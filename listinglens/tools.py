"""Actions the agent may take. Each tool returns a NEW image and never invents content:
only geometry, global tone/colour, denoise, mild sharpening and background replacement.
Anything stronger is out of scope on purpose (a product photo must stay truthful).
"""
from __future__ import annotations

import cv2
import numpy as np

from .perceive import segment_object, to_work, border_strip

TARGET_OCCUPANCY = 0.85
OUT_SIDE = 1600
MIN_SIDE = 1000
MAX_UPSCALE = 2.0
GRABCUT_SIDE = 640


def _full_res_mask(img: np.ndarray):
    work = to_work(img)
    om = segment_object(work)
    sx = img.shape[1] / work.shape[1]
    sy = img.shape[0] / work.shape[0]
    x, y, w, h = om.bbox
    mask = cv2.resize(om.mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask, (int(x * sx), int(y * sy), int(np.ceil(w * sx)), int(np.ceil(h * sy))), om


def _bg_colour(img: np.ndarray) -> np.ndarray:
    b = border_strip(*img.shape[:2])
    return np.median(img[b > 0], axis=0)


def recrop(img: np.ndarray, occupancy: float = TARGET_OCCUPANCY) -> np.ndarray:
    """Centre the product on a square canvas so it fills `occupancy` of the frame."""
    _, (x, y, w, h), om = _full_res_mask(img)
    if om.n_parts == 0:
        return img.copy()
    side = int(round(max(w, h) / occupancy))
    bg = _bg_colour(img).astype(np.uint8)
    canvas = np.empty((side, side, 3), np.uint8)
    canvas[:] = bg
    ox, oy = (side - w) // 2, (side - h) // 2
    canvas[oy:oy + h, ox:ox + w] = img[y:y + h, x:x + w]
    if side > OUT_SIDE:
        canvas = cv2.resize(canvas, (OUT_SIDE, OUT_SIDE), interpolation=cv2.INTER_AREA)
    elif side < MIN_SIDE and MIN_SIDE / side <= MAX_UPSCALE:
        # modest upscale only; beyond 2x the photo really lacks detail and the agent will ask for a reshoot
        canvas = cv2.resize(canvas, (MIN_SIDE, MIN_SIDE), interpolation=cv2.INTER_LANCZOS4)
    return canvas


def white_balance(img: np.ndarray) -> np.ndarray:
    """Neutralise the cast measured on the background (assumed neutral), applied globally in LAB."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    b = border_strip(*img.shape[:2])
    da = lab[..., 1][b > 0].mean() - 128
    db = lab[..., 2][b > 0].mean() - 128
    lab[..., 1] -= da
    lab[..., 2] -= db
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def exposure(img: np.ndarray) -> np.ndarray:
    """Levels on L: map the background to white (if it is light) or the object mid-tone to ~140, then mild CLAHE."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    b = border_strip(*img.shape[:2])
    bg_L = float(np.median(L[b > 0]))
    lo = float(np.percentile(L, 0.5))
    if bg_L > 120:  # light background that should be white: stretch so bg -> 255
        hi = max(bg_L, lo + 1)
        L2 = (L - lo) * (255.0 / (hi - lo))
    else:           # dark scene: gamma towards mid-tone
        mid = max(float(np.median(L)), 1.0)
        g = np.log(140 / 255) / np.log(mid / 255)
        L2 = 255.0 * (L / 255.0) ** g
    L2 = np.clip(L2, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    mask, _, _ = _full_res_mask(img)
    eq = clahe.apply(L2)
    L2 = np.where(mask > 0, eq, L2)  # local contrast only on the product, keep background flat
    lab[..., 0] = L2
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def whiten_background(img: np.ndarray, iters: int = 3) -> tuple[np.ndarray, float]:
    """GrabCut refinement seeded by our mask, then composite onto pure white with a feathered edge.
    Returns (image, agreement) where agreement = IoU between seed mask and GrabCut mask; low IoU means
    the separation is uncertain and a human should look."""
    work = to_work(img)
    s = GRABCUT_SIDE / max(work.shape[:2])
    if s < 1:  # GrabCut cost grows with pixel count; 640 px keeps the edge quality and is ~3x faster
        work = cv2.resize(work, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    om = segment_object(work)
    if om.n_parts == 0:
        return img.copy(), 0.0
    gc = np.full(work.shape[:2], cv2.GC_PR_BGD, np.uint8)
    gc[om.mask > 0] = cv2.GC_PR_FGD
    core = cv2.erode(om.mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    gc[core > 0] = cv2.GC_FGD
    gc[border_strip(*work.shape[:2], 0.02) > 0] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    cv2.grabCut(work, gc, None, bgd, fgd, iters, cv2.GC_INIT_WITH_MASK)
    fg = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    inter = np.logical_and(fg > 0, om.mask > 0).sum()
    union = np.logical_or(fg > 0, om.mask > 0).sum()
    iou = float(inter / union) if union else 0.0
    fg = cv2.resize(fg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    alpha = cv2.GaussianBlur(fg, (0, 0), 1.5).astype(np.float32)[..., None] / 255.0
    out = img.astype(np.float32) * alpha + 255.0 * (1 - alpha)
    return out.astype(np.uint8), iou


def denoise(img: np.ndarray) -> np.ndarray:
    """Non-local means with strength derived from the measured noise level."""
    from .perceive import noise_sigma
    s = noise_sigma(cv2.cvtColor(to_work(img), cv2.COLOR_BGR2GRAY))
    h = float(np.clip(s * 4.0, 3, 15))
    return cv2.fastNlMeansDenoisingColored(img, None, h, h, 7, 21)


def sharpen(img: np.ndarray, amount: float = 0.8, sigma: float = 1.6) -> np.ndarray:
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1 + amount, blur, -amount, 0)


TOOLS = {
    "recrop": recrop,
    "white_balance": white_balance,
    "exposure": exposure,
    "whiten_background": whiten_background,
    "denoise": denoise,
    "sharpen": sharpen,
}
