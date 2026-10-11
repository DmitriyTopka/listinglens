"""Perception: OpenCV 5 measurements of a product photo.

Every function here is pure: image in, numbers out. The agent never looks at
pixels directly, it only sees the Report produced by `perceive()`.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import cv2
import numpy as np

WORK_SIDE = 1024  # analysis resolution, longest side


@dataclass
class ObjectMask:
    mask: np.ndarray            # uint8 0/255, analysis resolution
    bbox: tuple[int, int, int, int]  # x, y, w, h in analysis resolution
    confidence: float           # 0..1, how cleanly the object separates from background
    n_parts: int


@dataclass
class Report:
    width: int
    height: int
    # framing
    occupancy: float            # max(bbox_w / W, bbox_h / H)
    center_offset: float        # distance of bbox center from image center, fraction of diagonal
    touches_edge: bool          # product bbox touches the frame
    edge_contact: float         # longest run of object pixels on a frame edge, share of that edge (cut-off signal)
    mask_confidence: float
    n_parts: int
    # background
    bg_lightness: float         # LAB L of border strip, 0..255
    bg_uniformity: float        # std of L in border strip
    bg_edge_density: float      # Canny edge share in border strip (clutter)
    bg_cast: float              # chroma of border strip, sqrt(da^2 + db^2)
    # exposure
    obj_lightness: float
    obj_clipped_high: float     # share of object pixels with L >= 250
    obj_clipped_low: float      # share of object pixels with L <= 8
    black_point: float          # 2nd percentile of L inside the product bbox (washed-out signal)
    # detail
    lap_var: float              # variance of Laplacian on object region
    blur_index: float           # Crete-Roffet no-reference blur metric, 0 sharp .. 1 blurry
    noise_sigma: float          # Immerkaer noise estimate
    phash: str = ""
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def to_work(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = WORK_SIDE / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def border_strip(h: int, w: int, frac: float = 0.04) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    b = max(2, int(round(min(h, w) * frac)))
    m[:b, :] = 255
    m[-b:, :] = 255
    m[:, :b] = 255
    m[:, -b:] = 255
    return m


def segment_object(img: np.ndarray) -> ObjectMask:
    """Foreground = pixels far (in LAB) from the border colour, cleaned with morphology."""
    h, w = img.shape[:2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    border = border_strip(h, w)
    bg = np.median(lab[border > 0], axis=0)
    dist = np.linalg.norm(lab - bg, axis=2)
    dist = cv2.GaussianBlur(dist, (5, 5), 0)
    # Otsu on the distance map, but never below a floor that ignores JPEG noise
    d8 = np.clip(dist * 3, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(d8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thr = max(otsu / 3.0, 10.0)
    fg = (dist > thr).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k)

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = 0.002 * h * w
    keep = [c for c in contours if cv2.contourArea(c) >= min_area]
    mask = np.zeros((h, w), np.uint8)
    if not keep:
        return ObjectMask(mask, (0, 0, w, h), 0.0, 0)
    cv2.drawContours(mask, keep, -1, 255, thickness=cv2.FILLED)
    x, y, bw, bh = cv2.boundingRect(np.vstack(keep))

    # confidence: contrast between distance inside vs outside the mask edge band
    edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, k)
    inside = dist[(mask > 0) & (edge == 0)]
    outside = dist[(mask == 0) & (edge == 0)]
    if inside.size == 0 or outside.size == 0:
        conf = 0.0
    else:
        sep = (np.median(inside) - np.median(outside)) / (np.std(outside) * 3 + thr)
        conf = float(np.clip(sep, 0, 1))
    return ObjectMask(mask, (x, y, bw, bh), conf, len(keep))


def blur_index(gray: np.ndarray) -> float:
    """Crete-Roffet et al. 2007: compare image to its own re-blurred copy."""
    g = gray.astype(np.float32)
    bv = cv2.blur(g, (1, 9))
    bh = cv2.blur(g, (9, 1))
    d_fv = np.abs(np.diff(g, axis=0))
    d_fh = np.abs(np.diff(g, axis=1))
    d_bv = np.abs(np.diff(bv, axis=0))
    d_bh = np.abs(np.diff(bh, axis=1))
    v_v = np.maximum(0, d_fv - d_bv)
    v_h = np.maximum(0, d_fh - d_bh)
    s_fv, s_fh = d_fv.sum(), d_fh.sum()
    if s_fv == 0 or s_fh == 0:
        return 1.0
    b_v = (s_fv - v_v.sum()) / s_fv
    b_h = (s_fh - v_h.sum()) / s_fh
    return float(max(b_v, b_h))


def noise_sigma(gray: np.ndarray) -> float:
    """Immerkaer 1996 fast noise estimate, computed on flat regions only (Sobel magnitude below the
    image's 60th percentile), so fabric weave or printed packaging is not mistaken for sensor noise."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    g = gray.astype(np.float32)
    conv = np.abs(cv2.filter2D(g, -1, k))[1:-1, 1:-1]
    mag = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))[1:-1, 1:-1]
    flat = mag <= np.percentile(mag, 60)
    if flat.sum() < 1000:
        flat = np.ones_like(flat)
    return float(np.sqrt(np.pi / 2) * conv[flat].mean() / 6)


def phash(img: np.ndarray) -> str:
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    g = cv2.resize(g, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    d = cv2.dct(g)[:8, :8].flatten()
    med = np.median(d[1:])
    bits = d > med
    return "".join("1" if b else "0" for b in bits)


def hamming(a: str, b: str) -> int:
    return sum(x != y for x, y in zip(a, b))


def perceive(img_full: np.ndarray) -> tuple[Report, ObjectMask]:
    H, W = img_full.shape[:2]
    img = to_work(img_full)
    h, w = img.shape[:2]
    om = segment_object(img)
    x, y, bw, bh = om.bbox

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L = lab[..., 0].astype(np.float32)
    border = border_strip(h, w)
    # background = border strip minus the (dilated) product, so a product touching the frame is not read as clutter
    near_obj = cv2.dilate(om.mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    bsel = (border > 0) & (near_obj == 0)
    if bsel.sum() < 0.1 * (border > 0).sum():
        bsel = border > 0
    bpx = lab[bsel].astype(np.float32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 60, 160)

    objm = om.mask > 0
    if objm.sum() < 50:
        objm = np.ones_like(objm)
    obj_L = L[objm]

    # detail on the object's bounding box, so a big white background does not dilute it
    pad = 4
    roi = gray[max(0, y - pad):y + bh + pad, max(0, x - pad):x + bw + pad]
    if roi.size < 400:
        roi = gray
    lap = cv2.Laplacian(roi, cv2.CV_64F)

    m = 1
    touches = om.n_parts > 0 and (x <= m or y <= m or x + bw >= w - m or y + bh >= h - m)
    if om.n_parts:
        sides = [om.mask[0, :], om.mask[-1, :], om.mask[:, 0], om.mask[:, -1]]
        contact = max(float((sd > 0).mean()) for sd in sides)
    else:
        contact = 0.0
    cx, cy = x + bw / 2, y + bh / 2
    off = float(np.hypot(cx - w / 2, cy - h / 2) / np.hypot(w, h))

    rep = Report(
        width=W, height=H,
        occupancy=float(max(bw / w, bh / h)) if om.n_parts else 0.0,
        center_offset=off,
        touches_edge=bool(touches),
        edge_contact=contact,
        mask_confidence=om.confidence,
        n_parts=om.n_parts,
        bg_lightness=float(bpx[:, 0].mean()),
        bg_uniformity=float(bpx[:, 0].std()),
        bg_edge_density=float((edges[bsel] > 0).mean()),
        bg_cast=float(np.hypot(bpx[:, 1].mean() - 128, bpx[:, 2].mean() - 128)),
        obj_lightness=float(obj_L.mean()),
        obj_clipped_high=float((obj_L >= 250).mean()),
        obj_clipped_low=float((obj_L <= 8).mean()),
        black_point=float(np.percentile(L[y:y + bh, x:x + bw], 2)) if om.n_parts else float(np.percentile(L, 2)),
        lap_var=float(lap.var()),
        blur_index=blur_index(roi),
        noise_sigma=noise_sigma(gray),
        phash=phash(img),
    )
    return rep, om
