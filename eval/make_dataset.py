"""Build the evaluation set from Amazon Berkeley Objects catalog photos (CC BY 4.0).

Each clean catalog photo is normalised (centred on white, 1200 px) and then degraded in
controlled ways, so the correct agent decision is known in advance. Photos that fail the
base filter (non-white or busy background, unclear product) are NOT degraded; they are kept
as a separate "real" set for the failure analysis.

Split: images sorted by id; first 150 = dev (thresholds were tuned here), the rest = test.
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from listinglens.perceive import perceive  # noqa: E402
from listinglens.tools import recrop, _full_res_mask  # noqa: E402

SIDE = 1200

# variant -> acceptable decisions
EXPECT = {
    "clean": {"accept"},
    "blur_strong": {"reshoot"},
    "motion_blur": {"reshoot"},
    "soft_focus": {"auto_fixed", "reshoot"},
    "dark": {"auto_fixed"},
    "colour_cast": {"auto_fixed"},
    "grey_background": {"auto_fixed"},
    "busy_background": {"auto_fixed", "human"},
    "small_offcentre": {"auto_fixed", "reshoot"},
    "cut_off": {"reshoot", "human"},
    "noisy": {"auto_fixed"},
    "overexposed": {"reshoot", "human"},
    "low_resolution": {"reshoot"},
}


def texture(h, w, rng):
    """Procedural cluttered background: wood-like stripes + random shapes (no third-party imagery)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    f = rng.uniform(0.02, 0.06)
    base = 0.5 + 0.5 * np.sin(xx * f + 6 * np.sin(yy * 0.01 + rng.uniform(0, 6)))
    col = np.array(rng.uniform(60, 200, 3), np.float32)
    img = (base[..., None] * col[None, None] * 0.6 + col * 0.4).astype(np.uint8)
    for _ in range(25):
        c = tuple(int(v) for v in rng.integers(0, 255, 3))
        if rng.random() < 0.5:
            cv2.circle(img, (int(rng.integers(0, w)), int(rng.integers(0, h))), int(rng.integers(10, 80)), c, -1)
        else:
            p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            cv2.rectangle(img, p1, (p1[0] + int(rng.integers(20, 150)), p1[1] + int(rng.integers(20, 150))), c, -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def replace_bg(img, colour_or_img):
    mask, _, _ = _full_res_mask(img)
    a = cv2.GaussianBlur(mask, (0, 0), 1.2).astype(np.float32)[..., None] / 255
    bg = colour_or_img if isinstance(colour_or_img, np.ndarray) and colour_or_img.ndim == 3 \
        else np.full_like(img, colour_or_img)
    return (img * a + bg * (1 - a)).astype(np.uint8)


def motion(img, k):
    ker = np.zeros((k, k), np.float32)
    ker[k // 2, :] = 1 / k
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), 30, 1)
    ker = cv2.warpAffine(ker, M, (k, k))
    ker /= ker.sum()
    return cv2.filter2D(img, -1, ker)


def variants(clean, rng):
    h, w = clean.shape[:2]
    out = {"clean": clean}
    out["blur_strong"] = cv2.GaussianBlur(clean, (0, 0), 4.5)
    out["motion_blur"] = motion(clean, 25)
    out["soft_focus"] = cv2.GaussianBlur(clean, (0, 0), 1.6)
    g = (clean / 255.0) ** 1.8 * 0.75
    out["dark"] = np.clip(g * 255, 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(clean, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 1] += rng.uniform(3, 7)
    lab[..., 2] += rng.uniform(10, 16)
    out["colour_cast"] = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    out["grey_background"] = replace_bg(clean, tuple(int(v) for v in rng.integers(150, 215, 1).repeat(3)))
    out["busy_background"] = replace_bg(clean, texture(h, w, rng))
    s = 0.42
    small = cv2.resize(clean, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    canvas = np.full_like(clean, 255)
    ox, oy = int(w * rng.uniform(0.02, 0.5)), int(h * rng.uniform(0.02, 0.5))
    canvas[oy:oy + small.shape[0], ox:ox + small.shape[1]] = small
    out["small_offcentre"] = canvas
    _, (bx, by, bw, bh), _ = _full_res_mask(clean)
    cut = rng.uniform(0.25, 0.4)  # share of the product pushed outside the frame
    dx = int(w - (bx + (1 - cut) * bw)) if rng.random() < 0.5 else -int(bx + cut * bw)
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    out["cut_off"] = cv2.warpAffine(clean, M, (w, h), borderValue=(255, 255, 255))
    out["noisy"] = np.clip(clean + rng.normal(0, 14, clean.shape), 0, 255).astype(np.uint8)
    out["overexposed"] = np.clip(clean.astype(np.float32) * 1.9 + 30, 0, 255).astype(np.uint8)
    out["low_resolution"] = cv2.resize(clean, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(os.path.dirname(__file__), "../../data/abo/original"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "../../data/synth"))
    ap.add_argument("--dev", type=int, default=150)
    a = ap.parse_args()
    rng = np.random.default_rng(2026)
    files = sorted(glob.glob(os.path.join(a.src, "*.jpg")))
    os.makedirs(a.out, exist_ok=True)
    rows, real = [], []
    for n, f in enumerate(files):
        split = "dev" if n < a.dev else "test"
        iid = os.path.splitext(os.path.basename(f))[0]
        img = cv2.imread(f)
        r, _ = perceive(img)
        ok = (r.bg_lightness >= 248 and r.bg_uniformity < 4 and r.mask_confidence >= 0.7
              and r.edge_contact < 0.05 and r.n_parts >= 1)
        if not ok:
            real.append((split, iid, f))
            continue
        clean = recrop(img)
        clean = cv2.resize(clean, (SIDE, SIDE), interpolation=cv2.INTER_AREA)
        for v, im in variants(clean, rng).items():
            d = os.path.join(a.out, split, v)
            os.makedirs(d, exist_ok=True)
            p = os.path.join(d, iid + ".jpg")
            cv2.imwrite(p, im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            rows.append((split, v, iid, os.path.relpath(p, a.out), "|".join(sorted(EXPECT[v]))))
    with open(os.path.join(a.out, "labels.csv"), "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["split", "variant", "image_id", "path", "expected"])
        w.writerows(rows)
    with open(os.path.join(a.out, "real_unfiltered.csv"), "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["split", "image_id", "path"])
        w.writerows(real)
    print(f"synthetic {len(rows)} images from {len(rows) // len(EXPECT)} clean bases; real/unfiltered {len(real)}")


if __name__ == "__main__":
    main()
