"""Boundary map: where does the agent switch from "accept" to "fix" to "reshoot"?

Takes clean held-out test photos and applies one defect at increasing strength (blur, darkness,
colour cast, sensor noise, overexposure). For every strength level it records the share of each decision.
Thresholds are NOT changed; this only shows where the deployed rules draw their lines.

python eval/boundary_map.py --n 40      # -> eval/results/boundary_map.json, docs/boundary_map.png
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from listinglens.agent import run  # noqa: E402
from listinglens.planners import RulePlanner  # noqa: E402

CLEAN = os.path.join(ROOT, "../data/synth/test/clean")
DECISIONS = ["accept", "auto_fixed", "human", "reshoot"]


def blur(img, s):
    return img if s == 0 else cv2.GaussianBlur(img, (0, 0), s)


def dark(img, s):  # s in 0..1: gamma 1 -> 2.2 and gain 1 -> 0.65
    g, k = 1 + 1.2 * s, 1 - 0.35 * s
    return np.clip((img / 255.0) ** g * k * 255, 0, 255).astype(np.uint8)


def cast(img, s):  # s = shift of LAB b* (yellow); a* moves by a third of it
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 1] += s / 3
    lab[..., 2] += s
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def noise(img, s, seed=0):
    rng = np.random.default_rng(seed)
    return img if s == 0 else np.clip(img + rng.normal(0, s, img.shape), 0, 255).astype(np.uint8)


def bright(img, s):  # s in 0..1: gain 1 -> 1.9 and offset 0 -> 30 (the "overexposed" test variant at s = 1)
    return np.clip(img.astype(np.float32) * (1 + 0.9 * s) + 30 * s, 0, 255).astype(np.uint8)


DEFECTS = {
    "blur (Gaussian sigma, px)": (blur, [0, 0.8, 1.6, 2.4, 3.2, 4.5, 6.0]),
    "darkness (0 = none, 1 = strong)": (dark, [0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0]),
    "colour cast (LAB b* shift)": (cast, [0, 4, 8, 12, 16, 20, 26]),
    "sensor noise (sigma, 8-bit)": (noise, [0, 4, 8, 12, 16, 20, 28]),
    "overexposure (0 = none, 1 = test variant)": (bright, [0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0]),
}


def one(job):
    path, defect, s = job
    fn = DEFECTS[defect][0]
    img = fn(cv2.imread(path), s)
    return defect, s, run(img, RulePlanner()).decision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    files = sorted(os.listdir(CLEAN))[: a.n]
    jobs = [(os.path.join(CLEAN, f), d, s) for f in files for d, (_, levels) in DEFECTS.items() for s in levels]
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        res = list(ex.map(one, jobs, chunksize=4))
    table = collections.defaultdict(lambda: collections.defaultdict(collections.Counter))
    for d, s, dec in res:
        table[d][s][dec] += 1
    out = {"photos": len(files), "defects": {d: {str(s): dict(table[d][s]) for s in lv}
                                             for d, (_, lv) in DEFECTS.items()}}
    json.dump(out, open(os.path.join(ROOT, "eval/results/boundary_map.json"), "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colours = {"accept": "#2e7d32", "auto_fixed": "#1565c0", "human": "#f9a825", "reshoot": "#c62828"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    axes[1, 2].axis("off")
    for ax, (d, (_, lv)) in zip(axes.flat, DEFECTS.items()):
        shares = np.array([[table[d][s][k] / len(files) for s in lv] for k in DECISIONS])
        ax.stackplot(range(len(lv)), shares, labels=DECISIONS, colors=[colours[k] for k in DECISIONS], alpha=0.9)
        ax.set_xticks(range(len(lv)), [str(s) for s in lv])
        ax.set_title(d, fontsize=11)
        ax.set_xlim(0, len(lv) - 1)
        ax.set_ylim(0, 1)
    axes[0, 0].set_ylabel("share of photos")
    axes[1, 0].set_ylabel("share of photos")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[1, 2].legend(handles, labels, loc="center", fontsize=11, title="agent decision")
    fig.suptitle(f"ListingLens decision boundaries: {len(files)} clean held-out photos, one defect at a time",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "docs/boundary_map.png"), dpi=130)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
