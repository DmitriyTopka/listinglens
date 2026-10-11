"""Benchmark the ListingLens workload under whichever OpenCV build the interpreter imports.

Run it twice on the same Graviton instance, once inside the COOL venv and once inside a venv
with the stock pip wheel, and compare the JSON lines it prints:

  /opt/cool/venvs/python_3.12/bin/python bench/bench_cool.py --label cool  --images imgs/
  /opt/stock/bin/python                  bench/bench_cool.py --label stock --images imgs/

Three parts:
  1. micro: median latency of the OpenCV calls the agent actually uses, on a 1600x1600 photo
  2. agent: the full perceive -> decide -> act -> verify loop on N photos x 13 defect variants
     (same generator as the evaluation set), latency p50/p95 and single-core throughput
  3. parity: a hash of every decision, so we can show both builds reach identical decisions
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import platform
import statistics
import sys
import time

import cv2
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))
from listinglens.agent import run  # noqa: E402
from listinglens.planners import RulePlanner  # noqa: E402
from listinglens.tools import recrop  # noqa: E402
from make_dataset import variants  # noqa: E402


def timeit(fn, reps):
    fn()  # warm-up
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(ts), 3)


def micro(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA)
    mask = (cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) < 200).astype(np.uint8) * 255
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    M = cv2.getRotationMatrix2D((800, 800), 7, 1.0)
    gc_mask = np.full(small.shape[:2], cv2.GC_PR_BGD, np.uint8)
    gc_mask[mask > 0] = cv2.GC_PR_FGD
    gc_mask[:10, :] = cv2.GC_BGD

    def grabcut():
        m = gc_mask.copy()
        cv2.grabCut(small, m, None, np.zeros((1, 65)), np.zeros((1, 65)), 2, cv2.GC_INIT_WITH_MASK)

    ops = {
        "resize_area_1600_to_1024": (lambda: cv2.resize(img, (1024, 1024), interpolation=cv2.INTER_AREA), 50),
        "resize_lanczos_up_2x": (lambda: cv2.resize(small, (2048, 2048), interpolation=cv2.INTER_LANCZOS4), 20),
        "cvtColor_BGR2LAB": (lambda: cv2.cvtColor(img, cv2.COLOR_BGR2LAB), 50),
        "cvtColor_BGR2GRAY": (lambda: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 50),
        "GaussianBlur_s1.6": (lambda: cv2.GaussianBlur(img, (0, 0), 1.6), 50),
        "blur_box_1x9": (lambda: cv2.blur(gray, (1, 9)), 50),
        "Laplacian_64F": (lambda: cv2.Laplacian(gray, cv2.CV_64F), 50),
        "filter2D_3x3": (lambda: cv2.filter2D(gray, cv2.CV_32F, np.ones((3, 3), np.float32)), 50),
        "Canny": (lambda: cv2.Canny(gray, 60, 160), 50),
        "threshold_otsu": (lambda: cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU), 50),
        "adaptiveThreshold_gauss": (lambda: cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                                  cv2.THRESH_BINARY, 31, 5), 50),
        "morphologyEx_close": (lambda: cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2), 50),
        "findContours": (lambda: cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE), 50),
        "warpAffine": (lambda: cv2.warpAffine(img, M, (1600, 1600)), 30),
        "addWeighted_unsharp": (lambda: cv2.addWeighted(img, 1.8, img, -0.8, 0), 50),
        "dct_32x32": (lambda: cv2.dct(np.float32(gray[:32, :32])), 200),
        "createCLAHE_apply": (lambda: cv2.createCLAHE(1.5, (8, 8)).apply(gray), 30),
        "imencode_jpeg": (lambda: cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88]), 20),
        "grabCut_1024_2it": (grabcut, 5),
        "fastNlMeansDenoisingColored_1024": (lambda: cv2.fastNlMeansDenoisingColored(small, None, 5, 5, 7, 21), 3),
    }
    return {name: timeit(fn, reps) for name, (fn, reps) in ops.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--skip-micro", action="store_true")
    a = ap.parse_args()
    cv2.setNumThreads(int(os.environ.get("LL_THREADS", "0")) or cv2.getNumThreads())

    info = cv2.getBuildInformation()
    env = {
        "label": a.label, "opencv": cv2.__version__, "cv2_file": cv2.__file__,
        "kleidicv": "kleidicv" in info.lower(), "threads": cv2.getNumThreads(),
        "machine": platform.machine(), "python": platform.python_version(),
        "cpu": next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.lower().startswith(("model name", "cpu part"))), "")
        if os.path.exists("/proc/cpuinfo") else platform.processor(),
    }
    files = sorted(glob.glob(os.path.join(a.images, "*.jpg")))[: a.n]
    rng = np.random.default_rng(2026)
    work = []
    for f in files:
        clean = cv2.resize(recrop(cv2.imread(f)), (1200, 1200), interpolation=cv2.INTER_AREA)
        for v, im in variants(clean, rng).items():
            work.append((os.path.basename(f), v, im))

    out = {"env": env}
    if not a.skip_micro:
        out["micro_ms"] = micro(cv2.resize(cv2.imread(files[0]), (1600, 1600)))

    lat, decisions = [], []
    t0 = time.perf_counter()
    for name, v, im in work:
        t = time.perf_counter()
        r = run(im, RulePlanner())
        lat.append((time.perf_counter() - t) * 1000)
        decisions.append(f"{name}:{v}:{r.decision}:{','.join(s.action for s in r.steps)}")
    wall = time.perf_counter() - t0
    lat.sort()
    out["agent"] = {
        "images": len(work),
        "p50_ms": round(lat[len(lat) // 2], 1),
        "p95_ms": round(lat[int(len(lat) * 0.95) - 1], 1),
        "mean_ms": round(sum(lat) / len(lat), 1),
        "throughput_img_per_s": round(len(work) / wall, 3),
        "decision_hash": hashlib.sha256("\n".join(decisions).encode()).hexdigest()[:16],
        "decisions": {d: sum(1 for x in decisions if x.split(":")[2] == d)
                      for d in ("accept", "auto_fixed", "reshoot", "human")},
    }
    print("LLRESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
