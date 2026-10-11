"""Ablation: what does the verify-and-revert loop buy?

Compares on the held-out test split:
  agent      - the full perceive -> decide -> act -> verify loop (deployed)
  open_loop  - same perception, same issue rules, same tools, but no re-measurement: apply the first
               remedy for every fixable issue once and trust it (a typical one-shot pipeline)
  rules_only - perception + issue rules only, no actions: fixable issues are reported as "auto_fixed"
               without fixing anything (upper bound of what thresholds alone can claim)

Scores are reported with the acceptable-answer sets used in the report ("lenient") and with the
original single-answer sets decided before seeing any results ("strict").
For open_loop we also count "false fixes": photos it calls auto_fixed whose output still fails the checks.

python eval/ablation.py --split test
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import cv2

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from listinglens.agent import run  # noqa: E402
from listinglens.perceive import perceive  # noqa: E402
from listinglens.planners import RulePlanner  # noqa: E402
from listinglens.policy import find_issues, REMEDIES  # noqa: E402
from listinglens.tools import TOOLS  # noqa: E402

SYNTH = os.path.join(ROOT, "../data/synth")
STRICT = {  # the sets written down before any result was seen
    "clean": {"accept"}, "blur_strong": {"reshoot"}, "motion_blur": {"reshoot"},
    "soft_focus": {"auto_fixed", "reshoot"}, "dark": {"auto_fixed"}, "colour_cast": {"auto_fixed"},
    "grey_background": {"auto_fixed"}, "busy_background": {"auto_fixed", "human"},
    "small_offcentre": {"auto_fixed"}, "cut_off": {"reshoot"}, "noisy": {"auto_fixed"},
    "overexposed": {"reshoot", "human"}, "low_resolution": {"reshoot"},
}
ORDER = ["exposure", "white_balance", "whiten_background", "denoise", "sharpen", "recrop"]


def open_loop(img):
    issues = find_issues(perceive(img)[0])
    if any(i.severity == "reshoot" for i in issues):
        return "reshoot", None
    if any(i.severity == "human" for i in issues):
        return "human", None
    tools = []
    for t in ORDER:
        if any(t == REMEDIES.get(i.code, [None])[0] for i in issues):
            tools.append(t)
    if not tools:
        return "accept", None
    out = img
    for t in tools:
        out = TOOLS[t](out)
        if isinstance(out, tuple):
            out = out[0]
    still = find_issues(perceive(out)[0])
    return "auto_fixed", bool(still)


def rules_only(img):
    issues = find_issues(perceive(img)[0])
    if any(i.severity == "reshoot" for i in issues):
        return "reshoot"
    if any(i.severity == "human" for i in issues):
        return "human"
    return "auto_fixed" if issues else "accept"


def one(path):
    img = cv2.imread(os.path.join(SYNTH, path))
    r = run(img, RulePlanner())
    ol, false_fix = open_loop(img)
    reverted = any(s.outcome.startswith("reverted") for s in r.steps)
    return {"agent": r.decision, "open_loop": ol, "false_fix": false_fix, "rules_only": rules_only(img),
            "reverted": reverted}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    rows = [r for r in csv.DictReader(open(os.path.join(SYNTH, "labels.csv"))) if r["split"] == a.split]
    with ProcessPoolExecutor(os.cpu_count()) as ex:
        res = list(ex.map(one, [r["path"] for r in rows], chunksize=4))
    score = collections.defaultdict(collections.Counter)
    for r, x in zip(rows, res):
        lenient, strict = set(r["expected"].split("|")), STRICT[r["variant"]]
        for m in ("agent", "open_loop", "rules_only"):
            score[m]["lenient"] += x[m] in lenient
            score[m]["strict"] += x[m] in strict
        if x["open_loop"] == "auto_fixed":
            score["open_loop"]["claimed_fixed"] += 1
            score["open_loop"]["false_fix"] += bool(x["false_fix"])
        if x["agent"] == "auto_fixed":
            score["agent"]["claimed_fixed"] += 1
        if x["reverted"]:
            score["agent"]["with_revert"] += 1
            score["agent"]["with_revert_correct"] += x["agent"] in lenient
            score["open_loop"]["on_reverted_correct"] += x["open_loop"] in lenient
    n = len(rows)
    out = {"split": a.split, "images": n}
    for m, c in score.items():
        out[m] = dict(c)
        out[m]["lenient_acc"] = round(c["lenient"] / n, 4)
        out[m]["strict_acc"] = round(c["strict"] / n, 4)
    os.makedirs(os.path.join(ROOT, "eval/results"), exist_ok=True)
    json.dump(out, open(os.path.join(ROOT, f"eval/results/ablation_{a.split}.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
