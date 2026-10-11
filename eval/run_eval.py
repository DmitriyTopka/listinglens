"""Run the agent over the labelled set and report decision accuracy per variant.

python eval/run_eval.py --split dev            # rules planner, all dev images
python eval/run_eval.py --split test --planner bedrock --per-variant 20
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import cv2

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from listinglens.agent import run  # noqa: E402
from listinglens.planners import RulePlanner, BedrockPlanner  # noqa: E402

SYNTH = os.path.join(ROOT, "../data/synth")


def _one(args):
    path, planner_name = args
    planner = BedrockPlanner() if planner_name == "bedrock" else RulePlanner()
    res = run(cv2.imread(os.path.join(SYNTH, path)), planner)
    t = res.trace()
    if planner_name == "bedrock":
        t["llm"] = {"calls": planner.calls, "in": planner.tokens_in, "out": planner.tokens_out}
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--planner", default="rules", choices=["rules", "bedrock"])
    ap.add_argument("--per-variant", type=int, default=0, help="limit images per variant (0 = all)")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(ROOT, "eval/results"))
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(os.path.join(SYNTH, "labels.csv"))) if r["split"] == a.split]
    if a.per_variant:
        seen = collections.Counter()
        keep = []
        for r in rows:
            if seen[r["variant"]] < a.per_variant:
                keep.append(r)
                seen[r["variant"]] += 1
        rows = keep
    t0 = time.time()
    workers = 4 if a.planner == "bedrock" else a.workers
    with ProcessPoolExecutor(workers) as ex:
        traces = list(ex.map(_one, [(r["path"], a.planner) for r in rows], chunksize=4))
    wall = time.time() - t0

    per = collections.defaultdict(lambda: collections.Counter())
    failures = []
    agentic = []  # traces where a measurement after an action changed the next step
    for r, t in zip(rows, traces):
        ok = t["decision"] in r["expected"].split("|")
        per[r["variant"]]["n"] += 1
        per[r["variant"]]["ok"] += ok
        per[r["variant"]][t["decision"]] += 1
        if not ok:
            failures.append({"path": r["path"], "variant": r["variant"], "expected": r["expected"],
                             "got": t["decision"], "issues": [i["code"] for i in t["initial_issues"]],
                             "steps": [(s["action"], s["outcome"]) for s in t["steps"]]})
        if any(s["outcome"].startswith("reverted") for s in t["steps"]):
            agentic.append({"path": r["path"], "decision": t["decision"],
                            "steps": [(s["action"], s["outcome"]) for s in t["steps"]]})
    total = sum(v["n"] for v in per.values())
    correct = sum(v["ok"] for v in per.values())
    summary = {
        "split": a.split, "planner": a.planner, "images": total, "correct": correct,
        "accuracy": round(correct / total, 4) if total else 0,
        "wall_s": round(wall, 1),
        "mean_ms_per_image": round(sum(t["ms_total"] for t in traces) / max(1, total), 1),
        "per_variant": {k: dict(v) for k, v in sorted(per.items())},
        "replanned_after_revert": len(agentic),
    }
    if a.planner == "bedrock":
        summary["llm_tokens_in"] = sum(t["llm"]["in"] for t in traces)
        summary["llm_tokens_out"] = sum(t["llm"]["out"] for t in traces)
        summary["llm_calls"] = sum(t["llm"]["calls"] for t in traces)
        summary["fallbacks"] = sum(1 for t in traces for s in t["steps"] if "fallback" in s["planner"])
    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.split}_{a.planner}"
    json.dump(summary, open(os.path.join(a.out, f"summary_{tag}.json"), "w"), indent=1)
    json.dump(failures, open(os.path.join(a.out, f"failures_{tag}.json"), "w"), indent=1)
    json.dump(agentic[:50], open(os.path.join(a.out, f"replans_{tag}.json"), "w"), indent=1)
    json.dump(traces[:30], open(os.path.join(a.out, f"traces_{tag}.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in summary.items() if k != "per_variant"}))
    for k, v in sorted(per.items()):
        print(f"{k:18s} {v['ok']:4d}/{v['n']:<4d} " + " ".join(f"{d}={v[d]}" for d in
              ("accept", "auto_fixed", "reshoot", "human") if v[d]))


if __name__ == "__main__":
    main()
