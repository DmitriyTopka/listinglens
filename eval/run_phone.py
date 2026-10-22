"""One-shot run of the deployed agent on the 24 phone photos (decisions only, labels are NOT read here).

python eval/run_phone.py <photo_dir> <out_csv>
"""
from __future__ import annotations

import csv
import glob
import os
import sys

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from listinglens.agent import run  # noqa: E402
from listinglens.planners import RulePlanner  # noqa: E402


def main(photo_dir, out_csv):
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["photo", "decision", "issues", "steps", "ms"])
        for p in sorted(glob.glob(os.path.join(photo_dir, "*.jpg"))):
            r = run(cv2.imread(p), RulePlanner())
            t = r.trace()
            w.writerow([os.path.basename(p), r.decision,
                        "|".join(i["code"] for i in t.get("initial_issues", [])),
                        "|".join(f'{s["action"]}:{s["outcome"].split(" ")[0]}' for s in t.get("steps", [])),
                        t.get("ms_total")])


if __name__ == "__main__":
    main(*sys.argv[1:3])
