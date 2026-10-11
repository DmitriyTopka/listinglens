"""Parse LLRESULT lines out of `aws ec2 get-console-output` text.

The serial console interleaves timestamps like [2026-10-11T11:25:08.430205] into long lines and every
line is printed twice (console + ttyS0); this strips both and prints one JSON object per run.
usage: python3 parse_console.py console.txt > results.jsonl
"""
import json
import re
import sys

raw = open(sys.argv[1], errors="ignore").read()
raw = re.sub(r"\[\d{4}-\d\d-\d\dT[\d:.]+\]", "", raw)
seen, out = set(), []
for m in re.finditer(r"LLRESULT (\{.*)", raw):
    s = m.group(1)
    depth, end = 0, None
    for i, ch in enumerate(s):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            end = i + 1
            break
    if end is None:
        continue
    try:
        d = json.loads(s[:end])
    except json.JSONDecodeError:
        continue
    if d["env"]["label"] in seen:
        continue
    seen.add(d["env"]["label"])
    out.append(d)
for d in out:
    print(json.dumps(d))
