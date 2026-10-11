"""Summarise bench results (results.jsonl from parse_console.py) into a markdown table.
Medians over rounds; speedup = stock / cool (>1 means COOL is faster)."""
import json
import statistics
import sys

R = [json.loads(l) for l in open(sys.argv[1])]
cool = [r for r in R if r["env"]["label"].startswith("cool")]
stock = [r for r in R if r["env"]["label"].startswith("stock")]
med = lambda rs, f: statistics.median(f(r) for r in rs)
print(f"rounds: cool {len(cool)}, stock {len(stock)}; cool {cool[0]['env']['opencv']} | stock {stock[0]['env']['opencv']}")
print(f"decision hashes: {sorted({r['agent']['decision_hash'] for r in R})}")
print("| operation | COOL ms | stock ms | speedup |")
print("|---|---:|---:|---:|")
for k in cool[0]["micro_ms"]:
    c, s = med(cool, lambda r: r["micro_ms"][k]), med(stock, lambda r: r["micro_ms"][k])
    print(f"| {k} | {c:.3f} | {s:.3f} | {s / c:.2f}x |")
for k in ("p50_ms", "p95_ms", "mean_ms", "throughput_img_per_s"):
    c, s = med(cool, lambda r: r["agent"][k]), med(stock, lambda r: r["agent"][k])
    ratio = c / s if k.startswith("throughput") else s / c
    print(f"| agent {k} | {c} | {s} | {ratio:.3f}x |")
