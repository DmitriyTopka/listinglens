# Statistics behind the headline numbers (16 Oct 2026)

All numbers recomputed from files in `eval/results/`; nothing in the deployed rules was changed.
95 % intervals are Wilson score intervals. McNemar tests are paired (same photos, two pipelines).

## Held-out synthetic test set (1,768 photos, `ablation_test_per_image.csv`)

| pipeline | correct | share | 95 % CI |
|---|---:|---:|---|
| agent (verify and roll back) | 1,584 | 89.6 % | 88.1-90.9 % |
| open loop (same rules and tools, no re-measure) | 1,507 | 85.2 % | 83.5-86.8 % |

Paired: agent right and open loop wrong on 97 photos, the reverse on 20.
McNemar exact two-sided p = 2.6e-13 (chi-square with continuity correction 49.4).
On the 371 photos where the agent rolled back one of its own actions: 351 correct (94.6 %, CI 91.8-96.5 %)
vs open loop 274 (73.9 %, CI 69.2-78.1 %).

Caveat: 13 variants are built from each of 136 base photos, so photos are not fully independent and
the intervals above are somewhat narrower than a fully independent sample would give.

## 106 real catalog photos, labelled by hand before the run (`real106_labels.csv`)

33 photos with a real problem, 73 fine. "Stopped" = reshoot or sent to a person (not published as is
and not auto-passed).

| | agent | open loop |
|---|---|---|
| problem photos stopped | 27 / 33 = 82 % (CI 66-91 %) | 11 / 33 = 33 % (CI 20-50 %) |
| "accept" verdicts that were right | 18 / 23 = 78 % (CI 58-90 %) | 18 / 23 |
| good photos sent to a person | 31 / 73 = 42 % (CI 32-54 %) | 17 / 73 = 23 % |

Paired on the 33 problem photos: the agent stopped 16 that the open loop let through, the reverse 0.
McNemar exact p = 3.1e-5.

### Operating points (what a shop gets with each policy, 106 real photos)

| policy | problem photos published | good photos held back (person or reshoot) |
|---|---:|---:|
| publish everything | 33 / 33 | 0 / 73 |
| open loop | 22 / 33 | 29 / 73 |
| **ListingLens agent** | **6 / 33** | **43 / 73** |
| a person checks everything | 0 / 33 | 73 / 73 (all reviewed) |

A full threshold sweep (one knob trading person workload against missed problems) needs the raw metrics
of the 106 photos re-measured under each threshold; it is not in this release.

## Cost per 1,000 photos

Measured on the live Lambda (arm64, 2,048 MB) on 16 Oct: the six sample photos took 0.18-4.4 s of agent
time, about 1.1-1.2 s on average (mean on the 1,768-photo test run on a laptop: 0.93 s).
At 2 GB x 1.2 s = 2.4 GB-s per photo and the x86 list price of $0.0000166667 per GB-s (Arm is cheaper),
1,000 photos cost under $0.05 plus $0.0002 for requests. The always-free tier (400,000 GB-s and
1 M requests a month) covers about 160,000 photos a month.
