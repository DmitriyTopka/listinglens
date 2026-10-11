# ListingLens: a quality gate for product photos with verifiable autonomy

ListingLens checks every product photo before it goes live. An OpenCV 5 agent measures the photo,
fixes only what it can fix safely, **measures again to prove the fix worked, and rolls it back if it
did not**. Whatever it cannot verify goes back to the seller as a concrete reshoot tip or to a person.

**In numbers:** on 1,768 held-out photos the agent makes the right call 89.6 % of the time; the same
pipeline without the re-measure step gets 85.2 % and one in four of its "fixed" photos still fails the
checks. On 106 hand-labelled real catalog photos it stops 27 of 33 problem photos (an open-loop
pipeline stops 11). Runs in about a second per photo on AWS Lambda (Graviton) inside the free tier.

- Live endpoint (AWS Lambda arm64 / Graviton): https://je7md23hxr42uzjol5luyteg2q0axshn.lambda-url.us-east-1.on.aws/
  (no photos at hand? click one of the six sample photos on the page)
- Results: 89.6 % correct decisions on 1768 held-out test photos; COOL on Graviton4 in [bench/RESULTS.md](bench/RESULTS.md)
- Architecture: [docs/architecture.svg](docs/architecture.svg)
- Technical report: [docs/REPORT.md](docs/REPORT.md)

## Why

Marketplaces reject or down-rank listings with blurry, dark, tinted, badly framed or cluttered photos,
and small sellers find out after the fact. Fixing photos by hand is slow; "AI enhancement" that
repaints the product is not acceptable because the photo must stay truthful. ListingLens only uses
classical, explainable OpenCV operations and shows the evidence for every decision.

## How it works

```
photo -> 1 PERCEIVE (OpenCV 5 measurements) -> issues
      -> 2 DECIDE   (planner picks ONE legal next action)
      -> 3 ACT      (OpenCV tool: recrop, white balance, exposure, GrabCut background, NLM denoise, sharpen)
      -> 4 VERIFY   (measure again: keep the result or revert it and mark the tool as failed)
      -> back to 2 until: accept | auto_fixed | reshoot | human
```

The OpenCV result after each action changes what happens next: a reverted tool is excluded, a different
remedy is tried, or the photo is escalated. Safety rails live in the loop, not in the planner:
step budget (6), automatic revert on regression, GrabCut/seed-mask agreement check, and a colour-fidelity
guard (product chroma may not move more than the measured background cast + 8).

| Module | What it does |
|---|---|
| `listinglens/perceive.py` | product mask from LAB distance to the border colour (Otsu + morphology + contours), framing, edge contact, background lightness/clutter/cast, exposure, Crete-Roffet blur index, Laplacian variance, Immerkaer noise, DCT perceptual hash |
| `listinglens/policy.py` | thresholds and issue rules in one place; tool-specific verification |
| `listinglens/tools.py` | the six actions (all deterministic OpenCV) |
| `listinglens/agent.py` | the perceive-decide-act-verify loop, trace, guards |
| `listinglens/planners.py` | `RulePlanner` (default) and `BedrockPlanner` (Amazon Nova via Bedrock Converse tool use, validated against the legal action set, falls back to rules) |
| `app/` | Lambda handler + single-file web page; `local_server.py` runs the same handler locally |
| `eval/` | dataset builder (13 controlled defects on CC BY 4.0 catalog photos) and the evaluation runner |
| `bench/bench_cool.py` | COOL vs stock OpenCV 5 benchmark on Graviton4 |
| `deploy/` | Lambda build + deploy (with kill switch), COOL benchmark user-data |

## Run locally (macOS / Linux, Python 3.10+)

```bash
pip install -r requirements.txt
python -m pytest -q                      # 8 tests
python app/local_server.py               # http://localhost:8765
```

## Reproduce the evaluation

```bash
# 1. download the 358 Amazon Berkeley Objects images listed in ../data/abo/sample.csv (public S3, CC BY 4.0)
# 2. build the labelled set (3276 images) and run the agent on it
python eval/make_dataset.py
python eval/run_eval.py --split dev          # thresholds were tuned on dev only
python eval/run_eval.py --split test         # held-out split, run once for the report
```

## Deploy to AWS (everything inside the always-free tier)

From AWS CloudShell in `us-east-1`, with this repository uploaded:

```bash
bash deploy/build_lambda.sh ~/build          # arm64 layers: OpenCV 5 + numpy, and the function zip
cd ~/build && ALERT_EMAIL=you@example.com bash ~/listinglens/deploy/deploy_lambda.sh
```

This creates: Lambda `listinglens` (arm64, Python 3.12, 2048 MB), a public Function URL, log retention
14 days, and a kill switch (CloudWatch alarm at >300 invocations/hour -> SNS -> `listinglens-killswitch`
sets reserved concurrency to 0). Re-enable with `aws lambda delete-function-concurrency --function-name listinglens`.

## COOL benchmark (Graviton4)

Subscribe to "Cloud Optimized OpenCV For AWS Graviton4" on AWS Marketplace (7-day free trial), then:

```bash
python deploy/make_cool_userdata.py          # -> build/userdata.sh.gz (source inlined, < 16 KB)
aws ec2 run-instances --instance-type c8g.large \
  --image-id resolve:ssm:/aws/service/marketplace/prod-k6u24vxijyzvg/cloud-optimized-opencv-3.1 \
  --user-data fileb://build/userdata.sh.gz --instance-initiated-shutdown-behavior terminate \
  --metadata-options HttpTokens=required
aws ec2 get-console-output --instance-id <id> --latest --output text | grep -aE "LLRESULT|LLCOOL|LLSTOCK"
```

The instance installs a stock `opencv-python-headless==5.0.0.93` venv next to COOL, runs the same
benchmark three times in each build, prints JSON lines to the serial console and terminates itself.

## Data and licences

- Photos: [Amazon Berkeley Objects](https://registry.opendata.aws/amazon-berkeley-objects/), CC BY 4.0
  (Collins et al., "ABO: Dataset and Benchmarks for Real-World 3D Object Understanding", CVPR 2022).
- OpenCV: Apache 2.0. No user photos are stored by the service.
