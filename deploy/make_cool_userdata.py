"""Build the gzipped EC2 user-data for the COOL benchmark: inlines the source files as heredocs
(EC2 limits user-data to 16 KB). Output: build/userdata.sh.gz

Launch (CloudShell, after subscribing to COOL on AWS Marketplace):
  aws ec2 run-instances --instance-type c8g.large \
    --image-id resolve:ssm:/aws/service/marketplace/prod-k6u24vxijyzvg/cloud-optimized-opencv-3.1 \
    --user-data fileb://userdata.sh.gz --instance-initiated-shutdown-behavior terminate \
    --metadata-options HttpTokens=required
  aws ec2 get-console-output --instance-id <id> --latest --output text | grep LL
"""
import glob
import gzip
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
os.chdir(ROOT)
files = sorted(glob.glob("listinglens/*.py")) + ["eval/make_dataset.py", "bench/bench_cool.py",
                                                 "bench/abo_bench_images.txt"]
parts = []
for f in files:
    body = open(f).read()
    if f.endswith(".py"):  # drop comment-only and blank lines to stay under the 16 KB user-data limit
        body = "\n".join(l for l in body.splitlines() if l.strip() and not l.strip().startswith("#")) + "\n"
    if not body.endswith("\n"):
        body += "\n"
    parts.append(f"mkdir -p $(dirname {f}) && cat > {f} <<'LLEOF'\n{body}LLEOF\n")
ud = open(os.path.join(HERE, "cool_userdata.sh")).read().replace("__CODE__\n", "".join(parts))
os.makedirs("build", exist_ok=True)
data = gzip.compress(ud.encode(), 9)
assert len(data) <= 16384, f"user-data too large: {len(data)} bytes"
open("build/userdata.sh.gz", "wb").write(data)
print(f"build/userdata.sh.gz {len(data)} bytes")
