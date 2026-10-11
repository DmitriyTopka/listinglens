"""Download the 358 Amazon Berkeley Objects catalog photos used in the evaluation (CC BY 4.0, public S3).

python eval/download_abo.py            # -> ../data/abo/original/<image_id>.jpg  (about 106 MB)
"""
import csv
import os
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "../../data/abo/original")
BASE = "https://amazon-berkeley-objects.s3.amazonaws.com/images/original/"
os.makedirs(OUT, exist_ok=True)
rows = list(csv.DictReader(open(os.path.join(HERE, "abo_images.csv"))))


def get(r):
    dst = os.path.join(OUT, r["image_id"] + ".jpg")
    if not os.path.exists(dst):
        try:
            urllib.request.urlretrieve(BASE + r["path"], dst)
        except Exception as e:  # a couple of ABO files are missing upstream
            return f"skip {r['image_id']}: {e}"
    return None


with ThreadPoolExecutor(12) as ex:
    errs = [e for e in ex.map(get, rows) if e]
print(f"{len(rows) - len(errs)} images in {OUT}; {len(errs)} skipped")
