"""AWS Lambda entry point (arm64 / Graviton), exposed through a Lambda Function URL.

GET  /            -> the web page (static HTML, no build step)
POST /api/check   -> {"image": "<base64 jpeg/png>", "listing_hashes": ["<phash>", ...]}
                     returns the agent decision, the full trace and the corrected image.

Images are processed in memory and never stored. One JSON log line per request goes
to CloudWatch (decision, issue codes, timings; no pixels, no personal data).
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from listinglens.agent import run  # noqa: E402
from listinglens.perceive import hamming  # noqa: E402
from listinglens.planners import RulePlanner, BedrockPlanner  # noqa: E402
from listinglens.policy import Issue  # noqa: E402

MAX_BYTES = 5 * 1024 * 1024
MAX_SIDE = 4000
DUP_HAMMING = 6
HERE = os.path.dirname(__file__)
PAGE = open(os.path.join(HERE, "index.html"), encoding="utf-8").read()
SAMPLES = os.path.join(HERE, "samples")


def _planner():
    return BedrockPlanner() if os.environ.get("LL_PLANNER") == "bedrock" else RulePlanner()


def _resp(code: int, body, ctype="application/json"):
    return {
        "statusCode": code,
        "headers": {"Content-Type": ctype, "Cache-Control": "no-store",
                    "X-Content-Type-Options": "nosniff"},
        "body": body if isinstance(body, str) else json.dumps(body),
    }


def _jpeg_b64(img: np.ndarray, side: int = 900) -> str:
    h, w = img.shape[:2]
    s = min(1.0, side / max(h, w))
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return base64.b64encode(buf.tobytes()).decode()


def check(payload: dict) -> tuple[int, dict]:
    raw = base64.b64decode(payload.get("image", ""), validate=False)
    if not raw:
        return 400, {"error": "no image"}
    if len(raw) > MAX_BYTES:
        return 413, {"error": "image larger than 5 MB"}
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return 415, {"error": "not a JPEG/PNG/WebP image"}
    if max(img.shape[:2]) > MAX_SIDE:
        s = MAX_SIDE / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    # duplicate check against the other photos of the same listing (hashes come from the browser)
    from listinglens.perceive import perceive
    ph = perceive(img)[0].phash
    extra = []
    for i, other in enumerate(payload.get("listing_hashes", [])[:50]):
        if isinstance(other, str) and len(other) == len(ph) and hamming(ph, other) <= DUP_HAMMING:
            extra.append(Issue("duplicate", "human", f"near-identical to photo #{i + 1} (hash distance {hamming(ph, other)})"))
            break

    res = run(img, _planner(), extra)
    out = res.trace()
    out["phash"] = ph
    out["image_out"] = _jpeg_b64(res.image) if res.decision == "auto_fixed" else None
    return 200, out


def handler(event, context=None):
    t0 = time.perf_counter()
    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    path = event.get("rawPath", "/")
    if method == "GET" and path in ("/", "/index.html"):
        return _resp(200, PAGE, "text/html; charset=utf-8")
    if method == "GET" and path.startswith("/samples/"):
        name = os.path.basename(path)
        if name == "list.json":
            files = sorted(f for f in os.listdir(SAMPLES) if f.endswith(".jpg"))
            return _resp(200, files)
        f = os.path.join(SAMPLES, name)
        if not (name.endswith(".jpg") and os.path.isfile(f)):
            return _resp(404, {"error": "not found"})
        r = _resp(200, base64.b64encode(open(f, "rb").read()).decode(), "image/jpeg")
        r["isBase64Encoded"] = True
        r["headers"]["Cache-Control"] = "public, max-age=86400"
        return r
    if method == "GET" and path == "/source.zip" and os.path.isfile(os.path.join(HERE, "source.zip")):
        r = _resp(200, base64.b64encode(open(os.path.join(HERE, "source.zip"), "rb").read()).decode(), "application/zip")
        r["isBase64Encoded"] = True
        r["headers"]["Content-Disposition"] = 'attachment; filename="listinglens_source.zip"'
        return r
    if method == "GET" and path == "/health":
        return _resp(200, {"ok": True, "opencv": cv2.__version__, "arch": os.uname().machine})
    if method != "POST" or path != "/api/check":
        return _resp(404, {"error": "not found"})
    try:
        body = event.get("body") or ""
        if event.get("isBase64Encoded"):
            body = base64.b64decode(body).decode()
        code, out = check(json.loads(body))
    except Exception as e:  # never leak a stack trace to the client
        code, out = 500, {"error": "internal error"}
        print(json.dumps({"level": "error", "type": type(e).__name__, "msg": str(e)[:200]}))
    print(json.dumps({
        "level": "info", "path": path, "status": code,
        "decision": out.get("decision"),
        "issues": [i["code"] for i in out.get("initial_issues", [])],
        "steps": [s["action"] for s in out.get("steps", [])],
        "agent_ms": out.get("ms_total"),
        "total_ms": round((time.perf_counter() - t0) * 1000, 1),
        "opencv": cv2.__version__, "arch": os.uname().machine,
    }))
    return _resp(code, out)
