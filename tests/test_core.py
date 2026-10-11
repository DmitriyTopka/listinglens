import cv2
import numpy as np

from listinglens.agent import run
from listinglens.perceive import perceive, hamming
from listinglens.planners import RulePlanner, BedrockPlanner, legal_actions
from listinglens.policy import find_issues


def product(side=1200, occ=0.85, bg=255, offset=(0, 0)):
    """A textured 'product' (checker + text) centred on a flat background."""
    img = np.full((side, side, 3), bg, np.uint8)
    s = int(side * occ)
    x0 = (side - s) // 2 + offset[0]
    y0 = (side - s) // 2 + offset[1]
    tile = np.indices((s, s)).sum(axis=0) // 40 % 2
    patch = np.where(tile[..., None] == 1, (40, 60, 160), (30, 140, 60)).astype(np.uint8)
    cv2.putText(patch, "LISTING", (s // 8, s // 2), cv2.FONT_HERSHEY_SIMPLEX, s / 300, (240, 240, 240), 6)
    img[y0:y0 + s, x0:x0 + s] = patch
    return img


def codes(img):
    return [i.code for i in find_issues(perceive(img)[0])]


def test_clean_photo_is_accepted():
    r = run(product(), RulePlanner())
    assert r.decision == "accept", r.trace()


def test_low_resolution_needs_reshoot():
    r = run(cv2.resize(product(), (600, 600)), RulePlanner())
    assert r.decision == "reshoot"


def test_strong_blur_needs_reshoot():
    r = run(cv2.GaussianBlur(product(), (0, 0), 6), RulePlanner())
    assert r.decision == "reshoot"


def test_grey_background_is_fixed_and_verified():
    r = run(product(bg=190), RulePlanner())
    assert "background_not_white" in [i["code"] for i in r.issues_initial]
    assert r.decision == "auto_fixed", r.trace()
    assert r.final["bg_lightness"] >= 244


def test_small_product_is_recropped():
    img = product(occ=0.5, offset=(200, 150))
    assert "framing" in codes(img)
    r = run(img, RulePlanner())
    assert r.decision in ("auto_fixed", "reshoot")
    if r.decision == "auto_fixed":
        assert r.final["occupancy"] > 0.8


def test_colour_cast_is_corrected():
    lab = cv2.cvtColor(product(), cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[..., 2] += 14
    img = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    r = run(img, RulePlanner())
    assert r.decision == "auto_fixed"
    assert r.final["bg_cast"] < 4


def test_phash_detects_near_duplicate():
    a = product()
    b = cv2.resize(a[20:-20, 20:-20], (1200, 1200))
    c = product(occ=0.5, offset=(200, 150))
    pa, pb, pc = (perceive(x)[0].phash for x in (a, b, c))
    assert hamming(pa, pb) <= 8 < hamming(pa, pc)


class _BadLLM:
    """Bedrock client stub that proposes an illegal action: the loop must fall back to rules."""
    def converse(self, **kw):
        return {"output": {"message": {"content": [{"toolUse": {"input": {"action": "accept", "reason": "looks fine"}}}]}},
                "usage": {"inputTokens": 10, "outputTokens": 5}}


def test_llm_illegal_choice_falls_back():
    r = run(product(bg=190), BedrockPlanner(client=_BadLLM()))
    first = r.steps[0]
    assert "fallback" in first.planner
    assert first.action != "accept"
