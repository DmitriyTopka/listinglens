"""Perception -> decision -> action -> verification loop.

The planner (rules or an LLM) only chooses WHAT to do next. The loop itself owns the
safety rails: it re-measures after every action, reverts actions that did not help or
broke something else, enforces a step budget, and checks the product still looks like
itself (truthfulness guard) before it auto-approves anything.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .perceive import perceive, Report, to_work
from .policy import find_issues, Issue, REMEDIES, verify
from .tools import TOOLS

MAX_STEPS = 6
FIDELITY_AB = 8.0    # product chroma may move at most (initial background cast + this) - truthfulness guard

FINAL = {"accept", "reshoot", "human"}


@dataclass
class Step:
    n: int
    action: str
    reason: str
    planner: str
    issues_before: list[str]
    issues_after: list[str] = field(default_factory=list)
    outcome: str = ""          # kept | reverted:<why> | final
    metrics_after: dict = field(default_factory=dict)
    ms: float = 0.0


@dataclass
class Result:
    decision: str              # accept | auto_fixed | reshoot | human
    message: str
    steps: list[Step]
    initial: dict
    final: dict
    issues_initial: list[dict]
    image: np.ndarray | None = None
    ms_total: float = 0.0

    def trace(self) -> dict:
        return {
            "decision": self.decision,
            "message": self.message,
            "initial_metrics": self.initial,
            "initial_issues": self.issues_initial,
            "final_metrics": self.final,
            "steps": [s.__dict__ for s in self.steps],
            "ms_total": round(self.ms_total, 1),
        }


def _codes(issues: list[Issue]) -> list[str]:
    return [i.code for i in issues]


def _obj_mean_lab(img: np.ndarray, rep_mask) -> np.ndarray:
    lab = cv2.cvtColor(to_work(img), cv2.COLOR_BGR2LAB).astype(np.float32)
    m = rep_mask.mask > 0
    if m.sum() < 50:
        m = np.ones(lab.shape[:2], bool)
    v = lab[m].mean(axis=0)
    return v[1:]  # a, b only: exposure may change lightness, but hue/chroma must stay put


def _metrics(r: Report) -> dict:
    d = r.to_dict()
    d.pop("extras", None)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}


def run(img: np.ndarray, planner, extra_issues: list[Issue] | None = None) -> Result:
    t0 = time.perf_counter()
    rep, mask = perceive(img)
    issues = find_issues(rep) + list(extra_issues or [])
    initial, initial_issues = _metrics(rep), [i.__dict__ for i in issues]
    ref_lab = _obj_mean_lab(img, mask)
    chroma_budget = rep.bg_cast + FIDELITY_AB
    cur = img
    steps: list[Step] = []
    failed: dict[str, set[str]] = {}   # issue code -> tools that did not fix it
    changed = False

    for n in range(1, MAX_STEPS + 1):
        ctx = {"issues": issues, "report": rep, "failed": failed, "history": steps, "changed": changed}
        choice = planner.decide(ctx)
        action, reason = choice["action"], choice.get("reason", "")
        step = Step(n, action, reason, choice.get("planner", planner.name), _codes(issues))
        if action in FINAL:
            step.outcome = "final"
            steps.append(step)
            decision = "auto_fixed" if (action == "accept" and changed) else action
            message = choice.get("message", reason)
            if decision == "auto_fixed":
                done = [s.action.replace("_", " ") for s in steps if s.outcome.startswith("kept")]
                message = "Fixed automatically (" + ", ".join(done) + ") and re-checked: all checks pass."
            elif decision == "accept":
                message = "Ready to publish: all checks pass."
            return _finish(decision, message, steps, initial, initial_issues, rep, cur, t0)

        ts = time.perf_counter()
        out = TOOLS[action](cur)
        gc_iou = None
        if isinstance(out, tuple):
            out, gc_iou = out
        new_rep, new_mask = perceive(out)
        new_issues = find_issues(new_rep) + [i for i in (extra_issues or [])]
        step.ms = (time.perf_counter() - ts) * 1000
        step.issues_after = _codes(new_issues)
        step.metrics_after = _metrics(new_rep)

        targets = [i.code for i in issues if action in REMEDIES.get(i.code, [])]
        before, after = set(_codes(issues)), set(_codes(new_issues))
        fixed = [c for c in targets if c not in after]
        introduced = after - before
        de = float(np.linalg.norm(_obj_mean_lab(out, new_mask) - ref_lab))

        why = verify(action, rep, new_rep)
        if why:
            pass
        elif gc_iou is not None and gc_iou < 0:
            why = "product continues past the frame (scene or in-use photo), background not replaced"
        elif gc_iou is not None and gc_iou < 0.80:
            why = f"grabcut disagrees with seed mask (IoU {gc_iou:.2f})"
        elif introduced:
            why = "introduced " + ",".join(sorted(introduced))
        elif not fixed:
            why = "did not fix " + ",".join(targets)
        elif de > chroma_budget:
            why = f"product colour shifted by {de:.1f} (budget {chroma_budget:.1f}), needs approval"

        if why:
            step.outcome = "reverted: " + why
            for c in targets:
                failed.setdefault(c, set()).add(action)
        else:
            step.outcome = f"kept (fixed {','.join(fixed)}; dE {de:.1f})"
            cur, rep, issues, changed = out, new_rep, new_issues, True
        steps.append(step)

    steps.append(Step(MAX_STEPS + 1, "human", "step budget exhausted", "loop", _codes(issues), outcome="final"))
    return _finish("human", "Could not resolve automatically within the step budget.", steps, initial,
                   initial_issues, rep, cur, t0)


def _finish(decision, message, steps, initial, initial_issues, rep, img, t0) -> Result:
    return Result(decision, message, steps, initial, _metrics(rep), initial_issues, img,
                  (time.perf_counter() - t0) * 1000)
