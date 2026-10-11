"""Planners choose the next action from the OpenCV evidence.

RulePlanner  - deterministic baseline, also the fallback.
BedrockPlanner - Amazon Nova (Bedrock Converse API, tool use). Its choice is validated
against the set of actions that are legal for the current evidence; an illegal or
failed call falls back to RulePlanner and the fallback is written to the trace.
"""
from __future__ import annotations

import json
import os

from .policy import REMEDIES, T, STUCK

RESHOOT_TIPS = {
    "low_resolution": "Use the main camera at full resolution (at least 1000 px on the short side).",
    "blurry": "Hold the phone steady or use a stand, tap the product to focus, use more light.",
    "cut_off": "Step back so the whole product is inside the frame with some space around it.",
    "blown_highlights": "Lower the exposure or move away from direct light: parts of the product are pure white.",
}


def legal_actions(ctx) -> list[str]:
    issues, failed = ctx["issues"], ctx["failed"]
    acts = []
    for i in issues:
        for tool in REMEDIES.get(i.code, []):
            if tool not in failed.get(i.code, set()) and tool not in acts:
                acts.append(tool)
    hard = [i for i in issues if i.severity in ("reshoot", "human")]
    stuck = [i for i in issues if i.severity == "fixable" and not any(
        t not in failed.get(i.code, set()) for t in REMEDIES.get(i.code, []))]
    finals = []
    if not issues:
        finals.append("accept")
    if hard or stuck:
        finals += ["reshoot", "human"]
    return acts + finals


class RulePlanner:
    name = "rules"
    ORDER = ["exposure", "white_balance", "whiten_background", "denoise", "sharpen", "recrop"]

    def decide(self, ctx) -> dict:
        issues, failed = ctx["issues"], ctx["failed"]
        if not issues:
            return {"action": "accept", "reason": "all checks pass", "planner": self.name}
        hard = [i for i in issues if i.severity == "reshoot"]
        if hard:
            msg = " ".join(RESHOOT_TIPS.get(i.code, i.evidence) for i in hard)
            return {"action": "reshoot", "reason": "; ".join(i.evidence for i in hard),
                    "message": msg, "planner": self.name}
        human = [i for i in issues if i.severity == "human"]
        if human:
            return {"action": "human", "reason": "; ".join(i.evidence for i in human),
                    "message": "Needs a person to review: " + "; ".join(i.code for i in human), "planner": self.name}
        legal = legal_actions(ctx)
        for tool in self.ORDER:
            if tool in legal:
                target = [i.code for i in issues if tool in REMEDIES.get(i.code, [])]
                return {"action": tool, "reason": "addresses " + ",".join(target), "planner": self.name}
        # fixable issues remain but every remedy was tried and reverted
        reshoot = [i for i in issues if STUCK.get(i.code, ("human",))[0] == "reshoot"]
        if reshoot and len(reshoot) == len(issues):
            return {"action": "reshoot", "reason": "automatic fixes failed: " + ",".join(i.code for i in issues),
                    "message": " ".join(STUCK[i.code][1] for i in reshoot), "planner": self.name}
        left = [f"{i.code} ({i.evidence})" for i in issues]
        msg = "Automatic fixes did not work, please review."
        if any(i.code == "busy_background" for i in issues):
            msg = ("The background could not be separated safely. If this is a lifestyle or in-use photo it can "
                   "work as a secondary image; for the main image shoot the product alone on a plain light background.")
        return {"action": "human", "reason": "automatic fixes failed: " + "; ".join(left),
                "message": msg, "planner": self.name}


SYSTEM = """You are the decision step of a product-photo QA agent for an online store.
You receive OpenCV 5 measurements of one photo, the issues they triggered, and which tools
were already tried and reverted. Choose exactly ONE next action from `legal_actions`.
Rules: never accept while issues remain; prefer the least invasive fix; if a fix was already
reverted for an issue, do not repeat it; ask for a reshoot when detail is physically missing
(blur, cut-off, clipped highlights, low resolution); ask a human when evidence is ambiguous or a
fix could misrepresent the product. When you choose reshoot or human, write a short, concrete
message for the seller (max 2 sentences, no jargon)."""

TOOL_SPEC = {
    "toolSpec": {
        "name": "choose_action",
        "description": "Choose the next action for this photo.",
        "inputSchema": {"json": {
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "reason": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["action", "reason"],
        }},
    }
}


class BedrockPlanner:
    name = "bedrock"

    def __init__(self, model_id: str | None = None, region: str | None = None, client=None):
        self.model_id = model_id or os.environ.get("LL_MODEL_ID", "amazon.nova-lite-v1:0")
        self.region = region or os.environ.get("AWS_REGION", "us-east-1")
        self._client = client
        self.fallback = RulePlanner()
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    @property
    def client(self):
        if self._client is None:
            import boto3
            self._client = boto3.client("bedrock-runtime", region_name=self.region)
        return self._client

    def _payload(self, ctx) -> str:
        r = ctx["report"]
        keys = ["width", "height", "occupancy", "center_offset", "edge_contact", "mask_confidence",
                "bg_lightness", "bg_uniformity", "bg_edge_density", "bg_cast", "obj_lightness",
                "obj_clipped_high", "blur_index", "noise_sigma"]
        return json.dumps({
            "metrics": {k: round(getattr(r, k), 3) if isinstance(getattr(r, k), float) else getattr(r, k) for k in keys},
            "thresholds": T,
            "issues": [i.__dict__ for i in ctx["issues"]],
            "reverted": {k: sorted(v) for k, v in ctx["failed"].items()},
            "steps_taken": [f"{s.action}: {s.outcome}" for s in ctx["history"]],
            "legal_actions": legal_actions(ctx),
        })

    def decide(self, ctx) -> dict:
        legal = legal_actions(ctx)
        try:
            resp = self.client.converse(
                modelId=self.model_id,
                system=[{"text": SYSTEM}],
                messages=[{"role": "user", "content": [{"text": self._payload(ctx)}]}],
                toolConfig={"tools": [TOOL_SPEC], "toolChoice": {"tool": {"name": "choose_action"}}},
                inferenceConfig={"maxTokens": 300, "temperature": 0},
            )
            self.calls += 1
            u = resp.get("usage", {})
            self.tokens_in += u.get("inputTokens", 0)
            self.tokens_out += u.get("outputTokens", 0)
            choice = next(c["toolUse"]["input"] for c in resp["output"]["message"]["content"] if "toolUse" in c)
        except Exception as e:  # network, throttling, parsing: fall back, never crash the loop
            d = self.fallback.decide(ctx)
            d["planner"] = f"rules (fallback: {type(e).__name__})"
            return d
        if choice.get("action") not in legal:
            d = self.fallback.decide(ctx)
            d["planner"] = f"rules (fallback: llm chose illegal '{choice.get('action')}')"
            return d
        choice["planner"] = self.name
        return choice
