"""Turns an OpenCV Report into named issues. Thresholds live in one place so the
evaluation can report exactly which numbers produced a decision."""
from __future__ import annotations

from dataclasses import dataclass

from .perceive import Report

T = {
    "min_side": 1000,          # marketplaces need >= 1000 px for zoom
    "blur_fixable": 0.47,      # blur_index above this AND little Laplacian energy: try sharpening once
    "lap_fixable": 45.0,
    "blur_reshoot": 0.62,      # blur_index above this AND almost no Laplacian energy: detail is gone
    "lap_reshoot": 8.0,
    "edge_contact": 0.06,      # product runs along the frame edge: possibly cut off (human)
    "edge_cut": 0.20,          # clearly cut off (reshoot)
    "occ_low": 0.62,
    "occ_high": 0.985,
    "center_offset": 0.07,
    "bg_white": 244.0,
    "bg_flat_std": 9.0,
    "bg_clutter_edges": 0.02,
    "cast": 4.0,
    "obj_dark": 70.0,
    "bg_dark": 200.0,         # a dark product on a white background is not underexposure
    "clip_high": 0.30,
    "noise": 1.2,
    "mask_conf": 0.45,
}


@dataclass
class Issue:
    code: str
    severity: str     # "fixable" | "reshoot" | "human"
    evidence: str


def find_issues(r: Report) -> list[Issue]:
    out: list[Issue] = []
    if min(r.width, r.height) < T["min_side"]:
        out.append(Issue("low_resolution", "reshoot", f"min side {min(r.width, r.height)} px < {T['min_side']}"))
    flat = r.bg_uniformity < T["bg_flat_std"] and r.bg_edge_density < T["bg_clutter_edges"]
    if not flat:
        # On a cluttered background the product mask, framing, focus and colour readings are unreliable.
        # Report only the clutter; the agent cleans the background first and then measures again.
        out.append(Issue("busy_background", "fixable",
                         f"border std {r.bg_uniformity:.1f}, edge share {r.bg_edge_density:.3f}"))
        return out
    if r.n_parts == 0 or r.mask_confidence < T["mask_conf"]:
        out.append(Issue("unclear_product", "human", f"mask confidence {r.mask_confidence:.2f}"))
    if r.blur_index > T["blur_reshoot"] and r.lap_var < T["lap_reshoot"]:
        out.append(Issue("blurry", "reshoot", f"blur index {r.blur_index:.2f}, Laplacian var {r.lap_var:.1f}"))
    elif r.blur_index > T["blur_fixable"] and r.lap_var < T["lap_fixable"]:
        out.append(Issue("soft_focus", "fixable", f"blur index {r.blur_index:.2f}, Laplacian var {r.lap_var:.1f}"))
    if r.edge_contact > T["edge_cut"]:
        out.append(Issue("cut_off", "reshoot", f"product runs along {r.edge_contact:.0%} of a frame edge"))
    elif r.edge_contact > T["edge_contact"]:
        out.append(Issue("maybe_cut_off", "human", f"product touches {r.edge_contact:.0%} of a frame edge"))
    elif r.n_parts and (r.occupancy < T["occ_low"] or r.occupancy > T["occ_high"] or r.center_offset > T["center_offset"]):
        out.append(Issue("framing", "fixable",
                         f"occupancy {r.occupancy:.2f}, centre offset {r.center_offset:.3f}"))
    if r.bg_lightness < T["bg_white"]:
        out.append(Issue("background_not_white", "fixable", f"background L {r.bg_lightness:.0f} < {T['bg_white']:.0f}"))
    if r.bg_cast > T["cast"]:
        out.append(Issue("colour_cast", "fixable", f"background chroma {r.bg_cast:.1f} > {T['cast']}"))
    if r.obj_lightness < T["obj_dark"] and r.bg_lightness < T["bg_dark"]:
        out.append(Issue("underexposed", "fixable", f"product L {r.obj_lightness:.0f} < {T['obj_dark']:.0f}"))
    if r.obj_clipped_high > T["clip_high"]:
        out.append(Issue("blown_highlights", "reshoot", f"{r.obj_clipped_high:.0%} of product pixels clipped"))
    if r.noise_sigma > T["noise"]:
        out.append(Issue("noisy", "fixable", f"noise sigma {r.noise_sigma:.2f} > {T['noise']}"))
    return out


# which tools can address which issue, in preferred order
REMEDIES = {
    "soft_focus": ["sharpen"],
    "framing": ["recrop"],
    "busy_background": ["whiten_background"],
    "background_not_white": ["exposure", "whiten_background"],
    "colour_cast": ["white_balance"],
    "underexposed": ["exposure"],
    "noisy": ["denoise"],
}


# when every remedy for a fixable issue was tried and reverted, where does the photo go
STUCK = {
    "soft_focus": ("reshoot", "Hold the phone steady or use a stand, tap the product to focus, add light."),
    "framing": ("reshoot", "Move closer so the product fills most of the frame."),
    "noisy": ("reshoot", "Shoot with more light; the photo is grainy."),
}


def verify(tool: str, before: Report, after: Report) -> str | None:
    """Tool-specific success criteria that a tool cannot satisfy by gaming the metric it targets."""
    if tool == "sharpen" and after.blur_index > T["blur_fixable"] - 0.07:
        return f"sharpening did not recover real detail (blur index {after.blur_index:.2f})"
    return None
