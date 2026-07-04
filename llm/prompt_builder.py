"""
llm/prompt_builder.py
======================
Builds prompts for Llama 3.1 — one SHORT analysis paragraph per track.

The old multi-section approach was unreliable (empty sections).
New design: ask for exactly 1 paragraph, template-generate everything else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# ─────────────────────────────────────────────────────────────
# Spatial attention → human-readable description
# ─────────────────────────────────────────────────────────────

_VERT_ZONES  = [(0.20, "head / neck region"),
                (0.42, "upper body / arms"),
                (0.65, "lower torso / hands"),
                (1.01, "lower body / ground region")]

_HORIZ_ZONES = [(0.33, "left side"),
                (0.66, "centre"),
                (1.01, "right side")]

_CONC_LEVELS = [(0.25, "low"),
                (0.50, "moderate"),
                (0.75, "high"),
                (1.01, "very high")]


def describe_attention(cam_3d: Optional[np.ndarray]) -> dict:
    """Convert cam_3d (T,14,14) into human-readable attention description."""
    if cam_3d is None:
        return {
            "vertical":     "full body region",
            "horizontal":   "centre of frame",
            "concentration":"unknown",
            "active_of":    "N/A",
        }

    cam_2d = cam_3d.mean(axis=0)            # (14,14)
    T, H, W = cam_3d.shape

    # Peak patch → body region
    ph, pw = np.unravel_index(cam_2d.argmax(), cam_2d.shape)
    hf, wf = float(ph) / H, float(pw) / W
    vert  = next(lbl for thr, lbl in _VERT_ZONES  if hf < thr)
    horiz = next(lbl for thr, lbl in _HORIZ_ZONES if wf < thr)

    # Concentration (peak / mean ratio normalised)
    peak, mean = float(cam_2d.max()), float(cam_2d.mean()) + 1e-8
    ratio = min(peak / mean / 10.0, 1.0)
    conc  = next(lbl for thr, lbl in _CONC_LEVELS if ratio < thr)

    # Active frames (mean > threshold)
    per_frame = cam_3d.mean(axis=(1, 2))
    thresh    = float(per_frame.mean()) * 0.5
    active    = int((per_frame >= thresh).sum())

    return {
        "vertical":     vert,
        "horizontal":   horiz,
        "concentration": conc,
        "active_of":    f"{active}/{T}",
    }


# ─────────────────────────────────────────────────────────────
# Risk level assignment (pure logic — no LLM needed)
# ─────────────────────────────────────────────────────────────

def assign_risk(is_abnormal: bool, fine_conf: float) -> tuple[str, str]:
    """
    Returns (risk_label, emoji).
    Abnormal events are always HIGH or CRITICAL regardless of confidence.
    """
    if not is_abnormal:
        return "LOW", "🟢"
    if fine_conf >= 0.70:
        return "CRITICAL", "🚨"
    if fine_conf >= 0.45:
        return "HIGH", "🔴"
    return "CRITICAL", "🚨"   # abnormal but low-conf → still flag critical


# ─────────────────────────────────────────────────────────────
# Prompt context
# ─────────────────────────────────────────────────────────────

@dataclass
class PromptContext:
    track_id:        int
    frame_start:     int
    frame_end:       int
    binary_label:    str
    fine_label:      str
    fine_conf:       float
    is_abnormal:     bool
    risk_label:      str
    risk_emoji:      str
    attn_vertical:   str
    attn_horizontal: str
    attn_conc:       str
    attn_active:     str
    t_start:         float
    t_end:           float


def build_context(
    classification,
    cam_3d: Optional[np.ndarray] = None,
    track_id: int = 0,
    frame_start: int = 0,
    frame_end: int = 0,
    fps: float = 30.0,
) -> PromptContext:
    attn = describe_attention(cam_3d)
    risk_label, risk_emoji = assign_risk(
        classification.is_abnormal, classification.fine_conf
    )
    t_start = float(frame_start) / fps
    t_end   = float(frame_end) / fps
    return PromptContext(
        track_id        = track_id,
        frame_start     = frame_start,
        frame_end       = frame_end,
        binary_label    = classification.binary_label,
        fine_label      = classification.fine_label,
        fine_conf       = classification.fine_conf,
        is_abnormal     = classification.is_abnormal,
        risk_label      = risk_label,
        risk_emoji      = risk_emoji,
        attn_vertical   = attn["vertical"],
        attn_horizontal = attn["horizontal"],
        attn_conc       = attn["concentration"],
        attn_active     = attn["active_of"],
        t_start         = t_start,
        t_end           = t_end,
    )


# ─────────────────────────────────────────────────────────────
# Prompt builder — asks for ONE paragraph only
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an AI surveillance analyst. Write exactly one concise paragraph (2-3 sentences)
analyzing the detected activity. Be factual and direct. No bullet points, no headers.
Start the paragraph exactly with: "Track {track_id} (frames {start}–{end}, {t_start:.2f}s–{t_end:.2f}s) was classified as"
"""

def build_llama3_prompt(ctx: PromptContext) -> str:
    """Build the Llama 3 chat prompt requesting a single analysis paragraph."""

    category = "ABNORMAL" if ctx.is_abnormal else "NORMAL"
    threat   = (
        f"This event poses a {ctx.risk_label} risk to public safety and may require "
        f"immediate attention."
        if ctx.is_abnormal else
        "No immediate threat was detected for this individual."
    )

    user_msg = (
        f"Track ID: {ctx.track_id}\n"
        f"Frame range: {ctx.frame_start}–{ctx.frame_end} ({ctx.t_start:.2f}s–{ctx.t_end:.2f}s)\n"
        f"Category: {category}\n"
        f"Detected action: {ctx.fine_label} ({ctx.fine_conf:.1%} confidence)\n"
        f"Risk level: {ctx.risk_label}\n"
        f"Spatial attention: focused on {ctx.attn_vertical} and {ctx.attn_horizontal} "
        f"with {ctx.attn_conc} motion/activity concentration; active in {ctx.attn_active} sampled frames.\n\n"
        f"Write exactly one paragraph starting with: "
        f"\"Track {ctx.track_id} (frames {ctx.frame_start}\u2013{ctx.frame_end}, {ctx.t_start:.2f}s\u2013{ctx.t_end:.2f}s) "
        f"was classified as {'ABNORMAL' if ctx.is_abnormal else 'NORMAL'} activity:\"\n"
        f"Include: action name, confidence, threat assessment, attention description, "
        f"risk level with emoji {ctx.risk_emoji}, and recommended action."
    )

    sys_prompt = SYSTEM_PROMPT.format(
        track_id=ctx.track_id,
        start=ctx.frame_start,
        end=ctx.frame_end,
        t_start=ctx.t_start,
        t_end=ctx.t_end,
    )

    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{sys_prompt}"
        "<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}"
        "<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def build_fallback_paragraph(ctx: PromptContext) -> str:
    """
    Template-generated paragraph used when LLM is disabled or fails.
    Matches the exact style of the desired report format.
    """
    category = "ABNORMAL" if ctx.is_abnormal else "NORMAL"
    threat   = (
        f"This event poses a {ctx.risk_label} risk to public safety and may require "
        f"immediate attention."
        if ctx.is_abnormal
        else "No immediate threat was detected for this individual."
    )
    action = (
        f"Dispatch security personnel to the area immediately and review footage."
        if ctx.risk_label in ("CRITICAL", "HIGH")
        else "Continue standard monitoring."
    )
    return (
        f"Track {ctx.track_id} (frames {ctx.frame_start}\u2013{ctx.frame_end}, {ctx.t_start:.2f}s\u2013{ctx.t_end:.2f}s) "
        f"was classified as {category} activity: '{ctx.fine_label}' with "
        f"{ctx.fine_conf:.1%} confidence. {threat} "
        f"Attention Analysis: Attention focused on {ctx.attn_vertical} "
        f"and {ctx.attn_horizontal} of frame with {ctx.attn_conc} "
        f"motion/activity concentration; active in {ctx.attn_active} sampled frames. "
        f" Risk Level: {ctx.risk_label} {ctx.risk_emoji} "
        f"Recommended Action: {action}"
    )
