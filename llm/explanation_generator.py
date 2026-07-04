"""
llm/explanation_generator.py
==============================
Generates one analysis paragraph per track (via LLM or fallback template).
Covers ALL tracks (Normal + Abnormal), not just abnormal ones.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import numpy as np

from .prompt_builder import (
    PromptContext, build_context, build_llama3_prompt,
    build_fallback_paragraph, assign_risk,
)
from .llama_engine import LlamaEngine

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Output dataclass
# ─────────────────────────────────────────────────────────────

@dataclass
class ExplanationOutput:
    """Analysis for one track — single paragraph style."""
    track_id:    int
    frame_start: int
    frame_end:   int
    fine_label:  str
    fine_conf:   float
    binary_label: str
    is_abnormal: bool
    risk_label:  str      # LOW / MEDIUM / HIGH / CRITICAL
    risk_emoji:  str      # 🟢 / 🔴 / 🚨
    analysis:    str      # the generated paragraph
    t_start:     float = 0.0
    t_end:       float = 0.0
    elapsed_s:   float = 0.0
    llm_used:    bool  = False

    def to_dict(self) -> dict:
        return {
            "track_id":    self.track_id,
            "frame_start": self.frame_start,
            "frame_end":   self.frame_end,
            "t_start":     round(self.t_start, 2),
            "t_end":       round(self.t_end, 2),
            "fine_label":  self.fine_label,
            "fine_conf":   round(self.fine_conf, 4),
            "binary_label":self.binary_label,
            "is_abnormal": self.is_abnormal,
            "risk_label":  self.risk_label,
            "risk_emoji":  self.risk_emoji,
            "analysis":    self.analysis,
            "elapsed_s":   round(self.elapsed_s, 2),
            "llm_used":    self.llm_used,
        }

    @property
    def is_high_risk(self) -> bool:
        return self.risk_label in ("HIGH", "CRITICAL")


# ─────────────────────────────────────────────────────────────
# Generator
# ─────────────────────────────────────────────────────────────

class SurveillanceExplainer:
    """
    Generates one analysis paragraph per track.

    Parameters
    ----------
    engine      : LlamaEngine (or None → pure template mode)
    deduplicate : explain each track only once per video run
    """

    def __init__(
        self,
        engine:      Optional[LlamaEngine] = None,
        deduplicate: bool = True,
        explain_all: bool = True,    # kept for API compatibility
    ) -> None:
        self.engine      = engine
        self.deduplicate = deduplicate
        self._explained: Set[int] = set()

    def reset(self) -> None:
        self._explained.clear()

    # ──────────────────────────────────────────
    # Single track
    # ──────────────────────────────────────────

    def explain(
        self,
        classification,
        cam_3d:      Optional[np.ndarray] = None,
        track_id:    int = 0,
        frame_start: int = 0,
        frame_end:   int = 0,
        force:       bool = False,
        temporal     = None,          # kept for API compatibility
        fps:         float = 30.0,
    ) -> Optional[ExplanationOutput]:

        if self.deduplicate and not force and track_id in self._explained:
            return None

        ctx = build_context(
            classification = classification,
            cam_3d         = cam_3d,
            track_id       = track_id,
            frame_start    = frame_start,
            frame_end      = frame_end,
            fps            = fps,
        )

        t0       = time.perf_counter()
        llm_used = False
        analysis = ""

        # Try LLM first
        if self.engine is not None:
            prompt = build_llama3_prompt(ctx)
            try:
                raw = self.engine.generate(prompt)
                # Clean: keep only the first paragraph
                paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
                analysis   = paragraphs[0] if paragraphs else raw.strip()
                llm_used   = True
            except Exception as exc:
                log.warning("[Explainer] LLM failed for track %d: %s — using template", track_id, exc)

        # Fallback to template if LLM failed or not enabled
        if not analysis:
            analysis = build_fallback_paragraph(ctx)

        elapsed = time.perf_counter() - t0

        out = ExplanationOutput(
            track_id    = track_id,
            frame_start = frame_start,
            frame_end   = frame_end,
            t_start     = ctx.t_start,
            t_end       = ctx.t_end,
            fine_label  = classification.fine_label,
            fine_conf   = classification.fine_conf,
            binary_label= classification.binary_label,
            is_abnormal = classification.is_abnormal,
            risk_label  = ctx.risk_label,
            risk_emoji  = ctx.risk_emoji,
            analysis    = analysis,
            elapsed_s   = elapsed,
            llm_used    = llm_used,
        )

        if self.deduplicate:
            self._explained.add(track_id)

        src = "LLM" if llm_used else "template"
        log.info("[Explainer] Track %d | %s | risk=%s | %.1fs [%s]",
                 track_id, classification.fine_label, ctx.risk_label, elapsed, src)
        return out

    # ──────────────────────────────────────────
    # Batch
    # ──────────────────────────────────────────

    def explain_batch(
        self,
        raw_results:    Dict,
        cam_map:        Dict = None,
        frame_idx:      int  = 0,
        frame_ranges:   Dict = None,    # {track_id: (start, end)}
        temporal_map:   Dict = None,    # kept for API compat
        fps:            float = 30.0,
    ) -> Dict[int, ExplanationOutput]:

        cam_map      = cam_map      or {}
        frame_ranges = frame_ranges or {}
        outputs: Dict[int, ExplanationOutput] = {}

        for tid, cls_result in raw_results.items():
            fs, fe = frame_ranges.get(tid, (0, frame_idx))
            out = self.explain(
                classification = cls_result,
                cam_3d         = cam_map.get(tid),
                track_id       = tid,
                frame_start    = fs,
                frame_end      = fe,
                fps            = fps,
            )
            if out is not None:
                outputs[tid] = out

        return outputs
