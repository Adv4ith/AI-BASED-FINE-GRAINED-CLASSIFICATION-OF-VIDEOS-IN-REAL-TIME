"""
visualiser.py — Overlay drawing utilities for the inference pipeline.

Draws on BGR frames:
  - Person bounding boxes (colour-coded by Normal/Abnormal/Unknown)
  - Track IDs + fine-grained classification label + confidence
  - Binary + fine-grained confidence bars
  - LLM risk level badge (CRITICAL / HIGH / MEDIUM / LOW)
  - LLM incident summary ticker line
  - Temporal analysis mini-HUD (active fraction, trend, peak frame)
  - Pipeline FPS + frame counter
"""

from __future__ import annotations

import time
import cv2
import numpy as np
from typing import Dict, Optional, List, Any

from config import CFG, VisConfig
from tracker import TrackedPerson
from vivit_model import ClassificationResult

# Risk level → BGR colour
_RISK_COLOURS = {
    "Critical": (0,   0,   220),   # red
    "High":     (0,   80,  230),   # orange-red
    "Medium":   (0,  165,  255),   # orange
    "Low":      (50, 200,   50),   # green
    "Unknown":  (130, 130, 130),   # grey
}


# ──────────────────────────────────────────────
# Helper: semi-transparent filled rectangle
# ──────────────────────────────────────────────

def _filled_rect_alpha(
    img: np.ndarray,
    pt1: tuple,
    pt2: tuple,
    colour: tuple,
    alpha: float = 0.4,
) -> None:
    overlay = img.copy()
    cv2.rectangle(overlay, pt1, pt2, colour, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


# ──────────────────────────────────────────────
# Helper: shadowed text
# ──────────────────────────────────────────────

def _shadow_text(
    img: np.ndarray,
    text: str,
    pos: tuple,
    scale: float = 0.5,
    color: tuple = (255, 255, 255),
    thickness: int = 1,
) -> None:
    cv2.putText(img, text, (pos[0]+1, pos[1]+1),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (0,0,0), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


# ──────────────────────────────────────────────
# Helper: confidence bar
# ──────────────────────────────────────────────

def _draw_conf_bar(
    img: np.ndarray,
    x: int, y: int,
    width: int, height: int,
    confidence: float,
    colour: tuple,
) -> None:
    cv2.rectangle(img, (x, y), (x + width, y + height), (50, 50, 50), -1)
    filled_w = int(width * confidence)
    if filled_w > 0:
        cv2.rectangle(img, (x, y), (x + filled_w, y + height), colour, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), (200, 200, 200), 1)


# ──────────────────────────────────────────────
# FPS tracker
# ──────────────────────────────────────────────

class FPSCounter:
    def __init__(self, window: int = 30) -> None:
        self._times: List[float] = []
        self._window = window

    def tick(self) -> float:
        now = time.perf_counter()
        self._times.append(now)
        if len(self._times) > self._window:
            self._times.pop(0)
        if len(self._times) < 2:
            return 0.0
        return (len(self._times) - 1) / (self._times[-1] - self._times[0])


# ──────────────────────────────────────────────
# Main Visualiser class
# ──────────────────────────────────────────────

class Visualiser:
    """
    Draws ALL inference overlays onto video frames.

    Layers (bottom → top):
      1. GradCAM heatmap blended inside bbox
      2. Bounding box rectangle + label banner
      3. Binary / fine-grained confidence bars
      4. LLM risk badge (top-right of bbox)
      5. Temporal HUD below confidence bars
      6. Global: FPS + frame counter (top-left)
      7. Global: LLM alert ticker (bottom strip, abnormal only)
      8. Global: pipeline status panel (top-right)
    """

    _CATEGORY_COLOUR = {
        "Normal":   CFG.vis.normal_colour,
        "Abnormal": CFG.vis.abnormal_colour,
        "Unknown":  CFG.vis.unknown_colour,
    }

    _FINE_TO_CATEGORY = {
        **{l: "Normal"   for l in ["Basketball", "Biking", "BoxingPunchingBag",
                                    "JumpingJack", "WalkingWithDog"]},
        **{l: "Abnormal" for l in ["Fighting", "Shooting", "Robbery",
                                    "Abuse", "Assault"]},
    }

    def __init__(self, cfg: VisConfig = CFG.vis) -> None:
        self.cfg = cfg
        self.fps = FPSCounter()
        # {track_id: ClassificationResult}
        self._last_results: Dict[int, ClassificationResult] = {}
        # {track_id: ExplanationOutput}   (from LLM)
        self._llm_reports:  Dict[int, Any] = {}
        # {track_id: TemporalAnalysisResult}
        self._temporal:     Dict[int, Any] = {}

    def update_results(self, results: Dict[int, ClassificationResult]) -> None:
        """Merge new ClassificationResults into cache."""
        self._last_results.update(results)

    def update_llm_reports(self, reports: Dict[int, Any]) -> None:
        """Store LLM ExplanationOutput objects per track."""
        self._llm_reports.update(reports)

    def update_temporal(self, temporal: Dict[int, Any]) -> None:
        """Store TemporalAnalysisResult objects per track."""
        self._temporal.update(temporal)

    def draw(
        self,
        frame:      np.ndarray,
        persons:    List[TrackedPerson],
        frame_idx:  int,
        show_fps:   bool = True,
    ) -> np.ndarray:
        """Draw all overlays on frame (in-place) and return it."""
        current_fps = self.fps.tick()

        # ── Per-person overlays ──────────────────────
        for person in persons:
            self._draw_person(frame, person)

        # ── Global overlays ──────────────────────────
        if show_fps:
            self._draw_fps_badge(frame, current_fps, frame_idx)

        self._draw_llm_alert_ticker(frame, persons)
        self._draw_pipeline_status(frame, persons, frame_idx)

        return frame

    # ──────────────────────────────────────────
    # Per-person rendering
    # ──────────────────────────────────────────

    def _draw_person(self, frame: np.ndarray, person: TrackedPerson) -> None:
        tid    = person.track_id
        result = self._last_results.get(tid)
        report = self._llm_reports.get(tid)
        temp   = self._temporal.get(tid)

        colour = self.cfg.unknown_colour
        main_label = f"ID:{tid}"

        if result is not None:
            category   = result.binary_label
            colour     = self._CATEGORY_COLOUR.get(category, self.cfg.unknown_colour)
            # If LLM gave a higher-accuracy risk colour, use that
            if report is not None:
                colour = _RISK_COLOURS.get(report.risk_level, colour)
            main_label = f"ID:{tid}  {result.fine_label}  {result.fine_conf:.0%}"

        x1, y1, x2, y2 = int(person.x1), int(person.y1), int(person.x2), int(person.y2)

        # 1. GradCAM heatmap blended inside bbox (drawn first — box on top)
        if result is not None and result.has_heatmap:
            self._blend_cam_on_bbox(frame, result.cam_3d, x1, y1, x2, y2)

        # 2. Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, self.cfg.thickness)

        # 3. Label banner (top of box)
        label_h = 22
        _filled_rect_alpha(frame, (x1, y1 - label_h), (x2, y1), colour, alpha=0.75)
        _shadow_text(frame, main_label, (x1 + 4, y1 - 5), 0.5, (255, 255, 255))

        # 4. CAM badge (small indicator top-right)
        if result is not None and result.has_heatmap:
            bx = x2 - 40
            by = y1 - label_h
            _filled_rect_alpha(frame, (bx, by), (x2, by + label_h), (200, 60, 0), 0.85)
            _shadow_text(frame, "CAM", (bx + 3, by + 14), 0.38, (255, 255, 255))

        # 5. LLM risk badge
        if report is not None:
            self._draw_risk_badge(frame, report, x1, y1, x2)

        # 6. Confidence bars + labels
        if result is not None:
            self._draw_classification_overlay(frame, person, result, colour)

        # 7. Temporal mini-HUD
        if temp is not None:
            self._draw_temporal_hud(frame, temp, x1, y2, x2)

    def _draw_risk_badge(
        self, frame: np.ndarray, report: Any,
        x1: int, y1: int, x2: int,
    ) -> None:
        """Draw coloured RISK level badge at top-right of bbox."""
        risk  = report.risk_level
        rcolour = _RISK_COLOURS.get(risk, (130, 130, 130))
        badge_text = f"⚠ {risk.upper()}" if risk in ("High", "Critical") else risk

        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        bx2 = x2
        bx1 = bx2 - tw - 10
        by1 = y1 - 46
        by2 = y1 - 24

        _filled_rect_alpha(frame, (bx1, by1), (bx2, by2), rcolour, 0.85)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 1)
        _shadow_text(frame, badge_text, (bx1 + 4, by2 - 4), 0.48, (255, 255, 255), 1)

    def _draw_classification_overlay(
        self,
        frame:  np.ndarray,
        person: TrackedPerson,
        result: ClassificationResult,
        colour: tuple,
    ) -> None:
        x1, y1, x2, y2 = int(person.x1), int(person.y1), int(person.x2), int(person.y2)
        panel_x = x1
        panel_y = y2 + 4
        bar_w   = max(x2 - x1, 80)

        # Binary confidence bar
        _draw_conf_bar(frame, panel_x, panel_y, bar_w, 7, result.binary_conf, colour)
        _shadow_text(frame,
                     f"{result.binary_label} {result.binary_conf:.0%}",
                     (panel_x, panel_y + 18), 0.45, colour)

        # Fine-grained
        if result.fine_label != "Unknown":
            fine_y = panel_y + 25
            fine_col = (60, 80, 230) if result.is_abnormal else (60, 200, 60)
            _draw_conf_bar(frame, panel_x, fine_y, bar_w, 7, result.fine_conf, fine_col)
            _shadow_text(frame,
                         f"{result.fine_label} {result.fine_conf:.0%}",
                         (panel_x, fine_y + 18), 0.45, fine_col)

    def _draw_temporal_hud(
        self, frame: np.ndarray, temp: Any,
        x1: int, y2: int, x2: int,
    ) -> None:
        """Draw a compact temporal analysis mini-HUD below the confidence bars."""
        bar_w = max(x2 - x1, 80)
        base_y = y2 + 56   # below the two confidence bars

        # Active-fraction mini-bar (teal)
        frac = float(getattr(temp, "active_fraction", 0.0))
        _draw_conf_bar(frame, x1, base_y, bar_w, 5, frac, (200, 180, 0))

        trend = str(getattr(temp, "trend_direction", ""))
        dom   = int(getattr(temp, "dominant_frame", 0))
        hud_text = f"act:{frac:.0%}  trend:{trend}  peak:f{dom}"
        _shadow_text(frame, hud_text, (x1, base_y + 16), 0.38, (200, 200, 100))

    # ──────────────────────────────────────────
    # Global overlays
    # ──────────────────────────────────────────

    def _draw_fps_badge(
        self, frame: np.ndarray, fps: float, frame_idx: int
    ) -> None:
        text = f"FPS: {fps:.1f}  |  Frame: {frame_idx}"
        _filled_rect_alpha(frame, (0, 0), (280, 28), (20, 20, 20), 0.65)
        _shadow_text(frame, text, (6, 19), 0.55, (220, 220, 220))

    def _draw_llm_alert_ticker(
        self, frame: np.ndarray, persons: List[TrackedPerson]
    ) -> None:
        """Bottom-strip ticker showing the latest LLM incident summary."""
        H, W = frame.shape[:2]
        # Collect active abnormal reports
        alerts = []
        for p in persons:
            rpt = self._llm_reports.get(p.track_id)
            if rpt and rpt.risk_level in ("High", "Critical"):
                short = rpt.incident_summary[:90] + "..." \
                    if len(rpt.incident_summary) > 90 else rpt.incident_summary
                alerts.append(f"[ID:{p.track_id}][{rpt.risk_level.upper()}] {short}")

        if not alerts:
            return

        ticker = "   ⚠   ".join(alerts)
        strip_h = 28
        _filled_rect_alpha(frame, (0, H - strip_h), (W, H), (10, 10, 40), 0.82)
        rcolour = _RISK_COLOURS.get("Critical", (0, 0, 220))
        cv2.rectangle(frame, (0, H - strip_h), (W, H - strip_h + 2), rcolour, -1)
        _shadow_text(frame, ticker, (8, H - 8), 0.45, (255, 220, 50))

    def _draw_pipeline_status(
        self, frame: np.ndarray,
        persons: List[TrackedPerson],
        frame_idx: int,
    ) -> None:
        """Top-right status panel: active tracks, abnormal count, LLM status."""
        H, W = frame.shape[:2]
        n_total    = len(persons)
        n_abnormal = sum(
            1 for p in persons
            if self._last_results.get(p.track_id, None) is not None
            and self._last_results[p.track_id].is_abnormal
        )
        n_llm = len(self._llm_reports)

        lines = [
            f"Tracks : {n_total}",
            f"Abnorm : {n_abnormal}",
            f"Reports: {n_llm}",
        ]
        panel_w, panel_h = 130, len(lines) * 20 + 8
        px = W - panel_w - 4
        py = 4
        _filled_rect_alpha(frame, (px, py), (px + panel_w, py + panel_h),
                           (20, 20, 20), 0.65)
        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h),
                      (80, 80, 80), 1)
        for i, line in enumerate(lines):
            _shadow_text(frame, line, (px + 6, py + 16 + i * 20), 0.42, (200, 220, 200))

    # ──────────────────────────────────────────
    # GradCAM heatmap blend
    # ──────────────────────────────────────────

    @staticmethod
    def _blend_cam_on_bbox(
        frame: np.ndarray,
        cam_3d: np.ndarray,            # (T_temporal, 14, 14) in [0,1]
        x1: int, y1: int, x2: int, y2: int,
        alpha: float = 0.55,           # base heatmap opacity
        colormap: int = cv2.COLORMAP_JET,
    ) -> None:
        """
        Project and blend the GradCAM activation map across the entire video frame.
        Smooths bounding box borders with a soft Gaussian filter and blends the
        colormap dynamically using the local activation strength.
        """
        H_f, W_f = frame.shape[:2]
        bx1 = max(0, x1);  by1 = max(0, y1)
        bx2 = min(W_f, x2); by2 = min(H_f, y2)
        bw  = bx2 - bx1;   bh  = by2 - by1
        if bw <= 0 or bh <= 0:
            return

        # Use temporal max projection across all frames
        cam_2d = cam_3d.max(axis=0).astype(np.float32)
        lo, hi = cam_2d.min(), cam_2d.max()
        if hi > lo:
            cam_2d = (cam_2d - lo) / (hi - lo + 1e-8)
        else:
            cam_2d = np.zeros_like(cam_2d)

        # Upsample to bbox size
        cam_up = cv2.resize(cam_2d, (bw, bh), interpolation=cv2.INTER_CUBIC)
        cam_up = np.clip(cam_up, 0.0, 1.0)

        # Project bounding box crop onto a full-frame activation canvas
        full_cam = np.zeros((H_f, W_f), dtype=np.float32)
        full_cam[by1:by2, bx1:bx2] = cam_up

        # Smooth transition edges globally to blend into the full frame context
        k = max(5, (min(W_f, H_f) // 16) | 1)
        full_cam = cv2.GaussianBlur(full_cam, (k, k), 0)
        full_cam = np.clip(full_cam, 0.0, 1.0)

        # Generate colormap over full frame
        heat_u8  = (full_cam * 255).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(heat_u8, colormap)

        # Perform dynamic pixel-wise blend across the entire frame
        # Regions with zero activation remain 100% clear.
        full_cam_3d = np.expand_dims(full_cam, axis=-1)
        dynamic_alpha = full_cam_3d * alpha

        frame_float = frame.astype(np.float32)
        heat_float  = heat_bgr.astype(np.float32)
        blended = frame_float * (1.0 - dynamic_alpha) + heat_float * dynamic_alpha
        frame[:] = np.clip(blended, 0, 255).astype(np.uint8)

        # Contour around hottest region (using local bounding box coordinates for overlay)
        thresh_val = max(int(cam_up.max() * 255 * 0.65), 1)
        heat_local_u8 = (cam_up * 255).astype(np.uint8)
        _, mask = cv2.threshold(heat_local_u8, thresh_val, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            shifted = [c + np.array([[[bx1, by1]]]) for c in contours]
            cv2.drawContours(frame, shifted, -1, (0, 255, 255), 1, cv2.LINE_AA)

    def reset(self) -> None:
        self._last_results.clear()
        self._llm_reports.clear()
        self._temporal.clear()
