"""
llm/report_writer.py
=====================
Generates the final surveillance report in the exact requested format:

  ======================================================================
           AI SURVEILLANCE ANALYSIS REPORT
  ======================================================================
  Video     : ab_test.mp4
  Generated : 2026-05-22T11:23:01
  ======================================================================

  INDIVIDUAL TRACK REPORTS
  ----------------------------------------------------------------------
  Track ID  : 1
  Frames    : 0 – 133
  Category  : NORMAL
  Action    : JumpingJack (29.1%)
  Risk      : LOW 🟢

  Analysis:
  Track 1 (frames 0–133) was classified as NORMAL activity: ...
  ----------------------------------------------------------------------

  SCENE SUMMARY
  ----------------------------------------------------------------------
  ...
  ======================================================================
"""

from __future__ import annotations

import json
import logging
import os
import textwrap
from datetime import datetime
from typing import Dict, List, Optional

from .explanation_generator import ExplanationOutput

log = logging.getLogger(__name__)

_W = 70   # report width


def _bar(char: str = "=") -> str:
    return char * _W


def _wrap(text: str, indent: int = 0) -> str:
    """Word-wrap a paragraph to fit report width."""
    prefix = " " * indent
    return textwrap.fill(text, width=_W, initial_indent=prefix,
                         subsequent_indent=prefix)


class ReportWriter:
    """
    Collects ExplanationOutput objects and writes the final formatted report.

    Parameters
    ----------
    output_dir : root directory for reports
    video_name : video stem (used as sub-folder)
    """

    def __init__(
        self,
        output_dir: str = os.path.join("output", "reports"),
        video_name: str = "run",
    ) -> None:
        self.run_dir    = os.path.join(output_dir, _sanitise(video_name))
        self.video_name = video_name
        os.makedirs(self.run_dir, exist_ok=True)
        self._reports:    List[ExplanationOutput] = []
        self._start_time  = datetime.now().isoformat()
        log.info("[ReportWriter] Reports → %s", self.run_dir)

    # ──────────────────────────────────────────
    # Accumulate reports
    # ──────────────────────────────────────────

    def save(self, report: ExplanationOutput) -> dict:
        """Add one track report. Returns paths dict after writing JSON."""
        self._reports.append(report)
        json_path = os.path.join(
            self.run_dir,
            f"track{report.track_id:03d}_{_sanitise(report.fine_label)}"
            f"_frame{report.frame_end:05d}.json"
        )
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(report.to_dict(), fh, indent=2, ensure_ascii=False)
        return {"json": json_path}

    def save_batch(self, reports: Dict[int, ExplanationOutput]) -> Dict[int, dict]:
        return {tid: self.save(r) for tid, r in reports.items()}

    # ──────────────────────────────────────────
    # Final report generation
    # ──────────────────────────────────────────

    def write_summary(self, video_source: str = "") -> str:
        """
        Write the full formatted report to all_reports.txt and summary.json.
        Returns path to all_reports.txt.
        """
        txt = self._build_report_text(video_source)

        # Human-readable full report
        txt_path = os.path.join(self.run_dir, "all_reports.txt")
        with open(txt_path, "w", encoding="utf-8") as fh:
            fh.write(txt)

        # Machine-readable summary
        summary = self._build_summary_dict(video_source)
        json_path = os.path.join(self.run_dir, "summary.json")
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)

        print(f"\n[ReportWriter] Full report -> {txt_path}")
        print(f"[ReportWriter] Summary     -> {json_path}")
        try:
            print(txt)   # also print to console
        except UnicodeEncodeError:
            print(txt.encode('ascii', errors='replace').decode('ascii'))

        return txt_path

    # ──────────────────────────────────────────
    # Console helpers
    # ──────────────────────────────────────────

    @staticmethod
    def print_report(report: ExplanationOutput) -> None:
        """Print a single track section to console immediately."""
        print(f"\n{'-'*_W}")
        print(f"Track ID  : {report.track_id}")
        print(f"Frames    : {report.frame_start} – {report.frame_end}")
        print(f"Category  : {'ABNORMAL' if report.is_abnormal else 'NORMAL'}")
        print(f"Action    : {report.fine_label} ({report.fine_conf:.1%})")
        print(f"Risk      : {report.risk_label} {report.risk_emoji}")
        print()
        print("Analysis:")
        try:
            print(_wrap(report.analysis, indent=0))
        except UnicodeEncodeError:
            print(_wrap(report.analysis.encode('ascii', errors='replace').decode('ascii'), indent=0))
        print(f"{'-'*_W}")

    def print_summary(self) -> None:
        """Print console summary of high-risk tracks."""
        high = [r for r in self._reports if r.is_high_risk]
        print(f"\n{'='*_W}")
        print(f"  Reports: {len(self._reports)} tracks | {len(high)} HIGH/CRITICAL")
        for r in high:
            print(f"  [WARN]  Track {r.track_id:3d} | {r.fine_label:20s} | "
                  f"{r.fine_conf:.0%} | {r.risk_label} {r.risk_emoji}")
        print(f"{'='*_W}\n")

    # ──────────────────────────────────────────
    # Private report builders
    # ──────────────────────────────────────────

    def _build_report_text(self, video_source: str) -> str:
        lines: List[str] = []
        now = datetime.now().isoformat()

        # ── Header ─────────────────────────────────────
        lines += [
            _bar("="),
            "         AI SURVEILLANCE ANALYSIS REPORT",
            _bar("="),
            f"Video     : {video_source or self.video_name}",
            f"Generated : {now}",
            _bar("="),
            "",
        ]

        # ── Individual track reports ────────────────────
        lines += ["INDIVIDUAL TRACK REPORTS", _bar("-")]

        sorted_reports = sorted(self._reports, key=lambda r: r.track_id)

        for rpt in sorted_reports:
            category = "ABNORMAL" if rpt.is_abnormal else "NORMAL"
            lines += [
                f"Track ID  : {rpt.track_id}",
                f"Frames    : {rpt.frame_start} \u2013 {rpt.frame_end} ({rpt.t_start:.2f}s \u2013 {rpt.t_end:.2f}s)",
                f"Category  : {category}",
                f"Action    : {rpt.fine_label} ({rpt.fine_conf:.1%})",
                f"Risk      : {rpt.risk_label} {rpt.risk_emoji}",
                "",
                "Analysis:",
            ]
            # Word-wrap the analysis paragraph
            lines.append(_wrap(rpt.analysis, indent=0))
            lines += [_bar("-"), ""]

        # ── Scene summary ────────────────────────────────
        n_total    = len(self._reports)
        n_abnormal = sum(1 for r in self._reports if r.is_abnormal)
        n_normal   = n_total - n_abnormal
        abnormal_actions = list({r.fine_label for r in self._reports if r.is_abnormal})
        overall_risk, overall_emoji = self._overall_risk()

        lines += [
            "SCENE SUMMARY",
            _bar("-"),
        ]

        stem = os.path.basename(str(video_source)) if video_source else self.video_name
        summary_text = (
            f"Scene analysis of '{stem}' completed. "
            f"Detected {n_total} person track(s): "
            f"{n_abnormal} abnormal, {n_normal} normal. "
        )
        if abnormal_actions:
            summary_text += f"Abnormal activities detected: {', '.join(abnormal_actions)}. "
            summary_text += "Immediate security review is recommended."
        else:
            summary_text += "No abnormal activities detected. Scene is clear."

        lines.append(_wrap(summary_text))
        lines += [
            f"Overall Scene Risk: {overall_risk}",
            "Recommended Actions:",
        ]

        if overall_risk in ("CRITICAL", "HIGH"):
            lines += [
                "  1. Archive this footage for record-keeping.",
                "  2. Dispatch security to the location.",
                "  3. Alert site supervisor.",
            ]
        else:
            lines += [
                "  1. Continue standard monitoring.",
                "  2. Archive footage per retention policy.",
            ]

        lines += [_bar("="), ""]
        return "\n".join(lines)

    def _overall_risk(self) -> tuple:
        if any(r.risk_label == "CRITICAL" for r in self._reports):
            return "CRITICAL", "🚨"
        if any(r.risk_label == "HIGH" for r in self._reports):
            return "HIGH", "🔴"
        if any(r.risk_label == "MEDIUM" for r in self._reports):
            return "MEDIUM", "🟡"
        return "LOW", "🟢"

    def _build_summary_dict(self, video_source: str) -> dict:
        overall_risk, emoji = self._overall_risk()
        return {
            "video":          video_source,
            "generated":      datetime.now().isoformat(),
            "total_tracks":   len(self._reports),
            "abnormal_tracks":sum(1 for r in self._reports if r.is_abnormal),
            "overall_risk":   overall_risk,
            "risk_counts": {
                lvl: sum(1 for r in self._reports if r.risk_label == lvl)
                for lvl in ("CRITICAL", "HIGH", "MEDIUM", "LOW")
            },
            "tracks": [r.to_dict() for r in sorted(self._reports, key=lambda r: r.track_id)],
        }


# ─────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────

def _sanitise(name: str) -> str:
    for ch in r'\/:*?"<>| ':
        name = name.replace(ch, "_")
    return name
