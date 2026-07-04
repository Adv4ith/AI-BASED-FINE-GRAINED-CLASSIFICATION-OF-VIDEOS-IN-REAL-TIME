"""
pipeline.py — Main orchestrator for the hierarchical ViViT inference pipeline.

Ties together:
  PersonDetector (YOLOv8)
    -> ByteTracker
    -> TubeManager  (robust per-track buffering, ghost padding, smoothing)
    -> ViViTInferenceEngine (binary -> normal/abnormal routing)
    -> Visualiser (overlay rendering)
    -> LLM Reporting (optional)
    -> VideoWriter (optional output)

Usage
-----
  from pipeline import HierarchicalPipeline

  pipeline = HierarchicalPipeline()
  pipeline.run("input.mp4", "output.mp4")
"""

from __future__ import annotations

import os
import time
import cv2
import torch
import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional, Dict, List

from config import CFG, PipelineConfig
from detector import PersonDetector
from tracker import ByteTracker
from tube_builder import TubeManager, SmoothedPrediction
from vivit_model import ViViTInferenceEngine, ClassificationResult
from visualiser import Visualiser
from llm import LlamaEngine, SurveillanceExplainer, ReportWriter


# ──────────────────────────────────────────────
# Device resolution helper
# ──────────────────────────────────────────────

def _resolve_device(device_str: str) -> str:
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


# ──────────────────────────────────────────────
# HierarchicalPipeline
# ──────────────────────────────────────────────

class HierarchicalPipeline:
    """
    End-to-end hierarchical inference pipeline.

    Model loading happens once at init. Call run() for each video.
    """

    def __init__(
        self,
        cfg:         PipelineConfig = CFG,
        enable_llm:  bool = False,
        llm_model:   str  = "meta-llama/Llama-3.1-8B-Instruct",
        llm_4bit:    bool = False,
        hf_token:    str  = None,
        llm_use_api: bool = True,
    ) -> None:
        self.cfg    = cfg
        self.device = _resolve_device(cfg.inference.device)

        print(f"\n{'='*60}")
        print(f"  Hierarchical ViViT Pipeline  |  device={self.device}")
        print(f"{'='*60}\n")

        # ── Core vision components ──────────────
        self.detector = PersonDetector(cfg.detector, device=self.device)
        self.tracker  = ByteTracker(cfg.tracker)
        self.manager  = TubeManager(cfg.tube, cfg.vivit)
        self.engine   = ViViTInferenceEngine(cfg.paths, cfg.vivit, device=self.device)
        self.vis      = Visualiser(cfg.vis)

        # Per-track frame range: {track_id: (first_frame, last_frame)}
        self._track_frame_ranges: Dict[int, tuple] = {}

        # GradCAM: rolling full-frame buffer per track (last N frames + bbox)
        # {tid: deque of (full_frame, (x1,y1,x2,y2))}
        _nf = getattr(cfg.vivit, 'finegrained_num_frames', 32)
        self._full_frame_buf: Dict[int, deque] = {}
        self._buf_maxlen = _nf
        # Accumulated clips: {tid: [{cam_3d, frames_bboxes, label, conf, frame_start}]}
        self._gradcam_clips: Dict[int, List[dict]] = {}

        # ── LLM explanation layer (optional) ────
        self._llm_enabled    = enable_llm
        self._llm_engine:    Optional[LlamaEngine]           = None
        self._llm_explainer: Optional[SurveillanceExplainer] = None
        self._report_writer: Optional[ReportWriter]          = None

        if enable_llm:
            self._llm_engine = LlamaEngine(
                model_name   = llm_model,
                use_api      = llm_use_api,
                device       = "auto",
                load_in_4bit = llm_4bit,
                hf_token     = hf_token,
            )
            mode = "HF API (no download)" if llm_use_api else "LOCAL model"
            print(f"  [LLM] Llama engine: {mode} | {llm_model}")
            self._llm_explainer = SurveillanceExplainer(
                engine      = self._llm_engine,
                explain_all = True,
                deduplicate = True,
            )
            print(f"  [LLM] Model loads lazily on first inference.")

        print(f"\n[Pipeline] All components ready.\n")

    # ──────────────────────────────────────────
    # Main entry: process a full video
    # ──────────────────────────────────────────

    def run(
        self,
        source:       str | int,
        output_path:  Optional[str] = None,
        show_preview: bool          = False,
        max_frames:   int           = 0,
    ) -> Dict[int, ClassificationResult]:
        """
        Process a video file (or webcam index) end-to-end.

        Parameters
        ----------
        source       : path to video file, or webcam index (0, 1, ...)
        output_path  : where to save the annotated video (None = no save)
        show_preview : whether to display frames in an OpenCV window
        max_frames   : stop after N frames (0 = process entire video)

        Returns
        -------
        Dict[track_id -> final ClassificationResult]
        """
        self._reset_state()

        # ── Open video source ───────────────────
        cap = cv2.VideoCapture(source if isinstance(source, int) else str(source))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        fps_in  = cap.get(cv2.CAP_PROP_FPS) or 30.0
        W       = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        H       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Pipeline] Source : {source}")
        print(f"[Pipeline] Size   : {W}x{H}  |  {fps_in:.1f} fps  |  {total_f} frames")

        # ── Open video writer ───────────────────
        writer   = None
        out_path = None
        if output_path or self.cfg.output.save_video:
            out_path = output_path or self._default_output_path(source)
            os.makedirs(
                os.path.dirname(out_path) if os.path.dirname(out_path) else ".",
                exist_ok=True,
            )
            fourcc = cv2.VideoWriter_fourcc(*self.cfg.output.fourcc)
            writer = cv2.VideoWriter(out_path, fourcc, fps_in, (W, H))
            print(f"[Pipeline] Output : {out_path}")

        # Sync tracker fps to actual video fps
        self.cfg.tracker.frame_rate = int(fps_in)

        all_raw_results: Dict[int, ClassificationResult] = {}
        frame_idx = 0
        t_start   = time.perf_counter()

        # ── Init LLM report writer for this run ──
        if self._llm_enabled and self._llm_explainer is not None:
            video_stem = Path(str(source)).stem if isinstance(source, (str, Path)) else f"webcam_{source}"
            self._report_writer = ReportWriter(
                output_dir = os.path.join(self.cfg.output.output_dir, "reports"),
                video_name = video_stem,
            )
            self._llm_explainer.reset()

        # ── Frame loop ──────────────────────────
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if max_frames and frame_idx >= max_frames:
                    break

                # ── Stage 1: YOLOv8 detection ─────
                detections = self.detector.detect(frame)

                # ── Stage 2: ByteTrack tracking ───
                persons = self.tracker.update(detections, frame_shape=(H, W))

                # Track first/last frame per track + buffer full frames for GradCAM
                for p in persons:
                    tid = p.track_id
                    if tid not in self._track_frame_ranges:
                        self._track_frame_ranges[tid] = (frame_idx, frame_idx)
                    else:
                        fs, _ = self._track_frame_ranges[tid]
                        self._track_frame_ranges[tid] = (fs, frame_idx)
                    # Rolling buffer: store full frame + person bbox
                    if tid not in self._full_frame_buf:
                        self._full_frame_buf[tid] = deque(maxlen=self._buf_maxlen)
                    bbox = (int(p.x1), int(p.y1), int(p.x2), int(p.y2))
                    self._full_frame_buf[tid].append((frame.copy(), bbox))

                # ── Stage 3: TubeManager update ───
                batch = self.manager.update(frame, persons, frame_idx)

                if batch.evicted:
                    for evicted_tid in batch.evicted:
                        self.engine.reset_best_window(evicted_tid)
                    print(f"  [Frame {frame_idx:5d}] Evicted tracks: {batch.evicted}")

                # ── Stage 4: ViViT inference ───────
                if batch.tubes:
                    raw_results = self.engine.classify_batch(
                        batch.tubes,
                        binary_conf_thresh = self.cfg.inference.binary_confidence_threshold,
                    )

                    self.manager.push_results(raw_results)
                    all_raw_results.update(raw_results)

                    # Capture GradCAM clips for tracks that have a heatmap
                    for tid, res in raw_results.items():
                        if res.has_heatmap and tid in self._full_frame_buf:
                            frames_bboxes = list(self._full_frame_buf[tid])
                            self._gradcam_clips.setdefault(tid, []).append({
                                "cam_3d":        res.cam_3d,
                                "frames_bboxes": frames_bboxes,
                                "label":         res.fine_label,
                                "conf":          res.fine_conf,
                                "frame_start":   frame_idx,
                            })

                    # ── LLM Explanation ────────────────────────────────────
                    if self._llm_enabled and self._llm_explainer is not None:
                        llm_reports = self._llm_explainer.explain_batch(
                            raw_results  = raw_results,
                            cam_map      = {},
                            frame_idx    = frame_idx,
                            frame_ranges = self._track_frame_ranges,
                            fps          = fps_in,
                        )
                        for tid, rpt in llm_reports.items():
                            ReportWriter.print_report(rpt)
                            if self._report_writer is not None:
                                self._report_writer.save(rpt)
                        if llm_reports:
                            self.vis.update_llm_reports(llm_reports)

                    # Log to console
                    for tid, res in raw_results.items():
                        smoothed = self.manager.get_smoothed(tid)
                        s_label  = smoothed.fine_label if smoothed else res.fine_label
                        s_conf   = smoothed.fine_conf  if smoothed else res.fine_conf
                        stable   = "OK" if (smoothed and smoothed.is_stable) else "~"
                        print(
                            f"  [Frame {frame_idx:5d}] Track {tid:3d} | "
                            f"S1: {res.binary_label}({res.binary_conf:.0%}) "
                            f"-> S2: {res.fine_label}({res.fine_conf:.0%}) "
                            f"| smooth({stable}): {s_label}({s_conf:.0%})"
                        )

                # ── Stage 5: Update visualiser ─────
                self._push_smoothed_to_vis(self.manager.get_all_smoothed())

                # ── Stage 6: Draw overlay ──────────
                annotated = self.vis.draw(frame, persons, frame_idx, show_fps=True)

                if writer is not None:
                    writer.write(annotated)

                if show_preview:
                    cv2.imshow("Hierarchical ViViT Pipeline", annotated)
                    key = cv2.waitKey(1)
                    if key in (27, ord("q")):
                        print("[Pipeline] User quit.")
                        break

                frame_idx += 1

                # Progress log every 100 frames
                if frame_idx % 100 == 0:
                    elapsed  = time.perf_counter() - t_start
                    proc_fps = frame_idx / max(elapsed, 1e-6)
                    phases   = self.manager.phase_counts()
                    print(
                        f"[Pipeline] {frame_idx}/{total_f} frames | "
                        f"{proc_fps:.1f} fps | "
                        f"tracks: {self.manager.active_track_count()} "
                        f"(FILLING={phases.get('FILLING',0)} "
                        f"READY={phases.get('READY',0)} "
                        f"COASTING={phases.get('COASTING',0)})"
                    )

        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()
            if self._llm_enabled and self._report_writer is not None:
                self._report_writer.print_summary()
                self._report_writer.write_summary(video_source=str(source))

        elapsed = time.perf_counter() - t_start
        print(f"\n[Pipeline] Done. {frame_idx} frames in {elapsed:.1f}s "
              f"({frame_idx / max(elapsed, 1e-6):.1f} fps avg)")

        if out_path:
            print(f"[Pipeline] Saved >> {out_path}")

        # Post-loop: write GradCAM heatmap videos
        self._write_gradcam_videos(source, out_path, fps_in)

        return all_raw_results

    # ──────────────────────────────────────────
    # Real-time: process a single frame
    # ──────────────────────────────────────────

    def process_frame(
        self,
        frame:     np.ndarray,
        frame_idx: int,
    ) -> tuple[np.ndarray, Dict[int, ClassificationResult]]:
        """
        Process one frame and return (annotated_frame, new_raw_results).
        For real-time / streaming use cases.
        """
        H, W = frame.shape[:2]

        detections = self.detector.detect(frame)
        persons    = self.tracker.update(detections, frame_shape=(H, W))
        batch      = self.manager.update(frame, persons, frame_idx)

        new_raw: Dict[int, ClassificationResult] = {}
        if batch.tubes:
            new_raw = self.engine.classify_batch(
                batch.tubes,
                binary_conf_thresh=self.cfg.inference.binary_confidence_threshold,
            )
            self.manager.push_results(new_raw)

        self._push_smoothed_to_vis(self.manager.get_all_smoothed())
        annotated = self.vis.draw(frame, persons, frame_idx)
        return annotated, new_raw

    # ──────────────────────────────────────────
    # Helpers
    # ──────────────────────────────────────────

    def _push_smoothed_to_vis(
        self, smoothed: Dict[int, SmoothedPrediction]
    ) -> None:
        from vivit_model import ClassificationResult as CR
        vis_results: Dict[int, CR] = {}
        for tid, sp in smoothed.items():
            cr = CR(
                track_id     = tid,
                binary_label = sp.binary_label,
                binary_conf  = sp.binary_conf,
                fine_label   = sp.fine_label,
                fine_conf    = sp.fine_conf,
                routed_to    = sp.routed_to,
            )
            vis_results[tid] = cr
        self.vis.update_results(vis_results)

    def _reset_state(self) -> None:
        """Full reset between video runs."""
        self.tracker.reset()
        self.manager.reset()
        self.vis.reset()
        self._track_frame_ranges.clear()
        self._full_frame_buf.clear()
        self._gradcam_clips.clear()

    def _default_output_path(self, source: str | int) -> str:
        from datetime import datetime
        out_dir   = self.cfg.output.output_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if isinstance(source, int):
            name = f"webcam_{source}_{timestamp}_output.mp4"
        else:
            stem = Path(str(source)).stem
            name = f"{stem}_{timestamp}_output.mp4"
        return os.path.join(out_dir, name)

    # ──────────────────────────────────────────
    # GradCAM video writer (post-loop)
    # ──────────────────────────────────────────

    def _write_gradcam_videos(
        self, source: str | int, out_path: Optional[str], fps: float
    ) -> None:
        """
        After the main frame loop:
        - For each track that has GradCAM clips, render a full-frame
          side-by-side heatmap video (original | heatmap overlay).
        - Patch all clips together into one compiled video.

        Output:
          output/heatmaps/<stem>_track007_Punching_gradcam.mp4   per track
          output/heatmaps/<stem>_gradcam_compiled.mp4            all tracks
        """
        if not self._gradcam_clips:
            return

        out_dir = os.path.join(
            os.path.dirname(out_path) if out_path and os.path.dirname(out_path)
            else self.cfg.output.output_dir,
            "heatmaps"
        )
        os.makedirs(out_dir, exist_ok=True)

        # Derive stem from out_path so heatmap files share the same timestamp
        # e.g. out_path = ".../ab_test_20260601_014623_output.mp4"
        #      stem     = "ab_test_20260601_014623_output"
        if out_path:
            stem = Path(out_path).stem          # inherits timestamp
        elif not isinstance(source, int):
            stem = Path(str(source)).stem
        else:
            stem = f"webcam_{source}"

        tub_size = getattr(self.cfg.vivit, 'tubelet_size', 2)
        compiled: List[np.ndarray] = []   # all frames for final compiled video
        frame_size: Optional[tuple] = None

        print(f"\n[GradCAM] Writing heatmap videos for "
              f"{len(self._gradcam_clips)} track(s) ...")

        for tid, clips in self._gradcam_clips.items():
            track_rendered: List[np.ndarray] = []

            for clip in clips:
                cam_3d        = clip["cam_3d"]          # (T_temp, 14, 14)
                frames_bboxes = clip["frames_bboxes"]   # [(frame, bbox), ...]
                label         = clip["label"]
                conf          = clip["conf"]
                T             = cam_3d.shape[0]

                for i, (frm, bbox) in enumerate(frames_bboxes):
                    t   = min(i // tub_size, T - 1)
                    cam = cam_3d[t]                     # (14,14)
                    rendered = self._render_comparison_frame(
                        frame=frm, cam=cam, bbox=bbox,
                        class_name=label,
                        conf=conf,
                        frame_no=i,
                    )
                    track_rendered.append(rendered)
                    if frame_size is None:
                        frame_size = (rendered.shape[1], rendered.shape[0])  # (W,H)

            if not track_rendered:
                continue

            # Add a 1-second title card between tracks in compiled video
            if compiled and frame_size:
                card = self._title_card(
                    f"Track {tid}  |  {clips[-1]['label']}  ({clips[-1]['conf']:.0%})",
                    frame_size, fps
                )
                compiled.extend(card)

            compiled.extend(track_rendered)

            # Write per-track video
            safe_label = clips[-1]['label'].replace(' ', '_')
            vid_path = os.path.join(
                out_dir, f"{stem}_track{tid:03d}_{safe_label}_gradcam.mp4"
            )
            self._write_video(track_rendered, vid_path, fps, frame_size)
            print(f"  [GradCAM] Track {tid:3d} >> {vid_path}")

        # Write compiled video
        if compiled and frame_size:
            comp_path = os.path.join(out_dir, f"{stem}_gradcam_compiled.mp4")
            self._write_video(compiled, comp_path, fps, frame_size)
            print(f"  [GradCAM] Compiled >> {comp_path}")

        print(f"[GradCAM] Heatmap videos saved to: {out_dir}")

    @staticmethod
    def _render_comparison_frame(
        frame:      np.ndarray,
        cam:        np.ndarray,
        bbox:       tuple,
        class_name: str,
        conf:       float,
        frame_no:   int,
        target_size: int = 256,
        alpha:      float = 0.55,
        colormap:   int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Produce a side-by-side comparison frame matching the academic reference style:
        - Left Panel: Clean original person crop labeled 'Original - <action>'
        - Right Panel: Person crop with dynamic GradCAM heatmap overlaid labeled 'Prediction - <action>'
        Both panels are framed with a white border and labeled in clean centered black text.
        """
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bx1 = max(0, int(x1))
        by1 = max(0, int(y1))
        bx2 = min(W, int(x2))
        by2 = min(H, int(y2))

        crop = frame[by1:by2, bx1:bx2].copy()
        if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
            crop = np.zeros((target_size, target_size, 3), dtype=np.uint8)

        # Resize crop to standard target size
        crop_resized = cv2.resize(crop, (target_size, target_size), interpolation=cv2.INTER_CUBIC)

        # ── 1. Original Panel Canvas (white margins) ──────────────────
        orig_canvas = np.ones((target_size + 40, target_size, 3), dtype=np.uint8) * 255
        orig_canvas[40:, :] = crop_resized

        label_txt = class_name.lower().replace("_", " ")
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness = 1
        
        # Center title text above original crop
        orig_title = f"Original - {label_txt}"
        (tw, th), _ = cv2.getTextSize(orig_title, font, font_scale, thickness)
        tx = max(2, (target_size - tw) // 2)
        cv2.putText(orig_canvas, orig_title, (tx, 24), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        # ── 2. Prediction Panel Canvas with Heatmap ────────────────────
        cam_u8 = (cam * 255).astype(np.uint8)
        cam_up = cv2.resize(cam_u8, (target_size, target_size), interpolation=cv2.INTER_CUBIC)
        
        # Apply square root peak sharpening
        cam_up_float = cam_up.astype(np.float32) / 255.0
        cam_up_float = np.sqrt(cam_up_float)
        cam_up_float = np.clip(cam_up_float, 0.0, 1.0)
        
        # Generate colormap
        heat = cv2.applyColorMap((cam_up_float * 255).astype(np.uint8), colormap)

        # Dynamic pixel-wise alpha blend over the resized crop
        crop_float = crop_resized.astype(np.float32)
        heat_float = heat.astype(np.float32)
        
        # Use constant alpha blend over the entire crop so that low intensity areas
        # (which map to dark blue in JET colormap) overlay a beautiful blue tint on the background.
        blended = crop_float * (1.0 - alpha) + heat_float * alpha
        pred_crop = np.clip(blended, 0, 255).astype(np.uint8)

        # Generate vertical colorbar panel of width 90
        colorbar_w = 90
        colorbar_panel = np.ones((target_size, colorbar_w, 3), dtype=np.uint8) * 255
        
        # Colorbar dimensions
        cb_h = 160
        cb_w = 18
        cb_x1 = (colorbar_w - cb_w) // 2
        cb_x2 = cb_x1 + cb_w
        cb_y1 = (target_size - cb_h) // 2
        cb_y2 = cb_y1 + cb_h
        
        # Generate vertical gradient (255 down to 0, so red is at the top, blue at the bottom)
        gradient = np.linspace(255, 0, cb_h, dtype=np.uint8).reshape(cb_h, 1)
        gradient_bgr = cv2.applyColorMap(gradient, colormap)
        gradient_block = np.tile(gradient_bgr, (1, cb_w, 1))
        colorbar_panel[cb_y1:cb_y2, cb_x1:cb_x2] = gradient_block
        
        # Draw thin border around colorbar
        cv2.rectangle(colorbar_panel, (cb_x1 - 1, cb_y1 - 1), (cb_x2, cb_y2), (80, 80, 80), 1)
        
        # Colorbar text labels (High Intensity / Low Intensity)
        font_cb = cv2.FONT_HERSHEY_SIMPLEX
        font_scale_cb = 0.36
        thickness_cb = 1
        color_cb = (0, 0, 0)
        
        # High Intensity text
        txt_high1 = "High"
        txt_high2 = "Intensity"
        (w1, h1), _ = cv2.getTextSize(txt_high1, font_cb, font_scale_cb, thickness_cb)
        (w2, h2), _ = cv2.getTextSize(txt_high2, font_cb, font_scale_cb, thickness_cb)
        cv2.putText(colorbar_panel, txt_high1, (colorbar_w // 2 - w1 // 2, cb_y1 - 22), font_cb, font_scale_cb, color_cb, thickness_cb, cv2.LINE_AA)
        cv2.putText(colorbar_panel, txt_high2, (colorbar_w // 2 - w2 // 2, cb_y1 - 8), font_cb, font_scale_cb, color_cb, thickness_cb, cv2.LINE_AA)
        
        # Low Intensity text
        txt_low1 = "Low"
        txt_low2 = "Intensity"
        (w3, h3), _ = cv2.getTextSize(txt_low1, font_cb, font_scale_cb, thickness_cb)
        (w4, h4), _ = cv2.getTextSize(txt_low2, font_cb, font_scale_cb, thickness_cb)
        cv2.putText(colorbar_panel, txt_low1, (colorbar_w // 2 - w3 // 2, cb_y2 + 16), font_cb, font_scale_cb, color_cb, thickness_cb, cv2.LINE_AA)
        cv2.putText(colorbar_panel, txt_low2, (colorbar_w // 2 - w4 // 2, cb_y2 + 30), font_cb, font_scale_cb, color_cb, thickness_cb, cv2.LINE_AA)

        # Stack prediction crop and colorbar horizontally
        pred_crop_with_cb = np.hstack((pred_crop, colorbar_panel))

        pred_canvas = np.ones((target_size + 40, target_size + colorbar_w, 3), dtype=np.uint8) * 255
        pred_canvas[40:, :] = pred_crop_with_cb

        # Center title text above prediction crop (including colorbar)
        pred_title = f"Prediction - {label_txt}"
        (tw, th), _ = cv2.getTextSize(pred_title, font, font_scale, thickness)
        tx = max(2, (target_size + colorbar_w - tw) // 2)
        cv2.putText(pred_canvas, pred_title, (tx, 24), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        # ── 3. Assemble side-by-side layout with white separator ───────
        sep = np.ones((target_size + 40, 8, 3), dtype=np.uint8) * 255
        combined = np.hstack((orig_canvas, sep, pred_canvas))
        return combined

    @staticmethod
    def _render_heatmap_frame(
        frame:    np.ndarray,   # full BGR video frame  (H, W, 3)
        cam:      np.ndarray,   # (14, 14) float32 in [0, 1]  — one temporal slice
        bbox:     tuple,        # (x1, y1, x2, y2) person bbox in full frame coords
        label:    str  = "",
        frame_no: int  = 0,
        alpha:    float = 0.55,
        colormap: int   = cv2.COLORMAP_JET,
        dim_bg:   float = 0.40,   # how much to darken non-person area
    ) -> np.ndarray:
        """
        Produce a FULL-FRAME GradCAM visualisation for one action frame.

        What it does
        ─────────────
        1. Dim the entire frame (non-action background → 40% brightness)
        2. Upsample cam (14×14) → person bbox size  (preserves spatial meaning:
           ViViT processed a 224×224 crop of this person, so cam maps to that crop)
        3. Gaussian smooth the upsampled cam to remove bicubic artifacts
        4. Alpha-blend heatmap over the person bbox on the full frame
        5. Draw cyan contour around the hottest activation region
        6. Restore the person's original pixels at 100% brightness outside the
           heatmap so the scene context is preserved
        7. Overlay HUD: label banner, frame counter, saliency score bar

        Result: full scene frame where the camera model's attention is visible
        as a coloured heatmap exactly where the action is happening.
        """
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        bx1 = max(0, x1);  by1 = max(0, y1)
        bx2 = min(W, x2);  by2 = min(H, y2)
        bw  = max(bx2 - bx1, 1)
        bh  = max(by2 - by1, 1)

        # ── 1. Dim background (everything outside the bbox) ───────────
        out = (frame.astype(np.float32) * dim_bg).astype(np.uint8)

        # ── 2. Restore person region at full brightness ────────────────
        out[by1:by2, bx1:bx2] = frame[by1:by2, bx1:bx2]

        # ── 3. Upsample cam (14×14) → bbox size ───────────────────────
        cam_u8  = (cam * 255).astype(np.uint8)
        cam_up  = cv2.resize(cam_u8, (bw, bh), interpolation=cv2.INTER_CUBIC)
        cam_up  = np.clip(cam_up.astype(np.float32) / 255.0, 0.0, 1.0)

        # ── 4. Project onto a full-frame canvas and smooth boundaries ──
        full_cam = np.zeros((H, W), dtype=np.float32)
        full_cam[by1:by2, bx1:bx2] = cam_up

        # Smooth transition edges globally so the heatmap blends into the scene
        k = max(5, (min(W, H) // 16) | 1)
        full_cam = cv2.GaussianBlur(full_cam, (k, k), 0)
        full_cam = np.clip(full_cam, 0.0, 1.0)

        # Renormalise full-frame map so the peak activation equals 1.0 (if any exist)
        hi_val = full_cam.max()
        if hi_val > 0:
            full_cam = full_cam / (hi_val + 1e-8)

        # Generate colormap over full frame
        heat_u8 = (full_cam * 255).astype(np.uint8)
        heat = cv2.applyColorMap(heat_u8, colormap)

        # Perform dynamic pixel-wise alpha blend over the full dimmed frame
        full_cam_3d = np.expand_dims(full_cam, axis=-1)
        dynamic_alpha = full_cam_3d * alpha

        out_float = out.astype(np.float32)
        heat_float = heat.astype(np.float32)
        blended = out_float * (1.0 - dynamic_alpha) + heat_float * dynamic_alpha
        out = np.clip(blended, 0, 255).astype(np.uint8)

        # ── 5. Cyan contour around hottest activation region ──────────
        thresh  = max(int(cam_up.max() * 255 * 0.65), 1)
        cam_local_u8 = (cam_up * 255).astype(np.uint8)
        _, mask = cv2.threshold(cam_local_u8, thresh, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            shifted = [c + np.array([[[bx1, by1]]]) for c in contours]
            cv2.drawContours(out, shifted, -1, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw bbox border in white
        cv2.rectangle(out, (bx1, by1), (bx2, by2), (255, 255, 255), 1)

        # ── 6. HUD ────────────────────────────────────────────────────
        score = float(cam.max())

        # Label banner (top-left)
        banner_h = 26
        cv2.rectangle(out, (0, 0), (W, banner_h), (10, 10, 10), -1)
        cv2.putText(out, label, (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(out, label, (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)

        # Frame counter (top-right)
        fc_txt = f"f={frame_no:03d}"
        (fw, _), _ = cv2.getTextSize(fc_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(out, fc_txt, (W - fw - 6, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # Saliency score bar (bottom strip)
        bar_h = max(4, H // 60)
        fill  = int(score * W)
        col   = (0, 200, 80) if score < 0.33 else (0, 165, 255) if score < 0.66 else (0, 50, 220)
        out[-bar_h:] = (20, 20, 20)
        out[-bar_h:, :fill] = col

        # Score text
        sc_txt = f"saliency={score:.2f}"
        cv2.putText(out, sc_txt, (6, H - bar_h - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1, cv2.LINE_AA)

        return out

    @staticmethod
    def _title_card(
        text: str, frame_size: tuple, fps: float, duration: float = 1.0
    ) -> List[np.ndarray]:
        """Generate N frames of a black title card with centred text."""
        W, H  = frame_size
        n     = max(1, int(fps * duration))
        card  = np.zeros((H, W, 3), dtype=np.uint8)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cx = (W - tw) // 2;  cy = H // 2 + th // 2
        cv2.putText(card, text, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
        return [card] * n

    @staticmethod
    def _write_video(
        frames: List[np.ndarray], path: str, fps: float, size: tuple
    ) -> None:
        """
        Write BGR frames to a video file.
        Uses MJPG (.avi) — guaranteed to work with OpenCV on Windows without
        any extra codec installation. Falls back to mp4v if MJPG fails.
        """
        if not frames:
            return

        W, H = size

        # MJPG in AVI container works on every Windows machine with OpenCV
        avi_path = path.replace(".mp4", ".avi")
        writer   = cv2.VideoWriter(
            avi_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            fps, (W, H)
        )

        if not writer.isOpened():
            # Last resort: mp4v
            writer = cv2.VideoWriter(
                path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps, (W, H)
            )

        for f in frames:
            if f.shape[1] != W or f.shape[0] != H:
                f = cv2.resize(f, (W, H))
            writer.write(f)
        writer.release()
        print(f"      saved: {avi_path if os.path.exists(avi_path) else path}")

    # ──────────────────────────────────────────
    # Debug helpers
    # ──────────────────────────────────────────

    def print_track_status(self) -> None:
        """Print all track buffer states (call anytime inside a loop)."""
        self.manager.print_status()
