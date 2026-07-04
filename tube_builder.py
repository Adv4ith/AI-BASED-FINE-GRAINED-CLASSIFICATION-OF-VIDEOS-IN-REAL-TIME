"""
tube_builder.py — Robust action tube management for multi-person ViViT inference.

Design
------
  TrackState      — per-track state machine + frame buffer
  SmoothingBuffer — sliding-window label smoother per track
  TubeManager     — central coordinator; owns all TrackStates

Key features
------------
  ✓ Per-track rolling frame buffer (deque with maxlen)
  ✓ Strict 32-frame gate before first inference
  ✓ Stride-based re-inference every N new frames
  ✓ Ghost-frame padding when a detection is temporarily missing
  ✓ Stale-track eviction after configurable miss tolerance
  ✓ Temporal smoothing via sliding window over past results
  ✓ Min-crop-size guard (rejects tiny / noisy boxes)
  ✓ Max-simultaneous-track cap (drops lowest-confidence extras)
  ✓ Pre-allocated ImageNet normalisation tensors (no re-allocation per frame)
  ✓ O(1) buffer append; uniform temporal sampling for any buffer depth
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

from config import CFG, TubeConfig, ViViTConfig
from tracker import TrackedPerson

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Pre-allocate normalisation constants once (avoid per-frame GC)
# ─────────────────────────────────────────────────────────────
_MEAN = torch.tensor(CFG.vivit.mean).view(3, 1, 1)   # (3,1,1)
_STD  = torch.tensor(CFG.vivit.std ).view(3, 1, 1)   # (3,1,1)


# ─────────────────────────────────────────────────────────────
# Track lifecycle states
# ─────────────────────────────────────────────────────────────
class TrackPhase(Enum):
    FILLING    = auto()   # < min_frames; not ready yet
    READY      = auto()   # >= min_frames; eligible for inference
    COASTING   = auto()   # lost detection; padding with ghost frames
    EVICTED    = auto()   # exceeded miss tolerance; awaiting cleanup


# ─────────────────────────────────────────────────────────────
# Temporal smoothing: sliding window over label + confidence
# ─────────────────────────────────────────────────────────────
@dataclass
class SmoothedPrediction:
    """Smoothed output for one track, updated after each inference call."""
    binary_label: str   = "Unknown"
    binary_conf:  float = 0.0
    fine_label:   str   = "Unknown"
    fine_conf:    float = 0.0
    routed_to:    str   = ""
    is_stable:    bool  = False   # True when window is full and dominant label is consistent


class SmoothingBuffer:
    """
    Maintains a fixed-size window of raw ClassificationResults and
    computes a smoothed prediction by majority vote + averaged confidence.

    Attributes
    ----------
    window : deque of (binary_label, binary_conf, fine_label, fine_conf, routed_to)
    """

    def __init__(self, window_size: int) -> None:
        self._w: Deque[tuple] = deque(maxlen=window_size)
        self._window_size = window_size

    def push(
        self,
        binary_label: str, binary_conf: float,
        fine_label:   str, fine_conf:   float,
        routed_to:    str,
    ) -> None:
        self._w.append((binary_label, binary_conf, fine_label, fine_conf, routed_to))

    def get(self) -> SmoothedPrediction:
        if not self._w:
            return SmoothedPrediction()

        # ── Binary majority vote ─────────────────
        bin_labels = [e[0] for e in self._w]
        bin_winner = max(set(bin_labels), key=bin_labels.count)
        bin_conf   = float(np.mean([e[1] for e in self._w if e[0] == bin_winner]))

        # ── Fine-grained majority vote ───────────
        fine_labels = [e[2] for e in self._w if e[2] != "Unknown"]
        if fine_labels:
            fine_winner = max(set(fine_labels), key=fine_labels.count)
            fine_conf   = float(np.mean([e[3] for e in self._w if e[2] == fine_winner]))
        else:
            fine_winner = "Unknown"
            fine_conf   = 0.0

        # ── Route ────────────────────────────────
        routes     = [e[4] for e in self._w if e[4]]
        routed_to  = max(set(routes), key=routes.count) if routes else ""

        # Stability = window full AND binary label unanimously agrees
        is_stable = (
            len(self._w) == self._window_size
            and len(set(bin_labels)) == 1
        )

        return SmoothedPrediction(
            binary_label=bin_winner,
            binary_conf=bin_conf,
            fine_label=fine_winner,
            fine_conf=fine_conf,
            routed_to=routed_to,
            is_stable=is_stable,
        )

    def clear(self) -> None:
        self._w.clear()

    def __len__(self) -> int:
        return len(self._w)


# ─────────────────────────────────────────────────────────────
# Per-track state
# ─────────────────────────────────────────────────────────────
class TrackState:
    """
    Full lifecycle state for one ByteTrack ID.

    Responsibilities
    ----------------
    - Store raw BGR crops in a bounded deque
    - Track lifecycle phase (FILLING → READY → COASTING → EVICTED)
    - Count frames since last real detection (miss counter)
    - Count frames added since last inference (stride counter)
    - Hold the temporal smoother
    - Build a (1, T, C, H, W) float32 tube tensor on demand
    """

    def __init__(
        self,
        track_id:  int,
        cfg_tube:  TubeConfig,
        cfg_vivit: ViViTConfig,
        created_at_frame: int = 0,
    ) -> None:
        self.track_id          = track_id
        self.cfg_tube          = cfg_tube
        self.cfg_vivit         = cfg_vivit
        self.created_at_frame  = created_at_frame

        # Raw crop store
        self._frames: Deque[np.ndarray] = deque(maxlen=cfg_tube.buffer_size)

        # Lifecycle
        self.phase           = TrackPhase.FILLING
        self.miss_counter    = 0     # consecutive frames without a detection
        self.stride_counter  = 0     # new frames added since last inference

        # Last known crop (used for ghost padding)
        self._last_crop: Optional[np.ndarray] = None

        # Temporal smoother
        self.smoother = SmoothingBuffer(cfg_tube.smoothing_window)

        # Stats
        self.total_frames_seen     = 0
        self.total_inferences_done = 0
        self.last_updated_at       = created_at_frame

    # ── Ingestion ─────────────────────────────

    def add_crop(self, crop: np.ndarray, frame_idx: int) -> None:
        """Push a new real crop. Resets miss counter, advances stride counter."""
        self._frames.append(crop)
        self._last_crop    = crop
        self.miss_counter  = 0
        self.stride_counter += 1
        self.total_frames_seen += 1
        self.last_updated_at   = frame_idx

        # Advance phase
        if self.phase in (TrackPhase.FILLING, TrackPhase.COASTING):
            if len(self._frames) >= self.cfg_tube.min_frames_for_inference:
                self.phase = TrackPhase.READY

    def add_ghost(self) -> None:
        """
        Pad buffer with the last known crop when detection is temporarily missing.
        Only pads if ghost_pad_enabled and we have a prior crop.
        Also promotes FILLING → READY if ghost pads push buffer to threshold.
        """
        if not self.cfg_tube.ghost_pad_enabled:
            return
        if self._last_crop is None:
            return
        self._frames.append(self._last_crop)
        self.stride_counter += 1

        # Ghost pads count toward the fill threshold
        if (
            self.phase == TrackPhase.FILLING
            and len(self._frames) >= self.cfg_tube.min_frames_for_inference
        ):
            self.phase = TrackPhase.READY

    def mark_missed(self) -> None:
        """
        Call once per frame when this track had no matching detection.
        Manages the miss counter and phase transitions.

        Transition table
        ----------------
        miss > tolerance          → EVICTED  (from any phase)
        READY + first miss        → COASTING
        FILLING + first miss      → COASTING (if already has enough frames)
        """
        self.miss_counter += 1
        self.add_ghost()   # may promote FILLING → READY internally

        if self.miss_counter > self.cfg_tube.missing_detection_tolerance:
            self.phase = TrackPhase.EVICTED
        elif self.phase == TrackPhase.READY:
            self.phase = TrackPhase.COASTING
        elif (
            self.phase == TrackPhase.FILLING
            and len(self._frames) >= self.cfg_tube.min_frames_for_inference
        ):
            # Buffer filled via ghost pads — downgrade to COASTING (not READY)
            # because we haven't seen a real detection recently
            self.phase = TrackPhase.COASTING

    # ── Inference gate ────────────────────────

    def is_ready(self) -> bool:
        """True when the buffer has enough frames for classification."""
        return (
            self.phase in (TrackPhase.READY, TrackPhase.COASTING)
            and len(self._frames) >= self.cfg_tube.min_frames_for_inference
        )

    def is_due(self) -> bool:
        """True when it's time to run inference again (stride reached)."""
        return self.stride_counter >= self.cfg_tube.inference_stride

    def is_evicted(self) -> bool:
        return self.phase == TrackPhase.EVICTED

    def mark_inferred(self) -> None:
        """Reset stride counter after inference fires."""
        self.stride_counter = 0
        self.total_inferences_done += 1

    # ── Tube construction ─────────────────────

    def build_tube(self, num_frames: int, image_size: int) -> Optional[torch.Tensor]:
        """
        Sample `num_frames` from the rolling buffer using uniform temporal sampling.

        Pipeline
        --------
        raw BGR crop → resize(224,224) → BGR→RGB → [0,1] float → ImageNet normalise
        → (C, H, W) tensor

        Stacks to (T, C, H, W), unsqueezes batch → (1, T, C, H, W).
        Returns None when buffer has fewer frames than required.
        """
        n_available = len(self._frames)
        if n_available < num_frames:
            return None

        frames = list(self._frames)   # snapshot: oldest → newest

        # Uniform temporal sampling (handles buffer_size != num_frames)
        indices = np.linspace(0, n_available - 1, num_frames, dtype=np.int32)
        sampled = [frames[i] for i in indices]

        processed: List[torch.Tensor] = []
        for crop in sampled:
            if crop is None or crop.size == 0:
                # Safety: insert a black frame rather than crashing
                rgb = np.zeros((image_size, image_size, 3), dtype=np.uint8)
            else:
                resized = cv2.resize(crop, (image_size, image_size),
                                     interpolation=cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            # (H,W,C) uint8 → (C,H,W) float32 [0,1]
            t = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255.0)
            # ImageNet normalise (using pre-allocated constants)
            t = (t - _MEAN) / _STD
            processed.append(t)

        # (T, C, H, W) → (1, T, C, H, W)
        return torch.stack(processed, dim=0).unsqueeze(0)

    # ── Stats / debug ─────────────────────────

    @property
    def buffer_fill(self) -> int:
        return len(self._frames)

    def summary(self) -> str:
        return (
            f"Track {self.track_id:3d} | phase={self.phase.name:9s} | "
            f"buf={self.buffer_fill:3d}/{self.cfg_tube.buffer_size} | "
            f"miss={self.miss_counter:2d} | stride={self.stride_counter:2d} | "
            f"inferences={self.total_inferences_done}"
        )


# ─────────────────────────────────────────────────────────────
# Central Tube Manager
# ─────────────────────────────────────────────────────────────
@dataclass
class TubeReadyBatch:
    """Returned by TubeManager.update() — everything needed for one inference pass."""
    # {track_id: (binary_tube, fine_tube)}   both (1, T, C, H, W)
    tubes:   Dict[int, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]] = field(
        default_factory=dict
    )
    # Track IDs that were evicted (buffer cleared) this frame
    evicted: List[int] = field(default_factory=list)


class TubeManager:
    """
    Central coordinator for all per-track TrackStates.

    Usage
    -----
        mgr = TubeManager()

        # Inside the frame loop:
        batch = mgr.update(frame, persons, frame_idx)
        if batch.tubes:
            results = engine.classify_batch(batch.tubes)
            mgr.push_results(results)    # feeds temporal smoother

        # Get smoothed prediction for display:
        smoothed = mgr.get_smoothed(track_id)
    """

    def __init__(
        self,
        cfg_tube:  TubeConfig  = CFG.tube,
        cfg_vivit: ViViTConfig = CFG.vivit,
    ) -> None:
        self.cfg_tube  = cfg_tube
        self.cfg_vivit = cfg_vivit

        self._states:  Dict[int, TrackState] = {}

        # Stats
        self._total_evictions   = 0
        self._total_inferences  = 0
        self._start_time        = time.perf_counter()

        log.info(
            "[TubeManager] Initialised | buffer=%d | min_frames=%d | "
            "stride=%d | miss_tol=%d | smooth_win=%d | max_tracks=%d",
            cfg_tube.buffer_size,
            cfg_tube.min_frames_for_inference,
            cfg_tube.inference_stride,
            cfg_tube.missing_detection_tolerance,
            cfg_tube.smoothing_window,
            cfg_tube.max_tracks,
        )

    # ══════════════════════════════════════════
    # Primary API
    # ══════════════════════════════════════════

    def update(
        self,
        frame:      np.ndarray,
        persons:    List[TrackedPerson],
        frame_idx:  int,
    ) -> TubeReadyBatch:
        """
        Call once per frame with the current set of tracked persons.

        Steps
        -----
        1. Enforce max-track cap (drop lowest-confidence persons)
        2. For each active person: crop → validate → push to TrackState
        3. For each known track NOT in active persons: mark_missed()
        4. Evict dead tracks (miss > tolerance)
        5. Collect tracks that are ready + due for inference
        6. Build and return TubeReadyBatch

        Parameters
        ----------
        frame     : BGR uint8 full video frame
        persons   : currently tracked persons from ByteTracker
        frame_idx : global frame counter (used for stats)

        Returns
        -------
        TubeReadyBatch
        """
        H, W = frame.shape[:2]

        # ── 1. Cap number of simultaneous tracks ──
        active_persons = self._cap_tracks(persons)
        active_ids     = {p.track_id for p in active_persons}

        # ── 2. Update active tracks ───────────────
        for person in active_persons:
            tid  = person.track_id
            crop = self._extract_crop(frame, person, H, W)
            if crop is None:
                continue

            if tid not in self._states:
                self._states[tid] = TrackState(
                    track_id=tid,
                    cfg_tube=self.cfg_tube,
                    cfg_vivit=self.cfg_vivit,
                    created_at_frame=frame_idx,
                )
                log.debug("[TubeManager] New track %d @ frame %d", tid, frame_idx)

            self._states[tid].add_crop(crop, frame_idx)

        # ── 3. Mark missing tracks ────────────────
        for tid, state in list(self._states.items()):
            if tid not in active_ids:
                state.mark_missed()

        # ── 4. Evict dead tracks ──────────────────
        evicted_ids = self._evict_dead()

        # ── 5. Collect inference-ready tracks ────
        batch = TubeReadyBatch(evicted=evicted_ids)
        for tid, state in self._states.items():
            if not state.is_ready():
                continue
            if not state.is_due():
                continue

            bt = state.build_tube(
                self.cfg_vivit.binary_num_frames,
                self.cfg_vivit.binary_image_size,
            )
            ft = state.build_tube(
                self.cfg_vivit.finegrained_num_frames,
                self.cfg_vivit.finegrained_image_size,
            )
            if bt is not None and ft is not None:
                batch.tubes[tid] = (bt, ft)
                state.mark_inferred()
                self._total_inferences += 1

        return batch

    def push_results(self, results: dict) -> None:
        """
        Feed raw ClassificationResults from the engine back into each
        track's temporal smoother.

        Parameters
        ----------
        results : Dict[track_id, ClassificationResult]
        """
        for tid, res in results.items():
            if tid not in self._states:
                continue
            self._states[tid].smoother.push(
                binary_label = res.binary_label,
                binary_conf  = res.binary_conf,
                fine_label   = res.fine_label,
                fine_conf    = res.fine_conf,
                routed_to    = res.routed_to,
            )

    def get_smoothed(self, track_id: int) -> Optional[SmoothedPrediction]:
        """
        Return the temporally-smoothed prediction for a track, or None if
        the track has no results yet.
        """
        state = self._states.get(track_id)
        if state is None or len(state.smoother) == 0:
            return None
        return state.smoother.get()

    def get_all_smoothed(self) -> Dict[int, SmoothedPrediction]:
        """Return smoothed predictions for every track with at least one result."""
        out: Dict[int, SmoothedPrediction] = {}
        for tid, state in self._states.items():
            if len(state.smoother) > 0:
                out[tid] = state.smoother.get()
        return out

    # ══════════════════════════════════════════
    # Diagnostics
    # ══════════════════════════════════════════

    def active_track_count(self) -> int:
        return len(self._states)

    def print_status(self) -> None:
        """Print per-track summary to stdout (useful for debugging)."""
        print(f"\n[TubeManager] {len(self._states)} active tracks | "
              f"total_inferences={self._total_inferences} | "
              f"total_evictions={self._total_evictions}")
        for state in sorted(self._states.values(), key=lambda s: s.track_id):
            print(f"  {state.summary()}")

    def phase_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {p.name: 0 for p in TrackPhase}
        for s in self._states.values():
            counts[s.phase.name] += 1
        return counts

    def reset(self) -> None:
        """Full reset between videos."""
        self._states.clear()
        self._total_evictions  = 0
        self._total_inferences = 0
        self._start_time       = time.perf_counter()
        log.info("[TubeManager] Reset.")

    # ══════════════════════════════════════════
    # Private helpers
    # ══════════════════════════════════════════

    def _cap_tracks(self, persons: List[TrackedPerson]) -> List[TrackedPerson]:
        """
        If more than max_tracks persons are detected, keep only the
        max_tracks with the highest detection confidence.
        """
        if len(persons) <= self.cfg_tube.max_tracks:
            return persons
        kept = sorted(persons, key=lambda p: p.conf, reverse=True)
        dropped = kept[self.cfg_tube.max_tracks:]
        log.debug(
            "[TubeManager] Cap: keeping %d/%d tracks (dropped IDs %s)",
            self.cfg_tube.max_tracks, len(persons),
            [p.track_id for p in dropped],
        )
        return kept[:self.cfg_tube.max_tracks]

    def _extract_crop(
        self,
        frame:  np.ndarray,
        person: TrackedPerson,
        H: int, W: int,
    ) -> Optional[np.ndarray]:
        """
        Clamp bbox, validate minimum size, slice crop from frame.
        Returns None for invalid or too-small crops.
        """
        x1 = max(0, int(person.x1))
        y1 = max(0, int(person.y1))
        x2 = min(W, int(person.x2))
        y2 = min(H, int(person.y2))

        crop_w = x2 - x1
        crop_h = y2 - y1

        # Reject degenerate boxes
        if crop_w <= 0 or crop_h <= 0:
            return None

        # Reject crops smaller than the minimum pixel threshold
        if crop_w < self.cfg_tube.min_crop_px or crop_h < self.cfg_tube.min_crop_px:
            log.debug(
                "[TubeManager] Track %d: crop too small (%dx%d), skipping",
                person.track_id, crop_w, crop_h,
            )
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        return crop

    def _evict_dead(self) -> List[int]:
        """
        Remove EVICTED tracks from the state dict.
        Returns list of evicted track IDs.
        """
        dead = [tid for tid, s in self._states.items() if s.is_evicted()]
        for tid in dead:
            log.debug("[TubeManager] Evicting track %d (miss=%d)",
                      tid, self._states[tid].miss_counter)
            del self._states[tid]
        self._total_evictions += len(dead)
        return dead


# ─────────────────────────────────────────────────────────────
# Backward-compat shim so pipeline.py doesn't need changes
# ─────────────────────────────────────────────────────────────
# pipeline.py uses `TubeBuilder`; alias TubeManager to it.
TubeBuilder = TubeManager
