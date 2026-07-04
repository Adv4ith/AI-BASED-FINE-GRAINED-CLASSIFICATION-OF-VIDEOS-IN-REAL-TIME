"""
tracker.py — ByteTrack multi-object tracker wrapper.

Uses the 'supervision' library's ByteTrack implementation,
which does not require a separate C++ build.

pip install supervision>=0.21.0
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

try:
    import supervision as sv
    _HAS_SUPERVISION = True
except ImportError:
    _HAS_SUPERVISION = False

from detector import Detection
from config import CFG, TrackerConfig


@dataclass
class TrackedPerson:
    """One tracked person with their current bounding box and history."""
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2


class ByteTracker:
    """
    Thin wrapper around supervision's ByteTracker.

    Converts List[Detection] → List[TrackedPerson] with stable IDs across frames.
    """

    def __init__(self, cfg: TrackerConfig = CFG.tracker) -> None:
        if not _HAS_SUPERVISION:
            raise ImportError(
                "supervision is not installed. Run: pip install supervision>=0.21.0"
            )
        self.cfg = cfg
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            self._tracker = sv.ByteTrack(
                track_activation_threshold=cfg.track_thresh,
                lost_track_buffer=cfg.track_buffer,
                minimum_matching_threshold=cfg.match_thresh,
                frame_rate=cfg.frame_rate,
            )
        print(f"[Tracker] ByteTrack initialised (thresh={cfg.track_thresh}, "
              f"buffer={cfg.track_buffer}f, fps={cfg.frame_rate})")

    def update(
        self,
        detections: List[Detection],
        frame_shape: tuple[int, int],   # (H, W)
    ) -> List[TrackedPerson]:
        """
        Update tracker with new detections for the current frame.

        Parameters
        ----------
        detections : List[Detection]
            Raw person detections from YOLOv8.
        frame_shape : (H, W)
            Frame dimensions (needed by supervision for boundary checks).

        Returns
        -------
        List[TrackedPerson]
            Active tracked persons with stable IDs.
        """
        if not detections:
            # Still call update so tracks age correctly
            sv_dets = sv.Detections.empty()
        else:
            xyxy  = np.array([d.as_xyxy() for d in detections], dtype=np.float32)
            confs = np.array([d.conf      for d in detections], dtype=np.float32)
            class_ids = np.zeros(len(detections), dtype=int)
            sv_dets = sv.Detections(xyxy=xyxy, confidence=confs, class_id=class_ids)

        tracked = self._tracker.update_with_detections(sv_dets)

        results: List[TrackedPerson] = []
        if tracked is None or len(tracked) == 0:
            return results

        for i in range(len(tracked)):
            x1, y1, x2, y2 = tracked.xyxy[i]
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 0.5
            tid  = int(tracked.tracker_id[i])
            results.append(TrackedPerson(
                track_id=tid, x1=x1, y1=y1, x2=x2, y2=y2, conf=conf
            ))

        return results

    def reset(self) -> None:
        """Reset tracker state (call between videos)."""
        self._tracker.reset()
