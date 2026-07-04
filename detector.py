"""
detector.py — YOLOv8 person detector wrapper.

Wraps Ultralytics YOLOv8 with clean detection output as typed dicts.
Only persons (COCO class 0) are returned.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from typing import List

from ultralytics import YOLO

from config import CFG, DetectorConfig


@dataclass
class Detection:
    """Single person detection in one frame."""
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    track_id: int = -1   # filled in by ByteTrack later

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    def as_tlwh(self) -> np.ndarray:
        """Convert to [top-left-x, top-left-y, w, h] for ByteTrack."""
        return np.array([self.x1, self.y1, self.width, self.height], dtype=np.float32)

    def as_xyxy(self) -> np.ndarray:
        return np.array([self.x1, self.y1, self.x2, self.y2], dtype=np.float32)


class PersonDetector:
    """
    YOLOv8 wrapper that:
      - Loads the model once
      - Returns only person detections above the confidence threshold
      - Is GPU-aware
    """

    def __init__(self, cfg: DetectorConfig = CFG.detector, device: str = "cpu") -> None:
        self.cfg = cfg
        self.device = device
        self._model = YOLO(cfg.model_name)
        self._model.to(device)
        print(f"[Detector] YOLOv8 loaded → {cfg.model_name} on {device}")

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Run inference on a single BGR frame (HWC uint8).

        Returns
        -------
        List[Detection]
            Filtered to persons only, above conf threshold, sorted by confidence desc.
        """
        results = self._model.predict(
            source=frame,
            imgsz=self.cfg.input_size,
            conf=self.cfg.conf_threshold,
            iou=self.cfg.iou_threshold,
            classes=[self.cfg.target_class_id],
            verbose=False,
            device=self.device,
        )

        detections: List[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detections.append(Detection(x1=x1, y1=y1, x2=x2, y2=y2, conf=conf))

        # Sort by confidence descending
        detections.sort(key=lambda d: d.conf, reverse=True)
        return detections
