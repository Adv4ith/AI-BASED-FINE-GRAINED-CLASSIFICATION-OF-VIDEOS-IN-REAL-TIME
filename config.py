"""
config.py — Centralised configuration for the hierarchical ViViT inference pipeline.
Edit ONLY this file to change paths, thresholds, or class names.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict

# ─────────────────────────────────────────────
# Project root (resolve relative to this file)
# ─────────────────────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(ROOT_DIR, "model")


# ─────────────────────────────────────────────
# Model Paths
# ─────────────────────────────────────────────
@dataclass
class ModelPaths:
    binary:   str = os.path.join(MODEL_DIR, "binary.pth")
    normal:   str = os.path.join(MODEL_DIR, "normal_vivit_32f.pth")
    abnormal: str = os.path.join(MODEL_DIR, "abnormal_vivit_32f.pth")


# ─────────────────────────────────────────────
# ViViT frame / resolution settings
# ─────────────────────────────────────────────
@dataclass
class ViViTConfig:
    # Binary model
    binary_num_frames: int = 32        # frames fed to binary ViViT
    binary_image_size: int = 224

    # Fine-grained models (normal + abnormal)
    finegrained_num_frames: int = 32   # both use 32 frames
    finegrained_image_size: int = 224

    # HuggingFace backbone
    hf_model_name: str = "google/vivit-b-16x2-kinetics400"

    # Pixel normalisation (ImageNet stats used by ViViT processor)
    mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    std:  List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])


# ─────────────────────────────────────────────
# Class Mappings
# ─────────────────────────────────────────────
BINARY_CLASSES: List[str] = ["Normal", "Abnormal"]   # idx 0 / 1

NORMAL_CLASSES: List[str] = [
    "Basketball",
    "Biking",
    "BoxingPunchingBag",
    "JumpingJack",
    "WalkingWithDog",
]

ABNORMAL_CLASSES: List[str] = [
    "Fighting",
    "Shooting",
    "Robbery",
    "Abuse",
    "Assault",
]


# ─────────────────────────────────────────────
# YOLOv8 Detection settings
# ─────────────────────────────────────────────
@dataclass
class DetectorConfig:
    model_name: str  = "yolov8n.pt"   # nano is fastest; swap to yolov8s/m for accuracy
    conf_threshold: float = 0.40
    iou_threshold:  float = 0.45
    target_class_id: int  = 0          # COCO class 0 = person
    input_size: int       = 640


# ─────────────────────────────────────────────
# ByteTrack / Tracking settings
# ─────────────────────────────────────────────
@dataclass
class TrackerConfig:
    track_thresh:   float = 0.50
    track_buffer:   int   = 30         # frames to keep a lost track alive
    match_thresh:   float = 0.80
    frame_rate:     int   = 30


# ─────────────────────────────────────────────
# Action-Tube / Buffer settings
# ─────────────────────────────────────────────
@dataclass
class TubeConfig:
    # Rolling frame buffer capacity per track (raw crops)
    buffer_size: int = 64

    # Minimum frames before first classification
    min_frames_for_inference: int = 32

    # Re-run classification every N new frames added to a track
    inference_stride: int = 8

    # ── Missing-detection tolerance ──────────
    # Max consecutive frames a track can be absent before its buffer is evicted
    missing_detection_tolerance: int = 15

    # When a detection is missing, repeat the last known crop this many times
    # so temporal continuity is preserved in the tube
    ghost_pad_enabled: bool = True

    # ── Temporal smoothing ───────────────────
    # Number of past ClassificationResults to average for label stability
    smoothing_window: int = 5

    # ── Safety caps ─────────────────────────
    # Maximum simultaneous tracked persons (ignore extras by descending conf)
    max_tracks: int = 20

    # Minimum pixel area for a crop to be accepted (filters tiny/noisy boxes)
    min_crop_px: int = 32             # both width AND height must exceed this


# ─────────────────────────────────────────────
# Inference / Classification thresholds
# ─────────────────────────────────────────────
@dataclass
class InferenceConfig:
    binary_confidence_threshold: float = 0.60   # below this -> uncertain, no routing
    device: str = "auto"                         # "auto" | "cuda" | "cpu"


# ─────────────────────────────────────────────
# Visualisation settings
# ─────────────────────────────────────────────
@dataclass
class VisConfig:
    font_scale: float = 0.55
    thickness:  int   = 2
    box_alpha:  float = 0.35
    # Colour palette per top-level category (BGR)
    normal_colour:   tuple = (0, 200, 80)     # green
    abnormal_colour: tuple = (0, 60, 220)     # red-ish (BGR → red)
    unknown_colour:  tuple = (150, 150, 150)  # grey


# ─────────────────────────────────────────────
# Output settings
# ─────────────────────────────────────────────
@dataclass
class OutputConfig:
    save_video:   bool = True
    output_dir:   str  = os.path.join(ROOT_DIR, "output")
    fourcc:       str  = "mp4v"


# ─────────────────────────────────────────────
# Master config (single import point)
# ─────────────────────────────────────────────
@dataclass
class PipelineConfig:
    paths:     ModelPaths    = field(default_factory=ModelPaths)
    vivit:     ViViTConfig   = field(default_factory=ViViTConfig)
    detector:  DetectorConfig = field(default_factory=DetectorConfig)
    tracker:   TrackerConfig  = field(default_factory=TrackerConfig)
    tube:      TubeConfig     = field(default_factory=TubeConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    vis:       VisConfig      = field(default_factory=VisConfig)
    output:    OutputConfig   = field(default_factory=OutputConfig)


# Singleton – import this everywhere
CFG = PipelineConfig()
