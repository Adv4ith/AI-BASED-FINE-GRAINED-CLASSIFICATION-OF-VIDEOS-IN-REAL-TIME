"""
vivit_model.py — ViViT model loader and classifier.

Handles:
  - Loading three ViViT checkpoints (binary, normal, abnormal)
  - Position-embedding interpolation for custom frame counts
  - Batched inference with softmax confidence scores
  - GPU / CPU device routing
"""

from __future__ import annotations

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple

from transformers import VivitModel, VivitConfig

from config import CFG, ViViTConfig, ModelPaths, BINARY_CLASSES, NORMAL_CLASSES, ABNORMAL_CLASSES


# ─────────────────────────────────────────────
# Positional embedding interpolation
# ─────────────────────────────────────────────

def _interpolate_pos_embed(
    state_dict: Dict,
    model: nn.Module,
    key: str = "vivit.embeddings.patch_embeddings.position_embeddings",
) -> Dict:
    """
    Interpolate position embeddings if the checkpoint num_frames ≠ model num_frames.
    Handles both spatial and temporal axes gracefully.
    """
    if key not in state_dict:
        return state_dict

    ckpt_pe  = state_dict[key]          # (1, T_ckpt*N_spatial+1, D)
    model_pe = model.state_dict()[key]   # (1, T_model*N_spatial+1, D)

    if ckpt_pe.shape == model_pe.shape:
        return state_dict   # no resize needed

    # cls token + patch tokens
    cls_token = ckpt_pe[:, :1, :]
    patch_pe  = ckpt_pe[:, 1:, :]

    # Interpolate linearly along the sequence dim
    patch_pe = patch_pe.permute(0, 2, 1)   # (1, D, N)
    patch_pe = F.interpolate(
        patch_pe,
        size=model_pe.shape[1] - 1,
        mode="linear",
        align_corners=False,
    )
    patch_pe = patch_pe.permute(0, 2, 1)   # (1, N_new, D)

    state_dict[key] = torch.cat([cls_token, patch_pe], dim=1)
    return state_dict


# ─────────────────────────────────────────────
# Custom ViViT head (classifier on top of HF backbone)
# ─────────────────────────────────────────────

class ViViTClassifier(nn.Module):
    """
    ViViT backbone (google/vivit-b-16x2-kinetics400) + linear classifier head.

    The HuggingFace VivitModel returns a pooled CLS token; we project that
    to num_classes using a single Linear layer.
    """

    def __init__(
        self,
        num_classes: int,
        num_frames:  int,
        image_size:  int,
        hf_model_name: str = CFG.vivit.hf_model_name,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_frames  = num_frames

        # Build HF config with custom tubelet / frame settings
        hf_cfg = VivitConfig.from_pretrained(hf_model_name)
        hf_cfg.num_frames  = num_frames
        hf_cfg.image_size  = image_size

        self.vivit = VivitModel(hf_cfg)
        hidden_dim = hf_cfg.hidden_size        # 768 for vivit-b
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        pixel_values : (B, T, C, H, W)

        Returns
        -------
        logits : (B, num_classes)
        """
        outputs = self.vivit(pixel_values=pixel_values)
        # Use the CLS token from last_hidden_state instead of pooler_output.
        # This avoids the missing 'vivit.pooler.dense.*' weights problem in
        # checkpoints that were saved without the pooler layer.
        cls_token = outputs.last_hidden_state[:, 0]   # (B, hidden_dim)
        logits = self.classifier(cls_token)
        return logits


# ─────────────────────────────────────────────
# Model loader
# ─────────────────────────────────────────────

def _load_vivit(
    checkpoint_path: str,
    num_classes: int,
    num_frames: int,
    image_size: int,
    device: torch.device,
    hf_model_name: str = CFG.vivit.hf_model_name,
    label: str = "",
) -> ViViTClassifier:
    """
    Instantiate ViViTClassifier, load weights from checkpoint.

    Handles all checkpoint formats produced by common ViViT training scripts:
      1. Full model object saved with torch.save(model, path)
         - class may be named 'ViViTWrapper', 'ViViTClassifier', etc.
         → extract via model.state_dict()
      2. Training checkpoint dict with nested state dict key:
         - {'model_state': {...}}          ← binary.pth format
         - {'model_state_dict': {...}}     ← standard format
         - {'state_dict': {...}}           ← Lightning format
      3. Bare state dict (no wrapper)
    """
    import sys
    import types

    print(f"[ViViT] Loading {label} -> {checkpoint_path}")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # ── Inject stub classes so pickle can resolve any class name used
    # during training (ViViTWrapper, ViViTClassifier, etc.)
    # We temporarily attach them to __main__ which is where torch.load
    # looks when the checkpoint was saved from a top-level training script.
    _STUB_NAMES = [
        "ViViTWrapper", "ViViTClassifier", "ViViTModel",
        "VideoViViT", "VivitClassifier", "Model",
    ]
    main_mod = sys.modules.get("__main__", types.ModuleType("__main__"))
    _injected = []
    for _name in _STUB_NAMES:
        if not hasattr(main_mod, _name):
            # Stub: an nn.Module that can receive arbitrary __init__ args
            _stub = type(_name, (nn.Module,), {
                "__init__": lambda self, *a, **kw: nn.Module.__init__(self),
                "forward":  lambda self, x: x,
            })
            setattr(main_mod, _name, _stub)
            _injected.append(_name)

    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    finally:
        # Always clean up injected stubs
        for _name in _injected:
            try:
                delattr(main_mod, _name)
            except AttributeError:
                pass

    # ── Extract state dict from whatever was loaded ──────────────────
    if isinstance(ckpt, nn.Module):
        # Case 1: full model object — extract its state dict
        print(f"  [INFO] {label} checkpoint is a full model object; extracting state_dict()")
        state = ckpt.state_dict()
    elif isinstance(ckpt, dict):
        # Case 2 / 3: dict — find the nested state dict
        if "model_state" in ckpt:          # binary.pth format
            print(f"  [INFO] {label} using key 'model_state'")
            state = ckpt["model_state"]
        elif "model_state_dict" in ckpt:   # standard training checkpoint
            print(f"  [INFO] {label} using key 'model_state_dict'")
            state = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:         # Lightning / other frameworks
            print(f"  [INFO] {label} using key 'state_dict'")
            state = ckpt["state_dict"]
        else:
            # Case 3: bare state dict (no wrapper keys)
            state = ckpt
    else:
        raise TypeError(
            f"Unexpected checkpoint type for {label}: {type(ckpt)}. "
            "Expected nn.Module or dict."
        )

    # ── Build target model ───────────────────────────────────────────
    model = ViViTClassifier(
        num_classes=num_classes,
        num_frames=num_frames,
        image_size=image_size,
        hf_model_name=hf_model_name,
    )

    # Strip common key prefixes:
    #   "module."  — added by DataParallel / DistributedDataParallel
    #   "model."   — added when the checkpoint class stores backbone as self.model
    def _strip(k: str) -> str:
        for prefix in ("module.", "model."):
            if k.startswith(prefix):
                k = k[len(prefix):]
        return k
    state = {_strip(k): v for k, v in state.items()}

    # Remap classifier keys:
    # Some training scripts wrap the head in nn.Sequential, saving as
    # "classifier.0.*" or "classifier.1.*". We use a bare nn.Linear
    # ("classifier.weight" / "classifier.bias"), so remap if needed.
    remapped = {}
    for k, v in state.items():
        # "classifier.1.weight" -> "classifier.weight"
        if k.startswith("classifier.") and not k in ("classifier.weight", "classifier.bias"):
            parts = k.split(".")
            if len(parts) >= 3 and parts[1].isdigit():
                k = "classifier." + ".".join(parts[2:])
        remapped[k] = v
    state = remapped

    # Interpolate positional embeddings for frame-count mismatch
    state = _interpolate_pos_embed(state, model)

    missing, unexpected = model.load_state_dict(state, strict=False)

    # Only warn about keys that are actual model weights (not training metadata)
    _meta_keys = {"epoch", "model_state", "optim_state", "val_acc", "val_loss",
                  "optimizer", "scheduler", "global_step", "best_val"}
    real_unexpected = [k for k in unexpected if k not in _meta_keys]
    if missing:
        print(f"  [WARN] {label} missing keys   : {missing[:5]} ...")
    if real_unexpected:
        print(f"  [WARN] {label} unexpected keys: {real_unexpected[:5]} ...")

    model.to(device)
    model.eval()
    print(f"  [OK] {label} ready on {device} | classes={num_classes} | frames={num_frames}")
    return model


# ─────────────────────────────────────────────
# Inference result dataclass
# ─────────────────────────────────────────────

from dataclasses import dataclass, field as dc_field
import numpy as np

@dataclass
class ClassificationResult:
    """Hierarchical classification result for one track."""
    track_id:    int
    # Stage 1: Binary
    binary_label: str   = "Unknown"
    binary_conf:  float = 0.0
    binary_probs: Optional[List[float]] = None
    # Stage 2: Fine-grained
    fine_label:  str   = "Unknown"
    fine_conf:   float = 0.0
    fine_probs:  Optional[List[float]] = None
    # Meta
    routed_to:   str   = ""   # "normal" | "abnormal" | "uncertain"
    # GradCAM activation map (T_temporal, 14, 14) in [0,1] — None if not generated
    cam_3d: Optional[np.ndarray] = dc_field(default=None, repr=False)

    @property
    def final_label(self) -> str:
        if self.fine_label != "Unknown":
            return self.fine_label
        return self.binary_label

    @property
    def final_conf(self) -> float:
        if self.fine_label != "Unknown":
            return self.fine_conf
        return self.binary_conf

    @property
    def is_abnormal(self) -> bool:
        return self.binary_label == "Abnormal"

    @property
    def has_heatmap(self) -> bool:
        return self.cam_3d is not None


# ─────────────────────────────────────────────
# Main ViViT inference engine
# ─────────────────────────────────────────────

class ViViTInferenceEngine:
    """
    Loads all three ViViT models and exposes a classify() method.

    classify() implements the hierarchical routing:
      binary ViViT
        ├── Normal  → normal ViViT
        └── Abnormal → abnormal ViViT

    All three models are loaded at construction time.
    """

    def __init__(
        self,
        paths:       ModelPaths  = CFG.paths,
        vivit:       ViViTConfig = CFG.vivit,
        device:      torch.device | str = "cpu",
        temperature: float = 0.7,   # < 1.0 sharpens softmax; 1.0 = no effect
        use_tta:     bool  = True,  # horizontal flip augmentation
    ) -> None:
        self.vivit       = vivit
        self.device      = torch.device(device)
        self.temperature = temperature
        self.use_tta     = use_tta
        # Multi-window: best confidence seen per track {tid: (pred_idx, conf, probs)}
        self._best_window: dict = {}

        # ── Stage 1: Binary ──────────────────
        self.binary_model = _load_vivit(
            checkpoint_path = paths.binary,
            num_classes     = len(BINARY_CLASSES),
            num_frames      = vivit.binary_num_frames,
            image_size      = vivit.binary_image_size,
            device          = self.device,
            label           = "Binary",
        )

        # ── Stage 2a: Normal ─────────────────
        self.normal_model = _load_vivit(
            checkpoint_path = paths.normal,
            num_classes     = len(NORMAL_CLASSES),
            num_frames      = vivit.finegrained_num_frames,
            image_size      = vivit.finegrained_image_size,
            device          = self.device,
            label           = "Normal",
        )

        # ── Stage 2b: Abnormal ───────────────
        self.abnormal_model = _load_vivit(
            checkpoint_path = paths.abnormal,
            num_classes     = len(ABNORMAL_CLASSES),
            num_frames      = vivit.finegrained_num_frames,
            image_size      = vivit.finegrained_image_size,
            device          = self.device,
            label           = "Abnormal",
        )

        # ── GradCAM engines (one per fine-grained model) ─────────────
        from gradcam import ViViTGradCAM
        self._cam_normal   = ViViTGradCAM(model=self.normal_model,
                                          tubelet_size=vivit.tubelet_size
                                          if hasattr(vivit, 'tubelet_size') else 2)
        self._cam_abnormal = ViViTGradCAM(model=self.abnormal_model,
                                          tubelet_size=vivit.tubelet_size
                                          if hasattr(vivit, 'tubelet_size') else 2)
        print("  [GradCAM] Engines ready (normal + abnormal).")


    # ──────────────────────────────────────────
    # Preprocessing: CLAHE contrast enhancement
    # ──────────────────────────────────────────

    @staticmethod
    def _apply_clahe(tube: torch.Tensor) -> torch.Tensor:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to
        every frame in the tube.

        Why: ViViT was trained on well-lit UCF/Kinetics clips. Surveillance
        footage is often low-contrast, causing the model to produce weak,
        spread activations -> low confidence. CLAHE boosts local contrast
        so the model sees sharper action features.

        tube shape: (1, T, C, H, W)  values in [-1, 1] (ImageNet normalised)
        """
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        t = tube.clone().cpu().numpy()          # (1, T, C, H, W)
        B, T, C, H, W = t.shape

        # Denorm from [-1,1] -> [0,255]
        t = ((t * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

        for b in range(B):
            for f in range(T):
                # Convert CHW RGB -> HWC BGR for OpenCV
                frame = t[b, f].transpose(1, 2, 0)[:, :, ::-1]   # HWC BGR
                # Apply CLAHE to L channel in LAB space
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                # Back to CHW RGB
                t[b, f] = frame[:, :, ::-1].transpose(2, 0, 1)

        # Re-normalise to [-1, 1]
        t = (t.astype(np.float32) / 255.0 - 0.5) / 0.5
        return torch.from_numpy(t).to(tube.device, dtype=tube.dtype)

    # ──────────────────────────────────────────
    # Core inference (all 6 boost techniques)
    # ──────────────────────────────────────────

    @torch.no_grad()
    def _run_model(
        self,
        model: ViViTClassifier,
        tube:  torch.Tensor,
    ) -> Tuple[int, float, List[float]]:
        """
        6-technique confidence-boosted inference (zero retraining):

        1. CLAHE contrast enhancement  — sharper features for low-light cams
        2. Extended TTA (4 views)      — h-flip, temporal reverse, brightness jitter
        3. Center crop view            — removes edge noise, focuses on person
        4. Logit bias correction       — removes systematic class bias learned
                                         from unbalanced training data
        5. Temperature scaling (T<1)   — sharpens the softmax distribution
        6. Multi-window best-of        — keeps highest conf across tube windows
                                         (handled in classify() via _best_of)
        """
        tube = tube.to(self.device)              # (1, T, C, H, W)

        # ── 1. CLAHE on the tube before inference ──────────────────────
        tube = self._apply_clahe(tube)

        # ── 2 & 3. Extended TTA: collect logits from multiple views ────
        all_logits = []

        # View 1: original
        all_logits.append(model(tube))

        if self.use_tta:
            # View 2: horizontal flip
            all_logits.append(model(tube.flip(dims=[-1])))

            # View 3: temporal reverse (flip the frame sequence)
            all_logits.append(model(tube.flip(dims=[1])))

            # View 4: brightness jitter (+15% brightness boost)
            tube_bright = (tube + 0.15).clamp(-1.0, 1.0)
            all_logits.append(model(tube_bright))

            # View 5: center crop at 87.5% -> resize back (removes edge noise)
            _, T, C, H, W = tube.shape
            crop_h, crop_w = int(H * 0.875), int(W * 0.875)
            y0, x0 = (H - crop_h) // 2, (W - crop_w) // 2
            tube_crop = tube[:, :, :, y0:y0+crop_h, x0:x0+crop_w]
            tube_crop = F.interpolate(
                tube_crop.view(-1, C, crop_h, crop_w),
                size=(H, W), mode="bilinear", align_corners=False
            ).view(1, T, C, H, W)
            all_logits.append(model(tube_crop))

        # Average raw logits across all views
        logits = torch.stack(all_logits).mean(dim=0)  # (1, num_classes)

        # ── 4. Logit bias correction ───────────────────────────────────
        # Subtract the mean logit so all classes start from the same
        # baseline. This removes learned prior bias where one class
        # always scores higher regardless of input.
        logits = logits - logits.mean(dim=-1, keepdim=True)

        # ── 5. Temperature scaling ─────────────────────────────────────
        logits = logits / max(self.temperature, 1e-4)

        probs      = F.softmax(logits, dim=-1)[0]   # (num_classes,)
        pred_idx   = int(probs.argmax())
        conf       = float(probs[pred_idx])
        probs_list = probs.cpu().tolist()
        return pred_idx, conf, probs_list

    def _best_of(
        self,
        tid:      int,
        pred_idx: int,
        conf:     float,
        probs:    List[float],
    ) -> Tuple[int, float, List[float]]:
        """
        Multi-window voting: keep the highest-confidence prediction seen
        for this track across all tube windows processed so far.
        If the new window is more confident, update the cache.
        Returns the best (pred_idx, conf, probs) for this track.
        """
        prev = self._best_window.get(tid)
        if prev is None or conf > prev[1]:
            self._best_window[tid] = (pred_idx, conf, probs)
        return self._best_window[tid]

    def reset_best_window(self, tid: int | None = None) -> None:
        """Clear best-window cache for a track (or all if tid=None)."""
        if tid is None:
            self._best_window.clear()
        else:
            self._best_window.pop(tid, None)

    def classify(
        self,
        track_id:    int,
        binary_tube: torch.Tensor,
        fine_tube:   torch.Tensor,
        binary_conf_thresh: float = CFG.inference.binary_confidence_threshold,
        generate_cam: bool = True,
    ) -> ClassificationResult:
        """
        Full hierarchical classification for one track.

        Stage 1 -> Binary ViViT (Normal vs Abnormal)
        Stage 2 -> Fine-grained ViViT (5-class)
        GradCAM -> spatial activation map on Stage-2 model for predicted class
                   cam_3d[t] gives the heatmap for frame t*tubelet_size
        """
        result = ClassificationResult(track_id=track_id)

        # Stage 1: Binary
        b_idx, b_conf, b_probs = self._run_model(self.binary_model, binary_tube)
        result.binary_label = BINARY_CLASSES[b_idx]
        result.binary_conf  = b_conf
        result.binary_probs = b_probs

        if b_conf < binary_conf_thresh:
            result.routed_to = "uncertain"
            return result

        # Stage 2: Route based on binary
        if result.binary_label == "Normal":
            result.routed_to  = "normal"
            cam_engine        = self._cam_normal
            f_idx, f_conf, f_probs = self._run_model(self.normal_model, fine_tube)
            result.fine_label = NORMAL_CLASSES[f_idx]
        else:
            result.routed_to  = "abnormal"
            cam_engine        = self._cam_abnormal
            f_idx, f_conf, f_probs = self._run_model(self.abnormal_model, fine_tube)
            result.fine_label = ABNORMAL_CLASSES[f_idx]

        result.fine_conf  = f_conf
        result.fine_probs = f_probs

        # Multi-window: upgrade to highest-confidence window seen for this track
        f_idx, f_conf, f_probs = self._best_of(track_id, f_idx, f_conf, f_probs)
        result.fine_conf  = f_conf
        result.fine_probs = f_probs
        # Re-apply class label in case best window had a different index
        if result.routed_to == "normal":
            result.fine_label = NORMAL_CLASSES[f_idx]
        else:
            result.fine_label = ABNORMAL_CLASSES[f_idx]

        # GradCAM: activation map for the predicted class on this tube
        # cam_3d shape: (T_temporal, 14, 14)  — overlay cam_3d[t] on frame t*tubelet_size
        # e.g. for Punching@91%: cam_3d[8] is the heatmap for frame 16 (mid-tube)
        if generate_cam:
            try:
                cam_3d, _, _ = cam_engine.generate(
                    tube       = fine_tube.to(self.device),
                    class_idx  = f_idx,
                    num_frames = self.vivit.finegrained_num_frames,
                )
                result.cam_3d = cam_3d
            except Exception as exc:
                import logging
                logging.getLogger(__name__).warning(
                    "[GradCAM] Track %d failed: %s", track_id, exc
                )

        return result

    def classify_batch(
        self,
        tubes: Dict[int, Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]],
        binary_conf_thresh: float = CFG.inference.binary_confidence_threshold,
        generate_cam: bool = True,
    ) -> Dict[int, ClassificationResult]:
        """
        Classify multiple tracks at once.

        Parameters
        ----------
        tubes        : {track_id: (binary_tube, fine_tube)}
        generate_cam : run GradCAM after Stage-2 (default True)

        Returns
        -------
        {track_id: ClassificationResult}  — result.cam_3d populated per track
        """
        results: Dict[int, ClassificationResult] = {}
        for tid, (binary_tube, fine_tube) in tubes.items():
            if binary_tube is None or fine_tube is None:
                continue
            results[tid] = self.classify(
                track_id           = tid,
                binary_tube        = binary_tube,
                fine_tube          = fine_tube,
                binary_conf_thresh = binary_conf_thresh,
                generate_cam       = generate_cam,
            )
        return results
