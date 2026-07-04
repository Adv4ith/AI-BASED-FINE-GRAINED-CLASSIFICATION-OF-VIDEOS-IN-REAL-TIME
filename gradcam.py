"""
gradcam.py — Standard GradCAM for ViViT.

Flow per track:
  1. ViViT predicts class (e.g. Punching, index=2, conf=91%)
  2. GradCAM.generate(model, tube, class_idx=2)
     a. Forward pass  → capture last-encoder-layer activations  A (B, N, D)
     b. Backward on class score → capture gradients              G (B, N, D)
     c. Channel weights:  w_k = mean_over_tokens(G[:, :, k])
     d. Weighted sum + ReLU:  CAM = ReLU(sum_k  w_k * A[:, :, k])
     e. Remove CLS token, reshape (N_patch,) → (T_temporal, 14, 14)
     f. Normalise → suppress background → sharpen peaks
  3. Returns cam_3d  (T_temporal, 14, 14)  in [0, 1]
  4. Visualiser upsamples each frame's slice → blends over person bbox
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_TUBELET_SIZE = 2
_PATCHES_SIDE = 14
_N_SPATIAL    = _PATCHES_SIDE * _PATCHES_SIDE   # 196
_BG_PERCENTILE = 75.0                            # suppress bottom 75 %


class ViViTGradCAM:
    """
    Standard Grad-CAM for ViViT classifiers.

    Parameters
    ----------
    model          : ViViTClassifier (has .vivit and .classifier attributes)
    target_layer   : nn.Module to hook (default: last encoder layer)
    tubelet_size   : temporal stride used when building tubes (default 2)
    patches_side   : spatial patches per side (14 for 224/16)
    bg_percentile  : percentile below which cam values are zeroed (noise removal)
    """

    def __init__(
        self,
        model:         nn.Module,
        target_layer:  Optional[nn.Module] = None,
        tubelet_size:  int   = _TUBELET_SIZE,
        patches_side:  int   = _PATCHES_SIDE,
        bg_percentile: float = _BG_PERCENTILE,
    ) -> None:
        self.model         = model
        self.tubelet_size  = tubelet_size
        self.patches_side  = patches_side
        self.n_spatial     = patches_side * patches_side
        self.bg_percentile = bg_percentile

        # Default: last ViViT encoder block
        self._target = target_layer or model.vivit.encoder.layer[-1]
        self._acts:  Optional[torch.Tensor] = None
        self._grads: Optional[torch.Tensor] = None
        self._hooks: list = []

    # ──────────────────────────────────────────
    # Hook management
    # ──────────────────────────────────────────

    def _register(self) -> None:
        def _fwd(mod, inp, out):
            self._acts = out[0] if isinstance(out, tuple) else out

        def _bwd(mod, gin, gout):
            # For Vision Transformers (ViViT), the output gradient (gout[0]) has zero gradients
            # for all spatial patch tokens because they are sliced out/discarded when extracting
            # the CLS token [:, 0, :] for the final classification head.
            # We capture the layer's input gradient (gin[0]) instead, which contains the rich,
            # attention-weighted backpropagated gradients for all patches.
            g = gin[0] if isinstance(gin, tuple) else gin
            self._grads = g.detach()

        self._hooks = [
            self._target.register_forward_hook(_fwd),
            self._target.register_full_backward_hook(_bwd),
        ]

    def _remove(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ──────────────────────────────────────────
    # Core
    # ──────────────────────────────────────────

    def generate(
        self,
        tube:      torch.Tensor,        # (1, T, C, H, W) — the 32-frame action tube
        class_idx: Optional[int] = None, # predicted class index from ViViT
        num_frames: int = 32,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Compute GradCAM activation map for one action tube.

        Parameters
        ----------
        tube       : (1, T, C, H, W) tensor — same tube used for classification
        class_idx  : class to explain (None = take argmax from forward pass)
        num_frames : number of frames in tube (T dimension)

        Returns
        -------
        cam_3d     : (T_temporal, 14, 14)  float32 in [0, 1]
        class_idx  : int   — class that was explained
        confidence : float — softmax probability for that class
        """
        T_temporal = num_frames // self.tubelet_size

        was_training = self.model.training
        self.model.train()          # required for gradient flow
        self._register()

        try:
            x = tube.clone().detach().float()
            x.requires_grad_(True)

            # ── Forward ───────────────────────────────────────────────
            out      = self.model.vivit(pixel_values=x)
            cls_feat = out.last_hidden_state[:, 0]     # CLS token (1, D)
            logits   = self.model.classifier(cls_feat) # (1, C)

            probs = torch.softmax(logits, dim=-1)
            if class_idx is None:
                class_idx = int(logits.argmax(-1).item())
            confidence = float(probs[0, class_idx].item())

            # ── Backward on target class score ────────────────────────
            self.model.zero_grad()
            logits[0, class_idx].backward()

            # ── Grad-CAM weights ──────────────────────────────────────
            acts  = self._acts.detach()[:, 1:, :]   # drop CLS  (1, N_patch, D)
            grads = self._grads[:, 1:, :]            # drop CLS  (1, N_patch, D)

            # w_k = global-average-pool gradients over all patch tokens
            weights = grads.mean(dim=1)              # (1, D)

            # Weighted sum + ReLU  →  CAM per token
            cam = F.relu(
                (acts * weights.unsqueeze(1)).sum(dim=-1)   # (1, N_patch)
            )
            cam_np = cam[0].cpu().numpy()            # (N_patch,)

        finally:
            self._remove()
            self.model.train(was_training)
            self.model.zero_grad()

        # ── Reshape (N_patch,) → (T_temporal, 14, 14) ─────────────────
        N_expected = T_temporal * self.n_spatial
        if cam_np.shape[0] > N_expected:
            cam_np = cam_np[:N_expected]
        elif cam_np.shape[0] < N_expected:
            cam_np = np.pad(cam_np, (0, N_expected - cam_np.shape[0]))
        cam_3d = cam_np.reshape(T_temporal, self.patches_side, self.patches_side)

        # ── Normalise ──────────────────────────────────────────────────
        lo, hi = cam_3d.min(), cam_3d.max()
        if hi <= lo:
            return np.zeros_like(cam_3d, dtype=np.float32), class_idx, confidence
        cam_3d = (cam_3d - lo) / (hi - lo + 1e-8)

        # ── Background suppression + peak sharpening ───────────────────
        thresh = np.percentile(cam_3d, self.bg_percentile)
        cam_3d = np.where(cam_3d >= thresh, cam_3d, 0.0)
        # Apply a square-root curve to strongly amplify the intensity of active regions
        cam_3d = np.sqrt(cam_3d)

        lo, hi = cam_3d.min(), cam_3d.max()
        if hi > lo:
            cam_3d = (cam_3d - lo) / (hi - lo + 1e-8)

        return cam_3d.astype(np.float32), class_idx, confidence
