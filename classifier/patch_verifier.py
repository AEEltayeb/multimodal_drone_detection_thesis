"""
patch_verifier.py — Inference wrapper for per-modality patch verifiers.

Loads a model checkpoint saved by train_patch_verifier.py and exposes a
simple interface:

    vf = PatchVerifier("classifier/runs/patches/patch_verifier_rgb.pt")
    p_drone = vf.predict_crop(bgr_image, xyxy_box)

For a whole-frame + list of boxes, use predict_boxes() which returns one
probability per box. Batches internally so many boxes per frame cost
roughly one forward pass.

The IR verifier was trained on real thermal crops. In the GUI's
grayscale-replicate mode, the "IR" frame is actually a desaturated RGB;
the caller is responsible for deciding when to trust the IR verifier
(see `is_real_thermal` heuristic in ir_gui/fusion/engine.py).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small


class PatchVerifier:
    def __init__(self, weights_path: str | Path, device: str | torch.device = "cuda"):
        self.device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        weights_path = Path(weights_path)
        ckpt = torch.load(str(weights_path), map_location=self.device,
                          weights_only=True)
        self.num_classes = int(ckpt.get("num_classes", 1))
        self.class_names = list(ckpt.get("class_names", []))
        confuser_defaults = ["airplane", "helicopter", "bird"]
        conf_names = list(ckpt.get("confuser_classes", confuser_defaults))
        self.confuser_indices = [i for i, n in enumerate(self.class_names)
                                  if n in conf_names]

        net = mobilenet_v3_small(weights=None)
        in_features = net.classifier[-1].in_features
        net.classifier[-1] = nn.Linear(in_features, self.num_classes)
        net.load_state_dict(ckpt["state_dict"])
        net.eval()
        net.to(self.device)
        self.net = net
        self.input_size = int(ckpt.get("input_size", 224))
        self.mean = np.array(ckpt.get("mean", [0.485, 0.456, 0.406]),
                             dtype=np.float32)
        self.std = np.array(ckpt.get("std", [0.229, 0.224, 0.225]),
                            dtype=np.float32)
        self.modality = ckpt.get("modality", "unknown")
        self.last_labels: list[str] = []  # populated on each predict call

        # Optional Mahalanobis OOD stats (produced by calibrate_confuser_ood.py)
        ood_path = weights_path.with_name(weights_path.stem + "_ood.npz")
        self.ood_mean = None
        self.ood_inv_cov = None
        self.ood_threshold = None
        if ood_path.exists():
            z = np.load(str(ood_path))
            self.ood_mean = torch.from_numpy(z["mean"]).to(self.device)
            self.ood_inv_cov = torch.from_numpy(z["inv_cov"]).to(self.device)
            self.ood_threshold = float(z["threshold"])

    @staticmethod
    def _crop_with_context(img_bgr: np.ndarray, xyxy, pad_frac: float = 0.5,
                           min_side: int = 24) -> np.ndarray | None:
        x1, y1, x2, y2 = xyxy
        ih, iw = img_bgr.shape[:2]
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        side = max(bw, bh) * (1.0 + 2.0 * pad_frac)
        side = max(side, float(min_side))
        ax1 = int(round(cx - side / 2))
        ay1 = int(round(cy - side / 2))
        ax2 = int(round(cx + side / 2))
        ay2 = int(round(cy + side / 2))
        ax1 = max(0, ax1); ay1 = max(0, ay1)
        ax2 = min(iw, ax2); ay2 = min(ih, ay2)
        if ax2 - ax1 < min_side or ay2 - ay1 < min_side:
            return None
        return img_bgr[ay1:ay2, ax1:ax2]

    def _preprocess(self, crop_bgr: np.ndarray) -> torch.Tensor:
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.input_size, self.input_size),
                         interpolation=cv2.INTER_AREA)
        arr = rgb.astype(np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return tensor

    def _forward(self, batch: torch.Tensor):
        """Run backbone + classifier, returning (penultimate_features, logits).

        Logits shape: (N,) for binary, (N, num_classes) for multi-class.
        """
        x = self.net.features(batch)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        for layer in self.net.classifier[:-1]:
            x = layer(x)
        penult = x
        logits = self.net.classifier[-1](x)
        if self.num_classes == 1:
            logits = logits.squeeze(1)
        return penult, logits

    def _confuser_probs(self, logits: torch.Tensor) -> np.ndarray:
        """Return effective P(confuser) per crop:
          - binary: sigmoid(logit)
          - 4-class: argmax_prob if argmax ∈ confusers else 0
        so the existing veto rule (p >= threshold) keeps the semantics
        from the training-time reject sweep.
        """
        if self.num_classes == 1:
            return torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        argmax = probs.argmax(axis=1)
        argmax_prob = probs[np.arange(len(probs)), argmax]
        is_conf = np.isin(argmax, self.confuser_indices)
        return np.where(is_conf, argmax_prob, 0.0).astype(np.float32)

    def _confuser_labels(self, logits: torch.Tensor) -> list[str]:
        """Return per-crop human-readable label, e.g. 'airplane:0.85' or 'pass'.
        For binary models returns 'confuser:0.XX' or 'pass'."""
        if self.num_classes == 1:
            p = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
            return [f"confuser:{v:.2f}" if v >= 0.3 else "pass" for v in p]
        probs = torch.softmax(logits, dim=1).cpu().numpy().astype(np.float32)
        argmax = probs.argmax(axis=1)
        labels = []
        for i in range(len(probs)):
            cls_idx = argmax[i]
            cls_prob = probs[i, cls_idx]
            if cls_idx < len(self.class_names):
                name = self.class_names[cls_idx]
            else:
                name = f"cls{cls_idx}"
            if cls_idx in self.confuser_indices:
                labels.append(f"{name}:{cls_prob:.2f}")
            else:
                # Show top confuser prob too
                conf_probs = [(self.class_names[j], probs[i, j])
                              for j in self.confuser_indices
                              if j < len(self.class_names)]
                if conf_probs:
                    top_c, top_p = max(conf_probs, key=lambda x: x[1])
                    labels.append(f"pass({top_c}:{top_p:.2f})")
                else:
                    labels.append("pass")
        return labels

    def _mahalanobis(self, feats: torch.Tensor) -> torch.Tensor:
        d = feats - self.ood_mean
        return torch.sqrt(torch.einsum("ni,ij,nj->n", d, self.ood_inv_cov, d))

    @torch.no_grad()
    def predict_crops(self, crops_bgr: Sequence[np.ndarray]) -> np.ndarray:
        if len(crops_bgr) == 0:
            return np.zeros(0, dtype=np.float32)
        batch = torch.stack([self._preprocess(c) for c in crops_bgr]).to(
            self.device, non_blocking=True)
        _, logits = self._forward(batch)
        self.last_labels = self._confuser_labels(logits)
        return self._confuser_probs(logits)

    @torch.no_grad()
    def predict_crops_with_ood(self, crops_bgr: Sequence[np.ndarray]):
        """Return (probs, ood_flags, distances). For 4-class models the
        built-in "other" class already covers OOD, so flags are always False
        unless Mahalanobis stats are explicitly loaded."""
        if len(crops_bgr) == 0:
            return (np.zeros(0, dtype=np.float32),
                    np.zeros(0, dtype=bool),
                    np.zeros(0, dtype=np.float32))
        batch = torch.stack([self._preprocess(c) for c in crops_bgr]).to(
            self.device, non_blocking=True)
        penult, logits = self._forward(batch)
        probs = self._confuser_probs(logits)
        self.last_labels = self._confuser_labels(logits)
        if self.ood_threshold is None:
            return probs, np.zeros(len(probs), dtype=bool), np.zeros(len(probs), np.float32)
        dists = self._mahalanobis(penult).cpu().numpy().astype(np.float32)
        flags = dists > self.ood_threshold
        return probs, flags, dists

    @torch.no_grad()
    def predict_boxes(self, img_bgr: np.ndarray,
                      boxes_xyxy: Iterable) -> np.ndarray:
        crops = []
        keep_idx = []
        for i, b in enumerate(boxes_xyxy):
            c = self._crop_with_context(img_bgr, b)
            if c is not None:
                crops.append(c)
                keep_idx.append(i)
        n = sum(1 for _ in boxes_xyxy) if hasattr(boxes_xyxy, "__len__") else len(crops)
        if isinstance(boxes_xyxy, (list, tuple)):
            n = len(boxes_xyxy)
        probs_out = np.zeros(n, dtype=np.float32)
        if not crops:
            return probs_out
        probs = self.predict_crops(crops)
        for i, p in zip(keep_idx, probs):
            probs_out[i] = p
        return probs_out

    @torch.no_grad()
    def predict_boxes_with_ood(self, img_bgr: np.ndarray, boxes_xyxy: Iterable):
        """Return (probs, ood_flags, distances) per box. Cropping failures
        are reported as OOD (no prediction)."""
        boxes_list = list(boxes_xyxy)
        n = len(boxes_list)
        probs_out = np.zeros(n, dtype=np.float32)
        ood_out = np.ones(n, dtype=bool)          # default OOD for dropped boxes
        dist_out = np.zeros(n, dtype=np.float32)
        crops = []
        keep_idx = []
        for i, b in enumerate(boxes_list):
            c = self._crop_with_context(img_bgr, b)
            if c is not None:
                crops.append(c)
                keep_idx.append(i)
        if not crops:
            return probs_out, ood_out, dist_out
        probs, flags, dists = self.predict_crops_with_ood(crops)
        for i, p, f, d in zip(keep_idx, probs, flags, dists):
            probs_out[i] = p
            ood_out[i] = bool(f)
            dist_out[i] = d
        # If no calibration is loaded, nothing is OOD.
        if self.ood_threshold is None:
            ood_out[:] = False
        return probs_out, ood_out, dist_out
