"""MLP V5 feature-distillation verifier — production-importable copy.

The V5 verifier reads YOLO's internal FPN features (P3 + P5) for each detection
and classifies drone-vs-confuser, replacing the heavy patch-CNN verifier at
~50x lower per-detection cost.

This module is a self-contained lift of the hook + ROI-pool + feature extractor
from `eval/distill_v5_p3p5_ft4.py` plus the inference wrapper from
`eval/eval_v4_vs_patch.py`, so the GUI/production stack does not import from
`eval/`. The feature schema is frozen by the checkpoint (input_dim is validated
at load); if you retrain V5 with a different layout, regenerate the checkpoint.

Source of truth for the schema: docs/analysis/mlp_v5_report.md §3.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn

# ── Feature schema (must match eval/distill_v5_p3p5_ft4.py) ──────────────────
# p3 (stride 8, 64ch) pooled to a 2x2 grid -> 256-D; p5 (stride 32, 256ch)
# pooled to 1x1 -> 256-D; plus 5 metadata features. Total 517-D.
P3_GRID = (2, 2)
P5_GRID = (1, 1)
_P3_CH = 64
_P5_CH = 256
_P3_DIM = _P3_CH * P3_GRID[0] * P3_GRID[1]   # 256
_P5_DIM = _P5_CH * P5_GRID[0] * P5_GRID[1]   # 256
YOLO_FEAT_DIM = _P3_DIM + _P5_DIM            # 512
META_DIM = 5
INPUT_DIM = META_DIM + YOLO_FEAT_DIM         # 517


class DetectInputHook:
    """Captures the 3 FPN feature maps fed into YOLO's Detect head.

    Register once on a YOLO model; `p3/p5` then hold the most recent forward's
    feature maps. Read them synchronously right after `model.predict(...)`,
    before the next forward overwrites them.
    """

    def __init__(self):
        self.p3: torch.Tensor | None = None   # stride ~8  (high res)
        self.p4: torch.Tensor | None = None   # stride ~16
        self.p5: torch.Tensor | None = None   # stride ~32 (most semantic)

    def clear(self):
        self.p3 = self.p4 = self.p5 = None

    def _hook(self, module, args):
        x = args[0]  # list of 3 feature maps from the neck
        self.p3 = x[0].detach()
        self.p4 = x[1].detach()
        self.p5 = x[2].detach()

    def register(self, model):
        """Attach to the Detect head's forward-pre-hook. Returns the handle."""
        detect_mod = model.model.model[-1]
        return detect_mod.register_forward_pre_hook(self._hook)


def roi_pool(feature_map: torch.Tensor, box_xyxy, img_shape, out_h=1, out_w=1):
    """Adaptive-average-pool a box region from a feature map -> flat numpy array
    of length C * out_h * out_w. box_xyxy is in *image* coords; img_shape is
    (H_img, W_img)."""
    _, C, H, W = feature_map.shape
    ih, iw = img_shape
    x1, y1, x2, y2 = box_xyxy
    fx1 = max(0, int(x1 / iw * W))
    fy1 = max(0, int(y1 / ih * H))
    fx2 = min(W, max(fx1 + 1, int(np.ceil(x2 / iw * W))))
    fy2 = min(H, max(fy1 + 1, int(np.ceil(y2 / ih * H))))
    crop = feature_map[0, :, fy1:fy2, fx1:fx2]
    pooled = nn.functional.adaptive_avg_pool2d(crop.unsqueeze(0), (out_h, out_w))
    return pooled.squeeze(0).flatten().cpu().numpy()


def extract_box_metadata(box_xyxy, conf, img_shape):
    """5 metadata features: conf, log_area, aspect, rel_cx, rel_cy."""
    x1, y1, x2, y2 = box_xyxy
    ih, iw = img_shape
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    area = bw * bh
    cx = (x1 + x2) / 2.0 / max(iw, 1)
    cy = (y1 + y2) / 2.0 / max(ih, 1)
    return np.array([
        float(conf),
        float(np.log(max(area, 1.0))),
        float(bw / max(bh, 1)),
        float(cx),
        float(cy),
    ], dtype=np.float32)


def extract_detection_features(hook: DetectInputHook, box_xyxy, img_shape, conf):
    """metadata + multi-scale p3+p5 features for one detection -> 517-D."""
    meta = extract_box_metadata(box_xyxy, conf, img_shape)
    if hook.p3 is not None:
        p3_feat = roi_pool(hook.p3, box_xyxy, img_shape, P3_GRID[0], P3_GRID[1])
    else:
        p3_feat = np.zeros(_P3_DIM, dtype=np.float32)
    if hook.p5 is not None:
        p5_feat = roi_pool(hook.p5, box_xyxy, img_shape, P5_GRID[0], P5_GRID[1])
    else:
        p5_feat = np.zeros(_P5_DIM, dtype=np.float32)
    return np.concatenate([meta, p3_feat, p5_feat]).astype(np.float32)


class MLPVerifier:
    """Loads an mlp_v5.pt checkpoint and scores per-detection P(drone).

    Architecture is reproduced from the checkpoint's saved hyperparameters:
    a stack of (Linear -> [BatchNorm1d] -> ReLU -> Dropout) ending in
    Linear(., 1). `predict_drone_probs` returns sigmoid(logits); keep a
    detection when its drone-prob >= threshold (default from the checkpoint,
    overridable by the caller).
    """

    def __init__(self, weights_path, device: str = "cuda"):
        self.device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        ckpt = torch.load(str(weights_path), map_location=self.device,
                          weights_only=True)
        self.input_dim = int(ckpt["input_dim"])
        self.hidden_dims = list(ckpt["hidden_dims"])
        self.threshold = float(ckpt.get("threshold", 0.5))
        self.cv_f1 = float(ckpt.get("cv_f1", -1.0))
        self.feature_schema = ckpt.get("feature_schema", "unknown")
        use_bn = bool(ckpt.get("use_batchnorm", False))
        dropout = float(ckpt.get("dropout", 0.2))

        dims = [self.input_dim, *self.hidden_dims, 1]
        layers: list[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if use_bn:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers).to(self.device).eval()
        self.net.load_state_dict(ckpt["state_dict"])

        sm, ss = ckpt["scaler_mean"], ckpt["scaler_scale"]
        if isinstance(sm, torch.Tensor):
            self.scaler_mean = sm.to(self.device).float()
            self.scaler_scale = ss.to(self.device).float()
        else:
            self.scaler_mean = torch.from_numpy(
                np.asarray(sm, dtype=np.float32)).to(self.device)
            self.scaler_scale = torch.from_numpy(
                np.asarray(ss, dtype=np.float32)).to(self.device)

        if self.input_dim != INPUT_DIM:
            raise ValueError(
                f"Checkpoint input_dim={self.input_dim} but this module's "
                f"feature extractor produces {INPUT_DIM}-D vectors. The "
                f"checkpoint schema does not match classifier/mlp_verifier.py.")

    @torch.no_grad()
    def predict_drone_probs(self, feats_np: np.ndarray) -> np.ndarray:
        """feats_np: (n, input_dim) raw (un-scaled) features. Returns (n,)
        P(drone) in [0, 1]."""
        if feats_np is None or len(feats_np) == 0:
            return np.zeros(0, dtype=np.float32)
        x = torch.from_numpy(feats_np.astype(np.float32)).to(self.device)
        x = (x - self.scaler_mean) / self.scaler_scale
        logits = self.net(x).squeeze(-1)
        return torch.sigmoid(logits).cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def score_dets(self, hook: DetectInputHook, dets: Sequence, img_shape) -> np.ndarray:
        """Score a list of detections using the hook's current feature maps.

        dets: iterable of [x1, y1, x2, y2, conf] (extra fields ignored).
        img_shape: (H, W) of the frame that produced `hook`'s features.
        Returns (n,) P(drone). Call immediately after the forward that
        produced these detections.
        """
        if not len(dets):
            return np.zeros(0, dtype=np.float32)
        feats = np.stack([
            extract_detection_features(hook, (d[0], d[1], d[2], d[3]), img_shape, d[4])
            for d in dets
        ])
        return self.predict_drone_probs(feats)
