#!/usr/bin/env python3
"""
V5 distillation — multi-scale p3+p5 on FT4 R3, ~56k-sample balanced corpus.

Three levers added on top of V4 to break the recall-collapse pathology:

  Lever 1 — Per-source quotas. Drone pool ~26k across 7 sources (Anti-UAV val,
            Svanstrom, Selcom mixed ft2 val, RGB dataset train+val, and
            RGB_video_rgb_dataset DRONE frames). Confuser pool ~30k across 6
            sources (rgb_confusers_merged train+val, Svanstrom drone-empty,
            RGB_video AIRPLANE/BIRD/HELI, RGB dataset drone-empty, Anti-UAV
            hard negatives). Selcom target capped at 150 since FT4 R3 R~=0.5
            on Selcom (no useful feature signal from frames the detector
            misses).

  Lever 2 — Focal loss with label smoothing + per-source sample weights.
            FocalLoss(alpha=0.75, gamma=2.0, label_smoothing=0.1). Svanstrom
            samples carry weight 2.5x, real-video 2.0x, Selcom 1.8x, others
            1.0x. Sample weights persisted to training_data.npz.

  Lever 3 — Multi-scale ROI pool. p3 -> 2x2 grid (4 * 64 = 256-D). p5 -> 1x1
            (256-D). YOLO features 512-D total (was 320-D). With 5 metadata
            features the MLP input is 517-D. Hidden dims bumped to
            (512, 256, 128, 64) with BatchNorm1d and dropout 0.3.

Note: RGB_video_rgb_dataset is the converted form of the drone-detection-
video-tests mp4s. Same source as pipeline_video_tests eval, so V5 head-to-
head MUST stay on svanstrom + confuser_test + antiuav only. Adding pipeline
video eval to V5 creates a train-test leak.

Saves mlp_v5.pt with the same checkpoint schema as mlp_v4.pt so the harness
at eval/eval_v4_vs_patch.py works unchanged via --mlp-weights.

Usage:
    python eval/distill_v5_p3p5_ft4.py              # full run, ~2-3h
    python eval/distill_v5_p3p5_ft4.py --quick      # smoke (~10 min)
    python eval/distill_v5_p3p5_ft4.py --phase 3    # resume at eval (cached)
    python eval/distill_v5_p3p5_ft4.py --phase 2    # rerun training only
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from ultralytics import YOLO

from metrics import compute_prf, score_detections

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = REPO / "eval"
OUT_DIR = EVAL_DIR / "results" / "_v5_p3p5_ft4_distill"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "classifiers").mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    # ES_Drone_Thesis layout: weights live under models/rgb/ (the pre-reorg
    # "RGB model/" dir no longer exists). Repointed 2026-06-17.
    "selcom_ft3_1280": str(REPO / "models" / "rgb" / "Yolo26n_selcom_mixed_ft3_1280" / "weights" / "best.pt"),
    "ft4_r3":          str(REPO / "models" / "rgb" / "Yolo26n_selcom_confuser_ft4_1280" / "weights" / "best.pt"),
}

ANTIUAV_DIR  = Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images")
ANTIUAV_VAL  = Path("G:/drone/Anti-UAV-RGBT_yolo_converted/val/RGB/images")
ANTIUAV_TEST = Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images")
SVANSTROM_DIR = Path("G:/drone/svanstrom_paired/RGB/images")
SELCOM_VAL   = Path("G:/drone/_finetune_selcom_mixed_ft2/images/val")
SELCOM_TRAIN = Path("G:/drone/_finetune_selcom_mixed_ft2/images/train")
CONFUSER_TRAIN = Path("G:/drone/rgb_confusers_merged/images/train")
CONFUSER_VAL   = Path("G:/drone/rgb_confusers_merged/images/val")
CONFUSER_TEST  = Path("G:/drone/rgb_confusers_merged/images/test")
# V5 additions
RGB_DATASET_TRAIN = Path("G:/drone/dataset/dataset/images/train")
RGB_DATASET_VAL   = Path("G:/drone/dataset/dataset/images/val")
RGB_VIDEO_TRAIN   = Path("G:/drone/RGB_video_rgb_dataset/train/images")
RGB_VIDEO_VAL     = Path("G:/drone/RGB_video_rgb_dataset/val/images")

# ── Parameters ──────────────────────────────────────────────────────────────
IMGSZ = 1280
CONF_THR = 0.25
IOU_THR = 0.5
IOP_THR = 0.5
SEED = 42

# V5 per-source quota registry.
#
# Each entry mines BOTH drone TPs and confuser FPs from the same scan when
# possible (e.g. Anti-UAV val supplies real drones AND hard-neg backgrounds).
# Targets sum to ~26k drones + ~30k confusers = ~56k total (vs patch-v2's
# ~23k RGB patches — 2.4x more).
#
# kind values:
#   "image_with_gt"   -- read YOLO .txt labels; det matches GT -> drone TP,
#                        det does not match -> confuser FP (hard negative).
#   "image_no_gt"     -- no labels read; every detection is a confuser FP
#                        (used for rgb_confusers_merged train/val).
#
# filter_prefixes restricts the per-image scan to files starting with one of
# the listed prefixes (used to split RGB_video_rgb_dataset by V_DRONE_ vs
# V_BIRD_/V_AIRPLANE_/V_HELICOPTER_).

@dataclass(frozen=True)
class SourceConfig:
    name: str
    path: Path
    stride: int
    kind: str                            # "image_with_gt" or "image_no_gt"
    target_drones: int = 0               # 0 -> skip drone collection
    target_confusers: int = 0            # 0 -> skip confuser collection
    weight_drone: float = 1.0            # sample weight for drone samples
    weight_confuser: float = 1.0         # sample weight for confuser samples
    filter_prefixes: tuple = ()          # if set, only files starting with one
    match_rule: str = "iou"              # "iou" or "iop"; Svan needs "iop"
                                          # because GT boxes are bigger than
                                          # the drone -> IoU under-counts
    imgsz: int = 640                     # YOLO input size. Production default
                                          # is 640 globally; Svanstrom needs
                                          # 1280 because native res is 640x480
                                          # and small drones are unresolvable
                                          # at 640. Selcom stays at 640.
    drone_class: int = 0                 # YOLO label class id for "drone".
                                          # 0 for all standard datasets; 1 for
                                          # CBAM (names=['B','D','P']).

SOURCES = [
    # --- Drone-rich datasets: mine both TPs and hard-neg FPs ---
    SourceConfig("antiuav_val",       ANTIUAV_VAL,       stride=3,  kind="image_with_gt",
                 target_drones=4000, target_confusers=2000,
                 weight_drone=1.0,   weight_confuser=1.0),
    SourceConfig("svanstrom",         SVANSTROM_DIR,     stride=1,  kind="image_with_gt",
                 target_drones=5000, target_confusers=6000,
                 weight_drone=2.5,   weight_confuser=2.5,
                 match_rule="iop",   # Svan GT boxes are larger than drones
                 imgsz=1280),        # 640x480 native -> 1280 needed for drones
    SourceConfig("selcom_train",      SELCOM_TRAIN,      stride=1,  kind="image_with_gt",
                 target_drones=3000, target_confusers=500,
                 weight_drone=1.8,   weight_confuser=1.5,
                 match_rule="iop",   # CCTV: IoP per EVIDENCE_LEDGER. NOTE: V5
                                      # uses selcom_TRAIN (not val) so val stays
                                      # clean as the head-to-head eval surface.
                 imgsz=1280),        # Selcom small CCTV drones need 1280 to
                                      # stay resolvable — at 640 FT4 R=0.10.
    SourceConfig("rgb_dataset_train", RGB_DATASET_TRAIN, stride=8,  kind="image_with_gt",
                 target_drones=8000, target_confusers=3000,
                 weight_drone=1.0,   weight_confuser=1.0),
    SourceConfig("rgb_dataset_val",   RGB_DATASET_VAL,   stride=3,  kind="image_with_gt",
                 target_drones=1500, target_confusers=0,
                 weight_drone=1.0,   weight_confuser=1.0),
    # --- Drone-detection-video-tests: split by filename prefix ---
    SourceConfig("rgb_video_train_drone", RGB_VIDEO_TRAIN, stride=2, kind="image_with_gt",
                 target_drones=4500, target_confusers=0,
                 weight_drone=2.0,   weight_confuser=2.0,
                 filter_prefixes=("V_DRONE_",)),
    SourceConfig("rgb_video_val_drone",   RGB_VIDEO_VAL,   stride=1, kind="image_with_gt",
                 target_drones=800,  target_confusers=0,
                 weight_drone=2.0,   weight_confuser=2.0,
                 filter_prefixes=("V_DRONE_",)),
    SourceConfig("rgb_video_train_conf",  RGB_VIDEO_TRAIN, stride=1, kind="image_with_gt",
                 target_drones=0,    target_confusers=3500,
                 weight_drone=2.0,   weight_confuser=2.0,
                 filter_prefixes=("V_AIRPLANE_", "V_BIRD_", "V_HELICOPTER_")),
    SourceConfig("rgb_video_val_conf",    RGB_VIDEO_VAL,   stride=1, kind="image_with_gt",
                 target_drones=0,    target_confusers=500,
                 weight_drone=2.0,   weight_confuser=2.0,
                 filter_prefixes=("V_AIRPLANE_", "V_BIRD_", "V_HELICOPTER_")),
    # --- Pure confuser datasets: no labels, all dets are FPs ---
    SourceConfig("confuser_train",    CONFUSER_TRAIN,    stride=2,  kind="image_no_gt",
                 target_drones=0,    target_confusers=12000,
                 weight_drone=1.0,   weight_confuser=1.0),
    SourceConfig("confuser_val",      CONFUSER_VAL,      stride=1,  kind="image_no_gt",
                 target_drones=0,    target_confusers=2500,
                 weight_drone=1.0,   weight_confuser=1.0),
]

# Evaluation strides (Phase 3 sanity check; real eval is in eval_v4_vs_patch.py)
EVAL_STRIDE = {
    "antiuav":  85,
    "svanstrom": 29,
    "selcom":    1,          # only 311 images
    "confuser":  3,
}

# ── Feature extraction hook ─────────────────────────────────────────────────

class DetectInputHook:
    """Captures the 3 FPN feature maps fed into YOLO's Detect head."""
    def __init__(self):
        self.p3: torch.Tensor | None = None   # stride ~8  (high res)
        self.p4: torch.Tensor | None = None   # stride ~16
        self.p5: torch.Tensor | None = None   # stride ~32 (most semantic)

    def clear(self):
        self.p3 = self.p4 = self.p5 = None

    def _hook(self, module, args):
        x = args[0]  # list of 3 feature maps from the neck
        self.p3 = x[0].detach()  # (1, 64, H3, W3)
        self.p4 = x[1].detach()  # (1, 128, H4, W4)
        self.p5 = x[2].detach()  # (1, 256, H5, W5)

    def register(self, model: YOLO):
        detect_mod = model.model.model[-1]
        return detect_mod.register_forward_pre_hook(self._hook)


def roi_pool(feature_map: torch.Tensor, box_xyxy, img_shape, out_h=1, out_w=1):
    """Adaptive average pool a box region from a feature map, returning a flat
    numpy array of length C * out_h * out_w.

    Args:
        feature_map: (1, C, H, W) tensor from the YOLO neck.
        box_xyxy: (x1, y1, x2, y2) in *image* coordinates.
        img_shape: (H_img, W_img) of the original image.
        out_h, out_w: spatial grid size to pool the ROI down to.
    """
    _, C, H, W = feature_map.shape
    ih, iw = img_shape
    x1, y1, x2, y2 = box_xyxy
    fx1 = max(0, int(x1 / iw * W))
    fy1 = max(0, int(y1 / ih * H))
    fx2 = min(W, max(fx1 + 1, int(np.ceil(x2 / iw * W))))
    fy2 = min(H, max(fy1 + 1, int(np.ceil(y2 / ih * H))))
    crop = feature_map[0, :, fy1:fy2, fx1:fx2]  # (C, fH, fW)
    pooled = nn.functional.adaptive_avg_pool2d(
        crop.unsqueeze(0), (out_h, out_w))  # (1, C, out_h, out_w)
    return pooled.squeeze(0).flatten().cpu().numpy()  # (C * out_h * out_w,)


# V5 multi-scale ROI pool. p3 is pooled to a 2x2 spatial grid (preserves
# coarse structure of the small-drone region) -> 4 cells * 64 channels = 256-D.
# p5 is pooled to a single 1x1 cell (semantics are global) -> 256-D.
# Concatenated YOLO features = 512-D. Plus 5 metadata = 517-D total input.
# V4 used 1x1 for both -> 320-D total. The 2x2 on p3 gives the MLP spatial
# context that V4 collapsed to a single mean — critical for small-drone vs
# confuser distinctions on Svanstrom.
P3_GRID = (2, 2)       # spatial grid for p3 pooling
P5_GRID = (1, 1)
YOLO_FEAT_DIM = 64 * P3_GRID[0] * P3_GRID[1] + 256 * P5_GRID[0] * P5_GRID[1]  # 256 + 256 = 512
META_DIM = 5
INPUT_DIM = META_DIM + YOLO_FEAT_DIM  # 517


def extract_box_metadata(box_xyxy, conf, img_shape):
    """Extract 5 metadata features from a detection: conf, log_area, aspect,
    rel_cx, rel_cy."""
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


# ── Data collection ────────────────────────────────────────────────────────

def is_jpg(p: Path) -> bool:
    return p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")


def _resolve_labels_dir(img_dir: Path) -> Path:
    """Find YOLO labels dir for this images dir, handling both layouts:

        Layout A (Anti-UAV, Svanstrom, RGB_video):
            <base>/.../images/<files>
            <base>/.../labels/<files>           (sibling of images dir)

        Layout B (Selcom, RGB dataset):
            <base>/images/<split>/<files>
            <base>/labels/<split>/<files>       (split mirrored under labels)
    """
    sibling = img_dir.parent / "labels"
    if sibling.exists():
        return sibling
    mirrored = img_dir.parent.parent / "labels" / img_dir.name
    if mirrored.exists():
        return mirrored
    return sibling  # default — fail later on missing files


def collect_from_source(model, hook, src: SourceConfig):
    """Mine drone TPs and/or confuser FPs from one source per its quotas.

    Returns numpy arrays plus per-sample weight vectors:
        X_tp, y_tp (=1), w_tp, X_fp, y_fp (=0), w_fp
    Each X is shape (n, INPUT_DIM). w arrays carry the per-source sample weight
    so the focal-loss training can up-weight thesis-critical surfaces.
    """
    if not src.path.exists():
        print(f"  SKIP {src.name}: path not found {src.path}")
        empty_X = np.empty((0, INPUT_DIM), dtype=np.float32)
        return empty_X, np.empty(0), np.empty(0), empty_X, np.empty(0), np.empty(0)

    all_images = sorted(p for p in src.path.iterdir() if is_jpg(p))
    if src.filter_prefixes:
        all_images = [p for p in all_images
                      if any(p.name.startswith(pre) for pre in src.filter_prefixes)]
    images = all_images[::src.stride]
    print(f"  Collecting from {src.name}: {len(images)} images "
          f"(stride={src.stride}, kind={src.kind}, rule={src.match_rule}, "
          f"target_d={src.target_drones}, target_c={src.target_confusers})")

    X_tp, X_fp = [], []
    labels_dir = _resolve_labels_dir(src.path) if src.kind == "image_with_gt" else None
    if labels_dir is not None:
        print(f"    labels dir: {labels_dir}  (exists={labels_dir.exists()})")
    t0 = time.time()
    n_imgs_processed = 0

    for img_path in images:
        # Early-stop both buckets met
        if (len(X_tp) >= src.target_drones and
                len(X_fp) >= src.target_confusers):
            break

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        n_imgs_processed += 1
        ih, iw = img_bgr.shape[:2]

        gt_boxes = []
        if labels_dir is not None:
            lbl_path = labels_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                for line in lbl_path.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5 and int(parts[0]) == src.drone_class:
                        xc, yc, bw, bh = map(float, parts[1:5])
                        x1 = (xc - bw / 2) * iw
                        y1 = (yc - bh / 2) * ih
                        x2 = (xc + bw / 2) * iw
                        y2 = (yc + bh / 2) * ih
                        gt_boxes.append((x1, y1, x2, y2))

        hook.clear()
        results = model.predict(img_bgr, imgsz=src.imgsz, conf=CONF_THR,
                                verbose=False, device="cuda")
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue

        dets = []
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            dets.append((tuple(xyxy), float(boxes.conf[i])))

        for det_box, det_conf in dets:
            # Decide label for this detection
            if src.kind == "image_no_gt":
                is_tp = False  # every detection is a confuser
            else:
                is_tp = _match_det_to_gt(det_box, gt_boxes, src.match_rule)

            # Quota gate
            if is_tp:
                if len(X_tp) >= src.target_drones:
                    continue
                feat = _extract_detection_features(
                    hook, det_box, (ih, iw), det_conf)
                X_tp.append(feat)
            else:
                if len(X_fp) >= src.target_confusers:
                    continue
                feat = _extract_detection_features(
                    hook, det_box, (ih, iw), det_conf)
                X_fp.append(feat)

    dt = max(time.time() - t0, 0.1)
    fps = n_imgs_processed / dt
    print(f"  Done {src.name}: {len(X_tp)} TPs + {len(X_fp)} FPs  "
          f"({n_imgs_processed} imgs, {fps:.1f} fps)")

    X_tp_arr = (np.array(X_tp, dtype=np.float32)
                if X_tp else np.empty((0, INPUT_DIM), dtype=np.float32))
    X_fp_arr = (np.array(X_fp, dtype=np.float32)
                if X_fp else np.empty((0, INPUT_DIM), dtype=np.float32))
    y_tp_arr = np.ones(len(X_tp), dtype=np.float32)
    y_fp_arr = np.zeros(len(X_fp), dtype=np.float32)
    w_tp_arr = np.full(len(X_tp), src.weight_drone, dtype=np.float32)
    w_fp_arr = np.full(len(X_fp), src.weight_confuser, dtype=np.float32)

    return X_tp_arr, y_tp_arr, w_tp_arr, X_fp_arr, y_fp_arr, w_fp_arr


def _iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * max(box_a[3] - box_a[1], 1)
    area_b = (box_b[2] - box_b[0]) * max(box_b[3] - box_b[1], 1)
    return inter / max(area_a + area_b - inter, 1)


def _iop(det_box, gt_box):
    """Intersection over Prediction area. For Svanstrom/Selcom where GT boxes
    are larger than the actual drone: a tight detection inside a loose GT box
    has IoU<0.5 but IoP=1.0. Production scoring uses this rule for Svanstrom."""
    x1 = max(det_box[0], gt_box[0])
    y1 = max(det_box[1], gt_box[1])
    x2 = min(det_box[2], gt_box[2])
    y2 = min(det_box[3], gt_box[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    det_area = (det_box[2] - det_box[0]) * max(det_box[3] - det_box[1], 1)
    return inter / max(det_area, 1)


def _match_det_to_gt(det_box, gt_boxes, rule: str) -> bool:
    """Return True if det matches ANY GT under the given rule (iou or iop)."""
    if not gt_boxes:
        return False
    if rule == "iop":
        return any(_iop(det_box, gt) >= IOP_THR for gt in gt_boxes)
    return any(_iou(det_box, gt) >= IOU_THR for gt in gt_boxes)


_P3_DIM = 64 * P3_GRID[0] * P3_GRID[1]   # 256
_P5_DIM = 256 * P5_GRID[0] * P5_GRID[1]  # 256


def _extract_detection_features(hook: DetectInputHook, box_xyxy, img_shape, conf):
    """Extract metadata + multi-scale p3+p5 features for one detection.

    V5 change: p3 is pooled to a P3_GRID (default 2x2) grid -> 256-D, retaining
    coarse spatial structure of the box. p5 is pooled to a P5_GRID (1x1)
    cell -> 256-D, semantic-only. Output: 5 + 256 + 256 = 517-D.
    """
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


# ── Classifier definitions (all follow sklearn interface) ──────────────────

class LogRegWrapper:
    """LogisticRegression. fit() accepts optional per-sample weights."""
    def __init__(self, C=1.0, class_weight="balanced", max_iter=2000, seed=SEED):
        self.C = C
        self.class_weight_val = class_weight
        self.max_iter = max_iter
        self.seed = seed
        self.model = None
    def fit(self, X, y, sample_weight=None):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(C=self.C, class_weight=self.class_weight_val,
                                        max_iter=self.max_iter, random_state=self.seed)
        self.model.fit(X, y, sample_weight=sample_weight)
        return self
    def predict(self, X):
        return self.model.predict(X)
    def predict_proba(self, X):
        return self.model.predict_proba(X)


class RFWrapper:
    """RandomForest. fit() accepts optional per-sample weights."""
    def __init__(self, n_estimators=150, max_depth=8, class_weight="balanced", seed=SEED):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight_val = class_weight
        self.seed = seed
        self.model = None
    def fit(self, X, y, sample_weight=None):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=self.n_estimators,
                                            max_depth=self.max_depth,
                                            class_weight=self.class_weight_val,
                                            random_state=self.seed, n_jobs=-1)
        self.model.fit(X, y, sample_weight=sample_weight)
        return self
    def predict(self, X):
        return self.model.predict(X)
    def predict_proba(self, X):
        return self.model.predict_proba(X)


class XGBWrapper:
    """XGBoost. fit() accepts optional per-sample weights."""
    def __init__(self, n_estimators=150, max_depth=5, learning_rate=0.1, seed=SEED):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        self.model = None
    def fit(self, X, y, sample_weight=None):
        from xgboost import XGBClassifier
        pos_w = (y == 0).sum() / max((y == 1).sum(), 1)
        self.model = XGBClassifier(n_estimators=self.n_estimators,
                                    max_depth=self.max_depth,
                                    learning_rate=self.learning_rate,
                                    scale_pos_weight=pos_w,
                                    random_state=self.seed, n_jobs=-1,
                                    verbosity=0, use_label_encoder=False)
        self.model.fit(X, y, sample_weight=sample_weight)
        return self
    def predict(self, X):
        return self.model.predict(X)
    def predict_proba(self, X):
        return self.model.predict_proba(X)


class FocalLoss(nn.Module):
    """Binary focal loss with label smoothing and optional per-sample weights.

    L = sample_weight * alpha_t * (1 - p_t)^gamma * BCE(logit, y_smooth)
    where alpha_t = alpha for positives (drones), 1-alpha for negatives.
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0,
                 label_smoothing: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = label_smoothing

    def forward(self, logits, targets, sample_weights=None):
        # Label smoothing: y=1 -> 1-eps/2, y=0 -> eps/2
        y_smooth = targets * (1.0 - self.eps) + 0.5 * self.eps
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, y_smooth, reduction="none")
        with torch.no_grad():
            p = torch.sigmoid(logits)
            pt = torch.where(targets >= 0.5, p, 1.0 - p)
            focal = (1.0 - pt).clamp(min=0.0) ** self.gamma
            alpha_t = torch.where(
                targets >= 0.5,
                torch.full_like(targets, self.alpha),
                torch.full_like(targets, 1.0 - self.alpha))
        loss = alpha_t * focal * bce
        if sample_weights is not None:
            loss = loss * sample_weights
        return loss.mean()


class MLPWrapper:
    """V5 MLP classifier: bigger arch, BatchNorm, focal loss, sample weights.

    Architecture: INPUT -> (Linear -> BN -> ReLU -> Dropout) * N -> Linear(1).
    Default hidden_dims=(512,256,128,64) gives ~300k params. Loss is focal
    with label smoothing 0.1; optional sample_weight up-weights thesis-
    critical surfaces (Svanstrom, real-video).
    """
    def __init__(self, input_dim, hidden_dims=(512, 256, 128, 64),
                 lr=1e-3, epochs=120, batch_size=128, device="cuda",
                 dropout=0.3, focal_alpha=0.75, focal_gamma=2.0,
                 label_smoothing=0.1, use_batchnorm=True):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.dropout = dropout
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.label_smoothing = label_smoothing
        self.use_batchnorm = use_batchnorm
        self.net = None
        self.scaler = None
        self.history = []
        self._fitted = False

    def _build_net(self):
        dims = [self.input_dim, *self.hidden_dims, 1]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if self.use_batchnorm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        return nn.Sequential(*layers).to(self.device)

    def fit(self, X, y, sample_weight=None):
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X).astype(np.float32)

        self.net = self._build_net()

        X_t = torch.from_numpy(Xs).to(self.device)
        y_t = torch.from_numpy(y.astype(np.float32)).to(self.device).unsqueeze(1)
        if sample_weight is not None:
            sw_t = torch.from_numpy(
                np.asarray(sample_weight, dtype=np.float32)
            ).to(self.device).unsqueeze(1)
        else:
            sw_t = None

        opt = torch.optim.AdamW(
            self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        # Cosine annealing from lr to lr/100 over training
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.epochs, eta_min=self.lr * 0.01)
        criterion = FocalLoss(self.focal_alpha, self.focal_gamma,
                              self.label_smoothing)

        n = len(X_t)
        for ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, device=self.device)
            losses = []
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                # BatchNorm needs >1 sample; skip a trailing 1-sample minibatch
                if len(idx) < 2 and self.use_batchnorm:
                    continue
                logit = self.net(X_t[idx])
                sw_batch = sw_t[idx] if sw_t is not None else None
                loss = criterion(logit, y_t[idx], sw_batch)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss))
            sched.step()
            if losses:
                self.history.append(float(np.mean(losses)))
        self._fitted = True
        return self

    @torch.no_grad()
    def _forward_eval(self, X):
        Xs = self.scaler.transform(X).astype(np.float32)
        X_t = torch.from_numpy(Xs).to(self.device)
        self.net.eval()  # BatchNorm: use running stats
        logit = self.net(X_t).squeeze(1)
        return torch.sigmoid(logit).cpu().numpy()

    def predict(self, X):
        p = self._forward_eval(X)
        return (p >= 0.5).astype(int)

    def predict_proba(self, X):
        p = self._forward_eval(X)
        return np.stack([1 - p, p], axis=1)


# ── Cross-validation ───────────────────────────────────────────────────────

def cross_val_score_f1(clf_class, clf_kwargs, X, y, sample_weight=None,
                        folds=5, seed=SEED):
    """Stratified CV returning (mean_f1, std_f1, best_model). If sample_weight
    is provided, slices and passes it to each fold's fit()."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    f1s, models = [], []
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        m = clf_class(**clf_kwargs)
        if sample_weight is not None:
            m.fit(X_tr, y_tr, sample_weight=sample_weight[tr_idx])
        else:
            m.fit(X_tr, y_tr)
        yp = m.predict(X_va)
        f1s.append(f1_score(y_va, yp, zero_division=0))
        models.append(m)
    mean_f1 = float(np.mean(f1s))
    std_f1 = float(np.std(f1s))
    best_idx = int(np.argmax(f1s))
    return mean_f1, std_f1, models[best_idx]


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate_detector_with_classifier(
    model, hook, classifier,
    ds_name: str, img_dir: Path, stride: int, has_drones: bool,
    rule: str = "iou", feat_type: str = "meta+yolo", imgsz: int = 640,
):
    """Run YOLO + confuser classifier filter on a dataset.

    A detection is kept only if:
      - It passes the YOLO confidence threshold AND
      - The confuser classifier predicts "drone" (y=1)

    Returns dict of metrics.
    """
    images = sorted(p for p in img_dir.iterdir() if is_jpg(p))[::stride]
    print(f"    Eval {ds_name}: {len(images)} images...", end=" ")

    labels_dir = _resolve_labels_dir(img_dir) if has_drones else None

    totals = {"tp": 0, "fp": 0, "fn": 0}
    n_confuser_rejected = 0
    total_dets_before = 0

    t0 = time.time()
    for img_path in images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        ih, iw = img_bgr.shape[:2]

        # Load GT if applicable
        gt_boxes = []
        if has_drones and labels_dir is not None:
            lbl_path = labels_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                for line in lbl_path.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5 and int(parts[0]) == 0:
                        xc, yc, bw, bh = map(float, parts[1:5])
                        x1 = (xc - bw / 2) * iw
                        y1 = (yc - bh / 2) * ih
                        x2 = (xc + bw / 2) * iw
                        y2 = (yc + bh / 2) * ih
                        gt_boxes.append((x1, y1, x2, y2))

        # Run YOLO and extract features (per-surface imgsz)
        hook.clear()
        results = model.predict(img_bgr, imgsz=imgsz, conf=CONF_THR,
                                verbose=False, device="cuda")
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            if has_drones and len(gt_boxes) > 0:
                totals["fn"] += len(gt_boxes)
            continue

        dets = []
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            conf = float(boxes.conf[i])
            dets.append((xyxy, conf))

        total_dets_before += len(dets)

        # Apply confuser classifier
        filtered_dets = []
        for det_box, det_conf in dets:
            feat = _extract_detection_features(
                hook, det_box, (ih, iw), det_conf)
            if feat_type == "meta_only":
                feat = feat[:5]
            elif feat_type == "yolo_only":
                feat = feat[5:]
            feat = feat.reshape(1, -1)
            yp = int(classifier.predict(feat)[0])
            if yp == 1:
                filtered_dets.append((det_box, det_conf))
            else:
                n_confuser_rejected += 1

        # Score filtered detections against GT
        if has_drones:
            tp, fp, fn = score_detections(
                filtered_dets, gt_boxes, rule=rule,
                iou_thr=IOU_THR, iop_thr=IOP_THR)
            totals["tp"] += tp
            totals["fp"] += fp
            totals["fn"] += fn
        else:
            # Confuser dataset: any remaining detection = FP
            totals["fp"] += len(filtered_dets)
            totals["fn"] += 0  # no GT

    elapsed = time.time() - t0
    fps = len(images) / max(elapsed, 0.01)
    print(f"{fps:.1f} fps")

    result = {
        "dataset": ds_name,
        "n_images": len(images),
        "totals": totals,
        "n_confuser_rejected": n_confuser_rejected,
        "total_dets_before": total_dets_before,
        "elapsed_s": round(elapsed, 1),
    }
    if has_drones:
        prf = compute_prf(totals["tp"], totals["fp"], totals["fn"])
        result["precision"] = prf["precision"]
        result["recall"] = prf["recall"]
        result["f1"] = prf["f1"]

    return result


def evaluate_bare(model, hook, ds_name, img_dir, stride, has_drones, rule="iou"):
    """Evaluate YOLO bare (no classifier filter). Used for baseline comparison."""
    return evaluate_detector_with_classifier(
        model, hook, _PassThroughClassifier(), ds_name, img_dir,
        stride, has_drones, rule)


class _PassThroughClassifier:
    """No-op classifier that passes all detections through."""
    def predict(self, X):
        return np.ones(len(X), dtype=int)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Overnight hard-negative distillation (Option 4)")
    parser.add_argument("--quick", action="store_true",
                        help="Use larger strides / fewer samples")
    parser.add_argument("--phase", type=int, default=0,
                        help="Resume from phase: 0=all, 1=features, 2=train, 3=eval")
    args = parser.parse_args()

    quick_mode = args.quick
    sources = SOURCES
    if quick_mode:
        # Smoke test: scale strides up ~5x and quotas down to ~5% of full
        global EVAL_STRIDE
        sources = [
            SourceConfig(
                name=s.name, path=s.path,
                stride=max(1, s.stride * 5),
                kind=s.kind,
                target_drones=max(0, s.target_drones // 20),
                target_confusers=max(0, s.target_confusers // 20),
                weight_drone=s.weight_drone,
                weight_confuser=s.weight_confuser,
                filter_prefixes=s.filter_prefixes,
            )
            for s in SOURCES
        ]
        for k in EVAL_STRIDE:
            EVAL_STRIDE[k] *= 2
        print("⚡ QUICK MODE — quotas /20, strides x5")

    print("=" * 72)
    print("  OPTION 4: Hard-negative distillation with YOLO embeddings")
    print("=" * 72)

    # ── Phase 0: Setup Model ─────────────────────────────────────────────
    # NOTE: Load models one at a time to avoid CUDA OOM (imgsz=1280).
    # ft4 baseline is loaded later in Phase 3.
    print("\n── Phase 0: Loading base model ──")

    # V4: train on FT4 R3 features directly (production-parity feature distribution).
    base_model_path = MODEL_PATHS["ft4_r3"]
    ft4_model_path = MODEL_PATHS["ft4_r3"]

    print(f"  Base detector (FT4 R3, production): {base_model_path}")
    model = YOLO(base_model_path)
    hook = DetectInputHook()
    handle = hook.register(model)

    # ── Phase 1: Collect training data ───────────────────────────────────
    phase1_cache = OUT_DIR / "training_data.npz"
    phase1_meta = OUT_DIR / "training_meta.json"

    if args.phase <= 1:
        print("\n── Phase 1: Collecting training data (V5 per-source quotas) ──")

        X_chunks, y_chunks, w_chunks = [], [], []
        source_counts = []  # for logging / meta JSON

        for src in sources:
            X_tp, y_tp, w_tp, X_fp, y_fp, w_fp = collect_from_source(
                model, hook, src)
            if len(X_tp) > 0:
                X_chunks.append(X_tp); y_chunks.append(y_tp); w_chunks.append(w_tp)
            if len(X_fp) > 0:
                X_chunks.append(X_fp); y_chunks.append(y_fp); w_chunks.append(w_fp)
            source_counts.append({
                "name": src.name,
                "n_drones": int(len(X_tp)),
                "n_confusers": int(len(X_fp)),
                "weight_drone": src.weight_drone,
                "weight_confuser": src.weight_confuser,
                "target_drones": src.target_drones,
                "target_confusers": src.target_confusers,
            })

        if not X_chunks:
            print("  FATAL: no samples collected. Check source paths.")
            return

        X_all = np.concatenate(X_chunks, axis=0)
        y_all = np.concatenate(y_chunks, axis=0)
        w_all = np.concatenate(w_chunks, axis=0)

        # Shuffle
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(len(X_all))
        X_all = X_all[perm]; y_all = y_all[perm]; w_all = w_all[perm]

        # Save (V5: also persist per-sample weights)
        np.savez_compressed(phase1_cache, X=X_all, y=y_all, w=w_all)
        with open(phase1_meta, "w") as f:
            json.dump({
                "n_total": int(len(X_all)),
                "n_drone": int((y_all == 1).sum()),
                "n_confuser": int((y_all == 0).sum()),
                "feature_dim": int(X_all.shape[1]),
                "metadata_dim": META_DIM,
                "yolo_feat_dim": int(X_all.shape[1] - META_DIM),
                "yolo_feat_source": (
                    f"p3 at {P3_GRID[0]}x{P3_GRID[1]} ({_P3_DIM}-D) + "
                    f"p5 at {P5_GRID[0]}x{P5_GRID[1]} ({_P5_DIM}-D) = "
                    f"{YOLO_FEAT_DIM}-D"),
                "base_detector": base_model_path,
                "per_source_counts": source_counts,
                "weight_mean": float(w_all.mean()),
                "weight_min": float(w_all.min()),
                "weight_max": float(w_all.max()),
            }, f, indent=2)

        n_d = int((y_all == 1).sum())
        n_c = int((y_all == 0).sum())
        print(f"\n  Collected: {n_d} drone + {n_c} confuser = {len(X_all)} total")
        print(f"  Feature dim: {X_all.shape[1]} = "
              f"{META_DIM} metadata + {X_all.shape[1]-META_DIM} YOLO")
        print(f"  Weight stats: min={w_all.min():.2f} mean={w_all.mean():.2f} max={w_all.max():.2f}")
    else:
        print("\n── Phase 1: Loading cached training data ──")
        z = np.load(phase1_cache)
        X_all, y_all = z["X"], z["y"]
        w_all = z["w"] if "w" in z.files else np.ones(len(y_all), dtype=np.float32)
        with open(phase1_meta) as f:
            meta_info = json.load(f)
        print(f"  Loaded: {meta_info['n_drone']} drone + {meta_info['n_confuser']} "
              f"confuser = {meta_info['n_total']} samples")
        print(f"  Feature dim: {meta_info['feature_dim']}")

    # ── Phase 2: Train classifiers ───────────────────────────────────────
    phase2_cache = OUT_DIR / "classifiers.pkl"

    if args.phase <= 2:
        print("\n── Phase 2: Training classifiers ──")

        # Metadata-only features (first 5 columns) as baseline
        X_meta = X_all[:, :5]
        X_yolo = X_all[:, 5:]
        X_full = X_all

        classifiers = {}

        for feat_name, feat_X in [("meta_only", X_meta), ("yolo_only", X_yolo), ("meta+yolo", X_full)]:
            name = f"logreg_{feat_name}"
            print(f"\n  Training {name} ({feat_X.shape[1]} features)...")
            mean_f1, std_f1, best_m = cross_val_score_f1(
                LogRegWrapper, {}, feat_X, y_all, sample_weight=w_all)
            classifiers[name] = (best_m, mean_f1, std_f1, feat_name)
            print(f"    CV F1: {mean_f1:.4f} ± {std_f1:.4f}")

            name = f"rf_{feat_name}"
            print(f"  Training {name}...")
            mean_f1, std_f1, best_m = cross_val_score_f1(
                RFWrapper, {}, feat_X, y_all, sample_weight=w_all)
            classifiers[name] = (best_m, mean_f1, std_f1, feat_name)
            print(f"    CV F1: {mean_f1:.4f} ± {std_f1:.4f}")

            name = f"xgb_{feat_name}"
            print(f"  Training {name}...")
            mean_f1, std_f1, best_m = cross_val_score_f1(
                XGBWrapper, {}, feat_X, y_all, sample_weight=w_all)
            classifiers[name] = (best_m, mean_f1, std_f1, feat_name)
            print(f"    CV F1: {mean_f1:.4f} ± {std_f1:.4f}")

            name = f"mlp_{feat_name}"
            print(f"  Training {name} (V5 arch: focal+BN+sample weights)...")
            mlp_kwargs = {"input_dim": feat_X.shape[1]}
            mean_f1, std_f1, best_m = cross_val_score_f1(
                MLPWrapper, mlp_kwargs, feat_X, y_all, sample_weight=w_all)
            classifiers[name] = (best_m, mean_f1, std_f1, feat_name)
            print(f"    CV F1: {mean_f1:.4f} ± {std_f1:.4f}")

        # Save all classifiers (V2-compatible cache)
        with open(phase2_cache, "wb") as f:
            pickle.dump(classifiers, f)
        print(f"\n  Saved {len(classifiers)} classifiers to {phase2_cache}")

        # V5: persist a callable verifier artifact for the head-to-head harness.
        # We save the best MLP-on-fused-features (mlp_meta+yolo) since that uses
        # the full multi-scale p3+p5+metadata input the harness will reproduce.
        mlp_key = "mlp_meta+yolo"
        if mlp_key in classifiers:
            best_mlp, cv_f1, cv_std, _ = classifiers[mlp_key]
            artifact_path = OUT_DIR / "classifiers" / "mlp_v5.pt"
            torch.save({
                "state_dict": best_mlp.net.state_dict(),
                "scaler_mean": torch.from_numpy(
                    best_mlp.scaler.mean_.astype(np.float32)),
                "scaler_scale": torch.from_numpy(
                    best_mlp.scaler.scale_.astype(np.float32)),
                "input_dim": int(best_mlp.input_dim),
                "hidden_dims": list(best_mlp.hidden_dims),
                "threshold": 0.5,
                "cv_f1": float(cv_f1),
                "cv_std": float(cv_std),
                "feature_schema": (
                    f"{META_DIM} metadata + p3@{P3_GRID[0]}x{P3_GRID[1]} ({_P3_DIM}-D) "
                    f"+ p5@{P5_GRID[0]}x{P5_GRID[1]} ({_P5_DIM}-D) = "
                    f"{META_DIM + YOLO_FEAT_DIM}-D"),
                "metadata_order": ["conf", "log_area", "aspect", "rel_cx", "rel_cy"],
                "p3_grid": list(P3_GRID),
                "p5_grid": list(P5_GRID),
                "use_batchnorm": True,
                "dropout": 0.3,
                "base_detector": base_model_path,
            }, artifact_path)
            print(f"  Saved callable V5 artifact: {artifact_path}  (CV F1={cv_f1:.4f})")
        else:
            print(f"  WARN: {mlp_key} not in classifiers; skipping mlp_v5.pt save")
    else:
        print("\n── Phase 2: Loading cached classifiers ──")
        with open(phase2_cache, "rb") as f:
            classifiers = pickle.load(f)
        print(f"  Loaded {len(classifiers)} classifiers")

    # ── Phase 3: Evaluate ────────────────────────────────────────────────
    print("\n── Phase 3: Evaluation ──")

    # Per-surface Phase 3 eval config: (ds_name, img_dir, has_drones, rule, imgsz)
    # Svanstrom needs imgsz=1280 (native 640x480, drones unresolvable at 640);
    # everything else uses production imgsz=640.
    phase3_surfaces = [
        ("confuser_test", CONFUSER_TEST,  False, "iou",  640),
        ("selcom_val",    SELCOM_VAL,     True,  "iop",  1280),  # FT4 R~0.10 at 640
        ("antiuav",       ANTIUAV_DIR,    True,  "iou",  640),
        ("svanstrom",     SVANSTROM_DIR,  True,  "iop",  1280),
    ]

    def run_surface(model, hook, classifier, ds_name, img_dir,
                    stride, has_drones, rule, imgsz):
        """Wrapper with error handling + per-surface imgsz."""
        if not img_dir.exists():
            print(f"  SKIP {ds_name}: {img_dir} not found")
            return None
        return evaluate_detector_with_classifier(
            model, hook, classifier, ds_name, img_dir,
            stride, has_drones, rule, imgsz=imgsz)

    # Collect bare detector results (no filter) + ft4 baseline
    eval_results = {}

    # Bare ft4-features model — phase 1's base detector is already FT4 R3
    print("\n  Baseline: bare FT4 R3 (no classifier)")
    for ds_name, img_dir, has_drones, rule, imgsz in phase3_surfaces:
        stride = EVAL_STRIDE.get(ds_name.split("_")[0], 1)
        r = run_surface(model, hook, _PassThroughClassifier(),
                        ds_name, img_dir, stride, has_drones, rule, imgsz)
        if r is not None:
            eval_results[f"bare_ft4_{ds_name}"] = r

    # Each classifier variant — only MLP + logreg (RF/XGB skipped for speed)
    for clf_name, (clf, cv_f1, cv_std, feat_type) in sorted(classifiers.items()):
        if clf_name.startswith("rf_") or clf_name.startswith("xgb_"):
            print(f"\n  Skipping {clf_name} (too slow)")
            continue
        print(f"\n  Classifier: {clf_name}  (CV F1={cv_f1:.4f})")

        for ds_name, img_dir, has_drones, rule, imgsz in phase3_surfaces:
            if not img_dir.exists():
                continue
            stride = EVAL_STRIDE.get(ds_name.split("_")[0], 1)
            tag = ds_name if has_drones else "confuser"
            r = evaluate_detector_with_classifier(
                model, hook, clf, f"{clf_name}_{tag}",
                img_dir, stride, has_drones, rule, feat_type, imgsz)
            eval_results[f"{clf_name}_{ds_name}"] = r

    # ── Print summary table ──────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)

    # Confuser hallucination comparison
    print(f"\n  CONFOUSER HALLUCINATION RATE (lower = better)")
    print(f"  {'Variant':<35s} {'Halluc':>8s} {'Rejected':>9s} {'Dets_before':>12s}")
    print(f"  {'-' * 66}")
    for key, r in sorted(eval_results.items()):
        if "confuser" not in key:
            continue
        hr = r["totals"]["fp"] / max(r["n_images"], 1)
        variant = key.replace("_confuser_test", "").replace("_confuser", "")
        print(f"  {variant:<35s} {hr:>7.2%}  {r['n_confuser_rejected']:>6d}"
              f"  {r['total_dets_before']:>9d}")

    # Drone P/R/F1 comparison
    for surface in ["selcom_val", "antiuav", "svanstrom"]:
        print(f"\n  {surface.upper()} — Drone P/R/F1 (higher = better)")
        print(f"  {'Variant':<35s} {'TP':>6s} {'FP':>5s} {'FN':>5s} "
              f"{'P':>7s} {'R':>7s} {'F1':>7s}")
        print(f"  {'-' * 73}")
        for key, r in sorted(eval_results.items()):
            if surface not in key:
                continue
            t = r["totals"]
            variant = key.replace(f"_{surface}", "")
            if "precision" in r:
                print(f"  {variant:<35s} {t['tp']:>5d}  {t['fp']:>4d}  {t['fn']:>4d}  "
                      f"{r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f}")
            else:
                print(f"  {variant:<35s} {t['tp']:>5d}  {t['fp']:>4d}  {t['fn']:>4d}  "
                      f"{'—':>7s} {'—':>7s} {'—':>7s}")

    # ── Winner pick ──────────────────────────────────────────────────────
    print("\n  CLASSIFIER CV RANKING (on training data):")
    cv_ranks = []
    for clf_name, (_, cv_f1, cv_std, _) in sorted(classifiers.items()):
        cv_ranks.append((cv_f1, clf_name, cv_std))
    cv_ranks.sort(reverse=True)
    for i, (f1, name, std) in enumerate(cv_ranks):
        medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"  {i+1}."
        print(f"  {medal} {name:<30s} CV F1={f1:.4f} ± {std:.4f}")

    # ── Save all results ─────────────────────────────────────────────────
    out_path = OUT_DIR / "distill_results.json"
    serializable = {}
    for key, r in eval_results.items():
        serializable[key] = r
    serializable["_classifier_cv"] = {
        name: {"cv_f1": cv_f1, "cv_std": cv_std, "feat_type": ft}
        for name, (_, cv_f1, cv_std, ft) in classifiers.items()
    }
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n  Results saved: {out_path}")

    # Cleanup
    handle.remove()
    print("\n── Done ──")


if __name__ == "__main__":
    main()