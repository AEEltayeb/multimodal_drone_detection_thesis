#!/usr/bin/env python3
"""
Overnight hard-negative distillation with YOLO feature embeddings (Option 4).

Extracts the pre-detect feature maps from YOLO's neck for each detection, then
trains a lightweight confuser classifier on those embeddings. Evaluates against
the ft4 R3 baseline on confuser hallucination AND drone P/R/F1 on all surfaces.

Classifiers compared: LogisticRegression, RandomForest, XGBoost, MLP.

Usage:
    python eval/overnight_confuser_distill.py              # full run (~2hr)
    python eval/overnight_confuser_distill.py --quick      # 1/3 sample (~45min)
    python eval/overnight_confuser_distill.py --phase 3    # resume at eval
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
import warnings
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
OUT_DIR = EVAL_DIR / "results" / "_overnight_distill"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATHS = {
    "selcom_ft3_1280": str(REPO / "RGB model" / "Yolo26n_selcom_mixed_ft3_1280" / "weights" / "best.pt"),
    "ft4_r3":          str(REPO / "RGB model" / "Yolo26n_selcom_confuser_ft4_1280" / "weights" / "best.pt"),
}

ANTIUAV_DIR  = Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images")
ANTIUAV_VAL  = Path("G:/drone/Anti-UAV-RGBT_yolo_converted/val/RGB/images")
SVANSTROM_DIR = Path("G:/drone/svanstrom_paired/RGB/images")
SELCOM_VAL   = Path("G:/drone/_finetune_selcom_mixed_ft2/images/val")
CONFUSER_TRAIN = Path("G:/drone/rgb_confusers_merged/images/train")
CONFUSER_VAL   = Path("G:/drone/rgb_confusers_merged/images/val")
CONFUSER_TEST  = Path("G:/drone/rgb_confusers_merged/images/test")

# ── Parameters ──────────────────────────────────────────────────────────────
IMGSZ = 1280
CONF_THR = 0.25
IOU_THR = 0.5
IOP_THR = 0.5
SEED = 42

# Sampling: strides for training data collection (1 = every frame)
TRAIN_STRIDE_CONFUSER = 5   # ~4,357 imgs → sample for FPs
TRAIN_STRIDE_DRONE    = 10  # sample for TPs
TRAIN_MAX_DRONE_TP    = 2000
TRAIN_MAX_CONFUSER_FP = 2000

# Evaluation strides (~1000 images each)
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
    """Adaptive average pool a box region from a feature map.

    Args:
        feature_map: (1, C, H, W) tensor.
        box_xyxy: (x1, y1, x2, y2) in *image* coordinates.
        img_shape: (H_img, W_img) of the original image.
    Returns:
        (C,) numpy array.
    """
    _, C, H, W = feature_map.shape
    ih, iw = img_shape
    x1, y1, x2, y2 = box_xyxy
    # Project to feature map coordinates (clamp to valid range)
    fx1 = max(0, int(x1 / iw * W))
    fy1 = max(0, int(y1 / ih * H))
    fx2 = min(W, max(fx1 + 1, int(np.ceil(x2 / iw * W))))
    fy2 = min(H, max(fy1 + 1, int(np.ceil(y2 / ih * H))))
    crop = feature_map[0, :, fy1:fy2, fx1:fx2]  # (C, fH, fW)
    pooled = nn.functional.adaptive_avg_pool2d(crop.unsqueeze(0), (out_h, out_w))
    return pooled.squeeze().cpu().numpy()


# YOLO feature dimension is always 256 (highest-level FPN scale, p5).
# This ensures all training samples have the same feature vector length:
#   5 metadata + 256 YOLO = 261 total
# Lower FPN scales (p3/p4) are not used because their channel counts differ
# and consistent dimensionality is required for classifier training.
YOLO_FEAT_DIM = 256


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


def collect_predictions(model, hook, img_dir: Path, stride: int, max_samples: int,
                        has_gt: bool = False, category: str = "unknown"):
    """Run YOLO predictions and collect per-detection features + labels.

    Args:
        model: YOLO instance.
        hook: DetectInputHook instance (registered).
        img_dir: Directory with images.
        stride: Take every N-th image.
        max_samples: Max detections to collect.
        has_gt: If True, load GT from ../labels/<name>.txt and label detections.
                If False, all detections are labelled as confuser (y=0).
        category: For logging.

    Returns:
        X: list of feature vectors (each is metadata + YOLO features).
        y: list of labels (1=drone, 0=confuser).
        meta: list of dicts with detection-level info.
    """
    images = sorted(p for p in img_dir.iterdir() if is_jpg(p))[::stride]
    print(f"  Collecting from {category}: {len(images)} images (stride={stride})")

    X, y, meta = [], [], []
    t0 = time.time()

    labels_dir = img_dir.parent / "labels" if has_gt else None

    for idx, img_path in enumerate(images):
        # Load image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        ih, iw = img_bgr.shape[:2]

        # Load GT if available
        gt_boxes = []
        if has_gt and labels_dir is not None:
            lbl_path = labels_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                for line in lbl_path.read_text().splitlines():
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cls_id = int(parts[0])
                        if cls_id == 0:  # drone class only
                            xc, yc, bw, bh = map(float, parts[1:5])
                            x1 = (xc - bw / 2) * iw
                            y1 = (yc - bh / 2) * ih
                            x2 = (xc + bw / 2) * iw
                            y2 = (yc + bh / 2) * ih
                            gt_boxes.append((x1, y1, x2, y2))

        # Run YOLO
        hook.clear()
        results = model.predict(img_bgr, imgsz=IMGSZ, conf=CONF_THR,
                                verbose=False, device="cuda")
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            continue

        dets = []
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            dets.append((tuple(xyxy), float(boxes.conf[i])))

        # Match each detection to GT or label as confuser
        if has_gt and gt_boxes:
            for det_box, det_conf in dets:
                # Check IoU with any GT box
                is_tp = False
                for gt_box in gt_boxes:
                    if _iou(det_box, gt_box) >= IOU_THR:
                        is_tp = True
                        break
                if not is_tp:
                    continue  # skip detections that don't match GT (ambiguous)

                # Extract features
                feat = _extract_detection_features(
                    hook, det_box, (ih, iw), det_conf)
                X.append(feat)
                y.append(1)  # drone
                meta.append({"img": str(img_path), "cat": category,
                             "label": 1, "conf": det_conf})
        else:
            # No GT: all detections are confuser FPs
            for det_box, det_conf in dets:
                feat = _extract_detection_features(
                    hook, det_box, (ih, iw), det_conf)
                X.append(feat)
                y.append(0)  # confuser
                meta.append({"img": str(img_path), "cat": category,
                             "label": 0, "conf": det_conf})

        if len(X) >= max_samples:
            X = X[:max_samples]
            y = y[:max_samples]
            meta = meta[:max_samples]
            break

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            fps = (idx + 1) / elapsed
            print(f"    {idx+1}/{len(images)}  {fps:.1f} fps  "
                  f"collected {len(X)} samples", end="\r")

    dt = time.time() - t0
    fps = len(images) / max(dt, 0.1)
    print(f"  Done: {len(X)} samples from {category}  ({fps:.1f} fps)")
    return np.array(X), np.array(y), meta


def _iou(box_a, box_b):
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * max(box_a[3] - box_a[1], 1)
    area_b = (box_b[2] - box_b[0]) * max(box_b[3] - box_b[1], 1)
    return inter / max(area_a + area_b - inter, 1)


def _extract_detection_features(hook: DetectInputHook, box_xyxy, img_shape, conf):
    """Extract metadata + YOLO backbone features for one detection.

    Always pools from the highest-level FPN scale (p5, 256ch) for consistent
    feature dimensionality across all detections.
    """
    # 5 metadata features
    meta = extract_box_metadata(box_xyxy, conf, img_shape)

    # YOLO features: always from p5 (256-D, highest semantic level)
    if hook.p5 is not None:
        yolo_feat = roi_pool(hook.p5, box_xyxy, img_shape)
    else:
        yolo_feat = np.zeros(YOLO_FEAT_DIM, dtype=np.float32)

    return np.concatenate([meta, yolo_feat]).astype(np.float32)


# ── Classifier definitions (all follow sklearn interface) ──────────────────

class LogRegWrapper:
    """LogisticRegression with sklearn fit/predict interface."""
    def __init__(self, C=1.0, class_weight="balanced", max_iter=2000, seed=SEED):
        self.C = C
        self.class_weight_val = class_weight
        self.max_iter = max_iter
        self.seed = seed
        self.model = None
    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression
        self.model = LogisticRegression(C=self.C, class_weight=self.class_weight_val,
                                        max_iter=self.max_iter, random_state=self.seed)
        self.model.fit(X, y)
        return self
    def predict(self, X):
        return self.model.predict(X)
    def predict_proba(self, X):
        return self.model.predict_proba(X)


class RFWrapper:
    """RandomForest with sklearn fit/predict interface."""
    def __init__(self, n_estimators=150, max_depth=8, class_weight="balanced", seed=SEED):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.class_weight_val = class_weight
        self.seed = seed
        self.model = None
    def fit(self, X, y):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(n_estimators=self.n_estimators,
                                            max_depth=self.max_depth,
                                            class_weight=self.class_weight_val,
                                            random_state=self.seed, n_jobs=-1)
        self.model.fit(X, y)
        return self
    def predict(self, X):
        return self.model.predict(X)
    def predict_proba(self, X):
        return self.model.predict_proba(X)


class XGBWrapper:
    """XGBoost with sklearn fit/predict interface."""
    def __init__(self, n_estimators=150, max_depth=5, learning_rate=0.1, seed=SEED):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        self.model = None
    def fit(self, X, y):
        from xgboost import XGBClassifier
        pos_w = (y == 0).sum() / max((y == 1).sum(), 1)
        self.model = XGBClassifier(n_estimators=self.n_estimators,
                                    max_depth=self.max_depth,
                                    learning_rate=self.learning_rate,
                                    scale_pos_weight=pos_w,
                                    random_state=self.seed, n_jobs=-1,
                                    verbosity=0, use_label_encoder=False)
        self.model.fit(X, y)
        return self
    def predict(self, X):
        return self.model.predict(X)
    def predict_proba(self, X):
        return self.model.predict_proba(X)


class MLPWrapper:
    """Lightweight MLP classifier with sklearn interface."""
    def __init__(self, input_dim, hidden_dims=(128, 64), lr=1e-3,
                 epochs=100, batch_size=64, device="cuda"):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.net = None
        self.history = []
        self._fitted = False

    def fit(self, X, y):
        import torch.nn.functional as F
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)

        dims = [self.input_dim, *self.hidden_dims, 1]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers).to(self.device)

        X_t = torch.from_numpy(Xs.astype(np.float32)).to(self.device)
        y_t = torch.from_numpy(y.astype(np.float32)).to(self.device).unsqueeze(1)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        pos_w = (y == 0).sum() / max((y == 1).sum(), 1)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w]).to(self.device))

        n = len(X_t)
        for ep in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            losses = []
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
                logit = self.net(X_t[idx])
                loss = criterion(logit, y_t[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss))
            self.history.append(float(np.mean(losses)))
        self._fitted = True
        return self

    def predict(self, X):
        Xs = self.scaler.transform(X)
        X_t = torch.from_numpy(Xs.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logit = self.net(X_t).squeeze(1)
        p = torch.sigmoid(logit).cpu().numpy()
        return (p >= 0.5).astype(int)

    def predict_proba(self, X):
        Xs = self.scaler.transform(X)
        X_t = torch.from_numpy(Xs.astype(np.float32)).to(self.device)
        with torch.no_grad():
            logit = self.net(X_t).squeeze(1)
        p = torch.sigmoid(logit).cpu().numpy()
        return np.stack([1 - p, p], axis=1)


# ── Cross-validation ───────────────────────────────────────────────────────

def cross_val_score_f1(clf_class, clf_kwargs, X, y, folds=5, seed=SEED):
    """Stratified CV returning (mean_f1, std_f1, all_models)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    f1s, models = [], []
    for tr_idx, va_idx in skf.split(X, y):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        m = clf_class(**clf_kwargs)
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
    rule: str = "iou", feat_type: str = "meta+yolo"
):
    """Run YOLO + confuser classifier filter on a dataset.

    A detection is kept only if:
      - It passes the YOLO confidence threshold AND
      - The confuser classifier predicts "drone" (y=1)

    Returns dict of metrics.
    """
    images = sorted(p for p in img_dir.iterdir() if is_jpg(p))[::stride]
    print(f"    Eval {ds_name}: {len(images)} images...", end=" ")

    labels_dir = img_dir.parent / "labels" if has_drones else None

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

        # Run YOLO and extract features
        hook.clear()
        results = model.predict(img_bgr, imgsz=IMGSZ, conf=CONF_THR,
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

    if args.quick:
        global TRAIN_STRIDE_CONFUSER, TRAIN_STRIDE_DRONE
        global TRAIN_MAX_DRONE_TP, TRAIN_MAX_CONFUSER_FP
        global EVAL_STRIDE
        TRAIN_STRIDE_CONFUSER = 15
        TRAIN_STRIDE_DRONE = 30
        TRAIN_MAX_DRONE_TP = 500
        TRAIN_MAX_CONFUSER_FP = 500
        for k in EVAL_STRIDE:
            EVAL_STRIDE[k] *= 2
        print("⚡ QUICK MODE (1/3 sample size)")

    print("=" * 72)
    print("  OPTION 4: Hard-negative distillation with YOLO embeddings")
    print("=" * 72)

    # ── Phase 0: Setup Model ─────────────────────────────────────────────
    # NOTE: Load models one at a time to avoid CUDA OOM (imgsz=1280).
    # ft4 baseline is loaded later in Phase 3.
    print("\n── Phase 0: Loading base model ──")

    base_model_path = MODEL_PATHS["selcom_ft3_1280"]
    ft4_model_path = MODEL_PATHS["ft4_r3"]

    print(f"  Base detector (best selcom): {base_model_path}")
    model = YOLO(base_model_path)
    hook = DetectInputHook()
    handle = hook.register(model)

    # ── Phase 1: Collect training data ───────────────────────────────────
    phase1_cache = OUT_DIR / "training_data.npz"
    phase1_meta = OUT_DIR / "training_meta.json"

    if args.phase <= 1:
        print("\n── Phase 1: Collecting training data ──")

        # Drone TPs (from Anti-UAV val + Svanstrom + selcom → GT-matched detections)
        # Evaluation uses Anti-UAV TEST — no overlap.
        # Svanstrom has no train/test split; different strides → different images.
        X_drone, y_drone, meta_drone = [], [], []
        n_drone_sources = 3  # antiuav_val, svanstrom, selcom
        max_per_source = max(1, TRAIN_MAX_DRONE_TP // n_drone_sources)
        for src_name, src_dir in [("antiuav_val", ANTIUAV_VAL),
                                   ("svanstrom", SVANSTROM_DIR),
                                   ("selcom_val", SELCOM_VAL)]:
            if not src_dir.exists():
                print(f"  SKIP {src_name}: {src_dir} not found")
                continue
            Xs, ys, ms = collect_predictions(
                model, hook, src_dir,
                stride=TRAIN_STRIDE_DRONE,
                max_samples=max_per_source,
                has_gt=True,
                category=src_name)
            X_drone.append(Xs)
            y_drone.append(ys)
            meta_drone.extend(ms)

        # Confuser FPs (from confuser train → all detections are FPs)
        X_conf, y_conf, meta_conf = [], [], []
        for src_name, src_dir in [("confuser_train", CONFUSER_TRAIN)]:
            if not src_dir.exists():
                print(f"  SKIP {src_name}: {src_dir} not found")
                continue
            Xs, ys, ms = collect_predictions(
                model, hook, src_dir,
                stride=TRAIN_STRIDE_CONFUSER,
                max_samples=TRAIN_MAX_CONFUSER_FP,
                has_gt=False,
                category=src_name)
            X_conf.append(Xs)
            y_conf.append(ys)
            meta_conf.extend(ms)

        # Combine and balance
        X_all = np.concatenate(X_drone + X_conf, axis=0)
        y_all = np.concatenate(y_drone + y_conf, axis=0)

        # Shuffle
        rng = np.random.RandomState(SEED)
        perm = rng.permutation(len(X_all))
        X_all = X_all[perm]
        y_all = y_all[perm]

        # Save
        np.savez_compressed(phase1_cache, X=X_all, y=y_all)
        with open(phase1_meta, "w") as f:
            json.dump({
                "n_total": int(len(X_all)),
                "n_drone": int((y_all == 1).sum()),
                "n_confuser": int((y_all == 0).sum()),
                "n_drone_sources": [{"name": m["cat"], "count": meta_drone.count(m)}
                                     for m in meta_drone],
                "feature_dim": int(X_all.shape[1]),
                "metadata_dim": 5,
                "yolo_feat_dim": int(X_all.shape[1] - 5),
                "yolo_feat_source": "p5 (highest FPN, 256ch)",
            }, f, indent=2)

        n_d = int((y_all == 1).sum())
        n_c = int((y_all == 0).sum())
        print(f"\n  Collected: {n_d} drone TP + {n_c} confuser FP = {len(X_all)} total")
        print(f"  Feature dim: {X_all.shape[1]} (5 metadata + {X_all.shape[1]-5} YOLO)")
    else:
        print("\n── Phase 1: Loading cached training data ──")
        z = np.load(phase1_cache)
        X_all, y_all = z["X"], z["y"]
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
                LogRegWrapper, {}, feat_X, y_all)
            classifiers[name] = (best_m, mean_f1, std_f1, feat_name)
            print(f"    CV F1: {mean_f1:.4f} ± {std_f1:.4f}")

            name = f"rf_{feat_name}"
            print(f"  Training {name}...")
            mean_f1, std_f1, best_m = cross_val_score_f1(
                RFWrapper, {}, feat_X, y_all)
            classifiers[name] = (best_m, mean_f1, std_f1, feat_name)
            print(f"    CV F1: {mean_f1:.4f} ± {std_f1:.4f}")

            name = f"xgb_{feat_name}"
            print(f"  Training {name}...")
            mean_f1, std_f1, best_m = cross_val_score_f1(
                XGBWrapper, {}, feat_X, y_all)
            classifiers[name] = (best_m, mean_f1, std_f1, feat_name)
            print(f"    CV F1: {mean_f1:.4f} ± {std_f1:.4f}")

            name = f"mlp_{feat_name}"
            print(f"  Training {name}...")
            mlp_kwargs = {"input_dim": feat_X.shape[1]}
            mean_f1, std_f1, best_m = cross_val_score_f1(
                MLPWrapper, mlp_kwargs, feat_X, y_all)
            classifiers[name] = (best_m, mean_f1, std_f1, feat_name)
            print(f"    CV F1: {mean_f1:.4f} ± {std_f1:.4f}")

        # Save classifiers
        with open(phase2_cache, "wb") as f:
            pickle.dump(classifiers, f)
        print(f"\n  Saved {len(classifiers)} classifiers to {phase2_cache}")
    else:
        print("\n── Phase 2: Loading cached classifiers ──")
        with open(phase2_cache, "rb") as f:
            classifiers = pickle.load(f)
        print(f"  Loaded {len(classifiers)} classifiers")

    # ── Phase 3: Evaluate ────────────────────────────────────────────────
    print("\n── Phase 3: Evaluation ──")

    def run_surface(model, hook, classifier, ds_name, img_dir,
                    stride, has_drones, rule):
        """Wrapper with error handling."""
        if not img_dir.exists():
            print(f"  SKIP {ds_name}: {img_dir} not found")
            return None
        return evaluate_detector_with_classifier(
            model, hook, classifier, ds_name, img_dir,
            stride, has_drones, rule)

    # Collect bare detector results (no filter) + ft4 baseline
    eval_results = {}

    # Bare ft3 detector (no filter)
    print("\n  Baseline: ft3_1280 bare (no classifier)")
    for ds_name, img_dir, has_drones, rule in [
        ("confuser_test", CONFUSER_TEST, False, "iou"),
        ("selcom_val", SELCOM_VAL, True, "iop"),
        ("antiuav", ANTIUAV_DIR, True, "iou"),
        ("svanstrom", SVANSTROM_DIR, True, "iop"),
    ]:
        r = run_surface(model, hook, _PassThroughClassifier(),
                        ds_name, img_dir, EVAL_STRIDE.get(ds_name.split("_")[0], 1),
                        has_drones, rule)
        if r is not None:
            eval_results[f"bare_ft3_{ds_name}"] = r

    # Bare ft4 detector — load/unload separately to avoid CUDA OOM
    print("\n  Baseline: ft4 R3 bare (no filter)")
    import gc
    model_ft4 = YOLO(ft4_model_path)
    hook_ft4 = DetectInputHook()
    handle_ft4 = hook_ft4.register(model_ft4)
    for ds_name, img_dir, has_drones, rule in [
        ("confuser_test", CONFUSER_TEST, False, "iou"),
        ("selcom_val", SELCOM_VAL, True, "iop"),
        ("antiuav", ANTIUAV_DIR, True, "iou"),
        ("svanstrom", SVANSTROM_DIR, True, "iop"),
    ]:
        r = run_surface(model_ft4, hook_ft4, _PassThroughClassifier(),
                        ds_name, img_dir, EVAL_STRIDE.get(ds_name.split("_")[0], 1),
                        has_drones, rule)
        if r is not None:
            eval_results[f"bare_ft4_{ds_name}"] = r
    handle_ft4.remove()
    del model_ft4, hook_ft4
    gc.collect()
    torch.cuda.empty_cache()

    # Each classifier variant
    for clf_name, (clf, cv_f1, cv_std, feat_type) in sorted(classifiers.items()):
        print(f"\n  Classifier: {clf_name}  (CV F1={cv_f1:.4f})")

        # Confuser test
        if CONFUSER_TEST.exists():
            r = evaluate_detector_with_classifier(
                model, hook, clf, f"{clf_name}_confuser",
                CONFUSER_TEST, EVAL_STRIDE["confuser"],
                has_drones=False, feat_type=feat_type)
            eval_results[f"{clf_name}_confuser_test"] = r

        # Selcom val
        if SELCOM_VAL.exists():
            r = evaluate_detector_with_classifier(
                model, hook, clf, f"{clf_name}_selcom",
                SELCOM_VAL, EVAL_STRIDE["selcom"],
                has_drones=True, rule="iop", feat_type=feat_type)
            eval_results[f"{clf_name}_selcom_val"] = r

        # Anti-UAV
        if ANTIUAV_DIR.exists():
            r = evaluate_detector_with_classifier(
                model, hook, clf, f"{clf_name}_antiuav",
                ANTIUAV_DIR, EVAL_STRIDE["antiuav"],
                has_drones=True, rule="iou", feat_type=feat_type)
            eval_results[f"{clf_name}_antiuav"] = r

        # Svanstrom
        if SVANSTROM_DIR.exists():
            r = evaluate_detector_with_classifier(
                model, hook, clf, f"{clf_name}_svanstrom",
                SVANSTROM_DIR, EVAL_STRIDE["svanstrom"],
                has_drones=True, rule="iop", feat_type=feat_type)
            eval_results[f"{clf_name}_svanstrom"] = r

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