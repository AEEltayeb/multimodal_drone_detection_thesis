"""
Multi-class LDA: Drone vs Bird vs Airplane vs Helicopter.

Separates RGB-model (selcom FT4) detections by confuser CATEGORY, not just the
aggregate drone-vs-confuser split.

Primary data source (verified 2026-05-30): Svanström, all 4 categories on one
in-domain surface at imgsz=1280. The category is encoded in the FILENAME PREFIX
(IR_DRONE_ 11695, IR_AIRPLANE_ 6090, IR_HELICOPTER_ 5627, IR_BIRD_ 5298 frames),
NOT in the GT label class — Svanström GT contains only drone (class-0) boxes;
confuser sequences have no boxes. Each sequence is a single-category scene, so any
detection inside it is an object of that category. We trust the prefix and apply
no IoU/IoP true-positive filter.

Optional (--include-video): also mine the cross-domain RGB-video confuser clips
(V_BIRD_/V_AIRPLANE_/V_HELICOPTER_, imgsz=640) for extra variety.

Output: docs/analysis/images/v5_lda_multiclass.png
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "docs" / "analysis" / "images"
OUT.mkdir(parents=True, exist_ok=True)

# ── Detector paths ────────────────────────────────────────────────────────────
BASE_MODEL = REPO / "RGB model" / "Yolo26n_selcom_confuser_ft4_1280" / "weights" / "best.pt"

# ── Data paths ────────────────────────────────────────────────────────────────
SVANSTROM_DIR = Path("G:/drone/svanstrom_paired/RGB/images")
RGB_VIDEO_TRAIN = Path("G:/drone/RGB_video_rgb_dataset/train/images")
RGB_VIDEO_VAL = Path("G:/drone/RGB_video_rgb_dataset/val/images")
CACHED_NPZ = REPO / "eval" / "results" / "_v5_p3p5_ft4_distill" / "training_data.npz"
CACHED_META = REPO / "eval" / "results" / "_v5_p3p5_ft4_distill" / "training_meta.json"

# ── Feature extraction (mirrors distill_v5_p3p5_ft4.py) ──────────────────────
P3_GRID = (2, 2)
P5_GRID = (1, 1)
META_DIM = 5
IOU_THR = 0.5
IOP_THR = 0.5
CONF_THR = 0.25


class DetectInputHook:
    """Captures YOLO FPN feature maps."""
    def __init__(self):
        self.p3 = self.p4 = self.p5 = None

    def clear(self):
        self.p3 = self.p4 = self.p5 = None

    def _hook(self, module, args):
        x = args[0]
        self.p3 = x[0].detach()
        self.p4 = x[1].detach()
        self.p5 = x[2].detach()

    def register(self, model: YOLO):
        detect_mod = model.model.model[-1]
        return detect_mod.register_forward_pre_hook(self._hook)


def roi_pool(feature_map, box_xyxy, img_shape, out_h=1, out_w=1):
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


def extract_features(hook, box_xyxy, img_shape, conf):
    p3_feat = roi_pool(hook.p3, box_xyxy, img_shape, P3_GRID[0], P3_GRID[1])
    p5_feat = roi_pool(hook.p5, box_xyxy, img_shape, P5_GRID[0], P5_GRID[1])
    x1, y1, x2, y2 = box_xyxy
    ih, iw = img_shape
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    area = bw * bh
    cx = (x1 + x2) / 2.0 / max(iw, 1)
    cy = (y1 + y2) / 2.0 / max(ih, 1)
    meta = np.array([conf, np.log(max(area, 1.0)), bw / max(bh, 1), cx, cy], dtype=np.float32)
    return np.concatenate([meta, p3_feat, p5_feat])


def iou_iop(box_a, box_b):
    """Compute IoU and IoP (intersection over smaller-box area)."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    a2 = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = a1 + a2 - inter
    iou = inter / max(union, 1)
    iop = inter / max(min(a1, a2), 1)
    return iou, iop


def load_yolo_labels(labels_path: Path) -> list[tuple]:
    """Load YOLO-format labels: list of (class_id, x1, y1, x2, y2) in pixel coords.
    Also returns image shape if available from a companion .meta file.
    """
    gts = []
    if not labels_path.exists():
        return gts
    text = labels_path.read_text().strip()
    if not text:
        return gts
    for line in text.split("\n"):
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        # Keep in normalized coords for now; caller converts with img_shape
        gts.append((cls, cx, cy, w, h))
    return gts


def resolve_labels_dir(img_path: Path) -> Path:
    """Find labels file for an image, handling both label dir layouts."""
    sibling = img_path.parent.parent / "labels" / img_path.name
    if sibling.with_suffix(".txt").exists():
        return sibling.with_suffix(".txt")
    # Try mirrored under labels/
    mirrored = img_path.parent.parent.parent / "labels" / img_path.parent.name / img_path.name
    if mirrored.with_suffix(".txt").exists():
        return mirrored.with_suffix(".txt")
    return img_path.parent.parent / "labels" / (img_path.stem + ".txt")


def normalize_label(cls, cx, cy, w, h, img_w, img_h):
    """Convert normalized YOLO coords to pixel coords."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    return cls, x1, y1, x2, y2


# ── Source mining ─────────────────────────────────────────────────────────────

CLASS_NAMES = {0: "Drone", 1: "Bird", 2: "Airplane", 3: "Helicopter", 4: "Other"}
CLASS_COLORS = {0: "green", 1: "red", 2: "blue", 3: "orange", 4: "gray"}


SVAN_PREFIX_MAP = {"IR_DRONE_": 0, "IR_BIRD_": 1, "IR_AIRPLANE_": 2, "IR_HELICOPTER_": 3}


def mine_svanstrom_by_prefix(model, hook, per_class_cap=2500, stride=1):
    """Mine Svanström, labelling each detection by the sequence's filename-prefix
    category (IR_DRONE_/IR_BIRD_/IR_AIRPLANE_/IR_HELICOPTER_).

    Svanström's GT label files only contain drone (class-0) boxes — confuser
    sequences have no boxes — so the GT *class* cannot tell drone from bird from
    airplane. The category is encoded in the filename instead. Each sequence is a
    single-object-category scene, so any detection the model fires inside it is an
    object of that category. We therefore trust the prefix and apply no IoU/IoP
    true-positive filter (this is the same logic as the RGB-video confuser mining,
    but Svanström is in-domain and far larger: ~17k confuser frames at imgsz=1280).
    """
    all_imgs = [p for p in sorted(Path(SVANSTROM_DIR).iterdir())
                if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
    by_class: dict[int, list[Path]] = {0: [], 1: [], 2: [], 3: []}
    for p in all_imgs:
        for pre, cls_id in SVAN_PREFIX_MAP.items():
            if p.name.startswith(pre):
                by_class[cls_id].append(p)
                break

    X_list, y_list = [], []
    for cls_id, imgs in by_class.items():
        imgs = imgs[::stride]
        cat = CLASS_NAMES[cls_id]
        kept = 0
        print(f"Mining Svanström {cat} ({len(imgs)} frames at stride={stride}, imgsz=1280, cap={per_class_cap}) ...")
        for i, img_path in enumerate(imgs):
            if kept >= per_class_cap:
                break
            if i > 0 and i % 200 == 0:
                print(f"  ... {cat}: scanned {i}/{len(imgs)}, kept {kept}")

            results = model(str(img_path), imgsz=1280, conf=CONF_THR, verbose=False)
            dets = results[0].boxes
            img = cv2.imread(str(img_path))
            img_h, img_w = img.shape[:2]

            if dets is None or len(dets) == 0:
                hook.clear()
                del results
                continue

            boxes_xyxy = dets.xyxy.cpu().numpy()
            confs = dets.conf.cpu().numpy()
            for bi in range(len(boxes_xyxy)):
                if kept >= per_class_cap:
                    break
                box = tuple(boxes_xyxy[bi])
                conf = float(confs[bi])
                feat = extract_features(hook, box, (img_h, img_w), conf)
                X_list.append(feat)
                y_list.append(cls_id)  # trust the sequence-level category
                kept += 1

            hook.clear()
            del results, dets
            if torch.cuda.is_available() and i % 100 == 0:
                torch.cuda.empty_cache()
        print(f"  Svanström {cat} done: kept {kept} detections")

    print(f"  Svanström done: {len(X_list)} samples")
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def mine_video_confusers(model, hook, per_class_cap=2500):
    """Mine RGB video confuser sources, labelling each detection by the clip's
    filename-prefix category.

    These clips have NO per-frame GT boxes (the .txt label files are empty); the
    category lives only in the filename prefix (V_BIRD_/V_AIRPLANE_/V_HELICOPTER_).
    Every clip is, by construction, a single-confuser-category scene, so any
    detection the model fires inside it is a confuser of that category. We
    therefore trust the prefix and do NOT apply an IoU true-positive filter
    (doing so would discard the only category signal we have).

    `per_class_cap` bounds detections kept per category so the LDA stays balanced.
    """
    prefix_map = {"V_BIRD_": 1, "V_AIRPLANE_": 2, "V_HELICOPTER_": 3}
    sources = [("rgb_video_train", RGB_VIDEO_TRAIN), ("rgb_video_val", RGB_VIDEO_VAL)]

    # Group images by category across both splits.
    by_class: dict[int, list[Path]] = {1: [], 2: [], 3: []}
    for src_name, src_path in sources:
        if not src_path.exists():
            print(f"  WARN: {src_path} not found, skipping {src_name}")
            continue
        for p in sorted(src_path.iterdir()):
            if p.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            for pre, cls_id in prefix_map.items():
                if p.name.startswith(pre):
                    by_class[cls_id].append(p)
                    break

    X_list, y_list = [], []
    for cls_id, imgs in by_class.items():
        cat = CLASS_NAMES[cls_id]
        kept = 0
        print(f"Mining {cat} confusers ({len(imgs)} frames available, cap={per_class_cap}) ...")
        for i, img_path in enumerate(imgs):
            if kept >= per_class_cap:
                break
            if i > 0 and i % 200 == 0:
                print(f"  ... {cat}: scanned {i}/{len(imgs)}, kept {kept}")

            results = model(str(img_path), imgsz=640, conf=CONF_THR, verbose=False)
            dets = results[0].boxes
            img = cv2.imread(str(img_path))
            img_h, img_w = img.shape[:2]

            if dets is None or len(dets) == 0:
                hook.clear()
                del results
                continue

            boxes_xyxy = dets.xyxy.cpu().numpy()
            confs = dets.conf.cpu().numpy()
            for bi in range(len(boxes_xyxy)):
                if kept >= per_class_cap:
                    break
                box = tuple(boxes_xyxy[bi])
                conf = float(confs[bi])
                feat = extract_features(hook, box, (img_h, img_w), conf)
                X_list.append(feat)
                y_list.append(cls_id)  # trust the clip-level category
                kept += 1

            hook.clear()
            del results, dets
            if torch.cuda.is_available() and i % 100 == 0:
                torch.cuda.empty_cache()
        print(f"  {cat} done: kept {kept} detections")

    print(f"  Video confusers done: {len(X_list)} samples")
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def load_npz_drones(cap=2500):
    """Load drone features from cached NPZ as 'Drone' class (subsampled to `cap`)."""
    z = np.load(CACHED_NPZ)
    mask = z["y"] == 1
    X = z["X"][mask].astype(np.float32)
    if len(X) > cap:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X), cap, replace=False)
        X = X[idx]
    y = np.full(X.shape[0], 0, dtype=np.int64)  # class 0 = Drone
    print(f"  NPZ drones: {len(X)} samples (cap={cap})")
    return X, y


def load_npz_other():
    """Load 'Other' confusers from cached NPZ (non-Svanström, non-video sources)."""
    pass  # We'll skip this — can't partition reliably without source labels


# ── Main ──────────────────────────────────────────────────────────────────────

def main(include_video=False, per_class_cap=2500, svan_stride=2):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load detector
    print(f"Loading YOLO from {BASE_MODEL} ...")
    model = YOLO(str(BASE_MODEL))
    model.to(device)
    hook = DetectInputHook()
    handle = hook.register(model)

    all_X, all_y = [], []

    # Primary source: Svanström by filename prefix — all 4 categories on one
    # in-domain surface at imgsz=1280 (IR_DRONE_/IR_BIRD_/IR_AIRPLANE_/IR_HELICOPTER_).
    X_sv, y_sv = mine_svanstrom_by_prefix(model, hook, per_class_cap=per_class_cap,
                                          stride=svan_stride)
    all_X.append(X_sv)
    all_y.append(y_sv)

    # Optional: add cross-domain RGB-video confusers (imgsz=640) for extra variety.
    if include_video:
        X_vid, y_vid = mine_video_confusers(model, hook, per_class_cap=per_class_cap)
        all_X.append(X_vid)
        all_y.append(y_vid)

    # Remove hook
    handle.remove()

    # Combine
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)

    print(f"\nTotal samples: {len(X)}")
    for cid in range(5):
        n = int((y == cid).sum())
        if n:
            print(f"  {CLASS_NAMES[cid]:12s}: {n:5d} samples")

    # ── Multi-class LDA ──────────────────────────────────────────────────────
    # Use YOLO features only (skip metadata) for cleaner visualization
    yolo_X = X[:, META_DIM:]  # 512-D

    # Only use classes with > 1 sample
    valid_classes = [c for c in range(5) if (y == c).sum() > 1]
    print(f"\nClasses with >1 sample: {[CLASS_NAMES[c] for c in valid_classes]}")

    mask = np.isin(y, valid_classes)
    X_filt = yolo_X[mask]
    y_filt = y[mask]

    # Re-map class IDs to contiguous 0..K-1
    class_map = {old: new for new, old in enumerate(valid_classes)}
    y_contiguous = np.array([class_map[c] for c in y_filt])
    valid_names = [CLASS_NAMES[c] for c in valid_classes]
    valid_colors = [CLASS_COLORS[c] for c in valid_classes]

    n_classes = len(valid_classes)
    n_components = min(n_classes - 1, X_filt.shape[1])

    lda = LinearDiscriminantAnalysis(n_components=n_components)
    Z = lda.fit_transform(X_filt, y_contiguous)
    acc = lda.score(X_filt, y_contiguous)
    print(f"LDA accuracy: {acc:.4f}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    if Z.shape[1] == 1:
        # 1-D LDA -> histogram per class
        fig, ax = plt.subplots(figsize=(14, 6))
        bins = 100
        for cid in range(n_classes):
            c_mask = y_contiguous == cid
            ax.hist(Z[c_mask, 0], bins=bins, alpha=0.5,
                    label=f"{valid_names[cid]} (n={(c_mask).sum()})",
                    color=valid_colors[cid])
        ax.set_xlabel("LDA Component 1 (discriminant axis)")
        ax.set_ylabel("Count")
        ax.set_title(f"V5 Multi-class LDA: YOLO features by sub-class\n"
                     f"Train accuracy: {acc:.4f}  |  {len(X_filt)} samples across {n_classes} classes")
        ax.legend()
        plt.tight_layout()

    elif Z.shape[1] >= 2:
        # 2-D LDA -> scatter
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        # Sub-figure A: histogram on LD1
        ax = axes[0]
        for cid in range(n_classes):
            c_mask = y_contiguous == cid
            vals = Z[c_mask, 0]
            ax.hist(vals, bins=80, alpha=0.5,
                    label=f"{valid_names[cid]} (n={(c_mask).sum()})",
                    color=valid_colors[cid])
        ax.set_xlabel("LDA Component 1")
        ax.set_ylabel("Count")
        ax.set_title("LDA Component 1 histogram")
        ax.legend(fontsize=8)

        # Sub-figure B: LD1 vs LD2 scatter
        ax = axes[1]
        for cid in range(n_classes):
            c_mask = y_contiguous == cid
            ax.scatter(Z[c_mask, 0], Z[c_mask, 1],
                       alpha=0.3, s=8, c=valid_colors[cid],
                       label=f"{valid_names[cid]}")
        ax.set_xlabel("LDA Component 1")
        ax.set_ylabel("LDA Component 2")
        ax.set_title("LD1 vs LD2 scatter")
        ax.legend(fontsize=8, markerscale=2)

        fig.suptitle(f"V5 Multi-class LDA: YOLO features by sub-class\n"
                     f"Train accuracy: {acc:.4f}  |  {len(X_filt)} samples across {n_classes} classes",
                     fontsize=13, y=1.02)
        plt.tight_layout()

    out_path = OUT / "v5_lda_multiclass.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"\nWrote {out_path}")

    # Print var ratios
    print(f"LDA explained variance ratio per component: {lda.explained_variance_ratio_.round(4)}")
    print("\nDone.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Multi-class by-category LDA on the RGB model's features.")
    ap.add_argument("--include-video", action="store_true",
                    help="Also mine cross-domain RGB-video confusers (imgsz=640) on top of Svanström.")
    ap.add_argument("--cap", type=int, default=2500, help="Max detections kept per class.")
    ap.add_argument("--svan-stride", type=int, default=2, help="Frame stride when scanning Svanström sequences.")
    args = ap.parse_args()

    sns.set_theme(style="whitegrid")
    main(include_video=args.include_video, per_class_cap=args.cap, svan_stride=args.svan_stride)