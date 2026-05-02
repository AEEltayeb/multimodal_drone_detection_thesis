"""
train_confuser_filter.py — Train a binary CNN that detects aerial confusers
(airplane, helicopter, bird) and passes everything else (including drones).

Key insight: Instead of learning "what drones look like" (which doesn't
generalize across datasets), learn "what confusers look like" (which
DOES generalize — birds/airplanes look similar regardless of sensor).

Architecture: MobileNetV3-Small, binary sigmoid.
  label=1 → confuser (airplane/helicopter/bird) → REJECT
  label=0 → pass (drone or anything else) → ACCEPT

Two models trained separately: RGB and IR.

Steps:
  1. (Optional) Extract Anti-UAV drone crops if needed
  2. Flip existing manifest labels: confuser vs pass
  3. Train RGB confuser filter
  4. Train IR confuser filter

Usage:
    python classifier/train_confuser_filter.py
    python classifier/train_confuser_filter.py --extract-antiuav
    python classifier/train_confuser_filter.py --epochs 15 --modality rgb
"""

import argparse
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

SCRIPT_DIR = Path(__file__).resolve().parent
PATCH_DIR = SCRIPT_DIR / "runs" / "patches"
MANIFEST_PATH = PATCH_DIR / "manifest.csv"
OUT_DIR = PATCH_DIR
PROJECT_ROOT = SCRIPT_DIR.parent  # es_drone_detection/
ANTIUAV_BASE = Path(r"G:\drone\Anti-UAV-RGBT_yolo_converted")

CONFUSER_CATEGORIES = {"airplane", "helicopter", "bird"}
MIN_CROP_SIZE = 16
PAD_FRAC = 0.5  # must match patch_verifier.py _crop_with_context default

SVANSTROM_ROOT = Path(r"G:\drone\svanstrom_paired")


def crop_with_context(img_bgr, x1, y1, x2, y2, pad_frac=PAD_FRAC, min_side=24):
    """Exact replica of PatchVerifier._crop_with_context for train/serve parity."""
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


# ── Anti-UAV EXTRACTION ─────────────────────────────────────────

def extract_antiuav_crops(max_per_split_mod=1500, sample_every=30, seed=42):
    """Extract drone crops from Anti-UAV YOLO dataset to balance negatives."""
    random.seed(seed)
    manifest = pd.read_csv(MANIFEST_PATH)
    existing_stems = set(manifest["stem"].values)
    print(f"Existing manifest: {len(manifest)} rows")

    new_rows = []
    for split in ["test", "val"]:
        for mod_key, mod_folder in [("rgb", "RGB"), ("ir", "IR")]:
            img_dir = ANTIUAV_BASE / split / mod_folder / "images"
            lbl_dir = ANTIUAV_BASE / split / mod_folder / "labels"
            if not img_dir.exists():
                print(f"  SKIP {split}/{mod_folder}: not found")
                continue

            img_files = sorted(
                list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
            )
            sampled = img_files[::sample_every]
            random.shuffle(sampled)
            print(f"  {split}/{mod_folder}: {len(img_files)} total, "
                  f"{len(sampled)} sampled (every {sample_every})")

            count = 0
            for img_path in sampled:
                if count >= max_per_split_mod:
                    break
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if not lbl_path.exists():
                    continue
                content = lbl_path.read_text().strip()
                if not content:
                    continue

                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]

                for li, line in enumerate(content.split("\n")):
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cx, cy, bw, bh = (
                        float(parts[1]), float(parts[2]),
                        float(parts[3]), float(parts[4]),
                    )
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                    # Use same crop_with_context as inference (square, 50% pad)
                    crop = crop_with_context(img, x1, y1, x2, y2)
                    if crop is None or crop.size == 0:
                        continue

                    stem = f"antiuav_{split}_{img_path.stem}_b{li}"
                    if stem in existing_stems:
                        continue

                    out_dir = PATCH_DIR / mod_key / "drone"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    out_path = out_dir / f"{stem}.jpg"
                    cv2.imwrite(str(out_path), crop,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])

                    seq_name = img_path.stem.rsplit("_", 1)[0]
                    new_rows.append({
                        "stem": stem,
                        "path": str(out_path.relative_to(PROJECT_ROOT)),
                        "modality": mod_key,
                        "label": "drone",
                        "category": "drone",
                        "video": f"antiuav_{split}_{seq_name}",
                    })
                    existing_stems.add(stem)
                    count += 1
                    break  # one box per frame
            print(f"    -> Extracted {count} crops")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        updated = pd.concat([manifest, new_df], ignore_index=True)
        updated.to_csv(MANIFEST_PATH, index=False)
        print(f"\nManifest updated: {len(manifest)} -> {len(updated)} rows")
    else:
        updated = manifest
        print("\nNo new crops extracted")

    print("\n=== Distribution after extraction ===")
    print(updated.groupby(["modality", "category"])
          .size().unstack(fill_value=0).to_string())
    return updated


def re_extract_all_crops():
    """Re-extract ALL crops in manifest from original frames using
    crop_with_context (matching inference). Skips entries where original
    frame cannot be found."""
    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"\n{'='*60}")
    print(f"  Re-extracting all crops with inference-matched padding")
    print(f"  (pad_frac={PAD_FRAC}, square crops)")
    print(f"{'='*60}")
    print(f"  Manifest: {len(manifest)} rows")

    # Build mapping: stem -> original frame path + label path
    updated = 0
    failed = 0
    skipped = 0

    for i, row in manifest.iterrows():
        stem = row["stem"]
        mod = row["modality"]
        category = row["category"]
        crop_path = PROJECT_ROOT / row["path"]

        # Svanstrom crops were already made with pad_frac=0.5 via
        # extract_patches.py:crop_with_context — they already match inference.
        # Only antiuav_* crops used the old 20% rect pad. Skip the rest.
        if not stem.startswith("antiuav_"):
            skipped += 1
            continue

        # Determine original frame + label path
        frame_path, label_path = _find_original_frame(stem, mod, category)

        if frame_path is None or not frame_path.exists():
            skipped += 1
            continue

        # Load original frame
        img = cv2.imread(str(frame_path))
        if img is None:
            failed += 1
            continue
        h, w = img.shape[:2]

        # Find GT box from label file
        box = _get_box_from_label(label_path, stem, w, h)
        if box is None:
            skipped += 1
            continue

        x1, y1, x2, y2 = box
        crop = crop_with_context(img, x1, y1, x2, y2)
        if crop is None or crop.size == 0:
            failed += 1
            continue

        # Overwrite existing crop
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        updated += 1

        if (i + 1) % 2000 == 0:
            print(f"    {i+1}/{len(manifest)} processed "
                  f"(updated={updated}, skipped={skipped}, failed={failed})")

    print(f"\n  Done: updated={updated}, skipped={skipped}, failed={failed}")


def _find_original_frame(stem, mod, category):
    """Map a manifest stem back to its original frame + label file."""
    # Anti-UAV entries: antiuav_{split}_{seq_name}_f{frame}_b{box}
    if stem.startswith("antiuav_"):
        parts = stem.split("_")
        split = parts[1]  # test or val
        # Reconstruct image filename
        # stem: antiuav_test_20190925_111757_1_8_visible_f000182_b0
        # image name: 20190925_111757_1_8_visible_f000182.jpg or .png
        idx_f = stem.rfind("_f")
        idx_b = stem.rfind("_b")
        img_stem = stem[len(f"antiuav_{split}_"):idx_b]  # without antiuav_split_ and _bN
        mod_folder = "RGB" if "visible" in stem else "IR"
        img_dir = ANTIUAV_BASE / split / mod_folder / "images"
        lbl_dir = ANTIUAV_BASE / split / mod_folder / "labels"
        for ext in [".jpg", ".png"]:
            p = img_dir / (img_stem + ext)
            if p.exists():
                return p, lbl_dir / (img_stem + ".txt")
        return None, None

    # Svanström entries: {mod}_{CATEGORY}_{NNN}_f{frame}_b{box}
    # IR crops: IR_DRONE_001_f000000_b0 → IR_DRONE_001_f000000_infrared.jpg
    # RGB crops: V_DRONE_001_f000000_b0 → IR_DRONE_001_f000000_visible.jpg
    #   (V_ is short for "visible" — the actual file uses IR_ prefix always)
    idx_f = stem.rfind("_f")
    idx_b = stem.rfind("_b")
    if idx_f < 0 or idx_b < 0:
        return None, None

    # video prefix: everything before _f{frame}
    video_prefix = stem[:idx_f]  # e.g. V_DRONE_001 or IR_DRONE_001
    frame_part = stem[idx_f+1:idx_b]  # e.g. f000000

    # V_ prefix → IR_ (the files in svanstrom_paired all use IR_ prefix)
    if video_prefix.startswith("V_"):
        video_prefix = "IR_" + video_prefix[2:]

    # Determine Svanström modality folder from the manifest modality
    if mod == "ir":
        svan_mod = "IR"
        suffix = "_infrared"
    else:
        svan_mod = "RGB"
        suffix = "_visible"

    img_stem_full = f"{video_prefix}_{frame_part}{suffix}"
    img_dir = SVANSTROM_ROOT / svan_mod / "images"
    lbl_dir = SVANSTROM_ROOT / svan_mod / "labels"

    for ext in [".jpg", ".png"]:
        p = img_dir / (img_stem_full + ext)
        if p.exists():
            return p, lbl_dir / (img_stem_full + ".txt")
    return None, None


def _get_box_from_label(label_path, stem, img_w, img_h):
    """Read the GT box corresponding to this stem's box index."""
    if label_path is None or not label_path.exists():
        return None

    # Extract box index from stem (last _bN)
    idx_b = stem.rfind("_b")
    box_idx = int(stem[idx_b+2:]) if idx_b >= 0 else 0

    content = label_path.read_text().strip()
    if not content:
        return None

    lines = content.split("\n")
    if box_idx >= len(lines):
        box_idx = 0

    parts = lines[box_idx].strip().split()
    if len(parts) < 5:
        return None

    cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
    x1 = int((cx - bw / 2) * img_w)
    y1 = int((cy - bh / 2) * img_h)
    x2 = int((cx + bw / 2) * img_w)
    y2 = int((cy + bh / 2) * img_h)
    return (x1, y1, x2, y2)


# ── DATASET ──────────────────────────────────────────────────────

class ConfuserDataset(Dataset):
    """Binary dataset: confuser (1) vs pass (0)."""

    def __init__(self, rows, tfm):
        self.rows = rows.reset_index(drop=True)
        self.tfm = tfm

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows.iloc[idx]
        # Resolve path relative to project root
        img_path = str(PROJECT_ROOT / r["path"])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((64, 64, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.tfm(img)
        # INVERTED: confuser=1, drone/pass=0
        y = 1.0 if r["category"] in CONFUSER_CATEGORIES else 0.0
        return img, torch.tensor(y, dtype=torch.float32), r["category"]


# ── MODEL ────────────────────────────────────────────────────────

def build_model(device):
    net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    in_features = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(in_features, 1)
    return net.to(device)


def sequence_split(df, test_frac=0.2, seed=42):
    rng = random.Random(seed)
    videos = sorted(df["video"].unique())
    rng.shuffle(videos)
    n_test = max(1, int(len(videos) * test_frac))
    test_vids = set(videos[:n_test])
    tr = df[~df["video"].isin(test_vids)].copy()
    va = df[df["video"].isin(test_vids)].copy()
    return tr, va


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps, cats = [], [], []
    for x, y, c in loader:
        x = x.to(device, non_blocking=True)
        logit = model(x).squeeze(1)
        p = torch.sigmoid(logit).cpu().numpy()
        ys.append(y.numpy())
        ps.append(p)
        cats.extend(c)
    y_all = np.concatenate(ys)
    p_all = np.concatenate(ps)
    yhat = (p_all >= 0.5).astype(int)
    auc = float(roc_auc_score(y_all, p_all)) if len(set(y_all.tolist())) > 1 else float("nan")
    acc = float((yhat == y_all).mean())
    cm = confusion_matrix(y_all, yhat, labels=[0, 1]).tolist()

    per_cat = {}
    cats_arr = np.array(cats)
    for c in np.unique(cats_arr):
        m = cats_arr == c
        mean_prob = float(p_all[m].mean())
        per_cat[c] = {
            "n": int(m.sum()),
            "mean_confuser_prob": round(mean_prob, 4),
            "frac_flagged_confuser": round(float((p_all[m] >= 0.5).mean()), 4),
        }
    return {"auc": auc, "acc": acc, "cm": cm, "per_cat": per_cat}


# ── TRAINING ─────────────────────────────────────────────────────

def train_one_modality(manifest, modality, out_dir, epochs, batch_size, lr, device):
    print(f"\n{'='*60}")
    print(f"Training {modality.upper()} CONFUSER FILTER")
    print(f"{'='*60}")

    df = manifest[manifest["modality"] == modality].copy()

    # Set binary label: confuser=1, drone/pass=0
    df["is_confuser"] = df["category"].isin(CONFUSER_CATEGORIES).astype(int)
    n_conf = df["is_confuser"].sum()
    n_pass = len(df) - n_conf

    print(f"  Rows: {len(df)}  videos: {df['video'].nunique()}")
    print(f"  Confuser (reject): {n_conf}  ({n_conf/len(df)*100:.1f}%)")
    print(f"  Pass (drone/other): {n_pass}  ({n_pass/len(df)*100:.1f}%)")
    print(f"  By category: {dict(df['category'].value_counts())}")

    tr_df, va_df = sequence_split(df)
    print(f"  Train: {len(tr_df)} ({tr_df['video'].nunique()} vids)   "
          f"Val: {len(va_df)} ({va_df['video'].nunique()} vids)")

    # Transforms: match inference pipeline (direct resize to 224, no CenterCrop)
    train_tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    tr_loader = DataLoader(
        ConfuserDataset(tr_df, train_tfm),
        batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True,
    )
    va_loader = DataLoader(
        ConfuserDataset(va_df, eval_tfm),
        batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True,
    )

    model = build_model(device)

    # Balanced pos_weight: ratio of pass / confuser
    pos_w = n_pass / max(1, n_conf)
    print(f"  pos_weight: {pos_w:.3f}")
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w], device=device)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_auc = -1.0
    best_state = None
    history = []

    for ep in range(1, epochs + 1):
        model.train()
        losses = []
        t0 = time.time()
        for x, y, _ in tr_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            logit = model(x).squeeze(1)
            loss = criterion(logit, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sched.step()
        tr_loss = float(np.mean(losses))
        val = evaluate(model, va_loader, device)
        dt = time.time() - t0
        print(f"  ep {ep:2d}  loss={tr_loss:.4f}  val_auc={val['auc']:.4f}  "
              f"val_acc={val['acc']:.4f}  ({dt:.1f}s)")
        history.append({"epoch": ep, "train_loss": tr_loss, **val})
        if val["auc"] > best_auc:
            best_auc = val["auc"]
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    final_val = evaluate(model, va_loader, device)
    print(f"\nBest val AUC: {best_auc:.4f}")
    print(f"Per-category (val):")
    for c, s in sorted(final_val["per_cat"].items()):
        expected = "REJECT" if c in CONFUSER_CATEGORIES else "pass"
        print(f"  {c:<12s} n={s['n']:4d}  "
              f"mean_P(confuser)={s['mean_confuser_prob']:.3f}  "
              f"frac_flagged={s['frac_flagged_confuser']:.3f}  "
              f"[expect: {expected}]")

    # Save model — IMPORTANT: save with "confuser_filter" name
    model_path = out_dir / f"confuser_filter_{modality}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "arch": "mobilenet_v3_small",
        "modality": modality,
        "mode": "confuser_filter",  # distinguish from old patch_verifier
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": 224,
    }, model_path)

    metrics_path = out_dir / f"confuser_filter_{modality}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "mode": "confuser_filter",
            "best_val_auc": best_auc,
            "final": final_val,
            "history": history,
            "n_train": len(tr_df),
            "n_val": len(va_df),
            "n_confuser": int(n_conf),
            "n_pass": int(n_pass),
            "confuser_categories": sorted(CONFUSER_CATEGORIES),
        }, f, indent=2)
    print(f"Saved {model_path.name} + metrics.")
    return final_val


# ── MAIN ─────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Train confuser filter (airplane/helicopter/bird detector)"
    )
    p.add_argument("--extract-antiuav", action="store_true",
                    help="Extract Anti-UAV drone crops first")
    p.add_argument("--max-antiuav", type=int, default=1500,
                    help="Max crops per split/modality from Anti-UAV")
    p.add_argument("--sample-every", type=int, default=30,
                    help="Sample every Nth Anti-UAV frame")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--modality", choices=["rgb", "ir", "both"], default="both")
    p.add_argument("--re-extract", action="store_true",
                    help="Re-extract ALL crops using inference-matched padding")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Step 0: optionally re-extract all crops with inference-matched padding
    if args.re_extract:
        re_extract_all_crops()

    # Step 1: optionally extract Anti-UAV crops
    if args.extract_antiuav:
        if not ANTIUAV_BASE.exists():
            print(f"[ERROR] Anti-UAV not found at {ANTIUAV_BASE}")
            return
        manifest = extract_antiuav_crops(
            max_per_split_mod=args.max_antiuav,
            sample_every=args.sample_every,
        )
    else:
        manifest = pd.read_csv(MANIFEST_PATH)

    print(f"\nManifest: {len(manifest)} rows")
    print(manifest.groupby(["modality", "category"])
          .size().unstack(fill_value=0).to_string())

    # Step 2: train
    modalities = ["rgb", "ir"] if args.modality == "both" else [args.modality]
    for m in modalities:
        train_one_modality(
            manifest, m, OUT_DIR, args.epochs,
            args.batch_size, args.lr, device,
        )

    print("\nDone! Models saved as confuser_filter_rgb.pt / confuser_filter_ir.pt")
    print("Integration: veto when P(confuser) > threshold instead of P(drone) < threshold")


if __name__ == "__main__":
    main()
