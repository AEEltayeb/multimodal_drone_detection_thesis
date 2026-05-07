"""
train_confuser_4class.py — 4-class confuser filter with an explicit
"other" class (softmax instead of sigmoid).

Classes:
    0: airplane
    1: helicopter
    2: bird
    3: other        ← drones + any additional diverse negatives

Reject rule at inference:
    argmax ∈ {airplane, helicopter, bird}  AND  softmax[argmax] ≥ threshold
Everything else (low confidence, or argmax == other) passes.

Why this beats the binary sigmoid:
  - Softmax over 4 classes can express "I'm 40% airplane, 60% other",
    so the network has a real "not a confuser" outcome.
  - Novel / OOD inputs drift toward "other" by default instead of
    collapsing onto a random side of a 1-D sigmoid.

Optional: pass --extra-negatives-dir DIR to mix in arbitrary diverse
crops (sky, clouds, ground, random COCO, etc.) as additional "other"
samples. Directory should contain images at any depth; sub-folder
per modality ("rgb"/"ir") is optional — unlabeled images are added
to both modalities.

Outputs:
    confuser_filter4_rgb.pt
    confuser_filter4_ir.pt
    confuser_filter4_{rgb,ir}_metrics.json

Usage:
    python classifier/train_confuser_4class.py
    python classifier/train_confuser_4class.py --epochs 15 --modality rgb
    python classifier/train_confuser_4class.py --extra-negatives-dir G:/drone/neg_crops
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

SCRIPT_DIR = Path(__file__).resolve().parent
PATCH_DIR = SCRIPT_DIR / "runs" / "patches"
MANIFEST_PATH = PATCH_DIR / "manifest.csv"
PROJECT_ROOT = SCRIPT_DIR.parent

CLASS_NAMES = ["airplane", "helicopter", "bird", "other"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
CONFUSER_CLASSES = {"airplane", "helicopter", "bird"}

# Manifest categories that count as "other"
# Everything not in CONFUSER_CLASSES maps to "other" (drone, etc.).


def manifest_category_to_class(cat: str) -> int:
    if cat in CONFUSER_CLASSES:
        return CLASS_TO_IDX[cat]
    return CLASS_TO_IDX["other"]


# ── DATASET ──────────────────────────────────────────────────────

class FourClassDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, tfm, modality_hint: str = ""):
        self.rows = rows.reset_index(drop=True)
        self.tfm = tfm
        self.modality_hint = modality_hint

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows.iloc[idx]
        p = r["path"]
        img_path = p if Path(p).is_absolute() else str(PROJECT_ROOT / p)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((64, 64, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.tfm(img)
        y = int(r["class_idx"])
        return img, torch.tensor(y, dtype=torch.long), r["category"]


# ── SPLIT ────────────────────────────────────────────────────────

def sequence_split(df: pd.DataFrame, test_frac=0.2, seed=42):
    rng = random.Random(seed)
    videos = sorted(df["video"].unique())
    rng.shuffle(videos)
    n_test = max(1, int(len(videos) * test_frac))
    test_vids = set(videos[:n_test])
    tr = df[~df["video"].isin(test_vids)].copy()
    va = df[df["video"].isin(test_vids)].copy()
    return tr, va


# ── EXTRA NEGATIVES ──────────────────────────────────────────────

def load_extra_negatives(root: Path, modality: str) -> pd.DataFrame:
    """Walk a directory for images and return a manifest-shaped frame.
    Sub-folder 'rgb' / 'ir' scopes by modality; otherwise images are
    used for both modalities."""
    if not root.exists():
        print(f"  extra-negatives dir not found: {root}")
        return pd.DataFrame(columns=["stem", "path", "modality", "label",
                                      "category", "video", "class_idx"])
    sub_scoped = root / modality
    search = sub_scoped if sub_scoped.exists() else root
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    rows = []
    for p in search.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            stem = f"neg_{modality}_{p.stem}"
            rows.append({
                "stem": stem,
                "path": str(p),
                "modality": modality,
                "label": "neg",
                "category": "other",
                "video": f"neg_{p.parent.name}",
                "class_idx": CLASS_TO_IDX["other"],
            })
    print(f"  loaded {len(rows)} extra negatives from {search}")
    return pd.DataFrame(rows)


# ── MODEL ────────────────────────────────────────────────────────

def build_model(device: torch.device, num_classes: int = 4):
    net = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    in_features = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(in_features, num_classes)
    return net.to(device)


# ── EVAL ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, preds, probs_all, cats = [], [], [], []
    for x, y, c in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds.append(probs.argmax(axis=1))
        probs_all.append(probs)
        ys.append(y.numpy())
        cats.extend(c)
    y_all = np.concatenate(ys)
    yhat = np.concatenate(preds)
    probs_all = np.concatenate(probs_all, axis=0)
    acc = float((yhat == y_all).mean())
    cm = confusion_matrix(y_all, yhat, labels=list(range(len(CLASS_NAMES)))).tolist()

    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        mask_gt = y_all == i
        mask_pred = yhat == i
        tp = int((mask_gt & mask_pred).sum())
        fn = int((mask_gt & ~mask_pred).sum())
        fp = int((~mask_gt & mask_pred).sum())
        n = int(mask_gt.sum())
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        per_class[name] = {"n": n, "TP": tp, "FP": fp, "FN": fn,
                            "P": round(prec, 4), "R": round(rec, 4)}

    # Reject rule sweep: veto if argmax ∈ confusers AND prob ≥ thr
    confuser_idx = [CLASS_TO_IDX[c] for c in CONFUSER_CLASSES]
    reject_metrics = {}
    for thr in [0.5, 0.6, 0.7, 0.8, 0.9]:
        argmax_is_conf = np.isin(yhat, confuser_idx)
        top_prob = probs_all[np.arange(len(yhat)), yhat]
        vetoed = argmax_is_conf & (top_prob >= thr)
        gt_is_conf = np.isin(y_all, confuser_idx)
        # "veto correct" = vetoed AND ground-truth is confuser
        tp = int((vetoed & gt_is_conf).sum())
        fp = int((vetoed & ~gt_is_conf).sum())
        fn = int((~vetoed & gt_is_conf).sum())
        tn = int((~vetoed & ~gt_is_conf).sum())
        reject_metrics[f"thr={thr}"] = {
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision_veto": round(tp / max(1, tp + fp), 4),
            "recall_veto": round(tp / max(1, tp + fn), 4),
            "pass_acc_on_drones": round(
                tn / max(1, ((y_all == CLASS_TO_IDX["other"]).sum())), 4),
        }

    return {"acc": acc, "cm": cm, "per_class": per_class,
            "reject_sweep": reject_metrics}


# ── TRAINING ─────────────────────────────────────────────────────

def train_one_modality(manifest, modality, out_dir, epochs, batch_size, lr,
                       device, extra_neg_dir: Path | None):
    print(f"\n{'='*60}")
    print(f"Training {modality.upper()} 4-CLASS CONFUSER FILTER")
    print(f"{'='*60}")

    df = manifest[manifest["modality"] == modality].copy()
    df["class_idx"] = df["category"].apply(manifest_category_to_class)

    if extra_neg_dir is not None:
        extra = load_extra_negatives(extra_neg_dir, modality)
        if len(extra):
            df = pd.concat([df, extra], ignore_index=True)

    print(f"  rows: {len(df)}  videos: {df['video'].nunique()}")
    for name in CLASS_NAMES:
        n = int((df["class_idx"] == CLASS_TO_IDX[name]).sum())
        print(f"    {name:<10s} {n}")

    tr_df, va_df = sequence_split(df)
    print(f"  train: {len(tr_df)}  val: {len(va_df)}")

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
        FourClassDataset(tr_df, train_tfm, modality),
        batch_size=batch_size, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True,
    )
    va_loader = DataLoader(
        FourClassDataset(va_df, eval_tfm, modality),
        batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    model = build_model(device, num_classes=len(CLASS_NAMES))

    # Inverse-frequency class weights
    counts = np.array([
        max(1, int((tr_df["class_idx"] == i).sum()))
        for i in range(len(CLASS_NAMES))
    ], dtype=np.float32)
    weights = (counts.sum() / (len(counts) * counts)).astype(np.float32)
    print(f"  class weights: {dict(zip(CLASS_NAMES, weights.round(3).tolist()))}")
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_acc = -1.0
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
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        sched.step()
        tr_loss = float(np.mean(losses))
        val = evaluate(model, va_loader, device)
        dt = time.time() - t0
        print(f"  ep {ep:2d}  loss={tr_loss:.4f}  val_acc={val['acc']:.4f}  ({dt:.1f}s)")
        history.append({"epoch": ep, "train_loss": tr_loss, "val_acc": val["acc"]})
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            best_state = {k: v.detach().cpu().clone()
                           for k, v in model.state_dict().items()}
            # Save checkpoint to disk immediately
            ckpt_path = out_dir / f"confuser_filter4_{modality}_ckpt.pt"
            torch.save({"state_dict": best_state, "epoch": ep,
                         "val_acc": best_acc}, ckpt_path)

    model.load_state_dict(best_state)
    final_val = evaluate(model, va_loader, device)
    print(f"\nBest val acc: {best_acc:.4f}")
    print("\nPer-class:")
    for name, s in final_val["per_class"].items():
        print(f"  {name:<10s} n={s['n']:5d}  P={s['P']:.3f}  R={s['R']:.3f}  "
              f"TP={s['TP']}  FP={s['FP']}  FN={s['FN']}")
    print("\nReject-rule sweep (veto if argmax in confusers & prob >= thr):")
    for k, s in final_val["reject_sweep"].items():
        print(f"  {k}  veto P={s['precision_veto']:.3f}  "
              f"veto R={s['recall_veto']:.3f}  "
              f"drones_passed={s['pass_acc_on_drones']:.3f}")

    model_path = out_dir / f"confuser_filter4_{modality}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "arch": "mobilenet_v3_small",
        "modality": modality,
        "mode": "confuser_filter_4class",
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "confuser_classes": sorted(CONFUSER_CLASSES),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": 224,
    }, model_path)

    metrics_path = out_dir / f"confuser_filter4_{modality}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "mode": "confuser_filter_4class",
            "class_names": CLASS_NAMES,
            "best_val_acc": best_acc,
            "final": final_val,
            "history": history,
            "n_train": len(tr_df),
            "n_val": len(va_df),
        }, f, indent=2)
    print(f"\nSaved {model_path.name} + metrics.")
    return final_val


# ── MAIN ─────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--modality", choices=["rgb", "ir", "both"], default="both")
    p.add_argument("--extra-negatives-dir", type=str, default=None,
                    help="Optional directory of diverse negative crops to "
                          "mix into the 'other' class.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"\nManifest: {len(manifest)} rows")
    print(manifest.groupby(["modality", "category"])
          .size().unstack(fill_value=0).to_string())

    extra_dir = Path(args.extra_negatives_dir) if args.extra_negatives_dir else None
    modalities = ["rgb", "ir"] if args.modality == "both" else [args.modality]
    for m in modalities:
        train_one_modality(
            manifest, m, PATCH_DIR, args.epochs,
            args.batch_size, args.lr, device, extra_dir,
        )

    print("\nDone. Next step: wire confuser_filter4_*.pt into PatchVerifier.")


if __name__ == "__main__":
    main()
