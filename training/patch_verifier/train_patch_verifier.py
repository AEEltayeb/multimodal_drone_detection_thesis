"""
train_patch_verifier.py — Small CNN that predicts P(drone) from a crop.

Trains two models separately:
  * patch_verifier_rgb.pt  on RGB crops
  * patch_verifier_ir.pt   on IR crops

Input: manifest from extract_patches.py.
Backbone: MobileNetV3-Small (torchvision), final head replaced with a
binary sigmoid. ~2.5M params, <5ms per crop on the GTX 1050 Ti.

Downstream consumer: classifier/fusion_classifier.py will load both
models and inject rgb_patch_drone_prob / ir_patch_drone_prob as extra
features into the 4-class fusion classifier.
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
import cv2
from sklearn.metrics import roc_auc_score, confusion_matrix


class PatchDataset(Dataset):
    def __init__(self, rows, tfm):
        self.rows = rows.reset_index(drop=True)
        self.tfm = tfm

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows.iloc[idx]
        img = cv2.imread(r["path"], cv2.IMREAD_COLOR)
        if img is None:
            img = np.zeros((64, 64, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.tfm(img)
        y = 1.0 if r["label"] == "drone" else 0.0
        return img, torch.tensor(y, dtype=torch.float32), r["category"]


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
    cats = np.array(cats)
    for c in np.unique(cats):
        m = cats == c
        mean_prob = float(p_all[m].mean())
        per_cat[c] = {
            "n": int(m.sum()),
            "mean_drone_prob": mean_prob,
            "frac_as_drone": float((p_all[m] >= 0.5).mean()),
        }
    return {"auc": auc, "acc": acc, "cm": cm, "per_cat": per_cat}


def train_one_modality(manifest, modality, out_dir, epochs, batch_size, lr, device):
    print(f"\n{'='*60}\nTraining {modality.upper()} patch verifier\n{'='*60}")
    df = manifest[manifest["modality"] == modality].copy()
    print(f"  Rows: {len(df)}  videos: {df['video'].nunique()}")
    print(f"  By label: {dict(df['label'].value_counts())}")
    print(f"  By category: {dict(df['category'].value_counts())}")

    tr_df, va_df = sequence_split(df)
    print(f"  Train: {len(tr_df)} ({tr_df['video'].nunique()} vids)   "
          f"Val: {len(va_df)} ({va_df['video'].nunique()} vids)")

    train_tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    eval_tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    tr_loader = DataLoader(PatchDataset(tr_df, train_tfm),
                           batch_size=batch_size, shuffle=True,
                           num_workers=0, pin_memory=True, drop_last=True)
    va_loader = DataLoader(PatchDataset(va_df, eval_tfm),
                           batch_size=batch_size, shuffle=False,
                           num_workers=0, pin_memory=True)

    model = build_model(device)
    pos_w = (df["label"] == "aerial").sum() / max(1, (df["label"] == "drone").sum())
    pos_w_tensor = torch.tensor([pos_w], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor)
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
    for c, s in final_val["per_cat"].items():
        print(f"  {c:<12s} n={s['n']:4d}  mean_p={s['mean_drone_prob']:.3f}  "
              f"frac_as_drone={s['frac_as_drone']:.3f}")

    model_path = out_dir / f"patch_verifier_{modality}.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "arch": "mobilenet_v3_small",
        "modality": modality,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "input_size": 224,
    }, model_path)
    metrics_path = out_dir / f"patch_verifier_{modality}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "best_val_auc": best_auc,
            "final": final_val,
            "history": history,
            "n_train": len(tr_df),
            "n_val": len(va_df),
            "train_videos": int(tr_df["video"].nunique()),
            "val_videos": int(va_df["video"].nunique()),
        }, f, indent=2)
    print(f"Saved {model_path.name} + metrics.")
    return final_val


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", default="models/patches/manifest.csv")
    p.add_argument("--out", default="models/patches")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--modality", choices=["rgb", "ir", "both"], default="both")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    print(f"Manifest: {len(manifest)} rows")

    modalities = ["rgb", "ir"] if args.modality == "both" else [args.modality]
    for m in modalities:
        train_one_modality(manifest, m, out_dir, args.epochs,
                           args.batch_size, args.lr, device)


if __name__ == "__main__":
    main()
