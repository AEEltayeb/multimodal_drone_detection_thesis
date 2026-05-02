"""
calibrate_confuser_ood.py — Fit a Mahalanobis OOD detector for each
confuser filter (RGB + IR).

For each trained confuser filter we:
  1. Extract penultimate-layer features (1024-d, the input to the final
     Linear head) for every training crop in the manifest — both classes.
  2. Fit a Gaussian: mean + shrinkage-regularised inverse covariance.
  3. Pick a threshold at a configurable percentile of in-distribution
     Mahalanobis distances (default 99th).

At inference the verifier can now flag crops whose distance exceeds the
threshold as OOD — the fusion engine will treat those as "verifier has
no opinion" instead of letting a collapsed sigmoid trigger a veto.

Outputs sit next to the weights:
    confuser_filter_rgb_ood.npz
    confuser_filter_ir_ood.npz
with keys: mean, inv_cov, threshold, percentile, n, train_dists_summary.

Usage:
    python classifier/calibrate_confuser_ood.py
    python classifier/calibrate_confuser_ood.py --percentile 97.5
    python classifier/calibrate_confuser_ood.py --modality rgb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

SCRIPT_DIR = Path(__file__).resolve().parent
PATCH_DIR = SCRIPT_DIR / "runs" / "patches"
MANIFEST_PATH = PATCH_DIR / "manifest.csv"
PROJECT_ROOT = SCRIPT_DIR.parent


def load_backbone(weights_path: Path, device: torch.device):
    ckpt = torch.load(str(weights_path), map_location=device, weights_only=True)
    net = mobilenet_v3_small(weights=None)
    in_features = net.classifier[-1].in_features
    net.classifier[-1] = nn.Linear(in_features, 1)
    net.load_state_dict(ckpt["state_dict"])
    net.eval().to(device)
    input_size = int(ckpt.get("input_size", 224))
    mean = np.array(ckpt.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
    std = np.array(ckpt.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
    return net, input_size, mean, std


@torch.no_grad()
def penultimate_features(net: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    """Return the 1024-d feature vector fed into the final Linear head."""
    x = net.features(batch)
    x = net.avgpool(x)
    x = torch.flatten(x, 1)
    for layer in net.classifier[:-1]:
        x = layer(x)
    return x


def preprocess(img_bgr: np.ndarray, size: int, mean: np.ndarray,
               std: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    arr = rgb.astype(np.float32) / 255.0
    arr = (arr - mean) / std
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def extract_features(net, rows: pd.DataFrame, size: int, mean, std,
                     device, batch_size: int = 64) -> np.ndarray:
    feats = []
    batch = []
    total = len(rows)
    for i, (_, r) in enumerate(rows.iterrows()):
        path = PROJECT_ROOT / r["path"]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        batch.append(preprocess(img, size, mean, std))
        if len(batch) >= batch_size or i == total - 1:
            tensor = torch.stack(batch).to(device, non_blocking=True)
            f = penultimate_features(net, tensor).cpu().numpy()
            feats.append(f)
            batch = []
        if (i + 1) % 1000 == 0:
            print(f"    {i+1}/{total}")
    if batch:
        tensor = torch.stack(batch).to(device, non_blocking=True)
        feats.append(penultimate_features(net, tensor).cpu().numpy())
    return np.concatenate(feats, axis=0) if feats else np.zeros((0, 1024), np.float32)


def fit_gaussian(feats: np.ndarray, ridge: float = 1e-3):
    """Mean + shrinkage-regularised inverse covariance."""
    mu = feats.mean(axis=0)
    centered = feats - mu
    cov = (centered.T @ centered) / max(1, len(feats) - 1)
    cov += ridge * np.trace(cov) / cov.shape[0] * np.eye(cov.shape[0], dtype=cov.dtype)
    inv_cov = np.linalg.inv(cov)
    return mu.astype(np.float32), inv_cov.astype(np.float32)


def mahalanobis(feats: np.ndarray, mu: np.ndarray, inv_cov: np.ndarray) -> np.ndarray:
    d = feats - mu
    return np.sqrt(np.einsum("ni,ij,nj->n", d, inv_cov, d))


def calibrate_one(modality: str, weights_path: Path, manifest: pd.DataFrame,
                  percentile: float, device: torch.device) -> dict:
    rows = manifest[manifest["modality"] == modality].reset_index(drop=True)
    print(f"\n[{modality.upper()}] {len(rows)} crops — loading {weights_path.name}")

    net, size, mean, std = load_backbone(weights_path, device)
    feats = extract_features(net, rows, size, mean, std, device)
    print(f"  extracted {len(feats)} feature vectors ({feats.shape[1]}-d)")

    mu, inv_cov = fit_gaussian(feats)
    dists = mahalanobis(feats, mu, inv_cov)
    threshold = float(np.percentile(dists, percentile))

    summary = {
        "min": float(dists.min()),
        "p50": float(np.percentile(dists, 50)),
        "p95": float(np.percentile(dists, 95)),
        "p99": float(np.percentile(dists, 99)),
        "max": float(dists.max()),
    }
    print(f"  distance stats: {summary}")
    print(f"  threshold @p{percentile}: {threshold:.3f}")

    out_path = weights_path.with_name(weights_path.stem + "_ood.npz")
    np.savez(out_path,
             mean=mu, inv_cov=inv_cov,
             threshold=np.float32(threshold),
             percentile=np.float32(percentile),
             n=np.int64(len(feats)))
    print(f"  saved {out_path.name}")

    return {"modality": modality, "threshold": threshold,
            "percentile": percentile, "n": int(len(feats)),
            "dist_summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--percentile", type=float, default=99.0,
                    help="percentile of train distances to use as OOD threshold")
    ap.add_argument("--modality", choices=["rgb", "ir", "both"], default="both")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    manifest = pd.read_csv(MANIFEST_PATH)

    results = []
    for modality in ["rgb", "ir"]:
        if args.modality not in (modality, "both"):
            continue
        w = PATCH_DIR / f"confuser_filter_{modality}.pt"
        if not w.exists():
            print(f"[{modality}] {w} missing — skip")
            continue
        results.append(calibrate_one(modality, w, manifest, args.percentile, device))

    out_json = PATCH_DIR / "confuser_ood_calibration.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\nSummary written to {out_json}")


if __name__ == "__main__":
    main()
