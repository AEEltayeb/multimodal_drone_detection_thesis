"""One-shot: verify mlp_v4.pt loads with weights_only=True."""
import torch
from pathlib import Path

ckpt = torch.load(
    "eval/results/_v4_p3p5_ft4_distill/classifiers/mlp_v4.pt",
    map_location="cpu",
    weights_only=True,
)
print("weights_only=True load: OK")
print(f"  input_dim   = {ckpt['input_dim']}")
print(f"  hidden_dims = {ckpt['hidden_dims']}")
print(f"  cv_f1       = {ckpt['cv_f1']:.4f} +/- {ckpt['cv_std']:.4f}")
print(f"  scaler_mean shape  = {tuple(ckpt['scaler_mean'].shape)} dtype={ckpt['scaler_mean'].dtype}")
print(f"  scaler_scale shape = {tuple(ckpt['scaler_scale'].shape)} dtype={ckpt['scaler_scale'].dtype}")
print(f"  schema      = {ckpt['feature_schema']}")
print(f"  base        = {Path(ckpt['base_detector']).name}")
