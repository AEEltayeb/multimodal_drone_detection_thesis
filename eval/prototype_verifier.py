"""
Inference wrapper for the generative drone-prototype verifier built by
eval/build_prototype_verifier.py.

Same interface as MLPv4Verifier in eval_v4_vs_patch.py:
    pv = PrototypeVerifier("eval/results/.../classifiers/prototype_v1.pt")
    drone_probs = pv.predict_drone_probs(features_np)  # (n, 517) raw features

Internally:
    1. Slice features to the top-K indices selected at build time.
    2. Standardize with the stored scaler.
    3. Compute Mahalanobis distance to drone prototype.
    4. Map distance to a [0, 1] score via score = sigmoid((tau - d) / scale).
       d == tau -> 0.5;  d < tau -> > 0.5 (drone-like);  d > tau -> < 0.5.

The harness applies a threshold at eval time, just like for the MLP.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class PrototypeVerifier:
    """Mahalanobis-distance verifier on top-K ANOVA-selected YOLO features."""

    def __init__(self, weights_path: str | Path, device: str = "cuda"):
        self.device = torch.device(
            device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
        )
        ckpt = torch.load(str(weights_path), map_location=self.device,
                          weights_only=True)

        if ckpt.get("verifier_kind") != "prototype_mahalanobis":
            raise ValueError(
                f"Expected verifier_kind='prototype_mahalanobis', got "
                f"{ckpt.get('verifier_kind')!r}. Use the right artifact "
                f"from eval/build_prototype_verifier.py.")

        self.input_dim = int(ckpt["input_dim"])
        self.top_indices = ckpt["top_indices"].to(self.device)
        self.scaler_mean = ckpt["scaler_mean"].to(self.device).float()
        self.scaler_scale = ckpt["scaler_scale"].to(self.device).float()
        self.mu_drone = ckpt["mu_drone"].to(self.device).float()
        self.sigma_inv_drone = ckpt["sigma_inv_drone"].to(self.device).float()
        self.tau = float(ckpt["tau"])
        self.scale = float(ckpt["scale"])
        self.feature_schema = ckpt.get("feature_schema", "unknown")
        # Surfacing for logging
        self.cv_f1 = -1.0  # not applicable; prototype has no CV F1
        self.threshold = 0.5

    @torch.no_grad()
    def predict_drone_probs(self, feats_np: np.ndarray) -> np.ndarray:
        """feats_np: (n, input_dim) raw V5 features.

        Returns an array of length n with values in (0, 1), where higher
        means closer to the drone prototype (= more drone-like).
        """
        if len(feats_np) == 0:
            return np.zeros(0, dtype=np.float32)

        x = torch.from_numpy(feats_np.astype(np.float32)).to(self.device)
        # Slice to top-K subspace
        x_sel = x[:, self.top_indices]
        # Standardize
        x_std = (x_sel - self.scaler_mean) / self.scaler_scale
        # Mahalanobis to drone prototype
        delta = x_std - self.mu_drone                       # (n, K)
        tmp = delta @ self.sigma_inv_drone                  # (n, K)
        d2 = (tmp * delta).sum(dim=1)                       # (n,)
        d2 = d2.clamp_min(0.0)
        d = d2.sqrt()                                       # (n,)
        # Map to [0, 1]: score = sigmoid((tau - d) / scale)
        # d == tau -> 0.5; d << tau -> ~1.0; d >> tau -> ~0.0
        score = torch.sigmoid((self.tau - d) / self.scale)
        return score.cpu().numpy().astype(np.float32)
