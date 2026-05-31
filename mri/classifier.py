"""
mri.classifier — confuser-vs-drone classifiers on the extracted feature space.

The production candidate is the focal-loss MLP with BatchNorm and per-sample
weights (the V5 recipe). LogReg / RF / XGB are kept as a comparison bench. All
follow the sklearn fit/predict/predict_proba interface and accept optional
per-sample weights, so cross_val_score_f1 treats them uniformly.

Lifted from eval/distill_v5_p3p5_ft4.py; the saved .pt artifact keeps the
mlp_v5.pt checkpoint schema so eval/eval_v4_vs_patch.py --mlp-weights loads it.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

SEED = 42


# ── sklearn-style baselines ─────────────────────────────────────────────────

class LogRegWrapper:
    def __init__(self, C=1.0, class_weight="balanced", max_iter=2000, seed=SEED):
        self.C, self.class_weight_val = C, class_weight
        self.max_iter, self.seed, self.model = max_iter, seed, None

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
    def __init__(self, n_estimators=150, max_depth=8, class_weight="balanced", seed=SEED):
        self.n_estimators, self.max_depth = n_estimators, max_depth
        self.class_weight_val, self.seed, self.model = class_weight, seed, None

    def fit(self, X, y, sample_weight=None):
        from sklearn.ensemble import RandomForestClassifier
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            class_weight=self.class_weight_val, random_state=self.seed, n_jobs=-1)
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


class XGBWrapper:
    def __init__(self, n_estimators=150, max_depth=5, learning_rate=0.1, seed=SEED):
        self.n_estimators, self.max_depth = n_estimators, max_depth
        self.learning_rate, self.seed, self.model = learning_rate, seed, None

    def fit(self, X, y, sample_weight=None):
        from xgboost import XGBClassifier
        pos_w = (y == 0).sum() / max((y == 1).sum(), 1)
        self.model = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, scale_pos_weight=pos_w,
            random_state=self.seed, n_jobs=-1, verbosity=0, use_label_encoder=False)
        self.model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)


# ── Focal-loss MLP (production candidate) ───────────────────────────────────

class FocalLoss(nn.Module):
    """Binary focal loss with label smoothing and optional per-sample weights."""

    def __init__(self, alpha=0.75, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha, self.gamma, self.eps = alpha, gamma, label_smoothing

    def forward(self, logits, targets, sample_weights=None):
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
    """Focal-loss MLP with BatchNorm, dropout, cosine LR, per-sample weights."""

    def __init__(self, input_dim, hidden_dims=(512, 256, 128, 64),
                 lr=1e-3, epochs=120, batch_size=128, device="cuda",
                 dropout=0.3, focal_alpha=0.75, focal_gamma=2.0,
                 label_smoothing=0.1, use_batchnorm=True):
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.lr, self.epochs, self.batch_size = lr, epochs, batch_size
        self.device = device if torch.cuda.is_available() or device == "cpu" else "cpu"
        self.dropout = dropout
        self.focal_alpha, self.focal_gamma = focal_alpha, focal_gamma
        self.label_smoothing, self.use_batchnorm = label_smoothing, use_batchnorm
        self.net = None
        self.scaler = None
        self.history = []

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
        sw_t = (torch.from_numpy(np.asarray(sample_weight, dtype=np.float32))
                .to(self.device).unsqueeze(1)) if sample_weight is not None else None

        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.epochs, eta_min=self.lr * 0.01)
        criterion = FocalLoss(self.focal_alpha, self.focal_gamma, self.label_smoothing)

        n = len(X_t)
        for _ep in range(self.epochs):
            self.net.train()
            perm = torch.randperm(n, device=self.device)
            losses = []
            for start in range(0, n, self.batch_size):
                idx = perm[start:start + self.batch_size]
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
        return self

    @torch.no_grad()
    def _forward_eval(self, X):
        Xs = self.scaler.transform(X).astype(np.float32)
        X_t = torch.from_numpy(Xs).to(self.device)
        self.net.eval()
        logit = self.net(X_t).squeeze(1)
        return torch.sigmoid(logit).cpu().numpy()

    def predict(self, X):
        return (self._forward_eval(X) >= 0.5).astype(int)

    def predict_proba(self, X):
        p = self._forward_eval(X)
        return np.stack([1 - p, p], axis=1)


# ── Cross-validation ────────────────────────────────────────────────────────

def cross_val_score_f1(clf_class, clf_kwargs, X, y, sample_weight=None,
                       folds=5, seed=SEED):
    """Stratified CV. Returns (mean_f1, std_f1, best_model, oof_proba).

    oof_proba is the out-of-fold P(drone) for every sample — used by the
    diagnosis layer to estimate held-out FP reduction without a separate split.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    f1s, models = [], []
    oof = np.full(len(y), np.nan, dtype=np.float32)
    for tr_idx, va_idx in skf.split(X, y):
        m = clf_class(**clf_kwargs)
        if sample_weight is not None:
            m.fit(X[tr_idx], y[tr_idx], sample_weight=sample_weight[tr_idx])
        else:
            m.fit(X[tr_idx], y[tr_idx])
        oof[va_idx] = m.predict_proba(X[va_idx])[:, 1]
        f1s.append(f1_score(y[va_idx], m.predict(X[va_idx]), zero_division=0))
        models.append(m)
    return (float(np.mean(f1s)), float(np.std(f1s)),
            models[int(np.argmax(f1s))], oof)


def save_mlp_artifact(mlp: MLPWrapper, out_path, schema, cv_f1, cv_std,
                      detector_path, threshold=0.5):
    """Persist a callable MLP checkpoint in the mlp_v5.pt schema."""
    torch.save({
        "state_dict": mlp.net.state_dict(),
        "scaler_mean": torch.from_numpy(mlp.scaler.mean_.astype(np.float32)),
        "scaler_scale": torch.from_numpy(mlp.scaler.scale_.astype(np.float32)),
        "input_dim": int(mlp.input_dim),
        "hidden_dims": list(mlp.hidden_dims),
        "threshold": threshold,
        "cv_f1": float(cv_f1),
        "cv_std": float(cv_std),
        "feature_schema": schema.to_dict(),
        "metadata_order": ["conf", "log_area", "aspect", "rel_cx", "rel_cy"],
        "layers": list(schema.layers),
        "grids": {k: list(v) for k, v in schema.grids.items()},
        "use_batchnorm": mlp.use_batchnorm,
        "dropout": mlp.dropout,
        "base_detector": str(detector_path),
    }, out_path)
