"""Merge yt confuser rows into the Lean-13 dataset and retrain Lean-13 + Lean-10
on the augmented dataset (Lean-13_yt and Lean-10_yt).

Combined dataset := fusion_dataset_lean13.csv + fusion_dataset_lean13 (yt-only).
"""
import json, re
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parent.parent
LEAN13_CSV = REPO / "models/routers/lean13/fusion_dataset_lean13.csv"
YT_CSV = REPO / "models/routers/lean13_yt_only/fusion_dataset_lean13.csv"
COMBINED_CSV = REPO / "models/routers/lean13_yt/fusion_dataset_lean13_yt.csv"

FEATURES_13 = [
    "rgb_max_conf", "ir_max_conf",
    "rgb_best_log_bbox_area", "ir_best_log_bbox_area",
    "rgb_best_aspect_ratio", "ir_best_aspect_ratio",
    "rgb_best_pos_y", "ir_best_pos_y",
    "rgb_best_local_contrast", "ir_best_local_contrast",
    "rgb_img_mean", "ir_img_mean", "rgb_img_std",
]
FEATURES_10 = FEATURES_13[:10]
LABEL_NAMES = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}
SEQ_RE = re.compile(r"^(.+?)(?:_f\d+|_frame\d+|_\d{4,})(?:_visible|_infrared)?$", re.I)


def seq_id(stem, source):
    m = SEQ_RE.match(str(stem))
    base = m.group(1).rstrip("_") if m else str(stem)
    return f"{source}::{base}"


def train_and_save(df, features, out_dir, tag):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df["sequence_id"] = df.apply(lambda r: seq_id(r["stem"], r["source"]), axis=1)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(gss.split(df, df["trust_label"], groups=df["sequence_id"].values))
    Xtr, Xte = df.iloc[tr][features].values, df.iloc[te][features].values
    ytr, yte = df.iloc[tr]["trust_label"].values, df.iloc[te]["trust_label"].values
    src_te = df.iloc[te]["source"].values
    print(f"  [{tag}] features={len(features)}  train={len(Xtr):,}  test={len(Xte):,}")

    model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=4,
        eval_metric="mlogloss", tree_method="hist",
        random_state=42, n_jobs=-1,
    )
    model.fit(Xtr, ytr, verbose=False)
    pred = model.predict(Xte)
    acc = float(accuracy_score(yte, pred))
    f1m = float(f1_score(yte, pred, average="macro", zero_division=0))
    f1w = float(f1_score(yte, pred, average="weighted", zero_division=0))
    print(f"  [{tag}] acc={acc:.4f}  f1m={f1m:.4f}  f1w={f1w:.4f}")

    per = {}
    for ds in np.unique(src_te):
        m = src_te == ds
        per[ds] = {
            "n": int(m.sum()),
            "acc": float(accuracy_score(yte[m], pred[m])),
            "f1_macro": float(f1_score(yte[m], pred[m], average="macro", zero_division=0)),
        }

    imp = dict(zip(features, model.feature_importances_.tolist()))
    joblib.dump({"model": model, "features": features}, out_dir / "model.joblib")
    json.dump({
        "tag": tag, "n_features": len(features), "features": features,
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "accuracy": acc, "f1_macro": f1m, "f1_weighted": f1w,
        "per_dataset": per, "feature_importance": imp,
    }, open(out_dir / "metrics.json", "w"), indent=2)
    print(f"  [{tag}] saved -> {out_dir}\n")
    return acc, f1m, per, imp


def main():
    print(f"Loading {LEAN13_CSV.name} + {YT_CSV.parent.name}/{YT_CSV.name}")
    df13 = pd.read_csv(LEAN13_CSV)
    dfyt = pd.read_csv(YT_CSV)
    print(f"  lean13 rows: {len(df13):,}   yt rows: {len(dfyt):,}")
    df = pd.concat([df13, dfyt], ignore_index=True)
    print(f"  combined : {len(df):,}")
    COMBINED_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(COMBINED_CSV, index=False)
    print(f"  saved -> {COMBINED_CSV}\n")

    print("=== Retraining Lean-13 on combined dataset ===")
    train_and_save(df, FEATURES_13, REPO / "models/routers/lean13_yt", "lean13_yt")
    print("=== Retraining Lean-10 on combined dataset ===")
    train_and_save(df, FEATURES_10, REPO / "models/routers/lean10_yt", "lean10_yt")


if __name__ == "__main__":
    main()
