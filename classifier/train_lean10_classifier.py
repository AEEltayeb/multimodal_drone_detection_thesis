"""
train_lean10_classifier.py - Lean-10 ablation of Lean-13.

Drops the 3 brightness/std scalars (rgb_img_mean, ir_img_mean, rgb_img_std)
that hypothesis says were acting as per-clip scene fingerprints during
sequence-split training, causing the model to memorize clip identity
rather than generalize on held-out drone+bird scenes.

Same training data + XGBoost config as train_lean13_classifier.py.
"""
import argparse, json, re
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

FEATURE_COLS = [
    "rgb_max_conf", "ir_max_conf",
    "rgb_best_log_bbox_area", "ir_best_log_bbox_area",
    "rgb_best_aspect_ratio", "ir_best_aspect_ratio",
    "rgb_best_pos_y", "ir_best_pos_y",
    "rgb_best_local_contrast", "ir_best_local_contrast",
]
LABEL_NAMES = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}
SEQ_RE = re.compile(r"^(.+?)(?:_f\d+|_frame\d+|_\d{4,})(?:_visible|_infrared)?$", re.I)

def extract_sequence_id(stem, source):
    m = SEQ_RE.match(stem)
    base = m.group(1).rstrip("_") if m else stem
    return f"{source}::{base}"

def sequence_split(df, test_size=0.25, seed=42):
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    return next(gss.split(df, df["trust_label"], groups=df["sequence_id"].values))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="classifier/fusion_models/lean13/fusion_dataset_lean13.csv")
    ap.add_argument("--output-dir", default="classifier/fusion_models/lean10")
    args = ap.parse_args()
    repo = Path(__file__).resolve().parent.parent
    csv_path = (repo / args.csv) if not Path(args.csv).is_absolute() else Path(args.csv)
    out_dir = (repo / args.output_dir) if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} rows")
    df["sequence_id"] = df.apply(lambda r: extract_sequence_id(str(r["stem"]), str(r["source"])), axis=1)

    train_idx, test_idx = sequence_split(df)
    X_train = df.iloc[train_idx][FEATURE_COLS].values
    X_test = df.iloc[test_idx][FEATURE_COLS].values
    y_train = df.iloc[train_idx]["trust_label"].values
    y_test = df.iloc[test_idx]["trust_label"].values
    src_test = df.iloc[test_idx]["source"].values
    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}  Features: {len(FEATURE_COLS)}")

    print("Training XGBoost (400 trees, depth 6, lr 0.05)...")
    model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=4,
        eval_metric="mlogloss", tree_method="hist",
        random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train, verbose=False)
    pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, pred))
    f1m = float(f1_score(y_test, pred, average="macro", zero_division=0))
    f1w = float(f1_score(y_test, pred, average="weighted", zero_division=0))
    print(f"\n  acc={acc:.4f}  f1m={f1m:.4f}  f1w={f1w:.4f}\n")
    print(classification_report(y_test, pred,
        target_names=[LABEL_NAMES[i] for i in range(4)], zero_division=0))

    per = {}
    print("  Per-source:")
    for ds in np.unique(src_test):
        m = src_test == ds
        a = float(accuracy_score(y_test[m], pred[m]))
        fm = float(f1_score(y_test[m], pred[m], average="macro", zero_division=0))
        per[ds] = {"n": int(m.sum()), "acc": a, "f1_macro": fm}
        print(f"    {ds:<60s} n={m.sum():>4d}  acc={a:.4f}  f1m={fm:.4f}")

    imp = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    print("\n  Feature importance:")
    for f, i in sorted(imp.items(), key=lambda x: x[1], reverse=True):
        print(f"    {f:<32s} {i:.4f}  {'#'*int(i*50)}")

    joblib.dump({"model": model, "features": FEATURE_COLS}, out_dir / "model.joblib")
    json.dump({
        "tag": "fusion_lean10", "n_features": len(FEATURE_COLS),
        "features": FEATURE_COLS, "n_train": int(len(X_train)),
        "n_test": int(len(X_test)), "accuracy": acc, "f1_macro": f1m,
        "f1_weighted": f1w, "per_dataset": per, "feature_importance": imp,
    }, open(out_dir / "metrics.json", "w"), indent=2)
    print(f"\n  Saved: {out_dir / 'model.joblib'}")
    print(f"  Saved: {out_dir / 'metrics.json'}")

if __name__ == "__main__":
    main()
