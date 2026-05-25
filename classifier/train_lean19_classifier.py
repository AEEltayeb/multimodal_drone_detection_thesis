"""train_lean19_classifier.py — 19-feature XGBoost (Lean-13 + 6 geometry).

Same XGBoost config as train_lean13_classifier.py for apples-to-apples.
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
    "rgb_img_mean", "ir_img_mean", "rgb_img_std",
    "rgb_best_pos_x", "ir_best_pos_x",
    "rgb_best_dist_to_center", "ir_best_dist_to_center",
    "rgb_best_target_bg_delta", "ir_best_target_bg_delta",
]
LABEL_NAMES = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}
SEQ_RE = re.compile(r"^(.+?)(?:_f\d+|_frame\d+|_\d{4,})(?:_visible|_infrared)?$", re.I)

def seq_id(stem, source):
    m = SEQ_RE.match(str(stem))
    base = m.group(1).rstrip("_") if m else str(stem)
    return f"{source}::{base}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="classifier/fusion_models/lean19/fusion_dataset_lean19.csv")
    ap.add_argument("--output-dir", default="classifier/fusion_models/lean19")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    csv_p = (repo / args.csv) if not Path(args.csv).is_absolute() else Path(args.csv)
    out_dir = (repo / args.output_dir) if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading: {csv_p}")
    df = pd.read_csv(csv_p)
    print(f"  {len(df):,} rows  features={len(FEATURE_COLS)}")
    df["sequence_id"] = df.apply(lambda r: seq_id(r["stem"], r["source"]), axis=1)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(gss.split(df, df["trust_label"], groups=df["sequence_id"].values))
    Xtr = df.iloc[tr][FEATURE_COLS].values; Xte = df.iloc[te][FEATURE_COLS].values
    ytr = df.iloc[tr]["trust_label"].values; yte = df.iloc[te]["trust_label"].values
    src_te = df.iloc[te]["source"].values
    print(f"  Train: {len(Xtr):,}  Test: {len(Xte):,}")

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
    print(f"\n  acc={acc:.4f}  f1m={f1m:.4f}  f1w={f1w:.4f}\n")
    print(classification_report(yte, pred,
        target_names=[LABEL_NAMES[i] for i in range(4)], zero_division=0))

    per = {}
    for ds in np.unique(src_te):
        m = src_te == ds
        per[ds] = {"n": int(m.sum()),
                   "acc": float(accuracy_score(yte[m], pred[m])),
                   "f1_macro": float(f1_score(yte[m], pred[m], average="macro", zero_division=0))}
        print(f"    {ds:<60s} n={m.sum():>4d}  acc={per[ds]['acc']:.4f}  f1m={per[ds]['f1_macro']:.4f}")

    imp = dict(zip(FEATURE_COLS, model.feature_importances_.tolist()))
    print("\n  Feature importance:")
    for f, i in sorted(imp.items(), key=lambda x: x[1], reverse=True):
        print(f"    {f:<32s} {i:.4f}  {'#'*int(i*50)}")

    joblib.dump({"model": model, "features": FEATURE_COLS}, out_dir / "model.joblib")
    json.dump({
        "tag": "fusion_lean19", "n_features": len(FEATURE_COLS),
        "features": FEATURE_COLS, "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "accuracy": acc, "f1_macro": f1m, "f1_weighted": f1w,
        "per_dataset": per, "feature_importance": imp,
    }, open(out_dir / "metrics.json", "w"), indent=2)
    print(f"\n  Saved: {out_dir/'model.joblib'}\n  Saved: {out_dir/'metrics.json'}")


if __name__ == "__main__":
    main()
