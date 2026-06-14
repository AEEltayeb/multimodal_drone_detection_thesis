"""train_ablations.py - Train 4 ablation variants of lean19_v2.

Variants:
  A   : original lean19 CSV (19 feat), class weights down-weighting class-3
  B   : strict-label CSV (19 feat), default weights
  C   : xmodal CSV (22 feat), default weights
  ABC : strict+xmodal CSV (22 feat), class weights

All share the same XGBoost config (400 trees, depth 6, lr 0.05) and the same
GroupShuffleSplit (75/25 seed=42) on sequence_id derived from (source, stem).
"""
import argparse, json, re
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parent.parent

LEAN19_FEATS = [
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
XMODAL_FEATS = ["xmodal_centroid_dist", "xmodal_scale_ratio", "xmodal_conf_ratio"]
ALL_BC = LEAN19_FEATS + XMODAL_FEATS

LABEL_NAMES = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}
SEQ_RE = re.compile(r"^(.+?)(?:_f\d+|_frame\d+|_\d{4,})(?:_visible|_infrared)?$", re.I)


def seq_id(stem, src):
    m = SEQ_RE.match(str(stem))
    base = m.group(1).rstrip("_") if m else str(stem)
    return f"{src}::{base}"


def train_variant(csv_path, feats, out_dir, tag, use_class_weights):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n=== {tag} ===\n  csv: {csv_path.relative_to(REPO) if csv_path.is_absolute() else csv_path}")
    df = pd.read_csv(csv_path)
    miss = [c for c in feats if c not in df.columns]
    if miss: raise SystemExit(f"  missing cols: {miss}")
    print(f"  rows: {len(df):,}  features: {len(feats)}")
    print(f"  label dist: {dict(df['trust_label'].value_counts().sort_index())}")
    df["sequence_id"] = df.apply(lambda r: seq_id(r["stem"], r["source"]), axis=1)
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=42)
                   .split(df, df["trust_label"], groups=df["sequence_id"].values))
    X_tr = df.iloc[tr][feats].values; X_te = df.iloc[te][feats].values
    y_tr = df.iloc[tr]["trust_label"].values; y_te = df.iloc[te]["trust_label"].values
    src_te = df.iloc[te]["source"].values
    print(f"  train: {len(X_tr):,}  test: {len(X_te):,}")

    sw = compute_sample_weight("balanced", y_tr) if use_class_weights else None
    if use_class_weights:
        # show effective class weights
        print(f"  class weights (balanced): "
              + ", ".join(f"{k}={sw[y_tr==k][0]:.2f}" for k in sorted(np.unique(y_tr))))

    model = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=4,
        eval_metric="mlogloss", tree_method="hist",
        random_state=42, n_jobs=-1,
    )
    if sw is not None:
        model.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
    else:
        model.fit(X_tr, y_tr, verbose=False)

    pred = model.predict(X_te)
    acc = float(accuracy_score(y_te, pred))
    f1m = float(f1_score(y_te, pred, average="macro", zero_division=0))
    print(f"  acc={acc:.4f}  f1m={f1m:.4f}")
    print(classification_report(y_te, pred,
        target_names=[LABEL_NAMES[i] for i in range(4)], zero_division=0))

    per = {}
    for ds in np.unique(src_te):
        m = src_te == ds
        per[ds] = {"n": int(m.sum()),
                   "acc": float(accuracy_score(y_te[m], pred[m])),
                   "f1_macro": float(f1_score(y_te[m], pred[m], average="macro", zero_division=0))}

    imp = dict(zip(feats, model.feature_importances_.tolist()))
    joblib.dump({"model": model, "features": feats}, out_dir / "model.joblib")
    json.dump({
        "tag": tag, "n_features": len(feats), "features": feats,
        "use_class_weights": use_class_weights,
        "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "accuracy": acc, "f1_macro": f1m, "per_dataset": per,
        "feature_importance": imp,
    }, open(out_dir / "metrics.json", "w"), indent=2)
    print(f"  saved -> {out_dir.relative_to(REPO)}")
    return acc, f1m


def main():
    plans = [
        ("A",
         REPO / "models/routers/lean19/fusion_dataset_lean19.csv",
         LEAN19_FEATS,
         REPO / "models/routers/lean19_v2_A",
         True),
        ("B",
         REPO / "models/routers/lean19_v2_B/fusion_dataset.csv",
         LEAN19_FEATS,
         REPO / "models/routers/lean19_v2_B",
         False),
        ("C",
         REPO / "models/routers/lean19_v2_C/fusion_dataset.csv",
         ALL_BC,
         REPO / "models/routers/lean19_v2_C",
         False),
        ("ABC",
         REPO / "models/routers/lean19_v2_BC/fusion_dataset.csv",
         ALL_BC,
         REPO / "models/routers/lean19_v2_ABC",
         True),
    ]
    results = {}
    for tag, csv, feats, out, cw in plans:
        if (out / "model.joblib").exists():
            print(f"\n[skip {tag}] model exists at {out.relative_to(REPO)}")
            continue
        try:
            a, f = train_variant(csv, feats, out, f"lean19_v2_{tag}", cw)
            results[tag] = (a, f)
        except SystemExit as e:
            print(f"[FAIL {tag}] {e}")

    print("\n=== Summary ===")
    for tag, (a, f) in results.items():
        print(f"  lean19_v2_{tag}: acc={a:.4f}  f1m={f:.4f}")


if __name__ == "__main__":
    main()
