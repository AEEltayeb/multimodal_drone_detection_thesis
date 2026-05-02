"""
train_classifier_v11.py — P2 retrain (fusion classifier v1.1).

Merges the three available CSVs:
  - fusion_dataset.csv              Anti-UAV-RGBT (paired, mostly drone pos)
  - svanstrom_frame_dataset.csv     Svanström (mostly real-thermal aerial)
  - youtube_aerial_dataset.csv      YouTube aerial (grayscale-replicate + real IR)

Performs a SEQUENCE-LEVEL train/val/test split (stem prefix up to `_f\\d+`)
so adjacent frames from the same video don't leak across splits. Retrains
XGBoost with rebalanced per-class weights and picks a threshold under both
(a) F1-optimal and (b) FPR-constrained (airplane_FPR, helicopter_FPR < 2%)
rules. Saves the FPR-constrained model as classifier_v1.1.joblib if it costs
≤1pp drone recall vs F1-optimal; otherwise ships F1-optimal with the per-class
FPRs reported.

Usage:
    python classifier/train_classifier_v11.py
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import precision_recall_curve, average_precision_score
from xgboost import XGBClassifier


CATEGORY_PATTERN = re.compile(r"^IR_([A-Z]+)_", re.IGNORECASE)
KNOWN_CATEGORIES = {"airplane", "helicopter", "bird", "drone"}


def derive_category(stem, label):
    """Match eval_aerial_negatives.derive_category, with label→'drone' fallback."""
    m = CATEGORY_PATTERN.match(str(stem))
    if m:
        cat = m.group(1).lower()
        if cat in KNOWN_CATEGORIES:
            return cat
    return "drone" if label == 1 else "other"


def sequence_id(stem):
    """'20190925_124000_1_1_f000175' -> '20190925_124000_1_1'
       'IR_AIRPLANE_YT_airplane_rgb_f000408' -> 'IR_AIRPLANE_YT_airplane_rgb'"""
    s = str(stem)
    m = re.match(r"^(.+?)_f\d+$", s)
    return m.group(1) if m else s


def load_and_merge(runs_dir):
    frames = []
    for name, source_tag in [
        ("fusion_dataset.csv", "antiuav"),
        ("svanstrom_frame_dataset.csv", "svanstrom"),
        ("youtube_aerial_dataset.csv", "youtube"),
    ]:
        path = runs_dir / name
        if not path.exists():
            print(f"  [SKIP] {path} missing")
            continue
        df = pd.read_csv(path)
        df["data_source"] = source_tag
        print(f"  {name}: {len(df)} rows, labels={df['label'].value_counts().to_dict()}")
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True, sort=False)

    for col in ["rgb_brightness", "ir_brightness"]:
        if col not in merged.columns:
            merged[col] = 0.0
        merged[col] = merged[col].fillna(0.0)

    if "ir_is_real_thermal" not in merged.columns:
        merged["ir_is_real_thermal"] = 1
    mask_youtube = merged["data_source"] == "youtube"
    merged.loc[~mask_youtube, "ir_is_real_thermal"] = merged.loc[
        ~mask_youtube, "ir_is_real_thermal"].fillna(1)
    merged["ir_is_real_thermal"] = merged["ir_is_real_thermal"].astype(int)

    if "negative_class" not in merged.columns:
        merged["negative_class"] = None
    for idx, row in merged.iterrows():
        if pd.isna(row.get("negative_class")) or row.get("negative_class") is None:
            merged.at[idx, "negative_class"] = derive_category(row["stem"], row["label"])

    merged["category"] = merged.apply(
        lambda r: derive_category(r["stem"], r["label"]), axis=1)

    merged["sequence_id"] = merged["stem"].apply(sequence_id)
    return merged


FEATURE_COLS = [
    "max_conf_rgb", "max_conf_ir", "conf_max", "conf_min", "conf_mean",
    "conf_delta", "both_detected", "n_dets_rgb", "n_dets_ir", "n_dets_total",
    "conf_rgb_2nd", "conf_ir_2nd", "rgb_area_norm", "ir_area_norm", "hour",
    "rgb_brightness", "ir_brightness",
    "time_of_day_day", "time_of_day_dusk_dawn", "time_of_day_night",
    "time_of_day_unknown",
]


def build_feature_matrix(df):
    out = df.copy()
    if "time_of_day" in out.columns:
        dummies = pd.get_dummies(out["time_of_day"].fillna("unknown"),
                                 prefix="time_of_day", dtype=float)
        for c in dummies.columns:
            out[c] = dummies[c]
    for col in FEATURE_COLS:
        if col not in out.columns:
            out[col] = 0.0
    X = out[FEATURE_COLS].astype(float).values
    return X


def group_split(df, test_frac=0.15, val_frac=0.15, seed=42):
    """Split by sequence_id, stratified approximately by (category, label)."""
    rng = np.random.default_rng(seed)
    seq_df = df.drop_duplicates("sequence_id").copy()
    seq_df["group"] = seq_df["category"] + "_" + seq_df["label"].astype(str)

    train, val, test = [], [], []
    for group, sub in seq_df.groupby("group"):
        seqs = sub["sequence_id"].tolist()
        rng.shuffle(seqs)
        n = len(seqs)
        n_test = max(1, int(round(n * test_frac))) if n > 2 else 0
        n_val = max(1, int(round(n * val_frac))) if n - n_test > 2 else 0
        test.extend(seqs[:n_test])
        val.extend(seqs[n_test:n_test + n_val])
        train.extend(seqs[n_test + n_val:])
    train_mask = df["sequence_id"].isin(train)
    val_mask = df["sequence_id"].isin(val)
    test_mask = df["sequence_id"].isin(test)
    return train_mask, val_mask, test_mask


def per_category_fpr(y_true, y_prob, categories, threshold):
    preds = (y_prob >= threshold).astype(int)
    results = {}
    for cat in ["airplane", "helicopter", "bird", "drone", "other"]:
        mask = categories == cat
        n = int(mask.sum())
        if n == 0:
            results[cat] = {"n": 0, "fpr": None, "recall": None}
            continue
        if cat == "drone":
            pos_mask = mask & (y_true == 1)
            tp = int((preds == 1)[pos_mask].sum())
            n_pos = int(pos_mask.sum())
            recall = tp / n_pos if n_pos else None
            results[cat] = {"n": n, "recall": round(recall, 4) if recall is not None else None,
                            "n_pos": n_pos}
            neg_mask = mask & (y_true == 0)
            n_neg = int(neg_mask.sum())
            fp = int((preds == 1)[neg_mask].sum())
            results[cat]["n_neg"] = n_neg
            results[cat]["fpr"] = round(fp / n_neg, 4) if n_neg else None
        else:
            neg_mask = mask & (y_true == 0)
            n_neg = int(neg_mask.sum())
            fp = int((preds == 1)[neg_mask].sum())
            results[cat] = {"n": n, "n_neg": n_neg,
                            "fpr": round(fp / n_neg, 4) if n_neg else None}
    return results


def sweep_threshold(y_true, y_prob):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    precisions, recalls = precisions[:-1], recalls[:-1]
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-12)
    best_i = int(np.argmax(f1s))
    return {"threshold": float(thresholds[best_i]),
            "precision": float(precisions[best_i]),
            "recall": float(recalls[best_i]), "f1": float(f1s[best_i])}


def find_fpr_constrained_threshold(y_true, y_prob, categories, max_fpr=0.02,
                                   bird_max_fpr=0.05, min_recall=0.90):
    """Sweep thresholds; pick the LOWEST threshold that meets all per-class
    FPR caps and the drone-recall floor. Low threshold = high recall."""
    ts = np.unique(np.round(np.linspace(0.01, 0.99, 99), 3))
    best = None
    for t in ts:
        res = per_category_fpr(y_true, y_prob, categories, t)
        plane = res["airplane"]["fpr"]
        heli = res["helicopter"]["fpr"]
        bird = res["bird"]["fpr"]
        drone_r = res["drone"]["recall"]
        if drone_r is None or drone_r < min_recall:
            continue
        if plane is not None and plane > max_fpr:
            continue
        if heli is not None and heli > max_fpr:
            continue
        if bird is not None and bird > bird_max_fpr:
            continue
        return {"threshold": float(t), "metrics": res,
                "drone_recall": drone_r,
                "meets_constraints": True}
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="classifier/runs")
    p.add_argument("--config", default="classifier/config.yaml")
    p.add_argument("--output", default="classifier/runs/classifier_v1.1.joblib")
    p.add_argument("--report", default="classifier/runs/aerial_eval/v1.1_train_report.json")
    p.add_argument("--max-fpr", type=float, default=0.02)
    p.add_argument("--bird-max-fpr", type=float, default=0.05)
    p.add_argument("--min-recall", type=float, default=0.95)
    args = p.parse_args()

    runs_dir = Path(args.runs_dir)
    print("Loading and merging CSVs...")
    df = load_and_merge(runs_dir)
    print(f"  merged: {len(df)} rows")
    print(f"  label dist:     {df['label'].value_counts().to_dict()}")
    print(f"  category dist:  {df['category'].value_counts().to_dict()}")
    print(f"  sources:        {df['data_source'].value_counts().to_dict()}")
    print(f"  ir_real_thermal:{df['ir_is_real_thermal'].value_counts().to_dict()}")

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    print("\nSplitting by sequence_id...")
    train_mask, val_mask, test_mask = group_split(df)
    for tag, m in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        sub = df[m]
        print(f"  {tag:5s}: {len(sub):6d} rows, "
              f"pos={int(sub['label'].sum()):5d} "
              f"cat={sub['category'].value_counts().to_dict()}")

    X = build_feature_matrix(df)
    y = df["label"].values.astype(int)
    cats = df["category"].values

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_va, y_va = X[val_mask], y[val_mask]
    X_te, y_te = X[test_mask], y[test_mask]
    cats_va = cats[val_mask]
    cats_te = cats[test_mask]

    n_pos = int(y_tr.sum())
    n_neg = len(y_tr) - n_pos
    aerial_mask_tr = np.isin(cats[train_mask], ["airplane", "helicopter", "bird"])
    sample_weight = np.ones(len(y_tr))
    sample_weight[aerial_mask_tr] = 3.0
    print(f"\n  class balance: pos={n_pos} neg={n_neg} "
          f"scale_pos_weight={n_neg / max(1, n_pos):.3f}")
    print(f"  aerial-negative upweight: 3.0 on {int(aerial_mask_tr.sum())} rows")

    xgb_cfg = cfg.get("xgboost", {})
    model = XGBClassifier(
        n_estimators=xgb_cfg.get("n_estimators", 300),
        max_depth=xgb_cfg.get("max_depth", 5),
        learning_rate=xgb_cfg.get("learning_rate", 0.08),
        min_child_weight=xgb_cfg.get("min_child_weight", 10),
        subsample=xgb_cfg.get("subsample", 0.8),
        colsample_bytree=xgb_cfg.get("colsample_bytree", 0.8),
        scale_pos_weight=float(n_neg / max(1, n_pos)),
        eval_metric="aucpr",
        random_state=42,
        verbosity=0,
    )
    print("\nTraining XGBoost...")
    model.fit(X_tr, y_tr, sample_weight=sample_weight,
              eval_set=[(X_va, y_va)], verbose=False)

    prob_va = model.predict_proba(X_va)[:, 1]
    prob_te = model.predict_proba(X_te)[:, 1]

    f1_pick = sweep_threshold(y_va, prob_va)
    print(f"\nF1-optimal @ val: t={f1_pick['threshold']:.4f} "
          f"P={f1_pick['precision']:.4f} R={f1_pick['recall']:.4f}")

    fpr_pick = find_fpr_constrained_threshold(
        y_va, prob_va, cats_va,
        max_fpr=args.max_fpr, bird_max_fpr=args.bird_max_fpr,
        min_recall=args.min_recall)
    if fpr_pick:
        print(f"FPR-constrained: t={fpr_pick['threshold']:.4f} "
              f"drone_R={fpr_pick['drone_recall']:.4f} "
              f"plane_FPR={fpr_pick['metrics']['airplane']['fpr']} "
              f"heli_FPR={fpr_pick['metrics']['helicopter']['fpr']} "
              f"bird_FPR={fpr_pick['metrics']['bird']['fpr']}")
    else:
        print("FPR-constrained: NO threshold meets all constraints "
              f"(min_recall={args.min_recall}, max_fpr={args.max_fpr})")

    # Choose final threshold: prefer FPR-constrained if within 2pp recall.
    final_t = f1_pick["threshold"]
    final_tag = "f1_optimal"
    if fpr_pick and fpr_pick["drone_recall"] >= f1_pick["recall"] - 0.02:
        final_t = fpr_pick["threshold"]
        final_tag = "fpr_constrained"
    print(f"\nSelected threshold: {final_t:.4f} ({final_tag})")

    test_metrics = per_category_fpr(y_te, prob_te, cats_te, final_t)
    print(f"\n=== Test-set metrics at t={final_t:.4f} ===")
    for cat in ["drone", "airplane", "helicopter", "bird", "other"]:
        r = test_metrics[cat]
        if cat == "drone":
            print(f"  drone:   n={r.get('n',0):5d}  recall={r.get('recall')}  fpr={r.get('fpr')}")
        else:
            print(f"  {cat:8s} n={r.get('n',0):5d}  fpr={r.get('fpr')}  (on {r.get('n_neg',0)} neg)")

    aucpr = float(average_precision_score(y_te, prob_te))
    print(f"\n  AUC-PR test: {aucpr:.4f}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "feature_cols": FEATURE_COLS,
        "threshold": float(final_t),
        "threshold_tag": final_tag,
        "f1_optimal_threshold": f1_pick["threshold"],
        "model_type": "xgboost",
        "fusion_mode": "frame",
        "version": "1.1",
    }, out_path)
    print(f"\nSaved model -> {out_path}")

    imps = dict(zip(FEATURE_COLS, [float(x) for x in model.feature_importances_]))
    imps_sorted = sorted(imps.items(), key=lambda x: x[1], reverse=True)
    print("\nFeature importance:")
    for f, i in imps_sorted:
        bar = "#" * int(i * 60)
        print(f"  {f:28s} {i:.4f}  {bar}")

    report = {
        "version": "1.1",
        "merged_row_counts": {
            k: int(v) for k, v in df["data_source"].value_counts().items()
        },
        "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().items()},
        "category_counts": {k: int(v) for k, v in df["category"].value_counts().items()},
        "split_sizes": {"train": int(train_mask.sum()),
                         "val": int(val_mask.sum()),
                         "test": int(test_mask.sum())},
        "f1_optimal": f1_pick,
        "fpr_constrained": fpr_pick,
        "final_threshold": float(final_t),
        "final_tag": final_tag,
        "test_metrics": test_metrics,
        "test_aucpr": aucpr,
        "feature_importance": imps,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2, default=str))
    print(f"Report -> {args.report}")


if __name__ == "__main__":
    main()
