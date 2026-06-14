"""
classifier/train_robust8_noreject.py — NO-REJECT (3-class) variants of robust8.

robust8 is a 4-class trust router {0 reject, 1 trust_rgb, 2 trust_ir, 3 both}. This trains two no-reject
variants on the SAME corpus / features / hyperparams as robust8 (eval/compare_routing_pipeline.py
train_new_router), with the reject class removed two ways:
  drop : keep only rows where >=1 modality has a TP  (trust_label in {1,2,3})
  both : remap reject (0) -> both (3)
The router then ALWAYS routes to a modality; the downstream verifier (mlp_v5 / mlp_v5_ir_aligned) does all
false-positive rejection.

XGBoost requires 0-indexed labels, so we fit on {0,1,2} = {trust_rgb, trust_ir, both} and store a
`label_map` ({0:1,1:2,2:3}) in the joblib so the eval harness recovers the {1,2,3} trust labels its gate()
expects (pipeline_eval_unified.batch_labels applies label_map). argmax decision (tau=None — the model
chooses; no reject is ever emitted). Zero-GPU (reads the cached full56 CSV).

Run:  py classifier/train_robust8_noreject.py
"""
from __future__ import annotations
import re
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parent.parent
FULL56 = REPO / "models/routers/optimal_v1/fusion_dataset_full56.csv"

# --- mirror eval/compare_routing_pipeline.py exactly (kept local to avoid the ultralytics import web) ---
ROBUST6 = ["rgb_max_conf", "ir_max_conf", "rgb_best_log_bbox_area",
           "ir_best_log_bbox_area", "rgb_best_aspect_ratio", "ir_best_aspect_ratio"]
F8 = ROBUST6 + ["rgb_mean_conf", "is_grayscale"]
THERMAL_SRC = {"antiuav", "svanstrom"}
SEQ_RE = re.compile(r"^(.+?)(?:_f\d+|_frame\d+|_\d{4,})(?:_visible|_infrared)?$", re.I)

LABEL_NAMES = {1: "trust_rgb", 2: "trust_ir", 3: "both"}
TRUST2FIT = {1: 0, 2: 1, 3: 2}            # harness trust label -> 0-indexed xgb label
FIT2TRUST = {0: 1, 1: 2, 2: 3}            # stored as label_map for inference


def seq_id(stem, src):
    m = SEQ_RE.match(str(stem)); base = m.group(1).rstrip("_") if m else str(stem)
    return f"{src}::{base}"


def xgb():
    # identical to robust8's train_new_router except num_class (4 -> inferred 3)
    return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                         colsample_bytree=0.8, objective="multi:softprob",
                         eval_metric="mlogloss", tree_method="hist", random_state=42, n_jobs=4)


def main():
    df = pd.read_csv(FULL56)
    df["regime"] = np.where(df["source"].isin(THERMAL_SRC), "thermal", "grayscale")
    df["is_grayscale"] = (df["regime"] == "grayscale").astype(int)
    df["_seq"] = [seq_id(s, c) for s, c in zip(df["stem"], df["source"])]
    dist = {int(k): int(v) for k, v in df["trust_label"].value_counts().sort_index().items()}
    print(f"full56: {len(df)} rows | 4-class trust_label dist (0=reject): {dist}")

    for variant in ("drop", "both"):
        d = df.copy()
        if variant == "drop":
            d = d[d["trust_label"].isin([1, 2, 3])].reset_index(drop=True)
        else:                                   # both: reject(0) -> both(3)
            d["trust_label"] = d["trust_label"].replace(0, 3)
        y_trust = d["trust_label"].values
        y_fit = np.array([TRUST2FIT[int(v)] for v in y_trust])
        tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=42).split(d, y_fit, d["_seq"]))
        m = xgb(); m.fit(d.iloc[tr][F8].values, y_fit[tr])

        pred_trust = np.array([FIT2TRUST[int(v)] for v in m.predict(d.iloc[te][F8].values)])
        yte = y_trust[te]
        print(f"\n=== robust8_nr_{variant} ===")
        print(f"  rows {len(d)} (train {len(tr)} / test {len(te)}) | classes_={list(m.classes_)} -> "
              f"label_map {FIT2TRUST}")
        print(f"  label dist: { {LABEL_NAMES[k]: int((y_trust == k).sum()) for k in (1, 2, 3)} }")
        for cls in (1, 2, 3):
            p, r, f, s = precision_recall_fscore_support(yte, pred_trust, labels=[cls], zero_division=0)
            print(f"    {LABEL_NAMES[cls]:<10} P={p[0]:.3f} R={r[0]:.3f} F1={f[0]:.3f} (n={int(s[0])})")
        mac = precision_recall_fscore_support(yte, pred_trust, labels=[1, 2, 3], average="macro", zero_division=0)
        print(f"    macro-F1={mac[2]:.3f}  acc={float((pred_trust == yte).mean()):.3f}")

        out = REPO / f"models/routers/robust8_noreject_{variant}"
        out.mkdir(parents=True, exist_ok=True)
        bundle = {"model": m, "features": F8, "feat_key": "f8", "tau": None, "label_map": FIT2TRUST}
        joblib.dump(bundle, out / "model.joblib")
        print(f"  saved -> {out / 'model.joblib'}")
        if variant == "drop":   # promote the chosen production router to the flat, obvious path
            flat = REPO / "models/routers/robust8_noreject.joblib"
            joblib.dump(bundle, flat)
            print(f"  saved (flat production path) -> {flat}")


if __name__ == "__main__":
    main()
