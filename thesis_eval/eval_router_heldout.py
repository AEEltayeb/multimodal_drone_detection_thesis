"""thesis_eval/eval_router_heldout.py — held-out classification metrics + confusion matrix
for the SHIPPED trust router robust8-nr (3-class, no reject).

Eval-only: loads the existing models/routers/robust8_noreject_drop/model.joblib (does NOT retrain
or re-save it) and reconstructs the EXACT held-out split used by classifier/train_robust8_noreject.py
(GroupShuffleSplit, test_size=0.25, random_state=42, grouped by sequence id over the dropped 3-class
corpus). Prints the 3x3 confusion matrix, per-class P/R/F1, macro-F1, accuracy, and class-pick freq.

Zero-GPU. Run: py thesis_eval/eval_router_heldout.py
"""
from __future__ import annotations
import re, json
from pathlib import Path
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

REPO = Path(__file__).resolve().parent.parent
FULL56 = REPO / "models/routers/optimal_v1/fusion_dataset_full56.csv"
MODEL = REPO / "models/routers/robust8_noreject_drop/model.joblib"

ROBUST6 = ["rgb_max_conf", "ir_max_conf", "rgb_best_log_bbox_area",
           "ir_best_log_bbox_area", "rgb_best_aspect_ratio", "ir_best_aspect_ratio"]
F8 = ROBUST6 + ["rgb_mean_conf", "is_grayscale"]
THERMAL_SRC = {"antiuav", "svanstrom"}
SEQ_RE = re.compile(r"^(.+?)(?:_f\d+|_frame\d+|_\d{4,})(?:_visible|_infrared)?$", re.I)
LABEL_NAMES = {1: "trust_rgb", 2: "trust_ir", 3: "both"}
TRUST2FIT = {1: 0, 2: 1, 3: 2}
FIT2TRUST = {0: 1, 1: 2, 2: 3}


def seq_id(stem, src):
    m = SEQ_RE.match(str(stem)); base = m.group(1).rstrip("_") if m else str(stem)
    return f"{src}::{base}"


def main():
    df = pd.read_csv(FULL56)
    df["regime"] = np.where(df["source"].isin(THERMAL_SRC), "thermal", "grayscale")
    df["is_grayscale"] = (df["regime"] == "grayscale").astype(int)
    df["_seq"] = [seq_id(s, c) for s, c in zip(df["stem"], df["source"])]

    # production router = "drop" variant: rows where >=1 modality has a TP (trust_label in {1,2,3})
    d = df[df["trust_label"].isin([1, 2, 3])].reset_index(drop=True)
    y_trust = d["trust_label"].values
    y_fit = np.array([TRUST2FIT[int(v)] for v in y_trust])
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=42).split(d, y_fit, d["_seq"]))

    bundle = joblib.load(MODEL)
    m = bundle["model"]
    pred = np.array([FIT2TRUST[int(v)] for v in m.predict(d.iloc[te][F8].values)])
    yte = y_trust[te]

    print(f"== robust8-nr held-out classification ==  (no retrain; shipped model loaded)")
    print(f"corpus rows {len(d)} (3-class) | train {len(tr)} / TEST {len(te)} | groups by sequence")
    print(f"train sequences {d.iloc[tr]['_seq'].nunique()} / test sequences {d.iloc[te]['_seq'].nunique()}")
    cm = confusion_matrix(yte, pred, labels=[1, 2, 3])
    print("\nconfusion matrix (rows = TRUE, cols = PRED) order [trust_rgb, trust_ir, both]:")
    print(cm)
    print()
    for cls in (1, 2, 3):
        p, r, f, s = precision_recall_fscore_support(yte, pred, labels=[cls], zero_division=0)
        print(f"  {LABEL_NAMES[cls]:<10} P={p[0]:.3f} R={r[0]:.3f} F1={f[0]:.3f} (support n={int(s[0])})")
    mac = precision_recall_fscore_support(yte, pred, labels=[1, 2, 3], average="macro", zero_division=0)
    print(f"  macro-F1 = {mac[2]:.3f}   accuracy = {float((pred == yte).mean()):.3f}")
    print(f"\n  true dist  (test): { {LABEL_NAMES[k]: int((yte == k).sum())  for k in (1, 2, 3)} }")
    print(f"  pick freq  (pred): { {LABEL_NAMES[k]: int((pred == k).sum()) for k in (1, 2, 3)} }")

    per = {}
    for cls in (1, 2, 3):
        p, r, f, s = precision_recall_fscore_support(yte, pred, labels=[cls], zero_division=0)
        per[LABEL_NAMES[cls]] = {"P": round(float(p[0]), 4), "R": round(float(r[0]), 4),
                                 "F1": round(float(f[0]), 4), "n": int(s[0])}
    out = {"model": "robust8-nr", "labels": ["trust_rgb", "trust_ir", "both"],
           "n_train": int(len(tr)), "n_test": int(len(te)),
           "train_seq": int(d.iloc[tr]["_seq"].nunique()), "test_seq": int(d.iloc[te]["_seq"].nunique()),
           "confusion_matrix": cm.tolist(), "per_class": per,
           "macro_f1": round(float(mac[2]), 4), "accuracy": round(float((pred == yte).mean()), 4),
           "pick_freq": {LABEL_NAMES[k]: int((pred == k).sum()) for k in (1, 2, 3)}}
    od = REPO / "thesis_eval" / "results" / "per_model_heldout"; od.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(od / "router_heldout.json", "w"), indent=2)
    print(f"\n  wrote {od / 'router_heldout.json'}")


if __name__ == "__main__":
    main()
