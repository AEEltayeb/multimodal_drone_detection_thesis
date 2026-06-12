"""train_lean_ft4.py — train + compare trust classifiers on the ft4-mined lean fusion
dataset (models/routers/lean_ft4/fusion_dataset_lean19.csv).

Tests the leakage hypothesis on the CURRENT detector: does dropping the scene-fingerprint
features (brightness img_mean/std, absolute position pos_x/y, dist_to_center) hurt or
*help* generalization — especially on OOD confuser video?

Feature sets (all from the 19 cheap lean columns):
  all19      — every lean feature (incl. fingerprints)
  no_fp      — drop brightness + absolute-position fingerprints (10 feats)
  robust6    — confidences + box geometry only
  meta4      — rgb/ir max_conf + rgb box geometry (meta5_geo analog, no xmodal)

GroupShuffleSplit by sequence (no frame leakage). XGBoost 4-class, same recipe as
ablation_feature_sets.py. Reports per-surface F1-macro incl. video breakdown; saves
best model + metrics. Compares to the recorded sa32 reference (OLD detector — anchor only).
"""
from __future__ import annotations
import json, re
from pathlib import Path

import numpy as np, pandas as pd
import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parent.parent
CSV = REPO / "models/routers/lean_ft4/fusion_dataset_lean19.csv"
SA32_METRICS = REPO / "models/routers/scene_aware_v3more_32feat/metrics.json"
OUT = REPO / "models/routers/lean_ft4"
LABEL_NAMES = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}
SEQ_RE = re.compile(r"^(.+?)(?:_f\d+|_frame\d+|_\d{4,})(?:_visible|_infrared)?$", re.I)

FINGERPRINTS = ["rgb_img_mean", "ir_img_mean", "rgb_img_std",
                "rgb_best_pos_x", "ir_best_pos_x", "rgb_best_pos_y", "ir_best_pos_y",
                "rgb_best_dist_to_center", "ir_best_dist_to_center"]
ROBUST6 = ["rgb_max_conf", "ir_max_conf", "rgb_best_log_bbox_area",
           "ir_best_log_bbox_area", "rgb_best_aspect_ratio", "ir_best_aspect_ratio"]
META4 = ["rgb_max_conf", "ir_max_conf", "rgb_best_log_bbox_area", "rgb_best_aspect_ratio"]


def seq_id(stem, src):
    m = SEQ_RE.match(str(stem)); base = m.group(1).rstrip("_") if m else str(stem)
    return f"{src}::{base}"


def surface(src: str) -> str:
    if src.startswith("video_drone"): return "video_drone(OOD)"
    if src.startswith(("video_birds", "video_airplanes", "video_helicopters")):
        return "video_confuser(OOD)"
    return src


def train_eval(df, tr, te, feats, tag):
    Xtr, Xte = df.iloc[tr][feats].values, df.iloc[te][feats].values
    ytr, yte = df.iloc[tr]["trust_label"].values, df.iloc[te]["trust_label"].values
    m = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, objective="multi:softprob",
                      num_class=4, eval_metric="mlogloss", tree_method="hist",
                      random_state=42, n_jobs=1)
    m.fit(Xtr, ytr)
    yp = m.predict(Xte)
    surf = df.iloc[te]["source"].map(surface).values
    per = {}
    for s in sorted(set(surf)):
        mask = surf == s
        per[s] = round(float(f1_score(yte[mask], yp[mask], average="macro", zero_division=0)), 4)
    imp = dict(sorted(zip(feats, [float(v) for v in m.feature_importances_]), key=lambda x: -x[1]))
    return {
        "tag": tag, "n_features": len(feats),
        "acc": round(float(accuracy_score(yte, yp)), 4),
        "f1_macro": round(float(f1_score(yte, yp, average="macro", zero_division=0)), 4),
        "per_surface": per, "top5": list(imp.items())[:5],
    }, m


def main():
    df = pd.read_csv(CSV)
    all19 = [c for c in df.columns if c not in ("trust_label", "stem", "source")]
    df["_seq"] = [seq_id(s, src) for s, src in zip(df["stem"], df["source"])]
    print(f"{len(df):,} rows, {len(all19)} features, {df['_seq'].nunique()} sequences")

    variants = {
        "all19": all19,
        "no_fp": [c for c in all19 if c not in FINGERPRINTS],
        "robust6": ROBUST6,
        "meta4": META4,
    }
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=42)
                  .split(df, df["trust_label"], df["_seq"]))
    print(f"train {len(tr):,} / test {len(te):,}\n")

    sa32 = json.load(open(SA32_METRICS)) if SA32_METRICS.exists() else {}
    results, best, best_tag, best_f1 = {}, None, None, -1
    for tag, feats in variants.items():
        r, m = train_eval(df, tr, te, feats, tag)
        results[tag] = r
        if r["f1_macro"] > best_f1:
            best, best_tag, best_f1 = m, tag, r["f1_macro"]
        print(f"[{tag}] {r['n_features']} feats  acc={r['acc']}  F1m={r['f1_macro']}")
        print(f"   top5: {', '.join(f'{k}({v:.2f})' for k,v in r['top5'])}")

    # summary table
    surfs = ["antiuav", "svanstrom", "video_drone(OOD)", "video_confuser(OOD)"]
    print(f"\n{'variant':<10}{'feats':>6}{'F1m':>8}   " + "".join(f"{s:>22}" for s in surfs))
    if sa32:
        print(f"{'sa32(ref*)':<10}{'32':>6}{sa32.get('f1_macro',0):>8.4f}   (*OLD detector + extra scene feats; anchor only)")
    for tag, r in results.items():
        row = f"{tag:<10}{r['n_features']:>6}{r['f1_macro']:>8.4f}   "
        row += "".join(f"{r['per_surface'].get(s, float('nan')):>22.4f}" for s in surfs)
        print(row)

    joblib.dump({"model": best, "features": variants[best_tag]}, OUT / f"trust_ft4_{best_tag}.joblib")
    json.dump(results, open(OUT / "lean_ft4_compare.json", "w"), indent=2)
    print(f"\nbest = {best_tag} (F1m {best_f1}); saved trust_ft4_{best_tag}.joblib + lean_ft4_compare.json")


if __name__ == "__main__":
    main()
