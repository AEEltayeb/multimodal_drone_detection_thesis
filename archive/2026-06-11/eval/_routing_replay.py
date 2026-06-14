"""_routing_replay.py — torch-free re-replay of the cached routing-pipeline comparison.

The caches in eval/results/_routing_pipeline_cmp/cache/*.pkl already hold per-frame
detection survivors + f8/f32 feature vectors + GT. This script does ONLY the CPU work:
train robust8 (= robust6 + rgb_mean_conf + is_grayscale) + pick tau, load sa32/robust6,
replay the cascade ablation for all routers (incl tau variants), rewrite comparison.md.
No ultralytics/torch import -> no CUDA-init hang. Seconds.

  py -u eval/_routing_replay.py
"""
from __future__ import annotations
import json, pickle, re, time
from pathlib import Path
import numpy as np, joblib, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import precision_recall_fscore_support
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "eval" / "results" / "_routing_pipeline_cmp"
SA32 = REPO / "models/routers/scene_aware_v3more_32feat/model.joblib"
ROBUST6_JBL = REPO / "models/routers/lean_ft4/trust_ft4_robust6.joblib"
FULL56 = REPO / "models/routers/optimal_v1/fusion_dataset_full56.csv"
ROBUST6 = ["rgb_max_conf", "ir_max_conf", "rgb_best_log_bbox_area",
           "ir_best_log_bbox_area", "rgb_best_aspect_ratio", "ir_best_aspect_ratio"]
F8 = ROBUST6 + ["rgb_mean_conf", "is_grayscale"]
THERMAL_SRC = {"antiuav", "svanstrom"}
SEQ_RE = re.compile(r"^(.+?)(?:_f\d+|_frame\d+|_\d{4,})(?:_visible|_infrared)?$", re.I)


def seq_id(stem, src):
    m = SEQ_RE.match(str(stem)); base = m.group(1).rstrip("_") if m else str(stem)
    return f"{src}::{base}"


def train_robust8(tau_override=None):
    df = pd.read_csv(FULL56)
    df["regime"] = np.where(df["source"].isin(THERMAL_SRC), "thermal", "grayscale")
    df["is_grayscale"] = (df["regime"] == "grayscale").astype(int)
    df["_seq"] = [seq_id(s, c) for s, c in zip(df["stem"], df["source"])]
    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=42).split(df, df["trust_label"], df["_seq"]))
    m = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, objective="multi:softprob", num_class=4,
                      eval_metric="mlogloss", tree_method="hist", random_state=42, n_jobs=4)
    m.fit(df.iloc[tr][F8].values, df.iloc[tr]["trust_label"].values)
    gte = te[df.iloc[te]["regime"].values == "grayscale"]
    yg = df.iloc[gte]["trust_label"].values
    pg = m.predict_proba(df.iloc[gte][F8].values)
    rows = []
    for tau in np.round(np.arange(0.05, 0.51, 0.05), 2):
        pred = np.where(pg[:, 1] >= tau, 1, pg.argmax(1))
        p, r, f, _ = precision_recall_fscore_support(yg, pred, labels=[1], zero_division=0)
        rows.append((float(tau), float(r[0]), float(p[0]), float(f[0])))
    best = max(rows, key=lambda x: x[3])
    tau = tau_override if tau_override is not None else best[0]
    print("robust8 = robust6 + rgb_mean_conf + is_grayscale | held-out grayscale trust_rgb tau sweep:")
    for t, r, p, f in rows:
        print(f"    tau={t:.2f}  R={r:.3f} P={p:.3f} F1={f:.3f}" + ("  <- chosen(max-F1)" if t == tau else ""))
    clf = {"model": m, "features": F8, "tau": tau, "feat_key": "f8"}
    joblib.dump(clf, OUT / "robust8.joblib")
    return clf


def predict_trust(clf, F8_order, F32_order, rows, which):
    order = F32_order if clf.get("feat_key") == "f32" else F8_order
    vec = f"f32_{which}" if clf.get("feat_key") == "f32" else f"f8_{which}"
    model, feats = clf["model"], clf.get("features")
    if feats is None:
        X = np.array([r[vec] for r in rows], float)
    else:
        idx = [order.index(f) for f in feats]
        X = np.array([[r[vec][i] for i in idx] for r in rows], float)
    if clf.get("tau") is not None:
        proba = model.predict_proba(X)
        return np.where(proba[:, 1] >= clf["tau"], 1, proba.argmax(1))
    return model.predict(X)


def ablate(meta, rows, F8_order, F32_order, classifiers):
    hd = meta["has_drones"]
    pos = np.array([r["n_gt"] for r in rows]) > 0
    rgb_any = np.array([r["rgb_any"] for r in rows]); ir_any = np.array([r["ir_any"] for r in rows])
    rgb_s = np.array([r["rgb_surv"] for r in rows]); ir_s = np.array([r["ir_surv"] for r in rows])

    def metrics(alert):
        tp = int((alert & pos).sum()); fp = int((alert & ~pos).sum())
        fn = int((~alert & pos).sum()); tn = int((~alert & ~pos).sum())
        out = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "fire_rate": round(fp / max(fp + tn, 1), 4)}
        if hd:
            p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
            out.update({"precision": round(p, 4), "recall": round(r, 4),
                        "f1": round(2 * p * r / max(p + r, 1e-9), 4)})
        return out

    res = {"bare": metrics(rgb_any | ir_any), "filter_only": metrics(rgb_s | ir_s)}
    for cname, clf in classifiers.items():
        t_all = predict_trust(clf, F8_order, F32_order, rows, "all")
        t_flt = predict_trust(clf, F8_order, F32_order, rows, "flt")
        trgb_a = np.isin(t_all, [1, 3]); tir_a = np.isin(t_all, [2, 3])
        res[f"clf_only[{cname}]"] = metrics(t_all != 0)
        res[f"clf->filter[{cname}]"] = metrics((trgb_a & rgb_s) | (tir_a & ir_s))
        res[f"filter->clf[{cname}]"] = metrics(t_flt != 0)
    return res


def main():
    r8 = train_robust8()
    sa32_raw = joblib.load(SA32)
    sa32 = ({"model": sa32_raw["model"], "features": sa32_raw.get("features"), "feat_key": "f32"}
            if isinstance(sa32_raw, dict) else {"model": sa32_raw, "features": None, "feat_key": "f32"})
    robust6 = joblib.load(ROBUST6_JBL); robust6["feat_key"] = "f8"
    classifiers = {"sa32": sa32, "robust6": robust6,
                   f"robust8@{r8['tau']:.2f}": r8,
                   "robust8@0.15": {**r8, "tau": 0.15},
                   "robust8@0.20": {**r8, "tau": 0.20}}

    order = ["antiuav", "svanstrom", "svanstrom_gray", "video_drone", "rgb_confuser", "video_confuser"]
    all_res, scorecard = {}, {}
    lines = ["# Routing-Classifier Full-Pipeline Comparison",
             f"{time.strftime('%Y-%m-%d %H:%M')} | robust8 tau={r8['tau']}\n"]
    for name in order:
        pkl = OUT / "cache" / f"{name}.pkl"
        if not pkl.exists():
            continue
        d = pickle.load(open(pkl, "rb")); meta = d["meta"]
        res = ablate(meta, d["rows"], d["F8"], d["F32"], classifiers)
        all_res[name] = res; scorecard[name] = (meta, res)
        lines.append(f"\n## {name} (n={meta['n']}, rule={meta['rule']}, drones={meta['has_drones']})\n")
        lines.append("| cell | TP | FP | FN | P | R | F1 | fire |\n|---|---|---|---|---|---|---|---|"
                     if meta["has_drones"] else "| cell | FP | fire_rate |\n|---|---|---|")
        for cell, m in res.items():
            if meta["has_drones"]:
                lines.append(f"| {cell} | {m['tp']} | {m['fp']} | {m['fn']} | {m.get('precision')} | {m.get('recall')} | {m.get('f1')} | {m['fire_rate']} |")
            else:
                lines.append(f"| {cell} | {m['fp']} | {m['fire_rate']} |")

    # head-to-head scorecard (clf->filter) — averaged within surface group
    GROUPS = {"thermal": ["antiuav", "svanstrom"], "graydrone": ["svanstrom_gray", "video_drone"],
              "confuser": ["rgb_confuser", "video_confuser"]}
    def grp(cellname, g, key):
        vals = [scorecard[s][1][cellname][key] for s in GROUPS[g]
                if s in scorecard and cellname in scorecard[s][1] and scorecard[s][1][cellname].get(key) is not None]
        return float(np.mean(vals)) if vals else float("nan")
    lines.append("\n## SCORECARD — clf->filter cell (production cascade)\n")
    lines.append("| router | thermal drone F1 | GRAYSCALE drone recall | confuser fire-rate |\n|---|---|---|---|")
    print("\n== SCORECARD (clf->filter): thermal_F1 | GRAY_drone_R | confuser_fire ==")
    for cname in classifiers:
        cell = f"clf->filter[{cname}]"
        tf1, gr, cf = grp(cell, "thermal", "f1"), grp(cell, "graydrone", "recall"), grp(cell, "confuser", "fire_rate")
        print(f"  {cname:<14} {tf1:.3f}  {gr:.3f}  {cf:.3f}")
        lines.append(f"| {cname} | {tf1:.3f} | {gr:.3f} | {cf:.3f} |")
    # also per-surface graydrone recall (the split matters)
    lines.append("\n## GRAYSCALE-drone recall per surface (clf->filter) — svanstrom_gray vs video_drone\n")
    lines.append("| router | svanstrom_gray R | video_drone R | svan_gray fire | video_confuser fire |\n|---|---|---|---|---|")
    for cname in classifiers:
        c = f"clf->filter[{cname}]"
        sg = scorecard.get("svanstrom_gray", (None, {}))[1].get(c, {})
        vd = scorecard.get("video_drone", (None, {}))[1].get(c, {})
        vc = scorecard.get("video_confuser", (None, {}))[1].get(c, {})
        lines.append(f"| {cname} | {sg.get('recall')} | {vd.get('recall')} | {sg.get('fire_rate')} | {vc.get('fire_rate')} |")

    json.dump(all_res, open(OUT / "comparison.json", "w"), indent=2)
    (OUT / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nDONE -> {OUT/'comparison.md'}")


if __name__ == "__main__":
    main()
