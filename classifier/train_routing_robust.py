"""train_routing_robust.py — Phase 1 trust-ROUTING trainer.

Goal (the real one): the trust classifier must ROUTE correctly —
  drone in RGB        -> trust_rgb (1)   [or trust_both]
  drone in IR/gray    -> trust_ir  (2)   [or trust_both]
  drone in both       -> trust_both (3)
  neither             -> reject (0)
... and be robust ACROSS REGIMES (thermal-IR vs grayscale-IR fallback).

So we report PER-CLASS P/R/F1 for the 4 trust classes, split by regime, NOT just
binary trust-vs-reject. The headline metric is trust_ir recall on the GRAYSCALE regime
(robust6's known failure) and trust_rgb routing quality.

Feature sets are auto-selected by which columns exist, so the SAME script serves:
  Phase 1a (now)  : robust6, robust6+conf_sum            (lean19 data; conf_sum derived)
  Phase 1b (later): + rgb_verifier_pdrone / ir_verifier_pdrone   (verifier-augmented CSV)

  py classifier/train_routing_robust.py [--csv PATH]
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

import numpy as np, pandas as pd
import joblib
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO / "models/routers/lean_ft4/fusion_dataset_lean19.csv"
OUT = REPO / "models/routers/routing_robust"
LABELS = [0, 1, 2, 3]
LABEL_NAMES = {0: "reject", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}
THERMAL_SRC = {"antiuav", "svanstrom"}
SEQ_RE = re.compile(r"^(.+?)(?:_f\d+|_frame\d+|_\d{4,})(?:_visible|_infrared)?$", re.I)

ROBUST6 = ["rgb_max_conf", "ir_max_conf", "rgb_best_log_bbox_area",
           "ir_best_log_bbox_area", "rgb_best_aspect_ratio", "ir_best_aspect_ratio"]


def seq_id(stem, src):
    m = SEQ_RE.match(str(stem)); base = m.group(1).rstrip("_") if m else str(stem)
    return f"{src}::{base}"


def feature_sets(cols):
    """Build candidate sets from columns that actually exist.

    The grayscale trust_rgb hole is an OPERATING-POINT problem: rgb_mean_conf separates
    drone-from-confuser at AUROC 0.934 on the hole, but the optimal threshold differs by regime
    (thermal ~0.73 / grayscale ~0.38). A free `is_grayscale` flag (1 bit, known at inference) lets
    the tree learn regime-conditional splits on it. These sets isolate feature-alone vs +regime-flag.
    See docs/analysis/2026-06-05_routing_free_features_exhausted.md."""
    s = {"robust6": list(ROBUST6)}
    if "rgb_mean_conf" in cols:
        s["robust6+rgb_mean_conf"] = ROBUST6 + ["rgb_mean_conf"]
    if "is_grayscale" in cols:
        s["robust6+is_grayscale"] = ROBUST6 + ["is_grayscale"]
    if "rgb_mean_conf" in cols and "is_grayscale" in cols:
        s["robust6+rgb_mean_conf+is_grayscale"] = ROBUST6 + ["rgb_mean_conf", "is_grayscale"]
    # Phase-1b verifier hooks (only if the re-mined CSV has them)
    if "rgb_verifier_pdrone" in cols:
        s["robust6+rgb_verif"] = ROBUST6 + ["rgb_verifier_pdrone"]
        if "is_grayscale" in cols:
            s["robust6+rgb_verif+is_grayscale"] = ROBUST6 + ["rgb_verifier_pdrone", "is_grayscale"]
    return {k: v for k, v in s.items() if all(f in cols for f in v)}


def per_class(y, yp):
    p, r, f, s = precision_recall_fscore_support(y, yp, labels=LABELS, zero_division=0)
    return {LABEL_NAMES[c]: dict(P=round(p[i], 4), R=round(r[i], 4), F1=round(f[i], 4), n=int(s[i]))
            for i, c in enumerate(LABELS)}


def macro(y, yp):
    p, r, f, _ = precision_recall_fscore_support(y, yp, labels=LABELS, average="macro", zero_division=0)
    return dict(P=round(p, 4), R=round(r, 4), F1=round(f, 4), acc=round(accuracy_score(y, yp), 4))


def fit_predict(df, tr, te, feats):
    m = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, objective="multi:softprob", num_class=4,
                      eval_metric="mlogloss", tree_method="hist", random_state=42, n_jobs=1)
    m.fit(df.iloc[tr][feats].values, df.iloc[tr]["trust_label"].values)
    return m, m.predict(df.iloc[te][feats].values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    if "conf_sum" not in df.columns:                       # Phase 1a: derive it
        df["conf_sum"] = df["rgb_max_conf"] + df["ir_max_conf"]
        print("[note] conf_sum derived = rgb_max_conf + ir_max_conf")
    df["_seq"] = [seq_id(s, src) for s, src in zip(df["stem"], df["source"])]
    df["regime"] = np.where(df["source"].isin(THERMAL_SRC), "thermal", "grayscale")
    df["is_grayscale"] = (df["regime"] == "grayscale").astype(int)   # free regime flag (1 bit)

    print(f"{len(df):,} rows | {df['_seq'].nunique()} seqs | "
          f"thermal {int((df.regime=='thermal').sum())} / grayscale {int((df.regime=='grayscale').sum())}")
    print("trust labels overall:", {LABEL_NAMES[k]: int((df.trust_label==k).sum()) for k in LABELS})
    for rg in ("thermal", "grayscale"):
        d = df[df.regime == rg]
        print(f"  {rg:<9}:", {LABEL_NAMES[k]: int((d.trust_label==k).sum()) for k in LABELS})

    tr, te = next(GroupShuffleSplit(1, test_size=0.25, random_state=42)
                  .split(df, df["trust_label"], df["_seq"]))
    teidx = df.index[te]
    reg_te = df.loc[teidx, "regime"].values

    yte = df.iloc[te]["trust_label"].values
    sets = feature_sets(df.columns)
    results, models, preds = {}, {}, {}
    for tag, feats in sets.items():
        m, yp = fit_predict(df, tr, te, feats)
        rec = {"features": feats, "n_features": len(feats),
               "overall_macro": macro(yte, yp), "overall_per_class": per_class(yte, yp),
               "by_regime": {}}
        for rg in ("thermal", "grayscale"):
            mask = reg_te == rg
            if mask.sum():
                rec["by_regime"][rg] = {"macro": macro(yte[mask], yp[mask]),
                                        "per_class": per_class(yte[mask], yp[mask])}
        results[tag], models[tag], preds[tag] = rec, m, yp

    # ── report ──────────────────────────────────────────────────────
    print("\n" + "=" * 96)
    print("OVERALL macro (the routing decision quality across all classes)")
    print(f"{'feature set':<30}{'feats':>6}{'P':>9}{'R':>9}{'F1':>9}{'acc':>9}")
    for tag, r in results.items():
        mo = r["overall_macro"]
        print(f"{tag:<30}{r['n_features']:>6}{mo['P']:>9.4f}{mo['R']:>9.4f}{mo['F1']:>9.4f}{mo['acc']:>9.4f}")

    # ── THE GOAL METRIC: grayscale trust_rgb recall (robust6's hole) ─────────────────
    # plus thermal macro-F1 (must NOT regress) and gray trust_rgb precision (don't trade recall for FPs)
    print("\nKEY: grayscale TRUST_RGB recall (the hole) — does the free regime flag let the model use it?")
    print(f"{'feature set':<38}{'gray_tRGB_R':>12}{'gray_tRGB_P':>12}{'gray_F1m':>10}{'therm_F1m':>11}")
    for tag, r in results.items():
        g = r["by_regime"].get("grayscale", {}); t = r["by_regime"].get("thermal", {})
        gt = g.get("per_class", {}).get("trust_rgb", {})
        print(f"{tag:<38}{gt.get('R', float('nan')):>12.4f}{gt.get('P', float('nan')):>12.4f}"
              f"{g.get('macro', {}).get('F1', float('nan')):>10.4f}{t.get('macro', {}).get('F1', float('nan')):>11.4f}")

    # ── bootstrap CI: grayscale trust_rgb recall gain of each set vs robust6 (paired) ──
    gmask = reg_te == "grayscale"
    yg = yte[gmask]
    def trgb_recall(y, yp):
        m = (y == 1)
        return float((yp[m] == 1).mean()) if m.sum() else float("nan")
    base_pred = preds["robust6"][gmask]
    print(f"\nbootstrap 95% CI of grayscale trust_rgb RECALL gain vs robust6 (1000 resamples, paired):")
    rng = np.random.RandomState(0); idx = np.arange(len(yg))
    for tag in results:
        if tag == "robust6":
            continue
        wp = preds[tag][gmask]
        gains = []
        for _ in range(1000):
            b = rng.choice(idx, len(idx), replace=True)
            gains.append(trgb_recall(yg[b], wp[b]) - trgb_recall(yg[b], base_pred[b]))
        lo, hi = np.nanpercentile(gains, [2.5, 97.5])
        sig = "SIGNIFICANT" if lo > 0 else "not sig (CI incl. 0)"
        results[tag]["gray_trgb_recall_gain_ci"] = [float(lo), float(hi)]
        print(f"  {tag:<38} [{lo:+.4f}, {hi:+.4f}]  {sig}")

    # ── plot (always-plot rule) ──────────────────────────────────────
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tags = list(results)
    gr = [results[t]["by_regime"].get("grayscale", {}).get("per_class", {}).get("trust_rgb", {}).get("R", np.nan) for t in tags]
    th = [results[t]["by_regime"].get("thermal", {}).get("macro", {}).get("F1", np.nan) for t in tags]
    x = np.arange(len(tags)); w = 0.38
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x - w/2, gr, w, color="#d62728", label="grayscale trust_rgb RECALL (the hole)")
    ax.bar(x + w/2, th, w, color="#1f77b4", alpha=.6, label="thermal macro-F1 (must not regress)")
    ax.axhline(gr[0], color="#d62728", ls=":", alpha=.6)
    for xi, v in zip(x - w/2, gr):
        ax.annotate(f"{v:.2f}", (xi, v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(tags, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("score"); ax.set_ylim(0, 1)
    ax.set_title("Does a free is_grayscale regime flag let the model exploit rgb_mean_conf\n"
                 "to close the grayscale trust_rgb recall hole?")
    ax.legend(); plt.tight_layout()
    IMG = REPO / "docs/analysis/images"; IMG.mkdir(parents=True, exist_ok=True)
    plt.savefig(IMG / "routing_regime_flag_test.png", dpi=160); plt.close()

    # ── save: best by grayscale trust_rgb recall s.t. thermal macro-F1 not regressed vs robust6 ──
    base_th = results["robust6"]["by_regime"]["thermal"]["macro"]["F1"]
    def gtrgb_R(t):
        return results[t]["by_regime"].get("grayscale", {}).get("per_class", {}).get("trust_rgb", {}).get("R", -1)
    cands = [t for t in results if results[t]["by_regime"].get("thermal", {}).get("macro", {}).get("F1", 0) >= base_th - 1e-3]
    best = max(cands or list(results), key=gtrgb_R)
    json.dump(results, open(OUT / "routing_regime_flag.json", "w"), indent=2)
    joblib.dump({"model": models[best], "features": sets[best], "tag": best}, OUT / "trust_routing_best.joblib")
    print(f"\nWINNER (best gray trust_rgb recall, thermal not regressed): {best} "
          f"(gray_tRGB_R={gtrgb_R(best):.4f} vs robust6 {gtrgb_R('robust6'):.4f})")
    print("saved routing_regime_flag.json + trust_routing_best.joblib + images/routing_regime_flag_test.png")


if __name__ == "__main__":
    main()
