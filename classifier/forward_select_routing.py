"""forward_select_routing.py — statistics-gated greedy forward feature selection
for the trust ROUTING classifier, judged on the goal metric.

GOAL (win-condition): beat robust6 on GRAYSCALE macro-F1 (+ grayscale trust_rgb recall)
WITHOUT regressing overall/thermal macro-F1. Procedure is thesis-documented.

Method:
  1. Candidate pool = all fusion features MINUS robust6, MINUS label-tautological detection/
     count flags, MINUS leaky features (leakage_ratio > LEAK_MAX from feature_stats_ranked.csv).
     -> selection only considers legitimately-additive, low-leakage features (statistics-gated).
  2. Greedy forward: start at robust6; each round add the candidate that most improves
     held-out OVERALL macro-F1 (grouped split by sequence -> no scene leakage).
  3. Track at every step: overall / thermal / grayscale macro-F1 + grayscale trust_rgb recall
     + the leakage of the added feature.
  4. Winner = the step with best grayscale F1 s.t. thermal F1 >= robust6 (no-regression);
     bootstrap-CI the winner's grayscale-F1 gain vs robust6.

  py classifier/forward_select_routing.py            # ~5-8 min, CPU only (n_jobs=1)
  py classifier/forward_select_routing.py --rounds 8 --subsample 22000
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parent.parent
CSV = REPO / "models/routers/optimal_v1/fusion_dataset_full56.csv"
RANKED = REPO / "models/routers/optimal_v1/feature_stats_ranked.csv"
OUT = REPO / "models/routers/routing_robust"
IMG = REPO / "docs/analysis/images"
LABELS = [0, 1, 2, 3]
THERMAL_SRC = {"antiuav", "svanstrom"}
SEQ_RE = re.compile(r"^(.+?)(?:_f\d+|_frame\d+|_\d{4,})(?:_visible|_infrared)?$", re.I)
META = {"trust_label", "stem", "source", "_seq"}
ROBUST6 = ["rgb_max_conf", "ir_max_conf", "rgb_best_log_bbox_area",
           "ir_best_log_bbox_area", "rgb_best_aspect_ratio", "ir_best_aspect_ratio"]
# label-tautological: the trust label is DERIVED from per-modality detection presence
TAUTOLOGICAL = {"neither_detect", "both_detect", "ir_detected", "rgb_detected",
                "rgb_only_detect", "ir_only_detect", "ir_n_dets", "rgb_n_dets",
                "total_dets", "det_count_diff"}
LEAK_MAX = 1.0


def seq_id(stem, src):
    m = SEQ_RE.match(str(stem)); base = m.group(1).rstrip("_") if m else str(stem)
    return f"{src}::{base}"


def fit_eval(Xtr, ytr, Xte, yte):
    m = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, objective="multi:softprob", num_class=4,
                      eval_metric="mlogloss", tree_method="hist", random_state=42, n_jobs=1)
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def f1m(y, yp):
    return f1_score(y, yp, labels=LABELS, average="macro", zero_division=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--subsample", type=int, default=22000)
    ap.add_argument("--artifact-min", type=float, default=0.55,
                    help="drop candidates whose mean WITHIN-source AUROC < this (clip-ID artifacts)")
    ap.add_argument("--free-only", action="store_true",
                    help="exclude expensive image-read scene statistics (production-deployable pool)")
    ap.add_argument("--mask-diffs", action="store_true",
                    help="neutralize cross-modal *_diff features on non-both-detect rows so they "
                         "measure ONLY genuine box agreement, not the (0,0)-default presence-leak")
    ap.add_argument("--out-tag", default="", help="suffix for output filenames")
    args = ap.parse_args()
    # expensive = needs a pixel read + OpenCV per frame (vs free detector-output/box features).
    # NOTE: tokens are substrings, so "contrast"/"bg_delta" also catch the derived *_diff
    # features (contrast_diff, bg_delta_diff) — those need the same pixel-stat pass and are NOT free.
    EXPENSIVE = ("img_mean", "img_std", "img_entropy", "blurriness", "edge_density",
                 "sky_ground", "dynamic_range", "contrast", "bg_delta",
                 "scene_entropy", "brightness")
    OUT.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(CSV)
    feats_all = [c for c in df.columns if c not in META]
    X = df[feats_all].to_numpy(float)
    X[~np.isfinite(X)] = np.nan
    med = np.nanmedian(X, 0); nm = np.isnan(X); X[nm] = np.take(med, np.where(nm)[1])
    df[feats_all] = X
    df["_seq"] = [seq_id(s, c) for s, c in zip(df["stem"], df["source"])]
    df["regime"] = np.where(df["source"].isin(THERMAL_SRC), "thermal", "grayscale")

    # ── presence-leak fix ────────────────────────────────────────────
    # Per-modality box features default to 0.0 when that modality has NO detection
    # (generate_lean19_data.build_row). The cross-modal *_diff features therefore blow up
    # to |x-0| whenever exactly one modality fires -> they smuggle the EXCLUDED tautological
    # presence flag (corr(pos_euclidean_diff, xor-detect)=0.93 on grayscale). Verified that the
    # GENUINE agreement signal lives only in both-detect rows (AUROC 0.72-0.87, trust_both vs
    # trust_rgb). Neutralize the diff on non-both-detect rows -> median-of-both-detect, so the
    # feature carries ONLY real geometry agreement; presence stays in *_max_conf (already in robust6).
    if args.mask_diffs:
        both = (df["rgb_max_conf"] > 0) & (df["ir_max_conf"] > 0)
        DIFFS = [f for f in ("pos_euclidean_diff", "area_diff", "aspect_ratio_diff") if f in df.columns]
        for f in DIFFS:
            df.loc[~both, f] = df.loc[both, f].median()
        print(f"[mask-diffs] neutralized {DIFFS} on {int((~both).sum()):,} non-both-detect rows "
              f"-> median-of-both-detect (removes (0,0)-default presence-leak)")

    if args.subsample and len(df) > args.subsample:
        frac = args.subsample / len(df)
        df = df.groupby("trust_label", group_keys=False).sample(frac=frac, random_state=42).reset_index(drop=True)
    print(f"{len(df):,} rows (subsampled) | thermal {int((df.regime=='thermal').sum())} / "
          f"gray {int((df.regime=='grayscale').sum())}")

    leak = {}
    if RANKED.exists():
        r = pd.read_csv(RANKED); leak = dict(zip(r.feature, r.leakage_ratio))

    # statistics-gated candidate pool: drop robust6, tautological flags, leaky features
    pool = [f for f in feats_all if f not in ROBUST6 and f not in TAUTOLOGICAL
            and leak.get(f, 0.0) <= LEAK_MAX]

    # ARTIFACT GATE: drop features whose discrimination is cross-source only (clip-ID proxies
    # like rgb_blurriness — pooled AUROC high but within-clip ~0.5; leakage_ratio misses these).
    yfull = (df["trust_label"] >= 1).astype(int)
    def within_auroc(feat):
        accs = []
        for _, g in df.groupby("source"):
            yy = (g["trust_label"] >= 1).astype(int).to_numpy()
            if yy.sum() >= 15 and (yy == 0).sum() >= 15:
                accs.append(roc_auc_score(yy, g[feat]))
        return float(np.mean(accs)) if accs else 0.5
    artifact = sorted([f for f in pool if within_auroc(f) < args.artifact_min])
    pool = [f for f in pool if f not in artifact]
    if args.free_only:
        expensive = sorted([f for f in pool if any(h in f for h in EXPENSIVE)])
        pool = [f for f in pool if f not in expensive]
        print(f"  free-only: also dropped {len(expensive)} expensive scene features: {expensive}")
    excluded = [f for f in feats_all if f not in ROBUST6 and f not in pool]
    print(f"candidate pool: {len(pool)} | artifact-gate dropped {len(artifact)}: {artifact}\n")

    tr, te = next(GroupShuffleSplit(1, test_size=0.30, random_state=42)
                  .split(df, df["trust_label"], df["_seq"]))
    yte = df.iloc[te]["trust_label"].values
    reg_te = df.iloc[te]["regime"].values
    th, gr = reg_te == "thermal", reg_te == "grayscale"

    def evaluate(feats):
        yp = fit_eval(df.iloc[tr][feats].values, df.iloc[tr]["trust_label"].values,
                      df.iloc[te][feats].values, yte)
        trgb_gray = precision_recall_fscore_support(
            yte[gr], yp[gr], labels=[1], zero_division=0)[1][0] if gr.sum() else float("nan")
        return dict(overall=f1m(yte, yp), thermal=f1m(yte[th], yp[th]),
                    grayscale=f1m(yte[gr], yp[gr]), gray_trust_rgb_R=float(trgb_gray)), yp

    sel = list(ROBUST6)
    base, _ = evaluate(sel)
    print(f"{'step':<22}{'overall':>9}{'thermal':>9}{'gray':>8}{'gray_tRGB_R':>13}{'leak':>8}")
    print(f"{'robust6 (base)':<22}{base['overall']:>9.4f}{base['thermal']:>9.4f}"
          f"{base['grayscale']:>8.4f}{base['gray_trust_rgb_R']:>13.4f}{'-':>8}")
    steps = [{"feature": "robust6", "n": 6, **base, "leakage": None}]
    remaining = list(pool)
    for rnd in range(args.rounds):
        best_f, best_score, best_metrics = None, -1, None
        for c in remaining:
            mtr, _ = evaluate(sel + [c])
            if mtr["overall"] > best_score:
                best_score, best_f, best_metrics = mtr["overall"], c, mtr
        sel.append(best_f); remaining.remove(best_f)
        lk = leak.get(best_f, float("nan"))
        steps.append({"feature": best_f, "n": len(sel), **best_metrics, "leakage": lk})
        print(f"+{best_f:<21}{best_metrics['overall']:>9.4f}{best_metrics['thermal']:>9.4f}"
              f"{best_metrics['grayscale']:>8.4f}{best_metrics['gray_trust_rgb_R']:>13.4f}{lk:>8.3f}")

    # winner: best grayscale F1 with NO thermal regression vs robust6
    cands = [s for s in steps if s["thermal"] >= base["thermal"] - 1e-6]
    winner = max(cands, key=lambda s: s["grayscale"]) if cands else steps[0]
    print(f"\nWINNER (best gray F1, thermal not regressed): "
          f"{'robust6' if winner['feature']=='robust6' else 'robust6+...+'+winner['feature']} "
          f"(n={winner['n']}) gray {winner['grayscale']:.4f} vs robust6 {base['grayscale']:.4f} "
          f"(delta {winner['grayscale']-base['grayscale']:+.4f})")

    # bootstrap CI on winner's grayscale-F1 gain vs robust6 (paired on the gray test set)
    if winner["feature"] != "robust6":
        wfeats = ROBUST6 + [s["feature"] for s in steps[1:steps.index(winner) + 1]]
        _, yp_w = evaluate(wfeats); _, yp_b = evaluate(ROBUST6)
        yg, wg, bg = yte[gr], yp_w[gr], yp_b[gr]
        rng = np.random.RandomState(0); gains = []
        idx = np.arange(len(yg))
        for _ in range(1000):
            b = rng.choice(idx, len(idx), replace=True)
            gains.append(f1m(yg[b], wg[b]) - f1m(yg[b], bg[b]))
        lo, hi = np.percentile(gains, [2.5, 97.5])
        print(f"bootstrap 95% CI of grayscale-F1 gain: [{lo:+.4f}, {hi:+.4f}]  "
              f"({'SIGNIFICANT' if lo > 0 else 'not significant (CI includes 0)'})")
        winner["gray_gain_ci"] = [float(lo), float(hi)]

    # ── PLOT: F1 vs n_features ───────────────────────────────────────
    ns = [s["n"] for s in steps]
    plt.figure(figsize=(9, 5.5))
    for k, c in [("overall", "#1f77b4"), ("thermal", "#d62728"), ("grayscale", "#2ca02c")]:
        plt.plot(ns, [s[k] for s in steps], "o-", color=c, label=k)
    plt.axhline(base["grayscale"], color="#2ca02c", ls=":", lw=1, alpha=0.6)
    for s in steps[1:]:
        plt.annotate(s["feature"], (s["n"], s["grayscale"]), fontsize=6, rotation=30, alpha=0.7)
    plt.xlabel("n features (robust6 = 6, forward-added)"); plt.ylabel("macro-F1")
    plt.title("Forward selection from robust6 — does adding features beat it on grayscale?")
    plt.legend(); plt.tight_layout()
    plt.savefig(IMG / f"forward_select_routing{args.out_tag}.png", dpi=160); plt.close()

    json.dump({"steps": steps, "winner": winner, "artifact_dropped": artifact, "excluded_from_pool": excluded},
              open(OUT / f"forward_select_routing{args.out_tag}.json", "w"), indent=2)
    print(f"\nsaved forward_select_routing{args.out_tag}.json + docs/analysis/images/forward_select_routing{args.out_tag}.png")


if __name__ == "__main__":
    main()
