"""ir_failopen_viability.py — is the OOD-abstain ('fail-open') rule that fixed the RGB
mlp_v5 recall drop also VIABLE for the IR ALIGNED verifier? (CPU, offline)

Mirrors eval/diagnose_mlp_recall_drop.py + test_failopen_verifier.py + eval_failopen_prepost.py
but for the aligned IR verifier on two paths:
  (a) native thermal  — mlp_aligned.pt      thr 0.05   surfaces: ir_dset_final, cbam (rule iou)
  (b) grayscale       — mlp_aligned_gray.pt thr 0.05   surface : gray_svan          (rule iop)

For each drone surface: match dets to GT, split GT-matched REAL drones into KEPT (P>=thr) vs
FALSELY-VETOED (P<thr). Characterize vetoed-vs-kept (LDA/ANOVA/AUROC). Build an
OOD-from-confuser score (kNN dist k=5 on the confuser feature distribution) and test whether
vetoed-drones (want HIGH -> release) separate from MLP-vetoed-confusers (want LOW -> keep
vetoed). Sweep tau by acceptable leak; report recovery; PRE/POST P/R/F1 (bare/aligned/+failopen).

Confuser distributions:
  grayscale path  -> gray_confuser.pkl   (clean grayscale confusers, NO drones)
  thermal  path   -> cbam non-drone, MLP-vetoed detections (thermal-confuser PROXY); a clean
                     native-thermal confuser cache does not exist (caveat). rgb_confuser as
                     cross-modal fallback only if the proxy is too small.

  py eval/ir_failopen_viability.py
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "eval"))
from mri.stats import anova_f, per_feature_auroc, lda_separability
from metrics import score_detections, compute_prf
from eval_v4_vs_patch import MLPv4Verifier

CACHE = REPO / "eval/results/_offline_pipeline/cache"
IMG = REPO / "docs/analysis/images"; IMG.mkdir(parents=True, exist_ok=True)
CK_THERMAL = REPO / "models/verifiers/ir_aligned/mlp_aligned.pt"
CK_GRAY = REPO / "models/verifiers/ir_aligned/mlp_aligned_gray.pt"
THR = 0.05  # aligned verifier operating threshold (both paths)

# 517-D layout: p3[0:256], p5[256:512], meta[512:517]=conf,log_area,aspect,rel_cx,rel_cy
META = {512: "conf", 513: "log_area", 514: "aspect", 515: "rel_cx", 516: "rel_cy"}
def fname(i): return META.get(i, f"p3_{i}" if i < 256 else f"p5_{i-256}")


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0., x2-x1)*max(0., y2-y1); ua = (a[2]-a[0])*(a[3]-a[1]); ub = (b[2]-b[0])*(b[3]-b[1])
    return i/(ua+ub-i) if ua+ub-i > 0 else 0.
def iop(d, g):
    x1, y1 = max(d[0], g[0]), max(d[1], g[1]); x2, y2 = min(d[2], g[2]), min(d[3], g[3])
    i = max(0., x2-x1)*max(0., y2-y1); da = (d[2]-d[0])*(d[3]-d[1])
    return i/da if da > 0 else 0.


def load(name):
    return pickle.load(open(CACHE / f"{name}.pkl", "rb"))


# ── 1. split GT-matched real drones into KEPT vs FALSELY-VETOED ────────────────
def split_drones(name, rule, mlp):
    """Return (X_kept, X_vetoed) feature arrays for GT-matched real drones."""
    d = load(name); match = iop if rule == "iop" else iou
    K, V = [], []
    for fr in d["frames"]:
        if len(fr["feats"]) == 0 or len(fr["gt_boxes"]) == 0:
            continue
        p = mlp.predict_drone_probs(fr["feats"])
        for i, box in enumerate(fr["boxes"]):
            if max((match(box, g) for g in fr["gt_boxes"]), default=0) >= 0.5:
                (K if p[i] >= THR else V).append(fr["feats"][i])
    return np.array(K), np.array(V)


def characterize(name, K, V):
    nk, nv = len(K), len(V)
    n = nk + nv
    print(f"\n=== {name} ===")
    print(f"  real drones matched: {n}  | KEPT {nk}  FALSELY-VETOED {nv}  "
          f"(recall loss {nv/max(n,1):.1%})")
    out = {"name": name, "n": n, "kept": nk, "vetoed": nv,
           "recall_loss": nv/max(n, 1)}
    if nv < 5 or nk < 5:
        print("  too few in a group for stats")
        # still report size deltas if any vetoed
        if nv and nk:
            for mi, lbl in ((512, "conf"), (513, "log_area")):
                out[f"{lbl}_kept"] = float(K[:, mi].mean()); out[f"{lbl}_vetoed"] = float(V[:, mi].mean())
        return out
    X = np.vstack([K, V]); y = np.array([1]*nk + [0]*nv)
    F = anova_f(X, y); auroc = per_feature_auroc(X, y); _, lda_acc, _ = lda_separability(X, y)
    top = np.argsort(F)[::-1][:8]
    print(f"  LDA separability kept-vs-vetoed: {lda_acc:.3f}  (overfit caveat: 517 feats, n={n})")
    print(f"  top discriminating features (ANOVA F / AUROC honest signal):")
    for i in top:
        print(f"    {fname(int(i)):<12} F={F[i]:>9.1f}  AUROC={auroc[i]:.3f}")
    out["lda_acc"] = float(lda_acc)
    out["top_auroc_max"] = float(auroc.max())
    for mi, lbl in ((512, "conf"), (513, "log_area"), (514, "aspect")):
        vk, vv = float(K[:, mi].mean()), float(V[:, mi].mean())
        print(f"  {lbl}: kept mean={vk:.3f}  vetoed mean={vv:.3f}  (Δ={vk-vv:+.3f})")
        out[f"{lbl}_kept"] = vk; out[f"{lbl}_vetoed"] = vv
    return out


# ── 2. build confuser feature distribution ────────────────────────────────────
def gray_confuser_feats():
    cf = load("gray_confuser")
    F = [f for fr in cf["frames"] for f in fr["feats"]]
    return np.array(F) if F else np.zeros((0, 517)), "gray_confuser (clean grayscale, no drones)"


def thermal_confuser_feats(mlp):
    """Native-thermal confuser PROXY: cbam detections that DON'T match GT (non-drone) AND are
    MLP-vetoed (the verifier's notion of confuser). No clean thermal-confuser cache exists."""
    d = load("cbam"); F = []
    for fr in d["frames"]:
        if len(fr["feats"]) == 0:
            continue
        gt = fr["gt_boxes"]
        p = mlp.predict_drone_probs(fr["feats"])
        for i, box in enumerate(fr["boxes"]):
            matched = max((iou(box, g) for g in gt), default=0) >= 0.5
            if (not matched) and p[i] < THR:
                F.append(fr["feats"][i])
    return np.array(F) if F else np.zeros((0, 517)), "cbam non-drone MLP-vetoed (thermal PROXY)"


def vetoed_confusers(name, mlp, rule="iou", has_gt=True):
    """MLP-vetoed confuser detections on a confuser surface (for leak calibration)."""
    d = load(name); V = []
    for fr in d["frames"]:
        if len(fr["feats"]) == 0:
            continue
        p = mlp.predict_drone_probs(fr["feats"])
        gt = fr["gt_boxes"] if has_gt else []
        for i, box in enumerate(fr["boxes"]):
            if has_gt and max((iou(box, g) for g in gt), default=0) >= 0.5:
                continue  # skip real drones
            if p[i] < THR:
                V.append(fr["feats"][i])
    return np.array(V) if V else np.zeros((0, 517))


# ── 3. OOD separability + fail-open sweep ──────────────────────────────────────
def build_ood(Cf):
    mu, sd = Cf.mean(0), Cf.std(0) + 1e-6
    z = lambda x: (x - mu) / sd
    nn = NearestNeighbors(n_neighbors=min(5, len(Cf))).fit(z(Cf))
    return (lambda X: nn.kneighbors(z(X))[0].mean(1) if len(X) else np.zeros(0)), z


def ood_analysis(tag, V_drones, V_conf, ood):
    ood_V = ood(V_drones)       # vetoed drones — want HIGH
    ood_C = ood(V_conf)         # vetoed confusers — want LOW
    print(f"\n  [{tag}] OOD-from-confuser  vetoed-drones median={np.median(ood_V):.2f}  "
          f"vetoed-confusers median={np.median(ood_C):.2f}  (n_dr={len(ood_V)} n_cf={len(ood_C)})")
    # AUROC of OOD score separating drones(1) from confusers(0)
    auroc = None
    if len(ood_V) and len(ood_C):
        sc = np.concatenate([ood_V, ood_C]); lab = np.array([1]*len(ood_V)+[0]*len(ood_C))
        auroc = float(per_feature_auroc(sc.reshape(-1, 1), lab)[0])
        print(f"  [{tag}] OOD-score separability AUROC (drone-vs-confuser) = {auroc:.3f}")
    return ood_V, ood_C, auroc


def sweep_recovery(ood_V, ood_C, leaks=(0.02, 0.05, 0.10)):
    """For each acceptable leak, tau = (1-leak) quantile of confuser OOD; report drone recovery."""
    rows = []
    for lk in leaks:
        if len(ood_C):
            tau = float(np.quantile(ood_C, 1 - lk))
        else:
            tau = float(np.inf)
        rec = float((ood_V > tau).mean()) if len(ood_V) else 0.0
        actual_leak = float((ood_C > tau).mean()) if len(ood_C) else 0.0
        rows.append((lk, tau, rec, actual_leak))
        print(f"  leak<={lk:.0%}: tau={tau:.2f}  recovered drones={rec:.1%}  (actual leak={actual_leak:.1%})")
    return rows


# ── 4. PRE vs POST P/R/F1 on a drone surface ──────────────────────────────────
def prepost(name, rule, mlp, ood, tau):
    d = load(name)
    agg = {v: {"tp": 0, "fp": 0, "fn": 0} for v in ("bare", "aligned", "aligned+failopen")}
    for fr in d["frames"]:
        n = len(fr["feats"]); gt = [tuple(g) for g in fr["gt_boxes"]]
        boxes = [tuple(b) for b in fr["boxes"]]; confs = [float(c) for c in fr["confs"]]
        if n:
            p = mlp.predict_drone_probs(fr["feats"]); o = ood(fr["feats"])
        keeps = {
            "bare": np.ones(n, bool) if n else np.zeros(0, bool),
            "aligned": (p >= THR) if n else np.zeros(0, bool),
            "aligned+failopen": ((p >= THR) | (o > tau)) if n else np.zeros(0, bool),
        }
        for v, keep in keeps.items():
            kept = [(boxes[i], confs[i]) for i in range(n) if keep[i]]
            t, f, fn = score_detections(kept, gt, rule=rule, iou_thr=0.5, iop_thr=0.5)
            agg[v]["tp"] += t; agg[v]["fp"] += f; agg[v]["fn"] += fn
    print(f"\n  PRE/POST {name} ({rule}, tau={tau:.2f}):")
    res = {}
    for v, m in agg.items():
        prf = compute_prf(m["tp"], m["fp"], m["fn"])
        res[v] = prf
        print(f"    {v:<18} P={prf['precision']:.4f} R={prf['recall']:.4f} F1={prf['f1']:.4f}"
              f"  (TP {m['tp']} FP {m['fp']} FN {m['fn']})")
    return res


# ── figures ────────────────────────────────────────────────────────────────────
def fig_ood_hist(tag, ood_V, ood_C, tau):
    if not (len(ood_V) and len(ood_C)):
        return None
    plt.figure(figsize=(7, 4))
    plt.hist(ood_V, bins=30, alpha=0.6, color="blue", density=True, label=f"vetoed DRONES (n={len(ood_V)})")
    plt.hist(ood_C, bins=30, alpha=0.6, color="red", density=True, label=f"vetoed CONFUSERS (n={len(ood_C)})")
    plt.axvline(tau, color="k", ls="--", label=f"fail-open τ={tau:.1f}")
    plt.xlabel("OOD-from-confuser score (kNN dist)"); plt.ylabel("density")
    plt.title(f"IR fail-open separability [{tag}]")
    plt.legend(); plt.tight_layout()
    out = IMG / f"ir_failopen_ood_hist_{tag}.png"; plt.savefig(out, dpi=160); plt.close()
    return out


def fig_tradeoff(tag, ood_V, ood_C):
    if not (len(ood_V) and len(ood_C)):
        return None
    taus = np.linspace(0, max(ood_V.max(), ood_C.max()), 60)
    rec = [(ood_V > t).mean()*100 for t in taus]; leak = [(ood_C > t).mean()*100 for t in taus]
    plt.figure(figsize=(6.5, 5))
    plt.plot(leak, rec, "-o", ms=3)
    plt.xlabel("confuser leak % (released — bad)"); plt.ylabel("vetoed real drones recovered % (good)")
    plt.title(f"IR fail-open trade-off [{tag}]"); plt.grid(alpha=0.3)
    plt.tight_layout(); out = IMG / f"ir_failopen_tradeoff_{tag}.png"; plt.savefig(out, dpi=160); plt.close()
    return out


def fig_pca(tag, K, V, Cf, z):
    if not (len(K) and len(V) and len(Cf)):
        return None
    P = PCA(2).fit(z(np.vstack([K, V, Cf])))
    plt.figure(figsize=(6.5, 5.5))
    for A, c, l in [(Cf, "red", "confusers"), (K, "green", "kept drones"), (V, "blue", "vetoed drones")]:
        Z = P.transform(z(A)); plt.scatter(Z[:, 0], Z[:, 1], s=8, alpha=0.5, c=c, label=l)
    plt.legend(); plt.title(f"IR aligned: kept vs vetoed drones vs confusers [{tag}]")
    plt.tight_layout(); out = IMG / f"ir_failopen_pca_{tag}.png"; plt.savefig(out, dpi=160); plt.close()
    return out


def run_path(tag, ckpt, drone_surfaces, Cf, conf_src, leak_calib):
    """drone_surfaces: list of (name, rule). Cf: confuser feats. leak_calib: vetoed-confuser feats."""
    print(f"\n{'#'*70}\n# PATH: {tag}   ckpt={ckpt.name}   confuser-dist: {conf_src}\n{'#'*70}")
    mlp = MLPv4Verifier(ckpt, device="cpu")
    ood, z = build_ood(Cf)
    summary = {"tag": tag, "ckpt": ckpt.name, "conf_src": conf_src,
               "n_confuser_feats": int(len(Cf)), "surfaces": {}}
    # all vetoed drones across drone surfaces (pooled for OOD separability)
    Vall, K0, V0 = [], None, None
    for name, rule in drone_surfaces:
        K, V = split_drones(name, rule, mlp)
        ch = characterize(name, K, V)
        summary["surfaces"][name] = ch
        if len(V):
            Vall.append(V)
        if K0 is None and len(K) and len(V):
            K0, V0 = K, V  # for PCA on first informative surface
    Vall = np.vstack(Vall) if Vall else np.zeros((0, 517))
    # OOD separability vs the leak-calibration confusers
    ood_V, ood_C, ood_auroc = ood_analysis(tag, Vall, leak_calib, ood)
    summary["ood_auroc"] = ood_auroc
    summary["vetoed_drones_pooled"] = int(len(Vall))
    summary["leak_calib_confusers"] = int(len(leak_calib))
    rows = sweep_recovery(ood_V, ood_C) if len(Vall) else []
    summary["sweep"] = [{"leak": l, "tau": t, "recovered": r, "actual_leak": al} for l, t, r, al in rows]
    # PRE/POST at 5% leak tau
    tau5 = float(np.quantile(ood_C, 0.95)) if len(ood_C) else float(np.inf)
    summary["tau5"] = tau5
    summary["prepost"] = {}
    for name, rule in drone_surfaces:
        if summary["surfaces"][name]["n"] == 0:
            continue
        summary["prepost"][name] = prepost(name, rule, mlp, ood, tau5)
    # figures
    f1 = fig_ood_hist(tag, ood_V, ood_C, tau5)
    f2 = fig_tradeoff(tag, ood_V, ood_C)
    f3 = fig_pca(tag, K0 if K0 is not None else np.zeros((0, 517)),
                 V0 if V0 is not None else np.zeros((0, 517)), Cf, z)
    summary["figures"] = [str(f) for f in (f1, f2, f3) if f]
    return summary


if __name__ == "__main__":
    # --- grayscale path (KEY case) ---
    Cf_gray, src_gray = gray_confuser_feats()
    print(f"gray confuser feats: {len(Cf_gray)}")
    mlp_g = MLPv4Verifier(CK_GRAY, device="cpu")
    leak_gray = vetoed_confusers("gray_confuser", mlp_g, has_gt=False)
    s_gray = run_path("gray", CK_GRAY, [("gray_svan", "iop")], Cf_gray, src_gray, leak_gray)

    # --- native thermal path ---
    mlp_t = MLPv4Verifier(CK_THERMAL, device="cpu")
    Cf_therm, src_therm = thermal_confuser_feats(mlp_t)
    print(f"\nthermal confuser proxy feats: {len(Cf_therm)}")
    if len(Cf_therm) < 30:
        cf = load("rgb_confuser")
        Cf_therm = np.array([f for fr in cf["frames"] for f in fr["feats"]])
        src_therm = "rgb_confuser (CROSS-MODAL FALLBACK — proxy too small)"
        print(f"  falling back to rgb_confuser: {len(Cf_therm)}")
    leak_therm = Cf_therm  # proxy confusers double as leak-calibration set (all are MLP-vetoed)
    s_therm = run_path("thermal", CK_THERMAL, [("ir_dset_final", "iou"), ("cbam", "iou")],
                       Cf_therm, src_therm, leak_therm)

    print("\n\n========== JSON SUMMARY ==========")
    import json
    print(json.dumps({"gray": s_gray, "thermal": s_therm}, indent=2, default=float))
