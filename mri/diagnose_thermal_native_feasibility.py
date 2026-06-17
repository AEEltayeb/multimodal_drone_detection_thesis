"""diagnose_thermal_native_feasibility.py — is a THERMAL-NATIVE confuser filter
feasible? i.e. does v3b's THERMAL feature space separate drone from confuser
(ir_confusers is ~76% airplane), bounding the value of a dedicated thermal-native
retrain on the labelled G:/drone/IR_confusers crops?  ZERO-GPU (cached feats).

Trust-aware: drone feats are GT-matched detections on the thermal drone surfaces;
confuser feats are every detection on ir_confusers (no GT). We report:
  - univariate per-feature AUROC (overfit-proof) — max + how many features >0.8
  - HELD-OUT LDA accuracy (70/30 stratified) — separability that generalises,
    not the train-acc=1.0 high-D artefact
  - the thermal confuser DETECTION count (data-scarcity caveat)

  py mri/diagnose_thermal_native_feasibility.py
"""
from __future__ import annotations
import pickle, sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "eval"))
from mri.stats import lda_separability, anova_f, per_feature_auroc   # noqa: E402
from diagnose_rgbtest_veto_mechanism import match                    # noqa: E402 (reuse)

CACHE = REPO / "eval/results/_offline_pipeline/cache"
DRONE_SURF = ["antiuav_ir", "ir_dset_final", "svanstrom_ir", "ir_video"]
CONF_SURF = "ir_confusers"
IMG = REPO / "docs/analysis/images"; IMG.mkdir(parents=True, exist_ok=True)
OUT_PNG = IMG / "2026-06-17_ir_thermal_native_feasibility.png"
OUT_JSON = REPO / "docs/analysis/2026-06-17_ir_thermal_native_feasibility.json"


def drone_feats(name):
    d = pickle.load(open(CACHE / f"{name}.pkl", "rb"))
    rule = d["meta"]["rule"]; out = []
    for fr in d["frames"]:
        if len(fr["feats"]) == 0 or len(fr["gt_boxes"]) == 0:
            continue
        for i, box in enumerate(fr["boxes"]):
            if max((match(box, g, rule) for g in fr["gt_boxes"]), default=0) >= 0.5:
                out.append(fr["feats"][i])
    return np.array(out, dtype=np.float32)


def conf_feats(name):
    d = pickle.load(open(CACHE / f"{name}.pkl", "rb"))
    return np.array([f for fr in d["frames"] for f in fr["feats"]], dtype=np.float32)


def main():
    Xd = np.vstack([drone_feats(n) for n in DRONE_SURF])
    Xc = conf_feats(CONF_SURF)
    print(f"thermal drone feats {len(Xd)} | thermal confuser feats {len(Xc)} (ir_confusers, ~76% airplane)")

    # balance for the LDA (cap drones to 5x confusers so acc isn't a prior)
    rng = np.random.RandomState(0)
    cap = min(len(Xd), 5 * len(Xc))
    Xd_b = Xd[rng.choice(len(Xd), cap, replace=False)]
    X = np.vstack([Xd_b, Xc]).astype(np.float32)
    y = np.r_[np.ones(len(Xd_b)), np.zeros(len(Xc))].astype(int)

    auroc = per_feature_auroc(X, y)
    n_strong = int((auroc > 0.8).sum()); n_mid = int((auroc > 0.7).sum())
    print(f"univariate AUROC: max {auroc.max():.3f} | {n_strong} feats >0.8 | {n_mid} feats >0.7")

    # held-out LDA (70/30 stratified)
    from sklearn.model_selection import train_test_split
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    _, train_acc, _ = lda_separability(Xtr, ytr)
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis().fit(Xtr, ytr)
    test_acc = float(lda.score(Xte, yte))
    # balanced test accuracy + per-class recall
    pred = lda.predict(Xte)
    rec_drone = float((pred[yte == 1] == 1).mean()); rec_conf = float((pred[yte == 0] == 0).mean())
    Z_full = lda.transform(X).ravel()
    print(f"held-out LDA: train acc {train_acc:.3f} | TEST acc {test_acc:.3f} | "
          f"test drone-recall {rec_drone:.3f} confuser-recall {rec_conf:.3f}")

    # ── figure ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    bins = np.linspace(Z_full.min(), Z_full.max(), 60)
    ax[0].hist(Z_full[y == 1], bins=bins, density=True, alpha=.6, label=f"drone (n={int((y==1).sum())})", color="tab:blue")
    ax[0].hist(Z_full[y == 0], bins=bins, density=True, alpha=.6, label=f"confuser (n={int((y==0).sum())})", color="tab:orange")
    ax[0].set_xlabel("LDA projection"); ax[0].set_ylabel("density")
    ax[0].set_title(f"(a) Thermal drone vs confuser in v3b space\nheld-out LDA acc {test_acc:.2f} (drone-R {rec_drone:.2f}, conf-R {rec_conf:.2f})")
    ax[0].legend(fontsize=9)

    top = np.sort(auroc)[::-1][:30]
    ax[1].bar(np.arange(len(top)), top, color="tab:green")
    ax[1].axhline(0.8, color="k", ls=":", lw=1, label="AUROC 0.8")
    ax[1].set_xlabel("feature rank"); ax[1].set_ylabel("univariate AUROC")
    ax[1].set_title(f"(b) Top-30 single-feature AUROC\n({n_strong} of 517 feats > 0.8)")
    ax[1].legend(fontsize=9); ax[1].set_ylim(0.5, 1.0)

    fig.suptitle("Thermal-native confuser-filter feasibility (zero-GPU proxy; thermal confuser data is scarce in-cache)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT_PNG, dpi=130); plt.close(fig)

    stats = {"n_thermal_drone_feats": int(len(Xd)), "n_thermal_confuser_feats": int(len(Xc)),
             "auroc_max": float(auroc.max()), "n_feats_auroc_gt_0.8": n_strong, "n_feats_auroc_gt_0.7": n_mid,
             "lda_train_acc": float(train_acc), "lda_test_acc": test_acc,
             "lda_test_drone_recall": rec_drone, "lda_test_confuser_recall": rec_conf,
             "figure": str(OUT_PNG)}
    OUT_JSON.write_text(json.dumps(stats, indent=2))
    print(f"\nsaved {OUT_PNG}\nsaved {OUT_JSON}")


if __name__ == "__main__":
    main()
