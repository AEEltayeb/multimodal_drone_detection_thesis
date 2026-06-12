"""
mri.diagnose — the punchline: does this detector need an FP-reduction classifier?

Combines four signals into a verdict:
  1. raw_halluc_rate  — bare-detector FP per confuser image (how often it fires
     on a non-drone). High = the detector has a hallucination problem.
  2. raw_drone_f1     — bare-detector F1 on the positive datasets. Memory
     (pipeline_useful_when): a classifier earns its keep mainly when raw F1<~0.7.
  3. lda_separability — can a linear boundary split drone vs confuser features?
     If not, no downstream classifier will either; fix the detector/data instead.
  4. recall_cost      — out-of-fold estimate of how much true-drone recall the
     trained classifier sacrifices to win that FP reduction.

Thresholds are tunable from the CLI. The output is one plain-English verdict
plus the evidence dict that report.py renders.
"""
from __future__ import annotations

import numpy as np

from metrics import compute_prf  # type: ignore  (eval/ on sys.path via mri.cli)

VERDICTS = {
    "not_needed": "No classifier needed — the detector is already clean on these confusers.",
    "recommended": "Classifier strongly recommended — large FP cut at low recall cost.",
    "wont_help": "Classifier won't help — features don't separate; fix the detector or data.",
    "marginal": "Marginal — the FP cut trades meaningful recall; decide from the curve.",
    "insufficient": "Insufficient data — not enough confusers and/or drones to decide.",
}


def diagnose(raws, sep_summary, oof=None, y=None, threshold=0.5,
             fp_rate_thr=0.05, sep_thr=0.90, recall_cost_thr=0.10):
    """Return a diagnosis dict. `oof`/`y` are the CV out-of-fold drone-probs and
    labels (present only when a classifier was trained)."""
    neg = [r for r in raws if r.get("role") == "neg" and r.get("n_images")]
    pos = [r for r in raws if r.get("role") == "pos" and r.get("n_images")]

    neg_imgs = sum(r["n_images"] for r in neg)
    neg_fp = sum(r["fp"] for r in neg)
    raw_halluc_rate = (neg_fp / neg_imgs) if neg_imgs else None

    raw_prf = None
    if pos:
        tp = sum(r["tp"] for r in pos)
        fp = sum(r["fp"] for r in pos)
        fn = sum(r["fn"] for r in pos)
        raw_prf = compute_prf(tp, fp, fn)

    lda = sep_summary.get("lda_train_accuracy")
    n_drone = sep_summary.get("n_drone", 0)
    n_conf = sep_summary.get("n_confuser", 0)

    out = {
        "raw_halluc_rate": raw_halluc_rate,
        "raw_drone_prf": raw_prf,
        "lda_separability": lda,
        "silhouette": sep_summary.get("silhouette"),
        "n_drone": n_drone,
        "n_confuser": n_conf,
        "meta_max_auroc": sep_summary.get("meta_max_auroc"),
        "yolo_max_auroc": sep_summary.get("yolo_max_auroc"),
        "thresholds": {"fp_rate": fp_rate_thr, "separability": sep_thr,
                       "recall_cost": recall_cost_thr},
    }

    # Out-of-fold projected effect of the trained classifier.
    if oof is not None and y is not None and not np.all(np.isnan(oof)):
        y = np.asarray(y)
        valid = ~np.isnan(oof)
        conf_mask = valid & (y == 0)
        drone_mask = valid & (y == 1)
        kept_conf = float((oof[conf_mask] >= threshold).mean()) if conf_mask.any() else None
        recall_ret = float((oof[drone_mask] >= threshold).mean()) if drone_mask.any() else None
        out["classifier_keeps_confuser_frac"] = kept_conf      # lower = better
        out["classifier_recall_retention"] = recall_ret        # higher = better
        out["recall_cost"] = (1 - recall_ret) if recall_ret is not None else None
        if kept_conf is not None:
            out["fp_reduction"] = 1 - kept_conf          # classifier-only (out-of-fold); no image stream needed
        if raw_halluc_rate is not None and kept_conf is not None:
            out["projected_fp_rate"] = raw_halluc_rate * kept_conf   # per-image: needs the bare hallucination rate

    # ── Verdict ──────────────────────────────────────────────────────────
    verdict = _decide(out, fp_rate_thr, sep_thr, recall_cost_thr)
    out["verdict"] = verdict
    out["verdict_text"] = VERDICTS[verdict]
    out["rationale"] = _rationale(out, verdict, fp_rate_thr, sep_thr, recall_cost_thr)
    return out


def _decide(d, fp_rate_thr, sep_thr, recall_cost_thr):
    """`insufficient` means genuinely too few detections to fit/judge — NOT the
    absence of an image-level hallucination stream. When `raw_halluc_rate` is
    unavailable (e.g. a cached feature-only corpus via --resume), we still decide
    from feature separability + the trained classifier's out-of-fold behavior."""
    n_drone, n_conf = d["n_drone"], d["n_confuser"]
    if n_conf < 50 or n_drone < 50:
        return "insufficient"

    halluc = d.get("raw_halluc_rate")
    lda = d.get("lda_separability")
    rc = d.get("recall_cost")

    # A measured-clean detector is the only case that needs the halluc rate.
    if halluc is not None and halluc <= fp_rate_thr:
        return "not_needed"

    # Features must separate for any downstream classifier to help.
    if lda is not None and lda < sep_thr:
        return "wont_help"

    # Separable. Use the trained classifier's recall cost if we have it;
    # otherwise (no classifier trained) recommend training to confirm.
    if rc is None:
        return "recommended"
    if rc <= recall_cost_thr:
        return "recommended"
    return "marginal"


def _rationale(d, verdict, fp_rate_thr, sep_thr, recall_cost_thr):
    bits = [f"{d['n_drone']} drones / {d['n_confuser']} confusers"]
    h = d.get("raw_halluc_rate")
    if h is not None:
        bits.append(f"raw hallucination {h:.1%}/img (thr {fp_rate_thr:.0%})")
    else:
        bits.append("raw hallucination not measured (feature-only input)")
    if d.get("lda_separability") is not None:
        bits.append(f"LDA separability {d['lda_separability']:.3f} (thr {sep_thr:.2f})")
    if d.get("recall_cost") is not None:
        bits.append(f"recall cost {d['recall_cost']:.1%} (thr {recall_cost_thr:.0%})")
    if d.get("fp_reduction") is not None:
        bits.append(f"projected FP cut {d['fp_reduction']:.0%}")
    elif d.get("classifier_keeps_confuser_frac") is not None:
        bits.append(f"classifier rejects {1 - d['classifier_keeps_confuser_frac']:.0%} of confusers")
    if d.get("raw_drone_prf"):
        bits.append(f"raw drone F1 {d['raw_drone_prf']['f1']:.3f}")
    return "; ".join(bits)
