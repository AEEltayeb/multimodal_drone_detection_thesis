"""sweep_email_thresholds.py - FREE MLP-threshold sweep on the email-recompute cache.

P(drone) is cached in f32, so sweeping is just re-thresholding (no GPU, no YOLO).
For each threshold, reports the single-modality filter config (trust-aware) F1 +
confuser-FP suppression, per surface. Shows the operating-point curve / Pareto.

  py eval/sweep_email_thresholds.py
"""
from __future__ import annotations
import glob, pickle, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))
from metrics import score_trust_aware  # noqa: E402

CACHE = REPO / "eval" / "results" / "_email_recompute" / "cache"
_D = 10000
THRS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
CONF_CATS = {"AIRPLANE", "BIRD", "HELICOPTER"}


def load(surface, rgb_conf=0.25, ir_conf=0.40):
    out = []
    for sp in sorted(glob.glob(str(CACHE / f"{surface}_*.pkl"))):
        for fr in pickle.load(open(sp, "rb"))["frames"]:
            rb, rc, rpd = fr["rgb"]["boxes"], fr["rgb"]["confs"], fr["rgb"]["pdrone"]
            ib, ic, ipd = fr["ir"]["boxes"], fr["ir"]["confs"], fr["ir"]["pdrone"]
            rk = [i for i in range(len(rb)) if rc[i] >= rgb_conf]
            ik = [i for i in range(len(ib)) if ic[i] >= ir_conf]
            out.append({
                "rgb": [(tuple(rb[i]), float(rc[i]), float(rpd[i])) for i in rk],
                "ir":  [(tuple(ib[i]), float(ic[i]), float(ipd[i])) for i in ik],
                "rgt": [tuple(x) for x in fr["rgb_gt"]],
                "igt": [tuple(x) for x in fr["ir_gt"]],
                "cat": fr.get("cat", "OTHER"),
            })
    return out


def sweep_modality(frames, mod, label, rule):
    """mod='rgb'|'ir'. Returns rows of (thr, P, R, F1, suppression%)."""
    gt_key = "rgt" if mod == "rgb" else "igt"
    rows = []
    for thr in THRS:
        tp = fp = fn = 0
        conf_before = conf_after = 0
        for fr in frames:
            dets = fr[mod]
            kept = [(b, c) for (b, c, p) in dets if p >= thr]
            kr = kept if mod == "rgb" else []
            ki = kept if mod == "ir" else []
            s = score_trust_aware(label, kr, ki, fr["rgt"], fr["igt"],
                                  _D, _D, _D, _D, is_paired=True, rule=rule)
            tp += sum(s[b]["tp"] for b in s)
            fp += sum(s[b]["fp"] for b in s)
            fn += sum(s[b]["fn"] for b in s)
            if fr["cat"] in CONF_CATS:           # confuser frame: every det is FP
                conf_before += len(dets)
                conf_after += len(kept)
        p = tp / (tp + fp) if tp + fp else 0.
        r = tp / (tp + fn) if tp + fn else 0.
        f1 = 2 * p * r / (p + r) if p + r else 0.
        supp = 1 - conf_after / conf_before if conf_before else 0.
        rows.append((thr, p, r, f1, supp, conf_before, conf_after))
    return rows


def main():
    for surface, rule in [("svanstrom", "iop"), ("antiuav", "iou")]:
        frames = load(surface)
        if not frames:
            print(f"[{surface}] no cache"); continue
        print(f"\n{'='*78}\n{surface.upper()} ({rule.upper()})  n_frames={len(frames):,}\n{'='*78}")
        for mod, label, name in [("rgb", 1, "rgb_filter"), ("ir", 2, "ir_filter")]:
            rows = sweep_modality(frames, mod, label, rule)
            print(f"\n{name} (sweep P(drone) threshold):")
            print(f"  {'thr':>5} {'P':>7} {'R':>7} {'F1':>7} {'confSuppr':>10} {'confFP':>8}")
            best = max(rows, key=lambda x: x[3])
            for (thr, p, r, f1, supp, cb, ca) in rows:
                star = "  <- best F1" if (thr, p, r, f1, supp, cb, ca) == best else ""
                print(f"  {thr:>5.2f} {p:>7.4f} {r:>7.4f} {f1:>7.4f} {supp:>9.1%} {ca:>8,d}{star}")


if __name__ == "__main__":
    main()
