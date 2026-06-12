"""pipeline_eval_paired.py - Phase B of the email-table recompute (CPU, instant).

Replays the supervisor email's 7 configs on the Phase-A cache, but with the NEW stack:
  classifier = robust6  (models/routers/lean_ft4/trust_ft4_robust6.joblib)
  filter     = V5 MLP   (RGB mlp_v5 / IR mlp_v5_ir_aligned 'mlp_aligned.pt')

No GPU: the MLP forward runs on CPU from the cached 517-D feats. Re-run freely to sweep
thresholds. Emits per-rule metrics CSVs + an OLD(email)-vs-NEW comparison .md per surface.

Scoring mirrors classifier/eval_six_configs.py exactly (per-det greedy match; classifier
configs see both modalities' GT -> 'trust_both' frames count TPs from both; single-mod
configs score only their own GT). Anti-UAV = IoU@0.5, Svanstrom = IoP@0.5 (memory rule).

robust6's 6 features are computed exactly as its training builder (generate_lean19_data.py):
best = max-conf det; log_bbox_area = log(pw*ph+1) in pixels; aspect = pw/ph; rounded 4dp.
Dets are fed at the pipeline operating points (rgb>=0.25, ir>=0.40); robust6 was trained at
0.25 both -> minor train/serve skew only on the rare IR-only det in [0.25,0.40).

  py eval/pipeline_eval_paired.py
  py eval/pipeline_eval_paired.py --ir-mlp-thr 0.05 --rgb-mlp-thr 0.5
"""
from __future__ import annotations
import argparse, csv, glob, json, pickle, sys
from pathlib import Path

import numpy as np, joblib

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "classifier"))
sys.path.insert(0, str(REPO / "eval"))
from mlp_verifier import MLPVerifier  # noqa: E402
from metrics import score_trust_aware  # noqa: E402  (the published scoring rule)

_D = 10000  # dummy img dims: trust-aware sums size buckets, so aggregate is dim-independent

CACHE = REPO / "eval" / "results" / "_email_recompute" / "cache"
OUT = REPO / "eval" / "results" / "_email_recompute"
ROBUST6 = REPO / "models/routers/lean_ft4/trust_ft4_robust6.joblib"
MLP_RGB = REPO / "models/verifiers/rgb_v5/mlp_v5.pt"
MLP_IR = REPO / "models/verifiers/ir_aligned/mlp_aligned.pt"

SVAN_CATS = ("AIRPLANE", "BIRD", "DRONE", "HELICOPTER", "OTHER")
CONFIGS = ["ir_only", "rgb_only", "classifier", "ir_filter", "rgb_filter",
           "filter_then_classifier", "classifier_then_filter"]
GT_SCOPE = {"ir_only": ("ir",), "rgb_only": ("rgb",), "ir_filter": ("ir",),
            "rgb_filter": ("rgb",), "classifier": ("rgb", "ir"),
            "filter_then_classifier": ("rgb", "ir"),
            "classifier_then_filter": ("rgb", "ir")}
# email labels for the comparison table
LABEL = {"ir_only": "ir_only", "rgb_only": "rgb_only", "classifier": "classifier",
         "ir_filter": "ir_filter", "rgb_filter": "rgb_filter",
         "filter_then_classifier": "filter→classifier",
         "classifier_then_filter": "classifier→filter"}

# OLD column = the supervisor email's numbers (transcribed) — (TP,FP,FN,P,R,F1)
EMAIL = {
    ("antiuav", "iou"): {
        "ir_only": (79360, 1461, 4818, .9819, .9428, .9619),
        "rgb_only": (79060, 921, 639, .9885, .9920, .9902),
        "classifier": (158416, 1872, 815, .9883, .9949, .9916),
        "ir_filter": (79158, 1457, 5020, .9819, .9404, .9607),
        "rgb_filter": (79038, 920, 661, .9885, .9917, .9901),
        "filter_then_classifier": (158192, 1877, 814, .9883, .9949, .9916),
        "classifier_then_filter": (158192, 1868, 1039, .9883, .9935, .9909),
    },
    ("svanstrom", "iop"): {
        "ir_only": (11156, 622, 329, .9472, .9714, .9591),
        "rgb_only": (8054, 10777, 3660, .4277, .6876, .5274),
        "classifier": (18950, 208, 32, .9891, .9983, .9937),
        "ir_filter": (10814, 570, 671, .9499, .9416, .9457),
        "rgb_filter": (7670, 2568, 4044, .7492, .6548, .6988),
        "filter_then_classifier": (18211, 215, 35, .9883, .9981, .9932),
        "classifier_then_filter": (18226, 190, 756, .9897, .9602, .9747),
    },
}


def iou_iop(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., ix2 - ix1), max(0., iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0., 0.
    aa = (a[2]-a[0]) * (a[3]-a[1]); bb = (b[2]-b[0]) * (b[3]-b[1])
    u = aa + bb - inter
    return (inter/u if u > 0 else 0.), (inter/aa if aa > 0 else 0.)


def score(dets, gts, rule, thr=0.5):
    """dets=[(box,conf)], gts=[box]. Per-det greedy match -> tp,fp,fn."""
    tp = fp = 0; matched = set()
    for db, _ in dets:
        best_s, best_i = 0., -1
        for gi, g in enumerate(gts):
            iu, ip = iou_iop(db, g); s = iu if rule == "iou" else ip
            if s > best_s:
                best_s, best_i = s, gi
        if best_s >= thr and best_i not in matched:
            tp += 1; matched.add(best_i)
        else:
            fp += 1
    return tp, fp, len(gts) - len(matched)


def best_feats(dets):
    """robust6 per-modality features from dets=[(box,conf)] -> (max_conf, log_area, aspect)."""
    if not dets:
        return 0.0, 0.0, 0.0
    bc = max(d[1] for d in dets)
    bb = max(dets, key=lambda d: d[1])[0]
    pw = max(1.0, bb[2] - bb[0]); ph = max(1.0, bb[3] - bb[1])
    return float(bc), round(float(np.log(pw * ph + 1.0)), 4), round(float(pw / ph), 4)


def r6_vec(order, rgb_dets, ir_dets):
    """robust6 6-feature vector (training order) from (box,conf) det lists."""
    rmc, rla, rar = best_feats(rgb_dets); imc, ila, iar = best_feats(ir_dets)
    fmap = {"rgb_max_conf": rmc, "ir_max_conf": imc,
            "rgb_best_log_bbox_area": rla, "ir_best_log_bbox_area": ila,
            "rgb_best_aspect_ratio": rar, "ir_best_aspect_ratio": iar}
    return [fmap[f] for f in order]


def evaluate(surface, rule, model, order, args):
    shards = sorted(glob.glob(str(CACHE / f"{surface}_*.pkl")))
    if args.max_shards:
        shards = shards[:args.max_shards]
    if not shards:
        print(f"[{surface}] no shards in {CACHE}"); return None
    # pass 1: load, threshold on the CACHED f32 P(drone), build robust6 vectors
    FR = []; Xr = []; Xf = []
    for sp in shards:
        data = pickle.load(open(sp, "rb"))
        for fr in data["frames"]:
            rb, rc = fr["rgb"]["boxes"], fr["rgb"]["confs"]
            ib, ic = fr["ir"]["boxes"], fr["ir"]["confs"]
            rpd, ipd = fr["rgb"]["pdrone"], fr["ir"]["pdrone"]   # f32, precomputed
            rgb_raw = [(tuple(rb[i]), float(rc[i])) for i in range(len(rb)) if rc[i] >= args.rgb_conf]
            ir_raw = [(tuple(ib[i]), float(ic[i])) for i in range(len(ib)) if ic[i] >= args.ir_conf]
            rgb_flt = [(tuple(rb[i]), float(rc[i])) for i in range(len(rb))
                       if rc[i] >= args.rgb_conf and rpd[i] >= args.rgb_mlp_thr]
            ir_flt = [(tuple(ib[i]), float(ic[i])) for i in range(len(ib))
                      if ic[i] >= args.ir_conf and ipd[i] >= args.ir_mlp_thr]
            FR.append((rgb_raw, ir_raw, rgb_flt, ir_flt,
                       [tuple(x) for x in fr["rgb_gt"]],
                       [tuple(x) for x in fr["ir_gt"]], fr.get("cat", "OTHER")))
            Xr.append(r6_vec(order, rgb_raw, ir_raw))
            Xf.append(r6_vec(order, rgb_flt, ir_flt))
    # batch robust6 (per-row XGBoost predict was the v1 5.3h bottleneck)
    lbl_raw = model.predict(np.asarray(Xr, np.float32)) if Xr else np.zeros(0, int)
    lbl_flt = model.predict(np.asarray(Xf, np.float32)) if Xf else np.zeros(0, int)
    # pass 2: trust-aware scoring
    ctr = {c: {"tp": 0, "fp": 0, "fn": 0} for c in CONFIGS}
    fpcat = {c: {k: 0 for k in SVAN_CATS} for c in CONFIGS}
    for idx, (rgb_raw, ir_raw, rgb_flt, ir_flt, rgt, igt, cat) in enumerate(FR):
        lr = int(lbl_raw[idx]); lf = int(lbl_flt[idx])
        # trust-aware (eval/metrics.py): trusting one modality EXCLUDES the
        # other's GT (no phantom FN); reject(0) penalizes both; trust_both(3) sums.
        specs = {
            "ir_only":                (2, [],       ir_raw),
            "rgb_only":               (1, rgb_raw,  []),
            "ir_filter":              (2, [],       ir_flt),
            "rgb_filter":             (1, rgb_flt,  []),
            "classifier":             (lr, rgb_raw, ir_raw),
            "filter_then_classifier": (lf, rgb_flt, ir_flt),
            "classifier_then_filter": (lr, rgb_flt, ir_flt),
        }
        for cname, (lbl, rd, idt) in specs.items():
            s = score_trust_aware(lbl, rd, idt, rgt, igt, _D, _D, _D, _D,
                                  is_paired=True, rule=rule)
            fp = sum(s[b]["fp"] for b in s)
            ctr[cname]["tp"] += sum(s[b]["tp"] for b in s)
            ctr[cname]["fp"] += fp
            ctr[cname]["fn"] += sum(s[b]["fn"] for b in s)
            fpcat[cname][cat] += fp
    n_frames = len(FR)
    rows = []
    for c in CONFIGS:
        tp, fp, fn = ctr[c]["tp"], ctr[c]["fp"], ctr[c]["fn"]
        p = tp/(tp+fp) if tp+fp else 0.; r = tp/(tp+fn) if tp+fn else 0.
        f1 = 2*p*r/(p+r) if p+r else 0.
        rows.append({"config": c, "TP": tp, "FP": fp, "FN": fn,
                     "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f"metrics_{surface}_{rule}.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["config", "TP", "FP", "FN", "precision", "recall", "f1"])
        w.writeheader(); w.writerows(rows)
    if surface == "svanstrom":
        with (OUT / f"fp_by_category_{surface}_{rule}.csv").open("w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["config", *SVAN_CATS, "total"])
            for c in CONFIGS:
                vals = [fpcat[c][k] for k in SVAN_CATS]
                w.writerow([c, *vals, sum(vals)])
    print(f"[{surface} {rule.upper()}] {n_frames:,} frames")
    for r in rows:
        print(f"  {r['config']:<24} TP{r['TP']:>7} FP{r['FP']:>6} FN{r['FN']:>6}  "
              f"P{r['precision']:.4f} R{r['recall']:.4f} F1{r['f1']:.4f}")
    return rows


def write_comparison(surface, rule, rows):
    old = EMAIL.get((surface, rule))
    if old is None or rows is None:
        return
    new = {r["config"]: r for r in rows}
    lines = [f"# {surface.title()} ({rule.upper()}@0.5) — OLD email vs NEW stack\n",
             "OLD = email (Yolo26n_trained + fusion_no_fn + CNN patch). "
             "NEW = ft4 + v3b + **robust6** + **V5 MLP** filter.\n",
             "| Config | OLD P | OLD R | OLD F1 | NEW P | NEW R | NEW F1 | ΔF1 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for c in CONFIGS:
        o = old[c]; n = new[c]
        d = n["f1"] - o[5]
        lines.append(f"| {LABEL[c]} | {o[3]:.4f} | {o[4]:.4f} | {o[5]:.4f} | "
                     f"{n['precision']:.4f} | {n['recall']:.4f} | {n['f1']:.4f} | "
                     f"{d:+.4f} |")
    lines += ["", "### Counts (NEW stack)",
              "| Config | TP | FP | FN |", "|---|---:|---:|---:|"]
    for c in CONFIGS:
        n = new[c]
        lines.append(f"| {LABEL[c]} | {n['TP']:,} | {n['FP']:,} | {n['FN']:,} |")
    (OUT / f"comparison_{surface}_{rule}.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"  -> comparison_{surface}_{rule}.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-conf", type=float, default=0.25)
    ap.add_argument("--ir-conf", type=float, default=0.40)
    ap.add_argument("--rgb-mlp-thr", type=float, default=0.5,
                    help="RGB det survives if P(drone) >= thr (mlp_v5 ckpt default 0.5)")
    ap.add_argument("--ir-mlp-thr", type=float, default=0.05,
                    help="IR det survives if P(drone) >= thr (aligned recall-safe op-point)")
    ap.add_argument("--max-shards", type=int, default=0, help="0 = all (quick-test knob)")
    args = ap.parse_args()

    print(f"robust6={ROBUST6.name}  rgb_mlp_thr={args.rgb_mlp_thr}  ir_mlp_thr={args.ir_mlp_thr}")
    bundle = joblib.load(ROBUST6)
    model, order = bundle["model"], bundle["features"]
    assert len(order) == 6, f"expected 6 robust6 feats, got {order}"

    for surface, rule in [("antiuav", "iou"), ("svanstrom", "iop")]:
        rows = evaluate(surface, rule, model, order, args)
        write_comparison(surface, rule, rows)
    print(f"\nPhase B done -> {OUT}")


if __name__ == "__main__":
    main()
