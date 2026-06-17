"""pooled_operating_point.py — the EXACT (conf, filter-thr) operating point that
maximises F1 over ALL datasets treated as ONE pooled set. (zero-GPU)

Method: pool every dataset into one micro-averaged set. Each surviving detection
is TP (covers a drone GT), FP (covers nothing -- this includes EVERY detection on
a confuser surface, which has no GT), and any GT with no surviving cover is FN.
Pooling the confuser surfaces in is what makes F1 a valid objective for the filter
(their detections are FPs in the objective). The operating point is a per-modality
vector because the three detectors/filters are distinct:
    RGB   detector-conf x mlp_v5 P(drone)
    IR    detector-conf x aligned P(drone)        (thermal)
    GRAY  detector-conf x aligned_gray P(drone)   (grayscale fallback)
Global micro-F1 is maximised over the full cross-product of the three (conf,thr)
operating points; scoring is per-modality trust-aware (each detection vs its own
modality's GT, never unioned).

Reads the conf=0.05 caches (so the full conf>=0.05 grid is available).
NOTE: svanstrom (paired flagship) + dut have no 0.05 cache -> excluded; re-cache
them at 0.05 and add to SURFACES to include (see docs/analysis note).

  py thesis_eval/pooled_operating_point.py
"""
from __future__ import annotations
import json, sys, itertools
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
for s in ("eval", "classifier", "thesis_eval"):
    sys.path.insert(0, str(REPO / s))
import pickle
from metrics import score_detections, compute_prf                      # noqa: E402
from pipeline_eval_unified import load_verifiers, batch_probs, dets2, gts  # noqa: E402


def conf_mask(slot, t):
    c = np.asarray(slot["confs"], np.float32)
    return c >= t if len(c) else np.zeros(0, bool)

CACHE = REPO / "thesis_eval/cache_conf005"
CONF_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
THR_GRID = [0.0, 0.05, 0.10, 0.15, 0.25, 0.40, 0.60]   # 0.0 = filter off

# (cache, slot, gt_key, rule, has_drones, verifier_key, modality)
SURFACES = [
    ("rgb_dataset_test", "rgb", "rgb_gt", "iou", True,  "mlp_v5",      "RGB"),
    ("selcom_val",       "rgb", "rgb_gt", "iop", True,  "mlp_v5",      "RGB"),
    ("rgb_confuser",     "rgb", "rgb_gt", "iou", False, "mlp_v5",      "RGB"),
    ("ir_dset_final",    "ir",  "ir_gt",  "iou", True,  "aligned",     "IR"),
    ("ir_confusers",     "ir",  "ir_gt",  "iou", False, "aligned",     "IR"),
    ("svanstrom_gray",   "ir",  "ir_gt",  "iop", True,  "aligned_gray","GRAY"),
    ("gray_confuser",    "ir",  "ir_gt",  "iou", False, "aligned_gray","GRAY"),
    # antiuav paired (500-frame subsample): RGB side -> RGB group, IR side -> IR group
    ("antiuav",          "rgb", "rgb_gt", "iou", True,  "mlp_v5",      "RGB"),
    ("antiuav",          "ir",  "ir_gt",  "iou", True,  "aligned",     "IR"),
    # FLAGSHIP paired surfaces -- need conf=0.05 cache first (GPU re-cache):
    #   py -u thesis_eval/pipeline_cache_unified.py --conf 0.05 --cache-dir thesis_eval/cache_conf005 \
    #      --no-patch --target 4000 --only svanstrom,dut_antiuav_960
    # Skipped automatically until the cache file exists.
    ("svanstrom",        "rgb", "rgb_gt", "iop", True,  "mlp_v5",      "RGB"),
    ("svanstrom",        "ir",  "ir_gt",  "iop", True,  "aligned",     "IR"),
    ("dut_antiuav_960",  "rgb", "rgb_gt", "iou", True,  "mlp_v5",      "RGB"),
    ("dut_antiuav_960",  "ir",  "ir_gt",  "iou", True,  "aligned_gray","GRAY"),
]
SHIPPED = {"RGB": (0.25, 0.25), "IR": (0.40, 0.05), "GRAY": (0.40, 0.25)}


def counts_table(frames, slot, gt_key, rule, drones, probs):
    """(conf,thr) -> (tp,fp,fn) pooled over this surface's frames."""
    tab = {}
    for t in CONF_GRID:
        masks = [conf_mask(fr[slot], t) for fr in frames]
        for vt in THR_GRID:
            tp = fp = fn = 0
            for i, fr in enumerate(frames):
                m = masks[i] & (probs[i] >= vt)
                if drones:
                    a, b, c = score_detections(dets2(fr[slot], m), gts(fr[gt_key]), rule=rule)
                    tp += a; fp += b; fn += c
                else:
                    fp += int(m.sum())          # confuser surface: every kept det is FP
            tab[(t, vt)] = (tp, fp, fn)
    return tab


def main():
    verifs = load_verifiers(device="cpu")
    # per-surface counts tables + per-modality pooled tables
    surf_tab = {}
    group_tab = {g: {(t, vt): [0, 0, 0] for t in CONF_GRID for vt in THR_GRID}
                 for g in ("RGB", "IR", "GRAY")}
    loaded = {}
    missing = []
    for (name, slot, gt_key, rule, drones, vkey, grp) in SURFACES:
        if name not in loaded:
            fp_cache = CACHE / f"{name}.pkl"
            if not fp_cache.exists():
                if name not in missing:
                    missing.append(name)
                    print(f"  [SKIP {name}: no conf=0.05 cache -> re-cache to include it]")
                continue
            loaded[name] = pickle.load(open(fp_cache, "rb"))
        frames = loaded[name]["frames"]
        probs = batch_probs(frames, slot, verifs[vkey])
        tab = counts_table(frames, slot, gt_key, rule, drones, probs)
        surf_tab[(name, slot)] = (tab, drones, grp)
        for k, (tp, fp, fn) in tab.items():
            group_tab[grp][k][0] += tp; group_tab[grp][k][1] += fp; group_tab[grp][k][2] += fn
        print(f"  scored {name}/{slot} ({grp}, drones={drones})")

    def f1_of(tp, fp, fn):
        return compute_prf(tp, fp, fn)["f1"], compute_prf(tp, fp, fn)

    # global grid search over the per-modality cross-product
    best = None
    keys = [(t, vt) for t in CONF_GRID for vt in THR_GRID]
    for rk in keys:
        rtp, rfp, rfn = group_tab["RGB"][rk]
        for ik in keys:
            itp, ifp, ifn = group_tab["IR"][ik]
            for gk in keys:
                gtp, gfp, gfn = group_tab["GRAY"][gk]
                tp, fp, fn = rtp + itp + gtp, rfp + ifp + gfp, rfn + ifn + gfn
                f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
                if best is None or f1 > best[0]:
                    best = (f1, rk, ik, gk, (tp, fp, fn))

    def report_point(label, pts):
        tp = fp = fn = 0
        for g in ("RGB", "IR", "GRAY"):
            a, b, c = group_tab[g][pts[g]]; tp += a; fp += b; fn += c
        prf = compute_prf(tp, fp, fn)
        print(f"\n{label}")
        for g in ("RGB", "IR", "GRAY"):
            print(f"    {g:<5} conf={pts[g][0]:<4} filt_thr={pts[g][1]:<4}")
        print(f"    GLOBAL micro  P={prf['precision']:.4f}  R={prf['recall']:.4f}  "
              f"F1={prf['f1']:.4f}   (TP={tp} FP={fp} FN={fn})")
        # per-surface F1 / fire at this point
        for (name, slot), (tab, drones, grp) in surf_tab.items():
            tp_, fp_, fn_ = tab[pts[grp]]
            if drones:
                print(f"      {name}/{slot:<3} F1={compute_prf(tp_,fp_,fn_)['f1']:.4f} "
                      f"(R={compute_prf(tp_,fp_,fn_)['recall']:.3f})")
            else:
                print(f"      {name}/{slot:<3} FP={fp_} (confuser; pure FP in pool)")
        return prf["f1"]

    opt = {"RGB": best[1], "IR": best[2], "GRAY": best[3]}
    print("\n" + "=" * 70)
    inc = sorted({n for (n, _s) in surf_tab})
    print(f"POOL COVERAGE: included {inc}")
    if missing:
        print(f"  *** MISSING (no conf=0.05 cache, NOT in pool): {missing} ***")
        print(f"  *** result is PARTIAL until these are re-cached and rerun ***")
    report_point("OPTIMAL pooled operating point (max global micro-F1):", opt)
    report_point("SHIPPED defaults:", SHIPPED)
    # filter-OFF baseline: best conf per modality at thr=0.0
    off = {}
    for g in ("RGB", "IR", "GRAY"):
        bf = max(((compute_prf(*group_tab[g][(t, 0.0)])["f1"], (t, 0.0)) for t in CONF_GRID))
        off[g] = bf[1]
    report_point("FILTER OFF (best conf per modality, thr=0):", off)

    out = {"optimal": {g: opt[g] for g in opt}, "optimal_f1": best[0],
           "shipped": SHIPPED, "filter_off": off,
           "group_tables": {g: {f"{k[0]}_{k[1]}": v for k, v in group_tab[g].items()}
                            for g in group_tab}}
    p = REPO / "thesis_eval/results/conf_sweep/pooled_operating_point.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nsaved -> {p}")


if __name__ == "__main__":
    main()
