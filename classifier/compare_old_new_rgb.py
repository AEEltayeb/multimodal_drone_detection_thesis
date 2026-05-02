"""
compare_old_new_rgb.py — Fast old-vs-new RGB comparison using cached detections.

Computes ir_only and rgb_only per-detection metrics for BOTH old and new caches.
No image loading, no classifier, no patch verifier — pure JSON + GT label files.
Runs in under a minute.

Usage:
    python classifier/compare_old_new_rgb.py
    python classifier/compare_old_new_rgb.py --rgb-conf-old 0.25 --rgb-conf-new 0.30
"""

from __future__ import annotations
import argparse, json, time
from collections import defaultdict
from pathlib import Path

OLD_ANTIUAV = Path(__file__).resolve().parent / "runs" / "raw_detections.json"
NEW_ANTIUAV = Path(__file__).resolve().parent / "runs" / "raw_detections.new.json"
OLD_SVAN    = Path(__file__).resolve().parent / "runs" / "svanstrom_detections.json"
NEW_SVAN    = Path(__file__).resolve().parent / "runs" / "svanstrom_detections.new.json"

SVAN_CATS = ("AIRPLANE", "BIRD", "DRONE", "HELICOPTER")


def svan_category(key):
    for c in SVAN_CATS:
        if f"_{c}_" in key:
            return c
    return "OTHER"


def read_yolo_labels(path, w, h):
    boxes = []
    p = Path(path)
    if not p.exists():
        return boxes
    for ln in p.read_text().splitlines():
        parts = ln.strip().split()
        if len(parts) < 5 or parts[0] != "0":
            continue
        cx, cy, bw, bh = map(float, parts[1:5])
        boxes.append(((cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h))
    return boxes


def iou_iop(a, b):
    ix1, iy1 = max(a[0],b[0]), max(a[1],b[1])
    ix2, iy2 = min(a[2],b[2]), min(a[3],b[3])
    iw, ih = max(0., ix2-ix1), max(0., iy2-iy1)
    inter = iw * ih
    if inter <= 0: return 0., 0.
    aa = (a[2]-a[0])*(a[3]-a[1])
    bb = (b[2]-b[0])*(b[3]-b[1])
    u = aa + bb - inter
    return (inter/u if u > 0 else 0.), (inter/aa if aa > 0 else 0.)


def score_dets(dets, gts, rule="iou", thr=0.5):
    tp = fp = 0; matched = set()
    for d_box, _ in dets:
        best_i, best_s = -1, 0.
        for gi, g in enumerate(gts):
            iu, ip = iou_iop(d_box, g)
            s = iu if rule == "iou" else ip
            if s > best_s: best_s, best_i = s, gi
        if best_s >= thr and best_i not in matched:
            tp += 1; matched.add(best_i)
        else:
            fp += 1
    fn = len(gts) - len(matched)
    return tp, fp, fn


CONFIGS = ["ir_only", "rgb_only"]
RULES = ("iou", "iop")


def evaluate_cache(ds_name, cache_path, rgb_conf, ir_conf, tag):
    print(f"[{ds_name}] [{tag}] loading {cache_path.name} ({cache_path.stat().st_size/1e6:.0f} MB)...")
    t_load = time.time()
    raw = json.loads(cache_path.read_text())
    print(f"[{ds_name}] [{tag}] loaded {len(raw):,} frames in {time.time()-t_load:.1f}s")

    keys = sorted(raw.keys())
    counters = {rule: {c: {"tp":0,"fp":0,"fn":0} for c in CONFIGS} for rule in RULES}
    fp_cat   = {rule: {c: defaultdict(int) for c in CONFIGS} for rule in RULES}

    t0 = time.time()
    for key in keys:
        entry = raw[key]
        rgb_dets = [((d[0],d[1],d[2],d[3]), d[4]) for d in entry["rgb_dets"] if d[4] >= rgb_conf]
        ir_dets  = [((d[0],d[1],d[2],d[3]), d[4]) for d in entry["ir_dets"]  if d[4] >= ir_conf]
        rw, rh = entry["rgb_w"], entry["rgb_h"]
        iw, ih = entry["ir_w"],  entry["ir_h"]
        rgb_gt = read_yolo_labels(entry["rgb_lbl"], rw, rh)
        ir_gt  = read_yolo_labels(entry["ir_lbl"],  iw, ih)

        configs = {
            "ir_only":  ([], ir_dets,  [], ir_gt),
            "rgb_only": (rgb_dets, [], rgb_gt, []),
        }
        cat = svan_category(key)
        for c_name, (kept_rgb, kept_ir, gt_rgb, gt_ir) in configs.items():
            for rule in RULES:
                tp = fp = fn = 0
                if kept_rgb or gt_rgb:
                    t_, f_, n_ = score_dets(kept_rgb, gt_rgb, rule=rule)
                    tp += t_; fp += f_; fn += n_
                if kept_ir or gt_ir:
                    t_, f_, n_ = score_dets(kept_ir, gt_ir, rule=rule)
                    tp += t_; fp += f_; fn += n_
                counters[rule][c_name]["tp"] += tp
                counters[rule][c_name]["fp"] += fp
                counters[rule][c_name]["fn"] += fn
                fp_cat[rule][c_name][cat] += fp

    elapsed = time.time() - t0
    print(f"[{ds_name}] [{tag}] scored {len(keys):,} frames in {elapsed:.1f}s")

    results = {}
    for rule in RULES:
        results[rule] = {}
        for c_name in CONFIGS:
            tp = counters[rule][c_name]["tp"]
            fp = counters[rule][c_name]["fp"]
            fn = counters[rule][c_name]["fn"]
            p = tp/(tp+fp) if (tp+fp) > 0 else 0.
            r = tp/(tp+fn) if (tp+fn) > 0 else 0.
            f1 = 2*p*r/(p+r) if (p+r) > 0 else 0.
            results[rule][c_name] = {
                "TP": tp, "FP": fp, "FN": fn,
                "P": round(p, 4), "R": round(r, 4), "F1": round(f1, 4),
                "fp_by_cat": dict(fp_cat[rule][c_name]),
            }
    return results


def print_comparison(ds_name, old_res, new_res, rule, conf_old, conf_new):
    print(f"\n{'='*95}")
    print(f"  {ds_name.upper()} — {rule.upper()} match  |  OLD RGB @{conf_old}  vs  NEW RGB @{conf_new}")
    print(f"{'='*95}")
    print(f"  {'config':<12s} {'':4s} {'TP':>9s} {'FP':>9s} {'FN':>9s} {'Prec':>9s} {'Rec':>9s} {'F1':>9s}")
    print(f"  {'-'*80}")
    for c in CONFIGS:
        o = old_res[rule][c]
        n = new_res[rule][c]
        print(f"  {c:<12s}  old {o['TP']:>9,} {o['FP']:>9,} {o['FN']:>9,} {o['P']:>9.4f} {o['R']:>9.4f} {o['F1']:>9.4f}")
        print(f"  {'':12s}  new {n['TP']:>9,} {n['FP']:>9,} {n['FN']:>9,} {n['P']:>9.4f} {n['R']:>9.4f} {n['F1']:>9.4f}")
        dfp = n['FP'] - o['FP']
        dtp = n['TP'] - o['TP']
        dfn = n['FN'] - o['FN']
        df1 = n['F1'] - o['F1']
        print(f"  {'':12s}   Δ  {dtp:>+9,} {dfp:>+9,} {dfn:>+9,} {'':>9s} {'':>9s} {df1:>+9.4f}")
        print()

    # FP by category
    cats = [*SVAN_CATS, "OTHER"]
    has_cats = any(old_res[rule][c]["fp_by_cat"].get(cat, 0) > 0
                   for c in CONFIGS for cat in SVAN_CATS)
    if has_cats:
        print(f"  FP by category ({rule.upper()}):")
        print(f"  {'config':<12s} {'':4s} " + " ".join(f"{c:>12s}" for c in cats))
        for c_name in CONFIGS:
            o = old_res[rule][c_name]["fp_by_cat"]
            n = new_res[rule][c_name]["fp_by_cat"]
            print(f"  {c_name:<12s}  old " + " ".join(f"{o.get(c,0):>12,}" for c in cats))
            print(f"  {'':12s}  new " + " ".join(f"{n.get(c,0):>12,}" for c in cats))
            delta = " ".join(f"{n.get(c,0)-o.get(c,0):>+12,}" for c in cats)
            print(f"  {'':12s}   Δ  {delta}")
            print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-conf-old", type=float, default=0.25)
    ap.add_argument("--rgb-conf-new", type=float, default=0.30)
    ap.add_argument("--ir-conf", type=float, default=0.40)
    ap.add_argument("--dataset", choices=["antiuav", "svanstrom", "both"], default="both")
    args = ap.parse_args()

    datasets = []
    if args.dataset in ("antiuav", "both"):
        datasets.append(("antiuav", OLD_ANTIUAV, NEW_ANTIUAV))
    if args.dataset in ("svanstrom", "both"):
        datasets.append(("svanstrom", OLD_SVAN, NEW_SVAN))

    for ds_name, old_path, new_path in datasets:
        if not old_path.exists():
            print(f"[{ds_name}] SKIP: {old_path.name} not found")
            continue
        if not new_path.exists():
            print(f"[{ds_name}] SKIP: {new_path.name} not found")
            continue

        old_res = evaluate_cache(ds_name, old_path, args.rgb_conf_old, args.ir_conf, "OLD")
        new_res = evaluate_cache(ds_name, new_path, args.rgb_conf_new, args.ir_conf, "NEW")

        for rule in RULES:
            print_comparison(ds_name, old_res, new_res, rule, args.rgb_conf_old, args.rgb_conf_new)

    # Also run new @ 0.25 for pure model comparison
    if args.rgb_conf_old != args.rgb_conf_new:
        print(f"\n\n{'#'*95}")
        print(f"  BONUS: New RGB also evaluated at conf={args.rgb_conf_old} (isolates model change from conf change)")
        print(f"{'#'*95}")
        for ds_name, old_path, new_path in datasets:
            if not new_path.exists(): continue
            old_res = evaluate_cache(ds_name, old_path, args.rgb_conf_old, args.ir_conf, "OLD@0.25")
            new_at_old = evaluate_cache(ds_name, new_path, args.rgb_conf_old, args.ir_conf, "NEW@0.25")
            for rule in RULES:
                print_comparison(ds_name, old_res, new_at_old, rule, args.rgb_conf_old, args.rgb_conf_old)


if __name__ == "__main__":
    main()
