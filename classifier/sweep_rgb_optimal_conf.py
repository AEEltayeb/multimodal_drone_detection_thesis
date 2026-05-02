"""
sweep_rgb_optimal_conf.py — Find F1-optimal RGB conf for OLD vs v3_more
on the three big paired datasets, using cached conf=0.001 detections.

For each (model, dataset, rule):
  for conf in [0.05, 0.10, ..., 0.60]:
    filter detections by conf, score TP/FP/FN at IoU=0.5 / IoP=0.5
  report F1-optimal conf and the resulting P/R/F1.

No GPU, no YOLO re-inference. Uses:
  classifier/runs/reliability/inference/{tag}.json          (old RGB)
  classifier/runs/reliability/inference_v3more/{tag}.json   (v3_more)

Usage:
  python classifier/sweep_rgb_optimal_conf.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
OLD_DIR = SCRIPT_DIR / "runs" / "reliability" / "inference"
NEW_DIR = SCRIPT_DIR / "runs" / "reliability" / "inference_v3more"

DATASETS = [
    ("antiuav_test_rgb", "antiuav_test"),
    ("rgb_dataset_test", "dataset_rgb"),
    ("svanstrom_rgb",    "svanstrom"),
]

CONF_GRID = np.round(np.arange(0.05, 0.61, 0.05), 3).tolist()


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    ab = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    u = aa + ab - inter
    return inter / u if u > 0 else 0.0


def iop(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    return inter / aa if aa > 0 else 0.0


def parse_yolo_gt(text, w, h):
    boxes = []
    if not text:
        return boxes
    for ln in text.strip().split("\n"):
        p = ln.strip().split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = map(float, p[1:5])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        boxes.append([x1, y1, x2, y2])
    return boxes


def score(dets, gts, rule):
    """dets list of [x1,y1,x2,y2,conf]. Returns TP, FP, FN."""
    if not dets and not gts:
        return 0, 0, 0
    if not gts:
        return 0, len(dets), 0
    if not dets:
        return 0, 0, len(gts)

    fn = score_fn = iou if rule == "iou" else iop
    pairs = []
    for di, d in enumerate(dets):
        for gi, g in enumerate(gts):
            s = score_fn(d[:4], g)
            if s >= 0.5:
                pairs.append((s, di, gi))
    pairs.sort(reverse=True)
    matched_d = set(); matched_g = set()
    for s, di, gi in pairs:
        if di in matched_d or gi in matched_g:
            continue
        matched_d.add(di); matched_g.add(gi)
    tp = len(matched_d)
    fp = len(dets) - tp
    fn_count = len(gts) - len(matched_g)
    return tp, fp, fn_count


def sweep_one(jsons_path: Path):
    print(f"  loading {jsons_path.name}...", end="", flush=True)
    raw = json.loads(jsons_path.read_text())
    print(f" {len(raw):,} frames")

    out = {"iou": {}, "iop": {}}
    for rule in ("iou", "iop"):
        for c in CONF_GRID:
            out[rule][c] = [0, 0, 0]  # TP, FP, FN

    for stem, entry in raw.items():
        w = entry.get("w"); h = entry.get("h")
        if w is None or h is None:
            continue
        gts = parse_yolo_gt(entry.get("gt", ""), w, h)
        dets_all = entry.get("dets", [])
        for c in CONF_GRID:
            kept = [d for d in dets_all if d[4] >= c]
            for rule in ("iou", "iop"):
                tp, fp, fn = score(kept, gts, rule)
                out[rule][c][0] += tp
                out[rule][c][1] += fp
                out[rule][c][2] += fn

    # F1 per threshold
    summary = {"iou": {}, "iop": {}}
    for rule in ("iou", "iop"):
        for c, (tp, fp, fn) in out[rule].items():
            p = tp / max(1, tp + fp)
            r = tp / max(1, tp + fn)
            f1 = 2 * p * r / max(1e-9, p + r)
            summary[rule][c] = {"tp": tp, "fp": fp, "fn": fn,
                                 "p": p, "r": r, "f1": f1}
    return summary


def best(summary, rule):
    items = summary[rule]
    return max(items.items(), key=lambda kv: kv[1]["f1"])


def main():
    results = {}
    for (tag, label) in DATASETS:
        print(f"\n{'=' * 70}\n{label} ({tag})\n{'=' * 70}")
        old_p = OLD_DIR / f"{tag}.json"
        new_p = NEW_DIR / f"{tag}.json"
        if not old_p.exists():
            print(f"  [skip] {old_p} missing"); continue
        if not new_p.exists():
            print(f"  [skip] {new_p} missing"); continue
        print("  OLD:")
        old = sweep_one(old_p)
        print("  v3_more:")
        new = sweep_one(new_p)
        results[label] = {"old": old, "new": new}

        for rule in ("iou", "iop"):
            o_c, o_m = best(old, rule)
            n_c, n_m = best(new, rule)
            o25 = old[rule][0.25]
            n25 = new[rule][0.25]
            print(f"\n  [{rule.upper()}] best-F1:")
            print(f"    OLD     conf={o_c:.2f}  P={o_m['p']:.4f}  R={o_m['r']:.4f}  F1={o_m['f1']:.4f}  "
                  f"TP={o_m['tp']} FP={o_m['fp']} FN={o_m['fn']}")
            print(f"    v3_more conf={n_c:.2f}  P={n_m['p']:.4f}  R={n_m['r']:.4f}  F1={n_m['f1']:.4f}  "
                  f"TP={n_m['tp']} FP={n_m['fp']} FN={n_m['fn']}")
            print(f"  [{rule.upper()}] at fixed conf=0.25:")
            print(f"    OLD     P={o25['p']:.4f}  R={o25['r']:.4f}  F1={o25['f1']:.4f}  "
                  f"TP={o25['tp']} FP={o25['fp']} FN={o25['fn']}")
            print(f"    v3_more P={n25['p']:.4f}  R={n25['r']:.4f}  F1={n25['f1']:.4f}  "
                  f"TP={n25['tp']} FP={n25['fp']} FN={n25['fn']}")

    out_path = SCRIPT_DIR / "runs" / "rgb_finetune_eval" / "conf_sweep.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
