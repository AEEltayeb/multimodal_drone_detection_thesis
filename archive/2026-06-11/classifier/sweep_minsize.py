"""Sweep min-box-size per modality on raw detections (pre-classifier/confuser).

Counts TP/FP using cached detections + YOLO-format GT. Applies the
already-optimal confidence thresholds (RGB=0.30, IR=0.40 real thermal).
"""
import json, os, sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INF = os.path.join(ROOT, "classifier", "runs", "reliability", "inference")

RGB_SETS = [
    ("svanstrom_rgb", "svanstrom_rgb.json"),
    ("antiuav_test_rgb", "antiuav_test_rgb.json"),
    ("rgb_dataset_test", "rgb_dataset_test.json"),
]
IR_SETS = [
    ("svanstrom_ir", "svanstrom_ir.json"),
    ("antiuav_test_ir", "antiuav_test_ir.json"),
    ("ir_dset_final_test", "ir_dset_final_test.json"),
]

CONF = {"rgb": 0.30, "ir": 0.40}
IOU_TP = 0.3


def parse_gt(gt_str, w, h):
    out = []
    if not gt_str:
        return out
    for ln in gt_str.strip().splitlines():
        p = ln.split()
        if len(p) < 5:
            continue
        cx, cy, bw, bh = map(float, p[1:5])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        out.append((x1, y1, x2, y2))
    return out


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def collect(modality, sets):
    """Yield (short_side_px, short_side_norm, area_px, is_tp, dataset)."""
    conf_thr = CONF[modality]
    rows = []
    per_ds = defaultdict(lambda: {"tp": 0, "fp": 0, "gt_total": 0, "frames": 0})
    for tag, fname in sets:
        path = os.path.join(INF, fname)
        if not os.path.exists(path):
            print(f"missing {path}")
            continue
        d = json.load(open(path))
        for stem, v in d.items():
            w, h = v["w"], v["h"]
            gts = parse_gt(v.get("gt", ""), w, h)
            per_ds[tag]["frames"] += 1
            per_ds[tag]["gt_total"] += len(gts)
            dets = [det for det in v.get("dets", []) if det[4] >= conf_thr]
            # match each GT to best det (greedy, no double-count)
            used = set()
            for g in gts:
                best_i, best_iou = -1, IOU_TP
                for i, det in enumerate(dets):
                    if i in used:
                        continue
                    u = iou(det[:4], g)
                    if u > best_iou:
                        best_iou, best_i = u, i
                if best_i >= 0:
                    used.add(best_i)
            for i, det in enumerate(dets):
                x1, y1, x2, y2, sc = det
                bw, bh = x2 - x1, y2 - y1
                short = min(bw, bh)
                short_norm = short / min(w, h)
                area = bw * bh
                is_tp = i in used
                rows.append((short, short_norm, area, is_tp, tag))
                per_ds[tag]["tp" if is_tp else "fp"] += 1
    return rows, per_ds


def sweep(rows, total_gt, thresholds, key_idx):
    """key_idx: 0=short_px, 1=short_norm, 2=area_px"""
    out = []
    for thr in thresholds:
        tp = sum(1 for r in rows if r[key_idx] >= thr and r[3])
        fp = sum(1 for r in rows if r[key_idx] >= thr and not r[3])
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / total_gt if total_gt else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        out.append((thr, tp, fp, prec, rec, f1))
    return out


def report(modality, sets):
    rows, per_ds = collect(modality, sets)
    total_gt = sum(v["gt_total"] for v in per_ds.values())
    print(f"\n=== {modality.upper()} (conf>={CONF[modality]}) ===")
    for tag, v in per_ds.items():
        print(f"  {tag}: frames={v['frames']} gt={v['gt_total']} TP={v['tp']} FP={v['fp']}")
    print(f"  TOTAL: gt={total_gt} TP={sum(v['tp'] for v in per_ds.values())} FP={sum(v['fp'] for v in per_ds.values())}")

    print("\n  -- min short-side (px) --")
    print(f"  {'thr':>6} {'TP':>6} {'FP':>6} {'prec':>6} {'rec':>6} {'F1':>6}")
    for thr, tp, fp, p, r, f in sweep(rows, total_gt, [0, 2, 4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64], 0):
        print(f"  {thr:>6.1f} {tp:>6d} {fp:>6d} {p:>6.3f} {r:>6.3f} {f:>6.3f}")

    print("\n  -- min short-side (frac of min(W,H)) --")
    print(f"  {'thr':>7} {'TP':>6} {'FP':>6} {'prec':>6} {'rec':>6} {'F1':>6}")
    for thr, tp, fp, p, r, f in sweep(rows, total_gt, [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.075, 0.10], 1):
        print(f"  {thr:>7.4f} {tp:>6d} {fp:>6d} {p:>6.3f} {r:>6.3f} {f:>6.3f}")


if __name__ == "__main__":
    report("rgb", RGB_SETS)
    report("ir", IR_SETS)
