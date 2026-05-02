"""
sweep_conf_threshold.py — Sweep confidence thresholds to find the optimal
operating point for FN/FP relabeling.

For each threshold in the sweep range, filters detections at conf >= threshold,
re-matches GT with asymmetric IoU (FN: 0.2 lenient, FP: 0.5 strict), and
reports per-dataset FN/FP rates.

Outputs:
  runs/reliability/failure_models/conf_sweep_results.json

Usage:
    python sweep_conf_threshold.py
    python sweep_conf_threshold.py --only rgb
    python sweep_conf_threshold.py --thresholds 0.1 0.2 0.3 0.4 0.5
"""

import argparse
import json
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from run_all_inference import DATASETS  # noqa: E402

INFERENCE_DIR = SCRIPT_DIR.parent.parent / "runs" / "reliability" / "inference"
OUTPUT_DIR    = SCRIPT_DIR.parent.parent / "runs" / "reliability" / "failure_models"


# ── IoU + GT PARSING ──────────────────────────────────────────────

def compute_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def parse_yolo_gt(gt_text, img_w, img_h):
    boxes = []
    if not gt_text.strip():
        return boxes
    for line in gt_text.strip().split("\n"):
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        boxes.append([x1, y1, x2, y2])
    return boxes


def match_gt_to_dets(gt_boxes, dets, iou_thresh):
    n_gt = len(gt_boxes)
    n_det = len(dets)
    gt_matched = [False] * n_gt
    det_matched = [False] * n_det

    if n_gt == 0 or n_det == 0:
        return gt_matched, det_matched

    pairs = []
    for gi in range(n_gt):
        for di in range(n_det):
            iou = compute_iou(gt_boxes[gi], dets[di][:4])
            if iou >= iou_thresh:
                pairs.append((iou, gi, di))
    pairs.sort(reverse=True)

    for _, gi, di in pairs:
        if not gt_matched[gi] and not det_matched[di]:
            gt_matched[gi] = True
            det_matched[di] = True

    return gt_matched, det_matched


# ── SWEEP ─────────────────────────────────────────────────────────

def sweep_one_dataset(tag, data, thresholds, fn_iou, fp_iou):
    """Sweep all thresholds for one dataset. Returns dict[thresh] -> stats."""
    results = {}
    n_frames = len(data)

    for thresh in thresholds:
        n_gt = 0
        n_fn = 0
        n_fp_frames = 0
        n_frames_proc = 0

        for stem, frame in data.items():
            filtered_dets = [d for d in frame["dets"] if d[4] >= thresh]
            gt_boxes = parse_yolo_gt(frame.get("gt", ""), frame["w"], frame["h"])

            # FN matching (lenient)
            gt_m, _ = match_gt_to_dets(gt_boxes, filtered_dets, fn_iou)
            n_gt += len(gt_boxes)
            n_fn += sum(1 for m in gt_m if not m)

            # FP matching (strict)
            _, det_m = match_gt_to_dets(gt_boxes, filtered_dets, fp_iou)
            n_fp = sum(1 for m in det_m if not m)
            n_frames_proc += 1
            if n_fp > 0:
                n_fp_frames += 1

        fn_rate = n_fn / n_gt * 100 if n_gt > 0 else 0.0
        fp_rate = n_fp_frames / n_frames_proc * 100 if n_frames_proc > 0 else 0.0

        results[thresh] = {
            "n_gt": n_gt,
            "n_fn": n_fn,
            "fn_rate": round(fn_rate, 2),
            "n_frames": n_frames_proc,
            "n_fp_frames": n_fp_frames,
            "fp_rate": round(fp_rate, 2),
        }

    return results


def print_table(modality, tags, all_results, thresholds, metric, label):
    """Print a formatted table for one metric (fn_rate or fp_rate)."""
    # Column headers
    short_names = []
    for tag in tags:
        short = tag.replace(f"_{modality}", "").replace("_final", "")
        if len(short) > 15:
            short = short[:15]
        short_names.append(short)

    header = f"  {'conf':>6s}"
    for s in short_names:
        header += f" {s:>15s}"
    header += f" {'OVERALL':>10s}"
    print(header)

    for thresh in thresholds:
        row = f"  {thresh:>6.2f}"
        total_num = 0
        total_den = 0
        num_key = "n_fn" if metric == "fn_rate" else "n_fp_frames"
        den_key = "n_gt" if metric == "fn_rate" else "n_frames"

        for tag in tags:
            r = all_results[tag][thresh]
            row += f" {r[metric]:>14.1f}%"
            total_num += r[num_key]
            total_den += r[den_key]

        overall = total_num / total_den * 100 if total_den > 0 else 0
        row += f" {overall:>9.1f}%"
        print(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
                        help="Confidence thresholds to sweep")
    parser.add_argument("--fn-iou", type=float, default=0.2,
                        help="IoU threshold for FN matching (default: 0.2)")
    parser.add_argument("--fp-iou", type=float, default=0.5,
                        help="IoU threshold for FP matching (default: 0.5)")
    parser.add_argument("--only", choices=["rgb", "ir"],
                        help="Sweep only one modality")
    args = parser.parse_args()

    thresholds = sorted(args.thresholds)

    print("=" * 70)
    print("Confidence threshold sweep for FN/FP relabeling")
    print("=" * 70)
    print(f"  Thresholds: {thresholds}")
    print(f"  FN IoU: {args.fn_iou} (lenient)")
    print(f"  FP IoU: {args.fp_iou} (strict)")

    modalities = [args.only] if args.only else ["rgb", "ir"]
    full_results = {}

    for modality in modalities:
        print(f"\n{'=' * 70}")
        print(f"{modality.upper()} confidence threshold sweep")
        print(f"{'=' * 70}")

        dataset_list = [(t, i) for t, i, l, m in DATASETS if m == modality]

        # Load inference JSONs
        all_data = {}
        for tag, _ in dataset_list:
            jp = INFERENCE_DIR / f"{tag}.json"
            if not jp.exists():
                print(f"  [SKIP] {tag}: no inference JSON")
                continue
            print(f"  Loading {tag}...", end="", flush=True)
            with open(jp, "r", encoding="utf-8") as f:
                all_data[tag] = json.load(f)
            print(f" {len(all_data[tag]):,} frames")

        # Sweep each dataset
        all_results = {}
        tags = list(all_data.keys())
        t0 = time.time()

        for i, tag in enumerate(tags):
            print(f"  Sweeping {tag} ({i+1}/{len(tags)})...", end="", flush=True)
            t_tag = time.time()
            all_results[tag] = sweep_one_dataset(
                tag, all_data[tag], thresholds, args.fn_iou, args.fp_iou,
            )
            elapsed_tag = time.time() - t_tag
            print(f" {elapsed_tag:.1f}s")

        elapsed = time.time() - t0
        print(f"\n  Total sweep time: {elapsed:.1f}s")

        # Print tables
        print(f"\nFN rate (%) by conf threshold (FN IoU={args.fn_iou}):")
        print_table(modality, tags, all_results, thresholds, "fn_rate", "FN")

        print(f"\nFP frame rate (%) by conf threshold (FP IoU={args.fp_iou}):")
        print_table(modality, tags, all_results, thresholds, "fp_rate", "FP")

        full_results[modality] = {
            tag: {str(t): v for t, v in res.items()}
            for tag, res in all_results.items()
        }

    # Save results
    out_path = OUTPUT_DIR / "conf_sweep_results.json"
    with open(out_path, "w") as f:
        json.dump(full_results, f, indent=2)
    print(f"\nResults saved to {out_path.name}")
    print("Done.")


if __name__ == "__main__":
    main()
