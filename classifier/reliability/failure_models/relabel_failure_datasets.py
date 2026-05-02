"""
relabel_failure_datasets.py — Re-label FN/FP datasets with a deployment-
realistic confidence threshold and asymmetric IoU matching.

Problem: build_failure_datasets.py ran inference at conf=0.001, so virtually
every GT gets a ghost detection match at IoU 0.2. This makes FN rates
artificially near-zero on some datasets (Anti-UAV: 0.1%, ir_dset: 0.0%).

Fix: reload cached inference JSONs, filter detections at --conf-thresh
(default 0.4 = deployment threshold), re-match with asymmetric IoU:
  - FN matching at --fn-iou (default 0.2): lenient, don't falsely label
    near-miss as "missed"
  - FP matching at --fp-iou (default 0.5): strict, don't label near-GT
    detections as hallucination

Features are NOT recomputed — only the "label" column (and FP's n_fp_in_frame)
are overwritten. This is fast (~1 min total, no image loading needed).

Outputs: overwrites {modality}_{kind}_dataset.csv in-place and writes
a relabel_summary.json with before/after stats.

Usage:
    python relabel_failure_datasets.py
    python relabel_failure_datasets.py --conf-thresh 0.4 --fn-iou 0.2 --fp-iou 0.5
    python relabel_failure_datasets.py --only rgb
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from run_all_inference import DATASETS  # noqa: E402

INFERENCE_DIR = SCRIPT_DIR.parent.parent / "runs" / "reliability" / "inference"
OUTPUT_DIR    = SCRIPT_DIR.parent.parent / "runs" / "reliability" / "failure_models"


# ── IoU + GT PARSING (duplicated from build script to stay self-contained) ──

def compute_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
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


# ── RELABEL LOGIC ─────────────────────────────────────────────────

def relabel_dataset(tag, fn_df_tag, fp_df_tag, conf_thresh, fn_iou, fp_iou):
    """Relabel FN/FP rows for one source_dataset tag.

    Returns updated (fn_df_tag, fp_df_tag) with new labels.
    """
    json_path = INFERENCE_DIR / f"{tag}.json"
    if not json_path.exists():
        print(f"    [SKIP] {tag}: inference JSON not found")
        return fn_df_tag, fp_df_tag

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Index FN rows by (stem, gt_idx) for fast lookup
    fn_index = {}
    if fn_df_tag is not None and len(fn_df_tag) > 0:
        for idx, row in fn_df_tag.iterrows():
            fn_index.setdefault(row["stem"], []).append(idx)

    # Index FP rows by stem
    fp_index = {}
    if fp_df_tag is not None and len(fp_df_tag) > 0:
        for idx, row in fp_df_tag.iterrows():
            fp_index[row["stem"]] = idx

    fn_changed = 0
    fp_changed = 0
    n_frames = 0

    for stem in data:
        frame = data[stem]
        all_dets = frame["dets"]
        img_w = frame["w"]
        img_h = frame["h"]
        gt_text = frame.get("gt", "")

        # Filter detections at conf threshold
        filtered_dets = [d for d in all_dets if d[4] >= conf_thresh]

        gt_boxes = parse_yolo_gt(gt_text, img_w, img_h)

        # --- FN relabel (lenient IoU) ---
        if stem in fn_index:
            gt_matched_fn, _ = match_gt_to_dets(gt_boxes, filtered_dets, fn_iou)
            row_indices = fn_index[stem]
            for ri in row_indices:
                gi = int(fn_df_tag.at[ri, "gt_idx"])
                new_label = 0 if (gi < len(gt_matched_fn) and gt_matched_fn[gi]) else 1
                old_label = int(fn_df_tag.at[ri, "label"])
                if new_label != old_label:
                    fn_changed += 1
                fn_df_tag.at[ri, "label"] = new_label

        # --- FP relabel (strict IoU) ---
        if stem in fp_index:
            _, det_matched_fp = match_gt_to_dets(gt_boxes, filtered_dets, fp_iou)
            n_fp = sum(1 for m in det_matched_fp if not m)
            ri = fp_index[stem]
            new_label = 1 if n_fp > 0 else 0
            old_label = int(fp_df_tag.at[ri, "label"])
            if new_label != old_label:
                fp_changed += 1
            fp_df_tag.at[ri, "label"] = new_label
            fp_df_tag.at[ri, "n_fp_in_frame"] = n_fp
            # Update n_dets_in_frame to reflect filtered count
            fp_df_tag.at[ri, "n_dets_in_frame"] = len(filtered_dets)

        n_frames += 1

    return fn_df_tag, fp_df_tag, fn_changed, fp_changed, n_frames


def get_datasets_for_modality(modality):
    return [(tag, img_dir) for tag, img_dir, lbl_dir, m in DATASETS if m == modality]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--conf-thresh", type=float, default=0.4,
                        help="Confidence threshold for filtering detections (default: 0.4)")
    parser.add_argument("--fn-iou", type=float, default=0.2,
                        help="IoU threshold for FN matching (default: 0.2, lenient)")
    parser.add_argument("--fp-iou", type=float, default=0.5,
                        help="IoU threshold for FP matching (default: 0.5, strict)")
    parser.add_argument("--only", choices=["rgb", "ir"],
                        help="Relabel only one modality")
    args = parser.parse_args()

    print("=" * 70)
    print("Relabel failure datasets with deployment-realistic thresholds")
    print("=" * 70)
    print(f"  Confidence threshold: {args.conf_thresh}")
    print(f"  FN IoU threshold:     {args.fn_iou} (lenient)")
    print(f"  FP IoU threshold:     {args.fp_iou} (strict)")
    print()

    modalities = [args.only] if args.only else ["rgb", "ir"]
    summary = {}

    # Back up original CSVs before any modifications
    backup_dir = OUTPUT_DIR / "backup_before_relabel"
    backup_dir.mkdir(parents=True, exist_ok=True)
    print("Backing up original CSVs...")
    for modality in modalities:
        for kind in ["fn", "fp"]:
            src = OUTPUT_DIR / f"{modality}_{kind}_dataset.csv"
            if src.exists():
                dst = backup_dir / f"{modality}_{kind}_dataset.csv"
                shutil.copy2(src, dst)
                print(f"  {src.name} -> backup_before_relabel/{dst.name}")
    print()

    for modality in modalities:
        print(f"\n{'=' * 70}")
        print(f"Relabeling {modality.upper()}")
        print(f"{'=' * 70}")

        fn_csv = OUTPUT_DIR / f"{modality}_fn_dataset.csv"
        fp_csv = OUTPUT_DIR / f"{modality}_fp_dataset.csv"

        if not fn_csv.exists() or not fp_csv.exists():
            print(f"  [SKIP] {modality}: CSVs not found. Run build_failure_datasets.py first.")
            continue

        fn_df = pd.read_csv(fn_csv)
        fp_df = pd.read_csv(fp_csv)

        old_fn_rate = (fn_df["label"] == 1).mean()
        old_fp_rate = (fp_df["label"] == 1).mean()

        print(f"  Loaded {len(fn_df):,} FN rows, {len(fp_df):,} FP rows")
        print(f"  BEFORE: FN rate = {old_fn_rate*100:.2f}%, FP rate = {old_fp_rate*100:.2f}%")

        dataset_list = get_datasets_for_modality(modality)
        total_fn_changed = 0
        total_fp_changed = 0

        t0 = time.time()
        for tag, _ in dataset_list:
            fn_tag = fn_df[fn_df["source_dataset"] == tag].copy()
            fp_tag = fp_df[fp_df["source_dataset"] == tag].copy()

            if len(fn_tag) == 0 and len(fp_tag) == 0:
                continue

            print(f"\n  Processing {tag}...", end="", flush=True)

            # Work on slices of the original dataframe
            fn_mask = fn_df["source_dataset"] == tag
            fp_mask = fp_df["source_dataset"] == tag

            fn_slice = fn_df.loc[fn_mask].copy()
            fp_slice = fp_df.loc[fp_mask].copy()

            result = relabel_dataset(
                tag, fn_slice, fp_slice,
                args.conf_thresh, args.fn_iou, args.fp_iou,
            )
            fn_slice, fp_slice, fn_ch, fp_ch, n_frames = result

            # Write back to master dataframes
            fn_df.loc[fn_mask, "label"] = fn_slice["label"].values
            fp_df.loc[fp_mask, "label"] = fp_slice["label"].values
            fp_df.loc[fp_mask, "n_fp_in_frame"] = fp_slice["n_fp_in_frame"].values
            fp_df.loc[fp_mask, "n_dets_in_frame"] = fp_slice["n_dets_in_frame"].values

            total_fn_changed += fn_ch
            total_fp_changed += fp_ch

            old_fn = (fn_tag["label"] == 1).mean() * 100
            new_fn = (fn_slice["label"] == 1).mean() * 100
            old_fp = (fp_tag["label"] == 1).mean() * 100
            new_fp = (fp_slice["label"] == 1).mean() * 100

            print(f" {n_frames:,} frames")
            print(f"    FN: {old_fn:.1f}% -> {new_fn:.1f}%  "
                  f"({fn_ch:,} labels changed / {len(fn_slice):,})")
            print(f"    FP: {old_fp:.1f}% -> {new_fp:.1f}%  "
                  f"({fp_ch:,} labels changed / {len(fp_slice):,})")

        elapsed = time.time() - t0
        new_fn_rate = (fn_df["label"] == 1).mean()
        new_fp_rate = (fp_df["label"] == 1).mean()

        print(f"\n  AFTER:  FN rate = {new_fn_rate*100:.2f}%, FP rate = {new_fp_rate*100:.2f}%")
        print(f"  Total labels changed: FN={total_fn_changed:,}, FP={total_fp_changed:,}")
        print(f"  Time: {elapsed:.1f}s")

        # Overwrite CSVs
        fn_df.to_csv(fn_csv, index=False)
        fp_df.to_csv(fp_csv, index=False)
        print(f"  Saved: {fn_csv.name}, {fp_csv.name}")

        # Per-dataset summary
        print(f"\n  Per-dataset after relabel:")
        print(f"    {'dataset':<25s} {'FN_rows':>8s} {'FN%':>7s} {'FP_rows':>8s} {'FP%':>7s}")
        for tag, _ in dataset_list:
            fn_s = fn_df[fn_df["source_dataset"] == tag]
            fp_s = fp_df[fp_df["source_dataset"] == tag]
            if len(fn_s) == 0 and len(fp_s) == 0:
                continue
            fn_r = (fn_s["label"] == 1).mean() * 100 if len(fn_s) > 0 else 0
            fp_r = (fp_s["label"] == 1).mean() * 100 if len(fp_s) > 0 else 0
            print(f"    {tag:<25s} {len(fn_s):>8,} {fn_r:>6.1f}% {len(fp_s):>8,} {fp_r:>6.1f}%")

        summary[modality] = {
            "conf_thresh": args.conf_thresh,
            "fn_iou": args.fn_iou,
            "fp_iou": args.fp_iou,
            "fn_rows": int(len(fn_df)),
            "fp_rows": int(len(fp_df)),
            "old_fn_rate": float(old_fn_rate),
            "new_fn_rate": float(new_fn_rate),
            "old_fp_rate": float(old_fp_rate),
            "new_fp_rate": float(new_fp_rate),
            "fn_labels_changed": total_fn_changed,
            "fp_labels_changed": total_fp_changed,
        }

    # Save summary
    summary_path = OUTPUT_DIR / "relabel_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path.name}")
    print("Done. Ready for train_failure_models.py")


if __name__ == "__main__":
    main()
