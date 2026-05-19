"""
eval_video_tests.py — Evaluate RGB models on all video test datasets.

Handles both:
  - Negative-only (confuser) categories: airplanes, birds, helicopters
    -> Every detection = FP. Reports FPPI, FP frame rate.
  - Positive (drone) category: has GT labels
    -> Reports TP/FP/FN, Precision/Recall/F1 via IoU and IoP matching.

Also runs a confidence sweep to find optimal threshold per model.

Usage:
    python eval/eval_video_tests.py
    python eval/eval_video_tests.py --categories drone
    python eval/eval_video_tests.py --categories airplanes birds helicopters
    python eval/eval_video_tests.py --models retrained_v2 selcom_1280
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent

sys.path.insert(0, str(EVAL_DIR))
from metrics import (compute_prf, score_detections, compute_frame_metrics,
                     classify_size, iou_iop, SIZE_BUCKETS)
from datasets import ImageDataset, read_yolo_labels

# ── Model registry ───────────────────────────────────────────────
MODELS = {
    "baseline_trained": {
        "weights": str(REPO / "RGB model" / "Yolo26n_trained" / "weights" / "best.pt"),
        "imgsz": 640,
        "grayscale": False,
    },
    "retrained_v2": {
        "weights": str(REPO / "RGB model" / "Yolo26n_retrained_v2" / "weights" / "best.pt"),
        "imgsz": 640,
        "grayscale": False,
    },
    "selcom_1280": {
        "weights": str(REPO / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt"),
        "imgsz": 1280,
        "grayscale": False,
    },
    "selcom_640": {
        "weights": str(REPO / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt"),
        "imgsz": 640,
        "grayscale": False,
    },
    "ir_final_gray": {
        "weights": str(REPO / "models" / "IR_final_cleaned" / "weights" / "best.pt"),
        "imgsz": 640,
        "grayscale": True,
    },
    "ir_final_rgb": {
        "weights": str(REPO / "models" / "IR_final_cleaned" / "weights" / "best.pt"),
        "imgsz": 640,
        "grayscale": False,
    },
}

DATASET_ROOT = REPO / "datasets" / "drone detection video tests" / "rgb"
NEGATIVE_CATEGORIES = {"airplanes", "birds", "helicopters"}
POSITIVE_CATEGORIES = {"drone"}
ALL_CATEGORIES = ["airplanes", "birds", "drone", "helicopters"]


def eval_on_dataset(model, model_name: str, imgsz: int,
                    ds_path: Path, category: str,
                    base_conf: float, device: str,
                    grayscale: bool = False,
                    prod_conf: float = 0.25) -> dict:
    """Evaluate a model on a single dataset.

    For negative categories: all dets = FP.
    For positive categories: match dets to GT labels via IoU/IoP.

    Runs TWO inference passes per frame:
      (a) at base_conf (low) -- detection set used for the conf sweep
          (sweep filters post-hoc; valid because all candidates are present)
      (b) at prod_conf (production threshold, default 0.25) -- detection set
          used for the headline `iop_25` / `iou_25` metric

    The two-pass approach exists because YOLO/Ultralytics applies the conf
    threshold *before* NMS, so a conf=0.05 inference filtered post-hoc to
    conf>=0.25 is NOT guaranteed to match a conf=0.25 inference directly.
    The pipeline-side eval (`eval_pipeline_video_tests.py`) runs YOLO at the
    production threshold; we match that here for cross-script consistency.
    """
    is_negative = category in NEGATIVE_CATEGORIES

    img_dir = ds_path / "images" / "test"
    lbl_dir = ds_path / "labels" / "test"

    if not img_dir.exists():
        return {}

    ds = ImageDataset(img_dir, lbl_dir)
    images = ds.list_images()
    if not images:
        return {}

    total_frames = len(images)

    # Accumulators
    all_dets_with_gt = []  # [(conf, matched_iou, matched_iop), ...] -- for legacy/reference
    per_frame_data = []  # [(dets_list, gt_list), ...] from base_conf inference -- for the sweep
    per_frame_data_prod = []  # [(dets_list, gt_list), ...] from prod_conf inference -- for iop_25
    totals = {r: {"tp": 0, "fp": 0, "fn": 0} for r in ("iou", "iop")}
    frame_counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    sizes = {"small": 0, "medium": 0, "large": 0}
    total_gt_boxes = 0
    total_dets = 0
    total_dets_prod = 0
    fp_frames = 0

    t0 = time.time()
    for idx, img_path in enumerate(images):
        frame = ds.load_frame(img_path)
        if frame is None:
            continue
        img = frame["img"]
        w, h = frame["w"], frame["h"]
        stem = frame["stem"]

        # Read GT labels
        if is_negative:
            gt = []  # No GT for confusers
        else:
            lbl_path = lbl_dir / f"{stem}.txt"
            gt = read_yolo_labels(lbl_path, w, h, drone_classes={0})

        total_gt_boxes += len(gt)

        # Optional grayscale conversion (for IR model on RGB footage)
        if grayscale:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Run inference at base_conf (for the conf sweep)
        res = model.predict(img, conf=base_conf, verbose=False,
                            imgsz=imgsz, device=device)
        boxes = res[0].boxes
        dets = []
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf_val = float(boxes.conf[i])
            dets.append(((float(xyxy[0]), float(xyxy[1]),
                          float(xyxy[2]), float(xyxy[3])), conf_val))

        total_dets += len(dets)

        # Run a second inference pass at prod_conf for the production iop_25 metric.
        # Cannot be derived by post-hoc filtering of the base_conf pass because
        # Ultralytics applies the conf threshold pre-NMS (different NMS candidate
        # pool yields different post-NMS detection sets).
        if abs(prod_conf - base_conf) < 1e-6:
            dets_prod = list(dets)
        else:
            res_prod = model.predict(img, conf=prod_conf, verbose=False,
                                     imgsz=imgsz, device=device)
            boxes_prod = res_prod[0].boxes
            dets_prod = []
            for i in range(len(boxes_prod)):
                xyxy = boxes_prod.xyxy[i].cpu().numpy()
                conf_val = float(boxes_prod.conf[i])
                dets_prod.append(((float(xyxy[0]), float(xyxy[1]),
                                   float(xyxy[2]), float(xyxy[3])), conf_val))
        total_dets_prod += len(dets_prod)
        per_frame_data_prod.append((dets_prod, gt))

        # Size distribution
        for d_box, _ in dets:
            sz = classify_size(d_box, w, h)
            sizes[sz] += 1

        # Store conf records for sweep
        for d_box, d_conf in dets:
            best_iu = best_ip = 0.0
            for g in gt:
                iu, ip = iou_iop(d_box, g)
                best_iu = max(best_iu, iu)
                best_ip = max(best_ip, ip)
            all_dets_with_gt.append((d_conf, best_iu >= 0.5, best_ip >= 0.5))

        per_frame_data.append((dets, gt))

        # Detection-level scoring at base_conf
        for rule in ("iou", "iop"):
            tp, fp, fn = score_detections(dets, gt, rule=rule,
                                          iou_thr=0.5, iop_thr=0.5)
            totals[rule]["tp"] += tp
            totals[rule]["fp"] += fp
            totals[rule]["fn"] += fn

        # Frame-level
        has_det = len(dets) > 0
        has_gt = len(gt) > 0
        ftp, ffp, ffn, ftn = compute_frame_metrics(has_det, has_gt)
        frame_counts["tp"] += ftp
        frame_counts["fp"] += ffp
        frame_counts["fn"] += ffn
        frame_counts["tn"] += ftn
        if has_det and not has_gt:
            fp_frames += 1
        elif has_det and is_negative:
            fp_frames += 1

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            fps = (idx + 1) / elapsed
            print(f"      {idx+1:>5d}/{total_frames}  {fps:.1f} fps")

    elapsed = time.time() - t0

    # Compute metrics at prod_conf (default 0.25) using YOLO inference run at that
    # threshold directly (matches the pipeline eval's NMS candidate pool).
    # Proper per-detection bipartite matching via score_detections.
    tp_iou_25 = fp_iou_25 = fn_iou_25 = 0
    tp_iop_25 = fp_iop_25 = fn_iop_25 = 0
    for dets_f, gt_f in per_frame_data_prod:
        tp, fp, fn = score_detections(dets_f, gt_f, rule="iou", iou_thr=0.5)
        tp_iou_25 += tp; fp_iou_25 += fp; fn_iou_25 += fn
        tp, fp, fn = score_detections(dets_f, gt_f, rule="iop", iop_thr=0.5)
        tp_iop_25 += tp; fp_iop_25 += fp; fn_iop_25 += fn

    m_iou_25 = compute_prf(tp_iou_25, fp_iou_25, fn_iou_25)
    m_iop_25 = compute_prf(tp_iop_25, fp_iop_25, fn_iop_25)

    fppi = (fp_iou_25) / max(total_frames, 1) if is_negative else None

    # Confidence sweep using proper bipartite matching
    sweep_thresholds = [round(t, 2) for t in np.arange(0.05, 0.96, 0.05)]
    sweep_results = []
    for thr in sweep_thresholds:
        s_tp_iu = s_fp_iu = s_fn_iu = 0
        s_tp_ip = s_fp_ip = s_fn_ip = 0
        n_dets_at_thr = 0
        for dets_f, gt_f in per_frame_data:
            dets_filtered = [(box, c) for box, c in dets_f if c >= thr]
            n_dets_at_thr += len(dets_filtered)
            tp, fp, fn = score_detections(dets_filtered, gt_f, rule="iou", iou_thr=0.5)
            s_tp_iu += tp; s_fp_iu += fp; s_fn_iu += fn
            tp, fp, fn = score_detections(dets_filtered, gt_f, rule="iop", iop_thr=0.5)
            s_tp_ip += tp; s_fp_ip += fp; s_fn_ip += fn

        m_iu = compute_prf(s_tp_iu, s_fp_iu, s_fn_iu)
        m_ip = compute_prf(s_tp_ip, s_fp_ip, s_fn_ip)

        sweep_results.append({
            "conf": thr,
            "n_dets": n_dets_at_thr,
            "iou": {"tp": s_tp_iu, "fp": s_fp_iu, "fn": s_fn_iu, **m_iu},
            "iop": {"tp": s_tp_ip, "fp": s_fp_ip, "fn": s_fn_ip, **m_ip},
        })

    # Find optimal conf (best F1 for positive, lowest FP for negative)
    if is_negative:
        # For negatives: find conf where FP drops to near-zero
        best_sweep = min(sweep_results,
                         key=lambda s: (s["iou"]["fp"], -s["conf"]))
    else:
        best_sweep = max(sweep_results,
                         key=lambda s: s["iop"]["f1"])

    # Confidence values for stats
    all_confs = [c for c, _, _ in all_dets_with_gt]

    return {
        "model": model_name,
        "dataset": ds_path.name,
        "category": category,
        "is_negative": is_negative,
        "total_frames": total_frames,
        "total_gt": total_gt_boxes,
        "total_dets_raw": total_dets,           # at base_conf (sweep pass)
        "total_dets_prod": total_dets_prod,     # at prod_conf (production pass)
        "base_conf": base_conf,
        "prod_conf": prod_conf,
        # Standard metrics at prod_conf (default 0.25), proper bipartite matching,
        # YOLO inference run AT prod_conf (matches the pipeline-eval NMS pool).
        "iou_25": {"tp": tp_iou_25, "fp": fp_iou_25, "fn": fn_iou_25, **m_iou_25},
        "iop_25": {"tp": tp_iop_25, "fp": fp_iop_25, "fn": fn_iop_25, **m_iop_25},
        "fppi": round(fp_iou_25 / max(total_frames, 1), 4) if is_negative else None,
        "fp_frame_rate": round(frame_counts["fp"] / max(frame_counts["fp"] + frame_counts["tn"], 1), 4),
        "frame_counts": frame_counts,
        "sizes": sizes,
        "max_conf": round(max(all_confs), 4) if all_confs else 0.0,
        "mean_conf": round(float(np.mean(all_confs)), 4) if all_confs else 0.0,
        # Sweep
        "sweep": sweep_results,
        "best_conf": best_sweep["conf"],
        "best_sweep": best_sweep,
        "elapsed_s": round(elapsed, 1),
    }


def print_positive_summary(rows, models_to_run):
    """Print summary for drone (positive) datasets."""
    print(f"\n  DRONE VIDEOS (Positive - IoP matching, conf=0.25):")
    print(f"  {'Dataset':<42s} {'Model':<18s} {'Frames':>6s} {'GT':>5s} "
          f"{'TP':>5s} {'FP':>5s} {'FN':>5s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s} {'BestC':>6s}")
    print(f"  {'-'*120}")
    for row in rows:
        m = row["iop_25"]
        print(f"  {row['dataset']:<42s} {row['model']:<18s} "
              f"{row['total_frames']:>6d} {row['total_gt']:>5d} "
              f"{m['TP']:>5d} {m['FP']:>5d} {m['FN']:>5d} "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f} "
              f"{row['best_conf']:>6.2f}")


def print_negative_summary(rows, models_to_run):
    """Print summary for confuser (negative) datasets."""
    print(f"\n  CONFUSER VIDEOS (Negative-only, conf=0.25):")
    print(f"  {'Dataset':<42s} {'Model':<18s} {'Frames':>6s} "
          f"{'FP':>6s} {'FPPI':>7s} {'FPR':>7s} {'MaxC':>6s} {'BestC':>6s}")
    print(f"  {'-'*100}")
    for row in rows:
        m = row["iou_25"]
        print(f"  {row['dataset']:<42s} {row['model']:<18s} "
              f"{row['total_frames']:>6d} "
              f"{m['FP']:>6d} {row['fppi']:>7.4f} "
              f"{row['fp_frame_rate']:>7.4f} {row['max_conf']:>6.3f} "
              f"{row['best_conf']:>6.2f}")


def print_aggregate(rows, models_to_run, category_filter=None, label=""):
    """Print per-model aggregate."""
    if category_filter:
        rows = [r for r in rows if r["category"] in category_filter]
    if not rows:
        return

    is_neg = rows[0]["is_negative"]
    print(f"\n  {label}")

    if is_neg:
        print(f"  {'Model':<18s} {'Frames':>7s} {'FP':>7s} {'FPPI':>7s} {'FPR':>7s}")
        print(f"  {'-'*50}")
    else:
        print(f"  {'Model':<18s} {'Frames':>7s} {'GT':>6s} {'TP':>6s} {'FP':>6s} "
              f"{'FN':>6s} {'Prec':>6s} {'Rec':>6s} {'F1':>6s}")
        print(f"  {'-'*75}")

    for model_name in models_to_run:
        model_rows = [r for r in rows if r["model"] == model_name]
        if not model_rows:
            continue
        tot_frames = sum(r["total_frames"] for r in model_rows)

        if is_neg:
            tot_fp = sum(r["iou_25"]["FP"] for r in model_rows)
            fppi = tot_fp / max(tot_frames, 1)
            tot_fp_fr = sum(r["frame_counts"]["fp"] for r in model_rows)
            tot_tn_fr = sum(r["frame_counts"]["tn"] for r in model_rows)
            fpr = tot_fp_fr / max(tot_fp_fr + tot_tn_fr, 1)
            print(f"  {model_name:<18s} {tot_frames:>7d} {tot_fp:>7d} "
                  f"{fppi:>7.4f} {fpr:>7.4f}")
        else:
            tot_gt = sum(r["total_gt"] for r in model_rows)
            tot_tp = sum(r["iop_25"]["TP"] for r in model_rows)
            tot_fp = sum(r["iop_25"]["FP"] for r in model_rows)
            tot_fn = sum(r["iop_25"]["FN"] for r in model_rows)
            m = compute_prf(tot_tp, tot_fp, tot_fn)
            print(f"  {model_name:<18s} {tot_frames:>7d} {tot_gt:>6d} "
                  f"{tot_tp:>6d} {tot_fp:>6d} {tot_fn:>6d} "
                  f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}")


def print_conf_sweep_summary(rows, models_to_run):
    """Print optimal confidence per model across all datasets."""
    print(f"\n  CONFIDENCE SWEEP SUMMARY (optimal per model):")

    for model_name in models_to_run:
        model_rows = [r for r in rows if r["model"] == model_name]
        if not model_rows:
            continue

        pos_rows = [r for r in model_rows if not r["is_negative"]]
        neg_rows = [r for r in model_rows if r["is_negative"]]

        # Aggregate sweep across all positive datasets
        if pos_rows:
            sweep_thresholds = [s["conf"] for s in pos_rows[0]["sweep"]]
            best_f1 = 0
            best_thr = 0.25
            for thr in sweep_thresholds:
                tot_tp = tot_fp = tot_fn = 0
                for r in pos_rows:
                    s = next(x for x in r["sweep"] if x["conf"] == thr)
                    tot_tp += s["iop"]["tp"]
                    tot_fp += s["iop"]["fp"]
                    tot_fn += s["iop"]["fn"]
                m = compute_prf(tot_tp, tot_fp, tot_fn)
                if m["f1"] > best_f1:
                    best_f1 = m["f1"]
                    best_thr = thr
                    best_m = m

            print(f"\n  {model_name} (DRONE - best F1):")
            print(f"    Optimal conf: {best_thr:.2f}")
            print(f"    P={best_m['precision']:.4f}  R={best_m['recall']:.4f}  F1={best_m['f1']:.4f}")

        # Aggregate sweep across all negative datasets
        if neg_rows:
            sweep_thresholds = [s["conf"] for s in neg_rows[0]["sweep"]]
            print(f"\n  {model_name} (CONFUSERS - FP at each conf):")
            print(f"    {'conf':>6s} {'FP':>7s} {'FPPI':>8s}")
            tot_frames_neg = sum(r["total_frames"] for r in neg_rows)
            for thr in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
                tot_fp = 0
                for r in neg_rows:
                    s = next((x for x in r["sweep"] if abs(x["conf"] - thr) < 0.01), None)
                    if s:
                        tot_fp += s["iou"]["fp"]
                fppi = tot_fp / max(tot_frames_neg, 1)
                print(f"    {thr:>6.2f} {tot_fp:>7d} {fppi:>8.4f}")


def main():
    ap = argparse.ArgumentParser(description="Evaluate RGB models on video test datasets")
    ap.add_argument("--base-conf", type=float, default=0.05,
                    help="Base confidence for sweep inference pass (default: 0.05)")
    ap.add_argument("--prod-conf", type=float, default=0.25,
                    help="Production confidence threshold for the headline iop_25 metric (default: 0.25). "
                         "A separate YOLO inference pass is run at this threshold; the result is what "
                         "the pipeline eval (eval_pipeline_video_tests.py) operates on.")
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--categories", nargs="*", default=None,
                    help="Categories to evaluate (default: all)")
    ap.add_argument("--models", nargs="*", default=None,
                    help="Models to evaluate (default: all)")
    ap.add_argument("--videos", nargs="*", default=None,
                    help="Specific video names (default: all)")
    args = ap.parse_args()

    if not DATASET_ROOT.exists():
        print(f"ERROR: {DATASET_ROOT} not found.")
        return

    categories = args.categories or ALL_CATEGORIES
    models_to_run = MODELS
    if args.models:
        models_to_run = {k: v for k, v in MODELS.items() if k in args.models}

    # Discover datasets
    datasets = []
    for cat in categories:
        cat_dir = DATASET_ROOT / cat
        if not cat_dir.exists():
            continue
        for ds_dir in sorted(cat_dir.iterdir()):
            if ds_dir.is_dir() and (ds_dir / "images" / "test").exists():
                if args.videos and ds_dir.name not in args.videos:
                    continue
                datasets.append((cat, ds_dir))

    print(f"Video Test Evaluation")
    print(f"  Datasets:   {len(datasets)}")
    print(f"  Models:     {len(models_to_run)}")
    print(f"  Base conf:  {args.base_conf} (sweep from 0.05 to 0.95)")
    print(f"  Prod conf:  {args.prod_conf} (headline iop_25 metric)")
    print(f"  Device:     {args.device}")

    out_dir = EVAL_DIR / "results" / "video_tests"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows = []

    # Load models once
    from ultralytics import YOLO
    loaded_models = {}
    for model_name, model_info in models_to_run.items():
        print(f"\n  Loading {model_name}...")
        loaded_models[model_name] = YOLO(model_info["weights"])

    for cat, ds_dir in datasets:
        ds_name = ds_dir.name
        n_imgs = len(list((ds_dir / "images" / "test").glob("*.jpg")))

        print(f"\n{'='*70}")
        print(f"  {cat}/{ds_name}  ({n_imgs} frames, {'NEGATIVE' if cat in NEGATIVE_CATEGORIES else 'POSITIVE'})")
        print(f"{'='*70}")

        vid_out = out_dir / cat / ds_name
        vid_out.mkdir(parents=True, exist_ok=True)

        for model_name, model_info in models_to_run.items():
            gs = model_info.get("grayscale", False)
            gs_tag = " [GRAYSCALE]" if gs else ""
            print(f"\n    {model_name} (imgsz={model_info['imgsz']}{gs_tag})")
            result = eval_on_dataset(
                loaded_models[model_name], model_name, model_info["imgsz"],
                ds_dir, cat, args.base_conf, args.device,
                grayscale=gs, prod_conf=args.prod_conf)
            if not result:
                continue

            # Quick inline summary
            if result["is_negative"]:
                m = result["iou_25"]
                print(f"    >> FP={m['FP']}  FPPI={result['fppi']:.4f}  "
                      f"best_conf={result['best_conf']:.2f}")
            else:
                m = result["iop_25"]
                print(f"    >> TP={m['TP']}  FP={m['FP']}  FN={m['FN']}  "
                      f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  "
                      f"best_conf={result['best_conf']:.2f}")

            # Save per-video per-model JSON
            save_result = {k: v for k, v in result.items() if k != "sweep"}
            save_result["sweep_summary"] = [
                {"conf": s["conf"], "iop_f1": s["iop"]["f1"],
                 "iou_fp": s["iou"]["fp"], "iop_tp": s["iop"]["tp"]}
                for s in result["sweep"]
            ]
            json_path = vid_out / f"{model_name}.json"
            with open(json_path, "w") as f:
                json.dump(save_result, f, indent=2)

            all_rows.append(result)

    # ── Summary tables ───────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}")

    pos_rows = [r for r in all_rows if not r["is_negative"]]
    neg_rows = [r for r in all_rows if r["is_negative"]]

    if pos_rows:
        print_positive_summary(pos_rows, models_to_run)
    if neg_rows:
        print_negative_summary(neg_rows, models_to_run)

    # Per-model aggregates
    if pos_rows:
        print_aggregate(pos_rows, models_to_run,
                        category_filter=POSITIVE_CATEGORIES,
                        label="AGGREGATE: DRONE (all videos)")
    for cat in sorted(NEGATIVE_CATEGORIES & set(categories)):
        cat_rows = [r for r in neg_rows if r["category"] == cat]
        if cat_rows:
            print_aggregate(cat_rows, models_to_run,
                            category_filter={cat},
                            label=f"AGGREGATE: {cat.upper()}")
    if neg_rows:
        print_aggregate(neg_rows, models_to_run,
                        category_filter=NEGATIVE_CATEGORIES,
                        label="AGGREGATE: ALL CONFUSERS")

    # Confidence sweep
    print_conf_sweep_summary(all_rows, models_to_run)

    # Save CSV
    csv_path = out_dir / "video_tests_comparison.csv"
    if all_rows:
        fields = ["dataset", "category", "model", "is_negative",
                   "total_frames", "total_gt",
                   "iop_tp", "iop_fp", "iop_fn", "iop_prec", "iop_rec", "iop_f1",
                   "fppi", "fp_frame_rate", "best_conf", "max_conf", "elapsed_s"]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in all_rows:
                flat = {
                    "dataset": row["dataset"], "category": row["category"],
                    "model": row["model"], "is_negative": row["is_negative"],
                    "total_frames": row["total_frames"], "total_gt": row["total_gt"],
                    "iop_tp": row["iop_25"]["TP"], "iop_fp": row["iop_25"]["FP"],
                    "iop_fn": row["iop_25"]["FN"],
                    "iop_prec": row["iop_25"]["precision"],
                    "iop_rec": row["iop_25"]["recall"],
                    "iop_f1": row["iop_25"]["f1"],
                    "fppi": row["fppi"], "fp_frame_rate": row["fp_frame_rate"],
                    "best_conf": row["best_conf"], "max_conf": row["max_conf"],
                    "elapsed_s": row["elapsed_s"],
                }
                w.writerow(flat)
        print(f"\n  Saved: {csv_path}")

    # Save full JSON
    json_rows = [{k: v for k, v in r.items() if k != "sweep"} for r in all_rows]
    json_path = out_dir / "video_tests_comparison.json"
    with open(json_path, "w") as f:
        json.dump(json_rows, f, indent=2)
    print(f"  Saved: {json_path}")

    print(f"\n[eval_video_tests] Done.")


if __name__ == "__main__":
    main()
