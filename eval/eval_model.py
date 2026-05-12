"""
eval_model.py — Raw YOLO model benchmarking on datasets.

Supports:
- Single or multi-model comparison
- IoU and IoP matching
- Per-source breakdown
- Confidence threshold sweep
- Size distribution (small/medium/large)
- FPPI on negative-only datasets
- Frame-level TP/FP/FN/TN
- Patch verifier (confuser filter) — raw vs filtered metrics side-by-side
- Temporal alert gate — simulated N-of-M window on sequential images
- Multi-class label remapping (--drone-classes)

Usage:
    python eval/eval_model.py --weights best.pt --dataset G:/drone/test
    python eval/eval_model.py --weights a.pt b.pt --dataset G:/drone/test
    python eval/eval_model.py --weights best.pt --dataset path --stride 10
    python eval/eval_model.py --weights best.pt --dataset path --negatives-only
    python eval/eval_model.py --weights best.pt --dataset path --patch-weights filter.pt --patch-thr 0.70
    python eval/eval_model.py --weights best.pt --dataset path --temporal --patch-weights filter.pt
    python eval/eval_model.py --weights best.pt --dataset path --drone-classes 0,1
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

from metrics import (
    score_detections, compute_prf, compute_frame_metrics,
    size_distribution, classify_size, iou_iop, score_per_size,
    SIZE_BUCKETS,
)
from datasets import load_config, resolve_path, ImageDataset, detect_category, read_yolo_labels
from run_manifest import write_manifest
from reporting import (
    print_metrics_table, print_size_distribution,
    save_metrics_csv, save_json,
    plot_metrics_bars, plot_confusion_matrices, plot_size_distribution,
)


def run_yolo_val(weights: str, dataset: str, imgsz: int, device: str):
    """Run YOLO val mode (mAP) using ultralytics built-in."""
    from ultralytics import YOLO
    model = YOLO(weights)
    results = model.val(data=dataset, imgsz=imgsz, device=device, verbose=True)
    print(f"\n  mAP50:    {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    return results


def evaluate_model(
    weights_path: str,
    dataset_path: str,
    args,
    model_name: str = "",
    drone_classes: set[int] | None = None,
    patch_verifier=None,
    patch_thr: float = 0.70,
    enable_temporal: bool = False,
) -> dict:
    """Evaluate a single YOLO model on an image dataset.

    Extended to support:
      - drone_classes: which class IDs in GT labels count as drone
      - patch_verifier: PatchVerifier instance for confuser filtering
      - enable_temporal: simulate N-of-M alert gate on sequential images
    """
    from ultralytics import YOLO
    model = YOLO(weights_path)
    if not model_name:
        model_name = Path(weights_path).stem
    if drone_classes is None:
        drone_classes = getattr(args, '_drone_classes', {0})
    # negatives-only: no drone GT, every detection = FP
    if getattr(args, 'negatives_only', False):
        drone_classes = set()

    # Determine images and labels dirs
    ds_path = Path(dataset_path)
    if (ds_path / "images").exists():
        ds = ImageDataset(ds_path / "images")
    elif ds_path.is_dir():
        ds = ImageDataset(ds_path)
    else:
        print(f"  Cannot find images in {ds_path}")
        return {}

    images = ds.list_images()
    if args.stride > 1:
        images = images[::args.stride]
    if args.limit:
        images = images[:args.limit]
    print(f"  [{model_name}] {len(images):,} images, conf={args.conf}, drone_classes={drone_classes}")

    # Per-source tracking
    per_source = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "tn": 0,
                                       "frames": 0, "det_frames": 0,
                                       "sizes": {"small": 0, "medium": 0, "large": 0}})
    # Aggregate — raw and filtered
    totals = {rule: {"tp": 0, "fp": 0, "fn": 0} for rule in ("iou", "iop")}
    filt_totals = {rule: {"tp": 0, "fp": 0, "fn": 0} for rule in ("iou", "iop")}
    frame_totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    filt_frame_totals = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    all_sizes = {"small": 0, "medium": 0, "large": 0}
    per_size_totals = {rule: {b: {"tp": 0, "fp": 0, "fn": 0} for b in SIZE_BUCKETS}
                       for rule in ("iou", "iop")}
    conf_records = []  # (conf, matched_iou, matched_iop) for PR/conf sweep

    # Temporal states (raw + filtered)
    temporal_raw = temporal_filt = None
    if enable_temporal:
        sys.path.insert(0, str(REPO.parent / "ir_gui")) if str(REPO / "ir_gui") not in sys.path else None
        sys.path.insert(0, str(REPO / "ir_gui"))
        from fusion.temporal import PerModalityTemporalState
        temporal_raw = PerModalityTemporalState(
            stride=1, warning_window=10, warning_require=9,
            alert_window=10, alert_require=9,
        )
        temporal_filt = PerModalityTemporalState(
            stride=1, warning_window=10, warning_require=9,
            alert_window=10, alert_require=9,
        )

    # Counters for patch verifier
    raw_det_count = 0
    filt_det_count = 0
    raw_det_frames = 0
    filt_det_frames = 0
    alert_raw_count = 0
    alert_filt_count = 0
    alert_suppressed_count = 0
    frame_records = []  # per-frame cache

    t0 = time.time()
    for idx, img_path in enumerate(images):
        frame = ds.load_frame(img_path)
        if frame is None:
            continue
        img = frame["img"]
        w, h = frame["w"], frame["h"]
        stem = frame["stem"]

        # Read GT with drone_classes
        lbl_path = ds.labels_dir / f"{stem}.txt"
        gt = read_yolo_labels(lbl_path, w, h, drone_classes=drone_classes)

        # Source = first part of stem before underscore or full stem
        parts = stem.split("_")
        source = parts[0] if len(parts) > 1 else stem

        # Optionally convert to grayscale (3-channel) for IR model testing
        if args.grayscale:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Run inference
        res = model.predict(img, conf=args.conf, verbose=False, imgsz=args.imgsz)
        boxes = res[0].boxes
        dets = []
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            conf_val = float(boxes.conf[i])
            dets.append(((float(xyxy[0]), float(xyxy[1]),
                          float(xyxy[2]), float(xyxy[3])), conf_val))

        raw_det_count += len(dets)
        if dets:
            raw_det_frames += 1

        # Patch verifier filtering
        dets_filt = dets
        if patch_verifier is not None and dets:
            xyxy_list = [d[0] for d in dets]
            probs = patch_verifier.predict_boxes(img, xyxy_list)
            dets_filt = [d for d, p in zip(dets, probs) if p < patch_thr]
        filt_det_count += len(dets_filt)
        if dets_filt:
            filt_det_frames += 1

        # Frame-level (raw)
        has_det = len(dets) > 0
        has_gt = len(gt) > 0
        ftp, ffp, ffn, ftn = compute_frame_metrics(has_det, has_gt)
        frame_totals["tp"] += ftp
        frame_totals["fp"] += ffp
        frame_totals["fn"] += ffn
        frame_totals["tn"] += ftn

        # Frame-level (filtered)
        has_det_f = len(dets_filt) > 0
        ftp_f, ffp_f, ffn_f, ftn_f = compute_frame_metrics(has_det_f, has_gt)
        filt_frame_totals["tp"] += ftp_f
        filt_frame_totals["fp"] += ffp_f
        filt_frame_totals["fn"] += ffn_f
        filt_frame_totals["tn"] += ftn_f

        # Per-source frame-level
        per_source[source]["frames"] += 1
        if has_det:
            per_source[source]["det_frames"] += 1

        # Size distribution
        for d_box, _ in dets:
            sz = classify_size(d_box, w, h)
            all_sizes[sz] += 1
            per_source[source]["sizes"][sz] += 1

        # Detection-level scoring (raw)
        for rule in ("iou", "iop"):
            tp, fp, fn = score_detections(dets, gt, rule=rule,
                                           iou_thr=args.iou_thr, iop_thr=args.iop_thr)
            totals[rule]["tp"] += tp
            totals[rule]["fp"] += fp
            totals[rule]["fn"] += fn
            if rule == "iou":
                per_source[source]["tp"] += tp
                per_source[source]["fp"] += fp
                per_source[source]["fn"] += fn

        # Detection-level scoring (filtered)
        if patch_verifier is not None:
            for rule in ("iou", "iop"):
                tp_f, fp_f, fn_f = score_detections(dets_filt, gt, rule=rule,
                                                     iou_thr=args.iou_thr, iop_thr=args.iop_thr)
                filt_totals[rule]["tp"] += tp_f
                filt_totals[rule]["fp"] += fp_f
                filt_totals[rule]["fn"] += fn_f

        # Per-size TP/FP/FN attribution
        ps = score_per_size(dets, gt, w, h,
                            iou_thr=args.iou_thr, iop_thr=args.iop_thr)
        for rule in ("iou", "iop"):
            for b in SIZE_BUCKETS:
                per_size_totals[rule][b]["tp"] += ps[rule][b]["tp"]
                per_size_totals[rule][b]["fp"] += ps[rule][b]["fp"]
                per_size_totals[rule][b]["fn"] += ps[rule][b]["fn"]

        # Conf records for sweep
        for d_box, d_conf in dets:
            best_iu = best_ip = 0.0
            for g in gt:
                iu, ip = iou_iop(d_box, g)
                best_iu = max(best_iu, iu)
                best_ip = max(best_ip, ip)
            conf_records.append((d_conf, best_iu >= args.iou_thr, best_ip >= args.iop_thr))

        # Temporal alert gate
        if temporal_raw is not None:
            dets_list_raw = [[d[0][0], d[0][1], d[0][2], d[0][3], d[1]] for d in dets]
            dets_list_filt = [[d[0][0], d[0][1], d[0][2], d[0][3], d[1]] for d in dets_filt]
            _, a_raw = temporal_raw.update(dets_list_raw, w, h)
            if a_raw and not getattr(temporal_raw, '_prev_alert', False):
                alert_raw_count += 1
            temporal_raw._prev_alert = a_raw

            _, a_filt_pre = temporal_filt.update(dets_list_raw, w, h)
            # Alert gate: when alert about to fire, check filter
            if a_filt_pre and not getattr(temporal_filt, '_prev_alert', False):
                # Check patch verifier on current dets
                suppressed = False
                if patch_verifier is not None and dets:
                    xyxy_list = [d[0] for d in dets]
                    probs = patch_verifier.predict_boxes(img, xyxy_list)
                    if any(p >= patch_thr for p in probs):
                        suppressed = True
                        alert_suppressed_count += 1
                        temporal_filt.alert_active = False
                        a_filt_pre = False
                if not suppressed:
                    alert_filt_count += 1
            temporal_filt._prev_alert = a_filt_pre

        # Per-frame record for caching
        tp_iop, fp_iop, fn_iop = score_detections(dets, gt, rule="iop",
                                                    iou_thr=args.iou_thr, iop_thr=args.iop_thr)
        tp_f_iop = fp_f_iop = fn_f_iop = 0
        if patch_verifier is not None:
            tp_f_iop, fp_f_iop, fn_f_iop = score_detections(dets_filt, gt, rule="iop",
                                                             iou_thr=args.iou_thr, iop_thr=args.iop_thr)
        det_boxes = [f"{d[0][0]:.1f},{d[0][1]:.1f},{d[0][2]:.1f},{d[0][3]:.1f},{d[1]:.3f}" for d in dets]
        det_sizes = [classify_size(d[0], w, h) for d in dets]
        frame_records.append({
            "stem": stem, "n_gt": len(gt),
            "n_raw": len(dets), "n_filt": len(dets_filt),
            "tp": tp_iop, "fp": fp_iop, "fn": fn_iop,
            "tp_f": tp_f_iop, "fp_f": fp_f_iop, "fn_f": fn_f_iop,
            "n_small": det_sizes.count("small"),
            "n_medium": det_sizes.count("medium"),
            "n_large": det_sizes.count("large"),
            "dets": "|".join(det_boxes),
            "sizes": "|".join(det_sizes),
        })

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t0
            fps = (idx + 1) / elapsed
            print(f"    {idx + 1:>6,}/{len(images):,}  {fps:.1f} fps")

    elapsed = time.time() - t0
    print(f"  [{model_name}] Done in {elapsed:.0f}s")

    # Build results
    result = {"model": model_name, "weights": weights_path}

    # Detection-level metrics (raw)
    det_rows = []
    for rule in ("iou", "iop"):
        row = compute_prf(totals[rule]["tp"], totals[rule]["fp"], totals[rule]["fn"])
        row["config"] = f"{model_name}_{rule}"
        det_rows.append(row)
    result["detection_metrics"] = det_rows

    # Detection-level metrics (filtered)
    if patch_verifier is not None:
        filt_rows = []
        for rule in ("iou", "iop"):
            row = compute_prf(filt_totals[rule]["tp"], filt_totals[rule]["fp"],
                              filt_totals[rule]["fn"])
            row["config"] = f"{model_name}_filtered_{rule}"
            filt_rows.append(row)
        result["filtered_metrics"] = filt_rows

    # Frame-level metrics
    ftot = frame_totals
    fp_rate = ftot["fp"] / max(ftot["fp"] + ftot["tn"], 1)
    fn_rate = ftot["fn"] / max(ftot["fn"] + ftot["tp"], 1)
    result["frame_metrics"] = {
        **ftot,
        "fp_rate": round(fp_rate, 4),
        "fn_rate": round(fn_rate, 4),
        "total_frames": sum(ftot.values()),
    }

    # Filter summary
    if patch_verifier is not None:
        result["filter_summary"] = {
            "raw_det_count": raw_det_count,
            "filt_det_count": filt_det_count,
            "raw_det_frames": raw_det_frames,
            "filt_det_frames": filt_det_frames,
            "det_suppression_rate": round(1 - filt_det_count / max(raw_det_count, 1), 4),
            "frame_suppression_rate": round(1 - filt_det_frames / max(raw_det_frames, 1), 4),
        }

    # Temporal summary
    if temporal_raw is not None:
        result["temporal_summary"] = {
            "alerts_raw": alert_raw_count,
            "alerts_filtered": alert_filt_count,
            "alerts_suppressed": alert_suppressed_count,
        }

    result["size_distribution"] = all_sizes
    result["per_size_metrics"] = per_size_totals
    result["per_source"] = dict(per_source) if args.per_source else {}
    result["conf_records"] = conf_records
    result["frame_records"] = frame_records

    return result


def print_per_size_metrics(per_size: dict, model_name: str):
    """Print TP/FP/FN/P/R/F1 per size bucket per matching rule."""
    for rule in ("iou", "iop"):
        print(f"\n  Per-size metrics ({rule.upper()}, model={model_name}):")
        print(f"  {'size':<8s} {'TP':>6s} {'FP':>6s} {'FN':>6s} "
              f"{'P':>7s} {'R':>7s} {'F1':>7s}")
        for b in ("small", "medium", "large"):
            d = per_size[rule][b]
            m = compute_prf(d["tp"], d["fp"], d["fn"])
            print(f"  {b:<8s} {d['tp']:>6d} {d['fp']:>6d} {d['fn']:>6d} "
                  f"{m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f}")


def run_conf_sweep(conf_records, total_gt_iou, total_gt_iop, thresholds, out_dir, model_name):
    """Sweep confidence thresholds and report metrics."""
    print(f"\n  Confidence threshold sweep:")
    print(f"  {'conf':>6s} {'P_iou':>7s} {'R_iou':>7s} {'F1_iou':>7s} "
          f"{'P_iop':>7s} {'R_iop':>7s} {'F1_iop':>7s}")
    rows = []
    for thr in thresholds:
        filtered = [(c, m_iu, m_ip) for c, m_iu, m_ip in conf_records if c >= thr]
        tp_iu = sum(1 for _, m, _ in filtered if m)
        fp_iu = sum(1 for _, m, _ in filtered if not m)
        tp_ip = sum(1 for _, _, m in filtered if m)
        fp_ip = sum(1 for _, _, m in filtered if not m)
        m_iu = compute_prf(tp_iu, fp_iu, total_gt_iou - tp_iu)
        m_ip = compute_prf(tp_ip, fp_ip, total_gt_iop - tp_ip)
        print(f"  {thr:>6.2f} {m_iu['precision']:>7.4f} {m_iu['recall']:>7.4f} "
              f"{m_iu['f1']:>7.4f} {m_ip['precision']:>7.4f} {m_ip['recall']:>7.4f} "
              f"{m_ip['f1']:>7.4f}")
        rows.append({"conf": thr, **{f"{k}_iou": v for k, v in m_iu.items()},
                      **{f"{k}_iop": v for k, v in m_ip.items()}})
    # Save
    if out_dir:
        csv_path = out_dir / f"{model_name}_conf_sweep.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  Saved: {csv_path}")


def main():
    ap = argparse.ArgumentParser(description="YOLO model benchmarking")
    ap.add_argument("--weights", nargs="+", required=True, help="Model weight files")
    ap.add_argument("--model-name", type=str, default="",
                    help="Override model name (default: weights filename stem)")
    ap.add_argument("--dataset", required=True, help="Dataset path or data.yaml")
    ap.add_argument("--mode", default="per-frame",
                    choices=["per-frame", "yolo-val"],
                    help="Evaluation mode")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou-thr", type=float, default=0.5)
    ap.add_argument("--iop-thr", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--per-source", action="store_true")
    ap.add_argument("--negatives-only", action="store_true",
                    help="All frames are negative (any det = FP)")
    ap.add_argument("--conf-sweep", type=str, default="",
                    help="Comma-separated conf thresholds to sweep")
    ap.add_argument("--grayscale", action="store_true",
                    help="Convert images to grayscale before inference (for IR model on RGB data)")
    ap.add_argument("--patch-weights", type=str, default="",
                    help="Patch verifier weights for confuser filtering")
    ap.add_argument("--patch-thr", type=float, default=0.70,
                    help="Patch verifier confuser threshold")
    ap.add_argument("--temporal", action="store_true",
                    help="Enable temporal alert gate simulation")
    ap.add_argument("--drone-classes", type=str, default="0",
                    help="Comma-separated class IDs that count as drone in GT labels")
    ap.add_argument("--output-dir", type=str, default=str(EVAL_DIR / "results"))
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()
    # Parse drone classes
    args._drone_classes = {int(c) for c in args.drone_classes.split(",")}

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load patch verifier if requested
    pv = None
    if args.patch_weights:
        sys.path.insert(0, str(REPO / "classifier"))
        from patch_verifier import PatchVerifier
        pv = PatchVerifier(args.patch_weights)
        print(f"  Loaded patch verifier: {Path(args.patch_weights).name}")

    # Provenance manifest — every weights file in the run gets hashed
    wp = {f"weights_{i}": w for i, w in enumerate(args.weights)}
    if args.patch_weights:
        wp["patch_weights"] = args.patch_weights
    write_manifest(
        out_dir=out_dir,
        args=args,
        weights_paths=wp,
        extra={"dataset": args.dataset, "stage": "eval_model", "mode": args.mode},
    )

    if args.mode == "yolo-val":
        for w in args.weights:
            run_yolo_val(w, args.dataset, args.imgsz, args.device)
        return

    all_results = []
    for w in args.weights:
        result = evaluate_model(w, args.dataset, args,
                                model_name=args.model_name or "",
                                patch_verifier=pv,
                                patch_thr=args.patch_thr,
                                enable_temporal=args.temporal)
        all_results.append(result)

        # ── Compact summary ──────────────────────────────
        model = result.get("model", "?")
        print(f"\n  ┌─── {model} ───────────────────────────────────────")

        # Raw detection metrics (IoP)
        dm = result.get("detection_metrics", [])
        iop_raw = dm[1] if len(dm) > 1 else (dm[0] if dm else {})
        if iop_raw:
            print(f"  │ RAW    TP={iop_raw.get('TP',0):>5d}  FP={iop_raw.get('FP',0):>5d}  "
                  f"FN={iop_raw.get('FN',0):>5d}  "
                  f"P={iop_raw.get('precision',0):.4f}  R={iop_raw.get('recall',0):.4f}  "
                  f"F1={iop_raw.get('f1',0):.4f}")

        # Filtered detection metrics (IoP)
        fm_list = result.get("filtered_metrics", [])
        iop_filt = fm_list[1] if len(fm_list) > 1 else (fm_list[0] if fm_list else None)
        if iop_filt:
            print(f"  │ FILT   TP={iop_filt.get('TP',0):>5d}  FP={iop_filt.get('FP',0):>5d}  "
                  f"FN={iop_filt.get('FN',0):>5d}  "
                  f"P={iop_filt.get('precision',0):.4f}  R={iop_filt.get('recall',0):.4f}  "
                  f"F1={iop_filt.get('f1',0):.4f}")

        # Frame-level
        fm = result.get("frame_metrics", {})
        if fm:
            print(f"  │ FRAME  TP={fm['tp']}  FP={fm['fp']}  FN={fm['fn']}  TN={fm['tn']}  "
                  f"FPR={fm['fp_rate']:.4f}")

        # Filter impact
        fs = result.get("filter_summary")
        if fs:
            print(f"  │ FILTER det_supp={fs['det_suppression_rate']:.1%}  "
                  f"({fs['raw_det_count']}→{fs['filt_det_count']} dets)  "
                  f"frame_supp={fs['frame_suppression_rate']:.1%}  "
                  f"({fs['raw_det_frames']}→{fs['filt_det_frames']} frames)")

        # Temporal alerts
        ts = result.get("temporal_summary")
        if ts:
            print(f"  │ ALERT  raw={ts['alerts_raw']}  filtered={ts['alerts_filtered']}  "
                  f"suppressed={ts['alerts_suppressed']}")

        # Size profile (IoP)
        psm = result.get("per_size_metrics", {})
        iop_psm = psm.get("iop", {})
        if iop_psm:
            tp_s = iop_psm.get("small",{}).get("tp",0)
            tp_m = iop_psm.get("medium",{}).get("tp",0)
            tp_l = iop_psm.get("large",{}).get("tp",0)
            fp_s = iop_psm.get("small",{}).get("fp",0)
            fp_m = iop_psm.get("medium",{}).get("fp",0)
            fp_l = iop_psm.get("large",{}).get("fp",0)
            print(f"  │ SIZE   TP  S={tp_s:<4d} M={tp_m:<4d} L={tp_l:<4d}  |  "
                  f"FP  S={fp_s:<4d} M={fp_m:<4d} L={fp_l:<4d}")

        print(f"  └{'─'*55}")

        # Size + per-size (only with --per-source for verbose output)
        if args.per_source:
            sd = result.get("size_distribution", {})
            if sd:
                print_size_distribution({result["model"]: sd},
                                         f"Size distribution: {result['model']}")
            psm = result.get("per_size_metrics")
            if psm:
                print_per_size_metrics(psm, result["model"])

        # Per-source
        if args.per_source and result.get("per_source"):
            print(f"\n  Per-source breakdown ({result['model']}):")
            print(f"  {'source':<30s} {'frames':>7s} {'TP':>6s} {'FP':>6s} "
                  f"{'FN':>6s} {'P':>7s} {'R':>7s} {'F1':>7s}")
            for src, data in sorted(result["per_source"].items()):
                m = compute_prf(data["tp"], data["fp"], data["fn"])
                print(f"  {src:<30s} {data['frames']:>7d} {data['tp']:>6d} "
                      f"{data['fp']:>6d} {data['fn']:>6d} "
                      f"{m['precision']:>7.4f} {m['recall']:>7.4f} {m['f1']:>7.4f}")

        # Conf sweep
        if args.conf_sweep and result.get("conf_records"):
            thresholds = [float(x) for x in args.conf_sweep.split(",")]
            # Need total GT counts — approximate from detection metrics
            det_m = result["detection_metrics"]
            gt_iou = det_m[0]["TP"] + det_m[0]["FN"] if det_m else 0
            gt_iop = det_m[1]["TP"] + det_m[1]["FN"] if len(det_m) > 1 else gt_iou
            run_conf_sweep(result["conf_records"], gt_iou, gt_iop,
                           thresholds, out_dir, result["model"])

        # Save results
        save_json({k: v for k, v in result.items()
                   if k not in ("conf_records", "frame_records")},
                  out_dir / f"{result['model']}_results.json")
        if result.get("detection_metrics"):
            save_metrics_csv(result["detection_metrics"],
                             out_dir / f"{result['model']}_metrics.csv")

        # Save per-frame detection cache
        fr = result.get("frame_records", [])
        if fr:
            det_csv = out_dir / f"{result['model']}_frame_detections.csv"
            with open(det_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(fr[0].keys()))
                w.writeheader()
                w.writerows(fr)
            print(f"  Cached: {det_csv}")

    # Cross-model comparison
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("CROSS-MODEL COMPARISON")
        all_rows = []
        for r in all_results:
            all_rows.extend(r.get("detection_metrics", []))
        if all_rows:
            print_metrics_table(all_rows, "All models")
            save_metrics_csv(all_rows, out_dir / "comparison.csv")
        if args.plot:
            plot_metrics_bars(all_rows, out_dir, "Model comparison")

    print(f"\n[eval_model] Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
