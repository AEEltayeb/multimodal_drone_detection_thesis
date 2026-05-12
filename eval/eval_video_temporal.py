"""eval_video_temporal.py — Video-mode eval that mimics the PySide GUI.

Runs YOLO on a video file frame-by-frame, optionally driving the same
PerModalityTemporalState the GUI uses (ROI re-crop fallback, alert/warning
windows, alert-gate patch verifier veto).

Designed for the Phase 4 grayscale-gap diagnostic: produce a per-frame log
under several conditions (temporal on/off, ROI fallback on/off, imgsz sweep,
grayscale-vs-paired) so we can localize *which* of those is responsible for
the GUI-vs-eval recall gap.

Usage examples:
  # Mimic the GUI exactly on a known-good clip
  python eval/eval_video_temporal.py --video ir_gui/demo_outputs/yt_DiN4s-MWvPg.mp4 \\
    --mode grayscale --temporal on --use-roi-fallback on \\
    --imgsz 640 --conf 0.40 --frame-range 565:end

  # Strip the temporal layer to isolate its contribution
  python eval/eval_video_temporal.py --video ... --mode grayscale \\
    --temporal off --use-roi-fallback off --imgsz 640 --conf 0.40
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(REPO / "ir_gui"))

from datasets import load_config, resolve_path
from run_manifest import write_manifest


# ── _run_with_roi: lifted from ir_gui/pyside_engine.py:43 ────────────────
# Copied (not imported) to avoid pulling Qt/UI deps into the eval script.

def _run_yolo(model, frame, conf, imgsz, nms_iou, device):
    r = model.predict(frame, conf=conf, iou=nms_iou,
                      imgsz=imgsz, verbose=False, device=device)[0]
    dets = []
    if r.boxes is not None:
        for i in range(len(r.boxes)):
            x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()
            c = float(r.boxes.conf[i])
            dets.append([float(x1), float(y1), float(x2), float(y2), c])
    return dets


def _run_with_roi(model, frame, conf, imgsz, nms_iou, device, temporal,
                  use_roi_fallback: bool):
    """Full-frame inference, with optional ROI re-crop fallback when
    full-frame returns nothing and a prior detection ROI is still alive."""
    dets = _run_yolo(model, frame, conf, imgsz, nms_iou, device)
    sources = ["full"] * len(dets)
    troi = []
    if (use_roi_fallback and temporal is not None and not dets
            and getattr(temporal, "last_roi", None) is not None
            and getattr(temporal, "roi_age", 0) > 0):
        h, w = frame.shape[:2]
        roi_result = temporal.get_roi_crop(frame, w, h)
        if roi_result:
            crop, (ox, oy) = roi_result
            troi.append((ox, oy, ox + crop.shape[1], oy + crop.shape[0]))
            crop_dets = _run_yolo(model, crop, conf * 0.8, imgsz, nms_iou, device)
            if crop_dets:
                # remap to global coords
                from fusion.temporal import PerModalityTemporalState
                dets = PerModalityTemporalState.remap_dets(crop_dets, (ox, oy))
                sources = ["troi"] * len(dets)
    return dets, sources, troi


# ── Helpers ──────────────────────────────────────────────────────────────

def _parse_frame_range(spec: str, total_frames: int) -> tuple[int, int]:
    """Parse 'A:B' (1-based, inclusive) → (start_idx_0based, end_idx_exclusive).
    'end' or empty B means to the last frame."""
    if not spec:
        return 0, total_frames
    a, b = spec.split(":")
    a_idx = max(0, int(a) - 1) if a else 0
    if not b or b.lower() == "end":
        b_idx = total_frames
    else:
        b_idx = min(total_frames, int(b))
    return a_idx, b_idx


def _grayscale_3ch(frame_bgr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.merge([g, g, g])


# ── Main eval loop ───────────────────────────────────────────────────────

def evaluate_single_video(args, cfg) -> dict:
    from ultralytics import YOLO

    # Pick the model for the chosen mode
    if args.mode == "rgb_only":
        weights = args.rgb_weights or str(resolve_path(cfg["rgb_weights"]))
        conf_default = args.conf if args.conf is not None else cfg["defaults"].get("rgb_conf", 0.25)
    else:
        # ir_only or grayscale → IR model
        weights = args.ir_weights or str(resolve_path(cfg["ir_weights"]))
        conf_default = args.conf if args.conf is not None else cfg["defaults"].get("ir_conf", 0.40)
    conf = float(conf_default)

    print(f"[video] weights: {Path(weights).name}")
    print(f"[video] mode={args.mode}  temporal={args.temporal}  "
          f"roi_fallback={args.use_roi_fallback}  "
          f"imgsz={args.imgsz}  conf={conf}")

    model = YOLO(weights)

    # Temporal state (only if --temporal on)
    temporal = None
    if args.temporal == "on":
        sys.path.insert(0, str(REPO / "ir_gui"))
        from fusion.temporal import PerModalityTemporalState
        temporal = PerModalityTemporalState(
            stride=1,
            warning_window=args.warning_window,
            warning_require=args.warning_require,
            alert_window=args.alert_window,
            alert_require=args.alert_require,
        )

    # Optional patch verifier for alert-gate cascade
    verifier = None
    if args.cascade == "alert_gate_only":
        sys.path.insert(0, str(REPO / "classifier"))
        from patch_verifier import PatchVerifier
        pv_key = "patch_ir_weights" if args.mode != "rgb_only" else "patch_rgb_weights"
        verifier = PatchVerifier(str(resolve_path(cfg[pv_key])))
        print(f"[video] alert-gate verifier: {Path(cfg[pv_key]).name}")

    # Open video
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fr_start, fr_end = _parse_frame_range(args.frame_range, total_frames)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[video] total frames: {total_frames}  "
          f"range: {fr_start + 1}..{fr_end}  ({fr_end - fr_start} frames eval'd)")

    # Per-frame log
    log_path = Path(args.output_dir) / "per_frame.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(log_path, "w", newline="", encoding="utf-8")
    log_w = csv.writer(log_f)
    log_w.writerow([
        "frame", "raw_full_n_dets", "roi_used", "roi_crop_n_dets",
        "n_dets_final", "max_conf", "warning_active", "alert_active",
        "alert_emitted_this_frame", "alert_revoked_by_verifier",
    ])

    # Aggregate counters
    raw_fired_frames = 0      # any det at all (full or ROI)
    full_fired_frames = 0     # full-frame YOLO fired
    roi_fired_frames = 0      # ROI fallback fired
    alert_events = 0
    warning_events = 0
    revoked_events = 0
    n_processed = 0

    # Seek to start
    if fr_start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr_start)

    prev_alert = False
    prev_warn = False

    t0 = time.time()
    for fi in range(fr_start, fr_end):
        ok, frame = cap.read()
        if not ok:
            break
        if args.mode == "grayscale":
            frame_in = _grayscale_3ch(frame)
        else:
            frame_in = frame

        # Inference
        if args.use_roi_fallback == "on":
            dets, sources, _ = _run_with_roi(
                model, frame_in, conf, args.imgsz, args.nms_iou, args.device,
                temporal, use_roi_fallback=True,
            )
        else:
            dets = _run_yolo(model, frame_in, conf, args.imgsz, args.nms_iou, args.device)
            sources = ["full"] * len(dets)

        full_n = sum(1 for s in sources if s == "full")
        roi_n = sum(1 for s in sources if s == "troi")

        # Drive temporal
        warn = alert = False
        revoked = False
        if temporal is not None:
            h, w = frame.shape[:2]
            warn, alert = temporal.update(dets, w, h)
            # Alert-gate cascade: if alert is about to fire and verifier flags
            # confusers, revoke the alert.
            if args.cascade == "alert_gate_only" and alert and verifier is not None and dets:
                xyxy = [(d[0], d[1], d[2], d[3]) for d in dets]
                probs = verifier.predict_boxes(frame_in, xyxy)
                if probs is not None and len(probs) > 0:
                    if any(p >= args.patch_thr for p in probs):
                        temporal.alert_active = False
                        temporal.confuser_suppressed = True
                        alert = False
                        revoked = True

        if dets:
            raw_fired_frames += 1
            if full_n > 0: full_fired_frames += 1
            if roi_n > 0:  roi_fired_frames += 1
        max_conf = max((d[4] for d in dets), default=0.0)

        alert_emit = bool(alert and not prev_alert)
        warn_emit = bool(warn and not prev_warn)
        if alert_emit: alert_events += 1
        if warn_emit:  warning_events += 1
        if revoked:    revoked_events += 1

        log_w.writerow([
            fi + 1, full_n, int(roi_n > 0), roi_n,
            len(dets), f"{max_conf:.4f}",
            int(warn), int(alert), int(alert_emit), int(revoked),
        ])

        prev_alert, prev_warn = alert, warn
        n_processed += 1

    cap.release()
    log_f.close()
    elapsed = time.time() - t0

    summary = {
        "video": str(args.video),
        "mode": args.mode,
        "temporal": args.temporal,
        "use_roi_fallback": args.use_roi_fallback,
        "cascade": args.cascade,
        "imgsz": args.imgsz,
        "conf": conf,
        "frame_range": [fr_start + 1, fr_end],
        "n_processed": n_processed,
        "raw_fired_frames": raw_fired_frames,
        "full_fired_frames": full_fired_frames,
        "roi_fired_frames": roi_fired_frames,
        "raw_fire_rate": raw_fired_frames / max(n_processed, 1),
        "full_fire_rate": full_fired_frames / max(n_processed, 1),
        "roi_fire_rate": roi_fired_frames / max(n_processed, 1),
        "warning_events": warning_events,
        "alert_events": alert_events,
        "revoked_events": revoked_events,
        "elapsed_s": round(elapsed, 1),
        "fps": round(n_processed / max(elapsed, 1e-6), 1),
    }
    summary_path = Path(args.output_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[video] {n_processed} frames in {elapsed:.1f}s  "
          f"({summary['fps']} fps)")
    print(f"[video] full_fire_rate={summary['full_fire_rate']:.3f}  "
          f"roi_fire_rate={summary['roi_fire_rate']:.3f}  "
          f"raw_fire_rate={summary['raw_fire_rate']:.3f}")
    print(f"[video] alert_events={alert_events}  "
          f"warning_events={warning_events}  "
          f"revoked={revoked_events}")
    print(f"[video] per-frame: {log_path}")
    print(f"[video] summary:   {summary_path}")
    return summary


def main():
    ap = argparse.ArgumentParser(description="Video-mode eval mimicking the GUI")
    # Source
    ap.add_argument("--video", type=str, required=True,
                    help="Path to a video file")
    ap.add_argument("--frame-range", type=str, default="",
                    help="1-based inclusive 'A:B' range; B='end' or empty = last frame")
    # Mode
    ap.add_argument("--mode", choices=["paired", "ir_only", "rgb_only", "grayscale"],
                    default="grayscale",
                    help="paired (RGB+IR videos — TODO), ir_only, rgb_only, "
                         "grayscale (RGB stream → IR model)")
    ap.add_argument("--temporal", choices=["off", "on"], default="off")
    ap.add_argument("--use-roi-fallback", choices=["off", "on"], default="off")
    ap.add_argument("--cascade", choices=["none", "alert_gate_only"], default="none",
                    help="alert_gate_only enables patch verifier veto when alert fires")
    # Inference knobs
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=None,
                    help="If unset, uses cfg.defaults.{rgb_conf,ir_conf}")
    ap.add_argument("--nms-iou", type=float, default=0.45)
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--patch-thr", type=float, default=0.70,
                    help="Patch verifier veto threshold (alert_gate_only cascade)")
    # Temporal knobs
    ap.add_argument("--warning-window", type=int, default=10)
    ap.add_argument("--warning-require", type=int, default=9)
    ap.add_argument("--alert-window", type=int, default=10)
    ap.add_argument("--alert-require", type=int, default=9)
    # Weights overrides
    ap.add_argument("--rgb-weights", type=str, default="")
    ap.add_argument("--ir-weights", type=str, default="")
    # Output
    ap.add_argument("--output-dir", type=str, required=True)
    args = ap.parse_args()

    cfg = load_config()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Provenance
    weights_paths = {
        "rgb_weights": args.rgb_weights or resolve_path(cfg.get("rgb_weights", "")),
        "ir_weights": args.ir_weights or resolve_path(cfg.get("ir_weights", "")),
    }
    if args.cascade == "alert_gate_only":
        weights_paths["patch_rgb_weights"] = resolve_path(cfg.get("patch_rgb_weights", ""))
        weights_paths["patch_ir_weights"] = resolve_path(cfg.get("patch_ir_weights", ""))
    write_manifest(
        out_dir=out,
        args=args,
        cfg=cfg,
        weights_paths=weights_paths,
        extra={"stage": "eval_video_temporal", "video": str(args.video)},
    )

    evaluate_single_video(args, cfg)


if __name__ == "__main__":
    main()
