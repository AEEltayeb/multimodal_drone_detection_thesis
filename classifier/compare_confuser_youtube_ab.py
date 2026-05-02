"""
compare_confuser_youtube_ab.py — Apples-to-apples A/B of OLD vs NEW
4-class confuser filter on out-of-distribution YouTube videos.

For each modality (rgb, ir):
  1. Load YOLO (from fusion_settings.json) once.
  2. Load NEW pt and OLD pt (v1_backup) verifiers.
  3. Walk YouTube videos at the given stride. Every Nth frame:
       a. Run YOLO once → get boxes.
       b. Score boxes through both verifiers.
       c. Count veto/pass per video per filter.
  4. Confuser videos: vetoed = good (FP killed).
     Drone videos:   vetoed = bad  (TP killed).

This is the leak-free verdict — neither model has seen these videos.

Usage:
  python classifier/compare_confuser_youtube_ab.py
  python classifier/compare_confuser_youtube_ab.py --modality rgb --stride 5
  python classifier/compare_confuser_youtube_ab.py --thresh 0.7
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO       = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

from patch_verifier import PatchVerifier  # noqa: E402

PATCH_DIR = SCRIPT_DIR / "runs" / "patches"
OUT_DIR   = PATCH_DIR

# ── modality config ────────────────────────────────────────────────
MODALITY_VIDEOS = {
    "rgb": Path(r"D:/Downloads/youtube_classifier_videos"),
    "ir":  REPO / "ir_gui" / "demo_outputs",
}

# Pull video catalogues from the existing eval scripts so we stay aligned.
def load_video_labels(modality: str) -> dict:
    if modality == "rgb":
        from eval_youtube_rgb_filter import VIDEO_LABELS
    else:
        from eval_youtube_ir_filter import VIDEO_LABELS
    return dict(VIDEO_LABELS)


def yolo_weights(modality: str) -> str:
    settings = json.loads((REPO / "ir_gui" / "fusion_settings.json").read_text())
    return settings["rgb_model" if modality == "rgb" else "ir_model"]


def confuser_pt_paths(modality: str):
    new = PATCH_DIR / f"confuser_filter4_{modality}.pt"
    old = PATCH_DIR / f"confuser_filter4_{modality}_v1_backup.pt"
    return new, old


# ── per-modality eval ───────────────────────────────────────────────

def eval_modality(modality: str, stride: int, thresh: float, conf: float,
                  max_frames_per_vid: int):
    print(f"\n{'=' * 72}")
    print(f"  {modality.upper()} — YouTube OOD A/B")
    print(f"{'=' * 72}")

    video_dir = MODALITY_VIDEOS[modality]
    if not video_dir.exists():
        print(f"  [skip] {video_dir} not found")
        return None

    labels = load_video_labels(modality)
    available = [(name, cat, video_dir / name)
                 for name, cat in labels.items()
                 if (video_dir / name).exists()]
    if not available:
        print(f"  [skip] no labeled videos in {video_dir}")
        return None
    available.sort(key=lambda x: (x[1], x[0]))

    by_cat = defaultdict(list)
    for n, c, _ in available:
        by_cat[c].append(n)
    print("  Videos:")
    for c in sorted(by_cat):
        print(f"    {c:<11s} {len(by_cat[c])}")

    new_pt, old_pt = confuser_pt_paths(modality)
    if not new_pt.exists() or not old_pt.exists():
        print(f"  [skip] missing pt: {new_pt.name} or {old_pt.name}")
        return None

    print("  Loading YOLO + verifiers...")
    from ultralytics import YOLO
    yolo = YOLO(yolo_weights(modality))
    vf_new = PatchVerifier(str(new_pt), device="cuda:0")
    vf_old = PatchVerifier(str(old_pt), device="cuda:0")

    # Per-video, per-filter counters
    rows = []
    overall = {
        "new": {"frames": 0, "det_frames": 0, "vetoed_frames": 0,
                 "by_cat": defaultdict(lambda: {"frames": 0, "det": 0, "veto": 0})},
        "old": {"frames": 0, "det_frames": 0, "vetoed_frames": 0,
                 "by_cat": defaultdict(lambda: {"frames": 0, "det": 0, "veto": 0})},
    }

    for vname, cat, vpath in available:
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            print(f"  [error] {vname}")
            continue
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        n_proc = 0
        det_frames = 0
        veto_new = 0
        veto_old = 0

        t0 = time.time()
        idx = -1
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % stride != 0:
                continue
            n_proc += 1

            # YOLO once per frame
            res = yolo.predict(frame, conf=conf, iou=0.45, imgsz=640,
                               verbose=False, device=0, max_det=300)[0]
            boxes = []
            if res.boxes is not None and len(res.boxes) > 0:
                xy = res.boxes.xyxy.cpu().numpy()
                for i in range(len(xy)):
                    boxes.append((float(xy[i, 0]), float(xy[i, 1]),
                                  float(xy[i, 2]), float(xy[i, 3])))

            if not boxes:
                # No detection — neither filter has anything to do
                pass
            else:
                det_frames += 1
                p_new = vf_new.predict_boxes(frame, boxes)
                p_old = vf_old.predict_boxes(frame, boxes)
                # veto_frame = ANY box gets confuser-prob >= thresh
                if p_new.size and float(p_new.max()) >= thresh:
                    veto_new += 1
                if p_old.size and float(p_old.max()) >= thresh:
                    veto_old += 1

            if max_frames_per_vid and n_proc >= max_frames_per_vid:
                break

        cap.release()
        dt = time.time() - t0

        rows.append({
            "video": vname,
            "category": cat,
            "frames_total": n_total,
            "frames_processed": n_proc,
            "frames_with_det": det_frames,
            "veto_new": veto_new,
            "veto_old": veto_old,
            "fps_proc": round(n_proc / max(dt, 1e-6), 1),
        })
        for tag, vc in (("new", veto_new), ("old", veto_old)):
            overall[tag]["frames"] += n_proc
            overall[tag]["det_frames"] += det_frames
            overall[tag]["vetoed_frames"] += vc
            d = overall[tag]["by_cat"][cat]
            d["frames"] += n_proc
            d["det"]    += det_frames
            d["veto"]   += vc

        any_rate_new = det_frames / max(n_proc, 1)
        survive_new  = (det_frames - veto_new) / max(n_proc, 1)
        survive_old  = (det_frames - veto_old) / max(n_proc, 1)
        print(f"  {cat:<10s} {vname:<35s} "
              f"det={det_frames}/{n_proc} ({any_rate_new:.1%})  "
              f"survive OLD={survive_old:.1%}  NEW={survive_new:.1%}  "
              f"({dt:.0f}s)")

    # Overall summary
    print(f"\n  {'-' * 70}")
    print("  SUMMARY (lower survive% on confusers = better; "
          "higher survive% on drones = better)")
    print(f"  {'-' * 70}")
    cats = sorted({r["category"] for r in rows})
    print(f"    {'category':<12s} {'frames':>9s} {'with_det':>9s}  "
          f"{'OLD survive':>13s}  {'NEW survive':>13s}")
    for c in cats:
        old_d = overall["old"]["by_cat"][c]
        new_d = overall["new"]["by_cat"][c]
        old_surv = (old_d["det"] - old_d["veto"]) / max(old_d["frames"], 1)
        new_surv = (new_d["det"] - new_d["veto"]) / max(new_d["frames"], 1)
        print(f"    {c:<12s} {old_d['frames']:>9d} {old_d['det']:>9d}  "
              f"{old_surv:>12.2%}   {new_surv:>12.2%}")

    return {"rows": rows, "overall": {
        "old": {**overall["old"], "by_cat": dict(overall["old"]["by_cat"])},
        "new": {**overall["new"], "by_cat": dict(overall["new"]["by_cat"])},
    }}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", choices=["rgb", "ir", "both"], default="both")
    ap.add_argument("--stride", type=int, default=5,
                    help="process every Nth frame (default 5)")
    ap.add_argument("--thresh", type=float, default=0.7,
                    help="patch confuser-prob veto threshold")
    ap.add_argument("--rgb-conf", type=float, default=0.30)
    ap.add_argument("--ir-conf",  type=float, default=0.40)
    ap.add_argument("--max-frames-per-vid", type=int, default=0,
                    help="0 = no cap")
    args = ap.parse_args()

    out = {}
    mods = ["rgb", "ir"] if args.modality == "both" else [args.modality]
    for m in mods:
        conf = args.rgb_conf if m == "rgb" else args.ir_conf
        res = eval_modality(m, args.stride, args.thresh, conf,
                            args.max_frames_per_vid)
        if res is not None:
            out[m] = res

    out_path = OUT_DIR / "confuser_filter4_youtube_ab.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
