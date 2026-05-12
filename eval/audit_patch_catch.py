"""Patch-verifier catch-rate audit on Svanstrom @ imgsz=1280.

For the production candidate RGB model, runs YOLO on every Svanstrom frame,
splits per-detection results into:
  - DRONE-TP   (det overlaps GT, frame category == DRONE)
  - DRONE-FP   (det in DRONE frame but no IoP match — model thinks there's a drone where there isn't)
  - BIRD       (frame category == BIRD; all dets are FPs by definition)
  - AIRPLANE   (same)
  - HELICOPTER (same)
then runs each detection's crop through the RGB patch verifier and reports
per-class catch rate at multiple thresholds plus drone-TP veto rate.

Output: eval/results/_patch_catch_audit/<model>_<verifier>/
  - per_detection.csv : every detection's (frame, class, conf, prob, label, matched_iop)
  - summary.json      : per-class catch rate at thr ∈ {0.3, 0.4, 0.5, 0.6, 0.7}
  - manifest.json
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))
sys.path.insert(0, str(REPO))

from datasets import ImageDataset, detect_category
from metrics import iou_iop
from run_manifest import write_manifest

DEFAULTS = {
    "baseline":      REPO / "RGB model" / "Yolo26n_trained"        / "weights" / "best.pt",
    "hardneg_v3more": REPO / "RGB model" / "Yolo26n_hardneg_v3_more" / "weights" / "best.pt",
    "retrained_v2":  REPO / "RGB model" / "Yolo26n_retrained_v2"   / "weights" / "best.pt",
}
PATCH_DEFAULTS = {
    "v1": REPO / "classifier" / "runs" / "patches" / "confuser_filter4_rgb_v1_backup.pt",
    "v2": REPO / "classifier" / "runs" / "patches" / "confuser_filter4_rgb_v2_backup.pt",
    "v3": REPO / "classifier" / "runs" / "patches" / "confuser_filter4_rgb_v3_backup.pt",
    "v4": REPO / "classifier" / "runs" / "patches" / "confuser_filter4_rgb.pt",
}

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]
IOP_THR = 0.50
SVANSTROM_RGB = Path("G:/drone/svanstrom_paired/RGB")


def categorize_detection(frame_category: str, det_box, gt_boxes, used_gt: set[int]) -> tuple[str, int, float]:
    """Return (bucket, matched_iop_int, best_iop_score).

    bucket ∈ {"DRONE_TP", "DRONE_FP", "BIRD", "AIRPLANE", "HELICOPTER", "OTHER"}.
    """
    if frame_category != "DRONE":
        return frame_category, 0, 0.0
    best_iop, best_idx = 0.0, -1
    for gi, g in enumerate(gt_boxes):
        _, ip = iou_iop(det_box, g)
        if ip > best_iop:
            best_iop, best_idx = ip, gi
    if best_iop >= IOP_THR and best_idx not in used_gt:
        used_gt.add(best_idx)
        return "DRONE_TP", 1, best_iop
    return "DRONE_FP", 0, best_iop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb-model", default="baseline", choices=list(DEFAULTS.keys()))
    ap.add_argument("--rgb-weights", default=None, help="override weights path")
    ap.add_argument("--patch-version", default="v2", choices=list(PATCH_DEFAULTS.keys()))
    ap.add_argument("--patch-weights", default=None, help="override patch verifier path")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--stride", type=int, default=9, help="image stride (9 matches diagnose_failures_all.py)")
    ap.add_argument("--limit", type=int, default=0, help="cap frames for smoke runs (0 = no cap)")
    ap.add_argument("--svan-rgb", default=str(SVANSTROM_RGB))
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    rgb_weights = Path(args.rgb_weights) if args.rgb_weights else DEFAULTS[args.rgb_model]
    patch_weights = Path(args.patch_weights) if args.patch_weights else PATCH_DEFAULTS[args.patch_version]
    out_dir = Path(args.output_dir) if args.output_dir else (
        REPO / "eval" / "results" / "_patch_catch_audit" / f"{args.rgb_model}_{args.patch_version}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[audit] RGB:  {rgb_weights}")
    print(f"[audit] patch: {patch_weights}")
    print(f"[audit] out:   {out_dir}")

    write_manifest(
        out_dir,
        args=vars(args),
        cfg={"svan_rgb": str(SVANSTROM_RGB)},
        weights_paths={"rgb": str(rgb_weights), "patch_rgb": str(patch_weights)},
        cache_paths=[],
        extra={"thresholds": THRESHOLDS, "iop_thr": IOP_THR},
    )

    from ultralytics import YOLO
    sys.path.insert(0, str(REPO / "classifier"))
    from patch_verifier import PatchVerifier

    model = YOLO(str(rgb_weights))
    verifier = PatchVerifier(str(patch_weights))
    print(f"[audit] verifier classes: {verifier.class_names} (confuser idx: {verifier.confuser_indices})")

    ds = ImageDataset(Path(args.svan_rgb) / "images", Path(args.svan_rgb) / "labels")
    imgs = ds.list_images()[::args.stride]
    if args.limit:
        imgs = imgs[: args.limit]
    print(f"[audit] {len(imgs)} frames @ imgsz={args.imgsz} stride={args.stride} conf={args.conf}")

    per_det: list[dict] = []
    by_bucket = defaultdict(lambda: {"n": 0, "probs": [], "labels": []})

    t0 = time.time()
    for idx, p in enumerate(imgs):
        f = ds.load_frame(p)
        if f is None:
            continue
        res = model.predict(f["img"], conf=args.conf, verbose=False, imgsz=args.imgsz)
        boxes = res[0].boxes
        if len(boxes) == 0:
            if (idx + 1) % 500 == 0:
                print(f"  {idx + 1}/{len(imgs)}  {(idx + 1) / (time.time() - t0):.1f} fps")
            continue

        det_boxes = [
            (float(boxes.xyxy[i][0]), float(boxes.xyxy[i][1]),
             float(boxes.xyxy[i][2]), float(boxes.xyxy[i][3]))
            for i in range(len(boxes))
        ]
        det_confs = [float(boxes.conf[i]) for i in range(len(boxes))]
        probs = verifier.predict_boxes(f["img"], det_boxes)
        labels = list(verifier.last_labels) if len(verifier.last_labels) == len(det_boxes) else ["?"] * len(det_boxes)

        used_gt: set[int] = set()
        for i, (db, dc) in enumerate(zip(det_boxes, det_confs)):
            bucket, matched, best_iop = categorize_detection(f["category"], db, f["gt"], used_gt)
            row = {
                "frame": f["stem"],
                "category": f["category"],
                "bucket": bucket,
                "det_conf": round(dc, 4),
                "best_iop": round(best_iop, 4),
                "matched_iop": matched,
                "patch_prob": round(float(probs[i]), 4),
                "patch_label": labels[i],
            }
            per_det.append(row)
            by_bucket[bucket]["n"] += 1
            by_bucket[bucket]["probs"].append(float(probs[i]))
            by_bucket[bucket]["labels"].append(labels[i])

        if (idx + 1) % 500 == 0:
            print(f"  {idx + 1}/{len(imgs)}  {(idx + 1) / (time.time() - t0):.1f} fps")

    # Aggregate
    summary = {}
    for bucket, s in by_bucket.items():
        probs = np.array(s["probs"], dtype=np.float32)
        per_thr = {}
        for t in THRESHOLDS:
            vetoed = int((probs >= t).sum())
            per_thr[f"{t:.2f}"] = {
                "vetoed": vetoed,
                "rate": round(vetoed / max(len(probs), 1), 4),
            }
        summary[bucket] = {
            "n_detections": s["n"],
            "median_patch_prob": round(float(np.median(probs)), 4) if len(probs) else 0.0,
            "veto_at_threshold": per_thr,
        }

    # Per-det CSV
    csv_path = out_dir / "per_detection.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("frame,category,bucket,det_conf,best_iop,matched_iop,patch_prob,patch_label\n")
        for r in per_det:
            fh.write(
                f"{r['frame']},{r['category']},{r['bucket']},{r['det_conf']},"
                f"{r['best_iop']},{r['matched_iop']},{r['patch_prob']},{r['patch_label']}\n"
            )

    # Summary JSON
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump({
            "rgb_model": args.rgb_model,
            "rgb_weights": str(rgb_weights),
            "patch_version": args.patch_version,
            "patch_weights": str(patch_weights),
            "imgsz": args.imgsz,
            "conf": args.conf,
            "stride": args.stride,
            "iop_thr": IOP_THR,
            "thresholds": THRESHOLDS,
            "n_frames": len(imgs),
            "n_detections_total": sum(s["n_detections"] for s in summary.values()),
            "by_bucket": summary,
        }, fh, indent=2)

    # Print table
    print(f"\n[audit] Done in {time.time() - t0:.1f}s. Out: {out_dir}")
    print(f"\n{'Bucket':12s} {'N':>5s} {'MedP':>6s}  " + "  ".join(f"v@{t:.1f}" for t in THRESHOLDS))
    print("-" * 70)
    for bucket in ["DRONE_TP", "DRONE_FP", "BIRD", "AIRPLANE", "HELICOPTER", "OTHER"]:
        if bucket not in summary:
            continue
        s = summary[bucket]
        rates = "  ".join(f"{s['veto_at_threshold'][f'{t:.2f}']['rate']:.3f}" for t in THRESHOLDS)
        print(f"{bucket:12s} {s['n_detections']:>5d} {s['median_patch_prob']:>6.3f}  {rates}")
    print()
    print(f"  → catch rate = veto rate for confuser buckets (BIRD/AIRPLANE/HELICOPTER); higher is better")
    print(f"  → drone-TP veto rate (DRONE_TP row); lower is better")


if __name__ == "__main__":
    main()
