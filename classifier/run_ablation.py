"""
run_ablation.py  —  Full OLD-vs-NEW ablation evaluation.

Produces everything needed for the ablation comparison document:
  1) Trust-scoped metrics CSVs for Anti-UAV + Svanström (from existing per_det.jsonl)
  2) YouTube IR eval with OLD filter and NEW filter at stride=3 (apples-to-apples)
  3) Plots (PR curves, confusion matrices, bar charts)

All outputs go to: classifier/runs/ablation_old_vs_new/
"""

import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO       = SCRIPT_DIR.parent
OUT_ROOT   = SCRIPT_DIR / "runs" / "ablation_old_vs_new"

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(REPO / "ir_gui"))

# ══════════════════════════════════════════════════════════════════════
# PART 1: Trust-scoped metrics from existing per_det.jsonl
# ══════════════════════════════════════════════════════════════════════

def compute_scoped_metrics(label: str, per_det_dir: Path, out_dir: Path):
    """Compute trust-scoped metrics from per_det.jsonl and save CSVs."""
    from generate_all_plots import compute_all_metrics, CONFIG_NAMES

    for ds in ["antiuav", "svanstrom"]:
        p = per_det_dir / ds / "per_det.jsonl"
        if not p.exists():
            print(f"  [SKIP] {p} not found")
            continue

        results = compute_all_metrics(p)
        if not results:
            print(f"  [SKIP] {ds}: no data in per_det.jsonl")
            continue

        ds_out = out_dir / ds
        ds_out.mkdir(parents=True, exist_ok=True)

        for rule in ["iou", "iop"]:
            out_csv = ds_out / f"metrics_scoped_{rule}.csv"
            with open(out_csv, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=[
                    "config", "TP", "FP", "FN", "TN",
                    "Precision", "Recall", "F1"
                ])
                w.writeheader()
                for c in CONFIG_NAMES:
                    r = results[c][rule]
                    w.writerow({
                        "config": c,
                        "TP": r["TP"], "FP": r["FP"],
                        "FN": r["FN"], "TN": r["TN"],
                        "Precision": round(r["precision"], 4),
                        "Recall": round(r["recall"], 4),
                        "F1": round(r["f1"], 4),
                    })

            # Print table
            print(f"\n  {label} / {ds} / {rule.upper()}:")
            hdr = f"  {'config':<28s} {'TP':>8s} {'FP':>8s} {'FN':>8s} {'TN':>8s} {'P':>8s} {'R':>8s} {'F1':>8s}"
            print(hdr)
            print("  " + "-" * (len(hdr) - 2))
            for c in CONFIG_NAMES:
                r = results[c][rule]
                print("  {:<28s} {:>8,} {:>8,} {:>8,} {:>8,} {:>8.4f} {:>8.4f} {:>8.4f}".format(
                    c, r["TP"], r["FP"], r["FN"], r["TN"],
                    r["precision"], r["recall"], r["f1"]))

            print(f"  -> saved {out_csv}")


# ══════════════════════════════════════════════════════════════════════
# PART 2: YouTube IR eval (stride=3, apples-to-apples)
# ══════════════════════════════════════════════════════════════════════

# Same VIDEO_LABELS as in eval_youtube_ir_filter.py
VIDEO_LABELS = {
    "yt_EdOX8tJZDzw.mp4": "HELICOPTER",
    "yt_gg0Da0AtWJk.mp4": "AIRPLANE",
    "yt_LflkvbKEEr8.mp4": "AIRPLANE",
    "yt_UwOMwAGVwvs.mp4": "AIRPLANE",
    "yt_oon2AjhmAE8.mp4": "AIRPLANE",
    "yt_vfLc8n8mcKo.mp4": "AIRPLANE",
    "yt_r5tBDvY7MrA.mp4": "AIRPLANE",
    "yt_5BYnJQfMvrg.mp4": "AIRPLANE",
    "yt_omoX_2UYb0s.mp4": "BIRD",
    "yt_NEANQ74oTew.mp4": "BIRD",
    "yt_zFu7hAi5mIc.mp4": "DRONE",
    "yt_oA8Bfc_bjFk.mp4": "DRONE",
    "yt_Y0epqCI7muk.mp4": "DRONE",
    "yt_nqk0NsTBlFI.mp4": "DRONE",
}

DRONE_QUALITY = {
    "yt_zFu7hAi5mIc.mp4": "CLEAN",
    "yt_oA8Bfc_bjFk.mp4": "LABELS",
    "yt_Y0epqCI7muk.mp4": "LABELS",
    "yt_nqk0NsTBlFI.mp4": "LABELS",
}


def eval_youtube_ir(label: str, patch_verifier_path: str, out_dir: Path,
                    stride: int = 3, ir_conf: float = 0.40, patch_thr: float = 0.70):
    """Run YouTube IR eval with a specific patch verifier."""
    import cv2
    import numpy as np
    from ultralytics import YOLO
    from patch_verifier import PatchVerifier

    video_dir = REPO / "ir_gui" / "demo_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Discover videos
    available = []
    for fname, cat in VIDEO_LABELS.items():
        fpath = video_dir / fname
        if fpath.exists():
            available.append((fname, cat, fpath))
    available.sort(key=lambda x: (x[1], x[0]))

    if not available:
        print(f"  [{label}] No videos found!")
        return

    print(f"\n  [{label}] Found {len(available)} videos, stride={stride}")

    # Load models
    with open(REPO / "ir_gui" / "fusion_settings.json") as f:
        settings = json.load(f)
    ir_model = YOLO(settings["ir_model"])
    ir_verifier = PatchVerifier(patch_verifier_path)

    # Per-video results
    per_video_rows = []

    for fname, cat, fpath in available:
        cap = cv2.VideoCapture(str(fpath))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_idx = 0
        processed = 0
        ir_only_det_frames = 0
        ir_only_dets = 0
        ir_filter_det_frames = 0
        ir_filter_dets = 0
        filter_rejected = 0
        filter_passed = 0
        reject_labels = defaultdict(int)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % stride != 0:
                frame_idx += 1
                continue

            processed += 1

            # IR YOLO inference
            results = ir_model(frame, conf=ir_conf, verbose=False)
            boxes = results[0].boxes
            n_dets = len(boxes)

            if n_dets > 0:
                ir_only_det_frames += 1
                ir_only_dets += n_dets

                # Patch verifier — batch all detections at once
                boxes_xyxy = boxes.xyxy.cpu().numpy()
                probs = ir_verifier.predict_boxes(frame, boxes_xyxy)
                labels = ir_verifier.last_labels  # e.g. "airplane:0.85" or "pass(bird:0.12)"

                # Detection SURVIVES filter when P(confuser) < threshold
                kept = int((probs < patch_thr).sum())

                for p, lbl in zip(probs.tolist(), labels):
                    if p >= patch_thr:
                        filter_rejected += 1
                        reject_labels[lbl] += 1
                    else:
                        filter_passed += 1

                if kept > 0:
                    ir_filter_det_frames += 1
                    ir_filter_dets += kept

            frame_idx += 1

        cap.release()

        # Compute rates
        ir_only_rate = ir_only_det_frames / processed if processed > 0 else 0
        ir_filter_rate = ir_filter_det_frames / processed if processed > 0 else 0
        suppression = 0
        if ir_only_dets > 0:
            suppression = filter_rejected / (filter_rejected + filter_passed)

        # Top reject labels
        top_rejects = sorted(reject_labels.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{lbl}({cnt})" for lbl, cnt in top_rejects) if top_rejects else "-"

        quality = DRONE_QUALITY.get(fname, "")

        per_video_rows.append({
            "video": fname,
            "category": cat,
            "frames": processed,
            "ir_only_det_frames": ir_only_det_frames,
            "ir_only_dets": ir_only_dets,
            "ir_only_det_rate": ir_only_rate,
            "ir_filter_det_frames": ir_filter_det_frames,
            "ir_filter_dets": ir_filter_dets,
            "ir_filter_det_rate": ir_filter_rate,
            "filter_suppression": suppression,
            "filter_rejected": filter_rejected,
            "filter_passed": filter_passed,
            "top_reject_labels": top_str,
            "quality": quality,
        })

        status = "CONFUSER" if cat != "DRONE" else f"DRONE_{quality}"
        print(f"    {fname:<30s} {cat:<12s} {processed:>5d} fr  "
              f"ir_only={ir_only_rate:.1%}  ir_filter={ir_filter_rate:.1%}  "
              f"suppress={suppression:.1%}")

    # Write per-video CSV
    csv_path = out_dir / "youtube_per_video.csv"
    fieldnames = list(per_video_rows[0].keys())
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        w.writeheader()
        w.writerows(per_video_rows)
    print(f"  -> saved {csv_path}")

    # Category summary
    cat_stats = defaultdict(lambda: {"frames": 0, "ir_det": 0, "filt_det": 0})
    for row in per_video_rows:
        cat = row["category"]
        q = row["quality"]
        if cat == "DRONE":
            key = f"DRONE_{q}"
        else:
            key = cat
        cat_stats[key]["frames"] += row["frames"]
        cat_stats[key]["ir_det"] += row["ir_only_det_frames"]
        cat_stats[key]["filt_det"] += row["ir_filter_det_frames"]

    # Also compute ALL_CONFUSERS
    confuser_frames = sum(v["frames"] for k, v in cat_stats.items()
                         if k in ("AIRPLANE", "BIRD", "HELICOPTER"))
    confuser_ir = sum(v["ir_det"] for k, v in cat_stats.items()
                      if k in ("AIRPLANE", "BIRD", "HELICOPTER"))
    confuser_filt = sum(v["filt_det"] for k, v in cat_stats.items()
                        if k in ("AIRPLANE", "BIRD", "HELICOPTER"))

    summary_path = out_dir / "category_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "total_frames", "ir_only_det_frames",
                     "ir_only_det_rate", "ir_filter_det_frames",
                     "ir_filter_det_rate", "suppression"])

        def write_cat(name, frames, ir, filt):
            ir_rate = ir / frames if frames > 0 else 0
            filt_rate = filt / frames if frames > 0 else 0
            supp = 1 - (filt / ir) if ir > 0 else 0
            w.writerow([name, frames, ir, f"{ir_rate:.4f}",
                        filt, f"{filt_rate:.4f}", f"{supp:.4f}"])
            return ir_rate, filt_rate, supp

        ir_r, f_r, s = write_cat("ALL_CONFUSERS", confuser_frames, confuser_ir, confuser_filt)
        print(f"\n  [{label}] ALL_CONFUSERS: {confuser_frames} fr, "
              f"ir={ir_r:.1%}, filt={f_r:.1%}, suppress={s:.1%}")

        for key in ["AIRPLANE", "BIRD", "HELICOPTER", "DRONE_CLEAN", "DRONE_LABELS"]:
            if key in cat_stats:
                st = cat_stats[key]
                ir_r, f_r, s = write_cat(key, st["frames"], st["ir_det"], st["filt_det"])
                print(f"  [{label}] {key}: {st['frames']} fr, "
                      f"ir={ir_r:.1%}, filt={f_r:.1%}, suppress={s:.1%}")

    print(f"  -> saved {summary_path}")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # ── Part 1: Trust-scoped metrics ─────────────────────────────────
    print("=" * 70)
    print("PART 1: Trust-scoped metrics from per_det.jsonl")
    print("=" * 70)

    old_perdet = SCRIPT_DIR / "runs" / "eval_six_configs"
    new_perdet = SCRIPT_DIR / "runs" / "eval_six_configs_v3more_32feat"

    print("\n[OLD pipeline]")
    compute_scoped_metrics("OLD", old_perdet, OUT_ROOT / "old")

    print("\n[NEW pipeline]")
    compute_scoped_metrics("NEW", new_perdet, OUT_ROOT / "new")

    t1 = time.time()
    print(f"\nPart 1 done in {t1 - t0:.0f}s")

    # ── Part 2: YouTube IR ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("PART 2: YouTube IR eval (stride=3, apples-to-apples)")
    print("=" * 70)

    old_filter = str(SCRIPT_DIR / "runs" / "patches" / "confuser_filter4_ir_v1_backup.pt")
    new_filter = str(SCRIPT_DIR / "runs" / "patches" / "confuser_filter4_ir.pt")

    # Check that both filter files exist
    for p, label in [(old_filter, "OLD"), (new_filter, "NEW")]:
        if not Path(p).exists():
            print(f"  [ERROR] {label} filter not found: {p}")
            return

    print("\n[OLD filter]")
    eval_youtube_ir("OLD", old_filter, OUT_ROOT / "old" / "youtube_ir", stride=3)

    t2 = time.time()
    print(f"\nOLD YouTube IR done in {t2 - t1:.0f}s")

    print("\n[NEW filter]")
    eval_youtube_ir("NEW", new_filter, OUT_ROOT / "new" / "youtube_ir", stride=3)

    t3 = time.time()
    print(f"\nNEW YouTube IR done in {t3 - t2:.0f}s")

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"ALL DONE in {t3 - t0:.0f}s ({(t3 - t0) / 60:.1f} min)")
    print(f"Output directory: {OUT_ROOT}")
    print("=" * 70)

    # List all output files
    for p in sorted(OUT_ROOT.rglob("*.csv")):
        rel = p.relative_to(OUT_ROOT)
        print(f"  {rel}")


if __name__ == "__main__":
    main()
