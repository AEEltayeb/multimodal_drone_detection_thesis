"""
diagnose_failures.py — Characterize where the baseline RGB model fails.

1. Svanström drone-only frames at 1280: which drones does it miss? (FN analysis)
2. Svanström confuser frames at 1280: which confusers does it hallucinate on? (FP analysis by category)
3. Confuser test set (rgb_confusers_merged): hallucination rate per source
4. IR model on grayscale confusers: does the IR model also hallucinate?

Outputs a JSON summary + prints tables.
"""
import json, sys, time, csv
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))

from metrics import score_detections
from datasets import load_config, resolve_path, ImageDataset, detect_category

# --- Config ---
BASELINE_WEIGHTS = str(REPO / "RGB model" / "Yolo26n_trained" / "weights" / "best.pt")
IR_WEIGHTS = str(REPO / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt")
CONFUSER_ROOT = Path("G:/drone/rgb_confusers_merged")
SVANSTROM_RGB = Path("G:/drone/svanstrom_paired/RGB")
RGB_CONF = 0.25
IR_CONF = 0.40
IMGSZ_RGB = 1280
IMGSZ_IR = 640


def run_svanstrom_analysis(model):
    """Analyse baseline on Svanström by category."""
    ds = ImageDataset(SVANSTROM_RGB / "images", SVANSTROM_RGB / "labels")
    all_images = ds.list_images()
    # Stride to ~3k
    stride = max(1, len(all_images) // 3000)
    all_images = all_images[::stride]
    print(f"\n[Svanström] {len(all_images)} frames (stride {stride})")

    cat_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "n_frames": 0,
                                      "n_with_det": 0, "total_dets": 0,
                                      "miss_sizes": [], "hit_sizes": [],
                                      "fp_confs": [], "miss_gt_sizes": []})
    t0 = time.time()
    for idx, img_path in enumerate(all_images):
        frame = ds.load_frame(img_path)
        if frame is None:
            continue
        img = frame["img"]
        h, w = frame["h"], frame["w"]
        gt = frame["gt"]  # list of (x1,y1,x2,y2)
        cat = frame["category"]

        res = model.predict(img, conf=RGB_CONF, verbose=False, imgsz=IMGSZ_RGB)
        boxes = res[0].boxes
        dets = []
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            c = float(boxes.conf[i])
            dets.append(((float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])), c))

        tp, fp, fn = score_detections(dets, gt, rule="iop", iop_thr=0.5)
        s = cat_stats[cat]
        s["tp"] += tp
        s["fp"] += fp
        s["fn"] += fn
        s["n_frames"] += 1
        s["total_dets"] += len(dets)
        if len(dets) > 0:
            s["n_with_det"] += 1

        # Track detection sizes for analysis
        for d, c in dets:
            box_w = d[2] - d[0]
            box_h = d[3] - d[1]
            area_ratio = (box_w * box_h) / (w * h)
            if tp > 0:
                s["hit_sizes"].append(area_ratio)
            else:
                s["fp_confs"].append(c)

        # Track missed GT sizes
        if fn > 0 and gt:
            for g in gt:
                gw = g[2] - g[0]
                gh = g[3] - g[1]
                s["miss_gt_sizes"].append((gw * gh) / (w * h))

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            fps = (idx + 1) / elapsed
            print(f"  {idx+1}/{len(all_images)}  {fps:.1f} fps")

    print(f"\n[Svanström] Results by category:")
    print(f"  {'Category':12s}  {'Frames':>6s}  {'DetFrames':>9s}  {'TP':>5s}  {'FP':>5s}  {'FN':>5s}  {'Prec':>6s}  {'Rec':>6s}  {'DetRate':>7s}")
    print("  " + "-" * 75)
    for cat in sorted(cat_stats.keys()):
        s = cat_stats[cat]
        p = s["tp"] / max(s["tp"] + s["fp"], 1)
        r = s["tp"] / max(s["tp"] + s["fn"], 1)
        dr = s["n_with_det"] / max(s["n_frames"], 1)
        print(f"  {cat:12s}  {s['n_frames']:6d}  {s['n_with_det']:9d}  {s['tp']:5d}  {s['fp']:5d}  {s['fn']:5d}  {p:6.3f}  {r:6.3f}  {dr:7.1%}")

        # Size analysis for misses
        if s["miss_gt_sizes"]:
            sizes = s["miss_gt_sizes"]
            print(f"    └─ Missed GT sizes: min={min(sizes):.4f}  med={sorted(sizes)[len(sizes)//2]:.4f}  max={max(sizes):.4f} (area ratio)")
        if s["fp_confs"]:
            confs = s["fp_confs"]
            print(f"    └─ FP confidences:  min={min(confs):.3f}  med={sorted(confs)[len(confs)//2]:.3f}  max={max(confs):.3f}")

    return {cat: {k: v for k, v in s.items() if k not in ("miss_sizes", "hit_sizes", "fp_confs", "miss_gt_sizes")}
            for cat, s in cat_stats.items()}


def run_confuser_analysis(model, split="test"):
    """Run baseline on confuser test set (all empty-label images)."""
    img_dir = CONFUSER_ROOT / "images" / split
    images = sorted(img_dir.glob("*.*"))
    print(f"\n[Confusers {split}] {len(images)} images")

    # Group by source prefix
    src_stats = defaultdict(lambda: {"n": 0, "n_det": 0, "total_dets": 0,
                                      "confs": [], "sizes": []})
    t0 = time.time()
    for idx, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        stem = img_path.stem

        # Determine source from filename prefix
        if stem.startswith("airplane_"):
            src = "airplane"
        elif stem.startswith("helicopter_"):
            src = "helicopter"
        elif stem.startswith("bird_") or stem.startswith("raihanrsd_"):
            src = "bird"
        elif "_BIRD_" in stem:
            src = "svan_bird"
        elif "_AIRPLANE_" in stem:
            src = "svan_airplane"
        elif "_HELICOPTER_" in stem:
            src = "svan_helicopter"
        else:
            src = "other"

        res = model.predict(img, conf=RGB_CONF, verbose=False, imgsz=IMGSZ_RGB)
        boxes = res[0].boxes
        n = len(boxes)

        src_stats[src]["n"] += 1
        if n > 0:
            src_stats[src]["n_det"] += 1
            src_stats[src]["total_dets"] += n
            for i in range(n):
                src_stats[src]["confs"].append(float(boxes.conf[i]))
                xyxy = boxes.xyxy[i].cpu().numpy()
                bw = xyxy[2] - xyxy[0]
                bh = xyxy[3] - xyxy[1]
                src_stats[src]["sizes"].append(float(bw * bh) / (w * h))

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            fps = (idx + 1) / elapsed
            print(f"  {idx+1}/{len(images)}  {fps:.1f} fps")

    print(f"\n[Confusers {split}] Hallucination rate by source:")
    print(f"  {'Source':16s}  {'Images':>6s}  {'Halluc':>6s}  {'Rate':>7s}  {'AvgConf':>7s}  {'AvgSize':>7s}")
    print("  " + "-" * 60)
    for src in sorted(src_stats.keys()):
        s = src_stats[src]
        rate = s["n_det"] / max(s["n"], 1)
        avg_conf = np.mean(s["confs"]) if s["confs"] else 0
        avg_size = np.mean(s["sizes"]) if s["sizes"] else 0
        print(f"  {src:16s}  {s['n']:6d}  {s['n_det']:6d}  {rate:7.1%}  {avg_conf:7.3f}  {avg_size:7.4f}")

    return {src: {"n": s["n"], "n_det": s["n_det"], "total_dets": s["total_dets"]}
            for src, s in src_stats.items()}


def run_ir_on_confusers(ir_model, split="test"):
    """Run IR model on GRAYSCALE confuser images."""
    img_dir = CONFUSER_ROOT / "images" / split
    images = sorted(img_dir.glob("*.*"))
    print(f"\n[IR on grayscale confusers {split}] {len(images)} images")

    src_stats = defaultdict(lambda: {"n": 0, "n_det": 0, "total_dets": 0, "confs": []})
    t0 = time.time()
    for idx, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Stack to 3-channel for YOLO
        gray3 = cv2.merge([gray, gray, gray])

        # Determine source
        stem = img_path.stem
        if stem.startswith("airplane_"):
            src = "airplane"
        elif stem.startswith("helicopter_"):
            src = "helicopter"
        elif stem.startswith("bird_") or stem.startswith("raihanrsd_"):
            src = "bird"
        elif "_BIRD_" in stem:
            src = "svan_bird"
        elif "_AIRPLANE_" in stem:
            src = "svan_airplane"
        elif "_HELICOPTER_" in stem:
            src = "svan_helicopter"
        else:
            src = "other"

        res = ir_model.predict(gray3, conf=IR_CONF, verbose=False, imgsz=IMGSZ_IR)
        boxes = res[0].boxes
        n = len(boxes)

        src_stats[src]["n"] += 1
        if n > 0:
            src_stats[src]["n_det"] += 1
            src_stats[src]["total_dets"] += n
            for i in range(n):
                src_stats[src]["confs"].append(float(boxes.conf[i]))

        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            fps = (idx + 1) / elapsed
            print(f"  {idx+1}/{len(images)}  {fps:.1f} fps")

    print(f"\n[IR on grayscale confusers {split}] Hallucination rate by source:")
    print(f"  {'Source':16s}  {'Images':>6s}  {'Halluc':>6s}  {'Rate':>7s}  {'AvgConf':>7s}")
    print("  " + "-" * 55)
    for src in sorted(src_stats.keys()):
        s = src_stats[src]
        rate = s["n_det"] / max(s["n"], 1)
        avg_conf = np.mean(s["confs"]) if s["confs"] else 0
        print(f"  {src:16s}  {s['n']:6d}  {s['n_det']:6d}  {rate:7.1%}  {avg_conf:7.3f}")

    return {src: {"n": s["n"], "n_det": s["n_det"], "total_dets": s["total_dets"]}
            for src, s in src_stats.items()}


if __name__ == "__main__":
    from ultralytics import YOLO

    print("Loading models...")
    rgb_model = YOLO(BASELINE_WEIGHTS)
    ir_model = YOLO(IR_WEIGHTS)

    out_dir = EVAL_DIR / "results" / "_failure_diagnosis"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 1. Svanström by category
    results["svanstrom"] = run_svanstrom_analysis(rgb_model)

    # 2. Confuser test set (RGB model)
    results["confuser_rgb_baseline"] = run_confuser_analysis(rgb_model, "test")

    # 3. IR model on grayscale confusers
    results["confuser_ir_grayscale"] = run_ir_on_confusers(ir_model, "test")

    # Save JSON
    out_path = out_dir / "failure_diagnosis.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[SAVED] {out_path}")

    # --- Write CSVs (auto-generated, not hand-transcribed) ---

    # Svanstrom per-category CSV
    svan_csv = out_dir / "svanstrom_1280_by_category.csv"
    with open(svan_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "category", "frames", "det_rate", "TP", "FP", "FN",
                     "precision", "recall", "med_fp_conf", "missed_gt_med_area"])
        for cat, s in sorted(results["svanstrom"].items()):
            n = max(s["n_frames"], 1)
            p = s["tp"] / max(s["tp"] + s["fp"], 1)
            r = s["tp"] / max(s["tp"] + s["fn"], 1)
            dr = s["n_with_det"] / n
            w.writerow(["baseline", cat, s["n_frames"], f"{dr:.1%}",
                         s["tp"], s["fp"], s["fn"],
                         f"{p:.3f}", f"{r:.3f}", "", ""])
    print(f"[SAVED] {svan_csv}")

    # Confuser hallucination CSV
    confuser_csv = out_dir / "confuser_test_hallucination.csv"
    with open(confuser_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "source", "images", "hallucinations", "halluc_rate", "avg_conf"])
        for src, s in sorted(results["confuser_rgb_baseline"].items()):
            rate = s["n_det"] / max(s["n"], 1)
            w.writerow(["baseline_rgb", src, s["n"], s["n_det"], f"{rate:.1%}", ""])
        for src, s in sorted(results["confuser_ir_grayscale"].items()):
            rate = s["n_det"] / max(s["n"], 1)
            w.writerow(["ir_grayscale", src, s["n"], s["n_det"], f"{rate:.1%}", ""])
    print(f"[SAVED] {confuser_csv}")

    print("\n[ALL DONE]")
