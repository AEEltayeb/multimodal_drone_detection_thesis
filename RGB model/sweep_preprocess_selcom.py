"""
sweep_preprocess_selcom.py — Overnight unattended preprocessing sweep.

Phases:
  0  Diagnose GT drone pixel sizes in selcom_val
  1  Sweep all preprocessing variants × 2 models on selcom_val
     Score at conf thresholds 0.05 / 0.10 / 0.25 (free, single inference pass)
  2  Regression check top-3 variants on dataset_rgb test split
  3  Write REPORT.md + CSV; print final tables

Run:
    python "RGB model/sweep_preprocess_selcom.py"
    python "RGB model/sweep_preprocess_selcom.py" --phase1-only   # skip Phase 2
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "RGB model"))
from preprocess_cctv import VARIANTS, apply as preprocess   # noqa: E402
from finetune_selcom import load_gt, iop                    # noqa: E402

SELCOM_VAL_IMAGES = Path(r"G:/drone/_finetune_selcom_mixed_ft1/images/val")
SELCOM_VAL_LABELS = Path(r"G:/drone/_finetune_selcom_mixed_ft1/labels/val")

DATASET_RGB_IMAGES = Path(r"G:/drone/dataset/dataset/images/test")
DATASET_RGB_LABELS = Path(r"G:/drone/dataset/dataset/labels/test")

OLD_BASELINE = ROOT / "RGB model" / "Yolo26n_trained" / "weights" / "best_pre_finetune.pt"
MIXED_FT1    = ROOT / "RGB model" / "Yolo26n_selcom_mixed_ft1" / "weights" / "best.pt"

OUT_DIR = ROOT / "runs" / "preprocess_sweep"
SAMPLES_DIR = OUT_DIR / "samples"

IMG_EXTS   = {".jpg", ".jpeg", ".png", ".bmp"}
IOP_THRESH = 0.5
CONF_LEVELS = [0.25, 0.10, 0.05]
IMGSZ      = 640
DATASET_RGB_STRIDE = 5
N_SAMPLE_IMGS = 5


# ── Helpers ───────────────────────────────────────────────────────────────────

def img_list(d: Path, stride: int = 1) -> list[Path]:
    imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)
    return imgs[::stride] if stride > 1 else imgs


def score_frame(preds_all, gt_boxes, conf_thr):
    """
    preds_all: list of (x1n,y1n,x2n,y2n,conf) in normalised coords
    Returns (tp, fp, fn) at the given conf_thr.
    """
    preds = [(p[0], p[1], p[2], p[3]) for p in preds_all if p[4] >= conf_thr]
    matched = set()
    tp = fp = 0
    for pred in preds:
        best, best_j = 0.0, -1
        for j, gt in enumerate(gt_boxes):
            s = iop(pred, gt)
            if s > best:
                best, best_j = s, j
        if best >= IOP_THRESH and best_j not in matched:
            tp += 1
            matched.add(best_j)
        else:
            fp += 1
    fn = len(gt_boxes) - len(matched)
    return tp, fp, fn


def prf(tp, fp, fn):
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f = 2 * p * r / max(p + r, 1e-9)
    return round(p, 4), round(r, 4), round(f, 4)


# ── Phase 0 — Diagnostic ──────────────────────────────────────────────────────

def phase0_diagnose():
    print("\n" + "=" * 72)
    print("PHASE 0 — GT box size diagnostic")
    print("=" * 72)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    imgs = img_list(SELCOM_VAL_IMAGES)
    if not imgs:
        print(f"[fatal] no images at {SELCOM_VAL_IMAGES}")
        sys.exit(1)

    sizes = []
    for img_path in imgs:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        gt = load_gt(SELCOM_VAL_LABELS / (img_path.stem + ".txt"))
        for (x1n, y1n, x2n, y2n) in gt:
            bw = (x2n - x1n) * w
            bh = (y2n - y1n) * h
            sizes.append(np.sqrt(bw * bh))

    if not sizes:
        print("[warn] no GT boxes found — labels may be empty or paths wrong")
        return

    a = np.array(sizes)
    print(f"GT drone sizes (sqrt area in px): "
          f"n={len(a)}  min={a.min():.1f}  p25={np.percentile(a,25):.1f}  "
          f"median={np.median(a):.1f}  p90={np.percentile(a,90):.1f}  max={a.max():.1f}")

    if np.median(a) < 10:
        print("[WARN] Median drone < 10 px — preprocessing is unlikely to recover "
              "sub-threshold features. Continuing sweep anyway.")
    else:
        print("[OK] Drone pixel sizes look workable for preprocessing.")

    # Save histogram CSV
    hist_path = OUT_DIR / "gt_size_histogram.csv"
    with hist_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sqrt_area_px"])
        for s in sizes:
            writer.writerow([round(s, 2)])
    print(f"Histogram saved: {hist_path}")

    # Save 5 raw sample images with GT overlaid
    rng = np.random.default_rng(0)
    sample_idx = rng.choice(len(imgs), size=min(N_SAMPLE_IMGS, len(imgs)), replace=False)
    for i, idx in enumerate(sample_idx):
        img_path = imgs[idx]
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        gt = load_gt(SELCOM_VAL_LABELS / (img_path.stem + ".txt"))
        for (x1n, y1n, x2n, y2n) in gt:
            x1, y1, x2, y2 = int(x1n*w), int(y1n*h), int(x2n*w), int(y2n*h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(str(SAMPLES_DIR / f"raw_{i:02d}_{img_path.stem}.jpg"), frame)
    print(f"Sample images saved to {SAMPLES_DIR}")


# ── Phase 1 — Variant sweep ───────────────────────────────────────────────────

def eval_variant_on_dataset(model, variant_name: str, img_dir: Path,
                             lbl_dir: Path, stride: int = 1
                             ) -> dict[float, dict]:
    """
    Returns {conf_thr: {tp, fp, fn, p, r, f1, mean_ms}}.
    Runs inference at conf=0.05 once per frame; re-thresholds for each level.
    """
    imgs = img_list(img_dir, stride)
    if not imgs:
        return {}

    counters = {c: [0, 0, 0] for c in CONF_LEVELS}  # [tp, fp, fn]
    times = []

    for img_path in imgs:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        t0 = time.perf_counter()
        frame_p = preprocess(frame, variant_name)
        r = model.predict(frame_p, conf=min(CONF_LEVELS), iou=0.45,
                           imgsz=IMGSZ, verbose=False, device=0)[0]
        times.append(time.perf_counter() - t0)

        preds_all = []
        if r.boxes is not None:
            for j in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[j].cpu().numpy()
                preds_all.append((x1/w, y1/h, x2/w, y2/h,
                                   float(r.boxes.conf[j])))

        gt_boxes = load_gt(lbl_dir / (img_path.stem + ".txt"))
        for c in CONF_LEVELS:
            tp, fp, fn = score_frame(preds_all, gt_boxes, c)
            counters[c][0] += tp
            counters[c][1] += fp
            counters[c][2] += fn

    mean_ms = 1000 * np.mean(times) if times else 0.0
    out = {}
    for c in CONF_LEVELS:
        tp, fp, fn = counters[c]
        p, r, f1 = prf(tp, fp, fn)
        out[c] = dict(tp=tp, fp=fp, fn=fn, precision=p, recall=r, f1=f1,
                      n_images=len(imgs), mean_ms=round(mean_ms, 1))
    return out


def save_variant_samples(model, variant_name: str, imgs: list[Path], n: int = 5):
    rng = np.random.default_rng(42)
    idxs = rng.choice(len(imgs), size=min(n, len(imgs)), replace=False)
    for i, idx in enumerate(idxs):
        frame = cv2.imread(str(imgs[idx]))
        if frame is None:
            continue
        h, w = frame.shape[:2]
        frame_p = preprocess(frame, variant_name)
        r = model.predict(frame_p, conf=0.10, iou=0.45, imgsz=IMGSZ,
                           verbose=False, device=0)[0]
        out = frame_p.copy()
        if r.boxes is not None:
            for j in range(len(r.boxes)):
                x1, y1, x2, y2 = [int(v) for v in r.boxes.xyxy[j].cpu().numpy()]
                c = float(r.boxes.conf[j])
                cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(out, f"{c:.2f}", (x1, max(y1-4, 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        fname = f"{variant_name}_{i:02d}.jpg"
        cv2.imwrite(str(SAMPLES_DIR / fname), out)


def phase1_sweep(csv_writer, flush_file):
    from ultralytics import YOLO

    print("\n" + "=" * 72)
    print("PHASE 1 — Preprocessing sweep on selcom_val")
    print("=" * 72)

    models = {
        "old_baseline": str(OLD_BASELINE) if OLD_BASELINE.exists() else None,
        "mixed_ft1":    str(MIXED_FT1)    if MIXED_FT1.exists()    else None,
    }
    for mname, mpath in models.items():
        if mpath is None:
            print(f"[warn] {mname} not found — skipping")

    loaded_models = {}
    for mname, mpath in models.items():
        if mpath:
            loaded_models[mname] = YOLO(mpath)

    imgs = img_list(SELCOM_VAL_IMAGES)
    n_variants = len(VARIANTS)
    n_total = n_variants * len(loaded_models)
    done = 0
    t_start = time.perf_counter()

    for v_idx, (vname, _) in enumerate(VARIANTS):
        for mname, model in loaded_models.items():
            done += 1
            elapsed = time.perf_counter() - t_start
            eta = (elapsed / done) * (n_total - done) if done > 1 else 0
            print(f"[{done}/{n_total}] {mname}  ×  {vname}  "
                  f"(elapsed {elapsed:.0f}s  ETA {eta:.0f}s)", flush=True)
            try:
                results = eval_variant_on_dataset(
                    model, vname, SELCOM_VAL_IMAGES, SELCOM_VAL_LABELS)
                for conf, metrics in results.items():
                    csv_writer.writerow(dict(
                        phase="phase1", dataset="selcom_val",
                        model=mname, variant=vname,
                        conf_thr=conf, **metrics,
                    ))
                flush_file()
                # Save detection samples for this variant (use old_baseline only to save time)
                if mname == "old_baseline":
                    save_variant_samples(model, vname, imgs)
            except Exception:
                print(f"  [ERROR] {mname} × {vname}:")
                traceback.print_exc()

    print(f"\nPhase 1 done in {time.perf_counter()-t_start:.0f}s")


# ── Phase 2 — Top-3 regression check on dataset_rgb ──────────────────────────

def phase2_regression(csv_path: Path, csv_writer, flush_file):
    from ultralytics import YOLO

    print("\n" + "=" * 72)
    print("PHASE 2 — Regression check on dataset_rgb (top-3 variants)")
    print("=" * 72)

    # Read phase1 results to find top-3 variants by mixed_ft1 F1 @ conf=0.25
    rows = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if (row.get("phase") == "phase1"
                    and row.get("dataset") == "selcom_val"
                    and row.get("model") == "mixed_ft1"
                    and float(row.get("conf_thr", 0)) == 0.25):
                rows.append((float(row["f1"]), row["variant"]))

    if not rows:
        print("[warn] No phase1 mixed_ft1 results found — skipping Phase 2")
        return []

    rows.sort(reverse=True)
    top3 = [v for _, v in rows[:3]]
    print(f"Top-3 variants by mixed_ft1 selcom_val F1@0.25: {top3}")

    models = {}
    for mname, mpath in [("old_baseline", OLD_BASELINE), ("mixed_ft1", MIXED_FT1)]:
        if mpath.exists():
            models[mname] = YOLO(str(mpath))
        else:
            print(f"[warn] {mname} not found")

    n_total = len(top3) * len(models)
    done = 0
    t_start = time.perf_counter()

    for vname in top3:
        for mname, model in models.items():
            done += 1
            elapsed = time.perf_counter() - t_start
            eta = (elapsed / done) * (n_total - done) if done > 1 else 0
            print(f"[{done}/{n_total}] {mname}  ×  {vname}  "
                  f"elapsed {elapsed:.0f}s  ETA {eta:.0f}s", flush=True)
            try:
                results = eval_variant_on_dataset(
                    model, vname,
                    DATASET_RGB_IMAGES, DATASET_RGB_LABELS,
                    stride=DATASET_RGB_STRIDE)
                for conf, metrics in results.items():
                    csv_writer.writerow(dict(
                        phase="phase2", dataset="dataset_rgb",
                        model=mname, variant=vname,
                        conf_thr=conf, **metrics,
                    ))
                flush_file()
            except Exception:
                print(f"  [ERROR] {mname} × {vname}:")
                traceback.print_exc()

    print(f"\nPhase 2 done in {time.perf_counter()-t_start:.0f}s")
    return top3


# ── Phase 3 — Report ──────────────────────────────────────────────────────────

def phase3_report(csv_path: Path, top3: list[str]):
    print("\n" + "=" * 72)
    print("PHASE 3 — Final report")
    print("=" * 72)

    rows = []
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    def get(dataset, model, variant, conf):
        for r in rows:
            if (r.get("dataset") == dataset and r.get("model") == model
                    and r.get("variant") == variant
                    and abs(float(r.get("conf_thr", -1)) - conf) < 1e-6):
                return r
        return {}

    report_lines = []

    def section(title):
        line = f"\n## {title}"
        print(line)
        report_lines.append(line)

    def trow(s):
        print(s)
        report_lines.append(s)

    conf = 0.25

    # selcom_val table
    section(f"selcom_val  (conf={conf}  IoP@0.5  sorted by mixed_ft1 F1)")
    header = (f"{'Variant':<30s}  {'old_baseline':>12s}   {'mixed_ft1':>12s}  "
              f"  {'old R':>6s}  {'new R':>6s}")
    trow(header)
    trow("-" * len(header))

    variant_scores = []
    for vname, _ in VARIANTS:
        old = get("selcom_val", "old_baseline", vname, conf)
        new = get("selcom_val", "mixed_ft1", vname, conf)
        old_f1 = float(old.get("f1", 0))
        new_f1 = float(new.get("f1", 0))
        old_r  = float(old.get("recall", 0))
        new_r  = float(new.get("recall", 0))
        variant_scores.append((new_f1, vname, old_f1, old_r, new_r))

    variant_scores.sort(reverse=True)
    for new_f1, vname, old_f1, old_r, new_r in variant_scores:
        trow(f"  {vname:<28s}  {old_f1:>6.4f}  {new_f1:>6.4f}  "
             f"  {old_r:>6.4f}  {new_r:>6.4f}")

    # dataset_rgb table (top-3 only)
    if top3:
        section(f"dataset_rgb  top-3 variants  (conf={conf}  stride={DATASET_RGB_STRIDE})")
        trow(f"{'Variant':<30s}  {'old_baseline F1':>16s}  {'mixed_ft1 F1':>14s}")
        trow("-" * 65)
        for vname in top3:
            old = get("dataset_rgb", "old_baseline", vname, conf)
            new = get("dataset_rgb", "mixed_ft1", vname, conf)
            trow(f"  {vname:<28s}  {float(old.get('f1',0)):>16.4f}  "
                 f"{float(new.get('f1',0)):>14.4f}")

    # Verdict per top-3
    section("Verdict")
    baseline_rgb = get("dataset_rgb", "old_baseline", "none", conf)
    baseline_rgb_f1 = float(baseline_rgb.get("f1", 0)) if baseline_rgb else 0.0

    for new_f1, vname, old_f1, old_r, new_r in variant_scores[:3]:
        rgb = get("dataset_rgb", "mixed_ft1", vname, conf)
        rgb_f1 = float(rgb.get("f1", 0)) if rgb else 0.0
        regression_ok = (baseline_rgb_f1 == 0) or (rgb_f1 >= baseline_rgb_f1 - 0.02)
        verdict = "PASS" if (old_r >= 0.3 or new_r >= 0.5) and regression_ok else "INVESTIGATE"
        trow(f"  {vname}: old_baseline R={old_r:.4f}  mixed_ft1 R={new_r:.4f}  "
             f"dataset_rgb F1={rgb_f1:.4f}  -> {verdict}")

    # Write REPORT.md
    report_path = OUT_DIR / "REPORT.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nReport written: {report_path}")
    print(f"Full CSV:       {csv_path}")
    print(f"Samples:        {SAMPLES_DIR}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-only", action="store_true",
                    help="Skip Phase 2 (dataset_rgb regression)")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUT_DIR / "selcom_val.csv"
    csv_file = csv_path.open("w", newline="")
    fieldnames = ["phase", "dataset", "model", "variant", "conf_thr",
                  "n_images", "tp", "fp", "fn",
                  "precision", "recall", "f1", "mean_ms"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    def flush():
        csv_file.flush()

    t_wall = time.perf_counter()
    try:
        phase0_diagnose()
        phase1_sweep(writer, flush)
        top3 = [] if args.phase1_only else phase2_regression(csv_path, writer, flush)
        csv_file.close()
        phase3_report(csv_path, top3)
    except KeyboardInterrupt:
        csv_file.close()
        print("\n[interrupted] Partial results saved to", csv_path)
        sys.exit(1)

    print(f"\nTotal wall time: {(time.perf_counter()-t_wall)/60:.1f} min")
    print("=" * 72 + "\nALL DONE.\n" + "=" * 72)


if __name__ == "__main__":
    main()
