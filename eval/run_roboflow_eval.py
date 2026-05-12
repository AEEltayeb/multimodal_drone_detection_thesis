"""
run_roboflow_eval.py — Orchestrator for Roboflow OOD dataset evaluation.

Extracts 9 Roboflow datasets (airplane/bird/drone/helicopter in RGB and IR),
then calls eval_model.py for each dataset with the correct YOLO model,
patch verifier (v2 backup) confuser filters, and temporal alert gate.

Compares baseline RGB model (Yolo26n_trained) vs retrained_v2 on RGB datasets.

Usage:
    python eval/run_roboflow_eval.py --quick          # ~500 images/dataset
    python eval/run_roboflow_eval.py --full            # all images
    python eval/run_roboflow_eval.py --extract-only    # just unzip
    python eval/run_roboflow_eval.py --skip-extract    # skip unzip, run eval
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
GDRIVE = Path("G:/drone")
EXTRACT_DIR = GDRIVE / "roboflow_eval"

# ── Dataset registry ────────────────────────────────────────────────

DATASETS = [
    {
        "name": "rgb_airplane",
        "zip": "Airplane.v1i.yolo26-datasets-84fcz-airplane-vgvsf roboflow.zip",
        "alt_dir": "Airplane.v1i.yolo26-datasets-84fcz-airplane-vgvsf roboflow",
        "modality": "rgb",
        "category": "AIRPLANE",
        "drone_classes": "",   # empty = negatives-only (no drone GT)
        "has_drones": False,
    },
    {
        "name": "rgb_bird",
        "zip": "bird.v1i.yolo26-birds-zekpr-bird-pn3pj.zip",
        "alt_dir": "bird.v1i.yolo26-birds-zekpr-bird-pn3pj",
        "modality": "rgb",
        "category": "BIRD",
        "drone_classes": "",
        "has_drones": False,
    },
    {
        "name": "rgb_drone",
        "zip": "Drone.v1i.yolo26-drone-blb9h-drone-evttd.zip",
        "alt_dir": "Drone.v1i.yolo26-drone-blb9h-drone-evttd",
        "modality": "rgb",
        "category": "DRONE",
        "drone_classes": "0",
        "has_drones": True,
    },
    {
        "name": "rgb_helicopter",
        "zip": "final helicopter.v1i.yolo26-new-workspace-0k81p-final-helicopter.zip",
        "alt_dir": "final helicopter.v1i.yolo26-new-workspace-0k81p-final-helicopter",
        "modality": "rgb",
        "category": "HELICOPTER",
        "drone_classes": "",
        "has_drones": False,
    },
    {
        "name": "ir_airplane_hors2",
        "zip": "infrared_airplane_hors2.v2i.yolo26-shooting-dypek-hors2.zip",
        "alt_dir": "infrared_airplane_hors2.v2i.yolo26-shooting-dypek-hors2",
        "modality": "ir",
        "category": "AIRPLANE",
        "drone_classes": "",
        "has_drones": False,
    },
    {
        "name": "ir_airplane_plane",
        "zip": "infrared_airplane_plane_040302.v1i.yolo26-1874899822-qq-com-plane_040302.zip",
        "alt_dir": "infrared_airplane_plane_040302.v1i.yolo26-1874899822-qq-com-plane_040302",
        "modality": "ir",
        "category": "AIRPLANE",
        "drone_classes": "",
        "has_drones": False,
    },
    {
        "name": "ir_bird",
        "zip": "infrared_bird_Bird.v1i.yolo26-tfnet-night-vision-bird-zrvw0.zip",
        "alt_dir": "infrared_bird_Bird.v1i.yolo26-tfnet-night-vision-bird-zrvw0",
        "modality": "ir",
        "category": "BIRD",
        "drone_classes": "",
        "has_drones": False,
    },
    {
        "name": "ir_mixed_cbam",
        "zip": "Infrared_bird_drone_airplane_CBAM_TF-Net.v1i.yolo26-maha-daxhh-cbam_tf-net.zip",
        "alt_dir": "Infrared_bird_drone_airplane_CBAM_TF-Net.v1i.yolo26-maha-daxhh-cbam_tf-net",
        "modality": "ir",
        "category": "MIXED",
        "drone_classes": "1",   # class D=1 is drone
        "has_drones": True,
    },
    {
        "name": "ir_drone_night",
        "zip": "infrared_drone_night.v6i.yolo26-siddhant-mc35z-drone_night-v3a9a.zip",
        "alt_dir": "infrared_drone_night.v6i.yolo26-siddhant-mc35z-drone_night-v3a9a",
        "modality": "ir",
        "category": "DRONE",
        "drone_classes": "0,1",  # both classes are drone
        "has_drones": True,
    },
]

# ── Model & filter paths (relative to REPO) ────────────────────────

MODELS = {
    "rgb_baseline":     REPO / "RGB model" / "Yolo26n_trained" / "weights" / "best.pt",
    "rgb_retrained_v2": REPO / "RGB model" / "Yolo26n_retrained_v2" / "weights" / "best.pt",
    "ir_model":         REPO / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt",
}

FILTERS = {
    "rgb": REPO / "classifier" / "runs" / "patches" / "confuser_filter4_rgb_v2_backup.pt",
    "ir":  REPO / "classifier" / "runs" / "patches" / "confuser_filter4_ir_v2_backup.pt",
}


def _green(s): return f"\033[32m{s}\033[0m"
def _yellow(s): return f"\033[33m{s}\033[0m"
def _red(s): return f"\033[31m{s}\033[0m"


# ── Extraction ──────────────────────────────────────────────────────

def extract_datasets():
    """Extract all dataset zips to EXTRACT_DIR/<name>/."""
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        dst = EXTRACT_DIR / ds["name"]
        src = GDRIVE / ds["zip"]
        if dst.exists():
            n = sum(1 for _ in dst.rglob("*.jpg")) + sum(1 for _ in dst.rglob("*.png"))
            print(_yellow(f"  SKIP (exists, {n} images): {ds['name']}"))
            continue
        if not src.exists():
            print(_red(f"  MISSING: {src}"))
            continue
        print(f"  Extracting: {ds['name']} ...")
        import zipfile
        with zipfile.ZipFile(str(src), 'r') as zf:
            zf.extractall(str(dst))
        print(f"    Done: {dst}")


# ── Dataset discovery ───────────────────────────────────────────────

def find_all_splits(ds_root: Path) -> list[tuple[str, Path]]:
    """Find all split directories with images. Returns [(split_name, split_path), ...]."""
    splits = []
    for split in ("train", "valid", "val", "test"):
        img_dir = ds_root / split / "images"
        if img_dir.exists():
            n = len(list(img_dir.iterdir()))
            if n >= 5:
                splits.append((split, ds_root / split))
    # Maybe images are at root level
    if not splits and (ds_root / "images").exists():
        splits.append(("root", ds_root))
    return splits


def count_images(split_dir: Path) -> int:
    img_dir = split_dir / "images" if (split_dir / "images").exists() else split_dir
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sum(1 for f in img_dir.iterdir() if f.suffix.lower() in exts)


def compute_stride(n_images: int, target: int = 500) -> int:
    """Compute stride to get ~target images from n_images."""
    if n_images <= target:
        return 1
    return max(1, n_images // target)


# ── Run eval_model.py ───────────────────────────────────────────────

def run_eval(weights: str, dataset_path: str, output_dir: str,
             conf: float, patch_weights: str, patch_thr: float,
             drone_classes: str, stride: int, temporal: bool,
             model_name: str = "") -> bool:
    """Call eval_model.py as a subprocess."""
    cmd = [
        sys.executable, str(EVAL_DIR / "eval_model.py"),
        "--weights", str(weights),
        "--dataset", str(dataset_path),
        "--output-dir", str(output_dir),
        "--conf", str(conf),
        "--stride", str(stride),
        "--imgsz", "640",
        "--model-name", model_name,
    ]
    if patch_weights:
        cmd += ["--patch-weights", str(patch_weights)]
        cmd += ["--patch-thr", str(patch_thr)]
    if drone_classes:
        cmd += ["--drone-classes", drone_classes]
    else:
        cmd += ["--negatives-only"]
    if temporal:
        cmd += ["--temporal"]

    print(_yellow(f"  CMD: {' '.join(cmd[-8:])}"))
    proc = subprocess.run(cmd, cwd=str(REPO))
    return proc.returncode == 0


# ── Aggregation ─────────────────────────────────────────────────────

def aggregate_results(results_dir: Path):
    """Scrape all results JSONs and build summary CSV."""
    summary_rows = []
    for json_path in sorted(results_dir.rglob("*_results.json")):
        try:
            data = json.loads(json_path.read_text())
        except Exception:
            continue
        # Derive dataset/split from path: results_dir/dataset/model/split/file.json
        rel = json_path.relative_to(results_dir)
        parts = rel.parts
        ds_name = parts[0] if len(parts) > 1 else "unknown"
        split_name = parts[2] if len(parts) > 3 else ""
        ds_label = f"{ds_name}/{split_name}" if split_name else ds_name
        model = data.get("model", "")
        dm = data.get("detection_metrics", [])
        fm = data.get("filtered_metrics", [])
        fs = data.get("filter_summary", {})
        ts = data.get("temporal_summary", {})

        # Use IoP metrics (index 1) for primary reporting
        raw_m = dm[1] if len(dm) > 1 else (dm[0] if dm else {})
        filt_m = fm[1] if len(fm) > 1 else (fm[0] if fm else {})

        # Per-size breakdown (IoP)
        psm = data.get("per_size_metrics", {}).get("iop", {})

        row = {
            "dataset": ds_label,
            "model": model,
            "raw_TP": raw_m.get("TP", 0),
            "raw_FP": raw_m.get("FP", 0),
            "raw_FN": raw_m.get("FN", 0),
            "raw_precision": raw_m.get("precision", 0),
            "raw_recall": raw_m.get("recall", 0),
            "raw_f1": raw_m.get("f1", 0),
            "filt_TP": filt_m.get("TP", ""),
            "filt_FP": filt_m.get("FP", ""),
            "filt_FN": filt_m.get("FN", ""),
            "filt_precision": filt_m.get("precision", ""),
            "filt_recall": filt_m.get("recall", ""),
            "filt_f1": filt_m.get("f1", ""),
            "det_suppression": fs.get("det_suppression_rate", ""),
            "frame_suppression": fs.get("frame_suppression_rate", ""),
            "alerts_raw": ts.get("alerts_raw", ""),
            "alerts_filtered": ts.get("alerts_filtered", ""),
            "alerts_suppressed": ts.get("alerts_suppressed", ""),
            "tp_S": psm.get("small", {}).get("tp", 0),
            "tp_M": psm.get("medium", {}).get("tp", 0),
            "tp_L": psm.get("large", {}).get("tp", 0),
            "fp_S": psm.get("small", {}).get("fp", 0),
            "fp_M": psm.get("medium", {}).get("fp", 0),
            "fp_L": psm.get("large", {}).get("fp", 0),
        }
        summary_rows.append(row)

    if not summary_rows:
        print(_yellow("  No results to aggregate"))
        return

    csv_path = results_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)
    print(_green(f"\n  Summary CSV: {csv_path}"))

    # Print summary table
    print(f"\n{'='*130}")
    print(f"  {'dataset':<28s} {'model':<20s} {'raw_FP':>6s} {'filt_FP':>7s} "
          f"{'supp%':>6s} {'raw_R':>6s} {'filt_R':>6s} "
          f"{'alert':>5s} {'a_sp':>4s}  "
          f"{'tp_S':>4s} {'tp_M':>4s} {'tp_L':>4s} "
          f"{'fp_S':>4s} {'fp_M':>4s} {'fp_L':>4s}")
    print(f"  {'-'*134}")
    for r in summary_rows:
        supp = f"{r['det_suppression']:.0%}" if r['det_suppression'] != "" else "N/A"
        raw_r = f"{r['raw_recall']:.3f}" if r['raw_recall'] else "N/A"
        filt_r = f"{r['filt_recall']:.3f}" if r['filt_recall'] != "" else "N/A"
        alerts = str(r['alerts_raw']) if r['alerts_raw'] != "" else "-"
        a_supp = str(r['alerts_suppressed']) if r['alerts_suppressed'] != "" else "-"
        print(f"  {r['dataset']:<28s} {r['model']:<20s} "
              f"{r['raw_FP']:>6} {str(r['filt_FP']):>7s} "
              f"{supp:>6s} {raw_r:>6s} {filt_r:>6s} "
              f"{alerts:>5s} {a_supp:>4s}  "
              f"{r['tp_S']:>4} {r['tp_M']:>4} {r['tp_L']:>4} "
              f"{r['fp_S']:>4} {r['fp_M']:>4} {r['fp_L']:>4}")


# ── Main ────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Roboflow OOD dataset evaluation")
    ap.add_argument("--quick", action="store_true",
                    help="Use stride to sample ~500 images per dataset")
    ap.add_argument("--full", action="store_true",
                    help="Use all images (stride=1)")
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Subset of dataset names to run")
    ap.add_argument("--patch-thr", type=float, default=0.70)
    ap.add_argument("--output-dir", type=str,
                    default=str(EVAL_DIR / "results" / "roboflow_ood"))
    args = ap.parse_args()

    results_dir = Path(args.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        aggregate_results(results_dir)
        return

    # Extract
    if not args.skip_extract:
        print(_green("\n[1/3] Extracting datasets..."))
        extract_datasets()

    if args.extract_only:
        return

    # Verify models exist
    for label, path in MODELS.items():
        if not path.exists():
            print(_red(f"  MISSING model: {label} → {path}"))
            return
    for label, path in FILTERS.items():
        if not path.exists():
            print(_red(f"  MISSING filter: {label} → {path}"))
            return

    # Run evaluations
    print(_green("\n[2/3] Running evaluations..."))
    t0 = time.time()
    n_ok = n_fail = 0

    for ds in DATASETS:
        if args.datasets and ds["name"] not in args.datasets:
            continue

        # Try canonical path first, then alt_dir in G:/drone/
        ds_root = EXTRACT_DIR / ds["name"]
        all_splits = find_all_splits(ds_root)
        if not all_splits and ds.get("alt_dir"):
            ds_root = GDRIVE / ds["alt_dir"]
            all_splits = find_all_splits(ds_root)
        if not all_splits:
            print(_red(f"  SKIP {ds['name']}: no images found"))
            n_fail += 1
            continue

        total_imgs = sum(count_images(sp) for _, sp in all_splits)
        split_names = "+".join(s for s, _ in all_splits)
        mode_label = "DRONE (TP/FP/FN)" if ds["has_drones"] else "CONFUSER (all-FP)"

        print(f"\n{'='*80}")
        print(_green(f"  DATASET: {ds['name']}"))
        print(f"  Modality: {ds['modality'].upper()}  |  Category: {ds['category']}  |  Mode: {mode_label}")
        print(f"  Total images: {total_imgs}  |  Splits: {split_names}")
        print(f"{'='*80}")

        # Determine which models to run
        if ds["modality"] == "rgb":
            model_runs = [
                ("rgb_baseline", MODELS["rgb_baseline"]),
                ("rgb_retrained_v2", MODELS["rgb_retrained_v2"]),
            ]
            patch_w = FILTERS["rgb"]
            conf = 0.25
        else:
            model_runs = [
                ("ir_model", MODELS["ir_model"]),
            ]
            patch_w = FILTERS["ir"]
            conf = 0.40

        for model_label, model_path in model_runs:
            print(f"\n  {'─'*70}")
            print(_green(f"  MODEL: {model_label}"))
            print(f"  Weights: {model_path.name}  ({model_path.parent.parent.name})")
            print(f"  Filter:  {patch_w.name}  |  Threshold: {args.patch_thr}")
            print(f"  {'─'*70}")

            for split_name, split_dir in all_splits:
                n_imgs = count_images(split_dir)
                stride = 1 if args.full else compute_stride(n_imgs, 500)
                print(f"\n    ▸ Split: {split_name}  ({n_imgs} imgs, stride={stride})")

                out = results_dir / ds["name"] / model_label / split_name
                ok = run_eval(
                    weights=model_path,
                    dataset_path=str(split_dir),
                    output_dir=str(out),
                    conf=conf,
                    patch_weights=str(patch_w),
                    patch_thr=args.patch_thr,
                    drone_classes=ds["drone_classes"],
                    stride=stride,
                    temporal=True,
                    model_name=model_label,
                )
                if ok:
                    n_ok += 1
                else:
                    n_fail += 1
                    print(_red(f"  FAILED: {ds['name']} / {model_label} / {split_name}"))

    elapsed = time.time() - t0
    print(_green(f"\n[2/3] Done: {n_ok} ok / {n_fail} failed in {elapsed/60:.1f} min"))

    # Aggregate
    print(_green("\n[3/3] Aggregating results..."))
    aggregate_results(results_dir)


if __name__ == "__main__":
    main()
