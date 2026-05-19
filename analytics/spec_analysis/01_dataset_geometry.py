"""
01_dataset_geometry.py — Per-dataset drone-size + background-clutter analysis.

For each YOLO-format dataset (images/ + labels/), compute:
  - Image dimensions distribution
  - GT box count, sqrt(area) percentiles (px), %frame-w/h percentiles
  - GT center-x/y distribution (where in the frame drones tend to be)
  - Background clutter score (mean Laplacian variance over a sample of images)
  - Per-source bucketing by filename prefix (first underscore-delimited token)

Outputs `analytics/spec_analysis/results/<dataset_tag>_geometry.csv`.

CPU-only. Doesn't conflict with a running GPU job.
"""

from __future__ import annotations
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "RGB model"))
from finetune_selcom import load_gt  # noqa: E402

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
OUT_DIR  = ROOT / "analytics" / "spec_analysis" / "results"

DATASETS = {
    "selcom_val":   dict(images=Path(r"G:/drone/_finetune_selcom_mixed_ft2/images/val"),
                          labels=Path(r"G:/drone/_finetune_selcom_mixed_ft2/labels/val"),
                          stride=1, max_sample=None),
    "selcom_cctv":  dict(images=Path(r"G:/drone/selcom_dataset/images"),
                          labels=Path(r"G:/drone/selcom_dataset/labels"),
                          stride=1, max_sample=None),
    "dataset_rgb":  dict(images=Path(r"G:/drone/dataset/dataset/images/test"),
                          labels=Path(r"G:/drone/dataset/dataset/labels/test"),
                          stride=1, max_sample=None),
    "antiuav_yolo": dict(images=Path(r"G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images"),
                          labels=Path(r"G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/labels"),
                          stride=20, max_sample=4000),   # 85k images, sample
    "svanstrom":    dict(images=Path(r"G:/drone/svanstrom_paired/RGB/images"),
                          labels=Path(r"G:/drone/svanstrom_paired/RGB/labels"),
                          stride=10, max_sample=3000),
}

CLUTTER_SAMPLE_N = 300   # how many images per dataset to compute Laplacian variance on


def source_bucket(stem: str) -> str:
    """First underscore-delimited token, with a couple of well-known multi-token
    sources collapsed to a single bucket."""
    if stem.startswith("anti_uav_"):       return "anti_uav"
    if stem.startswith("anti-muav-roboflow"): return "anti-muav-roboflow"
    if stem.startswith("wosdetc_"):        return "wosdetc"
    if stem.startswith("AirBird"):         return "AirBird"
    if stem.startswith("FBD-SV"):          return "FBD-SV"
    if stem.startswith("selcom"):          return "selcom"
    if stem.startswith("gen_"):            return "gen_mixed"
    if stem.startswith("mav"):             return "mav"
    if stem.startswith("dut"):             return "dut"
    if stem.startswith("BDD100K"):         return "BDD100K"
    if stem.startswith("VIRAT"):           return "VIRAT"
    if stem.startswith("UA-DETRAC"):       return "UA-DETRAC"
    return stem.split("_", 1)[0] if "_" in stem else "other"


def analyze_dataset(tag: str, cfg: dict) -> dict:
    img_dir: Path = cfg["images"]
    lbl_dir: Path = cfg["labels"]
    stride   = cfg.get("stride", 1)
    max_n    = cfg.get("max_sample")

    print(f"\n[{tag}] scanning {img_dir}")
    if not img_dir.exists():
        print(f"  [skip] {img_dir} does not exist")
        return None

    imgs = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
    if stride > 1:
        imgs = imgs[::stride]
    if max_n and len(imgs) > max_n:
        imgs = imgs[:max_n]
    print(f"  {len(imgs)} images (stride={stride}, max_sample={max_n})")

    # Per-source accumulators
    by_src = defaultdict(lambda: {
        "n_images": 0, "n_with_label": 0, "n_boxes": 0,
        "sqrt_areas_px": [], "rel_w": [], "rel_h": [],
        "cx": [], "cy": [],
    })
    img_w_seen, img_h_seen = set(), set()
    clutter_scores = []
    clutter_sample_paths = imgs[::max(1, len(imgs) // CLUTTER_SAMPLE_N)][:CLUTTER_SAMPLE_N]
    clutter_set = set(p.name for p in clutter_sample_paths)

    n_done = 0
    for img_path in imgs:
        n_done += 1
        if n_done % 500 == 0:
            print(f"    {n_done}/{len(imgs)} ...", flush=True)

        src = source_bucket(img_path.stem)
        b = by_src[src]
        b["n_images"] += 1

        # Image dims — read header only when needed (for box→px conversion)
        need_dims = True
        H = W = None
        # Try to skip imread when we only need labels by lazily loading
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        gt = load_gt(lbl_path) if lbl_path.exists() else []

        # We have to know W,H to compute px sizes
        if gt or img_path.name in clutter_set:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            H, W = frame.shape[:2]
            img_w_seen.add(W); img_h_seen.add(H)

            if img_path.name in clutter_set:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                clutter_scores.append(cv2.Laplacian(gray, cv2.CV_64F).var())

            if gt:
                b["n_with_label"] += 1
                b["n_boxes"] += len(gt)
                for (x1n, y1n, x2n, y2n) in gt:
                    bw = (x2n - x1n) * W
                    bh = (y2n - y1n) * H
                    b["sqrt_areas_px"].append(np.sqrt(max(bw * bh, 1e-6)))
                    b["rel_w"].append((x2n - x1n) * 100)
                    b["rel_h"].append((y2n - y1n) * 100)
                    b["cx"].append(0.5 * (x1n + x2n))
                    b["cy"].append(0.5 * (y1n + y2n))

    # Aggregate
    overall = {
        "n_images": sum(b["n_images"] for b in by_src.values()),
        "n_with_label": sum(b["n_with_label"] for b in by_src.values()),
        "n_boxes": sum(b["n_boxes"] for b in by_src.values()),
        "sqrt_areas_px": np.concatenate(
            [np.array(b["sqrt_areas_px"]) for b in by_src.values()
             if b["sqrt_areas_px"]]) if any(b["sqrt_areas_px"] for b in by_src.values()) else np.array([]),
        "rel_w": [v for b in by_src.values() for v in b["rel_w"]],
        "rel_h": [v for b in by_src.values() for v in b["rel_h"]],
        "cx": [v for b in by_src.values() for v in b["cx"]],
        "cy": [v for b in by_src.values() for v in b["cy"]],
    }

    def pct(arr, p):
        a = np.asarray(arr)
        return float(np.percentile(a, p)) if len(a) else float("nan")

    rows = []
    for src in sorted(by_src):
        b = by_src[src]
        rows.append({
            "dataset": tag, "source": src,
            "n_images": b["n_images"], "n_with_label": b["n_with_label"], "n_boxes": b["n_boxes"],
            "sqrt_px_p25": round(pct(b["sqrt_areas_px"], 25), 1),
            "sqrt_px_p50": round(pct(b["sqrt_areas_px"], 50), 1),
            "sqrt_px_p75": round(pct(b["sqrt_areas_px"], 75), 1),
            "sqrt_px_p90": round(pct(b["sqrt_areas_px"], 90), 1),
            "rel_w_p50": round(pct(b["rel_w"], 50), 2),
            "rel_h_p50": round(pct(b["rel_h"], 50), 2),
            "cx_p50": round(pct(b["cx"], 50), 3),
            "cy_p50": round(pct(b["cy"], 50), 3),
        })
    # overall row
    rows.append({
        "dataset": tag, "source": "_ALL",
        "n_images": overall["n_images"], "n_with_label": overall["n_with_label"], "n_boxes": overall["n_boxes"],
        "sqrt_px_p25": round(pct(overall["sqrt_areas_px"], 25), 1),
        "sqrt_px_p50": round(pct(overall["sqrt_areas_px"], 50), 1),
        "sqrt_px_p75": round(pct(overall["sqrt_areas_px"], 75), 1),
        "sqrt_px_p90": round(pct(overall["sqrt_areas_px"], 90), 1),
        "rel_w_p50": round(pct(overall["rel_w"], 50), 2),
        "rel_h_p50": round(pct(overall["rel_h"], 50), 2),
        "cx_p50": round(pct(overall["cx"], 50), 3),
        "cy_p50": round(pct(overall["cy"], 50), 3),
    })

    summary = {
        "dataset": tag,
        "image_dim_set": sorted(zip(sorted(img_w_seen), sorted(img_h_seen)))[:5],
        "clutter_laplacian_var_mean": round(float(np.mean(clutter_scores)), 1) if clutter_scores else None,
        "clutter_laplacian_var_median": round(float(np.median(clutter_scores)), 1) if clutter_scores else None,
        "clutter_n_sample": len(clutter_scores),
    }

    return {"rows": rows, "summary": summary}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_summary = {}
    for tag in args.datasets:
        if tag not in DATASETS:
            print(f"[warn] unknown dataset {tag}")
            continue
        result = analyze_dataset(tag, DATASETS[tag])
        if result is None:
            continue
        out_csv = OUT_DIR / f"{tag}_geometry.csv"
        with out_csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(result["rows"][0].keys()))
            writer.writeheader()
            writer.writerows(result["rows"])
        print(f"  -> {out_csv}")
        all_summary[tag] = result["summary"]

    # Write a master summary
    import json
    (OUT_DIR / "_geometry_summary.json").write_text(json.dumps(all_summary, indent=2, default=str))
    print(f"\nMaster summary: {OUT_DIR / '_geometry_summary.json'}")


if __name__ == "__main__":
    main()
