"""
08_failure_sampler.py — Sample worst-FN and worst-FP frames per (model, surface).

For each (model, surface) cell, pick:
  - up to 15 worst-FP frames (rank: confidence of highest-conf FP detection)
  - up to 15 worst-FN frames (rank: presence of GT with no matching TP)

Writes:
  docs/analysis/failure_samples/<surface>/<model>/<failure_type>_<rank>_<stem>.jpg
      512-px-wide crop with bbox overlays (red=FP, green=GT, yellow=GT-missed)
  docs/analysis/2026-05-19_failure_mode_tags.csv
      One row per sampled frame. Columns: model, surface, failure_type, stem,
      det_conf, gt_box_area, my_tags, user_tags, notes.

`my_tags` is populated heuristically (size + simple scene cues from the source
filename pattern). User can override `user_tags` later.

Operates on:
  - Roboflow OOD: eval/results/roboflow_ood/<ds>/<m>/<split>/<m>_frame_detections.csv
  - Real-video: eval/results/video_tests/<cat>/<clip>/<m>.json (sizes/dets summary)
"""

from __future__ import annotations
import argparse
import ast
import csv
import sys
from pathlib import Path

import cv2

EVAL_DIR = Path(__file__).resolve().parents[2] / "eval"
REPO = EVAL_DIR.parent
OUT_DIR = REPO / "docs" / "analysis" / "failure_samples"
TAG_CSV = REPO / "docs" / "analysis" / "2026-05-19_failure_mode_tags.csv"

CONTROLLED_VOCAB = [
    "very_small", "small", "medium", "large",
    "clutter_trees", "clutter_buildings", "clutter_horizon", "sky_only",
    "sun_glare", "cloud_edge",
    "bird_lookalike", "airplane_lookalike", "helicopter_lookalike",
    "edge_of_frame", "motion_blur", "occlusion", "low_contrast",
]


def parse_dets(s: str) -> list[tuple[float, float, float, float, float]]:
    """Parse semicolon-separated dets: 'x1,y1,x2,y2,conf;...'"""
    out = []
    if not s:
        return out
    for chunk in s.split(";"):
        parts = chunk.split(",")
        if len(parts) != 5:
            continue
        try:
            out.append(tuple(float(x) for x in parts))
        except ValueError:
            continue
    return out


def heuristic_tags(category: str, has_dets: bool, dets, sizes: str) -> list[str]:
    """Propose tags based on minimal cues."""
    tags = []
    sizes_list = sizes.split(";") if sizes else []
    if sizes_list:
        small_n = sum(1 for s in sizes_list if s == "small")
        if small_n == len(sizes_list):
            tags.append("small")
        elif sizes_list[0] == "large":
            tags.append("large")
        else:
            tags.append("medium")
    if category:
        cat = category.lower()
        if "bird" in cat:
            tags.append("bird_lookalike")
        elif "airplane" in cat or "plane" in cat:
            tags.append("airplane_lookalike")
        elif "heli" in cat:
            tags.append("helicopter_lookalike")
    return tags


def find_image(stem: str, candidate_dirs: list[Path]) -> Path | None:
    for d in candidate_dirs:
        for ext in (".jpg", ".jpeg", ".png", ".bmp"):
            p = d / f"{stem}{ext}"
            if p.exists():
                return p
    return None


def overlay_and_save(img_path: Path, out_path: Path, dets, gt_boxes, max_w: int = 1024):
    img = cv2.imread(str(img_path))
    if img is None:
        return False
    h, w = img.shape[:2]
    for d in dets:
        if len(d) < 4: continue
        x1, y1, x2, y2 = map(int, d[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # red = det
    for g in gt_boxes:
        x1, y1, x2, y2 = map(int, g[:4])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # green = GT
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (max_w, int(h * scale)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)
    return True


def sample_roboflow(n_per_cell: int = 10):
    """Walk Roboflow per-frame CSVs and sample worst FP + FN."""
    rob = EVAL_DIR / "results" / "roboflow_ood"
    extract = REPO / "datasets" / "roboflow_ood"  # candidate image roots
    gd = Path("G:/drone")
    rows_for_tag_csv = []

    for csv_path in sorted(rob.rglob("*_frame_detections.csv")):
        parts = csv_path.relative_to(rob).parts
        if len(parts) < 4:
            continue
        ds, model, split = parts[0], parts[1], parts[2]
        category = ds.split("_", 1)[1] if "_" in ds else ds  # rgb_bird -> bird
        # Candidate image roots (multiple known patterns)
        candidate_dirs = [
            extract / ds / split / "images",
            gd / "_dl_roboflow_ood" / ds / split / "images",
            gd / ds / split / "images",
        ]

        with csv_path.open() as f:
            data = list(csv.DictReader(f))
        # Sort by worst FP: highest det_conf in dets where fp>0
        fp_rows = []
        for r in data:
            if int(r.get("fp", 0)) == 0:
                continue
            dets = parse_dets(r.get("dets", ""))
            if not dets:
                continue
            top_conf = max(d[4] for d in dets)
            fp_rows.append((top_conf, r, dets))
        fp_rows.sort(reverse=True, key=lambda x: x[0])

        # FN frames: n_gt>0 and tp==0
        fn_rows = [(int(r.get("n_gt", 0)), r) for r in data
                   if int(r.get("n_gt", 0)) > 0 and int(r.get("tp", 0)) == 0]
        fn_rows.sort(reverse=True, key=lambda x: x[0])

        out_cell = OUT_DIR / f"roboflow_{ds}__{split}" / model
        for rank, (conf, r, dets) in enumerate(fp_rows[:n_per_cell], 1):
            stem = r["stem"]
            ip = find_image(stem, candidate_dirs)
            if not ip:
                continue
            op = out_cell / f"FP_{rank:02d}_conf{conf:.2f}_{stem[:60]}.jpg"
            overlay_and_save(ip, op, dets, [])
            rows_for_tag_csv.append({
                "model": model, "surface": f"roboflow_{ds}/{split}",
                "failure_type": "FP", "rank": rank, "stem": stem,
                "det_conf": round(conf, 4), "gt_box_area": "",
                "my_tags": ";".join(heuristic_tags(category, True, dets, r.get("sizes", ""))),
                "user_tags": "", "notes": "", "image_path": str(op.relative_to(REPO)),
            })
        for rank, (n_gt, r) in enumerate(fn_rows[:n_per_cell], 1):
            stem = r["stem"]
            ip = find_image(stem, candidate_dirs)
            if not ip:
                continue
            op = out_cell / f"FN_{rank:02d}_ngt{n_gt}_{stem[:60]}.jpg"
            overlay_and_save(ip, op, [], [])
            rows_for_tag_csv.append({
                "model": model, "surface": f"roboflow_{ds}/{split}",
                "failure_type": "FN", "rank": rank, "stem": stem,
                "det_conf": "", "gt_box_area": "",
                "my_tags": ";".join(heuristic_tags(category, False, [], "")),
                "user_tags": "", "notes": "", "image_path": str(op.relative_to(REPO)),
            })
    return rows_for_tag_csv


def sample_svanstrom(n_per_cell: int = 10):
    """Use _patch_catch_audit/baseline_v2/per_detection.csv to surface worst FPs."""
    pd_csv = EVAL_DIR / "results" / "_patch_catch_audit" / "baseline_v2" / "per_detection.csv"
    if not pd_csv.exists():
        return []
    rows = list(csv.DictReader(pd_csv.open()))
    # FP candidates: best_iop == 0 (no GT match) ranked by det_conf
    fp_by_cat = {}
    for r in rows:
        cat = r["category"]
        try:
            iop = float(r.get("matched_iop", 0))
            conf = float(r.get("det_conf", 0))
        except ValueError:
            continue
        if cat == "DRONE" and iop > 0:
            continue
        fp_by_cat.setdefault(cat, []).append((conf, r))
    out_rows = []
    img_root = Path("G:/drone/svanstrom_paired/RGB/images")
    for cat, items in fp_by_cat.items():
        items.sort(reverse=True, key=lambda x: x[0])
        for rank, (conf, r) in enumerate(items[:n_per_cell], 1):
            stem = r["frame"]
            ip = find_image(stem, [img_root])
            op = OUT_DIR / "svanstrom_baseline" / cat / f"FP_{rank:02d}_conf{conf:.2f}_{stem[:60]}.jpg"
            if ip:
                overlay_and_save(ip, op, [], [])
            out_rows.append({
                "model": "baseline", "surface": "svanstrom_rgb",
                "failure_type": "FP", "rank": rank, "stem": stem,
                "det_conf": round(conf, 4), "gt_box_area": "",
                "my_tags": ";".join(heuristic_tags(cat, True, [], r.get("bucket", ""))),
                "user_tags": "", "notes": f"category={cat}, patch_label={r.get('patch_label', '')}",
                "image_path": str(op.relative_to(REPO)) if ip else "",
            })
    return out_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="samples per (model, surface, failure_type)")
    ap.add_argument("--surfaces", nargs="*", default=["roboflow", "svanstrom"])
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    if "roboflow" in args.surfaces:
        print("Sampling Roboflow...")
        all_rows.extend(sample_roboflow(args.n))
    if "svanstrom" in args.surfaces:
        print("Sampling Svanstrom...")
        all_rows.extend(sample_svanstrom(args.n))

    if all_rows:
        with TAG_CSV.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"Wrote {len(all_rows)} sample rows to {TAG_CSV}")
        print(f"Image crops in {OUT_DIR}")
        print(f"\nControlled vocabulary for user_tags column:")
        print(f"  {', '.join(CONTROLLED_VOCAB)}")
    else:
        print("No samples produced.")


if __name__ == "__main__":
    main()
