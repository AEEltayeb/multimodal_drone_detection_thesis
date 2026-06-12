"""smart_merge_lean17.py - Build Lean-17 dataset without re-reading images.

Takes the existing Lean-13 CSV (59k rows) and cached YOLO detection
bboxes, and augments each row with 4 derivable geometry features:
  rgb_best_pos_x, ir_best_pos_x,
  rgb_best_dist_to_center, ir_best_dist_to_center.

We drop target_bg_delta (needs pixel data) since it correlates strongly
with local_contrast which is already in Lean-13.

Image dimensions come from one cv2.imread per *clip/source*, not per
frame, so total I/O is ~few hundred reads instead of 70k.

Runs in seconds. Outputs:
  models/routers/lean17/fusion_dataset_lean17.csv
"""
import json, re, sys
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
LEAN13_CSV = REPO / "models/routers/lean13/fusion_dataset_lean13.csv"
ANTIUAV_CACHE = REPO / "models/routers/lean13/cache_antiuav.json"
SVAN_CACHE = REPO / "models/routers/lean13/cache_svanstrom.json"
VIDEO_CACHE_DIR = REPO / "docs/analysis/full_pipeline_ablations/cache"
VIDEO_TAG = "selcom_1280_sz1280"
IR_TAG = "ir_grayscale_sz640"
OUT_DIR = REPO / "models/routers/lean17"

LEAN13_FEATS = [
    "rgb_max_conf", "ir_max_conf",
    "rgb_best_log_bbox_area", "ir_best_log_bbox_area",
    "rgb_best_aspect_ratio", "ir_best_aspect_ratio",
    "rgb_best_pos_y", "ir_best_pos_y",
    "rgb_best_local_contrast", "ir_best_local_contrast",
    "rgb_img_mean", "ir_img_mean", "rgb_img_std",
]
LEAN17_FEATS = LEAN13_FEATS + [
    "rgb_best_pos_x", "ir_best_pos_x",
    "rgb_best_dist_to_center", "ir_best_dist_to_center",
]

# Known image dims per source (Anti-UAV, Svanstrom). Verified against repo notes.
SOURCE_DIMS = {
    # source : (rgb_w, rgb_h, ir_w, ir_h)
    "antiuav":   (1920, 1080, 640, 512),
    "svanstrom": (640, 480, 640, 512),  # ir 640x512 thermal, rgb 640x480 native
}

# Cache for video-clip image dimensions
_clip_dims_cache = {}


def get_clip_dims(source_tag):
    """For sources like 'video_drone_drone_takeoff_short', read 1 image to get dims."""
    if source_tag in _clip_dims_cache:
        return _clip_dims_cache[source_tag]
    # source_tag like 'video_drone_<clip_name>' or 'video_birds_<clip_name>'
    m = re.match(r"video_([^_]+)_(.+)", source_tag)
    if not m:
        _clip_dims_cache[source_tag] = None
        return None
    cat, clip = m.group(1), m.group(2)
    base = REPO / "datasets" / "drone detection video tests" / "rgb" / cat / clip
    img_d = base / "images" / "test"
    if not img_d.exists():
        img_d = base / "images"
    if not img_d.exists():
        _clip_dims_cache[source_tag] = None
        return None
    for f in img_d.iterdir():
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            img = cv2.imread(str(f))
            if img is not None:
                h, w = img.shape[:2]
                # Grayscale mode: both branches use same image -> same dims
                _clip_dims_cache[source_tag] = (w, h, w, h)
                return (w, h, w, h)
    _clip_dims_cache[source_tag] = None
    return None


def best_bbox(dets, conf_thresh=0.25):
    """Return (x1,y1,x2,y2) of highest-conf detection >= threshold, or None."""
    valid = [d for d in dets if d[4] >= conf_thresh]
    if not valid:
        return None
    return max(valid, key=lambda d: d[4])[:4]


def compute_pos(bbox, w, h):
    """pos_x, pos_y, dist_to_center from bbox + image dims."""
    if bbox is None or w <= 0 or h <= 0:
        return 0.0, 0.0, 0.0
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    pos_x = cx / w
    pos_y = cy / h
    dist = float(np.sqrt((pos_x - 0.5) ** 2 + (pos_y - 0.5) ** 2))
    return round(pos_x, 4), round(pos_y, 4), round(dist, 4)


def load_video_cache(source_tag):
    """Return (rgb_dets_dict, ir_dets_dict) for a video_<cat>_<clip> tag."""
    rc = VIDEO_CACHE_DIR / f"{source_tag}_{VIDEO_TAG}.json"
    ic = VIDEO_CACHE_DIR / f"{source_tag}_{IR_TAG}.json"
    rgb = json.load(open(rc))["dets"] if rc.exists() else {}
    ir  = json.load(open(ic))["dets"] if ic.exists() else {}
    return rgb, ir


def main():
    print(f"Loading lean13 CSV: {LEAN13_CSV.name}")
    df = pd.read_csv(LEAN13_CSV)
    print(f"  {len(df):,} rows")
    print(f"  sources: {dict(df['source'].value_counts())}")

    print(f"Loading caches...")
    auv_cache = json.load(open(ANTIUAV_CACHE)) if ANTIUAV_CACHE.exists() else {}
    svan_cache = json.load(open(SVAN_CACHE)) if SVAN_CACHE.exists() else {}
    print(f"  antiuav: {len(auv_cache)} entries")
    print(f"  svanstrom: {len(svan_cache)} entries")

    # Preload video caches for all video sources present
    video_caches = {}  # source_tag -> (rgb, ir)
    for src in df["source"].unique():
        if src.startswith("video_"):
            video_caches[src] = load_video_cache(src)
    print(f"  video clips with cache loaded: {len(video_caches)}")

    n_missing_auv = n_missing_svan = n_missing_video = 0
    rows_out = []

    for i, row in df.iterrows():
        src = row["source"]; stem = row["stem"]
        rgb_pos_x = ir_pos_x = rgb_dist = ir_dist = 0.0

        if src == "antiuav":
            rw, rh, iw, ih = SOURCE_DIMS["antiuav"]
            if stem in auv_cache:
                e = auv_cache[stem]
                rb = best_bbox(e.get("rgb_dets", []))
                ib = best_bbox(e.get("ir_dets", []))
                rgb_pos_x, _, rgb_dist = compute_pos(rb, rw, rh)
                ir_pos_x, _, ir_dist = compute_pos(ib, iw, ih)
            else:
                n_missing_auv += 1
        elif src == "svanstrom":
            rw, rh, iw, ih = SOURCE_DIMS["svanstrom"]
            if stem in svan_cache:
                e = svan_cache[stem]
                rb = best_bbox(e.get("rgb_dets", []))
                ib = best_bbox(e.get("ir_dets", []))
                rgb_pos_x, _, rgb_dist = compute_pos(rb, rw, rh)
                ir_pos_x, _, ir_dist = compute_pos(ib, iw, ih)
            else:
                n_missing_svan += 1
        elif src.startswith("video_"):
            dims = get_clip_dims(src)
            if dims is None:
                n_missing_video += 1
            else:
                w, h, _, _ = dims
                rgb_c, ir_c = video_caches.get(src, ({}, {}))
                # In lean13 generator, video row stem = f"{tag}_{image_stem}"
                # Strip tag prefix to recover image stem
                prefix = src + "_"
                img_stem = stem[len(prefix):] if stem.startswith(prefix) else stem
                rb = best_bbox(rgb_c.get(img_stem, []))
                ib = best_bbox(ir_c.get(img_stem, []))
                rgb_pos_x, _, rgb_dist = compute_pos(rb, w, h)
                ir_pos_x, _, ir_dist = compute_pos(ib, w, h)
        # else: yt rows wouldn't be in lean13 CSV; we handle yt separately

        out = {k: row[k] for k in LEAN13_FEATS}
        out["rgb_best_pos_x"] = rgb_pos_x
        out["ir_best_pos_x"] = ir_pos_x
        out["rgb_best_dist_to_center"] = rgb_dist
        out["ir_best_dist_to_center"] = ir_dist
        out["trust_label"] = row["trust_label"]
        out["stem"] = row["stem"]
        out["source"] = row["source"]
        rows_out.append(out)

    print(f"\n  Cache misses: antiuav={n_missing_auv}  svanstrom={n_missing_svan}  video={n_missing_video}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "fusion_dataset_lean17.csv"
    out_df = pd.DataFrame(rows_out, columns=LEAN17_FEATS + ["trust_label", "stem", "source"])
    out_df.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv}  ({len(out_df):,} rows)")

    # Print sanity stats on new features
    print("\n  New-feature distribution (non-zero rows):")
    for f in ["rgb_best_pos_x", "ir_best_pos_x", "rgb_best_dist_to_center", "ir_best_dist_to_center"]:
        nz = (out_df[f] != 0.0).sum()
        print(f"    {f}: nonzero={nz}/{len(out_df)} ({100*nz/len(out_df):.1f}%)")


if __name__ == "__main__":
    main()
