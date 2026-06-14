"""build_ablation_datasets.py - Construct 3 ablation variant CSVs from
the existing fusion_dataset_lean19.csv, without re-running YOLO.

Variant B (strict trust_both labels):
  For each row with trust_label=3, look up the cached RGB and IR best-detection
  bboxes. Compute the normalised centroid distance between RGB and IR bboxes
  (each normalised against its own image dimensions). If that distance exceeds
  --max-dist (default 0.05 = 5% of frame diagonal), demote the label:
    rgb_max_conf >= ir_max_conf -> trust_RGB (1) ; else trust_IR (2).
  This punishes the 'IR-coincidentally-on-drone' pattern that the lean19
  generator labelled as trust_both, especially on grayscale-fed IR rows.

Variant C (xmodal features):
  Add 3 cross-modal agreement features computed from cached detections:
    xmodal_centroid_dist  - normalised L2 distance between RGB and IR best-bbox centroids
    xmodal_scale_ratio    - min(log_area)/max(log_area) of best detections
    xmodal_conf_ratio     - min(max_conf)/max(max_conf) of the two modalities

Variant BC: B + C combined.

Image dimensions are taken from SOURCE_DIMS (AntiUAV / Svanstrom) and from one
cv2.imread per video_* clip (cached in memory).

Outputs:
  models/routers/lean19_v2_B/fusion_dataset.csv  (19 features, strict labels)
  models/routers/lean19_v2_C/fusion_dataset.csv  (22 features, original labels)
  models/routers/lean19_v2_BC/fusion_dataset.csv (22 features, strict labels)
  meta.json in each dir with stats.
"""
import argparse, json, re
from pathlib import Path
from collections import Counter
import cv2, numpy as np, pandas as pd

REPO = Path(__file__).resolve().parent.parent
LEAN19_CSV = REPO / "models/routers/lean19/fusion_dataset_lean19.csv"
ANTIUAV_CACHE = REPO / "models/routers/lean19/cache_antiuav.json"
SVAN_CACHE = REPO / "models/routers/lean19/cache_svanstrom.json"
VIDEO_CACHE_DIR = REPO / "docs/analysis/full_pipeline_ablations/cache"
VIDEO_RGB_TAG = "selcom_1280_sz1280"
IR_TAG = "ir_grayscale_sz640"

LEAN19_FEATS = [
    "rgb_max_conf", "ir_max_conf",
    "rgb_best_log_bbox_area", "ir_best_log_bbox_area",
    "rgb_best_aspect_ratio", "ir_best_aspect_ratio",
    "rgb_best_pos_y", "ir_best_pos_y",
    "rgb_best_local_contrast", "ir_best_local_contrast",
    "rgb_img_mean", "ir_img_mean", "rgb_img_std",
    "rgb_best_pos_x", "ir_best_pos_x",
    "rgb_best_dist_to_center", "ir_best_dist_to_center",
    "rgb_best_target_bg_delta", "ir_best_target_bg_delta",
]
XMODAL_FEATS = ["xmodal_centroid_dist", "xmodal_scale_ratio", "xmodal_conf_ratio"]
ALL_FEATS_BC = LEAN19_FEATS + XMODAL_FEATS

SOURCE_DIMS = {
    "antiuav":   (1920, 1080, 640, 512),  # rgb_w rgb_h ir_w ir_h
    "svanstrom": (640, 480, 640, 512),
}

_clip_dim_cache = {}


def get_clip_dims(source):
    """For video_* sources, lazily read one image to get its dimensions."""
    if source in _clip_dim_cache: return _clip_dim_cache[source]
    if source.startswith("confuser_"):
        # yt clips - read source video to get dims. Used cached read.
        _clip_dim_cache[source] = (1280, 720, 1280, 720)  # typical yt; close enough for ratios
        return _clip_dim_cache[source]
    m = re.match(r"video_([^_]+)_(.+)", source)
    if not m:
        _clip_dim_cache[source] = None; return None
    cat, clip = m.group(1), m.group(2)
    base = REPO / "datasets" / "drone detection video tests" / "rgb" / cat / clip
    img_d = base / "images" / "test"
    if not img_d.exists(): img_d = base / "images"
    if not img_d.exists():
        _clip_dim_cache[source] = None; return None
    for f in img_d.iterdir():
        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"):
            img = cv2.imread(str(f))
            if img is not None:
                h, w = img.shape[:2]
                _clip_dim_cache[source] = (w, h, w, h)
                return _clip_dim_cache[source]
    _clip_dim_cache[source] = None; return None


def best_bbox(dets, thr=0.25):
    valid = [d for d in dets if d[4] >= thr]
    return max(valid, key=lambda d: d[4])[:4] if valid else None


def centroid_norm(bbox, w, h):
    if bbox is None or w <= 0 or h <= 0: return None
    return ((bbox[0] + bbox[2]) / 2 / w, (bbox[1] + bbox[3]) / 2 / h)


def log_area(bbox):
    if bbox is None: return 0.0
    pw = max(1.0, bbox[2] - bbox[0]); ph = max(1.0, bbox[3] - bbox[1])
    return float(np.log(pw * ph + 1.0))


def compute_xmodal(rgb_bbox, ir_bbox, dims):
    rw, rh, iw, ih = dims
    rc = centroid_norm(rgb_bbox, rw, rh)
    ic = centroid_norm(ir_bbox, iw, ih)
    if rc is None or ic is None:
        return 0.0, 0.0, 0.0
    dist = float(np.sqrt((rc[0]-ic[0])**2 + (rc[1]-ic[1])**2))
    rla, ila = log_area(rgb_bbox), log_area(ir_bbox)
    scale_ratio = min(rla, ila) / max(rla, ila) if max(rla, ila) > 0 else 0.0
    return round(dist, 4), round(scale_ratio, 4), 0.0  # conf_ratio filled separately


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-dist", type=float, default=0.05,
                    help="Centroid distance threshold for strict trust_both (default 0.05)")
    args = ap.parse_args()

    print(f"Loading {LEAN19_CSV.name} ...")
    df = pd.read_csv(LEAN19_CSV)
    print(f"  {len(df):,} rows")
    print(f"  trust_label dist: {dict(df['trust_label'].value_counts().sort_index())}")

    print("Loading caches ...")
    auv_cache = json.load(open(ANTIUAV_CACHE))
    svan_cache = json.load(open(SVAN_CACHE))
    print(f"  antiuav cache: {len(auv_cache):,}  svan cache: {len(svan_cache):,}")
    video_caches = {}
    for src in df["source"].unique():
        if src.startswith("video_"):
            rgb_c = VIDEO_CACHE_DIR / f"{src}_{VIDEO_RGB_TAG}.json"
            ir_c = VIDEO_CACHE_DIR / f"{src}_{IR_TAG}.json"
            if rgb_c.exists() and ir_c.exists():
                video_caches[src] = (
                    json.load(open(rgb_c))["dets"],
                    json.load(open(ir_c))["dets"],
                )
    print(f"  video caches: {len(video_caches)} clips loaded")

    # Compute xmodal features + strict labels for every row
    new_labels = []
    xmodal_dists = []
    xmodal_scales = []
    xmodal_confs = []
    n_demoted_to_rgb = n_demoted_to_ir = n_kept_both = 0
    n_missing_lookup = 0

    for i, row in df.iterrows():
        src = str(row["source"]); stem = str(row["stem"])
        rgb_max = float(row["rgb_max_conf"]); ir_max = float(row["ir_max_conf"])

        # Look up cached bboxes
        rgb_b = ir_b = None; dims = None
        if src == "antiuav":
            e = auv_cache.get(stem)
            if e:
                rgb_b = best_bbox(e.get("rgb_dets", []))
                ir_b = best_bbox(e.get("ir_dets", []))
            dims = SOURCE_DIMS["antiuav"]
        elif src == "svanstrom":
            e = svan_cache.get(stem)
            if e:
                rgb_b = best_bbox(e.get("rgb_dets", []))
                ir_b = best_bbox(e.get("ir_dets", []))
            dims = SOURCE_DIMS["svanstrom"]
        elif src.startswith("video_") and src in video_caches:
            rd_c, id_c = video_caches[src]
            prefix = src + "_"
            img_stem = stem[len(prefix):] if stem.startswith(prefix) else stem
            rgb_b = best_bbox(rd_c.get(img_stem, []))
            ir_b = best_bbox(id_c.get(img_stem, []))
            dims = get_clip_dims(src)
        elif src.startswith("confuser_"):
            # yt rows: detections were not cached. Use rgb_best_pos_x and rgb_best_pos_y from CSV
            # to derive bboxes is not possible; just compute xmodal with conf/area we have.
            dims = get_clip_dims(src)
        else:
            n_missing_lookup += 1

        if dims:
            d, s, _ = compute_xmodal(rgb_b, ir_b, dims)
        else:
            d, s = 0.0, 0.0

        # Conf ratio uses CSV values directly
        if rgb_max > 0 and ir_max > 0:
            c = float(min(rgb_max, ir_max) / max(rgb_max, ir_max))
        else:
            c = 0.0

        xmodal_dists.append(d); xmodal_scales.append(s); xmodal_confs.append(round(c, 4))

        # Strict label rule (variant B / BC)
        new_lab = int(row["trust_label"])
        if new_lab == 3:
            if rgb_b is None or ir_b is None or d > args.max_dist:
                # Demote
                if rgb_max >= ir_max:
                    new_lab = 1; n_demoted_to_rgb += 1
                else:
                    new_lab = 2; n_demoted_to_ir += 1
            else:
                n_kept_both += 1
        new_labels.append(new_lab)

    print(f"\nStrict-label transitions on class-3 rows:")
    print(f"  kept trust_both : {n_kept_both:,}")
    print(f"  demoted to RGB  : {n_demoted_to_rgb:,}")
    print(f"  demoted to IR   : {n_demoted_to_ir:,}")
    print(f"  lookup misses   : {n_missing_lookup:,}")

    df["xmodal_centroid_dist"] = xmodal_dists
    df["xmodal_scale_ratio"] = xmodal_scales
    df["xmodal_conf_ratio"] = xmodal_confs
    df["trust_label_strict"] = new_labels

    print(f"\nStrict trust_label dist:")
    print(f"  {dict(pd.Series(new_labels).value_counts().sort_index())}")

    # Write outputs
    for tag, label_col, cols in [
        ("B", "trust_label_strict", LEAN19_FEATS),
        ("C", "trust_label", ALL_FEATS_BC),
        ("BC", "trust_label_strict", ALL_FEATS_BC),
    ]:
        out_dir = REPO / "models/routers" / f"lean19_v2_{tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        sub = df[cols + [label_col, "stem", "source"]].copy()
        sub = sub.rename(columns={label_col: "trust_label"})
        out = out_dir / "fusion_dataset.csv"
        sub.to_csv(out, index=False)
        meta = {
            "tag": f"lean19_v2_{tag}",
            "n_features": len(cols), "features": cols,
            "label_col_source": label_col, "n_rows": len(sub),
            "trust_dist": dict(sub["trust_label"].value_counts().sort_index().items()),
            "strict_max_dist": args.max_dist if "B" in tag else None,
        }
        json.dump(meta, open(out_dir / "meta.json", "w"), indent=2)
        print(f"  -> {out.relative_to(REPO)}  ({len(sub):,} rows, {len(cols)} features)")


if __name__ == "__main__":
    main()
