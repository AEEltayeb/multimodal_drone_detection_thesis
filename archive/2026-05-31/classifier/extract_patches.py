"""
extract_patches.py — Build crop dataset for patch-verifier training.

For each source video, pulls per-frame GT boxes from the Svanström .mat
labels, crops around each box with configurable context padding, and
writes the crop to disk.

Output:
  classifier/runs/patches/
      ir/{drone,airplane,helicopter,bird}/{stem}.jpg
      rgb/{drone,airplane,helicopter,bird}/{stem}.jpg
      manifest.csv   (stem, modality, label, category, video)

Downstream: classifier/train_patch_verifier.py reads the manifest and
trains a small CNN for each modality (drone vs aerial-negative).
"""

import argparse
import csv
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import scipy.io


CATEGORIES = ["DRONE", "AIRPLANE", "HELICOPTER", "BIRD"]


def extract_bboxes_from_mat(mat_path, n_frames):
    """Return {frame_idx: [(x,y,w,h), ...]} from Svanström .mat."""
    try:
        mat = scipy.io.loadmat(str(mat_path), squeeze_me=False,
                               struct_as_record=True)
    except Exception:
        return {}
    if "__function_workspace__" not in mat:
        return {}
    raw = mat["__function_workspace__"].tobytes()

    candidates = []
    for off in range(0, len(raw) - 31, 8):
        vals = struct.unpack_from("<4d", raw, off)
        x, y, w, h = vals
        if (1 < x < 2000 and 1 < y < 2000 and 2 < w < 2000 and 2 < h < 2000
                and not any(np.isnan(vals)) and not any(np.isinf(vals))):
            candidates.append((off, x, y, w, h))
    if len(candidates) < 3:
        return {}

    spacings = {}
    for i in range(len(candidates) - 1):
        s = candidates[i + 1][0] - candidates[i][0]
        if s > 0:
            spacings[s] = spacings.get(s, 0) + 1
    if not spacings:
        return {}
    stride = max(spacings, key=spacings.get)

    best, cur = [], [candidates[0]]
    for i in range(1, len(candidates)):
        if candidates[i][0] - cur[-1][0] == stride:
            cur.append(candidates[i])
        else:
            if len(cur) > len(best):
                best = cur
            cur = [candidates[i]]
    if len(cur) > len(best):
        best = cur
    if len(best) < n_frames * 0.5:
        return {}

    out = {}
    for fi in range(min(len(best), n_frames)):
        _, x, y, w, h = best[fi]
        if w > 1 and h > 1:
            out[fi] = [(x, y, w, h)]
    return out


def crop_with_context(img, x, y, w, h, pad_frac=0.5, min_side=24):
    """Return square-ish crop around bbox with context padding."""
    ih, iw = img.shape[:2]
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * (1.0 + 2.0 * pad_frac)
    side = max(side, float(min_side))
    x1 = int(round(cx - side / 2))
    y1 = int(round(cy - side / 2))
    x2 = int(round(cx + side / 2))
    y2 = int(round(cy + side / 2))
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(iw, x2); y2 = min(ih, y2)
    if x2 - x1 < min_side or y2 - y1 < min_side:
        return None
    return img[y1:y2, x1:x2]


def process_video(video_path, mat_path, category, modality,
                  out_dirs, manifest_rows, sample_every, max_crops_per_vid):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    bboxes = extract_bboxes_from_mat(mat_path, n_frames) if mat_path.exists() else {}
    if not bboxes:
        cap.release()
        return 0

    label = "drone" if category == "DRONE" else "aerial"
    cat_lower = category.lower()
    out_root = out_dirs[modality] / cat_lower
    out_root.mkdir(parents=True, exist_ok=True)

    stem_base = video_path.stem
    count = 0
    frame_idx = 0
    while count < max_crops_per_vid:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % sample_every != 0 or frame_idx not in bboxes:
            frame_idx += 1
            continue
        for bi, (x, y, w, h) in enumerate(bboxes[frame_idx]):
            crop = crop_with_context(frame, x, y, w, h)
            if crop is None:
                continue
            stem = f"{stem_base}_f{frame_idx:06d}_b{bi}"
            out_path = out_root / f"{stem}.jpg"
            cv2.imwrite(str(out_path), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            manifest_rows.append({
                "stem": stem,
                "path": str(out_path),
                "modality": modality,
                "label": label,
                "category": cat_lower,
                "video": video_path.name,
            })
            count += 1
            if count >= max_crops_per_vid:
                break
        frame_idx += 1
    cap.release()
    return count


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source",
                   default="G:/drone/Drone-detection-dataset-must-cite/"
                           "Drone-detection-dataset-master/Data")
    p.add_argument("--out", default="classifier/runs/patches")
    p.add_argument("--sample-every", type=int, default=15,
                   help="Sample every Nth frame (default 15).")
    p.add_argument("--max-crops-per-vid", type=int, default=80,
                   help="Cap crops from a single video (default 80).")
    p.add_argument("--max-videos-per-cat", type=int, default=60,
                   help="Cap videos per category (default 60).")
    args = p.parse_args()

    src = Path(args.source)
    ir_dir = src / "Video_IR"
    vis_dir = src / "Video_V"
    out = Path(args.out)
    out_dirs = {"ir": out / "ir", "rgb": out / "rgb"}
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    t0 = time.time()

    for cat in CATEGORIES:
        print(f"\n=== {cat} ===")
        # IR
        ir_vids = sorted([v for v in ir_dir.glob(f"IR_{cat}_*.mp4")])
        ir_vids = ir_vids[:args.max_videos_per_cat]
        ir_count = 0
        for vi, v in enumerate(ir_vids):
            mat = v.parent / f"{v.stem}_LABELS.mat"
            n = process_video(v, mat, cat, "ir", out_dirs, manifest_rows,
                              args.sample_every, args.max_crops_per_vid)
            ir_count += n
            if (vi + 1) % 10 == 0:
                print(f"  ir [{vi+1}/{len(ir_vids)}] crops={ir_count}")
        print(f"  IR total: {ir_count}")

        # Visible
        vis_vids = []
        for v in ir_vids:
            vis_name = v.name.replace("IR_", "V_")
            vp = vis_dir / vis_name
            if vp.exists():
                vis_vids.append(vp)
        vis_count = 0
        for vi, v in enumerate(vis_vids):
            mat = v.parent / f"{v.stem}_LABELS.mat"
            n = process_video(v, mat, cat, "rgb", out_dirs, manifest_rows,
                              args.sample_every, args.max_crops_per_vid)
            vis_count += n
            if (vi + 1) % 10 == 0:
                print(f"  rgb [{vi+1}/{len(vis_vids)}] crops={vis_count}")
        print(f"  RGB total: {vis_count}")

    manifest_path = out / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["stem", "path", "modality", "label",
                                          "category", "video"])
        w.writeheader()
        for row in manifest_rows:
            w.writerow(row)

    print(f"\nTotal crops: {len(manifest_rows)} in {(time.time()-t0)/60:.1f} min")
    print(f"Manifest: {manifest_path}")
    # Summary
    by_cell = {}
    for r in manifest_rows:
        key = (r["modality"], r["category"])
        by_cell[key] = by_cell.get(key, 0) + 1
    for (mod, cat), n in sorted(by_cell.items()):
        print(f"  {mod:>3s}/{cat:<11s} {n:>6d}")


if __name__ == "__main__":
    main()
