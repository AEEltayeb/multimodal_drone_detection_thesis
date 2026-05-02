"""
extract_background_crops.py — Mine "other" class crops from Svanström
and Anti-UAV frames, avoiding all labeled GT boxes.

Sources:
  1. Svanström original (G:/drone/Drone-detection-dataset-must-cite/...)
     - .mat per-class GT for DRONE/AIRPLANE/HELICOPTER/BIRD — all excluded.
     - Both Video_IR (ir) and Video_V (rgb).
  2. Anti-UAV YOLO-converted (G:/drone/Anti-UAV-RGBT_yolo_converted)
     - Drone-only labels (the only class) — excluded.
     - Both RGB and IR folders.

Strategy per frame:
  - Read all GT boxes (every class).
  - Sample up to N random square patches with sizes mimicking real object
    crops (min_side..max_side).
  - Drop any patch overlapping any GT box above IoU threshold.
  - Save with crop_with_context-equivalent padding (patch *is* the final
    crop — no re-pad since there's no underlying "object box" here).

Output:
  classifier/runs/patches/{rgb,ir}/background/bg_{source}_{stem}_c{n}.jpg
  appends rows to classifier/runs/patches/manifest.csv with
  category=background (→ "other" class in train_confuser_4class.py).

Usage:
  python classifier/extract_background_crops.py
  python classifier/extract_background_crops.py --crops-per-frame 3 \
      --sample-every 20 --max-per-video 60
"""

from __future__ import annotations

import argparse
import random
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import scipy.io

SCRIPT_DIR = Path(__file__).resolve().parent
PATCH_DIR = SCRIPT_DIR / "runs" / "patches"
MANIFEST_PATH = PATCH_DIR / "manifest.csv"
PROJECT_ROOT = SCRIPT_DIR.parent

SVAN_ORIG = Path(r"G:\drone\Drone-detection-dataset-must-cite"
                  r"\Drone-detection-dataset-master\Data")
ANTIUAV = Path(r"G:\drone\Anti-UAV-RGBT_yolo_converted")

SVAN_CATS = ["DRONE", "AIRPLANE", "HELICOPTER", "BIRD"]


# ── .mat label parsing (same heuristic as extract_patches.py) ───

def extract_bboxes_from_mat(mat_path: Path, n_frames: int):
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


# ── Geometry helpers ─────────────────────────────────────────────

def box_overlap(a, b) -> float:
    """IoU on xyxy boxes (floats)."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1); ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    return inter / max(1e-6, a_area + b_area - inter)


def sample_background_patches(img, gt_boxes_xyxy, n_wanted, min_side, max_side,
                              iou_thr, max_tries, rng):
    """Sample n_wanted square patches avoiding GT boxes. Returns list of
    (x1,y1,x2,y2) ints."""
    h, w = img.shape[:2]
    out = []
    tries = 0
    while len(out) < n_wanted and tries < max_tries:
        tries += 1
        side = rng.randint(min_side, max_side)
        if side >= min(w, h):
            continue
        x1 = rng.randint(0, w - side)
        y1 = rng.randint(0, h - side)
        x2 = x1 + side; y2 = y1 + side
        cand = (x1, y1, x2, y2)
        if any(box_overlap(cand, g) > iou_thr for g in gt_boxes_xyxy):
            continue
        out.append(cand)
    return out


# ── Svanström (original) mining ──────────────────────────────────

def mine_svanstrom(out_dirs, manifest_rows, sample_every, crops_per_frame,
                   max_per_video, min_side, max_side, iou_thr, rng,
                   max_videos_per_cat: int):
    ir_dir = SVAN_ORIG / "Video_IR"
    vis_dir = SVAN_ORIG / "Video_V"
    if not ir_dir.exists():
        print(f"[svanstrom] not found: {SVAN_ORIG} — skip")
        return 0

    total = 0
    for cat in SVAN_CATS:
        ir_vids = sorted(ir_dir.glob(f"IR_{cat}_*.mp4"))[:max_videos_per_cat]
        print(f"\n[svanstrom] cat={cat}  ir_videos={len(ir_vids)}")
        for vi, vpath in enumerate(ir_vids):
            # Pair: IR + matching visible
            vis_path = vis_dir / vpath.name.replace("IR_", "V_")
            mat_path = vpath.parent / f"{vpath.stem}_LABELS.mat"
            if not mat_path.exists():
                continue

            for modality, frame_src in [("ir", vpath), ("rgb", vis_path)]:
                if not frame_src.exists():
                    continue
                cap = cv2.VideoCapture(str(frame_src))
                if not cap.isOpened():
                    continue
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                bboxes = extract_bboxes_from_mat(mat_path, n_frames)
                if not bboxes:
                    cap.release()
                    continue

                saved = 0
                frame_idx = 0
                while saved < max_per_video:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if frame_idx % sample_every != 0:
                        frame_idx += 1
                        continue
                    gt_xyxy = [(x, y, x + w, y + h)
                                for (x, y, w, h) in bboxes.get(frame_idx, [])]
                    patches = sample_background_patches(
                        frame, gt_xyxy, crops_per_frame, min_side, max_side,
                        iou_thr, max_tries=crops_per_frame * 20, rng=rng)
                    for ci, (x1, y1, x2, y2) in enumerate(patches):
                        crop = frame[y1:y2, x1:x2]
                        stem = f"bg_svan_{cat.lower()}_{vpath.stem}_f{frame_idx:06d}_c{ci}"
                        out_dir = out_dirs[modality]
                        out_path = out_dir / f"{stem}.jpg"
                        cv2.imwrite(str(out_path), crop,
                                    [cv2.IMWRITE_JPEG_QUALITY, 90])
                        manifest_rows.append({
                            "stem": stem,
                            "path": str(out_path.relative_to(PROJECT_ROOT)),
                            "modality": modality,
                            "label": "neg",
                            "category": "background",
                            "video": f"svan_bg_{vpath.stem}_{modality}",
                        })
                        saved += 1
                        total += 1
                        if saved >= max_per_video:
                            break
                    frame_idx += 1
                cap.release()
            if (vi + 1) % 10 == 0:
                print(f"  [{vi+1}/{len(ir_vids)}] total so far: {total}")
    print(f"[svanstrom] done, {total} crops")
    return total


# ── Anti-UAV mining ──────────────────────────────────────────────

def read_yolo_boxes(lbl_path: Path, w: int, h: int):
    boxes = []
    if not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        boxes.append((x1, y1, x2, y2))
    return boxes


def mine_antiuav(out_dirs, manifest_rows, sample_every, crops_per_frame,
                 max_per_seq, min_side, max_side, iou_thr, rng,
                 max_sequences: int):
    if not ANTIUAV.exists():
        print(f"[antiuav] not found: {ANTIUAV} — skip")
        return 0

    total = 0
    for split in ["test", "val", "train"]:
        for mod_key, mod_folder in [("rgb", "RGB"), ("ir", "IR")]:
            img_dir = ANTIUAV / split / mod_folder / "images"
            lbl_dir = ANTIUAV / split / mod_folder / "labels"
            if not img_dir.exists():
                continue
            # Group images by sequence (prefix before _fNNNNNN)
            imgs = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
            seqs: dict[str, list[Path]] = {}
            for p in imgs:
                prefix = p.stem.rsplit("_f", 1)[0] if "_f" in p.stem else p.stem
                seqs.setdefault(prefix, []).append(p)
            seq_names = sorted(seqs)
            rng.shuffle(seq_names)
            seq_names = seq_names[:max_sequences]
            print(f"\n[antiuav] split={split} mod={mod_folder} "
                  f"sequences={len(seq_names)}")

            for si, sname in enumerate(seq_names):
                frames = sorted(seqs[sname])[::sample_every]
                saved = 0
                for ipath in frames:
                    if saved >= max_per_seq:
                        break
                    img = cv2.imread(str(ipath))
                    if img is None:
                        continue
                    h, w = img.shape[:2]
                    gt = read_yolo_boxes(lbl_dir / (ipath.stem + ".txt"), w, h)
                    patches = sample_background_patches(
                        img, gt, crops_per_frame, min_side, max_side,
                        iou_thr, max_tries=crops_per_frame * 20, rng=rng)
                    for ci, (x1, y1, x2, y2) in enumerate(patches):
                        crop = img[y1:y2, x1:x2]
                        stem = f"bg_antiuav_{split}_{ipath.stem}_c{ci}"
                        out_dir = out_dirs[mod_key]
                        out_path = out_dir / f"{stem}.jpg"
                        cv2.imwrite(str(out_path), crop,
                                    [cv2.IMWRITE_JPEG_QUALITY, 90])
                        manifest_rows.append({
                            "stem": stem,
                            "path": str(out_path.relative_to(PROJECT_ROOT)),
                            "modality": mod_key,
                            "label": "neg",
                            "category": "background",
                            "video": f"antiuav_{split}_{sname}_{mod_key}",
                        })
                        saved += 1
                        total += 1
                        if saved >= max_per_seq:
                            break
                if (si + 1) % 20 == 0:
                    print(f"  [{si+1}/{len(seq_names)}] total so far: {total}")
    print(f"[antiuav] done, {total} crops")
    return total


# ── MAIN ─────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample-every", type=int, default=30,
                    help="Only look at every Nth frame of each video.")
    p.add_argument("--crops-per-frame", type=int, default=2)
    p.add_argument("--max-per-video", type=int, default=40,
                    help="Cap crops per Svanström video (per modality).")
    p.add_argument("--max-per-seq", type=int, default=20,
                    help="Cap crops per Anti-UAV sequence (per modality).")
    p.add_argument("--max-svan-videos-per-cat", type=int, default=40)
    p.add_argument("--max-antiuav-sequences", type=int, default=200)
    p.add_argument("--min-side", type=int, default=40,
                    help="Min square patch size in pixels.")
    p.add_argument("--max-side", type=int, default=180,
                    help="Max square patch size in pixels.")
    p.add_argument("--iou-threshold", type=float, default=0.02,
                    help="Reject patch if IoU with any GT > this.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--source", choices=["svanstrom", "antiuav", "both"],
                    default="both")
    args = p.parse_args()

    rng = random.Random(args.seed)
    out_dirs = {
        "ir": PATCH_DIR / "ir" / "background",
        "rgb": PATCH_DIR / "rgb" / "background",
    }
    for d in out_dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(MANIFEST_PATH)
    existing_stems = set(manifest["stem"].values)
    print(f"Existing manifest: {len(manifest)} rows")

    manifest_rows = []
    t0 = time.time()

    if args.source in ("svanstrom", "both"):
        mine_svanstrom(
            out_dirs, manifest_rows, args.sample_every, args.crops_per_frame,
            args.max_per_video, args.min_side, args.max_side,
            args.iou_threshold, rng, args.max_svan_videos_per_cat,
        )

    if args.source in ("antiuav", "both"):
        mine_antiuav(
            out_dirs, manifest_rows, args.sample_every, args.crops_per_frame,
            args.max_per_seq, args.min_side, args.max_side,
            args.iou_threshold, rng, args.max_antiuav_sequences,
        )

    # Deduplicate against existing stems and append
    fresh_rows = [r for r in manifest_rows if r["stem"] not in existing_stems]
    print(f"\nCollected {len(manifest_rows)} crops "
          f"({len(fresh_rows)} new after dedupe) in {time.time()-t0:.1f}s")

    if fresh_rows:
        new_df = pd.DataFrame(fresh_rows)
        updated = pd.concat([manifest, new_df], ignore_index=True)
        updated.to_csv(MANIFEST_PATH, index=False)
        print(f"Manifest: {len(manifest)} → {len(updated)} rows")
        print("\nBackground distribution by modality:")
        bg = new_df.groupby("modality").size().to_dict()
        for k, v in bg.items():
            print(f"  {k}: {v}")
    else:
        print("No new crops added.")


if __name__ == "__main__":
    main()
