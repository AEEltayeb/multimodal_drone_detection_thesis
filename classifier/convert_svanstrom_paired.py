"""
convert_svanstrom_paired.py — Convert Svanstrom IR+Visible videos to paired YOLO dataset.

Extracts frames from paired IR/Visible videos, parses MATLAB .mat labels,
converts to YOLO format, and outputs a paired dataset structure.

Only processes videos that exist in BOTH modalities.

Output structure:
    {output_root}/
        IR/images/   IR/labels/
        RGB/images/  RGB/labels/

Usage:
    python convert_svanstrom_paired.py
    python convert_svanstrom_paired.py --sample-every 5 --max-videos 10
"""

import argparse
import json
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import scipy.io


# ---------------------------------------------------------------------------
# MATLAB .mat label parsing
# ---------------------------------------------------------------------------

def extract_bboxes_from_mat(mat_path, n_frames):
    """
    Extract per-frame bounding boxes from MATLAB groundTruth .mat file.

    Scans __function_workspace__ for contiguous runs of 4-double arrays
    with a consistent stride that look like [x, y, w, h] bboxes.

    Returns: dict { frame_idx: list of (x, y, w, h) }
    """
    try:
        mat = scipy.io.loadmat(str(mat_path), squeeze_me=False, struct_as_record=True)
    except Exception as e:
        print(f"  WARNING: Could not load {mat_path}: {e}")
        return {}

    if '__function_workspace__' not in mat:
        return {}

    raw = mat['__function_workspace__'].tobytes()

    # Step 1: Find all candidate 4-double groups that look like bboxes
    candidates = []
    for offset in range(0, len(raw) - 31, 8):
        vals = struct.unpack_from('<4d', raw, offset)
        x, y, w, h = vals
        if (1 < x < 2000 and 1 < y < 2000 and 2 < w < 2000 and 2 < h < 2000
                and not any(np.isnan(vals)) and not any(np.isinf(vals))):
            candidates.append((offset, x, y, w, h))

    if len(candidates) < 3:
        return {}

    # Step 2: Find dominant stride between consecutive candidates
    spacings = {}
    for i in range(len(candidates) - 1):
        s = candidates[i + 1][0] - candidates[i][0]
        if s > 0:
            spacings[s] = spacings.get(s, 0) + 1

    if not spacings:
        return {}

    dominant_stride = max(spacings, key=spacings.get)

    # Step 3: Extract the longest contiguous run at this stride
    best_run = []
    current_run = [candidates[0]]
    for i in range(1, len(candidates)):
        if candidates[i][0] - current_run[-1][0] == dominant_stride:
            current_run.append(candidates[i])
        else:
            if len(current_run) > len(best_run):
                best_run = current_run
            current_run = [candidates[i]]
    if len(current_run) > len(best_run):
        best_run = current_run

    if len(best_run) < n_frames * 0.5:
        # Too few — likely not the right data
        return {}

    # Step 4: Assign to frames (1 bbox per frame)
    frame_bboxes = {}
    for fi in range(min(len(best_run), n_frames)):
        _, x, y, w, h = best_run[fi]
        if w > 1 and h > 1:
            frame_bboxes[fi] = [(x, y, w, h)]

    return frame_bboxes


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------

def bbox_to_yolo(x, y, w, h, img_w, img_h, cls=0):
    """Convert (x, y, w, h) top-left absolute to YOLO normalized (cx, cy, w, h)."""
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    # Clamp to [0, 1]
    cx = max(0, min(1, cx))
    cy = max(0, min(1, cy))
    nw = max(0, min(1, nw))
    nh = max(0, min(1, nh))
    return f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"


def process_video_pair(ir_video, vis_video, ir_mat, vis_mat,
                       ir_img_dir, ir_lbl_dir, vis_img_dir, vis_lbl_dir,
                       video_name, sample_every=3, is_drone=True):
    """Process one paired IR+Visible video. Returns frame count."""
    ir_cap = cv2.VideoCapture(str(ir_video))
    vis_cap = cv2.VideoCapture(str(vis_video))

    ir_n = int(ir_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vis_n = int(vis_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ir_w = int(ir_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    ir_h = int(ir_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vis_w = int(vis_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vis_h = int(vis_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    n_frames = min(ir_n, vis_n)

    # Parse labels
    ir_bboxes = extract_bboxes_from_mat(ir_mat, ir_n) if ir_mat.exists() else {}
    vis_bboxes = extract_bboxes_from_mat(vis_mat, vis_n) if vis_mat.exists() else {}

    count = 0
    for fi in range(n_frames):
        ir_ok, ir_frame = ir_cap.read()
        vis_ok, vis_frame = vis_cap.read()

        if not ir_ok or not vis_ok:
            break

        if fi % sample_every != 0:
            continue

        stem = f"{video_name}_f{fi:06d}"

        # Save images
        cv2.imwrite(str(ir_img_dir / f"{stem}_infrared.jpg"), ir_frame)
        cv2.imwrite(str(vis_img_dir / f"{stem}_visible.jpg"), vis_frame)

        # Save IR label
        ir_lbl_path = ir_lbl_dir / f"{stem}_infrared.txt"
        if is_drone and fi in ir_bboxes:
            lines = [bbox_to_yolo(*bb, ir_w, ir_h) for bb in ir_bboxes[fi]]
            ir_lbl_path.write_text("\n".join(lines))
        else:
            ir_lbl_path.write_text("")

        # Save visible label
        vis_lbl_path = vis_lbl_dir / f"{stem}_visible.txt"
        if is_drone and fi in vis_bboxes:
            lines = [bbox_to_yolo(*bb, vis_w, vis_h) for bb in vis_bboxes[fi]]
            vis_lbl_path.write_text("\n".join(lines))
        else:
            vis_lbl_path.write_text("")

        count += 1

    ir_cap.release()
    vis_cap.release()
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="G:/drone/Drone-detection-dataset-must-cite/Drone-detection-dataset-master/Data")
    parser.add_argument("--output", default="G:/drone/svanstrom_paired")
    parser.add_argument("--sample-every", type=int, default=3,
                        help="Sample every Nth frame (default: 3)")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Limit total videos processed (for testing)")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    ir_vid_dir = source / "Video_IR"
    vis_vid_dir = source / "Video_V"

    # Create output structure
    for modality in ["IR", "RGB"]:
        for sub in ["images", "labels"]:
            (output / modality / sub).mkdir(parents=True, exist_ok=True)

    ir_img_dir = output / "IR" / "images"
    ir_lbl_dir = output / "IR" / "labels"
    vis_img_dir = output / "RGB" / "images"
    vis_lbl_dir = output / "RGB" / "labels"

    # Find all paired videos
    ir_videos = sorted([f for f in ir_vid_dir.iterdir() if f.suffix == '.mp4'])
    pairs = []
    for ir_vid in ir_videos:
        vis_name = ir_vid.name.replace("IR_", "V_")
        vis_vid = vis_vid_dir / vis_name
        if vis_vid.exists():
            pairs.append((ir_vid, vis_vid))

    print(f"Found {len(pairs)} paired videos")

    if args.max_videos:
        pairs = pairs[:args.max_videos]
        print(f"  Limited to {len(pairs)} videos")

    # Process
    t_start = time.time()
    total_frames = 0
    stats = {"drone": 0, "airplane": 0, "bird": 0, "helicopter": 0}

    for idx, (ir_vid, vis_vid) in enumerate(pairs):
        video_name = ir_vid.stem  # e.g. IR_DRONE_001
        ir_mat = ir_vid.parent / f"{video_name}_LABELS.mat"
        vis_mat_name = vis_vid.stem + "_LABELS.mat"
        vis_mat = vis_vid.parent / vis_mat_name

        # Determine category
        is_drone = "_DRONE_" in video_name
        cat = video_name.split("_")[1].lower()  # drone/airplane/bird/helicopter

        n = process_video_pair(
            ir_vid, vis_vid, ir_mat, vis_mat,
            ir_img_dir, ir_lbl_dir, vis_img_dir, vis_lbl_dir,
            video_name, args.sample_every, is_drone=is_drone
        )
        total_frames += n
        stats[cat] = stats.get(cat, 0) + n

        if (idx + 1) % 20 == 0 or (idx + 1) == len(pairs):
            elapsed = time.time() - t_start
            vps = (idx + 1) / elapsed if elapsed > 0 else 0
            eta = (len(pairs) - idx - 1) / vps if vps > 0 else 0
            print(f"  [{idx+1}/{len(pairs)}] {total_frames} frames, "
                  f"{vps:.1f} vid/s, ETA {eta/60:.1f}min")

    # Summary
    print(f"\nDone. {total_frames} paired frames from {len(pairs)} videos")
    for cat, n in sorted(stats.items()):
        print(f"  {cat}: {n} frames")

    # Save metadata
    meta = {
        "total_frames": total_frames,
        "total_videos": len(pairs),
        "sample_every": args.sample_every,
        "stats": stats,
    }
    with open(output / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nOutput: {output}")


if __name__ == "__main__":
    main()
