"""
Convert paired image sequences (RGB + IR folders) into synchronized MP4 video pairs.

Usage:
    python convert_to_video.py \
        --rgb-dir G:\\drone\\Anti-UAV-RGBT\\yolo_test\\RGB\\images \
        --ir-dir  G:\\drone\\Anti-UAV-RGBT\\yolo_test\\IR\\images \
        --output-dir ./demo_videos \
        --fps 30

Groups images by sequence, strips modality suffixes (_visible, _infrared)
for pairing, sorts by frame number, writes one RGB.mp4 + one IR.mp4 per sequence.
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Regex: capture sequence ID (everything before _fNNNNNN or _frameNNNN or trailing _NNNN+)
SEQ_RE = re.compile(r"^(.+?)(?:_f\d+|_frame\d+)$", re.IGNORECASE)
# Modality suffixes to strip for pairing
MOD_RE = re.compile(r"_(visible|infrared)", re.IGNORECASE)
# Frame number extractor
FRAME_RE = re.compile(r"_f(\d+)$|_frame(\d+)$", re.IGNORECASE)


def strip_modality(stem: str) -> str:
    return MOD_RE.sub("", stem)


def extract_seq_id(stem: str) -> str:
    clean = strip_modality(stem)
    m = SEQ_RE.match(clean)
    return m.group(1) if m else clean


def extract_frame_num(stem: str) -> int:
    clean = strip_modality(stem)
    m = FRAME_RE.search(clean)
    if m:
        return int(m.group(1) or m.group(2))
    # Fallback: last group of digits
    nums = re.findall(r"\d+", clean)
    return int(nums[-1]) if nums else 0


def discover_images(img_dir: Path):
    """Returns dict: seq_id -> list of (frame_num, stem, path) sorted by frame_num."""
    seqs = defaultdict(list)
    for f in sorted(img_dir.iterdir()):
        if f.suffix.lower() not in IMG_EXTS:
            continue
        stem = f.stem
        seq_id = extract_seq_id(stem)
        frame_num = extract_frame_num(stem)
        seqs[seq_id].append((frame_num, stem, f))
    # Sort each sequence by frame number
    for seq_id in seqs:
        seqs[seq_id].sort(key=lambda x: x[0])
    return dict(seqs)


def pair_sequences(rgb_seqs, ir_seqs):
    """Match RGB and IR sequences by stripped sequence ID."""
    # Build maps: stripped_seq_id -> original_seq_id
    rgb_map = {}
    for seq_id in rgb_seqs:
        stripped = strip_modality(seq_id)
        rgb_map[stripped] = seq_id

    ir_map = {}
    for seq_id in ir_seqs:
        stripped = strip_modality(seq_id)
        ir_map[stripped] = seq_id

    common = sorted(set(rgb_map.keys()) & set(ir_map.keys()))
    pairs = []
    for stripped in common:
        pairs.append((stripped, rgb_map[stripped], ir_map[stripped]))
    return pairs


def write_video(frames_paths, output_path: Path, fps: int):
    """Write a list of image paths into an MP4 video."""
    if not frames_paths:
        return 0

    # Read first frame to get dimensions
    first = cv2.imread(str(frames_paths[0]))
    if first is None:
        print(f"  [WARN] Cannot read {frames_paths[0]}, skipping")
        return 0
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    writer.write(first)
    count = 1

    for p in frames_paths[1:]:
        img = cv2.imread(str(p))
        if img is None:
            continue
        # Resize if needed (some frames might differ slightly)
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        writer.write(img)
        count += 1

    writer.release()
    return count


def main():
    parser = argparse.ArgumentParser(description="Convert paired image sequences to video")
    parser.add_argument("--rgb-dir", required=True, help="RGB images directory")
    parser.add_argument("--ir-dir", required=True, help="IR images directory")
    parser.add_argument("--output-dir", default="./demo_videos", help="Output directory")
    parser.add_argument("--fps", type=int, default=30, help="Output video FPS")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Max frames per sequence (0 = all)")
    args = parser.parse_args()

    rgb_dir = Path(args.rgb_dir)
    ir_dir = Path(args.ir_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not rgb_dir.is_dir():
        print(f"ERROR: RGB dir not found: {rgb_dir}")
        sys.exit(1)
    if not ir_dir.is_dir():
        print(f"ERROR: IR dir not found: {ir_dir}")
        sys.exit(1)

    print(f"RGB dir: {rgb_dir}")
    print(f"IR dir:  {ir_dir}")
    print(f"Output:  {out_dir}")
    print(f"FPS:     {args.fps}")
    print()

    # Discover
    print("Scanning RGB images...")
    rgb_seqs = discover_images(rgb_dir)
    print(f"  Found {len(rgb_seqs)} sequences, {sum(len(v) for v in rgb_seqs.values())} frames")

    print("Scanning IR images...")
    ir_seqs = discover_images(ir_dir)
    print(f"  Found {len(ir_seqs)} sequences, {sum(len(v) for v in ir_seqs.values())} frames")

    # Pair
    pairs = pair_sequences(rgb_seqs, ir_seqs)
    print(f"\nMatched {len(pairs)} sequence pairs")

    if not pairs:
        print("No matching sequences found. Check directory structure and naming.")
        sys.exit(1)

    # Write videos
    total_rgb_frames = 0
    total_ir_frames = 0
    for stripped_id, rgb_seq_id, ir_seq_id in pairs:
        rgb_frames = rgb_seqs[rgb_seq_id]
        ir_frames = ir_seqs[ir_seq_id]

        if args.max_frames > 0:
            rgb_frames = rgb_frames[:args.max_frames]
            ir_frames = ir_frames[:args.max_frames]

        # Use min frame count for sync
        n = min(len(rgb_frames), len(ir_frames))
        rgb_frames = rgb_frames[:n]
        ir_frames = ir_frames[:n]

        # Clean filename
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', stripped_id)[:60]
        rgb_out = out_dir / f"{safe_name}_RGB.mp4"
        ir_out = out_dir / f"{safe_name}_IR.mp4"

        print(f"\n  {stripped_id}: {n} paired frames")
        print(f"    -> {rgb_out.name}")
        rc = write_video([f[2] for f in rgb_frames], rgb_out, args.fps)
        print(f"    -> {ir_out.name}")
        ic = write_video([f[2] for f in ir_frames], ir_out, args.fps)

        total_rgb_frames += rc
        total_ir_frames += ic

    print(f"\nDone. {len(pairs)} video pairs, "
          f"{total_rgb_frames} RGB frames, {total_ir_frames} IR frames")


if __name__ == "__main__":
    main()
