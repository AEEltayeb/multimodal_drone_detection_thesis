"""
convert_to_video.py — Convert paired image sequences to synchronized RGB+IR video pairs.

Groups images by sequence, strips modality suffixes for pairing,
and writes one RGB.mp4 + one IR.mp4 per sequence.

Usage:
    python convert_to_video.py --rgb-dir G:\\drone\\svanstrom_paired\\RGB\\images --ir-dir G:\\drone\\svanstrom_paired\\IR\\images --output-dir ./demo_videos
    python convert_to_video.py --rgb-dir G:\\drone\\Anti-UAV-RGBT_yolo_converted\\test\\RGB\\images --ir-dir G:\\drone\\Anti-UAV-RGBT_yolo_converted\\test\\IR\\images --output-dir ./demo_videos --fps 25
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import cv2

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Regex: extract sequence ID and frame number from stem
SEQ_RE = re.compile(
    r"^(.+?)_(?:f(\d+)|frame(\d+)|(\d{4,}))(?:_visible|_infrared|_ir|_rgb)?$",
    re.IGNORECASE,
)


def strip_modality_suffix(stem):
    return re.sub(r"_(visible|infrared|ir|rgb)$", "", stem, flags=re.IGNORECASE)


def extract_seq_and_frame(stem):
    """Extract (sequence_id, frame_number) from a stem like IR_DRONE_001_f000123_visible."""
    m = SEQ_RE.match(stem)
    if m:
        seq = m.group(1).rstrip("_")
        frame_num = int(m.group(2) or m.group(3) or m.group(4))
        return seq, frame_num
    return stem, 0


def list_images(img_dir):
    """List all image files sorted by name."""
    img_dir = Path(img_dir)
    return sorted([f for f in img_dir.iterdir() if f.suffix.lower() in IMG_EXTS])


def group_by_sequence(images):
    """Group images by sequence ID, with frame numbers for sorting."""
    groups = defaultdict(list)
    for path in images:
        base = strip_modality_suffix(path.stem)
        seq, frame_num = extract_seq_and_frame(path.stem)
        groups[seq].append((frame_num, base, path))
    # Sort each sequence by frame number
    for seq in groups:
        groups[seq].sort(key=lambda x: x[0])
    return groups


def pair_sequences(rgb_groups, ir_groups):
    """Find sequences that exist in both RGB and IR."""
    common = sorted(set(rgb_groups.keys()) & set(ir_groups.keys()))
    paired = []
    for seq in common:
        rgb_frames = rgb_groups[seq]
        ir_frames = ir_groups[seq]

        # Build base_stem -> path mapping for IR
        ir_map = {base: path for _, base, path in ir_frames}

        # Pair by base_stem
        pairs = []
        for _, base, rgb_path in rgb_frames:
            ir_path = ir_map.get(base)
            if ir_path is not None:
                pairs.append((rgb_path, ir_path))

        if pairs:
            paired.append((seq, pairs))

    return paired


def write_video(frames_paths, output_path, fps, is_ir=False):
    """Write a list of image paths to a video file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read first frame to get dimensions
    first = cv2.imread(str(frames_paths[0]))
    if first is None:
        print(f"    [ERROR] Cannot read {frames_paths[0]}")
        return False
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    for path in frames_paths:
        img = cv2.imread(str(path))
        if img is None:
            continue
        # If IR and single channel, convert to 3-channel for video
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        writer.write(img)

    writer.release()
    return True


def main():
    parser = argparse.ArgumentParser(description="Convert paired image sequences to videos")
    parser.add_argument("--rgb-dir", required=True, help="RGB images directory")
    parser.add_argument("--ir-dir", required=True, help="IR images directory")
    parser.add_argument("--output-dir", default="./demo_videos", help="Output directory")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS (default: 30)")
    parser.add_argument("--min-frames", type=int, default=30,
                        help="Skip sequences shorter than this (default: 30)")
    parser.add_argument("--max-sequences", type=int, default=0,
                        help="Limit number of sequences to convert (0=all)")
    parser.add_argument("--sequences", nargs="*", default=None,
                        help="Specific sequence names to convert (e.g., IR_DRONE_001)")
    args = parser.parse_args()

    rgb_dir = Path(args.rgb_dir)
    ir_dir = Path(args.ir_dir)
    out_dir = Path(args.output_dir)

    print("=" * 70)
    print("Dataset to Video Converter")
    print("=" * 70)
    print(f"  RGB dir: {rgb_dir}")
    print(f"  IR dir:  {ir_dir}")
    print(f"  Output:  {out_dir}")
    print(f"  FPS:     {args.fps}")
    print()

    # List and group
    print("Scanning RGB images...", end="", flush=True)
    rgb_images = list_images(rgb_dir)
    print(f" {len(rgb_images):,} found")

    print("Scanning IR images...", end="", flush=True)
    ir_images = list_images(ir_dir)
    print(f" {len(ir_images):,} found")

    print("Grouping by sequence...", end="", flush=True)
    rgb_groups = group_by_sequence(rgb_images)
    ir_groups = group_by_sequence(ir_images)
    print(f" RGB: {len(rgb_groups)} seqs, IR: {len(ir_groups)} seqs")

    # Pair
    paired = pair_sequences(rgb_groups, ir_groups)
    print(f"Paired sequences: {len(paired)}")

    # Filter
    if args.sequences:
        target = set(args.sequences)
        paired = [(seq, pairs) for seq, pairs in paired if seq in target]
        print(f"Filtered to {len(paired)} specified sequences")

    paired = [(seq, pairs) for seq, pairs in paired if len(pairs) >= args.min_frames]
    print(f"After min-frames filter ({args.min_frames}): {len(paired)} sequences")

    if args.max_sequences > 0:
        paired = paired[:args.max_sequences]
        print(f"Limited to {len(paired)} sequences")

    if not paired:
        print("\nNo sequences to convert.")
        return

    # Convert
    print(f"\nConverting {len(paired)} sequences...")
    out_dir.mkdir(parents=True, exist_ok=True)
    total_frames = 0

    for i, (seq, pairs) in enumerate(paired):
        n = len(pairs)
        rgb_paths = [p[0] for p in pairs]
        ir_paths = [p[1] for p in pairs]

        # Sanitize sequence name for filename
        safe_name = re.sub(r'[<>:"/\\|?*]', '_', seq)

        rgb_out = out_dir / f"{safe_name}_rgb.mp4"
        ir_out = out_dir / f"{safe_name}_ir.mp4"

        duration = n / args.fps
        print(f"\n  [{i+1}/{len(paired)}] {seq}: {n} frames ({duration:.1f}s at {args.fps}fps)")

        ok_rgb = write_video(rgb_paths, rgb_out, args.fps)
        ok_ir = write_video(ir_paths, ir_out, args.fps, is_ir=True)

        if ok_rgb and ok_ir:
            print(f"    -> {rgb_out.name}")
            print(f"    -> {ir_out.name}")
            total_frames += n
        else:
            print(f"    [ERROR] Failed")

    print(f"\n{'='*70}")
    print(f"Done. {len(paired)} sequences, {total_frames:,} total frames")
    print(f"Output: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
