"""
Download YouTube IR drone videos, extract frames at 1fps, auto-label with YOLO model.
Output: G:\drone\youtube_videos\ in YOLO format.
"""
import subprocess
import sys
import shutil
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUTPUT_ROOT = Path(r"G:\drone\youtube_videos")
MODEL_PATH = (
    r"C:\Users\User\Desktop\UNISA projects\Drone detection\es proj 3 thesis workspace"
    r"\ES_Drone_Detection\IR_FT_dsetV8_aug1_noclahe_200ep_s0"
    r"\IR_FT_dsetV8_aug1_noclahe_200ep_s0\weights\best.pt"
)
FPS = 1  # frames per second to extract
CONF_THRESH = 0.3  # auto-label confidence threshold

# (url, label, [(start_sec, end_sec), ...])   None end = till end of video
VIDEOS = [
    (
        "https://www.youtube.com/watch?v=CG8WoQdJf4Q",
        "CG8WoQdJf4Q",
        [(55, 82), (97, 117)],
    ),
    (
        "https://www.youtube.com/watch?v=JnbmIxe4TYs",
        "JnbmIxe4TYs",
        [(8, None)],
    ),
    (
        "https://www.youtube.com/watch?v=RYi0BrXBoHk",
        "RYi0BrXBoHk",
        [(6, 73)],
    ),
    (
        "https://www.youtube.com/watch?v=Uc_PVOzs4G0",
        "Uc_PVOzs4G0",
        [(0, None)],
    ),
    (
        "https://www.youtube.com/watch?v=v0c-R9qjUCw",
        "v0c-R9qjUCw",
        [(0, None)],
    ),
    (
        "https://www.youtube.com/watch?v=OkyqJ3Gckf0",
        "OkyqJ3Gckf0",
        [(0, None)],
    ),
    (
        "https://www.youtube.com/watch?v=QpMU2rmZLao",
        "QpMU2rmZLao",
        [(0, None)],
    ),
    (
        "https://www.youtube.com/watch?v=loJkpX-YrE8",
        "loJkpX-YrE8",
        [(0, None)],
    ),
]


def ensure_tool(name):
    """Check if a CLI tool is available."""
    if shutil.which(name) is None:
        print(f"ERROR: '{name}' not found. Install it first.")
        if name == "yt-dlp":
            print("  pip install yt-dlp")
        elif name == "ffmpeg":
            print("  Download from https://ffmpeg.org/download.html")
        sys.exit(1)


def download_video(url, video_id, tmp_dir):
    """Download video with yt-dlp. Returns path to downloaded file."""
    out_path = tmp_dir / f"{video_id}.mp4"
    if out_path.exists():
        print(f"  [SKIP] Already downloaded: {out_path.name}")
        return out_path

    print(f"  Downloading: {url}")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", str(out_path),
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [ERROR] yt-dlp failed: {result.stderr[:200]}")
        return None
    return out_path


def extract_frames(video_path, video_id, segments, images_dir, fps=1):
    """Extract frames from video at given fps for specified time segments."""
    total_frames = 0

    for seg_idx, (start, end) in enumerate(segments):
        # Build ffmpeg command
        cmd = ["ffmpeg", "-y"]

        # Input seeking (fast)
        if start > 0:
            cmd += ["-ss", str(start)]
        cmd += ["-i", str(video_path)]

        # Duration
        if end is not None:
            duration = end - start
            cmd += ["-t", str(duration)]

        # Output: 1fps, numbered frames
        prefix = f"{video_id}_s{seg_idx}"
        out_pattern = str(images_dir / f"{prefix}_%04d.jpg")

        cmd += [
            "-vf", f"fps={fps}",
            "-q:v", "2",  # high quality JPEG
            out_pattern,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  [ERROR] ffmpeg failed for segment {seg_idx}: {result.stderr[:200]}")
            continue

        # Count extracted frames
        extracted = list(images_dir.glob(f"{prefix}_*.jpg"))
        total_frames += len(extracted)
        seg_str = f"{start}s→{end}s" if end else f"{start}s→end"
        print(f"    Segment {seg_idx} ({seg_str}): {len(extracted)} frames")

    return total_frames


def auto_label(images_dir, labels_dir, model_path, conf=0.3):
    """Run YOLO model on all images and create YOLO-format label files."""
    from ultralytics import YOLO

    model = YOLO(model_path)
    images = sorted(images_dir.glob("*.jpg"))
    if not images:
        print("  No images to label!")
        return 0, 0

    print(f"  Auto-labeling {len(images)} images (conf={conf})...")

    labeled = 0
    empty = 0

    for i, img_path in enumerate(images):
        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(images)}]")

        results = model(str(img_path), conf=conf, verbose=False, device="0")

        label_lines = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            for box in r.boxes:
                xywhn = box.xywhn[0].cpu().numpy()
                cls = int(box.cls[0].cpu())
                label_lines.append(
                    f"{cls} {xywhn[0]:.6f} {xywhn[1]:.6f} {xywhn[2]:.6f} {xywhn[3]:.6f}"
                )

        lbl_path = labels_dir / f"{img_path.stem}.txt"
        lbl_path.write_text("\n".join(label_lines))

        if label_lines:
            labeled += 1
        else:
            empty += 1

    return labeled, empty


def create_dataset_yaml(output_root, video_urls):
    """Create dataset.yaml with source video links."""
    yaml_content = f"""# YouTube IR Drone Videos Dataset
# Auto-generated from YouTube thermal/IR drone footage
# Frames extracted at {FPS} fps, auto-labeled with YOLO model
#
# Source videos:
"""
    for url, vid_id, segments in video_urls:
        seg_strs = []
        for s, e in segments:
            seg_strs.append(f"{s}s-{e}s" if e else f"{s}s-end")
        yaml_content += f"#   {url}  [{', '.join(seg_strs)}]\n"

    yaml_content += f"""
path: {output_root}
train: train/images
val: val/images
test: test/images
nc: 1
names: ['drone']
"""
    yaml_path = output_root / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"  Saved: {yaml_path}")


def split_dataset(images_dir, labels_dir, output_root, train_ratio=0.8, val_ratio=0.1):
    """Split images into train/val/test."""
    import random
    random.seed(42)

    images = sorted(images_dir.glob("*.jpg"))
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split_name, split_images in splits.items():
        img_out = output_root / split_name / "images"
        lbl_out = output_root / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img in split_images:
            shutil.copy2(str(img), str(img_out / img.name))
            lbl = labels_dir / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy2(str(lbl), str(lbl_out / lbl.name))
            else:
                (lbl_out / f"{img.stem}.txt").write_text("")

        print(f"  {split_name}: {len(split_images)} images")


def main():
    ensure_tool("yt-dlp")
    ensure_tool("ffmpeg")

    # Temp working directories
    tmp_dir = OUTPUT_ROOT / "_tmp"
    tmp_images = OUTPUT_ROOT / "_tmp_images"
    tmp_labels = OUTPUT_ROOT / "_tmp_labels"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_images.mkdir(parents=True, exist_ok=True)
    tmp_labels.mkdir(parents=True, exist_ok=True)

    total_frames = 0

    # Step 1: Download and extract frames
    print("=" * 60)
    print("  STEP 1: Download videos & extract frames")
    print("=" * 60)

    for url, vid_id, segments in VIDEOS:
        print(f"\n[{vid_id}]")

        # Download
        video_path = download_video(url, vid_id, tmp_dir)
        if video_path is None:
            continue

        # Extract frames
        n = extract_frames(video_path, vid_id, segments, tmp_images, fps=FPS)
        total_frames += n
        print(f"  Total: {n} frames extracted")

    print(f"\n  TOTAL FRAMES: {total_frames}")

    # Step 2: Auto-label
    print("\n" + "=" * 60)
    print("  STEP 2: Auto-label with YOLO model")
    print("=" * 60)

    labeled, empty = auto_label(tmp_images, tmp_labels, MODEL_PATH, conf=CONF_THRESH)
    print(f"  Labeled: {labeled} images with detections")
    print(f"  Empty:   {empty} images (no detections)")

    # Step 3: Split into train/val/test
    print("\n" + "=" * 60)
    print("  STEP 3: Split into train/val/test (80/10/10)")
    print("=" * 60)

    split_dataset(tmp_images, tmp_labels, OUTPUT_ROOT)

    # Step 4: Create dataset.yaml
    create_dataset_yaml(OUTPUT_ROOT, VIDEOS)

    # Cleanup temp dirs
    print("\nCleaning up temp files...")
    shutil.rmtree(str(tmp_images), ignore_errors=True)
    shutil.rmtree(str(tmp_labels), ignore_errors=True)
    # Keep downloaded videos in _tmp in case re-run is needed

    print("\n" + "=" * 60)
    print(f"  DONE! Dataset at: {OUTPUT_ROOT}")
    print(f"  Total frames: {total_frames}")
    print(f"  Review labels in Label Reviewer before use!")
    print("=" * 60)


if __name__ == "__main__":
    main()
