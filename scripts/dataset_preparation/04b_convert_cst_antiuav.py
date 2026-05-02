import os
import shutil
import cv2
from pathlib import Path
from tqdm import tqdm

def convert_to_yolo(bbox_str, img_w, img_h):
    """
    Convert AntiUAV format (x, y, w, h) to YOLO format (cx_norm, cy_norm, w_norm, h_norm).
    x, y is the top-left corner in pixels.
    """
    parts = bbox_str.strip().split(',')
    if len(parts) != 4:
        return None
        
    x, y, w, h = map(float, parts)
    
    # If width or height is 0, invalid box
    if w <= 0 or h <= 0:
        return None
        
    # Calculate center
    cx = x + (w / 2.0)
    cy = y + (h / 2.0)
    
    # Normalize by image dimensions
    cx_norm = cx / img_w
    cy_norm = cy / img_h
    w_norm = w / img_w
    h_norm = h / img_h
    
    # Ensure they are bounded [0, 1]
    cx_norm = max(0.0, min(1.0, cx_norm))
    cy_norm = max(0.0, min(1.0, cy_norm))
    w_norm = max(0.0, min(1.0, w_norm))
    h_norm = max(0.0, min(1.0, h_norm))
    
    # Class ID is 0 for Drone
    return f"0 {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}"

def process_sequence(seq_dir, target_images_dir, target_labels_dir):
    """Process a single sequence folder (e.g. building_1)"""
    seq_name = seq_dir.name
    
    gt_file = seq_dir / "gt.txt"
    exist_file = seq_dir / "exist.txt"
    
    if not gt_file.exists() or not exist_file.exists():
        print(f"Skipping {seq_name}: missing gt.txt or exist.txt")
        return
        
    with open(gt_file, 'r') as f:
        gt_lines = [line.strip() for line in f.readlines()]
        
    with open(exist_file, 'r') as f:
        exist_lines = [line.strip() for line in f.readlines()]
        
    # Get all jpgs
    images = sorted([f for f in seq_dir.glob("*.jpg")])
    
    if not images:
        print(f"Skipping {seq_name}: no images found")
        return
        
    # Read the first image to get dimensions
    # Assuming all images in a sequence have the same dimensions
    first_img = cv2.imread(str(images[0]))
    if first_img is None:
        print(f"Error reading first image in {seq_name}")
        return
        
    img_h, img_w = first_img.shape[:2]
    
    # Make sure counts match roughly
    max_idx = min(len(images), len(gt_lines), len(exist_lines))
    
    for i in range(max_idx):
        img_path = images[i]
        
        # Format the new filename: seq_name_imgname.jpg
        # e.g., building_1_000001.jpg
        new_img_name = f"{seq_name}_{img_path.name}"
        new_label_name = f"{seq_name}_{img_path.stem}.txt"
        
        target_img_path = target_images_dir / new_img_name
        target_label_path = target_labels_dir / new_label_name
        
        # Copy image
        shutil.copy2(img_path, target_img_path)
        
        # Check if object exists and write label
        exist_flag = exist_lines[i]
        bbox_str = gt_lines[i]
        
        if exist_flag == '1':
            yolo_str = convert_to_yolo(bbox_str, img_w, img_h)
            if yolo_str:
                with open(target_label_path, 'w') as f:
                    f.write(yolo_str + '\n')
            else:
                # Invalid box but exist=1: Treat as negative (empty file)
                target_label_path.touch()
        else:
            # exist=0: Meaningful negative (empty file)
            target_label_path.touch()

def main():
    source_root = Path(r"G:\drone\CST-AntiUAV\CST-AntiUAV\CST-AntiUAV\CST-AntiUAV")
    target_root = Path(r"G:\drone\CST-AntiUAV_YOLO")
    
    print(f"Source: {source_root}")
    print(f"Target: {target_root}")
    
    splits = ["train", "val", "test"]
    
    for split in splits:
        source_split_dir = source_root / split
        if not source_split_dir.exists():
            print(f"Split {split} not found at {source_split_dir}")
            continue
            
        target_split_img_dir = target_root / split / "images"
        target_split_label_dir = target_root / split / "labels"
        
        # Also backup original structure (just nested up)
        backup_split_dir = target_root / "backup_original" / split
        shutil.copytree(source_split_dir, backup_split_dir, dirs_exist_ok=True)
        
        os.makedirs(target_split_img_dir, exist_ok=True)
        os.makedirs(target_split_label_dir, exist_ok=True)
        
        sequences = [d for d in source_split_dir.iterdir() if d.is_dir()]
        print(f"\nProcessing {split} split ({len(sequences)} sequences)")
        
        for seq_dir in tqdm(sequences):
            process_sequence(seq_dir, target_split_img_dir, target_split_label_dir)
            
    print(f"\nDone! YOLO structure saved to {target_root}")
    print(f"Originals backed up to {target_root / 'backup_original'}")

if __name__ == "__main__":
    main()
