import argparse
import json
import random
from pathlib import Path

import cv2
import joblib
import numpy as np
from ultralytics import YOLO

# --- Paths (same as eval_six_configs.py) ---
CLASSIFIER_DIR = Path(__file__).resolve().parent
REPO = CLASSIFIER_DIR.parent
CLASSIFIER_PATH = CLASSIFIER_DIR / "runs" / "reliability" / "fusion" / "fusion_no_fn_model.joblib"
PATCH_RGB_PATH  = CLASSIFIER_DIR / "runs" / "patches" / "confuser_filter4_rgb.pt"
PATCH_IR_PATH   = CLASSIFIER_DIR / "runs" / "patches" / "confuser_filter4_ir.pt"

# --- Thresholds ---
PATCH_THRESHOLD = 0.70
RGB_CONF = 0.25
IR_CONF = 0.40
IOU_MATCH = 0.5

def draw_boxes(img, boxes, color, label_text, thickness=2):
    """Draw bounding boxes on image."""
    for box in boxes:
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if label_text:
            conf_str = f" {box[-1]:.2f}" if len(box) > 4 else ""
            text = f"{label_text}{conf_str}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["antiuav", "svanstrom"], default="antiuav")
    parser.add_argument("--num-frames", type=int, default=20, help="Number of frames to visualize")
    args = parser.parse_args()

    # Determine paths based on dataset
    if args.dataset == "antiuav":
        root = Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test")
        rgb_img_dir = root / "RGB" / "images"
        ir_img_dir = root / "IR" / "images"
    else:
        root = Path("G:/drone/svanstrom_paired")
        rgb_img_dir = root / "RGB" / "images"
        ir_img_dir = root / "IR" / "images"

    out_dir = CLASSIFIER_DIR / "runs" / f"visualize_{args.dataset}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    # Load settings to get weights
    with open(REPO / "ir_gui" / "fusion_settings.json") as f:
        settings = json.load(f)
    rgb_yolo = YOLO(settings["rgb_model"])
    ir_yolo = YOLO(settings["ir_model"])
    
    # We will just do YOLO visualization for simplicity to show the concept
    
    # Gather pairs
    rgb_imgs = sorted(list(rgb_img_dir.glob("*.jpg")))
    
    # Pick random subset
    random.seed(42)
    sample_imgs = random.sample(rgb_imgs, min(len(rgb_imgs), args.num_frames))
    
    print(f"Visualizing {len(sample_imgs)} frames from {args.dataset} to {out_dir}...")
    
    for i, rgb_path in enumerate(sample_imgs):
        stem = rgb_path.stem
        if "_visible" in stem:
            ir_stem = stem.replace("_visible", "_infrared")
        else:
            ir_stem = stem
            
        ir_path = ir_img_dir / f"{ir_stem}.jpg"
        if not ir_path.exists():
            continue
            
        rgb_img = cv2.imread(str(rgb_path))
        ir_img = cv2.imread(str(ir_path))
        if rgb_img is None or ir_img is None:
            continue
            
        # Run YOLO
        rgb_res = rgb_yolo.predict(rgb_img, conf=RGB_CONF, verbose=False)[0]
        ir_res = ir_yolo.predict(ir_img, conf=IR_CONF, verbose=False)[0]
        
        rgb_boxes = rgb_res.boxes.data.cpu().numpy().tolist() if len(rgb_res.boxes) > 0 else []
        ir_boxes = ir_res.boxes.data.cpu().numpy().tolist() if len(ir_res.boxes) > 0 else []
        
        # Draw RGB predictions
        rgb_vis = rgb_img.copy()
        draw_boxes(rgb_vis, rgb_boxes, (0, 0, 255), "RGB", thickness=2)
        cv2.putText(rgb_vis, f"RGB (T={RGB_CONF})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        # Draw IR predictions
        ir_vis = ir_img.copy()
        draw_boxes(ir_vis, ir_boxes, (0, 0, 255), "IR", thickness=2)
        cv2.putText(ir_vis, f"IR (T={IR_CONF})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        # Resize IR to match RGB height for concatenation
        h, w = rgb_vis.shape[:2]
        ih, iw = ir_vis.shape[:2]
        scale = h / ih
        ir_resized = cv2.resize(ir_vis, (int(iw * scale), h))
        
        # Combine
        combined = np.hstack([rgb_vis, ir_resized])
        
        out_path = out_dir / f"{i:03d}_{stem}.jpg"
        cv2.imwrite(str(out_path), combined)
        
    print(f"Done! Check the images in {out_dir}")

if __name__ == "__main__":
    main()
