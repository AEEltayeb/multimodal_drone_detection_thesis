"""
Crop-and-zoom augmentation for dsetV9b1.
Targets medium-large drones (1,600-10,000 px²) only.
Creates zoomed crops centered on drones to shift mean bbox size up.
"""
import argparse, os, random, shutil
from pathlib import Path
from PIL import Image

def compute_pixel_area(w_norm, h_norm, img_w=640, img_h=512):
    return (w_norm * img_w) * (h_norm * img_h)

def crop_and_zoom(img_path, label_path, out_img_dir, out_lbl_dir,
                  min_area=1600, max_area=10000,
                  zoom_factors=[2.0, 3.0],
                  img_w=640, img_h=512, prefix="czoom"):
    """Create zoomed crops for qualifying drones."""
    with open(label_path, encoding='utf-8', errors='ignore') as f:
        lines = [l.strip() for l in f if l.strip()]
    
    if not lines:
        return 0
    
    # Parse bboxes and find medium-large ones
    targets = []
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = parts[0]
        cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        px_area = compute_pixel_area(w, h, img_w, img_h)
        if min_area <= px_area <= max_area:
            targets.append((cls, cx, cy, w, h, px_area))
    
    if not targets:
        return 0
    
    # Load image
    try:
        img = Image.open(img_path)
        actual_w, actual_h = img.size
    except Exception:
        return 0
    
    count = 0
    stem = img_path.stem
    
    for cls, cx, cy, bw, bh, px_area in targets:
        for zoom in zoom_factors:
            # Crop size = image_size / zoom
            crop_w = actual_w / zoom
            crop_h = actual_h / zoom
            
            # Center crop on the drone
            drone_cx_px = cx * actual_w
            drone_cy_px = cy * actual_h
            
            # Crop bounds (clamp to image edges)
            x1 = max(0, drone_cx_px - crop_w / 2)
            y1 = max(0, drone_cy_px - crop_h / 2)
            x2 = min(actual_w, x1 + crop_w)
            y2 = min(actual_h, y1 + crop_h)
            
            # Adjust if crop hit edge
            if x2 - x1 < crop_w:
                x1 = max(0, x2 - crop_w)
            if y2 - y1 < crop_h:
                y1 = max(0, y2 - crop_h)
            
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            if x2 - x1 < 32 or y2 - y1 < 32:
                continue
            
            # Crop and resize to original dimensions
            cropped = img.crop((x1, y1, x2, y2))
            cropped = cropped.resize((actual_w, actual_h), Image.LANCZOS)
            
            # Recalculate ALL bboxes relative to crop
            new_lines = []
            for line2 in lines:
                p2 = line2.split()
                if len(p2) < 5:
                    continue
                c2 = p2[0]
                bx, by, bww, bhh = float(p2[1]), float(p2[2]), float(p2[3]), float(p2[4])
                
                # Convert to pixel coords
                bx_px = bx * actual_w
                by_px = by * actual_h
                bw_px = bww * actual_w
                bh_px = bhh * actual_h
                
                # Bbox edges in original image
                bx1 = bx_px - bw_px / 2
                by1 = by_px - bh_px / 2
                bx2 = bx_px + bw_px / 2
                by2 = by_px + bh_px / 2
                
                # Clip to crop region
                bx1_c = max(x1, bx1) - x1
                by1_c = max(y1, by1) - y1
                bx2_c = min(x2, bx2) - x1
                by2_c = min(y2, by2) - y1
                
                # Skip if bbox is outside crop or too small
                if bx2_c <= bx1_c or by2_c <= by1_c:
                    continue
                
                # Check if enough of the bbox is visible (>50%)
                visible_area = (bx2_c - bx1_c) * (by2_c - by1_c)
                original_area = bw_px * bh_px
                if original_area > 0 and visible_area / original_area < 0.5:
                    continue
                
                # Convert to normalized coords relative to crop
                crop_w_actual = x2 - x1
                crop_h_actual = y2 - y1
                new_cx = (bx1_c + bx2_c) / 2 / crop_w_actual
                new_cy = (by1_c + by2_c) / 2 / crop_h_actual
                new_w = (bx2_c - bx1_c) / crop_w_actual
                new_h = (by2_c - by1_c) / crop_h_actual
                
                # Clamp to [0, 1]
                new_cx = max(0, min(1, new_cx))
                new_cy = max(0, min(1, new_cy))
                new_w = max(0.001, min(1, new_w))
                new_h = max(0.001, min(1, new_h))
                
                new_lines.append(f"{c2} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}")
            
            if not new_lines:
                continue
            
            # Save
            out_name = f"{prefix}_{stem}_z{zoom:.0f}"
            cropped.save(str(out_img_dir / (out_name + img_path.suffix)), quality=95)
            (out_lbl_dir / (out_name + ".txt")).write_text("\n".join(new_lines) + "\n")
            count += 1
    
    return count


def main():
    parser = argparse.ArgumentParser(description="Crop-and-zoom augmentation")
    parser.add_argument("--dataset", required=True, help="Path to dataset (e.g., G:\\drone\\IR_dsetV9b1)")
    parser.add_argument("--min-area", type=float, default=1600, help="Min bbox area to target (px²)")
    parser.add_argument("--max-area", type=float, default=10000, help="Max bbox area to target (px²)")
    parser.add_argument("--zoom-factors", type=float, nargs="+", default=[2.0, 3.0])
    parser.add_argument("--max-crops", type=int, default=10000, help="Max crops to generate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    dataset = Path(args.dataset)
    lbl_dir = dataset / "train" / "labels"
    img_dir = dataset / "train" / "images"
    
    print(f"Crop-and-zoom augmentation")
    print(f"  Dataset: {dataset}")
    print(f"  Target range: {args.min_area}-{args.max_area} px²")
    print(f"  Zoom factors: {args.zoom_factors}")
    print(f"  Max crops: {args.max_crops}")
    
    # Find qualifying labels
    qualifying = []
    for lf in sorted(lbl_dir.glob("*.txt")):
        with open(lf, encoding='utf-8', errors='ignore') as f:
            lines = [l.strip() for l in f if l.strip()]
        for line in lines:
            p = line.split()
            if len(p) < 5:
                continue
            area = compute_pixel_area(float(p[3]), float(p[4]))
            if args.min_area <= area <= args.max_area:
                qualifying.append(lf)
                break
    
    print(f"  Qualifying images: {len(qualifying)}")
    
    # Shuffle and limit
    rng = random.Random(args.seed)
    rng.shuffle(qualifying)
    max_imgs = args.max_crops // len(args.zoom_factors)
    qualifying = qualifying[:max_imgs]
    
    print(f"  Processing: {len(qualifying)} images × {len(args.zoom_factors)} zooms")
    
    total = 0
    for i, lf in enumerate(qualifying):
        if (i + 1) % 500 == 0:
            print(f"    [{i+1}/{len(qualifying)}] {total} crops so far")
        
        stem = lf.stem
        img_path = None
        for ext in [".png", ".jpg", ".jpeg"]:
            candidate = img_dir / (stem + ext)
            if candidate.exists():
                img_path = candidate
                break
        
        if img_path is None:
            continue
        
        n = crop_and_zoom(
            img_path, lf, img_dir, lbl_dir,
            min_area=args.min_area, max_area=args.max_area,
            zoom_factors=args.zoom_factors,
        )
        total += n
    
    print(f"\n  Done! Added {total} crop-zoom images to train split.")


if __name__ == "__main__":
    main()
