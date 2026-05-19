"""
Scan RGB and IR datasets and report comprehensive distribution info.
"""
import os
import json
from pathlib import Path
from collections import defaultdict

def count_images(folder):
    """Count image files in a folder."""
    if not os.path.isdir(folder):
        return 0
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
    return sum(1 for f in os.listdir(folder) if Path(f).suffix.lower() in exts)

def analyse_labels(folder):
    """Analyse YOLO-format label files in a folder.
    Returns: total_files, empty_files (negatives), files_with_labels (positives),
             class_counts dict, total_boxes
    """
    if not os.path.isdir(folder):
        return 0, 0, 0, {}, 0
    total = 0
    empty = 0
    with_labels = 0
    class_counts = defaultdict(int)
    total_boxes = 0
    for f in os.listdir(folder):
        if not f.endswith('.txt'):
            continue
        total += 1
        fpath = os.path.join(folder, f)
        size = os.path.getsize(fpath)
        if size == 0:
            empty += 1
            continue
        with open(fpath, 'r') as fh:
            lines = [l.strip() for l in fh.readlines() if l.strip()]
        if len(lines) == 0:
            empty += 1
        else:
            with_labels += 1
            for line in lines:
                parts = line.split()
                if parts:
                    cls = parts[0]
                    class_counts[cls] += 1
                    total_boxes += 1
    return total, empty, with_labels, dict(class_counts), total_boxes

def extract_sources_from_filenames(folder, sample_size=None):
    """Try to extract data source prefixes from filenames."""
    if not os.path.isdir(folder):
        return {}
    prefixes = defaultdict(int)
    files = os.listdir(folder)
    for f in files:
        name = Path(f).stem
        # Common patterns: source_number, source-number, etc.
        # Try splitting on common delimiters
        for delim in ['_', '-']:
            if delim in name:
                prefix = name.split(delim)[0]
                prefixes[prefix] += 1
                break
        else:
            prefixes[name] += 1
    return dict(prefixes)

def scan_rgb_dataset(base_path):
    """Scan the RGB dataset at G:/drone/dataset/dataset"""
    print("=" * 70)
    print("RGB DATASET SCAN")
    print(f"Path: {base_path}")
    print("=" * 70)
    
    results = {}
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(base_path, 'images', split)
        lbl_dir = os.path.join(base_path, 'labels', split)
        
        n_images = count_images(img_dir)
        n_labels, n_empty, n_positive, class_counts, n_boxes = analyse_labels(lbl_dir)
        
        # Source analysis
        sources = extract_sources_from_filenames(img_dir)
        
        results[split] = {
            'images': n_images,
            'label_files': n_labels,
            'positives': n_positive,
            'negatives': n_empty,
            'class_counts': class_counts,
            'total_boxes': n_boxes,
            'sources': sources,
        }
        
        print(f"\n--- {split.upper()} ---")
        print(f"  Images:       {n_images}")
        print(f"  Label files:  {n_labels}")
        print(f"  Positives:    {n_positive} ({n_positive/max(n_labels,1)*100:.1f}%)")
        print(f"  Negatives:    {n_empty} ({n_empty/max(n_labels,1)*100:.1f}%)")
        print(f"  Total boxes:  {n_boxes}")
        print(f"  Classes:      {class_counts}")
        
        # Show top source prefixes
        sorted_sources = sorted(sources.items(), key=lambda x: -x[1])[:20]
        print(f"  Top source prefixes ({len(sources)} unique):")
        for prefix, count in sorted_sources:
            print(f"    {prefix}: {count}")
    
    total_imgs = sum(r['images'] for r in results.values())
    total_pos = sum(r['positives'] for r in results.values())
    total_neg = sum(r['negatives'] for r in results.values())
    total_labels = sum(r['label_files'] for r in results.values())
    
    print(f"\n--- TOTALS ---")
    print(f"  Total images:    {total_imgs}")
    print(f"  Total labels:    {total_labels}")
    print(f"  Total positives: {total_pos}")
    print(f"  Total negatives: {total_neg}")
    print(f"  Pos:Neg ratio:   {total_pos}:{total_neg} = {total_pos/max(total_neg,1):.2f}:1")
    
    print(f"\n--- SPLIT RATIOS ---")
    for split in ['train', 'val', 'test']:
        r = results[split]
        pct = r['images'] / max(total_imgs, 1) * 100
        print(f"  {split}: {r['images']} ({pct:.1f}%)")
    
    return results

def scan_ir_dataset(base_path):
    """Scan the IR dataset at G:/drone/ir_dset_final"""
    print("\n" + "=" * 70)
    print("IR DATASET SCAN")
    print(f"Path: {base_path}")
    print("=" * 70)
    
    results = {}
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(base_path, split, 'images')
        lbl_dir = os.path.join(base_path, split, 'labels')
        
        n_images = count_images(img_dir)
        n_labels, n_empty, n_positive, class_counts, n_boxes = analyse_labels(lbl_dir)
        
        # Source analysis
        sources = extract_sources_from_filenames(img_dir)
        
        results[split] = {
            'images': n_images,
            'label_files': n_labels,
            'positives': n_positive,
            'negatives': n_empty,
            'class_counts': class_counts,
            'total_boxes': n_boxes,
            'sources': sources,
        }
        
        print(f"\n--- {split.upper()} ---")
        print(f"  Images:       {n_images}")
        print(f"  Label files:  {n_labels}")
        print(f"  Positives:    {n_positive} ({n_positive/max(n_labels,1)*100:.1f}%)")
        print(f"  Negatives:    {n_empty} ({n_empty/max(n_labels,1)*100:.1f}%)")
        print(f"  Total boxes:  {n_boxes}")
        print(f"  Classes:      {class_counts}")
        
        sorted_sources = sorted(sources.items(), key=lambda x: -x[1])[:20]
        print(f"  Top source prefixes ({len(sources)} unique):")
        for prefix, count in sorted_sources:
            print(f"    {prefix}: {count}")
    
    total_imgs = sum(r['images'] for r in results.values())
    total_pos = sum(r['positives'] for r in results.values())
    total_neg = sum(r['negatives'] for r in results.values())
    total_labels = sum(r['label_files'] for r in results.values())
    
    print(f"\n--- TOTALS ---")
    print(f"  Total images:    {total_imgs}")
    print(f"  Total labels:    {total_labels}")
    print(f"  Total positives: {total_pos}")
    print(f"  Total negatives: {total_neg}")
    print(f"  Pos:Neg ratio:   {total_pos}:{total_neg} = {total_pos/max(total_neg,1):.2f}:1")
    
    print(f"\n--- SPLIT RATIOS ---")
    for split in ['train', 'val', 'test']:
        r = results[split]
        pct = r['images'] / max(total_imgs, 1) * 100
        print(f"  {split}: {r['images']} ({pct:.1f}%)")
    
    return results

if __name__ == '__main__':
    rgb_path = r"G:\drone\dataset\dataset"
    ir_path = r"G:\drone\ir_dset_final"
    
    print("Starting dataset scan... (this may take a while on external drives)")
    
    rgb_results = scan_rgb_dataset(rgb_path)
    ir_results = scan_ir_dataset(ir_path)
    
    # Save raw results as JSON
    output = {
        'rgb': rgb_results,
        'ir': ir_results,
    }
    
    out_path = os.path.join(os.path.dirname(__file__), '..', 'docs', 'dataset_distributions.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nRaw results saved to: {out_path}")
    
    print("\n\nDONE.")
