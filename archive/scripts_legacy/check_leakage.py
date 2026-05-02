"""
Check for data leakage across train / val / test splits.

Reports:
  1. Exact filename collisions (stem match across splits)
  2. Same-video frame leakage (e.g. frame_001 in train, frame_002 in test)
     Uses heuristics: strips trailing _NNNNN / _frameNNNN suffixes to get a "video id"

Usage:
  python scripts/check_leakage.py --dataset-yaml G:/drone/IR_dset_final/dataset.yaml
  python scripts/check_leakage.py --train path/to/train/images --val path/to/val/images --test path/to/test/images
"""

import argparse
import os
import re
import yaml
from pathlib import Path
from collections import defaultdict


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Regex patterns to strip frame numbers and get a "video stem"
FRAME_SUFFIXES = [
    re.compile(r"_\d{3,6}$"),          # _001234
    re.compile(r"_frame_?\d+$", re.I), # _frame123, _Frame_0042
    re.compile(r"_f\d+$"),             # _f0042
]


def collect_stems(img_dir: Path) -> set[str]:
    """Return set of filename stems (no extension) from an image directory."""
    if not img_dir.exists():
        print(f"  [WARN] directory does not exist: {img_dir}")
        return set()
    stems = set()
    for f in img_dir.iterdir():
        if f.is_file() and f.suffix.lower() in IMG_EXTS:
            stems.add(f.stem)
    return stems


def get_video_id(stem: str) -> str:
    """Strip frame suffixes to get a rough 'video id'."""
    vid = stem
    # Strip czoom_ prefix (crop-zoom augmentation)
    if vid.startswith("czoom_"):
        vid = vid[6:]
    # Strip zoom level suffix like _z2, _z3
    vid = re.sub(r"_z\d+$", "", vid)
    for pat in FRAME_SUFFIXES:
        vid = pat.sub("", vid)
    return vid


def get_prefix(stem: str) -> str:
    """Extract a coarse dataset-level prefix (first 2 underscore tokens).

    Handles FLIR video hashes (hyphen-separated) and czoom_ augmentation.
    """
    s = stem
    if s.startswith("czoom_"):
        s = s[6:]
    # FLIR: flir_video-{HASH}-frame-... -> flir
    if s.startswith("flir_"):
        return "flir"
    parts = s.split("_")
    return "_".join(parts[:2]) if len(parts) >= 2 else parts[0]


def extract_frame_number(stem: str) -> int | None:
    """Try to pull a numeric frame index from the end of a stem."""
    s = stem
    if s.startswith("czoom_"):
        s = s[6:]
    s = re.sub(r"_z\d+$", "", s)
    m = re.search(r"_(\d+)$", s)
    return int(m.group(1)) if m else None


def prefix_breakdown(stem_sets: dict[str, set[str]]):
    """Print a per-prefix summary: file counts, video IDs, frame ranges."""
    # Build prefix -> split -> stems
    prefix_map = defaultdict(lambda: defaultdict(list))
    for split_name, stems in stem_sets.items():
        for s in stems:
            prefix_map[get_prefix(s)][split_name].append(s)

    # Collect all prefixes that appear in any eval split
    eval_prefixes = sorted(
        px for px, sp in prefix_map.items()
        if sp.get("val") or sp.get("test")
    )

    print("\nPER-PREFIX BREAKDOWN (eval prefixes)")
    print(f"  {'Prefix':<28s} | {'Split':6s} | {'Files':>7s} | {'VidIDs':>7s} | Frame range")
    print(f"  {'-'*28}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*30}")

    for px in eval_prefixes:
        sp = prefix_map[px]
        # Check if any video IDs are shared between train and eval
        train_vids = set(get_video_id(s) for s in sp.get("train", []))
        eval_vids = set(get_video_id(s) for s in sp.get("val", []))
        eval_vids |= set(get_video_id(s) for s in sp.get("test", []))
        shared = train_vids & eval_vids

        first = True
        for sn in ("train", "val", "test"):
            stems = sp.get(sn, [])
            if not stems:
                continue
            vid_ids = set(get_video_id(s) for s in stems)
            nums = sorted(n for s in stems if (n := extract_frame_number(s)) is not None)
            non_num = sum(1 for s in stems if extract_frame_number(s) is None)

            if nums:
                range_str = f"{nums[0]}-{nums[-1]} ({len(nums)} frames)"
            else:
                range_str = ""
            if non_num > 0:
                range_str += f" +{non_num} non-numeric" if range_str else f"{non_num} non-numeric"

            label = px if first else ""
            print(f"  {label:<28s} | {sn:6s} | {len(stems):>7,} | {len(vid_ids):>7,} | {range_str}")
            first = False

        # Warn if shared
        if shared:
            # Near-neighbour check for shared video IDs
            train_nums = set()
            for s in sp.get("train", []):
                if get_video_id(s) in shared:
                    n = extract_frame_number(s)
                    if n is not None:
                        train_nums.add(n)
            for esn in ("val", "test"):
                eval_stems = [s for s in sp.get(esn, []) if get_video_id(s) in shared]
                if not eval_stems:
                    continue
                e_nums = [n for s in eval_stems if (n := extract_frame_number(s)) is not None]
                near = 0
                for en in e_nums:
                    for d in range(-5, 6):
                        if d != 0 and (en + d) in train_nums:
                            near += 1
                            break
                near_str = f", {near}/{len(e_nums)} within +/-5 frames of train" if e_nums else ""
                print(f"  {'':28s}   ** {esn}: {len(shared)} shared vid ID(s) with train{near_str}")
        print(f"  {'-'*28}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*30}")


def check_pair(name_a: str, stems_a: set, name_b: str, stems_b: set):
    """Check overlap between two splits."""
    overlap = stems_a & stems_b
    if overlap:
        print(f"\n  [LEAK] {len(overlap)} exact filename collisions between {name_a} and {name_b}:")
        for s in sorted(overlap)[:20]:
            print(f"         {s}")
        if len(overlap) > 20:
            print(f"         ... and {len(overlap) - 20} more")
    else:
        print(f"  [OK]   {name_a} vs {name_b}: 0 exact collisions")
    return overlap


def check_video_leak(name_a: str, stems_a: set, name_b: str, stems_b: set):
    """Check if frames from the same video appear in both splits."""
    vids_a = defaultdict(set)
    vids_b = defaultdict(set)
    for s in stems_a:
        vids_a[get_video_id(s)].add(s)
    for s in stems_b:
        vids_b[get_video_id(s)].add(s)

    shared_vids = set(vids_a.keys()) & set(vids_b.keys())
    affected_a = 0
    affected_b = 0
    if shared_vids:
        print(f"\n  [WARN] {len(shared_vids)} video IDs appear in BOTH {name_a} and {name_b}:")
        for vid in sorted(shared_vids)[:15]:
            n_a = len(vids_a[vid])
            n_b = len(vids_b[vid])
            affected_a += n_a
            affected_b += n_b
            print(f"         '{vid}': {n_a} frames in {name_a}, {n_b} frames in {name_b}")
        if len(shared_vids) > 15:
            # Count remaining
            for vid in sorted(shared_vids)[15:]:
                affected_a += len(vids_a[vid])
                affected_b += len(vids_b[vid])
            print(f"         ... and {len(shared_vids) - 15} more")

        pct_a = affected_a / max(1, len(stems_a)) * 100
        pct_b = affected_b / max(1, len(stems_b)) * 100
        print(f"         Impact: {affected_a}/{len(stems_a)} ({pct_a:.2f}%) of {name_a}, "
              f"{affected_b}/{len(stems_b)} ({pct_b:.2f}%) of {name_b}")
    else:
        print(f"  [OK]   {name_a} vs {name_b}: 0 shared video IDs")
    return shared_vids, affected_a, affected_b


def main():
    ap = argparse.ArgumentParser(description="Check for data leakage across splits")
    ap.add_argument("--dataset-yaml", type=str, default=None, help="Path to dataset.yaml")
    ap.add_argument("--train", type=str, default=None, help="Path to train/images dir")
    ap.add_argument("--val", type=str, default=None, help="Path to val/images dir")
    ap.add_argument("--test", type=str, default=None, help="Path to test/images dir")
    args = ap.parse_args()

    splits = {}

    if args.dataset_yaml:
        with open(args.dataset_yaml) as f:
            cfg = yaml.safe_load(f)
        base = Path(cfg.get("path", Path(args.dataset_yaml).parent))
        for split_name in ["train", "val", "test"]:
            rel = cfg.get(split_name)
            if rel:
                p = Path(rel) if Path(rel).is_absolute() else base / rel
                splits[split_name] = p
    else:
        if args.train: splits["train"] = Path(args.train)
        if args.val:   splits["val"]   = Path(args.val)
        if args.test:  splits["test"]  = Path(args.test)

    if len(splits) < 2:
        print("Need at least 2 splits to check. Provide --dataset-yaml or --train/--val/--test.")
        return

    # Collect
    stem_sets = {}
    for name, path in splits.items():
        stems = collect_stems(path)
        stem_sets[name] = stems
        print(f"  {name}: {len(stems)} images in {path}")

    # Check all pairs
    print("EXACT FILENAME COLLISION CHECK")
    split_names = list(stem_sets.keys())
    total_leaks = 0
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            overlap = check_pair(split_names[i], stem_sets[split_names[i]],
                                  split_names[j], stem_sets[split_names[j]])
            total_leaks += len(overlap)

    print("VIDEO-LEVEL LEAKAGE CHECK (same source video in multiple splits)")
    total_vid_leaks = 0
    total_affected = defaultdict(int)  # split_name -> affected frames
    for i in range(len(split_names)):
        for j in range(i + 1, len(split_names)):
            shared, aff_a, aff_b = check_video_leak(
                split_names[i], stem_sets[split_names[i]],
                split_names[j], stem_sets[split_names[j]])
            total_vid_leaks += len(shared)
            total_affected[split_names[i]] = max(total_affected[split_names[i]], aff_a)
            total_affected[split_names[j]] = max(total_affected[split_names[j]], aff_b)

    # Per-prefix breakdown
    prefix_breakdown(stem_sets)

    # Summary
    print("\nSUMMARY")
    if total_leaks == 0 and total_vid_leaks == 0:
        print("  ALL CLEAR: No leakage detected.")
    else:
        if total_leaks > 0:
            print(f"  [LEAK] {total_leaks} exact filename collisions found")
        if total_vid_leaks > 0:
            print(f"  [WARN] {total_vid_leaks} shared video IDs found")
            # Compute eval impact (val + test only)
            eval_splits = [s for s in split_names if s != "train"]
            eval_total = sum(len(stem_sets[s]) for s in eval_splits)
            eval_affected = sum(total_affected.get(s, 0) for s in eval_splits)
            if eval_total > 0:
                print(f"  [IMPACT] {eval_affected}/{eval_total} eval images potentially affected "
                      f"({eval_affected/eval_total*100:.2f}%)")


if __name__ == "__main__":
    main()
