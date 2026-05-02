r"""
Merge IR_dsetV4 + 3rd Anti-UAV → IR_dsetV5.

Strategy:
  - dsetV4: keep existing train/val/test splits as-is (already leak-free)
  - Anti-UAV: pool all sequences from train+val, re-split at SEQUENCE
    level 80/10/10 to prevent video leakage
  - Copy images+labels into G:\drone\IR_dsetV5\{train,val,test}\{images,labels}
  - Prefix filenames: dv4_<name> and auv_<name> to avoid collisions
"""

import argparse
import json
import random
import shutil
from pathlib import Path

# ── Paths ──
DSETV4_DIR = Path(r"datasets\IR_dsetV4")
ANTIUAV_YOLO_DIR = Path(r"G:\drone\3rd_AntiUAV_yolo")
OUTPUT_DIR = Path(r"G:\drone\IR_dsetV5")

SEED = 42
SPLIT_RATIOS = (0.80, 0.10, 0.10)  # train, val, test


def collect_dsetv4_files(dsetv4_dir: Path) -> dict:
    """Collect dsetV4 files organized by split. Keep as-is."""
    result = {"train": [], "val": [], "test": []}

    for split in ["train", "val", "test"]:
        img_dir = dsetv4_dir / "images" / split
        lbl_dir = dsetv4_dir / "labels" / split

        if not img_dir.exists():
            print(f"  [WARN] dsetV4 {split} images not found: {img_dir}")
            continue

        # Pre-scan labels for fast lookup
        existing_labels = set()
        if lbl_dir.exists():
            existing_labels = {f.stem for f in lbl_dir.iterdir() if f.suffix == ".txt"}

        for img_file in sorted(img_dir.iterdir()):
            if img_file.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                lbl_file = lbl_dir / f"{img_file.stem}.txt"
                result[split].append({
                    "img_src": img_file,
                    "lbl_src": lbl_file if img_file.stem in existing_labels else None,
                    "out_name": f"dv4_{img_file.stem}",
                    "source": "dsetV4",
                })

    return result


def collect_antiuav_sequences(antiuav_dir: Path) -> list:
    """Collect Anti-UAV files grouped by sequence for leak-free splitting."""
    sequences = {}

    for split_name in ["train", "val"]:
        split_dir = antiuav_dir / split_name
        if not split_dir.exists():
            continue

        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"

        if not img_dir.exists():
            continue

        # Pre-scan labels for fast lookup
        existing_labels = set()
        if lbl_dir.exists():
            existing_labels = {f.stem for f in lbl_dir.iterdir() if f.suffix == ".txt"}

        for img_file in sorted(img_dir.iterdir()):
            if img_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                continue

            # Extract sequence name from filename pattern: seqname_NNNNNN.jpg
            stem = img_file.stem
            # Find the last underscore before the frame number
            parts = stem.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                seq_name = parts[0]
            else:
                seq_name = stem

            if seq_name not in sequences:
                sequences[seq_name] = []

            lbl_file = lbl_dir / f"{stem}.txt"
            sequences[seq_name].append({
                "img_src": img_file,
                "lbl_src": lbl_file if stem in existing_labels else None,
                "out_name": f"auv_{stem}",
                "source": "antiuav",
                "sequence": seq_name,
            })

    return sequences


def split_sequences(sequences: dict, ratios: tuple, seed: int) -> dict:
    """Split sequences into train/val/test at the sequence level."""
    rng = random.Random(seed)
    seq_names = sorted(sequences.keys())
    rng.shuffle(seq_names)

    n = len(seq_names)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    # rest goes to test

    train_seqs = seq_names[:n_train]
    val_seqs = seq_names[n_train:n_train + n_val]
    test_seqs = seq_names[n_train + n_val:]

    result = {"train": [], "val": [], "test": []}
    for seq in train_seqs:
        result["train"].extend(sequences[seq])
    for seq in val_seqs:
        result["val"].extend(sequences[seq])
    for seq in test_seqs:
        result["test"].extend(sequences[seq])

    print(f"\n  Anti-UAV sequence split:")
    print(f"    Train: {len(train_seqs)} sequences → {len(result['train'])} frames")
    print(f"    Val:   {len(val_seqs)} sequences → {len(result['val'])} frames")
    print(f"    Test:  {len(test_seqs)} sequences → {len(result['test'])} frames")

    return result


def copy_files(file_list: list, split: str, output_dir: Path, dry_run: bool):
    """Copy image+label pairs to output directory."""
    img_out = output_dir / split / "images"
    lbl_out = output_dir / split / "labels"

    if not dry_run:
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    for entry in file_list:
        out_name = entry["out_name"]
        img_src = entry["img_src"]
        lbl_src = entry["lbl_src"]

        if not dry_run:
            # Copy image
            shutil.copy2(str(img_src), str(img_out / f"{out_name}{img_src.suffix}"))

            # Copy or create empty label
            if lbl_src and lbl_src.exists():
                shutil.copy2(str(lbl_src), str(lbl_out / f"{out_name}.txt"))
            else:
                (lbl_out / f"{out_name}.txt").write_text("")

        copied += 1

    return copied


def main():
    parser = argparse.ArgumentParser(description="Merge dsetV4 + Anti-UAV → IR_dsetV5")
    parser.add_argument("--dsetv4", type=Path, default=DSETV4_DIR)
    parser.add_argument("--antiuav", type=Path, default=ANTIUAV_YOLO_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print(f"dsetV4:   {args.dsetv4}")
    print(f"Anti-UAV: {args.antiuav}")
    print(f"Output:   {args.output}")
    print(f"Seed:     {args.seed}")

    # Step 1: Collect dsetV4 files (keep existing splits)
    print("\n[1/4] Collecting dsetV4 files...")
    dv4_files = collect_dsetv4_files(args.dsetv4)
    for split, files in dv4_files.items():
        print(f"  dsetV4 {split}: {len(files)} files")

    # Step 2: Collect Anti-UAV sequences
    print("\n[2/4] Collecting Anti-UAV sequences...")
    auv_sequences = collect_antiuav_sequences(args.antiuav)
    total_auv = sum(len(frames) for frames in auv_sequences.values())
    print(f"  {len(auv_sequences)} sequences, {total_auv} total frames")

    # Step 3: Split Anti-UAV at sequence level
    print("\n[3/4] Splitting Anti-UAV sequences 80/10/10...")
    auv_files = split_sequences(auv_sequences, SPLIT_RATIOS, args.seed)

    # Step 4: Merge and copy
    print(f"\n[4/4] {'DRY RUN — ' if args.dry_run else ''}Copying files to {args.output}...")
    total_stats = {}

    for split in ["train", "val", "test"]:
        merged = dv4_files[split] + auv_files[split]
        n_dv4 = len(dv4_files[split])
        n_auv = len(auv_files[split])

        copied = copy_files(merged, split, args.output, args.dry_run)
        total_stats[split] = {"total": copied, "dsetV4": n_dv4, "antiuav": n_auv}

        print(f"  {split}: {copied} files (dsetV4={n_dv4}, antiuav={n_auv})")

    # Write dataset.yaml
    if not args.dry_run:
        yaml_content = f"""# IR_dsetV5 — dsetV4 + 3rd Anti-UAV merged dataset
# Sources: dsetV4 (6-source IR), 3rd Anti-UAV Challenge (IR)
# Split: dsetV4 splits preserved, Anti-UAV re-split at sequence level 80/10/10
# Seed: {args.seed}
# Class 0 = drone (UAV)

path: {args.output}

train: train/images
val: val/images
test: test/images

nc: 1
names: ['drone']
"""
        (args.output / "dataset.yaml").write_text(yaml_content, encoding="utf-8")
        print(f"\n  dataset.yaml written")

    # Write split manifest
    if not args.dry_run:
        manifest = {
            "seed": args.seed,
            "split_ratios": list(SPLIT_RATIOS),
            "sources": {
                "dsetV4": {
                    "path": str(args.dsetv4),
                    "strategy": "preserve_existing_splits",
                },
                "antiuav": {
                    "path": str(args.antiuav),
                    "strategy": "sequence_level_resplit",
                    "total_sequences": len(auv_sequences),
                },
            },
            "splits": total_stats,
        }
        (args.output / "split_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        print(f"  split_manifest.json written")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY — IR_dsetV5")
    print(f"{'='*60}")
    grand_total = 0
    for split, s in total_stats.items():
        pct = s["total"] / sum(ss["total"] for ss in total_stats.values()) * 100
        print(f"  {split:5s}: {s['total']:>6,} ({pct:5.1f}%)  "
              f"[dsetV4={s['dsetV4']:>6,}, antiuav={s['antiuav']:>6,}]")
        grand_total += s["total"]
    print(f"  {'total':5s}: {grand_total:>6,}")

    if args.dry_run:
        print("\n[DRY RUN] No files written.")


if __name__ == "__main__":
    main()
