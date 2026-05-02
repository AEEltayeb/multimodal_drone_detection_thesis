"""
curate_dataset.py — Build the Phase 1 honest curated frame-level dataset.

Reads:
  runs/fusion_dataset.csv            (Anti-UAV frame-level)
  runs/svanstrom_frame_dataset.csv   (Svanstrom frame-level)

Writes:
  runs/curated_frame_dataset.csv

Transformations:
  1. Add `sequence` column (extracted from stem via regex).
  2. Add `source` column: "anti_uav" or "svanstrom".
  3. Add `category` column:
       - Anti-UAV  -> "antiuav"
       - Svanstrom -> parsed from stem prefix (DRONE/BIRD/AIRPLANE/HELICOPTER)
  4. Add `lighting` column for reporting ONLY (not a feature):
       - Anti-UAV  -> from its native time_of_day column (day/dusk_dawn/night)
       - Svanstrom -> "unknown"
  5. Drop dataset-tag features that leak: hour, time_of_day, rgb_brightness,
     ir_brightness. These are present on Anti-UAV but absent/sentinel on
     Svanstrom -> any model using them learns "which dataset am I on".
  6. Stratified undersample Anti-UAV positives to ~20K, preserving sequence
     integrity (whole sequences in or out, not partial).
  7. Keep all Svanstrom rows (positives + hard negatives).

Usage:
    python curate_dataset.py
    python curate_dataset.py --anti-uav-positive-cap 20000
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Column policy
# ---------------------------------------------------------------------------

# Features actually fed to the classifier (intersection of both sources,
# excludes dataset-tag columns).
FEATURE_COLS = [
    "max_conf_rgb", "max_conf_ir",
    "conf_max", "conf_min", "conf_mean", "conf_delta",
    "both_detected",
    "n_dets_rgb", "n_dets_ir", "n_dets_total",
    "conf_rgb_2nd", "conf_ir_2nd",
    "rgb_area_norm", "ir_area_norm",
]

# Reporting / split metadata (not fed as features)
META_COLS = ["stem", "sequence", "source", "category", "lighting", "label"]

# Columns we explicitly drop because they encode dataset origin
LEAKAGE_COLS = ["hour", "time_of_day", "rgb_brightness", "ir_brightness"]

SEQ_RE = re.compile(r"^(.+)_f\d+$")


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def extract_sequence(stem: str) -> str:
    m = SEQ_RE.match(stem)
    return m.group(1) if m else stem


def extract_svanstrom_category(stem: str) -> str:
    # Svanstrom stems: IR_AIRPLANE_001_f000000, IR_DRONE_001_f000000, ...
    parts = stem.split("_")
    if len(parts) >= 2 and parts[0] == "IR":
        return parts[1]  # DRONE / BIRD / AIRPLANE / HELICOPTER
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Per-source loaders
# ---------------------------------------------------------------------------

def load_anti_uav(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source"] = "anti_uav"
    df["category"] = "antiuav"
    df["sequence"] = df["stem"].apply(extract_sequence)

    # Lighting for reporting only — use Anti-UAV's native time_of_day column
    if "time_of_day" in df.columns:
        df["lighting"] = df["time_of_day"].fillna("unknown")
    else:
        df["lighting"] = "unknown"

    return df


def load_svanstrom(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["source"] = "svanstrom"
    df["category"] = df["stem"].apply(extract_svanstrom_category)
    df["sequence"] = df["stem"].apply(extract_sequence)
    df["lighting"] = "unknown"  # no lighting info for Svanstrom
    return df


# ---------------------------------------------------------------------------
# Subsampling
# ---------------------------------------------------------------------------

def subsample_anti_uav(df: pd.DataFrame, positive_cap: int,
                       random_state: int = 42) -> pd.DataFrame:
    """
    Reduce Anti-UAV positives to `positive_cap` while keeping whole sequences.
    Negatives are kept as-is (there are only ~800 of them).
    """
    neg = df[df["label"] == 0]
    pos = df[df["label"] == 1]

    if len(pos) <= positive_cap:
        print(f"  Anti-UAV positives ({len(pos)}) already <= cap ({positive_cap}). "
              "Keeping all.")
        return df

    # Sample whole sequences until we hit the cap
    rng = np.random.default_rng(random_state)
    pos_by_seq = pos.groupby("sequence").size().sort_values(ascending=False)
    seq_order = list(pos_by_seq.index)
    rng.shuffle(seq_order)

    selected_seqs = []
    total = 0
    for seq in seq_order:
        seq_size = pos_by_seq[seq]
        if total + seq_size > positive_cap * 1.1:  # allow 10% overshoot
            continue
        selected_seqs.append(seq)
        total += seq_size
        if total >= positive_cap:
            break

    pos_sub = pos[pos["sequence"].isin(selected_seqs)]
    print(f"  Anti-UAV subsample: kept {len(selected_seqs)} sequences, "
          f"{len(pos_sub)} positive rows (target {positive_cap}).")
    return pd.concat([pos_sub, neg], ignore_index=True)


# ---------------------------------------------------------------------------
# Main curation
# ---------------------------------------------------------------------------

def curate(anti_uav_path: Path, svanstrom_path: Path,
           out_path: Path, positive_cap: int) -> pd.DataFrame:
    print("=" * 60)
    print("Curating Phase 1 frame-level dataset")
    print("=" * 60)

    print(f"\nLoading Anti-UAV from {anti_uav_path}")
    df_anti = load_anti_uav(anti_uav_path)
    print(f"  {len(df_anti)} rows "
          f"({df_anti['label'].sum()} pos, "
          f"{len(df_anti) - df_anti['label'].sum()} neg)")

    print(f"\nLoading Svanstrom from {svanstrom_path}")
    df_svan = load_svanstrom(svanstrom_path)
    print(f"  {len(df_svan)} rows "
          f"({df_svan['label'].sum()} pos, "
          f"{len(df_svan) - df_svan['label'].sum()} neg)")
    cat_counts = df_svan.groupby("category")["label"].agg(["count", "sum"])
    print("  Per-category:")
    for cat, row in cat_counts.iterrows():
        pos = int(row["sum"])
        total = int(row["count"])
        print(f"    {cat:<12s} {total:>6d} total, {pos:>6d} pos, {total - pos:>6d} neg")

    # Subsample Anti-UAV
    print(f"\nSubsampling Anti-UAV (positive cap = {positive_cap})")
    df_anti = subsample_anti_uav(df_anti, positive_cap)

    # Drop leakage columns
    for col in LEAKAGE_COLS:
        if col in df_anti.columns:
            df_anti = df_anti.drop(columns=[col])
        if col in df_svan.columns:
            df_svan = df_svan.drop(columns=[col])

    # Keep only the columns both sources share + our meta
    keep = FEATURE_COLS + META_COLS
    missing_anti = [c for c in keep if c not in df_anti.columns]
    missing_svan = [c for c in keep if c not in df_svan.columns]
    if missing_anti:
        raise ValueError(f"Anti-UAV missing required columns: {missing_anti}")
    if missing_svan:
        raise ValueError(f"Svanstrom missing required columns: {missing_svan}")

    df_anti = df_anti[keep]
    df_svan = df_svan[keep]

    # Concat
    df_merged = pd.concat([df_anti, df_svan], ignore_index=True)
    df_merged = df_merged.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Summary
    print("\n" + "=" * 60)
    print("Curated dataset summary")
    print("=" * 60)
    n_pos = int(df_merged["label"].sum())
    n_neg = int(len(df_merged) - n_pos)
    print(f"  Total: {len(df_merged)} rows ({n_pos} pos, {n_neg} neg)")
    print(f"  Sequences: {df_merged['sequence'].nunique()}")
    print("\n  Per source × category:")
    grp = df_merged.groupby(["source", "category"])["label"].agg(["count", "sum"])
    for (src, cat), row in grp.iterrows():
        total = int(row["count"])
        pos = int(row["sum"])
        print(f"    {src:<10s} {cat:<12s} {total:>6d} total, "
              f"{pos:>6d} pos, {total - pos:>6d} neg")

    # Write
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_csv(out_path, index=False)
    print(f"\n  Saved to {out_path}")

    return df_merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anti-uav-csv", default="runs/fusion_dataset.csv")
    parser.add_argument("--svanstrom-csv", default="runs/svanstrom_frame_dataset.csv")
    parser.add_argument("--out", default="runs/curated_frame_dataset.csv")
    parser.add_argument("--anti-uav-positive-cap", type=int, default=20000)
    args = parser.parse_args()

    # Resolve paths relative to classifier/ (parent of this file's folder)
    script_dir = Path(__file__).resolve().parent
    classifier_dir = script_dir.parent

    def resolve(p):
        p = Path(p)
        return p if p.is_absolute() else classifier_dir / p

    curate(
        anti_uav_path=resolve(args.anti_uav_csv),
        svanstrom_path=resolve(args.svanstrom_csv),
        out_path=resolve(args.out),
        positive_cap=args.anti_uav_positive_cap,
    )


if __name__ == "__main__":
    main()
