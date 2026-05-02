"""
split_sequences.py — Sequence-level train/val/test splitter.

Ensures no sequence appears in more than one split (prevents frame-level
leakage where adjacent frames from the same flight leak between splits).

Stratification is done per `source` so each split has proportional coverage
from every dataset.
"""

import numpy as np
import pandas as pd


def split_by_sequence(df: pd.DataFrame,
                      train_frac: float = 0.70,
                      val_frac: float = 0.15,
                      test_frac: float = 0.15,
                      random_state: int = 42,
                      stratify_col: str = "source") -> dict:
    """
    Split df into train/val/test by sequence, stratified by `stratify_col`.

    Returns a dict with keys 'train', 'val', 'test', each mapping to a
    boolean mask over df.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, \
        "Fractions must sum to 1.0"
    assert "sequence" in df.columns, "df must have a 'sequence' column"
    assert stratify_col in df.columns, f"df must have a '{stratify_col}' column"

    rng = np.random.default_rng(random_state)

    train_mask = np.zeros(len(df), dtype=bool)
    val_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)

    # For each stratum, split sequences independently
    for stratum, stratum_df in df.groupby(stratify_col):
        sequences = stratum_df["sequence"].unique()
        sequences = np.array(sorted(sequences))  # stable
        rng.shuffle(sequences)

        n = len(sequences)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))

        train_seqs = set(sequences[:n_train])
        val_seqs = set(sequences[n_train:n_train + n_val])
        test_seqs = set(sequences[n_train + n_val:])

        # Apply to the full df (not just stratum_df) via sequence membership
        in_stratum = (df[stratify_col] == stratum).values
        seq_col = df["sequence"].values

        train_mask |= in_stratum & np.array([s in train_seqs for s in seq_col])
        val_mask   |= in_stratum & np.array([s in val_seqs   for s in seq_col])
        test_mask  |= in_stratum & np.array([s in test_seqs  for s in seq_col])

    # Sanity: every row belongs to exactly one split
    assignment = train_mask.astype(int) + val_mask.astype(int) + test_mask.astype(int)
    if not (assignment == 1).all():
        n_bad = int((assignment != 1).sum())
        raise RuntimeError(f"Split assignment bug: {n_bad} rows not in exactly one split")

    return {"train": train_mask, "val": val_mask, "test": test_mask}


def print_split_summary(df: pd.DataFrame, masks: dict):
    """Pretty-print a breakdown of the split by source × label."""
    print("Split summary (by source × label):")
    print(f"  {'split':<7s} {'source':<12s} {'rows':>8s} {'pos':>8s} {'neg':>8s} {'seqs':>6s}")
    for split_name in ["train", "val", "test"]:
        mask = masks[split_name]
        sub = df[mask]
        for src in sorted(sub["source"].unique()):
            src_sub = sub[sub["source"] == src]
            pos = int(src_sub["label"].sum())
            neg = len(src_sub) - pos
            seqs = src_sub["sequence"].nunique()
            print(f"  {split_name:<7s} {src:<12s} {len(src_sub):>8d} "
                  f"{pos:>8d} {neg:>8d} {seqs:>6d}")
    # Leakage check
    train_seqs = set(df[masks["train"]]["sequence"].unique())
    val_seqs   = set(df[masks["val"]]["sequence"].unique())
    test_seqs  = set(df[masks["test"]]["sequence"].unique())
    overlap_tv = train_seqs & val_seqs
    overlap_tt = train_seqs & test_seqs
    overlap_vt = val_seqs & test_seqs
    print(f"\n  Leakage check: train&val={len(overlap_tv)}, "
          f"train&test={len(overlap_tt)}, val&test={len(overlap_vt)} "
          f"(all should be 0)")


if __name__ == "__main__":
    # Smoke test
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="runs/curated_frame_dataset.csv")
    args = parser.parse_args()

    from pathlib import Path
    script_dir = Path(__file__).resolve().parent
    classifier_dir = script_dir.parent
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = classifier_dir / csv_path

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}")
    masks = split_by_sequence(df)
    print_split_summary(df, masks)
