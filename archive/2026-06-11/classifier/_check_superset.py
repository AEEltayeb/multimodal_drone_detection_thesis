"""Quick check: what columns exist across all CSVs, can we merge on stem+source?"""
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent / "fusion_models"

csvs = {
    "lean19": BASE / "lean19/fusion_dataset_lean19.csv",
    "lean19_v2_C": BASE / "lean19_v2_C/fusion_dataset.csv",
    "retrained_v2_32feat": BASE / "retrained_v2_32feat/fusion_dataset.csv",
}

all_feat_cols = set()
for name, path in csvs.items():
    df = pd.read_csv(path, nrows=5)
    feat_cols = [c for c in df.columns if c not in ("trust_label", "trust_label_strict", "stem", "source", "sequence_id")]
    print(f"\n=== {name} ({len(pd.read_csv(path)):,} rows) ===")
    print(f"  Key cols: stem={'stem' in df.columns}  source={'source' in df.columns}")
    print(f"  Feature cols ({len(feat_cols)}): {feat_cols}")
    all_feat_cols.update(feat_cols)

print(f"\n=== SUPERSET: {len(all_feat_cols)} unique feature columns ===")
for c in sorted(all_feat_cols):
    present = []
    for name, path in csvs.items():
        hdr = pd.read_csv(path, nrows=0).columns
        if c in hdr:
            present.append(name)
    print(f"  {c:40s} in: {', '.join(present)}")

# Check merge feasibility
print("\n=== Merge check ===")
l19 = pd.read_csv(csvs["lean19"])
r32 = pd.read_csv(csvs["retrained_v2_32feat"])
merged = l19[["stem","source"]].merge(r32[["stem","source"]], on=["stem","source"], how="inner")
print(f"  lean19 rows: {len(l19):,}")
print(f"  32feat rows: {len(r32):,}")
print(f"  inner join on stem+source: {len(merged):,}")
