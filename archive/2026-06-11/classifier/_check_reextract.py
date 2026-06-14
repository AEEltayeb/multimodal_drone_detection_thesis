"""Quick diagnostic: check which manifest entries can't find original frames."""
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_confuser_filter import _find_original_frame

manifest = pd.read_csv(Path(__file__).resolve().parent / "runs" / "patches" / "manifest.csv")
missing = []
found = 0
for _, row in manifest.iterrows():
    fp, lp = _find_original_frame(row["stem"], row["modality"], row["category"])
    if fp is None or not fp.exists():
        missing.append({"stem": row["stem"], "mod": row["modality"], "cat": row["category"],
                        "tried_frame": str(fp), "tried_label": str(lp)})
    elif lp is None or not lp.exists():
        missing.append({"stem": row["stem"], "mod": row["modality"], "cat": row["category"],
                        "tried_frame": str(fp), "tried_label": str(lp), "issue": "no_label"})
    else:
        found += 1

print(f"Found: {found}/{len(manifest)}")
print(f"Missing: {len(missing)}")
if missing:
    df = pd.DataFrame(missing)
    print("\nMissing by category:")
    if "issue" in df.columns:
        print(df.groupby(["mod", "cat", "issue"]).size().to_string())
    else:
        print(df.groupby(["mod", "cat"]).size().to_string())
    print("\nFirst 10 missing:")
    for m in missing[:10]:
        print(f"  stem={m['stem']}")
        print(f"    tried: {m['tried_frame']}")
