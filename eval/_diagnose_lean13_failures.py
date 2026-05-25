"""Diagnose which clips the Lean-13 model fails on and why."""
import json, re
from pathlib import Path
from collections import Counter
import joblib, numpy as np, pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupShuffleSplit

REPO = Path(__file__).resolve().parent.parent
SEQ_RE = re.compile(r"^(.+?)(?:_f\d+|_frame\d+|_\d{4,})(?:_visible|_infrared)?$", re.I)
LABEL = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}

def seq_id(stem, source):
    m = SEQ_RE.match(str(stem))
    base = m.group(1).rstrip("_") if m else str(stem)
    return f"{source}::{base}"

csv = REPO / "classifier/fusion_models/lean13/fusion_dataset_lean13.csv"
mdl = REPO / "classifier/fusion_models/lean13/model.joblib"
df = pd.read_csv(csv)
b = joblib.load(mdl)
model = b["model"]; feats = b["features"]
df["sequence_id"] = df.apply(lambda r: seq_id(r["stem"], r["source"]), axis=1)

gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
_, test_idx = next(gss.split(df, df["trust_label"], groups=df["sequence_id"].values))
te = df.iloc[test_idx].copy()
te["pred"] = model.predict(te[feats].values)

# Focus on the failing video clips
fails = [
    "video_drone_two_birds_drone",
    "video_drone_drone_and_bird_sky_and_trees_short",
    "video_helicopters_helicopter_overhead_short",
]
for src in fails:
    sub = te[te["source"] == src]
    if sub.empty:
        print(f"[skip] {src}: no rows"); continue
    print(f"\n=== {src} (n={len(sub)}) ===")
    print("GT label distribution :", dict(Counter(sub["trust_label"])))
    print("PRED label distribution:", dict(Counter(sub["pred"])))
    print("Confusion matrix (rows=GT, cols=PRED):")
    cm = confusion_matrix(sub["trust_label"], sub["pred"], labels=[0, 1, 2, 3])
    print("            pred->  reject  rgb     ir      both")
    for i, name in enumerate(["reject_both", "trust_rgb", "trust_ir", "trust_both"]):
        print(f"  GT={name:13s} {cm[i,0]:>6d} {cm[i,1]:>6d} {cm[i,2]:>6d} {cm[i,3]:>6d}")
    # Mean detection conf when GT says trust_both but pred says reject
    miss = sub[(sub["trust_label"] == 3) & (sub["pred"] == 0)]
    if not miss.empty:
        print(f"\n  trust_both -> reject (n={len(miss)}): "
              f"mean rgb_conf={miss['rgb_max_conf'].mean():.3f}  "
              f"mean ir_conf={miss['ir_max_conf'].mean():.3f}  "
              f"mean ir_bbox_area={miss['ir_best_log_bbox_area'].mean():.2f}")
    miss2 = sub[(sub["trust_label"] == 3) & (sub["pred"] != 3)]
    if not miss2.empty:
        print(f"  trust_both -> non-both (n={len(miss2)}): "
              f"mean rgb_conf={miss2['rgb_max_conf'].mean():.3f}  "
              f"mean ir_conf={miss2['ir_max_conf'].mean():.3f}")

# Compare to a CORRECT clip for baseline
print("\n=== REFERENCE: antiuav (sample) ===")
sub = te[te["source"] == "antiuav"]
print(f"  rgb_conf when trust_both: {sub[sub['trust_label']==3]['rgb_max_conf'].mean():.3f}")
print(f"  ir_conf when trust_both : {sub[sub['trust_label']==3]['ir_max_conf'].mean():.3f}")
print(f"  ir_bbox_area when trust_both: {sub[sub['trust_label']==3]['ir_best_log_bbox_area'].mean():.2f}")
