import json
import pandas as pd
import numpy as np
import joblib

# Load model and data
m = joblib.load("fusion_models/retrained_v2_32feat/model.joblib")
model = m["model"]
features = m["features"]
df = pd.read_csv("fusion_models/retrained_v2_32feat/fusion_dataset.csv")

LABELS = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}

X = df[features].values
y_true = df["trust_label"].values
y_pred = model.predict(X)

print("PREDICTION DISTRIBUTION (all data):")
print(f"  {'Label':<15s} {'GT count':>10s} {'Pred count':>10s}")
for i in range(4):
    gt_n = (y_true == i).sum()
    pred_n = (y_pred == i).sum()
    print(f"  {LABELS[i]:<15s} {gt_n:>10,d} {pred_n:>10,d}")

# Now check: on Anti-UAV only (real paired data)
auv = df[df["source"] == "antiuav"]
X_a = auv[features].values
y_a = auv["trust_label"].values
p_a = model.predict(X_a)
print("\nANTI-UAV ONLY (real drone data):")
print(f"  {'Label':<15s} {'GT':>8s} {'Pred':>8s}")
for i in range(4):
    print(f"  {LABELS[i]:<15s} {(y_a==i).sum():>8d} {(p_a==i).sum():>8d}")

# Svanstrom only
svan = df[df["source"] == "svanstrom"]
if len(svan):
    X_s = svan[features].values
    y_s = svan["trust_label"].values
    p_s = model.predict(X_s)
    print("\nSVANSTROM ONLY:")
    print(f"  {'Label':<15s} {'GT':>8s} {'Pred':>8s}")
    for i in range(4):
        print(f"  {LABELS[i]:<15s} {(y_s==i).sum():>8d} {(p_s==i).sum():>8d}")

# Key question: when both models detect, does it prefer IR?
both_det = df[(df["rgb_max_conf"] > 0) & (df["ir_max_conf"] > 0)]
X_b = both_det[features].values
p_b = model.predict(X_b)
y_b = both_det["trust_label"].values
print(f"\nWHEN BOTH DETECT ({len(both_det)} frames):")
print(f"  {'Label':<15s} {'GT':>8s} {'Pred':>8s}")
for i in range(4):
    print(f"  {LABELS[i]:<15s} {(y_b==i).sum():>8d} {(p_b==i).sum():>8d}")
