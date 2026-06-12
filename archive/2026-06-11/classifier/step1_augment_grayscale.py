"""Step 1: Create grayscale-augmented fusion dataset."""
import pandas as pd
from pathlib import Path

DATA = Path(__file__).resolve().parent / "runs" / "reliability" / "fusion"
SRC = DATA / "fusion_dataset_v3more.csv"
OUT = DATA / "fusion_dataset_v3more_gray_aug.csv"

IR_SCENE = ["ir_img_mean","ir_img_std","ir_img_dynamic_range",
            "ir_img_entropy","ir_sky_ground_ratio","ir_edge_density","ir_blurriness"]
RGB_SCENE = [c.replace("ir_","rgb_") for c in IR_SCENE]

print(f"Loading {SRC.name}...")
df = pd.read_csv(SRC)
print(f"  Original: {len(df):,} rows")

# Create grayscale copy: replace IR scene globals with RGB values
gray = df.copy()
for ir_col, rgb_col in zip(IR_SCENE, RGB_SCENE):
    gray[ir_col] = gray[rgb_col]
gray["modality_mode"] = "grayscale"
df["modality_mode"] = "paired"

combined = pd.concat([df, gray], ignore_index=True)
combined.to_csv(OUT, index=False)
print(f"  Augmented: {len(combined):,} rows -> {OUT.name}")
print("Step 1 done.")
