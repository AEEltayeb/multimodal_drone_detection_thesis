"""Generate all thesis-ready plots for the professor email."""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO = Path(__file__).resolve().parent
OUT = REPO / "classifier" / "runs" / "professor_plots"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.facecolor": "white",
})

# ===========================================================
# 1. XGBoost Feature Importance (top 10)
# ===========================================================
print("1. Feature importance...")
import joblib
m = joblib.load(REPO / "classifier" / "runs" / "reliability" / "fusion" / "fusion_no_fn_model.joblib")
feats = m["features"]
imp = m["model"].feature_importances_
ranked = sorted(zip(feats, imp), key=lambda x: -x[1])[:10]
names = [r[0] for r in ranked][::-1]
vals = [r[1] for r in ranked][::-1]

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#4361ee" if "detect" in n or "detected" in n else "#7209b7" if "n_dets" in n else "#adb5bd" for n in names]
bars = ax.barh(range(len(names)), vals, color=colors, edgecolor="white", linewidth=0.5)
ax.set_yticks(range(len(names)))
ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=10)
ax.set_xlabel("Feature Importance (gain)")
ax.set_title("XGBoost Fusion Classifier — Top 10 Features")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# Add value labels
for bar, v in zip(bars, vals):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f"{v:.3f}", va="center", fontsize=9, color="#333")
# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor="#4361ee", label="Agreement features (96.3%)"),
                   Patch(facecolor="#7209b7", label="Detection count"),
                   Patch(facecolor="#adb5bd", label="Other")]
ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
plt.tight_layout()
fig.savefig(OUT / "1_feature_importance.png", dpi=200)
plt.close()
print(f"  -> {OUT / '1_feature_importance.png'}")

# ===========================================================
# 2. XGBoost Classifier Confusion Matrix (Anti-UAV)
# ===========================================================
print("2. Classifier confusion matrix...")
# From eval results: classifier TP=2949, TN=27, FP=1, FN=23
# 4 classes: reject_both(0), trust_rgb(1), trust_ir(2), trust_both(3)
# We only have binary drone/no-drone stats, so show that
cm_clf = np.array([[27, 1], [23, 2949]])
labels_clf = ["No Drone", "Drone"]
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm_clf, cmap="Blues")
for i in range(2):
    for j in range(2):
        color = "white" if cm_clf[i, j] > 1000 else "black"
        ax.text(j, i, f"{cm_clf[i, j]}", ha="center", va="center", fontsize=16, fontweight="bold", color=color)
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(labels_clf)
ax.set_yticklabels(labels_clf)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Fusion Classifier — Anti-UAV Test (n=3000)\nP=1.000  R=0.992")
plt.tight_layout()
fig.savefig(OUT / "2_classifier_confusion_matrix.png", dpi=200)
plt.close()
print(f"  -> {OUT / '2_classifier_confusion_matrix.png'}")

# ===========================================================
# 3. Confuser Filter Confusion Matrices (RGB + IR side by side)
# ===========================================================
print("3. Confuser filter confusion matrices...")
rgb_metrics = json.loads((REPO / "classifier" / "runs" / "patches" / "confuser_filter4_rgb_metrics.json").read_text())
ir_metrics = json.loads((REPO / "classifier" / "runs" / "patches" / "confuser_filter4_ir_metrics.json").read_text())

class_names = ["airplane", "helicopter", "bird", "other"]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for ax, metrics, title in [(ax1, rgb_metrics, "RGB Confuser Filter"), (ax2, ir_metrics, "IR Confuser Filter")]:
    cm = np.array(metrics["final"]["cm"])
    im = ax.imshow(cm, cmap="Blues")
    for i in range(4):
        for j in range(4):
            color = "white" if cm[i, j] > 500 else "black"
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=12, fontweight="bold", color=color)
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(class_names, rotation=35, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    acc = metrics["best_val_acc"]
    ax.set_title(f"{title}\nVal Acc: {acc*100:.1f}%")

plt.tight_layout()
fig.savefig(OUT / "3_confuser_confusion_matrices.png", dpi=200)
plt.close()
print(f"  -> {OUT / '3_confuser_confusion_matrices.png'}")

# ===========================================================
# 4. Threshold vs Performance Curve
# ===========================================================
print("4. Threshold sweep curve...")
# RGB sweep data from training
rgb_sweep = {
    0.5: {"confusers_caught": 0.972, "drones_passed": 0.987},
    0.6: {"confusers_caught": 0.953, "drones_passed": 0.989},
    0.7: {"confusers_caught": 0.929, "drones_passed": 0.991},
    0.8: {"confusers_caught": 0.884, "drones_passed": 0.993},
    0.9: {"confusers_caught": 0.822, "drones_passed": 0.995},
}
ir_sweep = {
    0.5: {"confusers_caught": 0.956, "drones_passed": 0.987},
    0.6: {"confusers_caught": 0.923, "drones_passed": 0.991},
    0.7: {"confusers_caught": 0.903, "drones_passed": 0.993},
    0.8: {"confusers_caught": 0.858, "drones_passed": 0.994},
    0.9: {"confusers_caught": 0.808, "drones_passed": 0.996},
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for ax, sweep, title in [(ax1, rgb_sweep, "RGB"), (ax2, ir_sweep, "IR")]:
    thrs = sorted(sweep.keys())
    caught = [sweep[t]["confusers_caught"] * 100 for t in thrs]
    passed = [sweep[t]["drones_passed"] * 100 for t in thrs]

    ax.plot(thrs, caught, "o-", color="#e63946", linewidth=2, markersize=8, label="Confusers caught (%)")
    ax.plot(thrs, passed, "s-", color="#2a9d8f", linewidth=2, markersize=8, label="Drones passed (%)")
    ax.axvline(x=0.70, color="#adb5bd", linestyle="--", linewidth=1.5, label="Production threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Rate (%)")
    ax.set_title(f"{title} Confuser Filter — Threshold Sweep")
    ax.legend(fontsize=9, loc="center left")
    ax.set_ylim(78, 101)
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

plt.tight_layout()
fig.savefig(OUT / "4_threshold_sweep.png", dpi=200)
plt.close()
print(f"  -> {OUT / '4_threshold_sweep.png'}")

# ===========================================================
# 5. Binary vs 4-Class FN Comparison Bar Chart
# ===========================================================
print("5. Binary vs 4-class comparison...")
fig, ax = plt.subplots(figsize=(9, 5))

datasets = ["Anti-UAV\nRGB", "Anti-UAV\nIR", "Train/Test\nRGB", "Train/Test\nIR"]
binary_fn = [854, 29, 9080, 1409]
fourclass_fn = [3, 7, 1173, 682]

x = np.arange(len(datasets))
w = 0.35
bars1 = ax.bar(x - w/2, binary_fn, w, label="Binary (old)", color="#e63946", alpha=0.85, edgecolor="white")
bars2 = ax.bar(x + w/2, fourclass_fn, w, label="4-Class (new)", color="#2a9d8f", alpha=0.85, edgecolor="white")

# Value labels
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + max(binary_fn)*0.01,
            f"{int(h):,}", ha="center", va="bottom", fontsize=9, color="#e63946", fontweight="bold")
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + max(binary_fn)*0.01,
            f"{int(h):,}", ha="center", va="bottom", fontsize=9, color="#2a9d8f", fontweight="bold")

# Reduction labels
reductions = ["-99.6%", "-75.9%", "-87.1%", "-51.6%"]
for i, red in enumerate(reductions):
    ax.text(x[i] + w/2, fourclass_fn[i] + max(binary_fn)*0.06,
            red, ha="center", fontsize=8, color="#333", fontstyle="italic")

ax.set_xticks(x)
ax.set_xticklabels(datasets)
ax.set_ylabel("False Rejections (FN — lower is better)")
ax.set_title("Confuser Filter: Drone False Rejections — Binary vs 4-Class")
ax.legend(fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"{int(y):,}"))
plt.tight_layout()
fig.savefig(OUT / "5_binary_vs_4class.png", dpi=200)
plt.close()
print(f"  -> {OUT / '5_binary_vs_4class.png'}")

# ===========================================================
# 6. Pipeline Diagram
# ===========================================================
print("6. Pipeline diagram...")
fig, ax = plt.subplots(figsize=(14, 3))
ax.set_xlim(0, 100)
ax.set_ylim(0, 10)
ax.axis("off")

boxes = [
    (5, "RGB YOLO\n(YOLOv8n)", "#4895ef"),
    (5, "IR YOLO\n(YOLOv8n)", "#4895ef"),
    (30, "XGBoost\nFusion\nClassifier", "#7209b7"),
    (52, "Temporal\nRolling\nWindow", "#f77f00"),
    (74, "Confuser\nFilter\n(MobileNetV3)", "#e63946"),
    (92, "⚠ ALERT", "#2a9d8f"),
]

# Draw boxes
y_single = 5
for x_pos, label, color in boxes:
    if "RGB" in label:
        rect = plt.Rectangle((x_pos, 6.5), 16, 3, facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.9, zorder=2)
        ax.add_patch(rect)
        ax.text(x_pos + 8, 8, label, ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=3)
    elif "IR" in label:
        rect = plt.Rectangle((x_pos, 2), 16, 3, facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.9, zorder=2)
        ax.add_patch(rect)
        ax.text(x_pos + 8, 3.5, label, ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=3)
    else:
        rect = plt.Rectangle((x_pos, 3), 16, 4.5, facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.9, zorder=2)
        ax.add_patch(rect)
        ax.text(x_pos + 8, 5.25, label, ha="center", va="center", fontsize=9, fontweight="bold", color="white", zorder=3)

# Arrows
arrow_style = dict(arrowstyle="-|>", color="#333", lw=1.5)
ax.annotate("", xy=(30, 5.25), xytext=(21, 8), arrowprops=arrow_style)
ax.annotate("", xy=(30, 5.25), xytext=(21, 3.5), arrowprops=arrow_style)
ax.annotate("", xy=(52, 5.25), xytext=(46, 5.25), arrowprops=arrow_style)
ax.annotate("", xy=(74, 5.25), xytext=(68, 5.25), arrowprops=arrow_style)
ax.annotate("", xy=(92, 5.25), xytext=(90, 5.25), arrowprops=arrow_style)

# Labels on arrows
ax.text(48, 6.5, "trust\nlabel", fontsize=7, ha="center", color="#555")
ax.text(70, 6.5, "9/10\nhits", fontsize=7, ha="center", color="#555")
ax.text(91, 6.5, "pass", fontsize=7, ha="center", color="#555")

# Suppress arrow from confuser
ax.annotate("", xy=(82, 1.5), xytext=(82, 3), arrowprops=dict(arrowstyle="-|>", color="#e63946", lw=1.5))
ax.text(82, 0.8, "suppress", fontsize=8, ha="center", color="#e63946", fontstyle="italic")

plt.tight_layout()
fig.savefig(OUT / "6_pipeline_diagram.png", dpi=200, bbox_inches="tight")
plt.close()
print(f"  -> {OUT / '6_pipeline_diagram.png'}")

print(f"\n✅ All plots saved to: {OUT}")
