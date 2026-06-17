"""filter_training_coverage_audit.py — what is UNDER-REPRESENTED in each filter's
training data, by SIZE (and category where knowable)?  ZERO-GPU.

Confuser objects are mined as "every detector fire" (IR_confusers/train has 0
labels; rgb_confusers labels are empty), so the only size signal available
without the detector is:
  - RGB filter TRAINING sizes: _v5_selcom_pure_1x8/training_data.npz (meta-first
    idx[1]=log_area), split by y (drone=1 / confuser=0).
  - Confuser-FIRE sizes (the FPs that actually occur): the cached detector fires
    on confuser surfaces (ir_confusers / rgb_confuser / rgb_bird_confuser).
Size proxy: equivalent square side px = sqrt(exp(log_area)).

Category is reported separately (filename-prefix counts; see the chat summary).
Exact IR confuser category x size needs the detector (GPU) — flagged, not faked.

  py eval/filter_training_coverage_audit.py
"""
from __future__ import annotations
import pickle, sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "eval/results/_offline_pipeline/cache"
RGB_NPZ = REPO / "eval/results/_v5_selcom_pure_1x8/training_data.npz"
IMG = REPO / "docs/analysis/images"; IMG.mkdir(parents=True, exist_ok=True)
OUT_PNG = IMG / "2026-06-17_filter_training_coverage.png"
OUT_JSON = REPO / "docs/analysis/2026-06-17_filter_training_coverage.json"
EDGES = [0, 16, 32, 64, 1e9]; NAMES = ["<16px", "16-32", "32-64", ">=64"]


def to_px(logarea):
    return np.sqrt(np.exp(np.asarray(logarea, dtype=np.float64)))


def bucketize(px):
    out = np.zeros(4, int)
    for i in range(4):
        out[i] = int(((px >= EDGES[i]) & (px < EDGES[i + 1])).sum())
    return out


def pct(b):
    s = b.sum() or 1
    return (100 * b / s).round(1)


def fire_logarea(name):
    p = CACHE / f"{name}.pkl"
    if not p.exists():
        return None
    d = pickle.load(open(p, "rb"))
    la = [f[1] for fr in d["frames"] for f in fr["feats"]]   # meta-first idx1 = log_area
    return np.array(la, dtype=np.float32)


def line(tag, b):
    return f"  {tag:<26} " + " ".join(f"{NAMES[i]}={b[i]:>5}({pct(b)[i]:>4}%)" for i in range(4)) + f"  | n={b.sum()}"


def main():
    rep = {}
    print("=== RGB filter TRAINING sizes (training_data.npz) ===")
    z = np.load(RGB_NPZ); la = z["X"][:, 1]; y = z["y"].astype(int)
    d_b = bucketize(to_px(la[y == 1])); c_b = bucketize(to_px(la[y == 0]))
    print(line("train DRONE", d_b)); print(line("train CONFUSER", c_b))
    rep["rgb_train_drone"] = d_b.tolist(); rep["rgb_train_confuser"] = c_b.tolist()

    print("\n=== Confuser-FIRE sizes (the FPs that occur; from caches) ===")
    fires = {}
    for nm in ["rgb_confuser", "rgb_bird_confuser", "gray_confuser", "ir_confusers"]:
        la_f = fire_logarea(nm)
        if la_f is None or len(la_f) == 0:
            print(f"  {nm}: (none)"); continue
        b = bucketize(to_px(la_f)); fires[nm] = b
        print(line(nm, b)); rep[f"fire_{nm}"] = b.tolist()

    # drone-fire sizes for reference (rgb_dataset_test GT-matched) to anchor "small"
    print("\n=== Reference: rgb_dataset_test real-drone detection sizes ===")
    dt = pickle.load(open(CACHE / "rgb_dataset_test.pkl", "rb"))
    dla = []
    for fr in dt["frames"]:
        if len(fr["gt_boxes"]) == 0:
            continue
        for i, box in enumerate(fr["boxes"]):
            x1, y1, x2, y2 = box
            inter_ok = any(min(box[2], g[2]) > max(box[0], g[0]) and min(box[3], g[3]) > max(box[1], g[1]) for g in fr["gt_boxes"])
            if inter_ok:
                dla.append(fr["feats"][i][1])
    db = bucketize(to_px(np.array(dla))); print(line("rgb_test drone dets", db)); rep["rgb_test_drone_dets"] = db.tolist()

    # ── figure ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(4); w = 0.38
    ax[0].bar(x - w/2, pct(d_b), w, label="train DRONE", color="tab:blue")
    ax[0].bar(x + w/2, pct(c_b), w, label="train CONFUSER", color="tab:orange")
    ax[0].set_xticks(x); ax[0].set_xticklabels(NAMES); ax[0].set_ylabel("% of class")
    ax[0].set_title("(a) RGB filter TRAINING size mix\n(drone vs confuser)"); ax[0].legend(fontsize=8)
    for nm, col in (("rgb_confuser", "tab:green"), ("rgb_bird_confuser", "tab:purple"), ("ir_confusers", "tab:red")):
        if nm in fires:
            ax[1].plot(x, pct(fires[nm]), "-o", label=f"{nm} FPs", color=col)
    ax[1].set_xticks(x); ax[1].set_xticklabels(NAMES); ax[1].set_ylabel("% of fires")
    ax[1].set_title("(b) Confuser-FIRE sizes (the FPs to catch)"); ax[1].legend(fontsize=8)
    fig.suptitle("Filter training coverage by size — what's under-represented", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT_PNG, dpi=130); plt.close(fig)

    OUT_JSON.write_text(json.dumps({"buckets": NAMES, **rep}, indent=2))
    print(f"\nsaved {OUT_PNG}\nsaved {OUT_JSON}")


if __name__ == "__main__":
    main()
