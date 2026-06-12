"""Benchmark the strided global-feature cache.

(1) Speed: time compute_global_features at different max_h, plus the
    near-zero cost of a cache hit.
(2) Accuracy: simulate stride caching on the cached fusion CSV by
    propagating scene features from frame 0 of each stride window to
    frames 1..N-1, then running the production classifier and comparing
    per-class P/R against the no-stride baseline.

Strided across rows (every Mth) for a cheap run; the temporal stride
simulation is independent and runs on every selected row.
"""
from __future__ import annotations
import argparse
import re
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ir_gui"))
from fusion.features import compute_global_features  # noqa: E402

CSV = ROOT / "classifier" / "runs" / "reliability" / "fusion" / "fusion_dataset_v3more.csv"
MODEL = ROOT / "classifier" / "fusion_models" / "scene_aware_v3more_32feat" / "model.joblib"

LABEL_NAMES = {0: "reject_both", 1: "trust_rgb", 2: "trust_ir", 3: "trust_both"}

SCENE_COLS = ["img_mean", "img_std", "img_dynamic_range", "img_entropy",
              "sky_ground_ratio", "edge_density", "blurriness"]
RGB_SCENE = [f"rgb_{c}" for c in SCENE_COLS]
IR_SCENE = [f"ir_{c}" for c in SCENE_COLS]
SEQ_RE = re.compile(r"^(.+?)_f(\d+)$")


# ── Speed ────────────────────────────────────────────────────────────

def bench_speed(reps=200):
    print("\n=== Speed benchmark ===")
    print("Synthetic 1080p grayscale frame, n=", reps)
    img_full = (np.random.rand(1080, 1920) * 255).astype(np.uint8)
    print(f"{'mode':<28} {'mean ms':>10} {'std ms':>10}")

    for label, max_h in [("compute (max_h=720)", 720),
                         ("compute (max_h=480, default)", 480),
                         ("compute (max_h=320)", 320),
                         ("compute (max_h=240)", 240)]:
        # Warmup
        for _ in range(5):
            compute_global_features(img_full, "all", "rgb", max_h=max_h)
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            compute_global_features(img_full, "all", "rgb", max_h=max_h)
            ts.append((time.perf_counter() - t0) * 1000)
        ts = np.array(ts)
        print(f"{label:<28} {ts.mean():>10.2f} {ts.std():>10.2f}")

    # Cache-hit cost: just a dict copy
    cached = compute_global_features(img_full, "all", "rgb", max_h=480)
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _ = dict(cached)  # what cache.get effectively returns
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    print(f"{'cache hit (dict copy)':<28} {ts.mean():>10.4f} {ts.std():>10.4f}")


# ── Accuracy: simulate temporal stride caching ──────────────────────

def derive_seq_and_frame(stems):
    seqs = []
    frames = []
    for s in stems:
        m = SEQ_RE.match(str(s))
        if m:
            seqs.append(m.group(1))
            frames.append(int(m.group(2)))
        else:
            seqs.append(str(s))
            frames.append(0)
    return seqs, frames


def apply_stride_cache(df: pd.DataFrame, stride: int,
                        scene_cut_delta: float = 15.0) -> pd.DataFrame:
    """For each (sequence, modality), propagate scene features within
    stride windows, with a scene-cut probe on img_mean."""
    if stride <= 1:
        return df.copy()

    out = df.copy()
    out = out.sort_values(["_seq", "_frame"]).reset_index(drop=True)

    for cols, mean_col in [(RGB_SCENE, "rgb_img_mean"),
                           (IR_SCENE, "ir_img_mean")]:
        # Build the cached version
        for seq, sub_idx in out.groupby("_seq").groups.items():
            sub_idx = list(sub_idx)
            cached_vals = None
            cached_mean = None
            counter = 0
            for i in sub_idx:
                row_mean = float(out.at[i, mean_col])
                # On stride boundary or scene cut → "recompute" (keep real values)
                on_stride = (counter % stride == 0)
                cut = (cached_mean is not None
                       and abs(row_mean - cached_mean) > scene_cut_delta)
                if on_stride or cached_vals is None or cut:
                    cached_vals = {c: float(out.at[i, c]) for c in cols}
                    cached_mean = row_mean
                else:
                    # Reuse cached values (overwrite the row's real values)
                    for c in cols:
                        out.at[i, c] = cached_vals[c]
                counter += 1
    return out


def per_class_prf(y_true, y_pred):
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2, 3], zero_division=0)
    return {LABEL_NAMES[i]: {"P": float(p[i]), "R": float(r[i]),
                              "F1": float(f[i]), "n": int(s[i])}
            for i in range(4)}


def bench_accuracy(stride_list, row_stride=10, scene_cut_delta=15.0):
    print("\n=== Accuracy benchmark (temporal stride simulation) ===")
    print(f"Loading {CSV.name} ...")
    df = pd.read_csv(CSV)
    df = df.iloc[::row_stride].reset_index(drop=True)
    df["_seq"], df["_frame"] = derive_seq_and_frame(df["base_stem"].values)
    print(f"  {len(df):,} rows after row-stride={row_stride}, "
          f"{df['_seq'].nunique():,} sequences")

    bundle = joblib.load(MODEL)
    model = bundle["model"]
    feat_cols = bundle["features"]
    print(f"  model: {len(feat_cols)} features")

    y_true = df["trust_label"].values

    # Baseline: no temporal caching
    y_base = model.predict(df[feat_cols].values.astype(np.float32))
    base_acc = float((y_base == y_true).mean())
    base_per = per_class_prf(y_true, y_base)

    print(f"\nBaseline (stride=1, every-frame compute): acc={base_acc:.4f}")

    rows = [("stride", "acc", "Δacc", "P_rgb", "R_rgb", "F1_rgb",
             "P_ir", "R_ir", "F1_ir", "P_both", "R_both", "F1_both",
             "P_rej", "R_rej", "F1_rej")]
    rows.append((1, base_acc, 0.0,
                 base_per["trust_rgb"]["P"], base_per["trust_rgb"]["R"],
                 base_per["trust_rgb"]["F1"],
                 base_per["trust_ir"]["P"], base_per["trust_ir"]["R"],
                 base_per["trust_ir"]["F1"],
                 base_per["trust_both"]["P"], base_per["trust_both"]["R"],
                 base_per["trust_both"]["F1"],
                 base_per["reject_both"]["P"], base_per["reject_both"]["R"],
                 base_per["reject_both"]["F1"]))

    for stride in stride_list:
        sim = apply_stride_cache(df, stride, scene_cut_delta=scene_cut_delta)
        # apply_stride_cache sorted by _seq, _frame; align y_true to that
        y_true_sorted = sim["trust_label"].values
        y_pred = model.predict(sim[feat_cols].values.astype(np.float32))
        acc = float((y_pred == y_true_sorted).mean())
        per = per_class_prf(y_true_sorted, y_pred)
        rows.append((stride, acc, acc - base_acc,
                     per["trust_rgb"]["P"], per["trust_rgb"]["R"],
                     per["trust_rgb"]["F1"],
                     per["trust_ir"]["P"], per["trust_ir"]["R"],
                     per["trust_ir"]["F1"],
                     per["trust_both"]["P"], per["trust_both"]["R"],
                     per["trust_both"]["F1"],
                     per["reject_both"]["P"], per["reject_both"]["R"],
                     per["reject_both"]["F1"]))

    # Print compact tables
    print(f"\n{'stride':>6} {'acc':>7} {'Δacc':>7}")
    for r in rows[1:]:
        print(f"{r[0]:>6} {r[1]:>7.4f} {r[2]:>+7.4f}")

    print(f"\n{'class':<14} {'stride':>7} {'P':>7} {'R':>7} {'F1':>7}")
    for cls_name in ["reject_both", "trust_rgb", "trust_ir", "trust_both"]:
        for r in rows[1:]:
            stride = r[0]
            if cls_name == "reject_both":
                p, rec, f = r[12], r[13], r[14]
            elif cls_name == "trust_rgb":
                p, rec, f = r[3], r[4], r[5]
            elif cls_name == "trust_ir":
                p, rec, f = r[6], r[7], r[8]
            else:
                p, rec, f = r[9], r[10], r[11]
            print(f"{cls_name:<14} {stride:>7} {p:>7.4f} {rec:>7.4f} {f:>7.4f}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row-stride", type=int, default=10,
                    help="Subsample rows for cheap run (default 10)")
    ap.add_argument("--strides", type=str, default="2,5,10,20",
                    help="Comma-separated temporal strides to test")
    ap.add_argument("--scene-cut-delta", type=float, default=15.0)
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--skip-accuracy", action="store_true")
    args = ap.parse_args()

    if not args.skip_speed:
        bench_speed()
    if not args.skip_accuracy:
        strides = [int(x) for x in args.strides.split(",")]
        bench_accuracy(strides, row_stride=args.row_stride,
                       scene_cut_delta=args.scene_cut_delta)


if __name__ == "__main__":
    main()
