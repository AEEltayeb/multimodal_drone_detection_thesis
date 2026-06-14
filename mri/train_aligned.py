"""
mri.train_aligned — the SYNTHESIS verifier: thermal-deployable, recall-safe AND
confuser-rich, by harvesting confusers from the abundant RGB->gray corpus and
domain-aligning them into thermal feature space.

Why this works (proven by mri.modality_align): the gray<->thermal feature gap is
a per-feature affine offset; per-modality z-score rescues gray->thermal transfer
from 0.50 to 0.919. So:
  positives = thermal drones (native, abundant; recall-safety via drone diversity)
  negatives = grayscale-harvested confusers (abundant) + the few real thermal ones
Each modality is z-scored to ITS OWN mean/std before concatenation, so the MLP
learns drone-vs-confuser in the shared aligned space, NOT the modality offset.

Deployment is on THERMAL. To keep the checkpoint loadable by the unchanged
MLPVerifier (one affine scaler applied to raw features), we COMPOSE the thermal
per-modality z-score with MLPWrapper's internal scaler into a single affine and
save that — so raw_thermal -> composed_scaler -> net reproduces training exactly.

CBAM is held OUT of training -> validate on it to test whether gray-harvested
confusers generalize to novel thermal aerial confusers.
"""
from __future__ import annotations
import argparse, json, time, sys
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from sklearn.preprocessing import StandardScaler
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from mri.extract import FeatureExtractor                 # noqa: E402
from mri.modality_align import mine_multi                # noqa: E402 (reuse miner)
from mri.classifier import MLPWrapper                    # noqa: E402

V3B = str(REPO / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt")
OUT = REPO / "mri" / "results" / "ir_aligned"
CONF_PRE = ("IR_BIRD_", "IR_AIRPLANE_", "IR_HELICOPTER_")

# (dir, prefix, imgsz) per group. CBAM is deliberately ABSENT (held out for eval).
GROUPS = {
    ("thermal", "drone"): [
        ("G:/drone/svanstrom_paired/IR/images", "IR_DRONE_", 1280),
        ("G:/drone/Anti-UAV-RGBT_yolo_converted/val/IR/images", "", 640),
        ("G:/drone/IR_dset_final/train/images", "", 640),
        ("G:/drone/IR_video_ir_dataset/train/images", "IR_DRONE_", 640),
        ("models/ir/corrective_finetune/dataset_v3/train/images", "", 640)],
    ("thermal", "conf"): [
        ("G:/drone/svanstrom_paired/IR/images", CONF_PRE, 1280),
        ("G:/drone/IR_video_ir_dataset/train/images", CONF_PRE, 640)],
    ("gray", "drone"): [
        ("G:/drone/svanstrom_paired/RGB/images", "IR_DRONE_", 1280),
        ("G:/drone/Anti-UAV-RGBT_yolo_converted/val/RGB/images", "", 640),
        ("G:/drone/dataset/dataset/images/train", "", 640)],
    ("gray", "conf"): [
        ("G:/drone/rgb_confusers_merged/images/train", "", 640),
        ("G:/drone/RGB_video_rgb_dataset/train/images",
         ("V_BIRD_", "V_AIRPLANE_", "V_HELICOPTER_"), 640),
        ("G:/drone/svanstrom_paired/RGB/images", CONF_PRE, 1280)],
}
CAP = {("thermal", "drone"): 9000, ("thermal", "conf"): 1800,
       ("gray", "drone"): 5000, ("gray", "conf"): 7000}
WEIGHT = {"thermal": 1.5, "gray": 1.0}   # bias the boundary toward the deploy modality


def main(no_gray: bool = False, out: Path = OUT):
    out.mkdir(parents=True, exist_ok=True)
    mode = "THERMAL-ONLY (no grayscale confusers; A/B counterfactual)" if no_gray else \
           "thermal + grayscale-harvested confusers (production)"
    print(f"== train_aligned [{mode}] ==\n  detector: {V3B}\n  out: {out}")
    yolo = YOLO(V3B)
    ex = FeatureExtractor(yolo, layers=("p3", "p5"), grids={"p3": (2, 2), "p5": (1, 1)})

    data = {}
    for (mod, kl), specs in GROUPS.items():
        if no_gray and mod == "gray":
            continue                   # A/B: drop the grayscale-harvested groups entirely
        t0 = time.time()
        X = mine_multi(ex, specs, grayscale=(mod == "gray"),
                       cap=CAP[(mod, kl)], stride=2, conf=0.25, device="cuda")
        data[(mod, kl)] = X            # FULL 517-D (deployable via MLPVerifier)
        print(f"  {mod:7} {kl:5} n={len(X):6}  ({time.time()-t0:.0f}s)")
    ex.close()

    # per-modality z-score, then concat into the shared aligned space (thermal only if --no-gray)
    Xt = np.vstack([data[("thermal", "drone")], data[("thermal", "conf")]])
    sct = StandardScaler().fit(Xt)
    Xtz = sct.transform(Xt)
    yt = np.r_[np.ones(len(data[("thermal", "drone")])), np.zeros(len(data[("thermal", "conf")]))]
    wt = np.full(len(yt), WEIGHT["thermal"])
    if no_gray:
        X = Xtz.astype(np.float32); y = yt.astype(np.float32); w = wt.astype(np.float32)
        print(f"\n  train set {X.shape}: {int(y.sum())} drone / {int((1-y).sum())} confuser "
              f"(thermal-only; grayscale groups dropped)")
    else:
        Xg = np.vstack([data[("gray", "drone")], data[("gray", "conf")]])
        scg = StandardScaler().fit(Xg)
        Xgz = scg.transform(Xg)
        yg = np.r_[np.ones(len(data[("gray", "drone")])), np.zeros(len(data[("gray", "conf")]))]
        wg = np.full(len(yg), WEIGHT["gray"])
        X = np.vstack([Xtz, Xgz]).astype(np.float32)
        y = np.r_[yt, yg].astype(np.float32)
        w = np.r_[wt, wg].astype(np.float32)
        print(f"\n  train set {X.shape}: {int(y.sum())} drone / {int((1-y).sum())} confuser "
              f"(thermal {len(yt)}, gray {len(yg)})")

    mlp = MLPWrapper(input_dim=X.shape[1], device="cuda", epochs=120)
    mlp.fit(X, y, sample_weight=w)

    # Compose thermal per-modality z-score (sct) with MLPWrapper's internal scaler
    # so deploy on RAW thermal reproduces training: raw -> sct -> mlp.scaler -> net.
    m_t, s_t = sct.mean_.astype(np.float32), sct.scale_.astype(np.float32)
    m_w, s_w = mlp.scaler.mean_.astype(np.float32), mlp.scaler.scale_.astype(np.float32)
    base = {
        "state_dict": mlp.net.state_dict(),
        "input_dim": int(mlp.input_dim), "hidden_dims": list(mlp.hidden_dims),
        "threshold": 0.5, "cv_f1": -1.0, "cv_std": 0.0,
        "use_batchnorm": mlp.use_batchnorm, "dropout": mlp.dropout,
        "feature_schema": ex.schema.to_dict(),
        "metadata_order": ["conf", "log_area", "aspect", "rel_cx", "rel_cy"],
        "p3_grid": [2, 2], "p5_grid": [1, 1],
    }
    (out / "classifiers").mkdir(exist_ok=True)
    tnote = ("aligned verifier (THERMAL deploy, NO-GRAY A/B). thermal drones + thermal confusers "
             "ONLY; scaler = compose(thermal_zscore, mlp_scaler). CBAM held out.") if no_gray else \
            ("aligned verifier (THERMAL deploy). thermal drones + per-modality-z gray confusers; "
             "scaler = compose(thermal_zscore, mlp_scaler). CBAM held out.")
    torch.save({**base,
                "scaler_mean": torch.from_numpy(m_t + s_t * m_w),
                "scaler_scale": torch.from_numpy(s_t * s_w),
                "note": tnote},
               out / "classifiers" / "mlp_aligned.pt")
    if not no_gray:
        # Same net, gray deploy scaler (per-modality z-score composed with mlp.scaler):
        #   gray: raw_gray -> scg -> mlp.scaler -> net   (deploy on grayscale-RGB)
        m_g, s_g = scg.mean_.astype(np.float32), scg.scale_.astype(np.float32)
        torch.save({**base,
                    "scaler_mean": torch.from_numpy(m_g + s_g * m_w),
                    "scaler_scale": torch.from_numpy(s_g * s_w),
                    "note": "aligned verifier (GRAYSCALE deploy). SAME net, gray per-modality scaler. "
                            "Use with --grayscale-input."},
                   out / "classifiers" / "mlp_aligned_gray.pt")
    (out / "train_meta.json").write_text(json.dumps({
        "no_gray": no_gray,
        "counts": {f"{m}_{k}": int(len(data[(m, k)])) for (m, k) in data},
        "n_train": int(len(X)), "weight": WEIGHT,
    }, indent=2))
    print(f"\n  saved {out/'classifiers'/'mlp_aligned.pt'}")
    print("  validate on CBAM-valid (held out of training):")
    print(f"    py -m mri.cli --yolo {V3B} --config mri/configs/ab_cbam_heldout.yaml \\")
    print(f"       --holdout-eval {out/'classifiers'/'mlp_aligned.pt'} \\")
    print("       --conf 0.40 --mlp-thr 0.05 --match-rule iop --imgsz 640 --device cuda --out mri/results/ab_nogray_cbam")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the aligned IR confuser filter (synthesis verifier).")
    ap.add_argument("--no-gray", action="store_true",
                    help="A/B counterfactual: drop grayscale-harvested confusers; thermal drones + "
                         "thermal confusers only (CBAM still held out). Writes to mri/results/ir_aligned_nogray.")
    ap.add_argument("--out", default=None, help="Output dir (default mri/results/ir_aligned[_nogray]).")
    a = ap.parse_args()
    out_dir = Path(a.out) if a.out else (REPO / "mri" / "results" / ("ir_aligned_nogray" if a.no_gray else "ir_aligned"))
    main(no_gray=a.no_gray, out=out_dir)
