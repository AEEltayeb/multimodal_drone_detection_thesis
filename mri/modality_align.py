"""
mri.modality_align — can we RESCUE gray->thermal transfer by domain alignment?

The probe found: same neurons fire across modality (Jaccard 0.71-0.88, act-corr
0.93-0.99) but naive transfer is chance (AUROC 0.53) because of a consistent
MODALITY OFFSET in absolute feature values. If that offset is a simple per-feature
shift/scale (or covariance shift), aligning it should recover transfer — which
would mean grayscale-mined confusers CAN substitute for scarce thermal confusers.

Alignment is estimated from the DRONE class only (svan + antiuav, both modalities;
population-level, no instance pairing), then APPLIED to confuser features, then we
test drone-vs-confuser transfer:
    raw           — train gray, test thermal (baseline; expect ~0.53)
    permod_z      — z-score EACH modality to its own mean/std, then transfer
    coral         — CORAL: match gray covariance+mean to thermal, then transfer
A jump toward the within-modality ceiling => the gap is alignable => harvest works.
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from mri.extract import FeatureExtractor          # noqa: E402
from mri.modality_probe import mine               # noqa: E402  (reuse miner)

IR_DETECTOR = REPO / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt"

# (dir, prefix, imgsz) lists per (modality, class). antiuav RGBT is paired; svan
# RGB/IR share the prefix scheme. grayscale flag is set by modality at mine time.
SVAN_IR, SVAN_RGB = "G:/drone/svanstrom_paired/IR/images", "G:/drone/svanstrom_paired/RGB/images"
AUV_IR = "G:/drone/Anti-UAV-RGBT_yolo_converted/val/IR/images"
AUV_RGB = "G:/drone/Anti-UAV-RGBT_yolo_converted/val/RGB/images"
RGB_CONF = "G:/drone/rgb_confusers_merged/images/train"
CONF_PRE = ("IR_BIRD_", "IR_AIRPLANE_", "IR_HELICOPTER_")

SOURCES = {
    ("thermal", "drone"): [(SVAN_IR, "IR_DRONE_", 1280), (AUV_IR, "", 640)],
    ("gray",    "drone"): [(SVAN_RGB, "IR_DRONE_", 1280), (AUV_RGB, "", 640)],
    ("thermal", "conf"):  [(SVAN_IR, CONF_PRE, 1280)],
    ("gray",    "conf"):  [(SVAN_RGB, CONF_PRE, 1280), (RGB_CONF, "", 640)],
}


def mine_multi(ex, specs, grayscale, cap, stride, conf, device):
    parts = []
    per = max(200, cap // max(1, len(specs)))
    for d, pre, imgsz in specs:
        if not Path(d).exists():
            print(f"    [skip missing source: {d}]")
            continue
        prefixes = pre if isinstance(pre, tuple) else (pre,)
        for p in prefixes:
            X = mine(ex, Path(d), p, grayscale, per, stride, imgsz, conf, device)
            if len(X):
                parts.append(X)
    return np.vstack(parts) if parts else np.zeros((0, ex.schema.total_dim), np.float32)


def _coral(Xs, Xt, eps=1e-3):
    """Align source Xs to target Xt (CORAL: whiten src cov, recolor to tgt)."""
    ms, mt = Xs.mean(0), Xt.mean(0)
    Cs = np.cov((Xs - ms).T) + eps * np.eye(Xs.shape[1])
    Ct = np.cov((Xt - mt).T) + eps * np.eye(Xt.shape[1])
    def msqrt(C, p):
        w, V = np.linalg.eigh(C); w = np.clip(w, 1e-8, None)
        return (V * (w ** p)) @ V.T
    A = msqrt(Cs, -0.5) @ msqrt(Ct, 0.5)
    return (Xs - ms) @ A + mt


def _auc(Xtr, ytr, Xte, yte):
    clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
    return float(roc_auc_score(yte, clf.decision_function(Xte)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yolo", default=str(IR_DETECTOR))
    ap.add_argument("--cap", type=int, default=1400, help="max dets per (modality,class)")
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(REPO / "mri" / "results" / "modality_align"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print(f"== modality align ==\n  detector: {args.yolo}\n  cap={args.cap} stride={args.stride}")
    yolo = YOLO(args.yolo)
    ex = FeatureExtractor(yolo, layers=("p3", "p5"), grids={"p3": (2, 2), "p5": (1, 1)})
    md = ex.schema.meta_dim

    data = {}
    for (mod, kl), specs in SOURCES.items():
        t0 = time.time()
        X = mine_multi(ex, specs, grayscale=(mod == "gray"),
                       cap=args.cap, stride=args.stride, conf=args.conf, device=args.device)
        data[(mod, kl)] = X[:, md:] if len(X) else X   # yolo-only appearance
        print(f"  {mod:7} {kl:5} n={len(X):5}  ({time.time()-t0:.0f}s)")
    ex.close()
    np.savez_compressed(out / "features.npz",
                        **{f"{m}_{k}": data[(m, k)] for (m, k) in data})

    # drone-vs-confuser sets per modality
    def setof(mod):
        Xd, Xc = data[(mod, "drone")], data[(mod, "conf")]
        X = np.vstack([Xd, Xc]); y = np.r_[np.ones(len(Xd)), np.zeros(len(Xc))]
        return X, y
    Xg, yg = setof("gray"); Xt, yt = setof("thermal")
    print(f"\n  gray set {Xg.shape} ({int(yg.sum())}d/{int((1-yg).sum())}c)  "
          f"thermal set {Xt.shape} ({int(yt.sum())}d/{int((1-yt).sum())}c)")

    res = {}
    # within-modality ceilings (5-fold CV AUROC)
    res["ceiling_thermal"] = float(cross_val_score(
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        StandardScaler().fit_transform(Xt), yt, cv=5, scoring="roc_auc").mean())
    res["ceiling_gray"] = float(cross_val_score(
        LogisticRegression(max_iter=2000, class_weight="balanced"),
        StandardScaler().fit_transform(Xg), yg, cv=5, scoring="roc_auc").mean())

    # 1) RAW transfer (single scaler fit on source)
    sc = StandardScaler().fit(Xg)
    res["raw_gray_to_thermal"] = _auc(sc.transform(Xg), yg, sc.transform(Xt), yt)

    # 2) PER-MODALITY z-score (each modality to its own mean/std)
    scg, sct = StandardScaler().fit(Xg), StandardScaler().fit(Xt)
    res["permod_gray_to_thermal"] = _auc(scg.transform(Xg), yg, sct.transform(Xt), yt)

    # 3) CORAL (align gray->thermal in raw feature space, then standardize by target)
    Xg_al = _coral(Xg, Xt)
    res["coral_gray_to_thermal"] = _auc(sct.transform(Xg_al), yg, sct.transform(Xt), yt)

    # drone-class centroid distance gray vs thermal (alignment difficulty gauge)
    dg, dt = data[("gray", "drone")].mean(0), data[("thermal", "drone")].mean(0)
    res["drone_centroid_cos"] = float(1 - np.dot(dg, dt) / (np.linalg.norm(dg)*np.linalg.norm(dt)))

    print("\n-- gray->thermal transfer AUROC (drone-vs-confuser) --")
    print(f"  within-modality ceilings : thermal {res['ceiling_thermal']:.3f}  gray {res['ceiling_gray']:.3f}")
    print(f"  raw (no alignment)       : {res['raw_gray_to_thermal']:.3f}")
    print(f"  per-modality z-score     : {res['permod_gray_to_thermal']:.3f}")
    print(f"  CORAL alignment          : {res['coral_gray_to_thermal']:.3f}")
    best = max(res['permod_gray_to_thermal'], res['coral_gray_to_thermal'])
    verdict = ("ALIGNABLE — grayscale confusers can substitute for thermal after alignment"
               if best >= 0.85 else
               "PARTIAL — alignment helps but doesn't fully close the gap"
               if best >= 0.70 else
               "NOT ALIGNABLE — the modality gap is not a simple shift; harvest won't transfer")
    res["verdict"] = verdict
    print(f"\n  VERDICT: {verdict}")
    (out / "modality_align.json").write_text(json.dumps(res, indent=2))
    print(f"  wrote {out/'modality_align.json'}")


if __name__ == "__main__":
    main()
