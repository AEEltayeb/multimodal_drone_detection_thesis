"""
mri.modality_probe — Is a class's signature the SAME in thermal vs grayscale?

Population-level (unpaired) test of whether the IR detector v3b represents an
object class the same way whether the input is real thermal or grayscale-RGB.
If yes, confusers can be HARVESTED from the abundant RGB corpus (RGB->gray) and
still be representative of thermal confusers — which makes the data problem easy.

Method (no pairing needed — we compare distributions, not instances):
  Populations from Svanstrom (category by filename prefix, both modalities present):
    thermal  = svanstrom_paired/IR/images   (real infrared)
    gray     = svanstrom_paired/RGB/images   (fed BGR->gray->3ch, the deploy op)
  Mine v3b features per (modality x category), then on the APPEARANCE subspace
  (yolo-only p3/p5; meta=conf/geometry are detection-stats, not signature):

  1. Modality separability  — per class, can a linear model tell thermal from
     gray? ~0.5 = modality-invariant (signatures coincide); ~1.0 = a real gap.
  2. Centroid distance      — cosine dist between class means across modality,
     vs a within-modality split-half noise floor, with DRONE as the reference
     (drones provably partially transfer, so their gap calibrates "close enough").
  3. Top-neuron overlap     — do the same channels carry the class? Jaccard of
     top-k mean-activation neurons + Pearson r of the mean activation vectors.
  4. Cross-modal transfer   — THE decision metric. Train drone-vs-confuser on the
     gray population, test catch on thermal (and reverse); compare to the
     within-modality CV ceiling. High transfer => grayscale harvest is valid.

Caveat (unpaired): a gap can be modality OR scene/dataset bias. We anchor to the
drone reference and lean on transfer (robust to centroid quirks).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold

REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO))
from mri.extract import FeatureExtractor  # noqa: E402

SVAN_IR = Path("G:/drone/svanstrom_paired/IR/images")
SVAN_RGB = Path("G:/drone/svanstrom_paired/RGB/images")
CATS = {"drone": "IR_DRONE_", "bird": "IR_BIRD_",
        "airplane": "IR_AIRPLANE_", "helicopter": "IR_HELICOPTER_"}
CONFUSER_CATS = ["bird", "airplane", "helicopter"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")
IR_DETECTOR = REPO / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt"


def mine(ex, img_dir, prefix, grayscale, cap, stride, imgsz, conf, device):
    imgs = sorted(p for p in img_dir.iterdir()
                  if p.suffix.lower() in IMG_EXTS and p.name.startswith(prefix))[::stride]
    feats = []
    for p in imgs:
        if len(feats) >= cap:
            break
        im = cv2.imread(str(p))
        if im is None:
            continue
        if grayscale:
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            im = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        ih, iw = im.shape[:2]
        ex.hook.clear()
        r = ex.model.predict(im, imgsz=imgsz, conf=conf, verbose=False, device=device)[0]
        if r.boxes is None:
            continue
        for i in range(len(r.boxes)):
            if len(feats) >= cap:
                break
            box = tuple(r.boxes.xyxy[i].cpu().numpy().tolist())
            feats.append(ex.extract_one(box, float(r.boxes.conf[i]), (ih, iw)))
    dim = ex.schema.total_dim
    return np.array(feats, np.float32) if feats else np.zeros((0, dim), np.float32)


def _cos_dist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(1.0 - np.dot(a, b) / (na * nb))


def _sep_acc(Xa, Xb, seed=42):
    """5-fold CV accuracy of a linear model separating two feature sets."""
    if len(Xa) < 10 or len(Xb) < 10:
        return float("nan")
    X = np.vstack([Xa, Xb]); y = np.r_[np.zeros(len(Xa)), np.ones(len(Xb))]
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    k = min(5, int(min(len(Xa), len(Xb))))
    if k < 2:
        return float("nan")
    return float(cross_val_score(clf, Xs, y,
                 cv=StratifiedKFold(k, shuffle=True, random_state=seed),
                 scoring="accuracy").mean())


def main():
    ap = argparse.ArgumentParser(description="thermal-vs-grayscale signature probe")
    ap.add_argument("--yolo", default=str(IR_DETECTOR))
    ap.add_argument("--cap", type=int, default=800, help="max dets per (modality,category)")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--imgsz", type=int, default=1280)   # Svanstrom rule
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--topk", type=int, default=30, help="top neurons for overlap")
    ap.add_argument("--out", default=str(REPO / "mri" / "results" / "modality_probe"))
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    print(f"== modality probe ==\n  detector: {args.yolo}\n  cap={args.cap} stride={args.stride} imgsz={args.imgsz} conf={args.conf}")
    yolo = YOLO(args.yolo)
    ex = FeatureExtractor(yolo, layers=("p3", "p5"), grids={"p3": (2, 2), "p5": (1, 1)})
    meta_dim = ex.schema.meta_dim

    data = {}
    for mod, (d, gray) in {"thermal": (SVAN_IR, False), "gray": (SVAN_RGB, True)}.items():
        if not d.exists():
            print(f"  FATAL: {d} missing"); return 1
        for cat, pre in CATS.items():
            t0 = time.time()
            X = mine(ex, d, pre, gray, args.cap, args.stride, args.imgsz, args.conf, args.device)
            data[(mod, cat)] = X
            print(f"  {mod:7} {cat:10} n={len(X):5}  ({time.time()-t0:.0f}s)")
    ex.close()

    def yolo_only(X):
        return X[:, meta_dim:] if len(X) else X

    report = {"detector": args.yolo, "cap": args.cap, "imgsz": args.imgsz,
              "counts": {f"{m}_{c}": int(len(data[(m, c)])) for m in ("thermal", "gray") for c in CATS},
              "modality_separability": {}, "centroid": {}, "neuron_overlap": {}, "transfer": {}}

    # 1 + 2 + 3 — per category
    print("\n-- per-class signature (yolo-only appearance subspace) --")
    print(f"  {'class':10} {'sep_acc':>8} {'cos_dist':>9} {'noise_fl':>9} {'neuron_J':>9} {'act_r':>7}")
    rng = np.random.RandomState(0)
    for cat in CATS:
        Xth, Xg = yolo_only(data[("thermal", cat)]), yolo_only(data[("gray", cat)])
        sep = _sep_acc(Xth, Xg)
        cos = nf = jac = r = float("nan")
        if len(Xth) >= 10 and len(Xg) >= 10:
            mth, mg = Xth.mean(0), Xg.mean(0)
            cos = _cos_dist(mth, mg)
            # within-thermal split-half noise floor
            idx = rng.permutation(len(Xth)); h = len(Xth) // 2
            nf = _cos_dist(Xth[idx[:h]].mean(0), Xth[idx[h:]].mean(0))
            kth = set(np.argsort(np.abs(mth))[-args.topk:])
            kg = set(np.argsort(np.abs(mg))[-args.topk:])
            jac = len(kth & kg) / len(kth | kg)
            r = float(np.corrcoef(mth, mg)[0, 1])
        report["modality_separability"][cat] = sep
        report["centroid"][cat] = {"cos_dist": cos, "noise_floor": nf}
        report["neuron_overlap"][cat] = {"jaccard_top%d" % args.topk: jac, "act_corr": r}
        print(f"  {cat:10} {sep:>8.3f} {cos:>9.3f} {nf:>9.3f} {jac:>9.3f} {r:>7.3f}")

    # 4 — cross-modal transfer (drone-vs-confuser)
    def build(mod):
        Xs, ys = [], []
        for cat in CATS:
            X = yolo_only(data[(mod, cat)])
            if not len(X):
                continue
            Xs.append(X); ys.append(np.full(len(X), 1 if cat == "drone" else 0))
        return (np.vstack(Xs), np.concatenate(ys)) if Xs else (np.zeros((0, 1)), np.zeros(0))

    Xg, yg = build("gray"); Xt, yt = build("thermal")
    print("\n-- cross-modal transfer: drone-vs-confuser (AUROC) --")
    if len(Xg) and len(Xt) and len(set(yg)) == 2 and len(set(yt)) == 2:
        sc = StandardScaler().fit(Xg)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(Xg), yg)
        auc_g2t = roc_auc_score(yt, clf.decision_function(sc.transform(Xt)))
        sc2 = StandardScaler().fit(Xt)
        clf2 = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc2.transform(Xt), yt)
        auc_t2g = roc_auc_score(yg, clf2.decision_function(sc2.transform(Xg)))
        # within-modality CV ceilings
        ceil_t = float(cross_val_score(LogisticRegression(max_iter=2000, class_weight="balanced"),
                       StandardScaler().fit_transform(Xt), yt, cv=5, scoring="roc_auc").mean())
        ceil_g = float(cross_val_score(LogisticRegression(max_iter=2000, class_weight="balanced"),
                       StandardScaler().fit_transform(Xg), yg, cv=5, scoring="roc_auc").mean())
        report["transfer"] = {"gray_to_thermal": auc_g2t, "thermal_to_gray": auc_t2g,
                              "ceiling_thermal": ceil_t, "ceiling_gray": ceil_g}
        print(f"  train GRAY -> test THERMAL : AUROC {auc_g2t:.3f}   (thermal ceiling {ceil_t:.3f})")
        print(f"  train THERMAL -> test GRAY : AUROC {auc_t2g:.3f}   (gray ceiling {ceil_g:.3f})")
        retain = auc_g2t / ceil_t if ceil_t else 0
        verdict = ("SIGNATURES ALIGN — grayscale harvest is representative"
                   if retain >= 0.92 and np.nanmean(list(report["modality_separability"].values())) < 0.85
                   else "MODALITY GAP — grayscale is NOT a clean substitute for thermal")
        report["verdict"] = verdict
        print(f"\n  VERDICT: {verdict}")
    else:
        print("  insufficient data for transfer test")

    (out / "modality_probe.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  wrote {out/'modality_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
