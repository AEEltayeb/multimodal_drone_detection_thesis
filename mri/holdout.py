"""
mri.holdout — held-out DEPLOYMENT eval of a trained verifier.

MRI's built-in verdict uses in-pool out-of-fold CV, which is optimistic: it
can't see the OOD-drone recall loss a verifier inflicts at deployment (this is
exactly what hid the IR-MLP problem — CV-F1 0.987 looked shippable). This module
is the honest gate: it runs the detector + the *shipped* verifier (V5 MLP and/or
patch CNN) across held-out surfaces and reports, per surface, bare vs verifier
P/R/F1 and the recall cost per FP removed.

Drone surfaces (pos, with main_class GT) measure recall cost; confuser surfaces
(neg, no GT) measure FP reduction. The gate: ship only if FP drops on confuser
surfaces with ~0 recall loss on drone surfaces.

Reuses the production inference path (classifier.mlp_verifier.MLPVerifier +
classifier.patch_verifier.PatchVerifier) so the eval reflects deployment exactly.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from metrics import score_detections, compute_prf  # eval/ on sys.path via cli
from .datasets import DatasetSpec, resolve_labels_dir
from .scan import _read_gt


def _score_branch(kept, gt, rule):
    return score_detections(kept, gt, rule=rule, iou_thr=0.5, iop_thr=0.5)


def run_holdout(yolo_path: str, specs: list[DatasetSpec], mlp_weights: str,
                patch_weights: str | None, conf: float, mlp_thr: float,
                patch_thr: float, device: str, out_dir: Path,
                grayscale: bool = False) -> int:
    """Evaluate bare vs MLP (+ optional patch) per held-out surface."""
    from classifier.mlp_verifier import MLPVerifier, DetectInputHook
    print(f"== MRI holdout eval ==\n  detector: {yolo_path}\n  mlp: {mlp_weights}"
          f"\n  patch: {patch_weights or '(none)'}\n  conf={conf} mlp_thr={mlp_thr} patch_thr={patch_thr}")

    yolo = YOLO(yolo_path)
    hook = DetectInputHook(); hook.register(yolo)
    mlp = MLPVerifier(mlp_weights, device)
    patch = None
    if patch_weights and Path(patch_weights).exists():
        from classifier.patch_verifier import PatchVerifier
        patch = PatchVerifier(patch_weights, device)
    elif patch_weights:
        print(f"  WARN: patch weights not found ({patch_weights}); skipping patch branch")

    results = []
    for spec in specs:
        imgs = spec.list_images()
        if not imgs:
            print(f"  SKIP {spec.name}: no images at {spec.path}"); continue
        labels_dir = resolve_labels_dir(spec.path) if spec.has_gt else None
        bare = dict(tp=0, fp=0, fn=0); mlpf = dict(tp=0, fp=0, fn=0); ptch = dict(tp=0, fp=0, fn=0)
        t0 = time.time()
        for ip in imgs:
            im = cv2.imread(str(ip))
            if im is None:
                continue
            if grayscale:
                im = cv2.cvtColor(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
            ih, iw = im.shape[:2]
            gt = _read_gt(labels_dir, ip.stem, ih, iw, spec.main_class) if labels_dir else []
            hook.clear()
            r = yolo.predict(im, imgsz=spec.imgsz, conf=conf, verbose=False, device=device)[0]
            dets, raw = [], []
            if r.boxes is not None:
                for i in range(len(r.boxes)):
                    x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()
                    c = float(r.boxes.conf[i])
                    dets.append(((float(x1), float(y1), float(x2), float(y2)), c))
                    raw.append([float(x1), float(y1), float(x2), float(y2), c])
            rule = spec.match_rule
            tp, fp, fn = _score_branch(dets, gt, rule)
            bare["tp"] += tp; bare["fp"] += fp; bare["fn"] += fn
            # MLP: keep if P(drone) >= mlp_thr (empty dets -> kept=[] so FN counted)
            kept = []
            if dets:
                probs = mlp.score_dets(hook, raw, (ih, iw))
                kept = [d for d, pr in zip(dets, probs) if float(pr) >= mlp_thr]
            tp, fp, fn = _score_branch(kept, gt, rule)
            mlpf["tp"] += tp; mlpf["fp"] += fp; mlpf["fn"] += fn
            # Patch: keep if P(confuser) < patch_thr (empty dets -> kept_p=[])
            if patch is not None:
                kept_p = []
                if dets:
                    cp = patch.predict_boxes(im, [d[0] for d in dets])
                    kept_p = [d for d, c in zip(dets, cp) if float(c) < patch_thr]
                tp, fp, fn = _score_branch(kept_p, gt, rule)
                ptch["tp"] += tp; ptch["fp"] += fp; ptch["fn"] += fn

        b, m = compute_prf(**bare), compute_prf(**mlpf)
        print(f"\n  {spec.name}  ({len(imgs)} imgs, role={spec.role}, {time.time()-t0:.0f}s)")
        print(f"    {'':6} {'TP':>5} {'FP':>5} {'FN':>5} {'P':>6} {'R':>6} {'F1':>6}  {'ΔR':>7} {'ΔF1':>7}")
        print(f"    {'bare':6} {bare['tp']:>5} {bare['fp']:>5} {bare['fn']:>5} "
              f"{b['precision']:>6.3f} {b['recall']:>6.3f} {b['f1']:>6.3f}")
        row = {"surface": spec.name, "role": spec.role, "bare": b, "mlp": m, "n_images": len(imgs)}
        if patch is not None:
            pp = compute_prf(**ptch)
            row["patch"] = pp
            print(f"    {'patch':6} {ptch['tp']:>5} {ptch['fp']:>5} {ptch['fn']:>5} "
                  f"{pp['precision']:>6.3f} {pp['recall']:>6.3f} {pp['f1']:>6.3f}  "
                  f"{pp['recall']-b['recall']:>+7.3f} {pp['f1']-b['f1']:>+7.3f}")
        print(f"    {'mlp':6} {mlpf['tp']:>5} {mlpf['fp']:>5} {mlpf['fn']:>5} "
              f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}  "
              f"{m['recall']-b['recall']:>+7.3f} {m['f1']-b['f1']:>+7.3f}")
        results.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "holdout.json").write_text(json.dumps({
        "detector": yolo_path, "mlp": mlp_weights, "patch": patch_weights,
        "conf": conf, "mlp_thr": mlp_thr, "patch_thr": patch_thr,
        "surfaces": results,
    }, indent=2, default=str))
    print(f"\n  wrote {out_dir/'holdout.json'}")
    print("\n  Gate: ship only if FP drops on confuser (neg) surfaces with ~0 recall "
          "loss on drone (pos) surfaces.")
    return 0
