#!/usr/bin/env python3
"""IR verifier head-to-head: bare v3b vs IR MLP V5, across held-out IR surfaces.

Answers the load-bearing question the necessity check raised: the IR MLP catches
~92% of thermal-confuser FPs on CBAM but loses ~17% recall there — is that recall
loss CBAM-specific or does it hit the main drone surfaces?

Surfaces (all held out from V5-IR mining, which used val/train splits):
  cbam        -- CBAM thermal confuser set (bird/drone/plane); drone=class 1.
  antiuav     -- Anti-UAV test IR (drone tracking, saturated); drone=class 0.
  ir_dset     -- IR_dset_final test (general IR benchmark); drone=class 0.
  ir_video    -- IR_video test (drone clips + airplane/bird/heli confuser clips);
                 drone=class 0, confuser clips have empty labels.

Reports per surface: bare P/R/F1 vs MLP@thr P/R/F1, and the recall delta.

Usage:
  python eval/ir_verifier_eval.py
  python eval/ir_verifier_eval.py --mlp-thr 0.15 --det-conf 0.40
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO))
from metrics import score_detections, compute_prf  # noqa: E402
from classifier.mlp_verifier import MLPVerifier, DetectInputHook  # noqa: E402
from classifier.patch_verifier import PatchVerifier  # noqa: E402

IR_DETECTOR = REPO / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt"
MLP_IR = REPO / "eval" / "results" / "_v5_ir_p3p5_v3b" / "classifiers" / "mlp_v5_ir.pt"
IR_PATCH = REPO / "classifier" / "runs" / "patches" / "confuser_filter4_ir_v2_backup.pt"

# name -> (img_dir, drone_class, rule, default_stride, imgsz)
SURFACES = {
    "cbam":     (Path("G:/drone/Infrared_bird_drone_airplane_CBAM_TF-Net.v1i.yolo26-maha-daxhh-cbam_tf-net/valid/images"), 1, "iou", 1,  640),
    "antiuav":  (Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/images"), 0, "iou", 20, 640),
    "ir_dset":  (Path("G:/drone/IR_dset_final/test/images"),                   0, "iou", 5,  640),
    "ir_video": (Path("G:/drone/IR_video_ir_dataset/test/images"),             0, "iou", 2,  640),
}


def load_drone_gt(lbl_path: Path, iw: int, ih: int, drone_cls: int):
    boxes = []
    if not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5 or int(p[0]) != drone_cls:
            continue
        xc, yc, bw, bh = map(float, p[1:5])
        boxes.append(((xc - bw/2)*iw, (yc - bh/2)*ih, (xc + bw/2)*iw, (yc + bh/2)*ih))
    return boxes


def eval_surface(name, yolo, hook, mlp, patch, det_conf, mlp_thr, patch_thr, args):
    img_dir, drone_cls, rule, dflt_stride, imgsz = SURFACES[name]
    if not img_dir.exists():
        print(f"  SKIP {name}: {img_dir} not found"); return None
    stride = args.stride if args.stride else dflt_stride
    imgs = sorted(p for p in img_dir.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))[::stride]
    lbl_dir = img_dir.parent / "labels"
    bare = dict(tp=0, fp=0, fn=0)
    mlpf = dict(tp=0, fp=0, fn=0)
    ptch = dict(tp=0, fp=0, fn=0)
    t0 = time.time()
    for ip in imgs:
        im = cv2.imread(str(ip))
        if im is None:
            continue
        ih, iw = im.shape[:2]
        gt = load_drone_gt(lbl_dir / (ip.stem + ".txt"), iw, ih, drone_cls)
        hook.clear()
        r = yolo.predict(im, imgsz=imgsz, conf=det_conf, verbose=False, device=0)[0]
        dets, raw = [], []
        if r.boxes is not None:
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()
                c = float(r.boxes.conf[i])
                dets.append(((float(x1), float(y1), float(x2), float(y2)), c))
                raw.append([float(x1), float(y1), float(x2), float(y2), c])
        # bare
        tp, fp, fn = score_detections(dets, gt, rule=rule, iou_thr=0.5, iop_thr=0.5)
        bare["tp"] += tp; bare["fp"] += fp; bare["fn"] += fn
        # mlp-vetoed (keep if P(drone) >= mlp_thr)
        if dets:
            probs = mlp.score_dets(hook, raw, (ih, iw))
            kept = [d for d, pr in zip(dets, probs) if float(pr) >= mlp_thr]
        else:
            kept = []
        tp, fp, fn = score_detections(kept, gt, rule=rule, iou_thr=0.5, iop_thr=0.5)
        mlpf["tp"] += tp; mlpf["fp"] += fp; mlpf["fn"] += fn
        # patch-vetoed (keep if P(confuser) < patch_thr). Must also score
        # zero-detection images so FN from missed GT is counted (else recall
        # is inflated — a veto can never recover a TP).
        if patch is not None:
            if dets:
                cprobs = patch.predict_boxes(im, [d[0] for d in dets])
                kept_p = [d for d, cp in zip(dets, cprobs) if float(cp) < patch_thr]
            else:
                kept_p = []
            tp, fp, fn = score_detections(kept_p, gt, rule=rule, iou_thr=0.5, iop_thr=0.5)
            ptch["tp"] += tp; ptch["fp"] += fp; ptch["fn"] += fn
    b, m = compute_prf(**bare), compute_prf(**mlpf)
    print(f"\n  {name}  ({len(imgs)} imgs, stride={stride}, {time.time()-t0:.0f}s)")
    print(f"    {'':6} {'TP':>5} {'FP':>5} {'FN':>5} {'P':>6} {'R':>6} {'F1':>6}  {'ΔR':>7} {'ΔF1':>7}")
    print(f"    {'bare':6} {bare['tp']:>5} {bare['fp']:>5} {bare['fn']:>5} "
          f"{b['precision']:>6.3f} {b['recall']:>6.3f} {b['f1']:>6.3f}")
    if patch is not None:
        pp = compute_prf(**ptch)
        print(f"    {'patch':6} {ptch['tp']:>5} {ptch['fp']:>5} {ptch['fn']:>5} "
              f"{pp['precision']:>6.3f} {pp['recall']:>6.3f} {pp['f1']:>6.3f}  "
              f"{pp['recall']-b['recall']:>+7.3f} {pp['f1']-b['f1']:>+7.3f}")
    else:
        pp = None
    print(f"    {'mlp':6} {mlpf['tp']:>5} {mlpf['fp']:>5} {mlpf['fn']:>5} "
          f"{m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}  "
          f"{m['recall']-b['recall']:>+7.3f} {m['f1']-b['f1']:>+7.3f}")
    return {"surface": name, "bare": b, "patch": pp, "mlp": m}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surfaces", default="cbam,antiuav,ir_dset,ir_video")
    ap.add_argument("--det-conf", type=float, default=0.40)
    ap.add_argument("--mlp-thr", type=float, default=0.15)
    ap.add_argument("--mlp", default=str(MLP_IR))
    ap.add_argument("--patch", default=str(IR_PATCH),
                    help="IR patch verifier weights; '' to skip the patch branch")
    ap.add_argument("--patch-thr", type=float, default=0.5,
                    help="keep detection if P(confuser) < this")
    ap.add_argument("--stride", type=int, default=0, help="override per-surface stride")
    args = ap.parse_args()

    print(f"Detector: {IR_DETECTOR}")
    print(f"IR MLP:   {args.mlp}")
    print(f"IR patch: {args.patch or '(skipped)'}")
    print(f"det_conf={args.det_conf}  mlp_thr={args.mlp_thr}  patch_thr={args.patch_thr}")
    yolo = YOLO(str(IR_DETECTOR))
    hook = DetectInputHook(); hook.register(yolo)
    mlp = MLPVerifier(args.mlp, "cuda")
    patch = None
    if args.patch and Path(args.patch).exists():
        patch = PatchVerifier(args.patch, "cuda")
    elif args.patch:
        print(f"  WARN: patch weights not found ({args.patch}); skipping patch branch")

    for name in [s.strip() for s in args.surfaces.split(",") if s.strip()]:
        if name not in SURFACES:
            print(f"  unknown surface {name}"); continue
        eval_surface(name, yolo, hook, mlp, patch, args.det_conf,
                     args.mlp_thr, args.patch_thr, args)

    print("\nGate: a verifier ships only if it cuts FP on confuser surfaces (cbam, "
          "ir_video) with ~0 recall loss on the drone surfaces (antiuav, ir_dset). "
          "Compare patch vs mlp on the recall cost per FP removed.")


if __name__ == "__main__":
    main()
