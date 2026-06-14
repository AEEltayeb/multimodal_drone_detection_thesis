#!/usr/bin/env python3
"""Does the IR detector (v3b) actually hallucinate on REAL THERMAL confusers?

Decides whether an IR confuser-verifier (patch or MLP V5) is necessary at all.
Scores v3b on held-out thermal datasets and reports, per conf threshold:
  - hallucination rate on confuser-only images (no drone GT) -> the headroom a
    verifier could recover.
  - drone recall (so we know the detector still works on this surface).

Surfaces:
  cbam   -- G:/drone/Infrared_bird_drone_airplane_CBAM_TF-Net (classes B/D/P,
            grayscale thermal). NEVER used in V5-IR mining -> fully held out.
            Drone = class 1; bird(0)/plane(2) are confusers.

Usage:
  python eval/ir_confuser_necessity.py
  python eval/ir_confuser_necessity.py --confs 0.25,0.40 --split valid
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO))
from metrics import score_detections, compute_prf  # noqa: E402


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _is_tp(box, drone_gt):
    return any(_iou(box, g) >= 0.5 for g in drone_gt)

IR_DETECTOR = REPO / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt"
CBAM = Path("G:/drone/Infrared_bird_drone_airplane_CBAM_TF-Net.v1i.yolo26-maha-daxhh-cbam_tf-net")
DRONE_CLASS = 1  # data.yaml names = ['B','D','P'] -> drone is index 1
IMGSZ = 640      # CBAM is 320x256 native; 640 is the deploy default for non-Svan IR


def load_gt(lbl_path: Path, iw: int, ih: int):
    """Return (drone_boxes, n_confuser_boxes). drone_boxes in xyxy pixels."""
    drones, n_conf = [], 0
    if not lbl_path.exists():
        return drones, n_conf
    for line in lbl_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls = int(p[0])
        xc, yc, bw, bh = map(float, p[1:5])
        if cls == DRONE_CLASS:
            drones.append(((xc - bw / 2) * iw, (yc - bh / 2) * ih,
                           (xc + bw / 2) * iw, (yc + bh / 2) * ih))
        else:
            n_conf += 1
    return drones, n_conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confs", default="0.25,0.40")
    ap.add_argument("--split", default="valid")
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--mlp", default="",
                    help="Path to mlp_v5_ir.pt; if set, also report MLP catch/loss "
                         "at the detector conf given by --mlp-det-conf.")
    ap.add_argument("--mlp-det-conf", type=float, default=0.40)
    ap.add_argument("--mlp-thrs", default="0.25,0.5")
    args = ap.parse_args()
    confs = [float(c) for c in args.confs.split(",")]

    img_dir = CBAM / args.split / "images"
    lbl_dir = CBAM / args.split / "labels"
    if not img_dir.exists():
        print(f"FATAL: {img_dir} not found"); sys.exit(1)
    imgs = sorted(p for p in img_dir.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"Detector: {IR_DETECTOR}")
    print(f"Surface : CBAM thermal / {args.split}  ({len(imgs)} images, imgsz={args.imgsz})")
    print(f"Classes : drone=class{DRONE_CLASS}; bird/plane = confusers\n")

    yolo = YOLO(str(IR_DETECTOR))

    # Pre-load GT + classify images as drone-bearing vs confuser-only
    gt = {}
    n_conf_only = n_drone_imgs = 0
    for p in imgs:
        im = cv2.imread(str(p))
        if im is None:
            continue
        ih, iw = im.shape[:2]
        dboxes, ncf = load_gt(lbl_dir / (p.stem + ".txt"), iw, ih)
        gt[p] = (dboxes, ncf, (ih, iw))
        if dboxes:
            n_drone_imgs += 1
        elif ncf:
            n_conf_only += 1
    print(f"  {n_drone_imgs} drone-bearing imgs | {n_conf_only} confuser-only imgs | "
          f"{len(gt)-n_drone_imgs-n_conf_only} empty\n")

    print(f"{'conf':>5} | {'det':>5} {'TP':>4} {'FP':>5} {'FN':>4} | "
          f"{'P':>6} {'R':>6} {'F1':>6} | {'halluc/conf-img':>15} {'%conf-img-fire':>14}")
    print("-" * 92)
    for conf in confs:
        TP = FP = FN = n_det = 0
        conf_img_fires = 0  # confuser-only images with >=1 detection
        for p, (dboxes, ncf, (ih, iw)) in gt.items():
            r = yolo.predict(str(p), imgsz=args.imgsz, conf=conf,
                             verbose=False, device=0)[0]
            dets = []
            if r.boxes is not None:
                for i in range(len(r.boxes)):
                    x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()
                    # metrics.score_detections expects (box_xyxy, conf) pairs
                    dets.append(((float(x1), float(y1), float(x2), float(y2)),
                                 float(r.boxes.conf[i])))
            n_det += len(dets)
            tp, fp, fn = score_detections(dets, dboxes, rule="iou", iou_thr=0.5)
            TP += tp; FP += fp; FN += fn
            if not dboxes and ncf and dets:
                conf_img_fires += 1
        prf = compute_prf(TP, FP, FN)
        halluc_per_conf = FP / max(n_conf_only, 1)
        pct_conf_fire = 100.0 * conf_img_fires / max(n_conf_only, 1)
        print(f"{conf:>5.2f} | {n_det:>5} {TP:>4} {FP:>5} {FN:>4} | "
              f"{prf['precision']:>6.3f} {prf['recall']:>6.3f} {prf['f1']:>6.3f} | "
              f"{halluc_per_conf:>15.3f} {pct_conf_fire:>13.1f}%")

    print("\nReading: high halluc/conf-img => IR detector DOES hallucinate on thermal "
          "confusers => a verifier could help. Low => verifier unnecessary.")

    # ── Optional: does the current IR MLP catch these thermal FPs? ──────────
    if args.mlp:
        from classifier.mlp_verifier import MLPVerifier, DetectInputHook
        mlp = MLPVerifier(args.mlp, "cuda")
        hook = DetectInputHook(); hook.register(yolo)
        dc = args.mlp_det_conf
        mlp_thrs = [float(t) for t in args.mlp_thrs.split(",")]
        # Collect per-detection (P_drone, is_tp) across the surface at det conf dc.
        tp_probs, fp_probs = [], []
        for p, (dboxes, ncf, (ih, iw)) in gt.items():
            hook.clear()
            r = yolo.predict(str(p), imgsz=args.imgsz, conf=dc, verbose=False, device=0)[0]
            if r.boxes is None or len(r.boxes) == 0:
                continue
            dets = []
            for i in range(len(r.boxes)):
                x1, y1, x2, y2 = r.boxes.xyxy[i].cpu().numpy()
                dets.append([float(x1), float(y1), float(x2), float(y2),
                             float(r.boxes.conf[i])])
            probs = mlp.score_dets(hook, dets, (ih, iw))
            for d, pr in zip(dets, probs):
                (tp_probs if _is_tp(d[:4], dboxes) else fp_probs).append(float(pr))
        tp_probs, fp_probs = np.array(tp_probs), np.array(fp_probs)
        print(f"\n── Current IR MLP ({Path(args.mlp).name}) on CBAM @ det-conf={dc} ──")
        print(f"  {len(tp_probs)} TP dets, {len(fp_probs)} FP dets")
        print(f"  TP P(drone): mean={tp_probs.mean():.3f}  FP P(drone): mean={fp_probs.mean():.3f}")
        print(f"  {'mlp_thr':>7} | {'FP caught (veto)':>17} | {'TP lost (veto)':>15}")
        print("  " + "-" * 48)
        for t in mlp_thrs:
            fp_caught = int((fp_probs < t).sum())
            tp_lost = int((tp_probs < t).sum())
            print(f"  {t:>7.2f} | {fp_caught:>4}/{len(fp_probs)} "
                  f"({100*fp_caught/max(len(fp_probs),1):>4.0f}%)      | "
                  f"{tp_lost:>3}/{len(tp_probs)} ({100*tp_lost/max(len(tp_probs),1):>4.0f}%)")
        print("  (Good = high FP-caught, low TP-lost. Low FP-caught => wrong-modality "
              "training; re-mine on thermal confusers.)")


if __name__ == "__main__":
    main()
