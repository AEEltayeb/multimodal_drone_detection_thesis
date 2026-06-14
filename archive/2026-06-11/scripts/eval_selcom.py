import time
import pickle
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO

# Import YOLO hook and V5 definitions
import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.append(str(REPO))
sys.path.append(str(REPO / "eval"))

from distill_v5_p3p5_ft4 import (
    DetectInputHook, _extract_detection_features, _PassThroughClassifier,
    MLPWrapper, FocalLoss, LogRegWrapper, RFWrapper, XGBWrapper
)
from metrics import score_detections, compute_prf

# Need to monkeypatch MLPWrapper into the correct module namespace for pickle
sys.modules['eval.distill_v4_p3p5_ft4'] = sys.modules['distill_v5_p3p5_ft4']
sys.modules['eval.distill_v5_p3p5_ft4'] = sys.modules['distill_v5_p3p5_ft4']

def evaluate_selcom(model, hook, classifier, name, img_dir, feat_type="meta+yolo"):
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in ('.jpg', '.png'))
    print(f"\nEvaluating {name} on {len(images)} SELCOM images...")

    labels_dir = img_dir.parent.parent / "labels" / img_dir.name
    if not labels_dir.exists():
        print(f"ERROR: Cannot find labels at {labels_dir}")
        return

    totals = {"tp": 0, "fp": 0, "fn": 0}
    t0 = time.time()

    for img_path in images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None: continue
        ih, iw = img_bgr.shape[:2]

        gt_boxes = []
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            for line in lbl_path.read_text().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5 and int(parts[0]) == 0:
                    xc, yc, bw, bh = map(float, parts[1:5])
                    x1 = (xc - bw / 2) * iw
                    y1 = (yc - bh / 2) * ih
                    x2 = (xc + bw / 2) * iw
                    y2 = (yc + bh / 2) * ih
                    gt_boxes.append((x1, y1, x2, y2))

        hook.clear()
        results = model.predict(img_bgr, imgsz=1280, conf=0.25, verbose=False, device="cuda")
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            if len(gt_boxes) > 0:
                totals["fn"] += len(gt_boxes)
            continue

        dets = []
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy().tolist()
            conf = float(boxes.conf[i])
            dets.append((xyxy, conf))

        filtered_dets = []
        for det_box, det_conf in dets:
            if isinstance(classifier, _PassThroughClassifier):
                filtered_dets.append((det_box, det_conf))
            else:
                feat = _extract_detection_features(hook, det_box, (ih, iw), det_conf)
                if feat_type == "meta_only":
                    feat = feat[:5]
                elif feat_type == "yolo_only":
                    feat = feat[5:]
                feat = feat.reshape(1, -1)
                yp = int(classifier.predict(feat)[0])
                if yp == 1:
                    filtered_dets.append((det_box, det_conf))

        tp, fp, fn = score_detections(filtered_dets, gt_boxes, rule="iop", iou_thr=0.5, iop_thr=0.5)
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn

    prf = compute_prf(totals["tp"], totals["fp"], totals["fn"])
    print(f"  TP: {prf['TP']} | FP: {prf['FP']} | FN: {prf['FN']}")
    print(f"  Precision: {prf['precision']:.4f} | Recall: {prf['recall']:.4f} | F1: {prf['f1']:.4f}")

if __name__ == "__main__":
    REPO = Path(__file__).resolve().parent.parent
    SELCOM_VAL = Path("G:/drone/_finetune_selcom_mixed_ft2/images/val")
    FT4_MODEL = REPO / "RGB model" / "Yolo26n_selcom_confuser_ft4_1280" / "weights" / "best.pt"
    
    print("Loading Baseline YOLO FT4...")
    model = YOLO(str(FT4_MODEL))
    hook = DetectInputHook()
    hook.register(model)

    # 1. Evaluate Baseline
    evaluate_selcom(model, hook, _PassThroughClassifier(), "Baseline YOLO (FT4)", SELCOM_VAL)

    # 2. Evaluate V4 (SKIPPED: V4 uses 325-D features, V5 uses 517-D. Extracting both simultaneously requires dynamic P3 grids)
    # v4_pkl = REPO / "eval/results/_v4_p3p5_ft4_distill/classifiers.pkl"
    # if v4_pkl.exists():
    #     with open(v4_pkl, "rb") as f:
    #         v4_clfs = pickle.load(f)
    #     v4_mlp, _, _, _ = v4_clfs["mlp_meta+yolo"]
    #     evaluate_selcom(model, hook, v4_mlp, "V4 (mlp_meta+yolo)", SELCOM_VAL)

    # 3. Evaluate V5
    v5_pkl = REPO / "eval/results/_v5_p3p5_ft4_distill/classifiers.pkl"
    if v5_pkl.exists():
        with open(v5_pkl, "rb") as f:
            v5_clfs = pickle.load(f)
        v5_mlp, _, _, _ = v5_clfs["mlp_meta+yolo"]
        evaluate_selcom(model, hook, v5_mlp, "V5 (mlp_meta+yolo)", SELCOM_VAL)
