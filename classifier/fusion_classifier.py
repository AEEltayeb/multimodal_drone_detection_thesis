"""
fusion_classifier.py — Inference module for the fusion classifier.

Loads the trained model and provides a clean API to classify detections
from paired RGB + IR YOLO outputs.

Usage as a module:
    from classifier.fusion_classifier import FusionClassifier

    clf = FusionClassifier("classifier/runs/classifier.joblib")
    final_dets = clf.classify(rgb_detections, ir_detections, img_w, img_h)

Usage standalone (demo on a single image pair):
    python classifier/fusion_classifier.py --rgb frame_rgb.jpg --ir frame_ir.jpg
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import yaml

from utils import align_detections, extract_features, representative_box


class FusionClassifier:
    """
    Wrapper around the trained fusion model.

    Takes raw detection lists from both YOLO models and returns
    filtered detections that the classifier accepts as true drones.
    """

    def __init__(self, model_path, config_path=None):
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.feature_cols = bundle["feature_cols"]
        self.threshold = bundle["threshold"]
        self.model_type = bundle["model_type"]

        self.alignment_iou = 0.3
        self.use_phase2 = False
        if config_path:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
            self.alignment_iou = cfg.get("alignment_iou", 0.3)
            self.use_phase2 = any(cfg.get(f"use_{k}", False)
                                  for k in ["frame_brightness", "n_dets_total",
                                            "aspect_ratio", "conf_delta"])

    def classify(self, rgb_dets, ir_dets, img_w, img_h, threshold=None):
        """
        Classify candidate detections from both models.

        Parameters
        ----------
        rgb_dets : list of (box, conf) — box is (x1, y1, x2, y2)
        ir_dets  : list of (box, conf)
        img_w, img_h : int — original image dimensions
        threshold : float, optional — override saved threshold

        Returns
        -------
        accepted : list of dict with keys:
            box: (x1, y1, x2, y2) — representative bounding box
            confidence: float — classifier probability
            source: str — "both", "rgb_only", or "ir_only"
            rgb_conf: float
            ir_conf: float
        """
        t = threshold if threshold is not None else self.threshold

        matched, rgb_only, ir_only = align_detections(
            rgb_dets, ir_dets, iou_thresh=self.alignment_iou
        )

        candidates = matched + rgb_only + ir_only
        if not candidates:
            return []

        # Tag source for each candidate
        sources = (["both"] * len(matched) +
                   ["rgb_only"] * len(rgb_only) +
                   ["ir_only"] * len(ir_only))

        img_area = img_w * img_h
        n_dets_total = len(candidates)

        # Extract features
        rows = []
        for cand in candidates:
            feats = extract_features(cand, img_area, n_dets_total=n_dets_total,
                                     use_phase2=self.use_phase2)
            rows.append([feats[c] for c in self.feature_cols])

        X = np.array(rows)
        probs = self.model.predict_proba(X)[:, 1]

        # Filter and build output
        accepted = []
        for i, (cand, prob, source) in enumerate(zip(candidates, probs, sources)):
            if prob >= t:
                accepted.append({
                    "box": representative_box(cand),
                    "confidence": float(prob),
                    "source": source,
                    "rgb_conf": cand.get("rgb_conf", 0.0),
                    "ir_conf": cand.get("ir_conf", 0.0),
                })

        return accepted


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fusion classifier inference demo")
    parser.add_argument("--rgb", required=True, help="Path to RGB image")
    parser.add_argument("--ir", required=True, help="Path to IR image")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", default="runs/classifier.joblib")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    print("Loading YOLO models...")
    from ultralytics import YOLO
    rgb_model = YOLO(cfg["rgb_weights"])
    ir_model = YOLO(cfg["ir_weights"])

    print("Loading fusion classifier...")
    clf = FusionClassifier(args.model, config_path=args.config)

    # Run YOLO inference
    def run_yolo(model, img_path):
        results = model.predict(source=img_path, conf=cfg["conf"],
                                iou=cfg["iou_nms"], imgsz=cfg["imgsz"],
                                device=cfg["device"], verbose=False, save=False)
        r = results[0]
        dets = []
        if r.boxes is not None and len(r.boxes) > 0:
            for i in range(len(r.boxes)):
                box = tuple(float(v) for v in r.boxes.xyxy[i].cpu().numpy())
                conf = float(r.boxes.conf[i].cpu())
                dets.append((box, conf))
        return dets, r.orig_shape[1], r.orig_shape[0]

    rgb_dets, w, h = run_yolo(rgb_model, args.rgb)
    ir_dets, _, _ = run_yolo(ir_model, args.ir)

    print(f"RGB detections: {len(rgb_dets)}, IR detections: {len(ir_dets)}")

    # Classify
    accepted = clf.classify(rgb_dets, ir_dets, w, h)
    print(f"\nAccepted detections: {len(accepted)}")
    for det in accepted:
        print(f"  box={det['box']}, prob={det['confidence']:.3f}, "
              f"source={det['source']}, rgb={det['rgb_conf']:.3f}, "
              f"ir={det['ir_conf']:.3f}")


if __name__ == "__main__":
    main()
