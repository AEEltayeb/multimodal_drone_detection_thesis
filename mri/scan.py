"""
mri.scan — run the detector across the datasets and mine the feature corpus.

For each DatasetSpec this:
  * runs YOLO on every (strided) image at the spec's imgsz,
  * captures the FPN features via the hook,
  * labels each detection drone(1)/confuser(0) (GT-match for pos dirs, all-FP
    for neg dirs),
  * records the *bare-detector* tallies (raw FP rate, raw TP/FP/FN) so the
    diagnosis layer can compare "needs a classifier?" against ground truth.

Reuses eval/metrics.py for IoU/IoP scoring (added to sys.path by mri.cli).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from .datasets import DatasetSpec, resolve_labels_dir
from .extract import FeatureExtractor

# eval/ is added to sys.path by mri.cli before this import runs.
from metrics import iou_iop, score_detections, compute_prf  # type: ignore


def _read_gt(labels_dir: Path, stem: str, ih: int, iw: int,
             main_class: int = 0) -> list[tuple]:
    boxes = []
    lbl = labels_dir / (stem + ".txt")
    if not lbl.exists():
        return boxes
    for line in lbl.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5 and int(float(parts[0])) == main_class:
            xc, yc, bw, bh = map(float, parts[1:5])
            boxes.append(((xc - bw / 2) * iw, (yc - bh / 2) * ih,
                          (xc + bw / 2) * iw, (yc + bh / 2) * ih))
    return boxes


def _match(det_box, gt_boxes, rule, iou_thr=0.5, iop_thr=0.5) -> bool:
    if not gt_boxes:
        return False
    thr = iop_thr if rule == "iop" else iou_thr
    for g in gt_boxes:
        iu, ip = iou_iop(det_box, g)
        if (ip if rule == "iop" else iu) >= thr:
            return True
    return False


def scan_source(extractor: FeatureExtractor, spec: DatasetSpec,
                conf_thr=0.25, device="cuda", iou_thr=0.5, iop_thr=0.5,
                grayscale=False):
    """Mine one dataset. Returns (X, y, w, raw_stats_dict, provenance).

    provenance is a list aligned with X: per-row {spec, path, box, conf} so an
    example detection can be re-rendered later (spatial activation panels)."""
    images = spec.list_images()
    if not images:
        print(f"  SKIP {spec.name}: no images at {spec.path}")
        return (np.empty((0, 0), np.float32), np.empty(0), np.empty(0),
                {"name": spec.name, "n_images": 0}, [])

    labels_dir = resolve_labels_dir(spec.path) if spec.has_gt else None
    print(f"  Scanning {spec.name}: {len(images)} imgs "
          f"(imgsz={spec.imgsz}, rule={spec.match_rule}, role={spec.role})")

    X, y, w, prov = [], [], [], []
    raw = {"name": spec.name, "role": spec.role, "n_images": 0,
           "tp": 0, "fp": 0, "fn": 0, "n_dets": 0}
    t0 = time.time()

    n_d = n_c = 0  # running counters (avoid O(n^2) re-summing)
    for img_path in images:
        # Stop on the spec's PRIMARY quota: drones for a pos dataset, confusers
        # for a neg dataset. Confusers mined from a pos dataset are incidental
        # hard-negatives — don't keep scanning a clean drone set hunting for them
        # (that over-scans datasets like Anti-UAV that rarely hallucinate).
        primary_done = ((spec.role == "pos" and spec.max_drones and n_d >= spec.max_drones)
                        or (spec.role == "neg" and spec.max_confusers and n_c >= spec.max_confusers))
        if primary_done:
            break

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if grayscale:  # grayscale-fallback mode: feed gray-3ch (the deploy op)
            img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        raw["n_images"] += 1
        ih, iw = img.shape[:2]

        gt = _read_gt(labels_dir, img_path.stem, ih, iw, spec.main_class) if labels_dir else []

        extractor.hook.clear()
        res = extractor.model.predict(img, imgsz=spec.imgsz, conf=conf_thr,
                                      verbose=False, device=device)
        boxes = res[0].boxes
        dets = []
        if boxes is not None and len(boxes):
            for i in range(len(boxes)):
                dets.append((tuple(boxes.xyxy[i].cpu().numpy().tolist()),
                             float(boxes.conf[i])))
        raw["n_dets"] += len(dets)

        # Bare-detector tally (no classifier) for the diagnosis baseline.
        if spec.has_gt:
            tp, fp, fn = score_detections(dets, gt, rule=spec.match_rule,
                                          iou_thr=iou_thr, iop_thr=iop_thr)
            raw["tp"] += tp; raw["fp"] += fp; raw["fn"] += fn
        else:
            raw["fp"] += len(dets)  # every det on a confuser dir is an FP

        # Mine features per detection.
        for det_box, det_conf in dets:
            is_drone = (spec.has_gt and
                        _match(det_box, gt, spec.match_rule, iou_thr, iop_thr))
            if is_drone:
                if spec.max_drones and n_d >= spec.max_drones:
                    continue
            else:
                if spec.max_confusers and n_c >= spec.max_confusers:
                    continue
            feat = extractor.extract_one(det_box, det_conf, (ih, iw))
            X.append(feat)
            y.append(1 if is_drone else 0)
            w.append(spec.weight_drone if is_drone else spec.weight_confuser)
            prov.append({"spec": spec.name, "path": str(img_path),
                         "box": tuple(float(v) for v in det_box),
                         "conf": float(det_conf)})
            if is_drone:
                n_d += 1
            else:
                n_c += 1

    dt = max(time.time() - t0, 0.1)
    n_d = int(sum(1 for v in y if v == 1))
    n_c = len(y) - n_d
    raw["mined_drones"] = n_d
    raw["mined_confusers"] = n_c
    raw["fps"] = round(raw["n_images"] / dt, 1)
    print(f"    -> {n_d} drone + {n_c} confuser feats "
          f"(bare tp={raw['tp']} fp={raw['fp']} fn={raw['fn']}, {raw['fps']} fps)")

    dim = extractor.schema.total_dim
    X_arr = np.array(X, np.float32) if X else np.empty((0, dim), np.float32)
    return X_arr, np.array(y, np.float32), np.array(w, np.float32), raw, prov


def collect(extractor: FeatureExtractor, specs: list[DatasetSpec],
            conf_thr=0.25, device="cuda", iou_thr=0.5, iop_thr=0.5, seed=42,
            grayscale=False):
    """Scan all specs, concatenate, shuffle. Returns (X, y, w, raws, provenance).

    provenance is a list aligned with the returned (shuffled) X rows."""
    Xs, ys, ws, raws, provs = [], [], [], [], []
    for spec in specs:
        X, y, w, raw, prov = scan_source(extractor, spec, conf_thr, device,
                                         iou_thr, iop_thr, grayscale=grayscale)
        raws.append(raw)
        if len(X):
            Xs.append(X); ys.append(y); ws.append(w); provs.extend(prov)
    if not Xs:
        return (np.empty((0, extractor.schema.total_dim), np.float32),
                np.empty(0), np.empty(0), raws, [])
    X = np.concatenate(Xs); y = np.concatenate(ys); w = np.concatenate(ws)
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(X))
    prov_shuffled = [provs[i] for i in perm]
    return X[perm], y[perm], w[perm], raws, prov_shuffled
