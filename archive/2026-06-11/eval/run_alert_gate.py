"""Quick script to compute alert gate for Svanstrom from existing temporal cache."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "classifier"))

from eval_detector import (
    DATASET_REGISTRY, MODEL_REGISTRY,
    group_frames_by_video, sample_temporal_frames, list_frames,
    read_yolo_labels, score_detections, compute_prf,
    PATCH_RGB_PATH, PATCH_IR_PATH,
)
from pathlib import Path
import json, cv2, time
import numpy as np

out_dir = Path(__file__).resolve().parent.parent / "eval" / "results" / "detector_eval"
ds = DATASET_REGISTRY["svanstrom"]
scoring = ds.get("scoring", "iop")

all_frames = list_frames(ds)
temporal_frames = sample_temporal_frames(all_frames, segment_size=3, target_windows=7, min_consec=15)

for mk in ["selcom_1280_960imgsz", "ir_v3b"]:
    weights, imgsz, modality, conf = MODEL_REGISTRY[mk]
    cache_path = out_dir / f"{mk}_temporal_detections.json"
    det_cache = json.loads(cache_path.read_text())
    matched = sum(1 for f in temporal_frames if f["stem"] in det_cache)
    print(f"\n{mk}: {matched}/{len(temporal_frames)} frames matched in cache")

    from patch_verifier import PatchVerifier
    patch_path = PATCH_IR_PATH if modality == "ir" else PATCH_RGB_PATH
    pv = PatchVerifier(str(patch_path))

    videos = group_frames_by_video(temporal_frames)
    tp_total = fp_total = fn_total = 0
    frm_tp = frm_fp = frm_fn = frm_tn = 0
    n_frames = n_suppressed = 0
    t0 = time.time()

    for vp, vframes in videos.items():
        frame_data = []
        for fr in vframes:
            if modality == "ir" and fr["ir_path"] is not None:
                img = cv2.imread(str(fr["ir_path"]))
                lbl_path = fr["ir_lbl"]
            else:
                img = cv2.imread(str(fr["rgb_path"]))
                lbl_path = fr["rgb_lbl"]
            if img is None: continue
            h, w = img.shape[:2]
            gts = read_yolo_labels(lbl_path, w, h, drone_classes={0}) if lbl_path else []
            raw = det_cache.get(fr["stem"], [])
            dets = [((d[0], d[1], d[2], d[3]), d[4]) for d in raw]
            frame_data.append({"dets": dets, "gts": gts,
                               "has_det": len(dets) > 0, "has_gt": len(gts) > 0,
                               "img": img})

        for seg_start in range(0, len(frame_data) - 2, 3):
            seg = frame_data[seg_start:seg_start + 3]
            if len(seg) < 3: break
            det_count = sum(1 for f in seg if f["has_det"])
            confirmed = det_count >= 2

            gate_suppressed = False
            if confirmed:
                for fd in [f for f in seg if f["has_det"]]:
                    boxes = [d[0] for d in fd["dets"]]
                    probs = pv.predict_boxes(fd["img"], boxes)
                    mp = float(np.max(probs)) if (isinstance(probs, np.ndarray) and probs.size > 0) else (float(max(probs)) if probs else 0.0)
                    if mp >= 0.70:
                        gate_suppressed = True
                        n_suppressed += 1
                        break

            for fd in seg:
                kept = fd["dets"] if (confirmed and not gate_suppressed) else []
                tp, fp, fn = score_detections(kept, fd["gts"], rule=scoring)
                tp_total += tp; fp_total += fp; fn_total += fn
                hd, hg = len(kept) > 0, fd["has_gt"]
                if hg and hd: frm_tp += 1
                elif hg and not hd: frm_fn += 1
                elif not hg and hd: frm_fp += 1
                else: frm_tn += 1
                n_frames += 1

    dt = time.time() - t0
    prf = compute_prf(tp_total, fp_total, fn_total)
    tot = frm_tp + frm_fp + frm_fn + frm_tn
    fp_pct = round(frm_fp / tot * 100, 2) if tot else 0
    tn_pct = round(frm_tn / tot * 100, 2) if tot else 0
    print(f"  {dt:.1f}s | {n_suppressed} segments suppressed | {n_frames} frames")
    print(f"  P={prf['precision']:.4f}  R={prf['recall']:.4f}  F1={prf['f1']:.4f}  FP%={fp_pct}%  TN%={tn_pct}%")

    # Save to CSV in exactly the expected format
    import pandas as pd
    out_csv = out_dir / f"alert_gate_{mk}_svanstrom.csv"
    res_df = pd.DataFrame([{
        "stage": f"alert_gate_{mk}",
        "TP": tp_total, "FP": fp_total, "FN": fn_total,
        "P": prf["precision"], "R": prf["recall"], "F1": prf["f1"],
        "FP_pct": fp_pct, "TN_pct": tn_pct, "n_frames": n_frames,
        "frame_tp": frm_tp, "frame_fp": frm_fp, "frame_fn": frm_fn, "frame_tn": frm_tn
    }])
    res_df.to_csv(out_csv, index=False)
    print(f"  Saved results to {out_csv.name}")
