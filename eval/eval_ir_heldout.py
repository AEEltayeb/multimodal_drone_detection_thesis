"""eval_ir_heldout.py — HONEST held-out IR confuser eval (the offline ir_confusers
cache is the TRAIN split the balanced filter trained on => leaky). Two held-out sets:
  - CBAM (cbam.pkl; a SEPARATE dataset held out of all training; ZERO-GPU): drone
    recall + confuser FP for shipped / native / balanced.
  - IR_confusers val+test (held out from the train split; GPU re-mine): confuser FP.
Compares shipped vs native vs balanced at thr 0.01 + 0.05.
  py -u eval/eval_ir_heldout.py
"""
from __future__ import annotations
import sys, time, pickle
from pathlib import Path
import numpy as np
import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "classifier")); sys.path.insert(0, str(REPO / "eval"))
from metrics import compute_prf, score_detections                      # noqa: E402
from pipeline_eval_offline import get_mlp, mlp_probs_per_frame, CACHE   # noqa: E402
from eval_v4_vs_patch import MLPv4Verifier                              # noqa: E402
import distill_v5_p3p5_ft4 as D                                         # noqa: E402
from ultralytics import YOLO                                            # noqa: E402

FILTERS = [("shipped", REPO / "models/verifiers/ir_aligned/mlp_aligned.pt"),
           ("native", REPO / "eval/results/_v5_ir_p3p5_v3b/classifiers/mlp_v5_ir.pt"),
           ("balanced", REPO / "mri/results/ir_aligned_balanced/classifiers/mlp_aligned.pt"),
           ("cbam", REPO / "mri/results/ir_aligned_cbam_thermalonly/classifiers/mlp_aligned.pt")]
V3B = REPO / "models/ir/corrective_finetune/finetune_v3b/weights/best.pt"
if not V3B.exists():
    V3B = REPO / "runs/corrective_finetune/finetune_v3b/weights/best.pt"
THRS = [0.01, 0.05]


def log(m): print(m, flush=True)


def cbam_eval():
    """ZERO-GPU CBAM held-out eval. Returns a results dict for JSON provenance."""
    log("=== CBAM (held out of ALL training; zero-GPU from cbam.pkl) ===")
    d = pickle.load(open(CACHE / "cbam.pkl", "rb")); fr = d["frames"]; rule = d["meta"]["rule"]
    log(f"  CBAM: {d['meta']['n_images']} imgs, has_drones={d['meta']['has_drones']}, rule={rule}")
    out = {"dataset": "CBAM", "split": "held-out (separate dataset, in no training set)",
           "n_images": d["meta"]["n_images"], "rule": rule, "results": {}}
    # bare
    tp = fp = fn = 0
    for f in fr:
        kept = [(tuple(f["boxes"][i]), float(f["confs"][i])) for i in range(len(f["confs"]))]
        t, f_, n_ = score_detections(kept, [tuple(g) for g in f["gt_boxes"]], rule=rule, iou_thr=0.5, iop_thr=0.5)
        tp += t; fp += f_; fn += n_
    out["results"]["bare"] = {"recall": round(compute_prf(tp, fp, fn)["recall"], 4), "confuser_FP": fp}
    log(f"  {'bare':<18} drone-R {compute_prf(tp,fp,fn)['recall']:.3f}  confuser-FP {fp}")
    for tag, wt in FILTERS:
        probs = mlp_probs_per_frame(fr, get_mlp(str(wt)))
        for thr in THRS:
            tp = fp = fn = 0
            for fi, f in enumerate(fr):
                n = len(f["confs"]); keep = probs[fi] >= thr if n else np.zeros(0, bool)
                kept = [(tuple(f["boxes"][i]), float(f["confs"][i])) for i in range(n) if keep[i]]
                t, f_, n_ = score_detections(kept, [tuple(g) for g in f["gt_boxes"]], rule=rule, iou_thr=0.5, iop_thr=0.5)
                tp += t; fp += f_; fn += n_
            out["results"][f"{tag}@{thr}"] = {"recall": round(compute_prf(tp, fp, fn)["recall"], 4), "confuser_FP": fp}
            log(f"  {tag+'@'+str(thr):<18} drone-R {compute_prf(tp,fp,fn)['recall']:.3f}  confuser-FP {fp}")
    return out


def irconf_valtest_eval():
    log("\n=== IR_confusers VAL+TEST (held out from the train split; GPU re-mine) ===")
    dirs = [Path("G:/drone/IR_confusers/images/val"), Path("G:/drone/IR_confusers/images/test")]
    imgs = [p for dd in dirs if dd.exists() for p in sorted(dd.iterdir()) if D.is_jpg(p)]
    log(f"  {len(imgs)} held-out confuser imgs (val+test)")
    model = YOLO(str(V3B)); hook = D.DetectInputHook(); h = hook.register(model)
    feats = []; t0 = time.time(); n = 0
    try:
        for i, p in enumerate(imgs):
            im = cv2.imread(str(p))
            if im is None:
                continue
            n += 1; ih, iw = im.shape[:2]; hook.clear()
            r = model.predict(im, imgsz=640, conf=0.40, verbose=False, device="cuda")[0]
            if r.boxes is not None:
                for j in range(len(r.boxes)):
                    box = tuple(r.boxes.xyxy[j].cpu().numpy().tolist())
                    feats.append(D._extract_detection_features(hook, box, (ih, iw), float(r.boxes.conf[j])))
            if n % 200 == 0:
                fps = n / (time.time() - t0)
                log(f"    [{i+1}/{len(imgs)}] {len(feats)} fires | {fps:.1f} img/s | ETA ~{(len(imgs)-i-1)/max(fps,0.1):.0f}s")
    finally:
        h.remove()
    Xc = np.array(feats, np.float32)
    log(f"  held-out confuser fires: {len(Xc)} (bare = all kept)")
    log(f"  {'bare':<18} confuser-FP {len(Xc)}")
    out = {"dataset": "IR_confusers val+test", "split": "held-out from the train split",
           "n_fires": int(len(Xc)), "results": {"bare": int(len(Xc))}}
    for tag, wt in FILTERS:
        p = MLPv4Verifier(Path(wt), device="cpu").predict_drone_probs(Xc)
        for thr in THRS:
            out["results"][f"{tag}@{thr}"] = int((p >= thr).sum())
            log(f"  {tag+'@'+str(thr):<18} confuser-FP {int((p>=thr).sum())} / {len(Xc)}")
    return out


def main():
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--cbam-only", action="store_true", help="zero-GPU CBAM held-out only (skip the GPU re-mine)")
    ap.add_argument("--out", default=str(REPO / "eval/results/ir_heldout_results.json"),
                    help="save the CBAM held-out numbers as JSON (provenance)")
    a = ap.parse_args()
    res = {"cbam": cbam_eval()}
    json.dump(res, open(a.out, "w"), indent=2)
    log(f"\nWROTE {a.out}")
    if not a.cbam_only:
        irconf_valtest_eval()   # GPU re-mine of IR_confusers val/test (90->22); prints only


if __name__ == "__main__":
    main()
