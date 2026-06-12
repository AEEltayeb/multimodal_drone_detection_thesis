"""pipeline_cache.py — Phase A of the offline full-pipeline eval.

ONE GPU pass. For each (modality, detector, surface), strided to ~N_TARGET frames,
runs the detector once and caches per-detection: box, conf, 517-D MLP feature vector,
patch-verifier P(confuser), plus per-frame GT boxes. Phase B (pipeline_eval_offline.py)
then replays every verifier/threshold variant with ZERO GPU.

Reuses the proven RGB harness internals (same 517-D extractor works for any YOLO, so
v3b IR uses it too). Defensive: missing surfaces are skipped; resumable (skips surfaces
whose cache already exists); per-frame try/except so one bad image can't kill the run.

  py -u eval/pipeline_cache.py
  py -u eval/pipeline_cache.py --target 1000 --overwrite
"""
from __future__ import annotations
import argparse, pickle, time, traceback
from pathlib import Path

import cv2, numpy as np, torch
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(REPO / "classifier")); sys.path.insert(0, str(REPO / "eval"))
from patch_verifier import PatchVerifier                       # noqa: E402
from distill_v5_p3p5_ft4 import DetectInputHook, _extract_detection_features, INPUT_DIM  # noqa: E402

OUT = REPO / "eval" / "results" / "_offline_pipeline" / "cache"
OUT.mkdir(parents=True, exist_ok=True)

FT4 = str(REPO / "models/rgb/Yolo26n_selcom_confuser_ft4_1280/weights/best.pt")
V3B = str(REPO / "models/ir/corrective_finetune/finetune_v3b/weights/best.pt")
RGB_PATCH = str(REPO / "models/patches/confuser_filter4_rgb_v2_backup.pt")
IR_PATCH = str(REPO / "models/patches/confuser_filter4_ir_v2_backup.pt")

# modality: which mlp family + patch Phase B applies. grayscale -> feed gray to v3b.
# (path, modality, detector, patch, conf, imgsz, rule, has_drones, gt_class, grayscale, prefix)
SURFACES = [
    # ---- RGB (ft4) ----
    ("svanstrom",      "G:/drone/svanstrom_paired/RGB/images",                         "rgb", FT4, RGB_PATCH, 0.25, 1280, "iop", True, 0, False, None),
    ("antiuav_rgb",    "G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images",         "rgb", FT4, RGB_PATCH, 0.25, 640,  "iou", True, 0, False, None),
    ("selcom_val",     "G:/drone/_finetune_selcom_mixed_ft2/images/val",               "rgb", FT4, RGB_PATCH, 0.25, 1280, "iop", True, 0, False, None),
    ("rgb_dataset_test","G:/drone/dataset/dataset/images/test",                        "rgb", FT4, RGB_PATCH, 0.25, 640,  "iou", True, 0, False, None),
    ("rgb_confuser",   "G:/drone/rgb_confusers_merged/images/test",                    "rgb", FT4, RGB_PATCH, 0.25, 640,  "iou", False,0, False, None),
    ("rgb_bird_confuser","G:/drone/bird.v1i.yolo26-birds-zekpr-bird-pn3pj/train/images","rgb", FT4, RGB_PATCH, 0.25, 640,  "iou", False,0, False, None),
    # ---- IR thermal (v3b) ----
    ("svanstrom_ir",   "G:/drone/svanstrom_paired/IR/images",                          "ir",  V3B, IR_PATCH, 0.40, 640,  "iop", True, 0, False, None),
    ("ir_dset_final",  "G:/drone/IR_dset_final/test/images",                           "ir",  V3B, IR_PATCH, 0.40, 640,  "iou", True, 0, False, None),
    ("ir_video",       "G:/drone/IR_video_ir_dataset/test/images",                     "ir",  V3B, IR_PATCH, 0.40, 640,  "iou", True, 0, False, "IR_DRONE_"),
    ("antiuav_ir",     "G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/images",          "ir",  V3B, IR_PATCH, 0.40, 640,  "iou", True, 0, False, None),
    ("cbam",           "G:/drone/Infrared_bird_drone_airplane_CBAM_TF-Net.v1i.yolo26-maha-daxhh-cbam_tf-net/valid/images", "ir", V3B, IR_PATCH, 0.40, 640, "iou", True, 1, False, None),
    ("ir_confusers",   "G:/drone/IR_confusers/images/train",                           "ir",  V3B, IR_PATCH, 0.40, 640,  "iou", False,0, False, None),
    # ---- grayscale fallback (v3b on RGB->gray) ----
    ("gray_svan",      "G:/drone/svanstrom_paired/RGB/images",                         "gray", V3B, RGB_PATCH, 0.40, 640, "iop", True, 0, True, None),
    ("gray_confuser",  "G:/drone/rgb_confusers_merged/images/test",                    "gray", V3B, RGB_PATCH, 0.40, 640, "iou", False,0, True, None),
]


def list_imgs(d: Path, prefix):
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in exts) if d.exists() else []
    if prefix:
        imgs = [p for p in imgs if p.name.startswith(prefix)]
    return imgs


def gt_boxes_for(img_path: Path, ih, iw, gt_class):
    a = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
    b = img_path.parent.parent.parent / "labels" / img_path.parent.name / (img_path.stem + ".txt")
    lbl = a if a.exists() else (b if b.exists() else None)
    if lbl is None:
        return []
    out = []
    for line in lbl.read_text().splitlines():
        p = line.split()
        if len(p) >= 5 and int(p[0]) == gt_class:
            xc, yc, bw, bh = map(float, p[1:5])
            out.append(((xc - bw/2)*iw, (yc - bh/2)*ih, (xc + bw/2)*iw, (yc + bh/2)*ih))
    return out


def cache_surface(name, path, modality, det_w, patch_w, conf, imgsz, rule,
                  has_drones, gt_class, grayscale, prefix, target, yolo_cache, patch_cache):
    imgs = list_imgs(Path(path), prefix)
    if not imgs:
        print(f"  [skip:no-imgs] {name} ({path})"); return None
    stride = max(1, len(imgs) // target)
    imgs = imgs[::stride][:target]
    if det_w not in yolo_cache:
        y = YOLO(det_w); h = DetectInputHook(); h.register(y); yolo_cache[det_w] = (y, h)
    yolo, hook = yolo_cache[det_w]
    patch = patch_cache.setdefault(patch_w, PatchVerifier(patch_w))
    frames, n_det = [], 0
    t0 = time.time()
    for ip in imgs:
        try:
            img = cv2.imread(str(ip))
            if img is None:
                continue
            if grayscale:
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            ih, iw = img.shape[:2]
            gtb = gt_boxes_for(ip, ih, iw, gt_class) if has_drones else []
            hook.clear()
            res = yolo.predict(img, imgsz=imgsz, conf=conf, verbose=False, device="cuda")
            b = res[0].boxes
            if b is None or len(b) == 0:
                frames.append({"boxes": np.zeros((0, 4), np.float32), "confs": np.zeros(0, np.float32),
                               "feats": np.zeros((0, INPUT_DIM), np.float32), "patch": np.zeros(0, np.float32),
                               "gt_boxes": np.array(gtb, np.float32) if gtb else np.zeros((0, 4), np.float32)})
                continue
            boxes = [tuple(b.xyxy[i].cpu().numpy().tolist()) for i in range(len(b))]
            confs = [float(b.conf[i]) for i in range(len(b))]
            feats = np.stack([_extract_detection_features(hook, db, (ih, iw), dc)
                              for db, dc in zip(boxes, confs)])
            pprob = np.asarray(patch.predict_boxes(img, boxes), np.float32)
            n_det += len(boxes)
            frames.append({"boxes": np.array(boxes, np.float32), "confs": np.array(confs, np.float32),
                           "feats": feats.astype(np.float32), "patch": pprob,
                           "gt_boxes": np.array(gtb, np.float32) if gtb else np.zeros((0, 4), np.float32)})
        except Exception:
            print(f"    [frame-err] {ip.name}\n{traceback.format_exc(limit=1)}")
    meta = {"name": name, "modality": modality, "conf": conf, "imgsz": imgsz, "rule": rule,
            "has_drones": has_drones, "n_images": len(frames), "n_dets": n_det,
            "stride": stride, "detector": det_w, "patch": patch_w, "grayscale": grayscale}
    pickle.dump({"meta": meta, "frames": frames}, open(OUT / f"{name}.pkl", "wb"))
    print(f"  [{name}] {len(frames)} frames, {n_det} dets, stride={stride}, "
          f"{len(frames)/max(time.time()-t0,0.01):.1f} fps -> {name}.pkl")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    print(f"Phase A: caching {len(SURFACES)} surfaces, ~{args.target} strided frames each")
    print(f"feature dim = {INPUT_DIM}\n")
    yolo_cache, patch_cache = {}, {}
    done = 0
    for (name, path, modality, det, patch, conf, imgsz, rule, hd, gc, gray, pref) in SURFACES:
        if (OUT / f"{name}.pkl").exists() and not args.overwrite:
            print(f"  [skip:cached] {name}"); done += 1; continue
        try:
            if cache_surface(name, path, modality, det, patch, conf, imgsz, rule,
                             hd, gc, gray, pref, args.target, yolo_cache, patch_cache):
                done += 1
        except Exception:
            print(f"  [SURFACE-ERR] {name}\n{traceback.format_exc()}")
    print(f"\nPhase A done: {done}/{len(SURFACES)} surfaces cached in {OUT}")


if __name__ == "__main__":
    main()
