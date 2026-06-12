"""
thesis_eval/pipeline_cache_unified.py — ONE unified detection cache for the whole thesis evaluation.

Detect ONCE per surface with the production detectors **ft4 (RGB) + v3b (IR)**, then every downstream
eval replays with ZERO GPU. For each kept detection we store {xyxy, conf, 517-D verifier feature,
patch-CNN P(confuser)}; per frame we store the trust-classifier feature rows {f8_all, f32_all}
computed with the SAME build_row the shipped routers (robust8/sa32/robust6) were trained on — so the
classifier replay cannot drift; per modality we store GT boxes (scored per-modality, NEVER unioned);
and we keep the frame's sequence key so the Tier-2 FULL-FRAMES build can later add temporal voting
(impossible on a strided sample).

Reuses compare_routing_pipeline verbatim (gray3, run_det, f8_vec, build_row32/FCOLS32, F8, CONF, FT4,
V3B, parse_yolo_gt, the paired iterators, the 517-D feature hook) + PatchVerifier. Nothing is
reimplemented, so features match the deployed routing classifier exactly. Detectors run at CONF=0.25
for BOTH modalities (the routing classifier's training setting — note the older offline matrix used
IR@0.40; the verifier is threshold-independent so this only matters for the classifier features).

TIER-1 (this run, --target 4000): ~4k frames/surface by EVEN STRIDE (every Nth, NOT first-N), all
frames if a surface is smaller. PER-FRAME only — temporal (C->F->T vs F->C->T) is a PLACEHOLDER,
deferred to Tier-2 because 2-of-3 voting needs consecutive frames.
TIER-2 (--full, later, gated on Tier-1): every consecutive frame -> final numbers + temporal.

  py -u thesis_eval/pipeline_cache_unified.py --target 4000 --overwrite     # Tier-1 strided screening
  py -u thesis_eval/pipeline_cache_unified.py --full --overwrite            # Tier-2 full frames (later)

LOW-CONF SWEEP MODE (2026-06-11): --conf lowers the detector floor (e.g. 0.05) so a downstream
conf-sweep replay (thesis_eval/conf_sweep_replay.py) can test "increased recall + verifier filter"
below the production operating points (rgb 0.25 / ir 0.40). Use --cache-dir to keep these caches
SEPARATE from the Tier-1 cache, and --no-patch to skip the (slow, unneeded) patch-CNN scoring:

  py -u thesis_eval/pipeline_cache_unified.py --conf 0.05 --cache-dir thesis_eval/cache_conf005 --no-patch --target 500 --only antiuav
"""
from __future__ import annotations
import argparse, pickle, time, traceback
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
import sys
for _sub in ("eval", "classifier"):
    sys.path.insert(0, str(REPO / _sub))

# --- reuse routing-harness internals verbatim (guarantees feature parity with the shipped routers) ---
from compare_routing_pipeline import (                       # noqa: E402
    gray3, run_det, f8_vec, build_row32, FCOLS32, F8, CONF, FT4, V3B,
    parse_yolo_gt, iter_antiuav_pairs, iter_svanstrom_pairs,
)
from distill_v5_p3p5_ft4 import DetectInputHook, _extract_detection_features, INPUT_DIM  # noqa: E402
from patch_verifier import PatchVerifier                     # noqa: E402

RGB_PATCH = str(REPO / "models/patches/confuser_filter4_rgb_v2_backup.pt")
IR_PATCH  = str(REPO / "models/patches/confuser_filter4_ir_v2_backup.pt")
OUT = REPO / "thesis_eval" / "cache"
OUT.mkdir(parents=True, exist_ok=True)

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def iter_dir(images_dir):
    d = Path(images_dir)
    imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS) if d.exists() else []
    for p in imgs:
        yield {"key": p.stem, "img": p}


VIDEO_ROOT = REPO / "datasets" / "drone detection video tests" / "rgb"


def iter_video_clips(categories):
    """The 19-clip real-video set: frames are CONSECUTIVE (as extracted) within each clip,
    so the temporal replay can vote on them. seq = category/clip for window grouping."""
    for cat in categories:
        cat_dir = VIDEO_ROOT / cat
        if not cat_dir.exists():
            continue
        for clip in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
            img_dir = clip / "images" / "test"
            if not img_dir.exists():
                continue
            for p in sorted(q for q in img_dir.iterdir() if q.suffix.lower() in EXTS):
                yield {"key": p.stem, "img": p, "seq": f"{cat}/{clip.name}"}


def find_label(img_path):
    """Resolve the YOLO sibling label for an image (handles images/<split>/ and images/ layouts)."""
    ip = Path(img_path)
    for c in (ip.parents[2] / "labels" / ip.parent.name / (ip.stem + ".txt"),     # root/images/<split>/ -> root/labels/<split>/
              ip.parent.parent / "labels" / ip.parent.name / (ip.stem + ".txt"),  # images/<split>/->labels/<split>/
              ip.parent.parent / "labels" / (ip.stem + ".txt"),                    # images/->../labels/
              ip.parent.with_name("labels") / (ip.stem + ".txt")):                 # sibling labels/
        if c.exists():
            return c
    return None


def gt_boxes(lbl, img_path, w, h):
    """Prefer the iterator's explicit label path (authoritative for paired sets); else derive from image."""
    p = lbl if (lbl and Path(lbl).exists()) else find_label(img_path)
    if p is None:
        return []
    try:
        return list(parse_yolo_gt(str(p), w, h))
    except Exception:
        return []


def empty_slot():
    return {"boxes": np.zeros((0, 4), np.float32), "confs": np.zeros(0, np.float32),
            "feats": np.zeros((0, INPUT_DIM), np.float32), "patch": np.zeros(0, np.float32)}


def det_slot(img, dets, feats, patch_vf):
    """dets = list of (x1,y1,x2,y2,conf). Returns the stored per-detection record incl. patch P(confuser).
    patch_vf=None (--no-patch) stores zeros — the sweep replay only uses the MLP verifier."""
    if not dets:
        return empty_slot()
    boxes = np.array([d[:4] for d in dets], np.float32)
    confs = np.array([d[4] for d in dets], np.float32)
    patch = (np.zeros(len(dets), np.float32) if patch_vf is None else
             np.asarray(patch_vf.predict_boxes(img, [tuple(b) for b in boxes.tolist()]), np.float32))
    return {"boxes": boxes, "confs": confs, "feats": feats.astype(np.float32), "patch": patch}


# (name, kind, iter_factory, rgb_imgsz, ir_imgsz, rule, has_drones, gt_class)
#   kind: paired (ft4+v3b) | rgb (ft4) | ir (v3b) | gray (v3b on RGB->gray, stored in ir slot)
# SMALLEST / FASTEST surfaces FIRST -> first .pkl lands in minutes (visible progress + early replay
# validation); the heavy 85k/28k paired sets (antiuav, svanstrom*) run LAST.
SURFACES = [
    ("selcom_val",      "rgb",    lambda: iter_dir("G:/drone/_finetune_selcom_mixed_ft2/images/val"),1280, 640, "iop", True,  0),
    ("rgb_confuser",    "rgb",    lambda: iter_dir("G:/drone/rgb_confusers_merged/images/test"),     640, 640, "iou", False, 0),
    ("gray_confuser",   "gray",   lambda: iter_dir("G:/drone/rgb_confusers_merged/images/test"),     640, 640, "iou", False, 0),
    ("ir_confusers",    "ir",     lambda: iter_dir("G:/drone/IR_confusers/images/train"),            640, 640, "iou", False, 0),
    ("ir_dset_final",   "ir",     lambda: iter_dir("G:/drone/IR_dset_final/test/images"),           640, 640, "iou", True,  0),
    ("rgb_dataset_test","rgb",    lambda: iter_dir("G:/drone/dataset/dataset/images/test"),         640, 640, "iou", True,  0),
    ("antiuav",         "paired", iter_antiuav_pairs,                                              640, 640, "iou", True,  0),
    ("svanstrom",       "paired", iter_svanstrom_pairs,                                            1280, 640, "iop", True,  0),
    ("svanstrom_gray",  "gray",   iter_svanstrom_pairs,                                            1280, 640, "iop", True,  0),
    # rawrgb = v3b fed the UNCONVERTED 3-channel RGB frame: the control leg of the grayscale 3-way
    # (RGB bare vs IR-on-rawRGB vs IR-on-gray). Diagnostic only — not a production mode.
    ("svanstrom_rawrgb", "rawrgb", iter_svanstrom_pairs,                                           1280, 640, "iop", True,  0),
    # grayrgb_paired = the production RGB-VIDEO regime (and the GUI's no-thermal mode): ft4 on the
    # RGB frame + v3b on gray(RGB) as the second channel, is_grayscale=1. CONSECUTIVE frames ->
    # the temporal replay (thesis_eval/temporal_replay.py) votes 2-of-3 on these two surfaces.
    ("video_drone",    "grayrgb_paired", lambda: iter_video_clips(["drone"]),                      640, 640, "iop", True,  0),
    ("video_confuser", "grayrgb_paired", lambda: iter_video_clips(["airplanes", "birds", "helicopters"]), 640, 640, "iop", False, 0),
]


def _read(it, *keys):
    for k in keys:
        if it.get(k):
            img = cv2.imread(str(it[k]))
            if img is not None:
                return img, Path(it[k])
    return None, None


def cache_surface(name, kind, items, rsz, isz, rule, has_drones, gt_class, target, M, full,
                  conf=CONF, out_dir=OUT):
    print(f"  [{name}] building file list (this can take a minute on big sets)...", flush=True)
    items = list(items)
    if not items:
        print(f"  [skip:no-imgs] {name}"); return None
    n_all = len(items)
    if full or n_all <= target:
        st = 1
    else:
        # even spread across the WHOLE sorted list (a floor-stride + first-N cap silently
        # truncates the alphabetical tail — on ir_confusers that dropped bird+heli entirely)
        idx = np.linspace(0, n_all - 1, target).round().astype(int)
        items = [items[i] for i in idx]
        st = max(1, round(n_all / target))
    print(f"  [{name}] {len(items)}/{n_all} frames (eff. stride={st}, even spread, "
          f"imgsz rgb={rsz}/ir={isz}) — detecting...", flush=True)
    is_gray = 1 if kind in ("gray", "grayrgb_paired") else 0   # the IR channel is grayscale-fed
    frames, n_det, t0 = [], 0, time.time()
    for k, it in enumerate(items):
        if k and k % 100 == 0:
            el = time.time() - t0; fps = k / max(el, .01)
            print(f"    [{name}] {k}/{len(items)}  {fps:.1f} fps  {n_det} dets  "
                  f"ETA {(len(items)-k)/max(fps, .01):.0f}s", flush=True)
        try:
            rgb_slot, ir_slot = empty_slot(), empty_slot()
            rgb_gt, ir_gt = [], []
            rgb_g = ir_g = None
            rwh = iwh = (640, 640)

            # ---- RGB detector (ft4) ----
            if kind in ("paired", "rgb", "grayrgb_paired"):
                rgb_img, rgb_p = _read(it, "rgb_img", "img")
                if rgb_img is None and kind == "rgb":
                    continue
                if rgb_img is not None:
                    rh, rw = rgb_img.shape[:2]; rwh = (rw, rh)
                    _, rgb_g = gray3(rgb_img)
                    dets, feats = run_det(M["yr"], M["hr"], rgb_img, rsz)
                    rgb_slot = det_slot(rgb_img, dets, feats, M["pr"]); n_det += len(dets)
                    if has_drones:
                        rgb_gt = gt_boxes(it.get("rgb_lbl"), rgb_p, rw, rh)

            # ---- IR detector (v3b) on real thermal ----
            if kind in ("paired", "ir"):
                ir_img, ir_p = _read(it, "ir_img", "img")
                if ir_img is None and kind == "ir":
                    continue
                if ir_img is not None:
                    ih, iw = ir_img.shape[:2]; iwh = (iw, ih)
                    _, ir_g = gray3(ir_img)
                    dets, feats = run_det(M["yi"], M["hi"], ir_img, isz)
                    ir_slot = det_slot(ir_img, dets, feats, M["pi"]); n_det += len(dets)
                    if has_drones:
                        ir_gt = gt_boxes(it.get("ir_lbl"), ir_p, iw, ih)

            # ---- GRAY path: v3b on RGB->gray (is_grayscale=1). RAWRGB control: v3b on the raw
            # 3-channel RGB frame (no conversion). Both store detections in the IR slot. ----
            if kind in ("gray", "rawrgb"):
                rgb_img, rgb_p = _read(it, "rgb_img", "img")
                if rgb_img is None:
                    continue
                rh, rw = rgb_img.shape[:2]; rwh = iwh = (rw, rh)
                rgb3, rgb_g = gray3(rgb_img); ir_g = rgb_g
                feed = rgb3 if kind == "gray" else rgb_img
                dets, feats = run_det(M["yi"], M["hi"], feed, isz)
                ir_slot = det_slot(feed, dets, feats, M["pr"]); n_det += len(dets)   # RGB-content patch
                if has_drones:
                    rgb_gt = ir_gt = gt_boxes(it.get("rgb_lbl"), rgb_p, rw, rh)      # drone GT lives on the RGB frame

            # ---- GRAYRGB_PAIRED second channel: v3b on gray(RGB), the GUI's no-thermal mode ----
            if kind == "grayrgb_paired" and rgb_g is not None:
                g3, _ = gray3(rgb_img)
                dets, feats = run_det(M["yi"], M["hi"], g3, isz)
                ir_slot = det_slot(g3, dets, feats, M["pr"]); n_det += len(dets)     # RGB-content patch
                ir_g, iwh = rgb_g, rwh
                if has_drones:
                    ir_gt = rgb_gt                                                    # same frame, same GT

            # dummy gray for the absent modality (F8/F32 detection cols are pixel-free; scene cols unused on solo)
            if rgb_g is None:
                rgb_g, rwh = ir_g, iwh
            if ir_g is None:
                ir_g, iwh = rgb_g, rwh

            rgb_dets = [tuple(rgb_slot["boxes"][i].tolist()) + (float(rgb_slot["confs"][i]),) for i in range(len(rgb_slot["confs"]))]
            ir_dets = [tuple(ir_slot["boxes"][i].tolist()) + (float(ir_slot["confs"][i]),) for i in range(len(ir_slot["confs"]))]
            f8 = f8_vec(rgb_dets, ir_dets, rgb_g, ir_g, rwh, iwh, is_gray, k, name)
            r32 = build_row32(rgb_dets, ir_dets, rgb_g, ir_g, rwh, iwh, 0, k, name, conf)
            f32 = [float(r32[f]) for f in FCOLS32]

            frames.append({
                "key": it.get("key", str(k)), "seq": it.get("seq") or _seq(it.get("key", str(k))),
                "rgb": rgb_slot, "ir": ir_slot,
                "rgb_gt": np.array(rgb_gt, np.float32) if rgb_gt else np.zeros((0, 4), np.float32),
                "ir_gt": np.array(ir_gt, np.float32) if ir_gt else np.zeros((0, 4), np.float32),
                "f8_all": np.array(f8, np.float32), "f32_all": np.array(f32, np.float32),
            })
        except Exception:
            print(f"    [frame-err {name} #{k}] {traceback.format_exc(limit=1)}")
    meta = {"name": name, "kind": kind, "rule": rule, "has_drones": has_drones,
            "is_grayscale": is_gray, "rgb_imgsz": rsz, "ir_imgsz": isz, "conf": conf,
            "detector_rgb": FT4 if kind in ("paired", "rgb", "grayrgb_paired") else None,
            "detector_ir": V3B if kind in ("paired", "ir", "gray", "rawrgb", "grayrgb_paired") else None,
            "n": len(frames), "n_source": n_all, "stride": st,
            "tier": "2-full" if full else "1-strided-screening",
            "F8": F8, "F32": FCOLS32}
    pickle.dump({"meta": meta, "frames": frames}, open(out_dir / f"{name}.pkl", "wb"))
    print(f"  [{name}] {len(frames)} frames, {n_det} dets, stride={st}, "
          f"{len(frames)/max(time.time()-t0,.01):.1f} fps -> {name}.pkl")
    return meta


import re
_SEQ = re.compile(r"^(.+?)(?:_f\d+|_frame\d+|_\d{3,})(?:_visible|_infrared)?$", re.I)
def _seq(stem):
    m = _SEQ.match(str(stem)); return m.group(1).rstrip("_") if m else str(stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=4000, help="Tier-1 strided cap (~frames/surface)")
    ap.add_argument("--full", action="store_true", help="Tier-2: every consecutive frame (enables temporal later)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--only", default="", help="comma list of surface names to (re)build")
    ap.add_argument("--conf", type=float, default=None,
                    help=f"detector conf floor override (default {CONF}); use a LOW value (e.g. 0.05) for conf-sweep caches")
    ap.add_argument("--cache-dir", default=None,
                    help="output dir override — keep non-default-conf caches SEPARATE from the Tier-1 cache")
    ap.add_argument("--no-patch", action="store_true",
                    help="skip patch-CNN scoring (stores zeros); big speedup when only the MLP verifier is replayed")
    args = ap.parse_args()

    conf = CONF if args.conf is None else args.conf
    out_dir = OUT if args.cache_dir is None else (REPO / args.cache_dir if not Path(args.cache_dir).is_absolute() else Path(args.cache_dir))
    out_dir.mkdir(parents=True, exist_ok=True)
    if conf != CONF:
        import compare_routing_pipeline as _crp
        _crp.CONF = conf                      # run_det reads the module global
        if out_dir == OUT:
            raise SystemExit("Refusing to write a non-default-conf cache into the Tier-1 cache dir — pass --cache-dir.")

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    print(f"Unified cache -> {out_dir}  | tier={'2-FULL' if args.full else '1-STRIDED ~'+str(args.target)} "
          f"| conf={conf} | patch={'OFF' if args.no_patch else 'on'} | feat_dim={INPUT_DIM}")
    yr = YOLO(FT4); hr = DetectInputHook(); hr.register(yr)
    yi = YOLO(V3B); hi = DetectInputHook(); hi.register(yi)
    M = {"yr": yr, "hr": hr, "yi": yi, "hi": hi,
         "pr": None if args.no_patch else PatchVerifier(RGB_PATCH),
         "pi": None if args.no_patch else PatchVerifier(IR_PATCH)}
    done = 0
    for (name, kind, itf, rsz, isz, rule, hd, gc) in SURFACES:
        if only and name not in only:
            continue
        if (out_dir / f"{name}.pkl").exists() and not args.overwrite:
            print(f"  [skip:cached] {name} (use --overwrite to rebuild)"); done += 1; continue
        try:
            if cache_surface(name, kind, itf(), rsz, isz, rule, hd, gc, args.target, M, args.full,
                             conf=conf, out_dir=out_dir):
                done += 1
        except Exception:
            print(f"  [SURFACE-ERR {name}]\n{traceback.format_exc()}")
    print(f"\nDone: {done}/{len(SURFACES)} surfaces cached in {out_dir}")


if __name__ == "__main__":
    main()
