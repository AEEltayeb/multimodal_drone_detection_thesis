"""pipeline_cache_paired.py - Phase A of the email-table recompute (PAIRED, full-set).

Reproduces the EXACT frame sets of the supervisor email by driving off the same
manifests the email used (classifier/runs/{raw_detections,svanstrom_detections}.json
-> keys + GT label paths), then re-running the NEW detectors with the V5 feature hook:

  RGB = ft4  (Yolo26n_selcom_confuser_ft4_1280)   <- mlp_v5 was distilled from this
  IR  = v3b  (finetune_v3b)                         <- mlp_v5_ir_aligned was distilled from this

For every paired frame it caches, per detection: box(xyxy,px), conf, and the 517-D
V5 feature vector (P3@2x2 + P5@1x1 + 5 meta) for BOTH modalities, plus per-frame GT
boxes. Phase B (pipeline_eval_paired.py) then replays robust6 + the V5 MLP filters and
every config with ZERO GPU (the MLP forward runs on CPU from the cached feats).

Detections are cached at a LOW conf floor (0.10) so Phase B can threshold to the
email's operating points (rgb>=0.25, ir>=0.40) AND feed robust6 its training-faithful
0.25 both-modality dets, without re-running YOLO.

Offline (no internet), GPU-gated (waits for a free GPU before starting), resumable
(sharded; skips shards already on disk).

  py -u eval/pipeline_cache_paired.py --surface both
  py -u eval/pipeline_cache_paired.py --surface antiuav --rgb-imgsz 1280
"""
from __future__ import annotations
import argparse, json, pickle, subprocess, sys, time, traceback
import datetime as dt
from pathlib import Path

import cv2, numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))
sys.path.insert(0, str(REPO / "classifier"))
from distill_v5_p3p5_ft4 import DetectInputHook, _extract_detection_features, INPUT_DIM  # noqa: E402
from mlp_verifier import MLPVerifier  # noqa: E402

FT4 = str(REPO / "models/rgb/Yolo26n_selcom_confuser_ft4_1280/weights/best.pt")
V3B = str(REPO / "models/ir/corrective_finetune/finetune_v3b/weights/best.pt")
MLP_RGB = str(REPO / "models/verifiers/rgb_v5/mlp_v5.pt")
MLP_IR = str(REPO / "models/verifiers/ir_aligned/mlp_aligned.pt")
OUT = REPO / "eval" / "results" / "_email_recompute" / "cache"
OUT.mkdir(parents=True, exist_ok=True)

# per-surface imgsz — matches eval_v4_vs_patch.py (Svan 1280: drones unresolvable
# at 640; everything else 640, the production default). (rgb, ir).
SURFACE_IMGSZ = {"antiuav": (640, 640), "svanstrom": (1280, 640)}

MANIFESTS = {
    "antiuav":   REPO / "classifier/runs/raw_detections.json",
    "svanstrom": REPO / "classifier/runs/svanstrom_detections.json",
}
CONF_FLOOR = 0.10          # cache everything >= this; Phase B re-thresholds
SHARD = 2000               # frames per shard pickle
SVAN_CATS = ("AIRPLANE", "BIRD", "DRONE", "HELICOPTER")


def svan_category(key: str) -> str:
    for c in SVAN_CATS:
        if f"_{c}_" in key:
            return c
    return "OTHER"


def img_from_label(lbl_path: str):
    lbl = Path(lbl_path)
    img_dir = lbl.parent.parent / "images"
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = img_dir / f"{lbl.stem}{ext}"
        if p.exists():
            return p
    return None


def parse_gt(lbl_path: str, w: int, h: int):
    p = Path(lbl_path)
    out = []
    if not p.exists():
        return np.zeros((0, 4), np.float32)
    for ln in p.read_text().splitlines():
        s = ln.split()
        if len(s) >= 5 and s[0] == "0":
            cx, cy, bw, bh = map(float, s[1:5])
            out.append(((cx - bw/2)*w, (cy - bh/2)*h, (cx + bw/2)*w, (cy + bh/2)*h))
    return np.array(out, np.float32) if out else np.zeros((0, 4), np.float32)


def run_det(yolo, hook, img, imgsz, conf, device="cuda"):
    """Return boxes(n,4)f32, confs(n,)f32, feats(n,517)f16 for one image."""
    hook.clear()
    res = yolo.predict(img, imgsz=imgsz, conf=conf, verbose=False, device=device)
    b = res[0].boxes
    h, w = img.shape[:2]
    if b is None or len(b) == 0:
        return (np.zeros((0, 4), np.float32), np.zeros(0, np.float32),
                np.zeros((0, INPUT_DIM), np.float32))
    boxes = [tuple(b.xyxy[i].cpu().numpy().tolist()) for i in range(len(b))]
    confs = [float(b.conf[i]) for i in range(len(b))]
    # float32 is REQUIRED: the MLP scaler divides by per-feature std (some tiny),
    # so float16 rounding gets amplified -> sigmoid saturates to 0 (the v1 bug).
    feats = np.stack([_extract_detection_features(hook, db, (h, w), dc)
                      for db, dc in zip(boxes, confs)]).astype(np.float32)
    return np.array(boxes, np.float32), np.array(confs, np.float32), feats


# ── GPU gate / log ────────────────────────────────────────────────
def gpu_used():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        return int(o.stdout.strip().splitlines()[0])
    except Exception:
        return None


def wait_for_gpu(thr=1600, stable=3, interval=30, timeout=5*3600):
    print(f"[gpu] waiting for <{thr} MB ({stable}x)...", flush=True)
    t0, ok = time.time(), 0
    while time.time() - t0 < timeout:
        u = gpu_used()
        if u is None:
            print("[gpu] no nvidia-smi; proceeding", flush=True); return
        ok = ok + 1 if u < thr else 0
        if ok >= stable:
            print(f"[gpu] free (used={u}); go", flush=True); return
        time.sleep(interval)
    print("[gpu] timeout; proceeding", flush=True)


def cache_surface(surface, rgb_imgsz, ir_imgsz, limit, device="cuda"):
    man = json.loads(MANIFESTS[surface].read_text())
    keys = sorted(man.keys())
    if limit:
        keys = keys[:limit]
    n_shards = (len(keys) + SHARD - 1) // SHARD
    print(f"\n== {surface}: {len(keys):,} frames -> {n_shards} shards "
          f"(rgb@{rgb_imgsz}, ir@{ir_imgsz}, conf>={CONF_FLOOR}) ==", flush=True)

    from ultralytics import YOLO
    rgb_y = YOLO(FT4); rgb_h = DetectInputHook(); rgb_h.register(rgb_y)
    ir_y = YOLO(V3B);  ir_h = DetectInputHook();  ir_h.register(ir_y)
    mlp_rgb = MLPVerifier(MLP_RGB, device=device)
    mlp_ir = MLPVerifier(MLP_IR, device=device)

    for si in range(n_shards):
        shard_path = OUT / f"{surface}_{si:04d}.pkl"
        if shard_path.exists():
            print(f"  [skip:cached] {shard_path.name}", flush=True); continue
        shard_keys = keys[si*SHARD:(si+1)*SHARD]
        frames, n_det, t0 = [], 0, time.time()
        for key in shard_keys:
            try:
                e = man[key]
                rp, ip = img_from_label(e["rgb_lbl"]), img_from_label(e["ir_lbl"])
                if rp is None or ip is None:
                    continue
                rimg, iimg = cv2.imread(str(rp)), cv2.imread(str(ip))
                if rimg is None or iimg is None:
                    continue
                rh, rw = rimg.shape[:2]; ih, iw = iimg.shape[:2]
                rb, rc, rf = run_det(rgb_y, rgb_h, rimg, rgb_imgsz, CONF_FLOOR, device)
                ib, ic, if_ = run_det(ir_y, ir_h, iimg, ir_imgsz, CONF_FLOOR, device)
                n_det += len(rb) + len(ib)
                # precompute P(drone) in f32 now (threshold-independent -> free sweep in B)
                rpd = (mlp_rgb.predict_drone_probs(rf).astype(np.float32)
                       if len(rf) else np.zeros(0, np.float32))
                ipd = (mlp_ir.predict_drone_probs(if_).astype(np.float32)
                       if len(if_) else np.zeros(0, np.float32))
                frames.append({
                    "key": key, "cat": svan_category(key),
                    "rgb": {"boxes": rb, "confs": rc, "feats": rf, "pdrone": rpd},
                    "ir":  {"boxes": ib, "confs": ic, "feats": if_, "pdrone": ipd},
                    "rgb_gt": parse_gt(e["rgb_lbl"], rw, rh),
                    "ir_gt":  parse_gt(e["ir_lbl"], iw, ih),
                })
            except Exception:
                print(f"    [frame-err] {key}\n{traceback.format_exc(limit=1)}", flush=True)
        meta = {"surface": surface, "shard": si, "rgb_imgsz": rgb_imgsz,
                "ir_imgsz": ir_imgsz, "conf_floor": CONF_FLOOR,
                "detector_rgb": FT4, "detector_ir": V3B,
                "n_frames": len(frames), "n_dets": n_det}
        pickle.dump({"meta": meta, "frames": frames}, open(shard_path, "wb"))
        fps = len(frames) / max(time.time() - t0, 0.01)
        print(f"  [{shard_path.name}] {len(frames)} frames, {n_det} dets, "
              f"{fps:.1f} fps", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface", choices=["antiuav", "svanstrom", "both"], default="both")
    ap.add_argument("--rgb-imgsz", type=int, default=0,
                    help="override RGB imgsz (0 = per-surface SURFACE_IMGSZ: antiuav 640, svan 1280)")
    ap.add_argument("--ir-imgsz", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0, help="0 = all frames")
    ap.add_argument("--device", default="cuda", help="cuda | cpu (cpu = smoke test)")
    ap.add_argument("--no-gpu-gate", action="store_true")
    args = ap.parse_args()

    print(f"== pipeline_cache_paired ==  {dt.datetime.now():%Y-%m-%d %H:%M:%S}", flush=True)
    if not args.no_gpu_gate and args.device != "cpu":
        wait_for_gpu()
    limit = args.limit or None
    for s in (["antiuav", "svanstrom"] if args.surface == "both" else [args.surface]):
        rgb_sz, ir_sz = SURFACE_IMGSZ[s]
        if args.rgb_imgsz:
            rgb_sz = args.rgb_imgsz
        try:
            cache_surface(s, rgb_sz, ir_sz, limit, args.device)
        except Exception:
            print(f"[SURFACE-ERR] {s}\n{traceback.format_exc()}", flush=True)
    print(f"\nPhase A done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
