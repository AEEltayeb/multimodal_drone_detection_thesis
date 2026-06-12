"""eval_rgb_video_ood.py - OLD CNN-patch vs NEW MLP on RGB OOD *video* clips (FT4).

The RGB mirror of domain-3: drone_video_tests/rgb has confuser clips
(airplanes/birds/helicopters -> suppression) AND drone clips (-> recall), so we get
both Pareto axes. One GPU pass caches per-det (conf, P(drone) via mlp_v5, P(confuser)
via CNN patch v2, best-IoP-to-GT); the sweep is then offline. IoP@0.5 (video convention).

  py -u eval/eval_rgb_video_ood.py            # GPU cache (if missing) + sweep
  py -u eval/eval_rgb_video_ood.py --resweep  # offline sweep only
"""
from __future__ import annotations
import argparse, pickle, sys, time
from pathlib import Path
import cv2, numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval")); sys.path.insert(0, str(REPO / "classifier"))
from distill_v5_p3p5_ft4 import DetectInputHook, _extract_detection_features  # noqa: E402
from mlp_verifier import MLPVerifier        # noqa: E402
from patch_verifier import PatchVerifier    # noqa: E402
from metrics import iou_iop                  # noqa: E402

FT4 = str(REPO / "models/rgb/Yolo26n_selcom_confuser_ft4_1280/weights/best.pt")
MLP = str(REPO / "models/verifiers/rgb_v5/mlp_v5.pt")
PATCH = str(REPO / "models/patches/confuser_filter4_rgb_v2_backup.pt")
VID_ROOT = REPO / "datasets" / "drone detection video tests" / "rgb"
OUT = REPO / "eval" / "results" / "_rgb_video_ood"; OUT.mkdir(parents=True, exist_ok=True)
CATS = ["airplanes", "birds", "helicopters", "drone"]
IMGSZ, CONF = 1280, 0.25
THRS = [0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


def clip_dirs(cat):
    d = VID_ROOT / cat
    return sorted(p for p in d.iterdir() if p.is_dir()) if d.exists() else []


def imgs_lbls(clip):
    for sub in ("images/test", "images"):
        idir = clip / sub
        if idir.exists():
            ldir = clip / ("labels/test" if sub == "images/test" else "labels")
            ims = sorted(p for p in idir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
            return ims, ldir
    return [], None


def gt0(lbl, w, h):
    out = []
    if lbl and lbl.exists():
        for ln in lbl.read_text().splitlines():
            p = ln.split()
            if len(p) >= 5 and p[0] == "0":
                cx, cy, bw, bh = map(float, p[1:5])
                out.append(((cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h))
    return out


def cache():
    from ultralytics import YOLO
    y = YOLO(FT4); hk = DetectInputHook(); hk.register(y)
    mlp = MLPVerifier(MLP, device="cuda"); patch = PatchVerifier(PATCH)
    frames = []; t0 = time.time(); n = 0
    for cat in CATS:
        for clip in clip_dirs(cat):
            ims, ldir = imgs_lbls(clip)
            for ip in ims:
                img = cv2.imread(str(ip))
                if img is None:
                    continue
                h, w = img.shape[:2]
                hk.clear()
                r = y.predict(img, imgsz=IMGSZ, conf=CONF, verbose=False, device="cuda")
                b = r[0].boxes
                gts = gt0(ldir / f"{ip.stem}.txt", w, h) if cat == "drone" else []
                dets = []
                if b is not None and len(b):
                    boxes = [tuple(b.xyxy[i].cpu().numpy().tolist()) for i in range(len(b))]
                    confs = [float(b.conf[i]) for i in range(len(b))]
                    feats = np.stack([_extract_detection_features(hk, db, (h, w), dc)
                                      for db, dc in zip(boxes, confs)]).astype(np.float32)
                    pdr = mlp.predict_drone_probs(feats)
                    pp = np.asarray(patch.predict_boxes(img, boxes), np.float32)
                    for i, db in enumerate(boxes):
                        biop, bgt = 0.0, -1
                        for gi, g in enumerate(gts):
                            ip_ = iou_iop(db, g)[1]
                            if ip_ > biop:
                                biop, bgt = ip_, gi
                        dets.append((float(confs[i]), float(pdr[i]), float(pp[i]),
                                     round(biop, 4), bgt))
                frames.append({"cat": cat, "is_conf": cat != "drone",
                               "dets": dets, "n_gt": len(gts)})
                n += 1
                if n % 300 == 0:
                    print(f"  {n} frames  {n/(time.time()-t0):.1f} fps", flush=True)
    pickle.dump(frames, open(OUT / "perdet.pkl", "wb"))
    print(f"cached {len(frames)} frames -> perdet.pkl ({(time.time()-t0)/60:.1f} min)", flush=True)
    return frames


def recall_at(frames, keep):
    tp = gt = 0
    for f in frames:
        if not f["is_conf"]:
            gt += f["n_gt"]
            taken = set()
            kept = [d for d in f["dets"] if keep(d)]
            for d in sorted(kept, key=lambda d: -d[3]):
                if d[3] >= 0.5 and d[4] >= 0 and d[4] not in taken:
                    taken.add(d[4]); tp += 1
    return tp / gt if gt else 0.0


def suppr_at(frames, keep, cat=None):
    before = after = 0
    for f in frames:
        if not f["is_conf"] or (cat and f["cat"] != cat):
            continue
        before += len(f["dets"]); after += sum(1 for d in f["dets"] if keep(d))
    return (1 - after/before if before else 0.0), before


def sweep(frames):
    cats = ["airplanes", "birds", "helicopters"]
    def report(title, kf):
        print(f"\n=== {title} ===")
        print(f"  {'thr':>5} {'ALLsuppr':>9} " + " ".join(f"{c[:4]:>7}" for c in cats) + f" {'droneR':>8}")
        for thr in THRS:
            k = kf(thr)
            alls, _ = suppr_at(frames, k)
            per = [suppr_at(frames, k, c)[0] for c in cats]
            print(f"  {thr:>5.2f} {alls:>8.1%} " + " ".join(f"{p:>6.1%}" for p in per)
                  + f" {recall_at(frames, k):>8.4f}")
    nd = sum(len(f['dets']) for f in frames if f['is_conf'])
    ng = sum(f['n_gt'] for f in frames if not f['is_conf'])
    print(f"\nconfuser dets={nd}  drone GT={ng}  frames={len(frames)}")
    report("NEW MLP (survive if P(drone) >= thr)", lambda t: (lambda d: d[1] >= t))
    report("OLD CNN patch v2 (survive if P(confuser) < thr)", lambda t: (lambda d: d[2] < t))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--resweep", action="store_true")
    a = ap.parse_args()
    p = OUT / "perdet.pkl"
    frames = pickle.load(open(p, "rb")) if (a.resweep and p.exists()) else cache()
    sweep(frames)


if __name__ == "__main__":
    main()
