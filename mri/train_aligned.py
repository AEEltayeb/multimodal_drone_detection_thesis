"""
mri.train_aligned — the SYNTHESIS verifier: thermal-deployable, recall-safe AND
confuser-rich, by harvesting confusers from the abundant RGB->gray corpus and
domain-aligning them into thermal feature space.

Why this works (proven by mri.modality_align): the gray<->thermal feature gap is
a per-feature affine offset; per-modality z-score rescues gray->thermal transfer
from 0.50 to 0.919. So:
  positives = thermal drones (native, abundant; recall-safety via drone diversity)
  negatives = grayscale-harvested confusers (abundant) + the few real thermal ones
Each modality is z-scored to ITS OWN mean/std before concatenation, so the MLP
learns drone-vs-confuser in the shared aligned space, NOT the modality offset.

Deployment is on THERMAL. To keep the checkpoint loadable by the unchanged
MLPVerifier (one affine scaler applied to raw features), we COMPOSE the thermal
per-modality z-score with MLPWrapper's internal scaler into a single affine and
save that — so raw_thermal -> composed_scaler -> net reproduces training exactly.

CBAM is held OUT of training -> validate on it to test whether gray-harvested
confusers generalize to novel thermal aerial confusers.
"""
from __future__ import annotations
import argparse, json, time, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import cv2
from ultralytics import YOLO
from sklearn.preprocessing import StandardScaler
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from mri.extract import FeatureExtractor                 # noqa: E402
from mri.modality_align import mine_multi                # noqa: E402 (reuse miner)
from mri.modality_probe import mine                      # noqa: E402 (per-source/prefix miner)
from mri.classifier import MLPWrapper                    # noqa: E402

V3B = str(REPO / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt")
OUT = REPO / "mri" / "results" / "ir_aligned"
CONF_PRE = ("IR_BIRD_", "IR_AIRPLANE_", "IR_HELICOPTER_")

# (dir, prefix, imgsz) per group. CBAM is deliberately ABSENT (held out for eval).
GROUPS = {
    ("thermal", "drone"): [
        ("G:/drone/svanstrom_paired/IR/images", "IR_DRONE_", 1280),
        ("G:/drone/Anti-UAV-RGBT_yolo_converted/val/IR/images", "", 640),
        ("G:/drone/IR_dset_final/train/images", "", 640),
        ("G:/drone/IR_video_ir_dataset/train/images", "IR_DRONE_", 640),
        ("models/ir/corrective_finetune/dataset_v3/train/images", "", 640)],
    ("thermal", "conf"): [
        ("G:/drone/svanstrom_paired/IR/images", CONF_PRE, 1280),
        ("G:/drone/IR_video_ir_dataset/train/images", CONF_PRE, 640)],
    ("gray", "drone"): [
        ("G:/drone/svanstrom_paired/RGB/images", "IR_DRONE_", 1280),
        ("G:/drone/Anti-UAV-RGBT_yolo_converted/val/RGB/images", "", 640),
        ("G:/drone/dataset/dataset/images/train", "", 640)],
    ("gray", "conf"): [
        ("G:/drone/rgb_confusers_merged/images/train", "", 640),
        ("G:/drone/RGB_video_rgb_dataset/train/images",
         ("V_BIRD_", "V_AIRPLANE_", "V_HELICOPTER_"), 640),
        ("G:/drone/svanstrom_paired/RGB/images", CONF_PRE, 1280)],
}
CAP = {("thermal", "drone"): 9000, ("thermal", "conf"): 1800,
       ("gray", "drone"): 5000, ("gray", "conf"): 7000}
WEIGHT = {"thermal": 1.5, "gray": 1.0}   # bias the boundary toward the deploy modality

# Per-group raw-feature cache: re-runs REUSE unchanged groups instead of re-mining
# (gray ~67 min + thermal-drone ~12 min are flag-independent). Delete this dir to
# force a fresh mine if the source lists change.
FEAT_CACHE = REPO / "mri" / "results" / "_feat_cache"


def _cached(key, fn):
    FEAT_CACHE.mkdir(parents=True, exist_ok=True)
    fp = FEAT_CACHE / f"{key}.npy"
    if fp.exists():
        X = np.load(fp); print(f"  [cache hit] {key}: {len(X)} (skipped mining)"); return X
    X = fn(); np.save(fp, X); return X

# --thermal-confusers: mine the thermal confuser pool BALANCED by (category x size),
# adding the THERMAL-NATIVE IR_confusers TRAIN split (the airplane hole; grayscale-
# harvested confusers are OOD to thermal airplanes). The thermal confuser FPs that
# occur are SMALL (cache: 40% <16px), so we cap per (category x size) cell, not per
# category only. Held OUT for eval: IR_confusers val/ + test/ (+ CBAM).
# Audit the mineable cells first to set the cap: py eval/ir_confuser_mine_audit.py
IR_CONF_SOURCES = [
    ("G:/drone/IR_confusers/images/train", 640),       # NEW thermal-native (airplane 3984 / bird 1140 / heli 113)
    ("G:/drone/svanstrom_paired/IR/images", 1280),     # existing svan thermal confusers
    ("G:/drone/IR_video_ir_dataset/train/images", 640),
]
IR_CONF_CATS = {                                         # category -> filename prefixes
    "airplane":   ("airplane_", "IR_AIRPLANE_"),
    "bird":       ("bird_", "IR_BIRD_"),
    "helicopter": ("helicopter_", "IR_HELICOPTER_"),
}
CONF_SIZE_EDGES = (16, 32, 64)                           # detection short-side px
CONF_SIZE_NAMES = ("xs", "s", "m", "l")

# --cbam: add the CBAM TRAIN split (classes Bird/Drone/airPlane) as GT-AWARE thermal
# data -- fires matching a class-D (drone) GT box -> DRONE positives, else -> confusers.
# Closes the held-out CBAM drone-recall collapse (CBAM drones ARE separable from CBAM
# airplanes, AUROC 0.964 -> MOVABLE). CBAM-VALID stays held out for eval (disjoint split).
CBAM_TRAIN_IMG = "G:/drone/Infrared_bird_drone_airplane_CBAM_TF-Net.v1i.yolo26-maha-daxhh-cbam_tf-net/train/images"
CBAM_TRAIN_LBL = "G:/drone/Infrared_bird_drone_airplane_CBAM_TF-Net.v1i.yolo26-maha-daxhh-cbam_tf-net/train/labels"
CBAM_DRONE_CLASS = 1   # data.yaml names ['B','D','P'] -> D = drone


def _size_bin_from_logarea(log_area: float) -> int:
    px = float(np.sqrt(np.exp(log_area)))               # meta-first idx[1]; equiv square side
    for i, e in enumerate(CONF_SIZE_EDGES):
        if px < e:
            return i
    return 3


def mine_thermal_confusers_balanced(ex, per_cell_cap, stride, conf, device, mine_cap=5000):
    """Mine the thermal confuser pool BALANCED by (category x size): mine each
    (source, category-prefix), bucket every fire by detection size (idx[1]=log_area),
    then subsample each (category x size) cell to per_cell_cap (scarce cells keep all).
    Returns FULL 517-D features; prints the achieved kept/mined grid for verification."""
    rng = np.random.RandomState(0)
    by_cell = defaultdict(list)
    for d, imgsz in IR_CONF_SOURCES:
        if not Path(d).exists():
            print(f"    [skip missing source: {d}]"); continue
        for cat, prefs in IR_CONF_CATS.items():
            for pre in prefs:
                X = mine(ex, Path(d), pre, False, mine_cap, stride, imgsz, conf, device)
                for row in X:
                    by_cell[(cat, _size_bin_from_logarea(row[1]))].append(row)
    parts = []
    print("    thermal-confuser cells kept/mined (category x size):")
    for cat in IR_CONF_CATS:
        cells = []
        for sb in range(4):
            rows = by_cell.get((cat, sb), [])
            arr = np.array(rows, np.float32) if rows else np.zeros((0, ex.schema.total_dim), np.float32)
            if len(arr) > per_cell_cap:
                arr = arr[rng.choice(len(arr), per_cell_cap, replace=False)]
            if len(arr):
                parts.append(arr)
            cells.append(f"{CONF_SIZE_NAMES[sb]}={len(arr)}/{len(rows)}")
        print(f"      {cat:<11} " + "  ".join(cells))
    return np.vstack(parts) if parts else np.zeros((0, ex.schema.total_dim), np.float32)


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0., x2 - x1) * max(0., y2 - y1)
    aa = (a[2] - a[0]) * (a[3] - a[1]); bb = (b[2] - b[0]) * (b[3] - b[1])
    return inter / max(aa + bb - inter, 1.)


def mine_cbam(ex, cap_drone=4000, cap_conf=4000, stride=2, conf=0.25, imgsz=640, device="cuda"):
    """GT-aware CBAM-TRAIN miner: fires matching a class-D drone GT box -> DRONE,
    else -> confuser. Returns (X_drone, X_conf), FULL 517-D. Verbose + ETA."""
    img_dir, lbl_dir = Path(CBAM_TRAIN_IMG), Path(CBAM_TRAIN_LBL)
    empty = np.zeros((0, ex.schema.total_dim), np.float32)
    if not img_dir.exists():
        print(f"    [skip CBAM: {img_dir} missing]"); return empty, empty
    imgs = sorted(p for p in img_dir.iterdir()
                  if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))[::stride]
    print(f"    [cbam] {len(imgs)} CBAM-train imgs (drone class {CBAM_DRONE_CLASS}); caps d={cap_drone}/c={cap_conf}")
    Xd, Xc = [], []; t0 = time.time(); n = 0
    for i, p in enumerate(imgs):
        if len(Xd) >= cap_drone and len(Xc) >= cap_conf:
            break
        im = cv2.imread(str(p))
        if im is None:
            continue
        n += 1; ih, iw = im.shape[:2]
        gt = []
        lp = lbl_dir / (p.stem + ".txt")
        if lp.exists():
            for ln in lp.read_text().splitlines():
                t = ln.split()
                if len(t) >= 5 and int(float(t[0])) == CBAM_DRONE_CLASS:
                    xc, yc, bw, bh = map(float, t[1:5])
                    gt.append(((xc-bw/2)*iw, (yc-bh/2)*ih, (xc+bw/2)*iw, (yc+bh/2)*ih))
        ex.hook.clear()
        r = ex.model.predict(im, imgsz=imgsz, conf=conf, verbose=False, device=device)[0]
        if r.boxes is None:
            continue
        for j in range(len(r.boxes)):
            box = tuple(r.boxes.xyxy[j].cpu().numpy().tolist())
            is_d = bool(gt) and max((_iou(box, g) for g in gt), default=0) >= 0.5
            if is_d and len(Xd) < cap_drone:
                Xd.append(ex.extract_one(box, float(r.boxes.conf[j]), (ih, iw)))
            elif (not is_d) and len(Xc) < cap_conf:
                Xc.append(ex.extract_one(box, float(r.boxes.conf[j]), (ih, iw)))
        if n % 200 == 0:
            fps = n / (time.time() - t0)
            print(f"      [{i+1}/{len(imgs)}] {len(Xd)}d/{len(Xc)}c | {fps:.1f} img/s | "
                  f"ETA ~{(len(imgs)-i-1)/max(fps,0.1):.0f}s", flush=True)
    print(f"    [cbam] done: {len(Xd)} drones + {len(Xc)} confusers ({n} imgs, {time.time()-t0:.0f}s)")
    return (np.array(Xd, np.float32) if Xd else empty), (np.array(Xc, np.float32) if Xc else empty)


def main(no_gray: bool = False, thermal_confusers: bool = False,
         conf_cell_cap: int = 1000, cbam: bool = False, out: Path = OUT):
    out.mkdir(parents=True, exist_ok=True)
    mode = "THERMAL-ONLY (no grayscale confusers; A/B counterfactual)" if no_gray else \
           "thermal + grayscale-harvested confusers (production)"
    if thermal_confusers:
        mode += " + IR_confusers TRAIN (thermal-native, balanced by category x size)"
    if cbam:
        mode += " + CBAM-TRAIN (GT-aware drones+confusers; CBAM-valid held out)"
    print(f"== train_aligned [{mode}] ==\n  detector: {V3B}\n  out: {out}")

    groups = {k: list(v) for k, v in GROUPS.items()}
    caps = dict(CAP)
    if thermal_confusers:
        print(f"  [+thermal-confusers] thermal confusers mined BALANCED by (category x size), "
              f"per-cell cap {conf_cell_cap}; sources: IR_confusers train + svan IR + IR_video. "
              f"HELD OUT for eval: IR_confusers val/ + test/ (+ CBAM).")

    yolo = YOLO(V3B)
    ex = FeatureExtractor(yolo, layers=("p3", "p5"), grids={"p3": (2, 2), "p5": (1, 1)})

    data = {}
    for (mod, kl), specs in groups.items():
        if no_gray and mod == "gray":
            continue                   # A/B: drop the grayscale-harvested groups entirely
        t0 = time.time()
        if thermal_confusers and (mod, kl) == ("thermal", "conf"):
            X = _cached(f"thermal_conf_balanced_cap{conf_cell_cap}",
                        lambda: mine_thermal_confusers_balanced(ex, conf_cell_cap, 2, 0.25, "cuda"))
        else:
            key = f"{mod}_{kl}_base" if (mod, kl) == ("thermal", "conf") else f"{mod}_{kl}"
            X = _cached(key, (lambda specs=specs, mk=(mod, kl):
                              mine_multi(ex, specs, grayscale=(mk[0] == "gray"),
                                         cap=caps[mk], stride=2, conf=0.25, device="cuda")))
        data[(mod, kl)] = X            # FULL 517-D (deployable via MLPVerifier)
        print(f"  {mod:7} {kl:5} n={len(X):6}  ({time.time()-t0:.0f}s)")

    if cbam:                            # GT-aware CBAM-TRAIN -> append to the thermal groups
        cd, cc = FEAT_CACHE / "cbam_drone.npy", FEAT_CACHE / "cbam_conf.npy"
        if cd.exists() and cc.exists():
            Xcd, Xcc = np.load(cd), np.load(cc); print(f"  [cache hit] cbam: {len(Xcd)}d/{len(Xcc)}c")
        else:
            Xcd, Xcc = mine_cbam(ex, cap_drone=4000, cap_conf=4000, stride=2, conf=0.25, imgsz=640, device="cuda")
            FEAT_CACHE.mkdir(parents=True, exist_ok=True); np.save(cd, Xcd); np.save(cc, Xcc)
        if len(Xcd):
            data[("thermal", "drone")] = np.vstack([data[("thermal", "drone")], Xcd])
        if len(Xcc):
            data[("thermal", "conf")] = np.vstack([data[("thermal", "conf")], Xcc])
        print(f"  [+cbam] +{len(Xcd)} CBAM drones, +{len(Xcc)} CBAM confusers -> "
              f"thermal drone {len(data[('thermal','drone')])} / conf {len(data[('thermal','conf')])}")
    ex.close()

    # per-modality z-score, then concat into the shared aligned space (thermal only if --no-gray)
    Xt = np.vstack([data[("thermal", "drone")], data[("thermal", "conf")]])
    sct = StandardScaler().fit(Xt)
    Xtz = sct.transform(Xt)
    yt = np.r_[np.ones(len(data[("thermal", "drone")])), np.zeros(len(data[("thermal", "conf")]))]
    wt = np.full(len(yt), WEIGHT["thermal"])
    if no_gray:
        X = Xtz.astype(np.float32); y = yt.astype(np.float32); w = wt.astype(np.float32)
        print(f"\n  train set {X.shape}: {int(y.sum())} drone / {int((1-y).sum())} confuser "
              f"(thermal-only; grayscale groups dropped)")
    else:
        Xg = np.vstack([data[("gray", "drone")], data[("gray", "conf")]])
        scg = StandardScaler().fit(Xg)
        Xgz = scg.transform(Xg)
        yg = np.r_[np.ones(len(data[("gray", "drone")])), np.zeros(len(data[("gray", "conf")]))]
        wg = np.full(len(yg), WEIGHT["gray"])
        X = np.vstack([Xtz, Xgz]).astype(np.float32)
        y = np.r_[yt, yg].astype(np.float32)
        w = np.r_[wt, wg].astype(np.float32)
        print(f"\n  train set {X.shape}: {int(y.sum())} drone / {int((1-y).sum())} confuser "
              f"(thermal {len(yt)}, gray {len(yg)})")

    mlp = MLPWrapper(input_dim=X.shape[1], device="cuda", epochs=120)
    mlp.fit(X, y, sample_weight=w)

    # Compose thermal per-modality z-score (sct) with MLPWrapper's internal scaler
    # so deploy on RAW thermal reproduces training: raw -> sct -> mlp.scaler -> net.
    m_t, s_t = sct.mean_.astype(np.float32), sct.scale_.astype(np.float32)
    m_w, s_w = mlp.scaler.mean_.astype(np.float32), mlp.scaler.scale_.astype(np.float32)
    base = {
        "state_dict": mlp.net.state_dict(),
        "input_dim": int(mlp.input_dim), "hidden_dims": list(mlp.hidden_dims),
        "threshold": 0.5, "cv_f1": -1.0, "cv_std": 0.0,
        "use_batchnorm": mlp.use_batchnorm, "dropout": mlp.dropout,
        "feature_schema": ex.schema.to_dict(),
        "metadata_order": ["conf", "log_area", "aspect", "rel_cx", "rel_cy"],
        "p3_grid": [2, 2], "p5_grid": [1, 1],
    }
    (out / "classifiers").mkdir(exist_ok=True)
    tnote = ("aligned verifier (THERMAL deploy, NO-GRAY A/B). thermal drones + thermal confusers "
             "ONLY; scaler = compose(thermal_zscore, mlp_scaler). CBAM held out.") if no_gray else \
            ("aligned verifier (THERMAL deploy). thermal drones + per-modality-z gray confusers; "
             "scaler = compose(thermal_zscore, mlp_scaler). CBAM held out.")
    torch.save({**base,
                "scaler_mean": torch.from_numpy(m_t + s_t * m_w),
                "scaler_scale": torch.from_numpy(s_t * s_w),
                "note": tnote},
               out / "classifiers" / "mlp_aligned.pt")
    if not no_gray:
        # Same net, gray deploy scaler (per-modality z-score composed with mlp.scaler):
        #   gray: raw_gray -> scg -> mlp.scaler -> net   (deploy on grayscale-RGB)
        m_g, s_g = scg.mean_.astype(np.float32), scg.scale_.astype(np.float32)
        torch.save({**base,
                    "scaler_mean": torch.from_numpy(m_g + s_g * m_w),
                    "scaler_scale": torch.from_numpy(s_g * s_w),
                    "note": "aligned verifier (GRAYSCALE deploy). SAME net, gray per-modality scaler. "
                            "Use with --grayscale-input."},
                   out / "classifiers" / "mlp_aligned_gray.pt")
    (out / "train_meta.json").write_text(json.dumps({
        "no_gray": no_gray,
        "counts": {f"{m}_{k}": int(len(data[(m, k)])) for (m, k) in data},
        "n_train": int(len(X)), "weight": WEIGHT,
    }, indent=2))
    print(f"\n  saved {out/'classifiers'/'mlp_aligned.pt'}")
    print("  validate on CBAM-valid (held out of training):")
    print(f"    py -m mri.cli --yolo {V3B} --config mri/configs/ab_cbam_heldout.yaml \\")
    print(f"       --holdout-eval {out/'classifiers'/'mlp_aligned.pt'} \\")
    print("       --conf 0.40 --mlp-thr 0.05 --match-rule iop --imgsz 640 --device cuda --out mri/results/ab_nogray_cbam")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train the aligned IR confuser filter (synthesis verifier).")
    ap.add_argument("--no-gray", action="store_true",
                    help="A/B counterfactual: drop grayscale-harvested confusers; thermal drones + "
                         "thermal confusers only (CBAM still held out). Writes to mri/results/ir_aligned_nogray.")
    ap.add_argument("--thermal-confusers", action="store_true",
                    help="Airplane-hole fix: mine thermal confusers BALANCED by (category x size) incl. the "
                         "IR_confusers TRAIN split (val/+test/ held out). Writes mri/results/ir_aligned_balanced.")
    ap.add_argument("--conf-cell-cap", type=int, default=1000,
                    help="Max thermal confuser fires per (category x size) cell (set from ir_confuser_mine_audit.py).")
    ap.add_argument("--cbam", action="store_true",
                    help="Add the CBAM TRAIN split as GT-aware thermal drones+confusers (closes the held-out "
                         "CBAM drone-recall collapse; CBAM-valid stays held out). Writes mri/results/ir_aligned_cbam.")
    ap.add_argument("--out", default=None, help="Output dir (default mri/results/ir_aligned[_nogray|_balanced|_cbam]).")
    a = ap.parse_args()
    default_name = ("ir_aligned_nogray" if a.no_gray else
                    "ir_aligned_cbam" if a.cbam else
                    "ir_aligned_balanced" if a.thermal_confusers else "ir_aligned")
    out_dir = Path(a.out) if a.out else (REPO / "mri" / "results" / default_name)
    main(no_gray=a.no_gray, thermal_confusers=a.thermal_confusers,
         conf_cell_cap=a.conf_cell_cap, cbam=a.cbam, out=out_dir)
