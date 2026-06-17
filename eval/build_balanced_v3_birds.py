"""build_balanced_v3_birds.py — v3: add BIRD confusers to the v2 cache + retrain,
to fix the bird-FP leak. Birds are separable from drones (AUROC 0.981), so this
teaches rejection WITHOUT touching drone recall. Reuses the v2 cache (fast — only
mines birds). VERBOSE: prints progress + ETA so you never wait blind.

  py -u eval/build_balanced_v3_birds.py            # -u = live (unbuffered) output
  py -u eval/build_balanced_v3_birds.py --bird-cap 4000 --bird-weight 1.5

Outputs: eval/results/_v5_balanced_v3/classifiers/mlp_v5_balanced_v3.pt
Then:    py eval/filter_acceptance_eval.py --mode rgb --candidate <that path>
"""
from __future__ import annotations
import argparse, time, sys, json
from pathlib import Path
import numpy as np
import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))
import distill_v5_p3p5_ft4 as D
import distill_v5_swap_selcom as S          # train_v5_mlp
from ultralytics import YOLO                 # noqa: E402

V2 = REPO / "eval/results/_v5_balanced_v2/training_data.npz"
OUT = REPO / "eval/results/_v5_balanced_v3"
# TRAINING bird sources only. rgb_bird_confuser EVAL = G:/drone/bird.v1i.yolo26-birds-...
# (a DISJOINT Roboflow set) — none of these prefixes touch it, so no train/eval leak.
BIRD_SOURCES = [
    ("G:/drone/rgb_confusers_merged/images/train", ("svan_bird", "raihanbird"), 640),  # 4586 + 1187 imgs
    ("G:/drone/RGB_video_rgb_dataset/train/images", ("V_BIRD_",), 640),                # 1265 imgs
]


def log(m): print(m, flush=True)   # flush => live output even if piped to a file


def mine_birds(model, hook, bird_cap):
    imgs = []
    for d, prefs, imgsz in BIRD_SOURCES:
        dd = Path(d)
        if not dd.exists():
            log(f"  [skip missing source: {d}]"); continue
        these = [(p, imgsz) for p in sorted(dd.iterdir())
                 if D.is_jpg(p) and any(p.name.startswith(pre) for pre in prefs)]
        log(f"  source {d}: {len(these)} bird images")
        imgs += these
    log(f"[mine birds] {len(imgs)} bird images total; target {bird_cap} fires @conf {D.CONF_THR}, imgsz 640")
    feats = []; t0 = time.time(); n_proc = 0
    for i, (p, imgsz) in enumerate(imgs):
        if len(feats) >= bird_cap:
            log(f"  reached bird cap {bird_cap} at image {i}/{len(imgs)}"); break
        im = cv2.imread(str(p))
        if im is None:
            continue
        n_proc += 1; ih, iw = im.shape[:2]
        hook.clear()
        r = model.predict(im, imgsz=imgsz, conf=D.CONF_THR, verbose=False, device="cuda")[0]
        if r.boxes is not None:
            for j in range(len(r.boxes)):
                box = tuple(r.boxes.xyxy[j].cpu().numpy().tolist())
                feats.append(D._extract_detection_features(hook, box, (ih, iw), float(r.boxes.conf[j])))
        if n_proc % 200 == 0:
            el = time.time() - t0; fps = n_proc / el; rem = len(imgs) - i - 1
            log(f"  [{i+1}/{len(imgs)}] {len(feats)} bird fires | {fps:.1f} img/s | ETA ~{rem/max(fps,0.1):.0f}s")
    log(f"  done: {len(feats)} bird fires from {n_proc} images in {time.time()-t0:.0f}s")
    return np.array(feats, np.float32) if feats else np.empty((0, D.INPUT_DIM), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bird-cap", type=int, default=4000)
    ap.add_argument("--bird-weight", type=float, default=1.5)
    a = ap.parse_args()
    (OUT / "classifiers").mkdir(parents=True, exist_ok=True)

    log("=" * 64); log("  v3: ADD BIRD CONFUSERS to the v2 cache + retrain"); log("=" * 64)
    t_all = time.time()
    log(f"[1/4] loading v2 cache: {V2}")
    z = np.load(V2); X, y, w = z["X"].astype(np.float32), z["y"].astype(np.float32), z["w"].astype(np.float32)
    log(f"      v2 corpus: {len(X)} ({int((y==1).sum())} drone / {int((y==0).sum())} confuser)")

    log(f"[2/4] loading FT4 detector + mining birds (the long step; ETA printed below)...")
    model = YOLO(D.MODEL_PATHS["ft4_r3"]); hook = D.DetectInputHook(); h = hook.register(model)
    try:
        Xbird = mine_birds(model, hook, a.bird_cap)
    finally:
        h.remove()

    log(f"[3/4] adding {len(Xbird)} bird confusers (y=0, weight {a.bird_weight}) and shuffling")
    X = np.vstack([X, Xbird]); y = np.r_[y, np.zeros(len(Xbird), np.float32)]
    w = np.r_[w, np.full(len(Xbird), a.bird_weight, np.float32)]
    rng = np.random.RandomState(D.SEED); pm = rng.permutation(len(X)); X, y, w = X[pm], y[pm], w[pm]
    log(f"      v3 corpus: {len(X)} ({int((y==1).sum())} drone / {int((y==0).sum())} confuser)  "
        f"[drone:conf ratio {(y==1).sum()/max((y==0).sum(),1):.2f}]")
    np.savez_compressed(OUT / "training_data.npz", X=X, y=y, w=w)
    (OUT / "training_meta.json").write_text(json.dumps({
        "variant": "balanced_v3_birds", "from": str(V2),
        "n_bird_confusers_added": int(len(Xbird)), "bird_weight": a.bird_weight,
        "n_total": int(len(X)), "n_drone": int((y == 1).sum()), "n_confuser": int((y == 0).sum()),
    }, indent=2))

    log(f"[4/4] training MLP (V5 arch, 5-fold CV — ~3-6 min)...")
    S.train_v5_mlp(X, y, w, OUT / "classifiers" / "mlp_v5_balanced_v3.pt")
    log(f"\nDONE in {time.time()-t_all:.0f}s  ->  {OUT/'classifiers'/'mlp_v5_balanced_v3.pt'}")
    log(f"NEXT: py eval/filter_acceptance_eval.py --mode rgb --candidate {OUT/'classifiers'/'mlp_v5_balanced_v3.pt'}")


if __name__ == "__main__":
    main()
