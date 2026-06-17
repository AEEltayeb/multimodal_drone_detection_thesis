"""build_balanced_v4_birdsplit.py — PROPER held-out test of the bird fix.
Split bird.v1i (the rgb_bird_confuser EVAL source) into train/test, train the
filter on the TRAIN birds (in-distribution), eval on the HELD-OUT TEST birds.
v3 failed because it trained on DISJOINT birds (no transfer); this tests whether
in-distribution bird coverage actually fixes the bird FP -- with NO leak (we never
eval on a bird image we trained on). VERBOSE: progress + ETA.

  py -u eval/build_balanced_v4_birdsplit.py            # 60/40 split, seed 0
  py -u eval/build_balanced_v4_birdsplit.py --test-frac 0.4 --bird-weight 1.5

Output: eval/results/_v5_balanced_v4/classifiers/mlp_v5_balanced_v4.pt  (+ split.json)
Then (NON-bird surfaces only, leak-free): py eval/filter_acceptance_eval.py --mode rgb --candidate <that>
  (IGNORE that run's rgb_bird_confuser number — it uses the FULL bird.v1i incl. train = leak;
   the held-out bird-TEST FP is printed by THIS script.)
"""
from __future__ import annotations
import argparse, time, sys, json
from pathlib import Path
import numpy as np
import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))
import distill_v5_p3p5_ft4 as D
import distill_v5_swap_selcom as S                 # train_v5_mlp
from eval_v4_vs_patch import MLPv4Verifier          # noqa: E402
from ultralytics import YOLO                         # noqa: E402

V2_NPZ = REPO / "eval/results/_v5_balanced_v2/training_data.npz"
V2_WT = REPO / "eval/results/_v5_balanced_v2/classifiers/mlp_v5_balanced_v2.pt"
SHIPPED = REPO / "models/verifiers/rgb_v5/mlp_v5.pt"
OUT = REPO / "eval/results/_v5_balanced_v4"
BIRD_DIR = Path("G:/drone/bird.v1i.yolo26-birds-zekpr-bird-pn3pj/train/images")
THR = 0.25


def log(m): print(m, flush=True)


def mine(model, hook, imgs, tag):
    feats = []; t0 = time.time(); n = 0
    log(f"  [mine {tag}] {len(imgs)} images @conf {D.CONF_THR}, imgsz 640")
    for i, p in enumerate(imgs):
        im = cv2.imread(str(p))
        if im is None:
            continue
        n += 1; ih, iw = im.shape[:2]
        hook.clear()
        r = model.predict(im, imgsz=640, conf=D.CONF_THR, verbose=False, device="cuda")[0]
        if r.boxes is not None:
            for j in range(len(r.boxes)):
                box = tuple(r.boxes.xyxy[j].cpu().numpy().tolist())
                feats.append(D._extract_detection_features(hook, box, (ih, iw), float(r.boxes.conf[j])))
        if n % 200 == 0:
            el = time.time() - t0; fps = n / el
            log(f"    [{i+1}/{len(imgs)}] {len(feats)} fires | {fps:.1f} img/s | ETA ~{(len(imgs)-i-1)/max(fps,0.1):.0f}s")
    log(f"  [mine {tag}] done: {len(feats)} fires from {n} imgs in {time.time()-t0:.0f}s")
    return np.array(feats, np.float32) if feats else np.empty((0, D.INPUT_DIM), np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frac", type=float, default=0.4)
    ap.add_argument("--bird-weight", type=float, default=1.5)
    a = ap.parse_args()
    (OUT / "classifiers").mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    log("=" * 64); log("  v4: held-out bird.v1i split (train-birds -> retrain; eval on test-birds)"); log("=" * 64)

    imgs = sorted(p for p in BIRD_DIR.iterdir() if D.is_jpg(p))
    rng = np.random.RandomState(0); perm = rng.permutation(len(imgs))
    n_test = int(len(imgs) * a.test_frac)
    test_names = {imgs[i].name for i in perm[:n_test]}
    train_imgs = [p for p in imgs if p.name not in test_names]
    test_imgs = [p for p in imgs if p.name in test_names]
    log(f"[1/4] bird.v1i: {len(imgs)} imgs -> TRAIN {len(train_imgs)} / TEST {len(test_imgs)} (seed 0, no overlap)")

    log("[2/4] loading FT4 detector + mining train/test bird fires...")
    model = YOLO(D.MODEL_PATHS["ft4_r3"]); hook = D.DetectInputHook(); h = hook.register(model)
    try:
        Xtr = mine(model, hook, train_imgs, "TRAIN birds")
        Xte = mine(model, hook, test_imgs, "TEST birds (held out)")
    finally:
        h.remove()

    log(f"[3/4] v4 = v2 cache + {len(Xtr)} TRAIN-bird confusers (y=0, w={a.bird_weight}); retrain")
    z = np.load(V2_NPZ); X, y, w = z["X"].astype(np.float32), z["y"].astype(np.float32), z["w"].astype(np.float32)
    X = np.vstack([X, Xtr]); y = np.r_[y, np.zeros(len(Xtr), np.float32)]; w = np.r_[w, np.full(len(Xtr), a.bird_weight, np.float32)]
    pm = np.random.RandomState(D.SEED).permutation(len(X)); X, y, w = X[pm], y[pm], w[pm]
    log(f"      v4 corpus: {len(X)} ({int((y==1).sum())} drone / {int((y==0).sum())} confuser)")
    np.savez_compressed(OUT / "training_data.npz", X=X, y=y, w=w)
    (OUT / "split.json").write_text(json.dumps({"test_frac": a.test_frac, "n_train_imgs": len(train_imgs),
                                                 "n_test_imgs": len(test_imgs), "test_names": sorted(test_names)}, indent=2))
    v4_wt = OUT / "classifiers" / "mlp_v5_balanced_v4.pt"
    S.train_v5_mlp(X, y, w, v4_wt)

    log(f"\n[4/4] HELD-OUT bird.v1i TEST FP (fires kept at P>={THR}; lower=better; {len(Xte)} test fires):")
    for tag, wt in (("bare (no filter)", None), ("shipped mlp_v5", SHIPPED), ("v2 (no birds)", V2_WT), ("v4 (train-birds)", v4_wt)):
        if wt is None:
            fp = len(Xte)
        else:
            p = MLPv4Verifier(Path(wt), device="cpu").predict_drone_probs(Xte); fp = int((p >= THR).sum())
        log(f"   {tag:<20} {fp:>5} / {len(Xte)}  ({fp/max(len(Xte),1):.0%} kept)")
    log(f"\nDONE in {time.time()-t_all:.0f}s -> {v4_wt}")
    log("If v4 << shipped on held-out TEST birds, in-distribution bird training transfers (the win).")
    log(f"NON-bird surfaces (leak-free): py eval/filter_acceptance_eval.py --mode rgb --candidate {v4_wt}")


if __name__ == "__main__":
    main()
