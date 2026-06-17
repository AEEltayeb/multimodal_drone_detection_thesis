"""eval_birdtest_heldout.py — HONEST held-out bird eval. Re-mines the bird.v1i TEST
split (the 484 imgs v4 NEVER trained on; from _v5_balanced_v4/split.json) and reports
FP (fires kept @0.25) for bare / shipped / v2 / v4. No leak. VERBOSE.
  py -u eval/eval_birdtest_heldout.py
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import cv2

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))
import distill_v5_p3p5_ft4 as D
from eval_v4_vs_patch import MLPv4Verifier         # noqa: E402
from ultralytics import YOLO                        # noqa: E402

BIRD_DIR = Path("G:/drone/bird.v1i.yolo26-birds-zekpr-bird-pn3pj/train/images")
SPLIT = REPO / "eval/results/_v5_balanced_v4/split.json"
WTS = [("bare (no filter)", None),
       ("shipped mlp_v5", REPO / "models/verifiers/rgb_v5/mlp_v5.pt"),
       ("v2 (no birds)", REPO / "eval/results/_v5_balanced_v2/classifiers/mlp_v5_balanced_v2.pt"),
       ("v4 (train-birds)", REPO / "eval/results/_v5_balanced_v4/classifiers/mlp_v5_balanced_v4.pt")]
THR = 0.25


def log(m): print(m, flush=True)


def main():
    test_names = set(json.load(open(SPLIT))["test_names"])
    test_imgs = [BIRD_DIR / n for n in sorted(test_names) if (BIRD_DIR / n).exists()]
    log(f"held-out bird.v1i TEST images: {len(test_imgs)} (never trained on)")

    model = YOLO(D.MODEL_PATHS["ft4_r3"]); hook = D.DetectInputHook(); h = hook.register(model)
    feats = []; t0 = time.time(); n = 0
    try:
        for i, p in enumerate(test_imgs):
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
            if n % 150 == 0:
                fps = n / (time.time() - t0)
                log(f"  [{i+1}/{len(test_imgs)}] {len(feats)} fires | {fps:.1f} img/s | ETA ~{(len(test_imgs)-i-1)/max(fps,0.1):.0f}s")
    finally:
        h.remove()
    Xte = np.array(feats, np.float32)
    log(f"test bird fires: {len(Xte)} (mined in {time.time()-t0:.0f}s)\n")

    log(f"HELD-OUT bird.v1i TEST FP (fires kept @{THR}; lower=better):")
    out = {"dataset": "bird.v1i", "split": "TEST (484 imgs, never trained on)",
           "n_test_imgs": len(test_imgs), "n_fires": int(len(Xte)), "thr": THR, "results": {}}
    for tag, wt in WTS:
        if wt is None:
            fp = len(Xte)
        else:
            fp = int((MLPv4Verifier(Path(wt), device="cpu").predict_drone_probs(Xte) >= THR).sum())
        out["results"][tag] = {"FP": fp, "kept_frac": round(fp / max(len(Xte), 1), 4)}
        log(f"  {tag:<20} {fp:>5} / {len(Xte)}  ({fp/max(len(Xte),1):.0%} kept)")
    op = REPO / "eval/results/birdtest_heldout_results.json"
    json.dump(out, open(op, "w"), indent=2); log(f"\nWROTE {op}")


if __name__ == "__main__":
    main()
