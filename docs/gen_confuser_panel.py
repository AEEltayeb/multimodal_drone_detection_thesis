#!/usr/bin/env python
"""gen_confuser_panel.py - the 'why the cascade exists' figure. The production RGB
detector fires on BOTH a bird and a drone; the feature-reuse verifier reads the
detector's own ROI features and separates them - vetoing the bird, keeping the drone.
Left: a bird the detector fired on, verifier P(drone) low (VETO). Right: a drone,
P(drone) high (KEEP).

  py docs/gen_confuser_panel.py
Outputs: docs/figures/fig_confuser_panel.{png,pdf}
"""
import os, glob, sys, cv2, numpy as np
sys.path.insert(0, "classifier")
from ultralytics import YOLO
from mlp_verifier import DetectInputHook, MLPVerifier

RGB_W = "models/rgb/Yolo26n_selcom_confuser_ft4_1280/weights/best.pt"
MLP = "models/verifiers/rgb_v5/mlp_v5.pt"
BIRD = "datasets/drone detection video tests/rgb/birds"
DRONE = "datasets/drone detection video tests/rgb/drone/flock_of_seagulls_attack_drone_beach/images/test"
DLBL = DRONE.replace("/images/", "/labels/")
OUT = "docs/figures/fig_confuser_panel"


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def gt_box(fp, W, H):
    p = os.path.join(DLBL, os.path.splitext(os.path.basename(fp))[0] + ".txt")
    if not os.path.exists(p):
        return None
    for ln in open(p):
        a = ln.split()
        if len(a) >= 5:
            cx, cy, bw, bh = (float(x) for x in a[1:5])
            return [(cx-bw/2)*W, (cy-bh/2)*H, (cx+bw/2)*W, (cy+bh/2)*H]
    return None


def scan(model, hook, ver, frames, want_high, need_gt, n=60):
    best = None
    for fp in frames[:n]:
        img = cv2.imread(fp)
        if img is None:
            continue
        res = model.predict(img, imgsz=1280, conf=0.25, verbose=False)[0]
        dets = [[*b.xyxy[0].tolist(), float(b.conf[0])] for b in res.boxes]
        if not dets:
            continue
        pd = ver.score_dets(hook, dets, img.shape[:2])
        for d, p in zip(dets, pd):
            box, conf = d[:4], d[4]
            if need_gt:
                gt = gt_box(fp, img.shape[1], img.shape[0])
                if not gt or iou(box, gt) < 0.2:
                    continue
            key = float(p) if want_high else -float(p)
            if best is None or key > best[0]:
                best = (key, img.copy(), box, conf, float(p))
    return best


def crop_cell(img, box, pdrone, conf, verdict, vcolor):
    cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
    s = int(max(300, max(box[2]-box[0], box[3]-box[1]) * 6)); s = min(s, min(img.shape[:2]))
    x0 = int(max(0, min(img.shape[1]-s, cx-s/2))); y0 = int(max(0, min(img.shape[0]-s, cy-s/2)))
    c = cv2.resize(img[y0:y0+s, x0:x0+s], (460, 460))
    sc = 460/s
    bx0, by0 = (box[0]-x0)*sc, (box[1]-y0)*sc; bx1, by1 = (box[2]-x0)*sc, (box[3]-y0)*sc
    cv2.rectangle(c, (int(bx0), int(by0)), (int(bx1), int(by1)), (40, 40, 255), 2)
    cv2.rectangle(c, (0, 0), (460, 28), (0, 0, 0), -1)
    cv2.putText(c, f"detector fired (conf {conf:.2f})", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.rectangle(c, (0, 432), (460, 460), vcolor, -1)
    cv2.putText(c, f"verifier P(drone)={pdrone:.2f}  ->  {verdict}", (6, 452), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return c


def main():
    model = YOLO(RGB_W); hook = DetectInputHook(); hook.register(model)
    ver = MLPVerifier(MLP, device="cuda")
    birds = sorted(glob.glob(os.path.join(BIRD, "**", "*.jpg"), recursive=True))
    drones = sorted(glob.glob(os.path.join(DRONE, "*.jpg")))
    print(f"  bird frames {len(birds)}, drone frames {len(drones)}")
    b = scan(model, hook, ver, birds, want_high=False, need_gt=False)
    d = scan(model, hook, ver, drones, want_high=True, need_gt=True)
    if not b or not d:
        print("  could not find both a bird-FP and a drone-TP detection"); return
    bird_cell = crop_cell(b[1], b[2], b[4], b[3], "VETO", (30, 30, 170))
    drone_cell = crop_cell(d[1], d[2], d[4], d[3], "KEEP", (30, 140, 30))
    panel = np.hstack([bird_cell, np.full((460, 6, 3), 255, np.uint8), drone_cell])
    os.makedirs("docs/figures", exist_ok=True)
    cv2.imwrite(OUT + ".png", panel)
    from PIL import Image
    Image.open(OUT + ".png").save(OUT + ".pdf")
    print(f"  wrote {OUT}  bird P(drone)={b[4]:.2f}  drone P(drone)={d[4]:.2f}")


if __name__ == "__main__":
    main()
