#!/usr/bin/env python
"""gen_grayscale_panel.py - the cross-modal transfer figure (RQ3), on Anti-UAV RGB
(drones are large/central, so the boxes are actually legible). Three zoom-cropped panels
on one frame: (1) production RGB detector on the RGB frame, (2) IR detector v3b on the
grayscale-converted frame -- catches the drone, (3) IR detector v3b on the raw 3-channel
RGB -- collapses. Picks the frame with the strongest gray-catches / raw-collapses contrast.

  py docs/gen_grayscale_panel.py
Outputs: docs/figures/fig_grayscale_panel.{png,pdf}
"""
import os, glob, cv2, numpy as np
from ultralytics import YOLO

CLIP = "G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images"
LBL  = "G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/labels"
RGB_W = "models/rgb/Yolo26n_selcom_confuser_ft4_1280/weights/best.pt"
IR_W  = "models/ir/corrective_finetune/finetune_v3b/weights/best.pt"
OUT   = "docs/figures/fig_grayscale_panel"


def gt_box(stem, W, H):
    p = os.path.join(LBL, stem + ".txt")
    if not os.path.exists(p):
        return None
    for ln in open(p):
        a = ln.split()
        if len(a) >= 5:
            cx, cy, bw, bh = (float(x) for x in a[1:5])
            return [int((cx-bw/2)*W), int((cy-bh/2)*H), int((cx+bw/2)*W), int((cy+bh/2)*H)]
    return None


def dets(model, img, imgsz):
    r = model.predict(img, imgsz=imgsz, conf=0.25, verbose=False)[0]
    return [(b.xyxy[0].tolist(), float(b.conf[0])) for b in r.boxes]


def iou(a, b):
    if not a or not b:
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1-ix0), max(0, iy1-iy0); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0.0


def gray3(img):
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def crop_panel(im, boxes, gt, title, iou_txt, iou_color):
    """Zoom-crop around the drone; draw GT (green) + detector boxes (red) + readouts."""
    cx, cy = (gt[0]+gt[2])//2, (gt[1]+gt[3])//2
    dim = max(gt[2]-gt[0], gt[3]-gt[1])
    s = int(min(min(im.shape[:2]), max(300, dim * 6)))           # ~6x drone, clamped
    x0 = int(max(0, min(im.shape[1]-s, cx - s//2)))
    y0 = int(max(0, min(im.shape[0]-s, cy - s//2)))
    crop = cv2.resize(im[y0:y0+s, x0:x0+s].copy(), (460, 460)); sc = 460.0/s
    def R(b, col, th):
        cv2.rectangle(crop, (int((b[0]-x0)*sc), int((b[1]-y0)*sc)),
                      (int((b[2]-x0)*sc), int((b[3]-y0)*sc)), col, th)
    R(gt, (0, 220, 0), 2)                                        # GT (green)
    for (bx0, by0, bx1, by1), conf in boxes:                     # detections (red)
        R((bx0, by0, bx1, by1), (40, 40, 255), 2)
        cv2.putText(crop, f"{conf:.2f}", (int((bx0-x0)*sc), int((by1-y0)*sc)+16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 255), 1)
    cv2.rectangle(crop, (0, 0), (460, 26), (0, 0, 0), -1)
    cv2.putText(crop, title, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
    cv2.rectangle(crop, (0, 432), (460, 460), iou_color, -1)
    cv2.putText(crop, iou_txt, (6, 452), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return crop


def main():
    rgb, ir = YOLO(RGB_W), YOLO(IR_W)
    frames = sorted(glob.glob(os.path.join(CLIP, "*.jpg")))
    cands = frames[::max(1, len(frames)//220)]                  # ~220 candidates across sequences
    print(f"  scanning {len(cands)} candidate frames of {len(frames)}")
    best = None
    for fp in cands:
        img = cv2.imread(fp)
        if img is None:
            continue
        H, W = img.shape[:2]
        gt = gt_box(os.path.splitext(os.path.basename(fp))[0], W, H)
        if not gt:
            continue
        g_iou = max((iou(b, gt) for b, _ in dets(ir, gray3(img), 640)), default=0.0)
        if g_iou < 0.45:                                         # grayscale must clearly catch
            continue
        r_iou = max((iou(b, gt) for b, _ in dets(ir, img, 640)), default=0.0)
        if r_iou > 0.10:                                         # raw RGB must clearly collapse
            continue
        area = (gt[2]-gt[0]) * (gt[3]-gt[1]) / float(W*H)
        score = g_iou - r_iou + 2.0 * min(area, 0.04)           # strong contrast + a visible drone
        if best is None or score > best[0]:
            best = (score, fp, img, gt, g_iou, r_iou)
    if not best:
        print("  no frame with gray-catches/raw-collapses contrast found; aborting"); return
    _, fp, img, gt, gi, ri = best
    print(f"  picked {os.path.basename(fp)}: ir_gray_iou={gi:.2f} ir_raw_iou={ri:.2f}")
    gray = gray3(img)
    panels = [
        crop_panel(img,  dets(rgb, img, 1280), gt, "1. RGB detector (ft4) on RGB",
                   "drone present (reference)", (60, 60, 60)),
        crop_panel(gray, dets(ir, gray, 640),  gt, "2. IR detector on GRAYSCALE: catches",
                   f"IoU {gi:.2f} with GT  ->  KEEP", (30, 120, 30)),
        crop_panel(img,  dets(ir, img, 640),   gt, "3. IR detector on RAW RGB: collapses",
                   f"IoU {ri:.2f}  ->  no drone box", (30, 30, 170)),
    ]
    panel = np.hstack([panels[0], np.full((460, 6, 3), 255, np.uint8),
                       panels[1], np.full((460, 6, 3), 255, np.uint8), panels[2]])
    os.makedirs("docs/figures", exist_ok=True)
    cv2.imwrite(OUT + ".png", panel)
    from PIL import Image
    Image.open(OUT + ".png").save(OUT + ".pdf")
    print(f"  wrote {OUT}.{{png,pdf}}  frame={os.path.basename(fp)}")


if __name__ == "__main__":
    main()
