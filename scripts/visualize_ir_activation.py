"""visualize_ir_activation.py — IR analog of the RGB v5_activation_* panels.

Runs the production IR detector (v3b) on an IR DRONE example and an IR CONFUSER example,
and renders, for each: the detection crop + the P3 (high-res spatial) and P5 (semantic)
activation heatmaps over the detection region — "the IR model's brain overlaid on the box."

  py scripts/visualize_ir_activation.py
Outputs: docs/analysis/images/v5_ir_activation_drone_example.png
         docs/analysis/images/v5_ir_activation_confuser_example.png
"""
from __future__ import annotations
from pathlib import Path
import numpy as np, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ultralytics import YOLO
import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "eval"))
from distill_v5_p3p5_ft4 import DetectInputHook  # noqa

V3B = str(REPO / "models/ir/corrective_finetune/finetune_v3b/weights/best.pt")
IR_DRONE_DIR = Path("G:/drone/IR_dset_final/test/images")
CBAM_DIR = Path("G:/drone/Infrared_bird_drone_airplane_CBAM_TF-Net.v1i.yolo26-maha-daxhh-cbam_tf-net/valid/images")
OUT = REPO / "docs/analysis/images"; OUT.mkdir(parents=True, exist_ok=True)
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def gt_boxes(img_path, w, h, cls):
    a = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
    if not a.exists():
        return []
    out = []
    for ln in a.read_text().strip().splitlines():
        p = ln.split()
        if len(p) >= 5 and int(p[0]) == cls:
            cx, cy, bw, bh = map(float, p[1:5])
            out.append(((cx-bw/2)*w, (cy-bh/2)*h, (cx+bw/2)*w, (cy+bh/2)*h))
    return out


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1]); x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    i = max(0., x2-x1)*max(0., y2-y1); ua = (a[2]-a[0])*(a[3]-a[1]); ub = (b[2]-b[0])*(b[3]-b[1])
    return i/(ua+ub-i) if ua+ub-i > 0 else 0.


def find_example(yolo, hook, d, want_drone, drone_cls, max_scan=400):
    imgs = sorted(p for p in d.iterdir() if p.suffix.lower() in EXTS)[:max_scan] if d.exists() else []
    best = None
    for ip in imgs:
        img = cv2.imread(str(ip))
        if img is None:
            continue
        h, w = img.shape[:2]
        hook.clear()
        r = yolo.predict(img, imgsz=640, conf=0.25, verbose=False, device="cuda")[0]
        if r.boxes is None or len(r.boxes) == 0:
            continue
        gts = gt_boxes(ip, w, h, drone_cls)
        for i in range(len(r.boxes)):
            box = tuple(r.boxes.xyxy[i].cpu().numpy().tolist()); conf = float(r.boxes.conf[i])
            matched = any(iou(box, g) >= 0.5 for g in gts)
            ok = matched if want_drone else (not matched)   # confuser = fires but not on a drone
            if ok and (best is None or conf > best["conf"]):
                best = {"path": ip, "box": box, "conf": conf, "img": img.copy()}
    return best


def heat(fmap, box, img_shape, crop_hw):
    """box-region mean activation of fmap (1,C,H,W) -> upsampled to crop size."""
    _, C, H, W = fmap.shape
    ih, iw = img_shape
    x1, y1, x2, y2 = box
    fx1, fy1 = max(0, int(x1/iw*W)), max(0, int(y1/ih*H))
    fx2, fy2 = min(W, max(fx1+1, int(np.ceil(x2/iw*W)))), min(H, max(fy1+1, int(np.ceil(y2/ih*H))))
    m = fmap[0, :, fy1:fy2, fx1:fx2].mean(0).cpu().numpy()
    m = (m - m.min()) / (np.ptp(m) + 1e-6)
    return cv2.resize(m, (crop_hw[1], crop_hw[0]), interpolation=cv2.INTER_CUBIC)


def panel(yolo, hook, ex, title, out):
    img = ex["img"]; h, w = img.shape[:2]; x1, y1, x2, y2 = map(int, ex["box"])
    pad = int(0.3 * max(x2-x1, y2-y1))
    cx1, cy1, cx2, cy2 = max(0, x1-pad), max(0, y1-pad), min(w, x2+pad), min(h, y2+pad)
    hook.clear(); yolo.predict(img, imgsz=640, conf=0.25, verbose=False, device="cuda")
    crop = img[cy1:cy2, cx1:cx2]; ch, cw = crop.shape[:2]
    p3 = heat(hook.p3, ex["box"], (h, w), (ch, cw))
    p5 = heat(hook.p5, ex["box"], (h, w), (ch, cw))
    fig, ax = plt.subplots(1, 3, figsize=(11, 4))
    ax[0].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)); ax[0].set_title(f"IR crop (conf={ex['conf']:.2f})")
    ax[1].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cmap="gray"); ax[1].imshow(p3, cmap="jet", alpha=0.55); ax[1].set_title("P3 activation (spatial, stride 8)")
    ax[2].imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cmap="gray"); ax[2].imshow(p5, cmap="jet", alpha=0.55); ax[2].set_title("P5 activation (semantic, stride 32)")
    for a in ax: a.axis("off")
    fig.suptitle(title); plt.tight_layout(); plt.savefig(out, dpi=170, bbox_inches="tight"); plt.close()
    print(f"  wrote {out.name}  (from {ex['path'].name})")


def main():
    yolo = YOLO(V3B); hook = DetectInputHook(); hook.register(yolo)
    print("scanning IR drone example (IR_dset_final)...")
    drone = find_example(yolo, hook, IR_DRONE_DIR, want_drone=True, drone_cls=0)
    print("scanning IR confuser example (CBAM, drone=class1)...")
    conf = find_example(yolo, hook, CBAM_DIR, want_drone=False, drone_cls=1)
    if drone:
        panel(yolo, hook, drone, "IR detector (v3b) — DRONE: P3/P5 activation overlay", OUT / "v5_ir_activation_drone_example.png")
    else:
        print("  no IR drone example found")
    if conf:
        panel(yolo, hook, conf, "IR detector (v3b) — CONFUSER (false fire): P3/P5 activation overlay", OUT / "v5_ir_activation_confuser_example.png")
    else:
        print("  no IR confuser example found")


if __name__ == "__main__":
    main()
