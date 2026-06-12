"""Quick probe: run production RGB + IR models on G:/drone/drone assets and
print conf/bbox for each pair. No GT - just whether a detection fires."""
from pathlib import Path
import cv2
from ultralytics import YOLO

REPO = Path(__file__).resolve().parent.parent
ROOT = Path(r"G:/drone/drone assets")
RGB_W = REPO / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt"
IR_W  = REPO / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt"
RGB_CONF, IR_CONF = 0.25, 0.40

rgb = YOLO(str(RGB_W)); ir = YOLO(str(IR_W))
print(f"{'asset':<24} {'side':<3} {'shape':<14} {'n_det':<5} {'top_conf':<8} bbox")
for n in range(1, 6):
    for side, mod, conf, sz in [("rgb", rgb, RGB_CONF, 1280), ("ir", ir, IR_CONF, 640)]:
        p = ROOT / f"drone {n} {side}.png"
        img = cv2.imread(str(p))
        if img is None: print(f"{p.name:<24} MISSING"); continue
        res = mod.predict(img, imgsz=sz, conf=conf, verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) else []
        confs = res.boxes.conf.cpu().numpy().tolist() if len(res.boxes) else []
        top = f"{max(confs):.3f}" if confs else "MISS"
        bb = boxes[confs.index(max(confs))].astype(int).tolist() if confs else "-"
        print(f"{p.name:<24} {side:<3} {str(img.shape):<14} {len(boxes):<5} {top:<8} {bb}")
