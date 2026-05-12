"""Quick drone-only P/R at 1280 on Svanstrom."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from metrics import compute_prf, score_detections
from datasets import ImageDataset

MODELS = {
    "baseline": str(Path(__file__).parent.parent / "RGB model/Yolo26n_trained/weights/best.pt"),
    "hardneg_v3more": str(Path(__file__).parent.parent / "RGB model/Yolo26n_hardneg_v3_more/weights/best.pt"),
    "retrained_v2": str(Path(__file__).parent.parent / "RGB model/Yolo26n_retrained_v2/weights/best.pt"),
}

from ultralytics import YOLO

ds = ImageDataset(Path("G:/drone/svanstrom_paired/RGB/images"))
images = ds.list_images()[::7]
print(f"{len(images)} images")

for mname, wpath in MODELS.items():
    model = YOLO(wpath)
    fp_by_cat = {}
    tp_total = fp_total = fn_total = 0
    t0 = time.time()
    for idx, img_path in enumerate(images):
        frame = ds.load_frame(img_path)
        if frame is None:
            continue
        img, gt = frame["img"], frame["gt"]
        cat = frame.get("category", "OTHER")
        res = model.predict(img, conf=0.25, verbose=False, imgsz=1280)
        boxes = res[0].boxes
        dets = []
        for i in range(len(boxes)):
            xyxy = boxes.xyxy[i].cpu().numpy()
            dets.append(((float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])), float(boxes.conf[i])))
        tp, fp, fn = score_detections(dets, gt, rule="iop", iou_thr=0.5, iop_thr=0.5)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        if fp > 0:
            fp_by_cat[cat] = fp_by_cat.get(cat, 0) + fp
        if (idx + 1) % 1000 == 0:
            print(f"  {mname} {idx+1}/{len(images)}")

    elapsed = time.time() - t0
    m = compute_prf(tp_total, fp_total, fn_total)
    drone_fp = fp_by_cat.get("DRONE", 0)
    dm = compute_prf(tp_total, drone_fp, fn_total)

    print(f"\n{mname} (IoP, 1280):")
    print(f"  Overall:    TP={tp_total} FP={fp_total} FN={fn_total}  P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f}")
    print(f"  FP by cat:  {fp_by_cat}")
    print(f"  DRONE-only: TP={tp_total} FP={drone_fp} FN={fn_total}  P={dm['precision']:.4f} R={dm['recall']:.4f} F1={dm['f1']:.4f}")
    print(f"  ({elapsed:.0f}s)")
