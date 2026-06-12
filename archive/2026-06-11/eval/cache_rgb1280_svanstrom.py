"""
Generate Svanstrom caches with RGB @ imgsz=1280, IR @ imgsz=640.
One cache per RGB model. Then run eval_pipeline on each.

28710 images / stride 10 = 2871 (~3k).
"""
import json, sys, time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent
sys.path.insert(0, str(EVAL_DIR))

from datasets import load_config, resolve_path, img_from_label
import cv2

RGB_MODELS = {
    "baseline":       str(REPO / "RGB model" / "Yolo26n_trained" / "weights" / "best.pt"),
    "hardneg_v3more": str(REPO / "RGB model" / "Yolo26n_hardneg_v3_more" / "weights" / "best.pt"),
    "retrained_v2":   str(REPO / "RGB model" / "Yolo26n_retrained_v2" / "weights" / "best.pt"),
}

IR_WEIGHTS = str(REPO / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt")
STRIDE = 10
CONF = 0.001  # Cache at very low conf; eval_pipeline applies threshold later


def generate_cache(rgb_name, rgb_weights):
    from ultralytics import YOLO

    cfg = load_config()
    ds_cfg = cfg["datasets"]["svanstrom"]
    root = Path(ds_cfg["root"])
    rgb_img_dir = root / ds_cfg["rgb_images"]
    rgb_lbl_dir = root / ds_cfg["rgb_labels"]
    ir_img_dir  = root / ds_cfg["ir_images"]
    ir_lbl_dir  = root / ds_cfg["ir_labels"]
    rgb_suffix  = ds_cfg.get("rgb_stem_suffix", "")
    ir_suffix   = ds_cfg.get("ir_stem_suffix", "")

    rgb_model = YOLO(rgb_weights)
    ir_model  = YOLO(IR_WEIGHTS)

    stems = sorted(p.stem for p in rgb_lbl_dir.glob("*.txt"))
    stems = stems[::STRIDE]
    print(f"\n[{rgb_name}] {len(stems)} frames, RGB@1280 IR@640")

    cache_dir = EVAL_DIR / "cache"
    cache_dir.mkdir(exist_ok=True)
    out_path = cache_dir / f"raw_detections_svanstrom_rgb1280_{rgb_name}.json"

    data = {}
    t0 = time.time()
    for idx, stem in enumerate(stems):
        ir_stem = stem.replace(rgb_suffix, ir_suffix) if rgb_suffix and ir_suffix else stem
        rgb_lbl = rgb_lbl_dir / f"{stem}.txt"
        ir_lbl  = ir_lbl_dir / f"{ir_stem}.txt"
        rgb_path = img_from_label(rgb_lbl, rgb_img_dir)
        ir_path  = img_from_label(ir_lbl, ir_img_dir)
        if rgb_path is None or ir_path is None:
            continue

        rgb_img = cv2.imread(str(rgb_path))
        ir_img  = cv2.imread(str(ir_path))
        if rgb_img is None or ir_img is None:
            continue

        rh, rw = rgb_img.shape[:2]
        ih, iw = ir_img.shape[:2]

        # RGB at 1280, IR at 640
        rgb_res = rgb_model.predict(rgb_img, conf=CONF, verbose=False, imgsz=1280)
        ir_res  = ir_model.predict(ir_img,  conf=CONF, verbose=False, imgsz=640)

        def _extract(res):
            boxes = res[0].boxes
            dets = []
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                c = float(boxes.conf[i])
                dets.append([round(float(xyxy[0]),1), round(float(xyxy[1]),1),
                             round(float(xyxy[2]),1), round(float(xyxy[3]),1),
                             round(c, 4)])
            return dets

        data[stem] = {
            "rgb_dets": _extract(rgb_res),
            "ir_dets":  _extract(ir_res),
            "rgb_w": rw, "rgb_h": rh,
            "ir_w": iw, "ir_h": ih,
            "rgb_lbl": str(rgb_lbl),
            "ir_lbl":  str(ir_lbl),
        }

        if (idx + 1) % 500 == 0:
            out_path.write_text(json.dumps(data))
            elapsed = time.time() - t0
            fps = (idx + 1) / elapsed
            eta = (len(stems) - idx - 1) / fps
            print(f"  {idx+1}/{len(stems)}  {fps:.1f} fps  ETA {eta:.0f}s")

    out_path.write_text(json.dumps(data))
    elapsed = time.time() - t0
    print(f"  [{rgb_name}] Done: {len(data)} frames in {elapsed:.0f}s -> {out_path.name}")
    return out_path


def run_pipeline(cache_path, rgb_name):
    """Run eval_pipeline using the generated cache."""
    import subprocess
    out_dir = EVAL_DIR / "results" / "_rgb1280_pipeline" / rgb_name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(EVAL_DIR / "eval_pipeline.py"),
        "--dataset", "svanstrom",
        "--stride", "1",  # cache already at stride 10
        "--imgsz", "640",  # doesn't matter, using cache
        "--rgb-conf", "0.25",
        "--ir-conf", "0.4",
        "--patch-thr", "0.7",
        "--scoring", "trust_aware",
        "--cache-tag", f"rgb1280_{rgb_name}",
        "--output-dir", str(out_dir),
    ]
    print(f"\n[{rgb_name}] Running pipeline eval...")
    print(f"  cmd: {' '.join(cmd[-8:])}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


if __name__ == "__main__":
    # Step 1: Generate caches
    cache_paths = {}
    for name, weights in RGB_MODELS.items():
        cache_paths[name] = generate_cache(name, weights)

    # Step 2: Run pipeline eval on each
    for name in RGB_MODELS:
        run_pipeline(cache_paths[name], name)

    print("\n[ALL DONE]")
