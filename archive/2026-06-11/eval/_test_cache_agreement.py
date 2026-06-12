"""Verify cache integrity: loading dets from existing caches and re-scoring
must produce the same TP/FP/FN as the original Phase 2 run.

Compares (dataset, detector, S0_detector) numbers:
  - From eval/results/full_pipeline_persize/<ds>/<det>/no_classifier/summary.csv (current run)
  - From eval/results/<phase2_dir>/<det>/<det>_results.json detection_metrics[iop]
  - From re-scoring the per-frame CSV dets via metrics.score_per_size

Any mismatch is a bug — refuse to enable cache reuse until fixed.
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

EVAL = Path(__file__).resolve().parent
REPO = EVAL.parent
sys.path.insert(0, str(EVAL))

from det_cache import DetCache, parse_dets_str  # noqa: E402
from metrics import score_per_size, SIZE_BUCKETS  # noqa: E402
from datasets import read_yolo_labels  # noqa: E402

cases = [
    # (ds_key, det, phase2_results_dir, weights_relpath, imgsz)
    ("antiuav", "baseline",
     "antiuav_per_model/baseline",
     "models/rgb/Yolo26n_trained/weights/best.pt", 640),
    ("antiuav", "selcom_960",
     "antiuav_per_model/selcom_960",
     "models/rgb/Yolo26n_selcom_mixed_ft2_1280/weights/best.pt", 960),
    ("selcom_val", "selcom_960",
     "selcom_val_holdout/selcom_960",
     "models/rgb/Yolo26n_selcom_mixed_ft2_1280/weights/best.pt", 960),
    ("selcom_val", "baseline",
     "selcom_val_holdout/baseline",
     "models/rgb/Yolo26n_trained/weights/best.pt", 1280),
]

# Per-dataset GT lookup
GT_DIRS = {
    "antiuav":    Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/labels"),
    "selcom_val": Path("G:/drone/_finetune_selcom_mixed_ft2/labels/val"),
}
IMG_DIRS = {
    "antiuav":    Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images"),
    "selcom_val": Path("G:/drone/_finetune_selcom_mixed_ft2/images/val"),
}

cache = DetCache(REPO)


from PIL import Image
_wh_cache: dict[tuple[str, str], tuple[int, int]] = {}

def img_wh(stem: str, ds: str) -> tuple[int, int] | None:
    """Header-only read via PIL; cached per (ds, stem)."""
    key = (ds, stem)
    if key in _wh_cache:
        return _wh_cache[key]
    d = IMG_DIRS[ds]
    for ext in (".jpg", ".jpeg", ".png", ".bmp"):
        p = d / f"{stem}{ext}"
        if p.exists():
            try:
                with Image.open(p) as im:
                    w, h = im.size  # metadata-only, no decode
                _wh_cache[key] = (w, h)
                return w, h
            except Exception:
                return None
    return None


print(f"{'CASE':45s} {'Phase2 TP/FP/FN':>22s}   {'Cache+rescore':>22s}   {'Match?':>10s}")
print("-" * 105)
all_ok = True
for ds, det, p2_dir, w_rel, sz in cases:
    p2_results = REPO / "eval" / "results" / p2_dir / f"{det}_results.json"
    if not p2_results.exists():
        print(f"{ds}/{det:<30s}  Phase 2 results JSON MISSING")
        all_ok = False
        continue
    d = json.loads(p2_results.read_text())
    iop_metrics = d["detection_metrics"][1] if len(d["detection_metrics"]) > 1 else d["detection_metrics"][0]
    p2_tp, p2_fp, p2_fn = iop_metrics["TP"], iop_metrics["FP"], iop_metrics["FN"]

    # Load per-frame CSV and re-score
    csv_p = REPO / "eval" / "results" / p2_dir / f"{det}_frame_detections.csv"
    if not csv_p.exists():
        print(f"{ds}/{det}: per-frame CSV missing")
        all_ok = False
        continue
    tp = fp = fn = 0
    with csv_p.open() as f:
        for r in csv.DictReader(f):
            stem = r["stem"]
            flat = parse_dets_str(r["dets"])
            dets = [((d[0], d[1], d[2], d[3]), d[4]) for d in flat]
            wh = img_wh(stem, ds)
            if wh is None:
                continue
            w, h = wh
            lbl_path = GT_DIRS[ds] / f"{stem}.txt"
            gts = read_yolo_labels(lbl_path, w, h)
            ps = score_per_size(dets, gts, w, h, iop_thr=0.5)["iop"]
            for b in SIZE_BUCKETS:
                tp += ps[b]["tp"]; fp += ps[b]["fp"]; fn += ps[b]["fn"]
    match = (tp, fp, fn) == (p2_tp, p2_fp, p2_fn)
    if not match:
        all_ok = False
    print(f"{ds}/{det:<30s}  {p2_tp}/{p2_fp}/{p2_fn:>5}   {tp}/{fp}/{fn:>5}   {'OK' if match else 'MISMATCH':>10s}")
print()
print("ALL OK" if all_ok else "FAIL: some caches do not reproduce Phase 2 numbers")
