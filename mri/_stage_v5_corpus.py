"""
Stage an existing V5 training corpus (eval/results/<tag>/training_data.npz) into
an MRI run dir so `python -m mri --resume` regenerates the report figures + stats
+ verdict from the *exact same features* — apples-to-apples, no re-extraction
sampling noise.

Usage:
    python mri/_stage_v5_corpus.py [corpus_tag] [out_name]
        corpus_tag default = _v5_selcom_pure_1x8  (the SHIPPED production model)
        out_name   default = v5_report_regen

The 517-D V5 schema is: 5 meta + p3@2x2 (4*64=256) + p5@1x1 (256). Per-source
'raws' are reconstructed from training_meta.json so diagnosis has a class signal.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
TAG = sys.argv[1] if len(sys.argv) > 1 else "_v5_selcom_pure_1x8"
OUT_NAME = sys.argv[2] if len(sys.argv) > 2 else "v5_report_regen"

SRC = REPO / "eval" / "results" / TAG
OUT = REPO / "mri" / "results" / OUT_NAME
OUT.mkdir(parents=True, exist_ok=True)

z = np.load(SRC / "training_data.npz")
X = z["X"].astype(np.float32)
y = z["y"].astype(np.float32)
w = z["w"].astype(np.float32) if "w" in z.files else np.ones(len(y), np.float32)
meta = json.loads((SRC / "training_meta.json").read_text())

print(f"corpus {TAG}: X{X.shape}  drones={int((y==1).sum())}  confusers={int((y==0).sum())}")
assert X.shape[1] == 517, f"expected 517-D, got {X.shape[1]}"

schema = {
    "layers": ["p3", "p5"],
    "grids": {"p3": [2, 2], "p5": [1, 1]},
    "layer_dims": {"p3": 256, "p5": 256},
    "meta_dim": 5,
    "total_dim": 517,
    "metadata_order": ["conf", "log_area", "aspect", "rel_cx", "rel_cy"],
}

raws = []
for s in meta.get("per_source_counts", []):
    role = "neg" if (s.get("target_drones", 0) == 0) else "pos"
    raws.append({
        "name": s["name"], "role": role,
        "n_images": 0,  # unknown at corpus level
        "n_dets": s.get("n_drones", 0) + s.get("n_confusers", 0),
        "tp": s.get("n_drones", 0), "fp": s.get("n_confusers", 0), "fn": 0,
        "mined_drones": s.get("n_drones", 0),
        "mined_confusers": s.get("n_confusers", 0),
    })

np.savez_compressed(OUT / "features.npz", X=X, y=y, w=w)
(OUT / "features_meta.json").write_text(json.dumps({
    "schema": schema, "raws": raws,
    "n_total": int(len(X)), "n_drone": int((y == 1).sum()),
    "n_confuser": int((y == 0).sum()),
    "note": f"Staged from eval/results/{TAG} for MRI --resume regeneration.",
}, indent=2))
print(f"staged {TAG} -> {OUT}")
