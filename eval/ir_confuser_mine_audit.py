"""ir_confuser_mine_audit.py — EXACT category x size of MINEABLE thermal confusers.
Small GPU pass (~10-15 min): runs v3b on the thermal confuser sources BY PREFIX and
records the size (detection short-side px) of every fire, so the --thermal-confusers
balancing can be capped per (category x size) cell from real numbers, not assumption.

Confuser objects aren't class-0-labelled, so the only way to know what the detector
actually fires on (and at what size) is to run it. This is that audit.

Sources (the IR filter's thermal confuser pool):
  IR_confusers train (NEW)  airplane_/bird_/helicopter_   imgsz 640
  svanstrom IR              IR_AIRPLANE_/IR_BIRD_/IR_HELICOPTER_  imgsz 1280
  IR_video train           same prefixes                 imgsz 640

  py eval/ir_confuser_mine_audit.py            # train split (default)
  py eval/ir_confuser_mine_audit.py --split val
"""
from __future__ import annotations
import argparse, json, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from ultralytics import YOLO   # noqa: E402

V3B = REPO / "models" / "ir" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt"
if not V3B.exists():
    V3B = REPO / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt"
EDGES = [0, 16, 32, 64, 1e9]; NAMES = ["<16px", "16-32", "32-64", ">=64"]
CATS = {"airplane": ("airplane_", "IR_AIRPLANE_"), "bird": ("bird_", "IR_BIRD_"),
        "helicopter": ("helicopter_", "IR_HELICOPTER_")}
IMG = REPO / "docs/analysis/images"; IMG.mkdir(parents=True, exist_ok=True)


def cat_of(name):
    n = name.lower()
    if "airplan" in n: return "airplane"
    if "bird" in n: return "bird"
    if "heli" in n: return "helicopter"
    return None


def sbin(box):
    s = min(box[2] - box[0], box[3] - box[1])
    for i in range(4):
        if EDGES[i] <= s < EDGES[i + 1]:
            return i
    return 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()
    sources = [
        (f"G:/drone/IR_confusers/images/{a.split}", 640),
        ("G:/drone/svanstrom_paired/IR/images", 1280),
        ("G:/drone/IR_video_ir_dataset/train/images", 640),
    ]
    print(f"detector: {V3B}\nsplit: {a.split}  conf {a.conf}  stride {a.stride}")
    yolo = YOLO(str(V3B))
    grid = defaultdict(int)          # (cat, sizebin) -> fire count
    per_src = defaultdict(lambda: defaultdict(int))
    for d, imgsz in sources:
        dd = Path(d)
        if not dd.exists():
            print(f"  [skip missing {d}]"); continue
        imgs = [p for p in sorted(dd.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp") and cat_of(p.name)][::a.stride]
        print(f"  {d}: {len(imgs)} prefixed imgs")
        t0 = time.time()
        for p in imgs:
            c = cat_of(p.name)
            im = cv2.imread(str(p))
            if im is None:
                continue
            r = yolo.predict(im, imgsz=imgsz, conf=a.conf, verbose=False, device="cuda")[0]
            if r.boxes is None:
                continue
            for i in range(len(r.boxes)):
                box = tuple(r.boxes.xyxy[i].cpu().numpy().tolist())
                grid[(c, sbin(box))] += 1; per_src[Path(d).parts[-3] if "IR_confusers" in d else Path(d).parts[1]][(c, sbin(box))] += 1
        print(f"    ({time.time()-t0:.0f}s)")

    cats = ["airplane", "bird", "helicopter"]
    print("\n=== MINEABLE thermal confuser FIRES: category x size ===")
    print(f"{'category':<12} " + " ".join(f"{n:>8}" for n in NAMES) + f" {'total':>8}")
    rep = {}
    for c in cats:
        row = [grid[(c, i)] for i in range(4)]
        rep[c] = row
        print(f"{c:<12} " + " ".join(f"{v:>8}" for v in row) + f" {sum(row):>8}")
    tot = [sum(grid[(c, i)] for c in cats) for i in range(4)]
    print(f"{'TOTAL':<12} " + " ".join(f"{v:>8}" for v in tot) + f" {sum(tot):>8}")

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(4); w = 0.25
    for k, c in enumerate(cats):
        ax.bar(x + (k-1)*w, [grid[(c, i)] for i in range(4)], w, label=c)
    ax.set_xticks(x); ax.set_xticklabels(NAMES); ax.set_ylabel("mineable fires")
    ax.set_title(f"Mineable thermal confuser fires by category x size ({a.split})\nv3b @conf {a.conf}")
    ax.legend(); fig.tight_layout()
    png = IMG / f"2026-06-17_ir_confuser_mineable_{a.split}.png"
    fig.savefig(png, dpi=130); plt.close(fig)
    out = REPO / "docs/analysis" / f"2026-06-17_ir_confuser_mineable_{a.split}.json"
    out.write_text(json.dumps({"buckets": NAMES, "grid": rep, "total_by_size": tot}, indent=2))
    print(f"\nsaved {png}\nsaved {out}")


if __name__ == "__main__":
    main()
