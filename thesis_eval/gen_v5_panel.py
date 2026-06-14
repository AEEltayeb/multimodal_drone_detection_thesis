"""gen_v5_panel.py — assemble fig_v5_regression: frames whose drone was UNLABELLED in the V5-era
corpus (G:/drone/IR_dsetV5) but labelled in the corrected corpus (G:/drone/IR_dset_final), with the
V5 detector firing on it — i.e. the detection that was scored FP purely because the dataset state,
not the model state, was wrong.

Candidates: stems present in BOTH corpora where the V5-era label is empty/absent and the final label
has >=1 box. V5 (models/ir/IR_dsetV5_269ep/best.pt) is run on each candidate; kept if a V5 box overlaps
the corrected GT (IoU >= 0.3). Renders a 3-row x 2-col panel into the Overleaf figures dir.

  py -u thesis_eval/gen_v5_panel.py
"""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NEW = Path(__file__).resolve().parent.parent
OLD = NEW   # legacy alias: artifacts are now resident in ES_Drone_Thesis (was ES_Drone_Detection)
V5_W = OLD / "models/ir/IR_dsetV5_269ep/best.pt"          # read-only from the archive tree
ERA = Path(r"G:/drone/IR_dsetV5")
FINAL = Path(r"G:/drone/IR_dset_final")
FIGDIR = NEW / "docs/thesis_working_distilling_overleaf/figures"
EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def label_index(root, splits):
    idx = {}
    for s in splits:
        d = root / s / "labels"
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.suffix == ".txt":
                idx.setdefault(p.stem, p)
    return idx


def image_index(root, splits):
    idx = {}
    for s in splits:
        d = root / s / "images"
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.suffix.lower() in EXTS:
                idx.setdefault(p.stem, p)
    return idx


def boxes(lbl_path, w, h):
    out = []
    if lbl_path is None or not lbl_path.exists():
        return out
    for line in lbl_path.read_text().splitlines():
        f = line.split()
        if len(f) >= 5:
            cx, cy, bw, bh = (float(x) for x in f[1:5])
            out.append((int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                        int((cx + bw / 2) * w), int((cy + bh / 2) * h)))
    return out


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1)


def main():
    era_lbl = label_index(ERA, ("train", "val", "test"))
    era_img = image_index(ERA, ("train", "val", "test"))
    fin_lbl = label_index(FINAL, ("train", "val", "test"))
    fin_img = image_index(FINAL, ("train", "val", "test"))
    print(f"era: {len(era_lbl)} labels / {len(era_img)} imgs ; final: {len(fin_lbl)} labels / {len(fin_img)} imgs")

    cands = []
    for stem, fl in fin_lbl.items():
        if stem not in era_img:
            continue
        el = era_lbl.get(stem)
        era_empty = (el is None) or (el.read_text().strip() == "")
        if not era_empty:
            continue
        if fl.read_text().strip() == "":
            continue
        cands.append(stem)
    print(f"candidates (era-unlabelled, final-labelled): {len(cands)}")
    if not cands:
        print("NO CANDIDATES — the bulk batch may not be stem-shared; panel not assemblable this way.")
        return

    from ultralytics import YOLO
    model = YOLO(str(V5_W))
    picked, seen_prefix = [], set()
    for stem in cands:
        if len(picked) >= 3:
            break
        prefix = stem[:14]
        if prefix in seen_prefix:
            continue
        ip = fin_img.get(stem) or era_img[stem]
        img = cv2.imread(str(ip))
        if img is None:
            continue
        h, w = img.shape[:2]
        gt = boxes(fin_lbl[stem], w, h)
        r = model.predict(img, conf=0.25, imgsz=640, verbose=False)[0]
        dets = [tuple(map(float, r.boxes.xyxy[i].cpu().numpy())) for i in range(len(r.boxes))] if r.boxes is not None else []
        hit = [d for d in dets if any(iou(d, g) >= 0.3 for g in gt)]
        if hit:
            picked.append((stem, img, hit[0], gt))
            seen_prefix.add(prefix)
            print(f"  picked {stem}  (V5 det IoU-matched corrected GT)")

    if not picked:
        print("V5 fired on none of the candidates at conf 0.25 — try conf 0.10 manually.")
        return

    n = len(picked)
    fig, axes = plt.subplots(n, 2, figsize=(9, 2.9 * n))
    axes = np.atleast_2d(axes)
    for i, (stem, img, det, gt) in enumerate(picked):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for j in range(2):
            ax = axes[i][j]
            ax.imshow(rgb)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                x1, y1, x2, y2 = det
                ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, color="red", lw=2))
                ax.set_title(f"V5 detection — V5-era GT: none (scored FP)", fontsize=8)
            else:
                for g in gt:
                    ax.add_patch(plt.Rectangle((g[0], g[1]), g[2] - g[0], g[3] - g[1], fill=False, color="lime", lw=2))
                ax.set_title("corrected corpus GT — same detection is a TP", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGDIR / f"fig_v5_regression.{ext}", dpi=200, bbox_inches="tight")
    print(f"wrote {FIGDIR}/fig_v5_regression.png + .pdf  ({n} rows)")


if __name__ == "__main__":
    main()
