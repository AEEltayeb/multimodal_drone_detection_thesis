"""gen_failure_profile_sheets.py — contact sheets for manual background tagging (notes round 1).

One representative (middle) RGB frame per Svanström sequence (273 seqs in the Tier-1 cache) and one
per video_drone clip, tiled 40 per sheet with running index + seq name. A human tags each sequence
with a coarse background class (sky / sky+treeline / sky+structures / ground-clutter); because
background is constant within a Svanström sequence, the tag propagates to every cached frame of the
sequence, and per-background P/R is then computed from the cached per-frame TP/FP/FN by
failure_profile_aggregate.py using the tag CSV this script's companion produces.

  py -u thesis_eval/gen_failure_profile_sheets.py
Writes thesis_eval/results/_failure_profile/sheet_*.png + seq_index.csv (index -> seq, path).
"""
from __future__ import annotations
import csv, pickle
from pathlib import Path
import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
CACHE = REPO / "thesis_eval/cache"
OUT = REPO / "thesis_eval/results/_failure_profile"
SVAN_RGB_IMG = Path("G:/drone/svanstrom_paired/RGB/images")   # cache key = stem minus "_visible"
VIDEO_ROOT = REPO / "datasets" / "drone detection video tests" / "rgb"
EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def locate(seq, key):
    if seq.startswith("VIDEO::"):
        cat_clip = seq.split("::", 1)[1]
        d = VIDEO_ROOT / cat_clip / "images" / "test"
        for ext in EXTS:
            p = d / (key + ext)
            if p.exists():
                return p
        return None
    for stem in (key + "_visible", key):
        for ext in EXTS:
            p = SVAN_RGB_IMG / (stem + ext)
            if p.exists():
                return p
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    tiles = []

    d = pickle.load(open(CACHE / "svanstrom.pkl", "rb"))
    by_seq = {}
    for fr in d["frames"]:
        by_seq.setdefault(fr["seq"], []).append(fr["key"])
    for seq in sorted(by_seq):
        keys = by_seq[seq]
        rows.append((seq, keys[len(keys) // 2]))

    dv = pickle.load(open(CACHE / "video_drone.pkl", "rb"))
    by_clip = {}
    for fr in dv["frames"]:
        by_clip.setdefault(fr["seq"], []).append(fr["key"])
    for seq in sorted(by_clip):
        keys = by_clip[seq]
        rows.append(("VIDEO::" + seq, keys[len(keys) // 2]))

    # locate images (svanstrom keys are *_visible stems; video keys live under datasets/)
    index = []
    misses = 0
    for i, (seq, key) in enumerate(rows):
        p = locate(seq, key)
        if p is None:
            misses += 1
        index.append((i, seq, key, str(p) if p else ""))
    with open(OUT / "seq_index.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["idx", "seq", "key", "img_path", "background"])
        for r in index:
            w.writerow(list(r) + [""])
    print(f"{len(index)} sequences, {misses} images not located")

    # contact sheets
    TH, TW, COLS, ROWS_PER = 150, 200, 8, 5
    per = COLS * ROWS_PER
    sheet_n = 0
    for s0 in range(0, len(index), per):
        chunk = index[s0:s0 + per]
        canvas = np.full((ROWS_PER * (TH + 24), COLS * TW, 3), 245, np.uint8)
        for j, (i, seq, key, p) in enumerate(chunk):
            r, c = divmod(j, COLS)
            y, x = r * (TH + 24), c * TW
            if p:
                img = cv2.imread(p)
                if img is not None:
                    img = cv2.resize(img, (TW - 4, TH - 4))
                    canvas[y + 20:y + 20 + TH - 4, x + 2:x + 2 + TW - 4] = img
            label = f"{i}: {seq[:24]}"
            cv2.putText(canvas, label, (x + 3, y + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1)
        out = OUT / f"sheet_{sheet_n:02d}.png"
        cv2.imwrite(str(out), canvas)
        print("wrote", out.name)
        sheet_n += 1


if __name__ == "__main__":
    main()
