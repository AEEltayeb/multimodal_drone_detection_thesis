"""rgbtest_balanced_quota_dryrun.py — ZERO-GPU planning dry-run for the
size x source-balanced re-mine of the RGB filter distill corpus.

Reads ONLY GT labels (no detector, no image decode) for rgb_dataset train, and:
  1. tallies available GT-drone SUPPLY per (sub-source prefix x size-bucket),
  2. simulates the CURRENT collector (alphabetical sorted scan, stride 8, stop at
     8000 drones) -> shows the skew (which cells the shipped corpus starved),
  3. proposes a BALANCED quota: equal-ish per (prefix x size) cell up to a cap,
     capped by availability -> shows the small-drone tail gets populated.

Size-bucket = GT normalized short side (min(w,h)); the production mining buckets
by detection short-side in px (done on GPU). Buckets chosen to mirror the
<16/16-32/32-64/>=64 px veto analysis at the dataset's typical resolution.

  py eval/rgbtest_balanced_quota_dryrun.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
IMGS = Path("G:/drone/dataset/dataset/images/train")
LBLS = Path("G:/drone/dataset/dataset/labels/train")
STRIDE = 8                 # matches SourceConfig rgb_dataset_train
CUR_QUOTA = 8000           # current target_drones for rgb_dataset_train
DRONE_CLASS = 0
# normalized short-side edges -> 4 size bins (xs/s/m/l). At ~1080-1280px native
# these approximate <~16px / 16-32 / 32-64 / >=64.
EDGES = [0.0, 0.015, 0.03, 0.06, 1.01]
SIZE_NAMES = ["xs(<~16px)", "s(16-32)", "m(32-64)", "l(>=64)"]
PER_CELL_CAP = 1200        # proposed per (prefix x size) drone cap
IMG = REPO / "docs/analysis/images"; IMG.mkdir(parents=True, exist_ok=True)
OUT_PNG = IMG / "2026-06-17_rgbtest_balanced_quota.png"
OUT_JSON = REPO / "docs/analysis/2026-06-17_rgbtest_balanced_quota.json"


def prefix_of(name: str) -> str:
    return name.replace("-", "_").split("_")[0]


def size_bin(short: float) -> int:
    for i in range(4):
        if EDGES[i] <= short < EDGES[i + 1]:
            return i
    return 3


def main():
    imgs = sorted(p for p in IMGS.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))
    scan = imgs[::STRIDE]
    print(f"{len(imgs)} train images; scanning {len(scan)} at stride {STRIDE} (labels only)")

    # records in ALPHABETICAL order: (prefix, sizebin)
    recs = []
    supply = defaultdict(int)
    for p in scan:
        lp = LBLS / (p.stem + ".txt")
        if not lp.exists():
            continue
        try:
            lines = lp.read_text().splitlines()
        except Exception:
            continue
        pre = prefix_of(p.name)
        for ln in lines:
            t = ln.split()
            if len(t) >= 5 and int(t[0]) == DRONE_CLASS:
                bw, bh = float(t[3]), float(t[4])
                sb = size_bin(min(bw, bh))
                recs.append((pre, sb)); supply[(pre, sb)] += 1

    prefixes = sorted({k[0] for k in supply})
    total_supply = sum(supply.values())
    print(f"total GT drones in scan: {total_supply}")

    # 1. CURRENT: alphabetical accumulate until CUR_QUOTA
    current = defaultdict(int); got = 0
    for pre, sb in recs:
        if got >= CUR_QUOTA:
            break
        current[(pre, sb)] += 1; got += 1

    # 2. BALANCED: min(supply, cap) per cell
    balanced = {k: min(v, PER_CELL_CAP) for k, v in supply.items()}

    def by_size(d):
        out = [0, 0, 0, 0]
        for (pre, sb), v in d.items():
            out[sb] += v
        return out
    def by_prefix(d):
        out = defaultdict(int)
        for (pre, sb), v in d.items():
            out[pre] += v
        return out

    cur_size, bal_size, sup_size = by_size(current), by_size(balanced), by_size(supply)
    cur_pre, bal_pre, sup_pre = by_prefix(current), by_prefix(balanced), by_prefix(supply)

    print("\n=== drones per SIZE bucket ===")
    print(f"{'bucket':<12} {'supply':>8} {'CURRENT(8000)':>14} {'BALANCED':>10}")
    for i, nm in enumerate(SIZE_NAMES):
        print(f"{nm:<12} {sup_size[i]:>8} {cur_size[i]:>14} {bal_size[i]:>10}")
    print(f"{'TOTAL':<12} {sum(sup_size):>8} {sum(cur_size):>14} {sum(bal_size):>10}")

    print("\n=== drones per SUB-SOURCE ===")
    print(f"{'prefix':<12} {'supply':>8} {'CURRENT':>9} {'BALANCED':>9}")
    for pre in prefixes:
        print(f"{pre:<12} {sup_pre[pre]:>8} {cur_pre.get(pre,0):>9} {bal_pre.get(pre,0):>9}")

    # ── figure ───────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(4); w = 0.4
    ax[0].bar(x - w/2, cur_size, w, label="CURRENT (alpha 8000-quota)", color="tab:red")
    ax[0].bar(x + w/2, bal_size, w, label=f"BALANCED (cap {PER_CELL_CAP}/cell)", color="tab:green")
    ax[0].set_xticks(x); ax[0].set_xticklabels(SIZE_NAMES, rotation=15)
    ax[0].set_ylabel("drones collected"); ax[0].set_title("(a) Drones by SIZE bucket\ncurrent corpus starves the small end")
    ax[0].legend(fontsize=9)
    xp = np.arange(len(prefixes))
    ax[1].bar(xp - w/2, [cur_pre.get(p,0) for p in prefixes], w, label="CURRENT", color="tab:red")
    ax[1].bar(xp + w/2, [bal_pre.get(p,0) for p in prefixes], w, label="BALANCED", color="tab:green")
    ax[1].set_xticks(xp); ax[1].set_xticklabels(prefixes, rotation=40, ha="right", fontsize=8)
    ax[1].set_ylabel("drones collected"); ax[1].set_title("(b) Drones by SUB-SOURCE\nwosdetc (small, late-alphabet) is starved")
    ax[1].legend(fontsize=9)
    fig.suptitle("RGB distill re-mine: size x source balancing populates the under-covered small-drone manifold", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(OUT_PNG, dpi=130); plt.close(fig)

    OUT_JSON.write_text(json.dumps({
        "scan_images": len(scan), "stride": STRIDE, "total_supply_scanned": total_supply,
        "edges_norm_short_side": EDGES, "size_names": SIZE_NAMES, "per_cell_cap": PER_CELL_CAP,
        "by_size": {"supply": sup_size, "current": cur_size, "balanced": bal_size},
        "by_prefix": {p: {"supply": sup_pre[p], "current": cur_pre.get(p, 0), "balanced": bal_pre.get(p, 0)} for p in prefixes},
        "figure": str(OUT_PNG),
    }, indent=2))
    print(f"\nsaved {OUT_PNG}\nsaved {OUT_JSON}")


if __name__ == "__main__":
    main()
