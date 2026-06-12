"""failure_profile_aggregate.py — background failure profile from manual seq-level tags.

282 sequences (273 Svanström + 9 video_drone clips) were tagged by a human from the contact sheets
produced by gen_failure_profile_sheets.py (one representative frame per sequence, tagged 2026-06-11)
into three coarse background classes:

  sky      near-pure sky (incl. clouds)
  horizon  sky with a visible ground / treeline / building strip
  ground   ground, vegetation, structures, or beach dominant

Because background is constant within a Svanström sequence, the tag propagates to every cached frame
of that sequence; per-background bare-detector P/R/F1 (per modality vs its OWN GT, Tier-1 rules) and
the FP-frame distribution are then exact replays of the cached per-frame TP/FP/FN. Caveats printed
with the output: tags are coarse, single-frame-per-sequence, one annotator.

  py -u thesis_eval/failure_profile_aggregate.py
"""
from __future__ import annotations
import csv, json, pickle
from pathlib import Path
import numpy as np
import sys

REPO = Path(__file__).resolve().parent.parent
for _sub in ("eval", "thesis_eval"):
    sys.path.insert(0, str(REPO / _sub))
from metrics import score_detections, compute_prf            # noqa: E402
from pipeline_eval_unified import dets2, gts                  # noqa: E402

CACHE = REPO / "thesis_eval/cache"
OUT = REPO / "thesis_eval/results"
IDX = REPO / "thesis_eval/results/_failure_profile/seq_index.csv"

# manual tags, encoded as (start, end_inclusive, tag) runs over the seq_index.csv idx column
RUNS = [
    (0, 4, "horizon"), (5, 7, "sky"), (8, 11, "horizon"), (12, 12, "sky"), (13, 14, "horizon"),
    (15, 15, "sky"), (16, 17, "horizon"), (18, 19, "sky"), (20, 20, "horizon"), (21, 21, "sky"),
    (22, 23, "horizon"), (24, 27, "sky"), (28, 28, "horizon"), (29, 29, "sky"), (30, 30, "horizon"),
    (31, 36, "sky"), (37, 37, "horizon"), (38, 47, "sky"), (48, 49, "horizon"), (50, 55, "sky"),
    (56, 62, "horizon"), (63, 63, "sky"), (64, 65, "horizon"), (66, 66, "ground"), (67, 67, "sky"),
    (68, 69, "horizon"), (70, 70, "sky"), (71, 74, "horizon"), (75, 76, "sky"), (77, 79, "horizon"),
    (80, 80, "sky"), (81, 81, "horizon"), (82, 82, "sky"), (83, 83, "horizon"), (84, 87, "sky"),
    (88, 90, "horizon"), (91, 92, "ground"), (93, 93, "horizon"), (94, 95, "sky"),
    (96, 100, "horizon"), (101, 101, "sky"), (102, 102, "horizon"), (103, 103, "sky"),
    (104, 104, "horizon"), (105, 105, "ground"), (106, 107, "horizon"), (108, 108, "sky"),
    (109, 109, "horizon"), (110, 110, "ground"), (111, 113, "horizon"), (114, 114, "ground"),
    (115, 116, "sky"), (117, 119, "horizon"), (120, 127, "sky"), (128, 130, "horizon"),
    (131, 135, "sky"), (136, 149, "horizon"), (150, 151, "sky"), (152, 153, "horizon"),
    (154, 162, "ground"), (163, 208, "horizon"), (209, 209, "sky"), (210, 212, "horizon"),
    (213, 215, "sky"), (216, 216, "horizon"), (217, 218, "sky"), (219, 219, "horizon"),
    (220, 228, "sky"), (229, 231, "horizon"), (232, 235, "ground"), (236, 237, "horizon"),
    (238, 239, "ground"), (240, 272, "sky"), (273, 273, "horizon"), (274, 274, "ground"),
    (275, 275, "horizon"), (276, 276, "sky"), (277, 279, "ground"), (280, 280, "ground"),
    (281, 281, "sky"),
]


def tags_by_seq():
    tag_of_idx = {}
    for a, b, t in RUNS:
        for i in range(a, b + 1):
            tag_of_idx[i] = t
    seq_tag = {}
    with open(IDX, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            seq_tag[row["seq"].replace("VIDEO::", "")] = tag_of_idx[int(row["idx"])]
    return seq_tag


def main():
    seq_tag = tags_by_seq()
    results, L = {}, ["# Background failure profile (manual seq-level tags, coarse 3-class)",
                      "Tags: sky / horizon (sky + ground strip) / ground-dominant. One representative "
                      "frame per sequence tagged by one annotator (2026-06-11); the tag propagates to "
                      "all cached frames of the sequence. Bare detectors, per modality vs own GT, "
                      "Tier-1 rules.\n"]
    for surf, slots in (("svanstrom", [("rgb", "rgb_gt", "ft4/rgb"), ("ir", "ir_gt", "v3b/ir")]),
                        ("video_drone", [("rgb", "rgb_gt", "ft4/rgb")])):
        d = pickle.load(open(CACHE / f"{surf}.pkl", "rb"))
        meta, frames = d["meta"], d["frames"]
        res = {}
        for slot, gkey, label in slots:
            agg = {}
            for fr in frames:
                t = seq_tag.get(fr["seq"])
                if t is None:
                    continue
                # split by sequence CONTENT to avoid the background-category confound (pure-sky
                # sequences are disproportionately the helicopter/airplane clips)
                kind = "drone-seqs" if (fr["seq"].startswith("IR_DRONE") or "/" in fr["seq"]) \
                    else "confuser-seqs"
                g = gts(fr[gkey])
                tp, fp, fn = score_detections(dets2(fr[slot]), g, rule=meta["rule"])
                a = agg.setdefault((kind, t), {"tp": 0, "fp": 0, "fn": 0, "n": 0, "n_gt": 0, "fp_frames": 0})
                a["tp"] += tp; a["fp"] += fp; a["fn"] += fn
                a["n"] += 1; a["n_gt"] += len(g); a["fp_frames"] += int(fp > 0)
            res[label] = {f"{k}/{t}": {**compute_prf(v["tp"], v["fp"], v["fn"]),
                                       "n_frames": v["n"], "n_gt": v["n_gt"], "fp_frames": v["fp_frames"],
                                       "fp_frame_rate": round(v["fp_frames"] / max(v["n"], 1), 4)}
                          for (k, t), v in sorted(agg.items())}
        results[surf] = res
        L.append(f"\n## {surf} (n={meta['n']}, rule={meta['rule']})\n")
        for label, blk in res.items():
            L.append(f"**{label}**\n")
            L.append("| background | n_frames | n_gt | P | R | F1 | FP-frame rate |\n|---|---|---|---|---|---|---|")
            for t, p in blk.items():
                r = "—" if p["n_gt"] == 0 else p["recall"]
                f1 = "—" if p["n_gt"] == 0 else p["f1"]
                L.append(f"| {t} | {p['n_frames']} | {p['n_gt']} | {p['precision']} | {r} | {f1} | {p['fp_frame_rate']} |")
            L.append("")
    (OUT / "failure_profile_results.md").write_text("\n".join(L), encoding="utf-8")
    json.dump(results, open(OUT / "failure_profile_results.json", "w"), indent=2, default=float)
    print("DONE ->", OUT / "failure_profile_results.md")


if __name__ == "__main__":
    main()
