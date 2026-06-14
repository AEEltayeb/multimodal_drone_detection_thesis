"""
thesis_eval/leakage_controlled_replay.py — round-3 N12/N13 leakage-controlled replays (2026-06-12).

Plan-mode audit found the §3.3 "Anti-UAV cascade numbers are fully clean" claim false:
  - IR_dset_final/train holds 17,314 svan_* (+1,325 czoom_svan_*) frames; 37.3% of the Tier-1
    Svanström eval subset are exact training images (renamed svan_<key> <-> <key>_infrared).
  - IR_dset_final/train holds 31,394 *auv_* Anti-UAV frames; 30 of the eval test dir's 91
    segments overlap training (6.2% exact frames in the eval subset).
  - classifier/generate_routing_data.py mines the trust router's Anti-UAV rows from the very
    same test dir (--auv-root default), and its Svanström rows from svanstrom_paired.

This script produces the honest numbers, all zero-GPU replays of the Tier-1 unified cache:

  DET-CLEAN  per surface, replay every pipeline arm (part_a bare + part_b ablation) on the
             frames whose SEQUENCE has zero frames in IR_dset_final/train:
               svanstrom: 54/279 sequences (never in IR train)
               antiuav:   61/91 segments  (never in IR train)
             Sequence-level, not frame-level: the detector saw sibling frames 3 frames away,
             so frame exclusion alone would still leak.
  ROUTER     reconstruct the shipped router's training groups exactly (lean19 CSV +
             GroupShuffleSplit(test_size=0.25, random_state=42) on _seq, identical to
             classifier/train_routing_robust.py) and intersect with the det-clean sets.
  CASCADE-CLEAN  if det-clean ∩ router-untrained survives, replay those frames too.

Clean sets are DERIVED LIVE from the G: listings (not hardcoded), so the output JSON is its
own provenance: it records the overlap counts it measured plus the clean sequence lists.

  py -u thesis_eval/leakage_controlled_replay.py
Outputs: thesis_eval/results/leakage_controlled.{json,md}
"""
from __future__ import annotations
import json, os, pickle, re, time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
import sys
for _sub in ("eval", "classifier", "thesis_eval"):
    sys.path.insert(0, str(REPO / _sub))

from pipeline_eval_unified import (load_classifiers, load_verifiers, part_a, part_b)  # noqa: E402

CACHE_DIR = REPO / "thesis_eval" / "cache"
OUT_DIR = REPO / "thesis_eval" / "results"
IR_TRAIN_IMAGES = Path("G:/drone/IR_dset_final/train/images")
LEAN19_CSV = REPO / "models/routers/lean_ft4/fusion_dataset_lean19.csv"

SVAN_SEQ_RE = re.compile(r"(IR_[A-Z]+_\d+)")
AUV_SEG_RE = re.compile(r"(20\d{6}_\d{6}_\d+_\d+)")
PATCH_THR = 0.5  # same default as pipeline_eval_unified main()

# arms worth reporting in the md (json keeps everything part_b emits)
MD_ARMS = ("bare", "filt_mlp", "clf[robust8]", "clf->filt[robust8]")


def train_contaminated_ids():
    """Scan IR_dset_final/train ONCE; return (svan seqs, auv segments) present in training,
    plus the EXACT-FRAME key sets (full-resolution training images only; czoom_* zoom crops are
    augmented derivatives, counted at sequence level but not as exact frames).
    Catches czoom_/dv5_ prefixed copies because the regexes match anywhere in the name."""
    svan, auv = set(), set()
    svan_exact, auv_exact = set(), set()
    n = n_svan_full = n_czoom_svan = n_auv_all = 0
    for name in os.listdir(IR_TRAIN_IMAGES):
        n += 1
        stem = name.rsplit(".", 1)[0]
        m = SVAN_SEQ_RE.search(name)
        if m and "svan" in name:
            svan.add(m.group(1))
            if stem.startswith("svan_"):
                svan_exact.add(stem[len("svan_"):])   # svan_IR_DRONE_001_f000000 -> cache key
                n_svan_full += 1
            elif stem.startswith("czoom_svan_"):
                n_czoom_svan += 1
            continue
        m = AUV_SEG_RE.search(name)
        if m:
            auv.add(m.group(1))
            n_auv_all += 1
            if not stem.startswith("czoom_"):
                # dv5_auv_<seg>_<frame6> -> <seg>_f<frame6> (cache key convention)
                k = stem.split("auv_", 1)[-1]
                seg, frame = k.rsplit("_", 1)
                if frame.isdigit():
                    auv_exact.add(f"{seg}_f{frame}")
    counts = {"svan_full_train_frames": n_svan_full, "czoom_svan_train_frames": n_czoom_svan,
              "auv_train_frames_incl_czoom": n_auv_all}
    return svan, auv, n, svan_exact, auv_exact, counts


def router_trained_ids():
    """Replicate train_routing_robust.py's split exactly; return svan seqs + auv segments
    whose rows are in the router's TRAIN portion."""
    import pandas as pd
    from sklearn.model_selection import GroupShuffleSplit
    from train_routing_robust import seq_id
    df = pd.read_csv(LEAN19_CSV)
    df["_seq"] = [seq_id(s, src) for s, src in zip(df["stem"], df["source"])]
    tr, _te = next(GroupShuffleSplit(1, test_size=0.25, random_state=42)
                   .split(df, df["trust_label"], df["_seq"]))
    train_seqs = set(df.iloc[tr]["_seq"])
    svan, auv = set(), set()
    for s in train_seqs:
        m = SVAN_SEQ_RE.search(s)
        if m and s.startswith("svanstrom::"):
            svan.add(m.group(1))
        m = AUV_SEG_RE.search(s)
        if m and s.startswith("antiuav::"):
            auv.add(m.group(1))
    meta = {"csv": str(LEAN19_CSV), "rows": len(df), "train_rows": len(tr),
            "split": "GroupShuffleSplit(test_size=0.25, random_state=42) on _seq (identical to train_routing_robust.py)"}
    return svan, auv, meta


def replay(meta, frames, clfs, verifs):
    res = {"A_bare": part_a(meta, frames), "B_pipeline": part_b(meta, frames, clfs, verifs, PATCH_THR)}
    res["n"] = len(frames)
    res["n_gt_rgb"] = int(sum(len(f["rgb_gt"]) for f in frames))
    res["n_gt_ir"] = int(sum(len(f["ir_gt"]) for f in frames))
    return res


def seq_of(surface, fr):
    rex = SVAN_SEQ_RE if surface == "svanstrom" else AUV_SEG_RE
    m = rex.search(fr["seq"])
    return m.group(1) if m else fr["seq"]


def md_rows(tag, block, headline):
    L = [f"\n**{tag}** (n={block['n']}, GT rgb/ir = {block['n_gt_rgb']}/{block['n_gt_ir']})\n",
         "| arm | clean P | clean R | clean F1 [95% CI] | headline F1 |",
         "|---|---|---|---|---|"]
    for arm in MD_ARMS:
        p = block["B_pipeline"].get(arm)
        if not p:
            continue
        ci = p.get("f1_ci")
        h = headline["B_pipeline"].get(arm, {}).get("f1", "—")
        L.append(f"| {arm} | {p['precision']} | {p['recall']} | {p['f1']}"
                 + (f" [{ci[0]}–{ci[1]}]" if ci else "") + f" | {h} |")
    for lab, p in block["A_bare"].items():
        hh = headline["A_bare"].get(lab, {}).get("f1", "—")
        L.append(f"| bare {lab} (own GT) | {p['precision']} | {p['recall']} | {p['f1']} | {hh} |")
    return L


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("deriving contaminated ids from", IR_TRAIN_IMAGES)
    tr_svan, tr_auv, n_train, svan_exact, auv_exact, train_counts = train_contaminated_ids()
    print(f"  IR train: {n_train:,} files -> {len(tr_svan)} svan seqs, {len(tr_auv)} auv segments contaminated")

    # exact-frame overlap of the FULL eval corpora with IR training (renamed copies)
    svan_corpus = {f.rsplit(".", 1)[0].replace("_infrared", "")
                   for f in os.listdir("G:/drone/svanstrom_paired/IR/images")}
    auv_corpus = {f.rsplit(".", 1)[0].replace("_infrared_f", "_f")
                  for f in os.listdir("G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/images")}
    exact_full = {"svan": {"overlap": len(svan_corpus & svan_exact), "corpus": len(svan_corpus)},
                  "auv": {"overlap": len(auv_corpus & auv_exact), "corpus": len(auv_corpus)}}
    print(f"  exact frames in IR train: svan {exact_full['svan']['overlap']}/{exact_full['svan']['corpus']}"
          f" | auv {exact_full['auv']['overlap']}/{exact_full['auv']['corpus']} (full corpora)")

    r_svan, r_auv, router_meta = router_trained_ids()
    print(f"  router train groups: {len(r_svan)} svan seqs, {len(r_auv)} auv segments "
          f"({router_meta['train_rows']}/{router_meta['rows']} rows)")

    clfs, verifs = load_classifiers(), load_verifiers()
    headline = json.load(open(OUT_DIR / "tier1_results.json"))

    out = {"derived": {"ir_train_files": n_train, **train_counts,
                       "svan_train_seqs": len(tr_svan), "auv_train_segments": len(tr_auv),
                       "exact_frame_overlap_full_corpus": exact_full,
                       "router": router_meta,
                       "router_train_svan_seqs": len(r_svan), "router_train_auv_segments": len(r_auv)}}
    L = ["# Leakage-controlled replays (round-3 N12/N13) — sequence-level clean subsets",
         f"{time.strftime('%Y-%m-%d %H:%M')} | Tier-1 cache replay, zero-GPU | patch_thr={PATCH_THR}",
         "det-clean = eval sequences with ZERO frames in IR_dset_final/train;",
         "cascade-clean = det-clean AND not in the shipped router's reconstructed train groups."]

    for surface, contaminated, router_trained in (("svanstrom", tr_svan, r_svan),
                                                  ("antiuav", tr_auv, r_auv)):
        d = pickle.load(open(CACHE_DIR / f"{surface}.pkl", "rb"))
        meta, frames = d["meta"], d["frames"]
        all_seqs = {seq_of(surface, f) for f in frames}
        det_clean_seqs = {s for s in all_seqs if s not in contaminated}
        casc_clean_seqs = {s for s in det_clean_seqs if s not in router_trained}

        det_frames = [f for f in frames if seq_of(surface, f) in det_clean_seqs]
        casc_frames = [f for f in frames if seq_of(surface, f) in casc_clean_seqs]
        exact = svan_exact if surface == "svanstrom" else auv_exact
        n_exact_subset = sum(1 for f in frames if f["key"] in exact)
        print(f"[{surface}] eval seqs {len(all_seqs)} | exact-in-train {n_exact_subset}/{len(frames)} "
              f"| det-clean {len(det_clean_seqs)} ({len(det_frames)} frames) "
              f"| cascade-clean {len(casc_clean_seqs)} ({len(casc_frames)} frames)")

        blk = {"eval_seqs": len(all_seqs), "n_eval_frames": len(frames),
               "exact_train_frames_in_eval_subset": n_exact_subset,
               "det_clean": {"seqs": sorted(det_clean_seqs)},
               "cascade_clean": {"seqs": sorted(casc_clean_seqs)}}
        blk["det_clean"].update(replay(meta, det_frames, clfs, verifs))
        L += [f"\n## {surface} — det-clean: {len(det_clean_seqs)}/{len(all_seqs)} sequences"]
        L += md_rows("det-clean (IR detector never trained on these sequences)",
                     blk["det_clean"], headline[surface])

        if casc_clean_seqs and len(casc_frames) >= 100:
            blk["cascade_clean"].update(replay(meta, casc_frames, clfs, verifs))
            L += md_rows(f"cascade-clean ({len(casc_clean_seqs)} seqs; also outside router training)",
                         blk["cascade_clean"], headline[surface])
        else:
            blk["cascade_clean"]["note"] = f"only {len(casc_frames)} frames — too small to score"
            L += [f"\ncascade-clean: only {len(casc_clean_seqs)} seqs / {len(casc_frames)} frames — too small to score."]
        out[surface] = blk

    json.dump(out, open(OUT_DIR / "leakage_controlled.json", "w"), indent=2, default=float)
    (OUT_DIR / "leakage_controlled.md").write_text("\n".join(L), encoding="utf-8")
    print(f"\nDONE {time.time()-t0:.1f}s -> {OUT_DIR/'leakage_controlled.md'} + leakage_controlled.json")


if __name__ == "__main__":
    main()
