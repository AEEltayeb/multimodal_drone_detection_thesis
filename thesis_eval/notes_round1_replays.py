"""
thesis_eval/notes_round1_replays.py — notes-round-1 zero-GPU replay extensions (2026-06-11).

Three analyses the thesis review asked for, all replayed from the Tier-1 unified cache
(thesis_eval/cache/*.pkl, written by pipeline_cache_unified.py) with no detector forward pass:

  PART M   modality A/B on the TRUE two-sensor paired surfaces (svanstrom, antiuav):
           rgb_only / ir_only / both / routed[robust8], each bare and +confuser-filter.
           COVERAGE scoring, which differs from the headline trust-aware rule on purpose:
           EVERY modality's GT always counts, and a missing/distrusted modality's GT becomes
           FN. The headline rule (score_trust_aware) EXCLUDES the distrusted side's GT — it
           answers "is the trusted side right"; this table answers "what does a single-camera
           system physically miss", so misses must count. Both GT sides are summed and every
           arm uses the same convention, so arms are directly comparable.
  PART SZ  per-size GT buckets (sqrt-area in ORIGINAL-image pixels: <16 / 16-32 / 32-64 /
           >=64) on every drone surface, bare and +filter. TP/FN bucketed by GT box size,
           FP by predicted box size. n_gt printed per bucket — R in a bucket with n_gt=0 is
           meaningless and is rendered as "—". Median GT sqrt-area printed per surface.
  PART CAT per-confuser-category fire rates (bird / airplane / helicopter / other, derived
           from the filename stem — confuser corpora are unlabeled, category lives in the
           name) on rgb_confuser / gray_confuser / ir_confusers:
           bare -> clf[robust8] -> filt_mlp -> clf->filt, 95% bootstrap CIs on fire rates.
           (gray_confuser router cells excluded — wrong regime, same rationale as part_c.)

  py -u thesis_eval/notes_round1_replays.py            # all relevant surfaces
  py -u thesis_eval/notes_round1_replays.py --only svanstrom,rgb_confuser
"""
from __future__ import annotations
import argparse, json, pickle, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
import sys
for _sub in ("eval", "classifier", "thesis_eval"):
    sys.path.insert(0, str(REPO / _sub))

from metrics import score_detections, compute_prf, iou_iop                      # noqa: E402
from pipeline_eval_unified import (load_classifiers, load_verifiers, batch_labels,   # noqa: E402
                                   batch_probs, dets2, gts, Cells, boot_rate_ci,
                                   KIND_VERIFIER, SLOTS, VERIFIER_LABELS,
                                   RGB_THR_MLP, IR_THR_MLP, GRAY_THR_MLP)

CACHE_DIR = REPO / "thesis_eval" / "cache"
OUT_DIR = REPO / "thesis_eval" / "results"

PAIRED_AB = {"svanstrom", "antiuav"}       # true two-sensor surfaces only (video_drone's second
                                           # channel is DERIVED grayscale, not a second sensor)
SIZE_EDGES = [(0, 16, "<16px"), (16, 32, "16-32px"), (32, 64, "32-64px"), (64, 1e9, ">=64px")]


def sqrt_area(box):
    return float(np.sqrt(max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)))


def size_bucket(box):
    s = sqrt_area(box)
    for lo, hi, name in SIZE_EDGES:
        if lo <= s < hi:
            return name
    return SIZE_EDGES[-1][2]


def match_flags(dets, gt, rule, thr=0.5):
    """Greedy matching identical to metrics.score_detections, but returning WHICH gts matched
    and per-detection match status (needed to bucket TP/FN by GT size and FP by det size)."""
    matched, det_ok = set(), []
    for d_box, _conf in dets:
        best_idx, best_score = -1, 0.0
        for gi, g in enumerate(gt):
            iu, ip = iou_iop(d_box, g)
            s = iu if rule == "iou" else ip
            if s > best_score:
                best_score, best_idx = s, gi
        ok = best_score >= thr and best_idx not in matched
        if ok:
            matched.add(best_idx)
        det_ok.append(ok)
    return matched, det_ok


# ── PART M: modality A/B under coverage scoring ──────────────────────────────────────────────
def part_modality_ab(meta, frames, clfs, verifs):
    rule, is_gray = meta["rule"], meta["is_grayscale"]
    F8, F32 = meta["F8"], meta["F32"]
    cells = Cells()
    labels = batch_labels(clfs["robust8"], np.stack([f["f8_all"] for f in frames]),
                          np.stack([f["f32_all"] for f in frames]), F8, F32) if "robust8" in clfs else None
    ir_vkey, ir_thr = ("aligned_gray", GRAY_THR_MLP) if is_gray else ("aligned", IR_THR_MLP)
    rgb_probs = batch_probs(frames, "rgb", verifs["mlp_v5"])
    ir_probs = batch_probs(frames, "ir", verifs[ir_vkey])

    def side(d, g):
        return score_detections(d, g, rule=rule)

    for i, fr in enumerate(frames):
        rgb_g, ir_g = gts(fr["rgb_gt"]), gts(fr["ir_gt"])
        rm, im = rgb_probs[i] >= RGB_THR_MLP, ir_probs[i] >= ir_thr
        r_all, r_flt = dets2(fr["rgb"]), dets2(fr["rgb"], rm)
        i_all, i_flt = dets2(fr["ir"]), dets2(fr["ir"], im)
        fn_r, fn_i = (0, 0, len(rgb_g)), (0, 0, len(ir_g))

        def add(arm, rs, is_):
            cells.add(arm, *(a + b for a, b in zip(rs, is_)))

        add("rgb_only bare",  side(r_all, rgb_g), fn_i)
        add("rgb_only +filt", side(r_flt, rgb_g), fn_i)
        add("ir_only bare",   fn_r, side(i_all, ir_g))
        add("ir_only +filt",  fn_r, side(i_flt, ir_g))
        add("both bare",      side(r_all, rgb_g), side(i_all, ir_g))
        add("both +filt",     side(r_flt, rgb_g), side(i_flt, ir_g))
        if labels is not None:
            L = int(labels[i])
            rs = side(r_all, rgb_g) if L in (1, 3) else fn_r
            is_ = side(i_all, ir_g) if L in (2, 3) else fn_i
            add("routed[robust8] bare", rs, is_)
            rs = side(r_flt, rgb_g) if L in (1, 3) else fn_r
            is_ = side(i_flt, ir_g) if L in (2, 3) else fn_i
            add("routed[robust8] +filt", rs, is_)
    return cells.report()


# ── PART SZ: per-size buckets (GT sqrt-area, original-image px) ─────────────────────────────
def part_per_size(meta, frames, verifs):
    rule = meta["rule"]
    out = {}
    for slot_key, gt_key in SLOTS[meta["kind"]]:
        vslot, vkey, thr = KIND_VERIFIER.get(meta["kind"], ("rgb", "mlp_v5", RGB_THR_MLP))
        if meta["kind"] in ("paired", "grayrgb_paired"):
            vkey, thr = (("mlp_v5", RGB_THR_MLP) if slot_key == "rgb" else
                         (("aligned_gray", GRAY_THR_MLP) if meta["is_grayscale"] else ("aligned", IR_THR_MLP)))
        if vkey not in verifs:
            continue
        probs = batch_probs(frames, slot_key, verifs[vkey])
        gname = "rgb_gt" if gt_key == "gt_rgb" else "ir_gt"
        stats = {arm: {b[2]: {"tp": 0, "fp": 0, "fn": 0} for b in SIZE_EDGES} for arm in ("bare", "filt")}
        areas = []
        for i, fr in enumerate(frames):
            gt = gts(fr[gname])
            areas += [sqrt_area(g) for g in gt]
            for arm, mask in (("bare", None), ("filt", probs[i] >= thr)):
                dd = dets2(fr[slot_key], mask)
                matched, det_ok = match_flags(dd, gt, rule)
                for gi, g in enumerate(gt):
                    stats[arm][size_bucket(g)]["tp" if gi in matched else "fn"] += 1
                for (box, _c), ok in zip(dd, det_ok):
                    if not ok:
                        stats[arm][size_bucket(box)]["fp"] += 1
        label = f"{'ft4' if slot_key == 'rgb' else 'v3b'}/{slot_key}"
        out[label] = {"median_gt_sqrt_area_px": round(float(np.median(areas)), 1) if areas else None,
                      "n_gt_total": len(areas), "verifier": vkey,
                      "buckets": {arm: {b: compute_prf(v["tp"], v["fp"], v["fn"]) | {"n_gt": v["tp"] + v["fn"]}
                                        for b, v in arm_stats.items()}
                                  for arm, arm_stats in stats.items()}}
    return out


# ── PART CAT: per-category confuser fire ─────────────────────────────────────────────────────
def cat_of(key):
    k = str(key).lower()
    if "airplane" in k or "plane" in k:
        return "airplane"
    if "bird" in k:
        return "bird"
    if "heli" in k:
        return "helicopter"
    return "other"


def part_per_category(meta, frames, clfs, verifs):
    slot_key, vkey, thr = KIND_VERIFIER[meta["kind"]]
    probs = batch_probs(frames, slot_key, verifs[vkey])
    labels = None
    if "robust8" in clfs and meta["kind"] != "gray":   # gray router cells = wrong regime (see part_c)
        labels = batch_labels(clfs["robust8"], np.stack([f["f8_all"] for f in frames]),
                              np.stack([f["f32_all"] for f in frames]), meta["F8"], meta["F32"])
    stages = ["bare", "filt_mlp"] + (["clf[robust8]", "clf->filt[robust8]"] if labels is not None else [])
    agg = {}
    for i, fr in enumerate(frames):
        c = cat_of(fr["key"])
        a = agg.setdefault(c, {s: {"fp": 0, "flags": []} for s in stages} | {"n": 0})
        a["n"] += 1
        n = len(fr[slot_key]["confs"])
        k_mlp = int((probs[i] >= thr).sum())
        vals = {"bare": n, "filt_mlp": k_mlp}
        if labels is not None:
            keep = int(labels[i]) == 3 or (int(labels[i]) == 1 and slot_key == "rgb") \
                   or (int(labels[i]) == 2 and slot_key == "ir")
            vals["clf[robust8]"] = n if keep else 0
            vals["clf->filt[robust8]"] = k_mlp if keep else 0
        for s, k in vals.items():
            a[s]["fp"] += k
            a[s]["flags"].append(int(k > 0))
    out = {}
    for c, a in sorted(agg.items()):
        out[c] = {"n": a["n"], "verifier": vkey}
        for s in stages:
            ci = boot_rate_ci(a[s]["flags"])
            cell = {"FP": a[s]["fp"], "fired": int(sum(a[s]["flags"])),
                    "fire_rate": round(sum(a[s]["flags"]) / max(a["n"], 1), 4)}
            if ci:
                cell["fire_ci"] = [round(ci[0], 4), round(ci[1], 4)]
            out[c][s] = cell
    return out


# ── report ────────────────────────────────────────────────────────────────────────────────────
def _f1c(p):
    ci = p.get("f1_ci")
    return f"{p['f1']}" + (f" [{ci[0]}–{ci[1]}]" if ci else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=str(CACHE_DIR))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    cdir, outdir = Path(args.cache_dir), Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    print(f"Notes-round-1 replays <- {cdir} (zero-GPU)")
    clfs, verifs = load_classifiers(), load_verifiers()
    results = {}
    L = ["# Notes-round-1 replay extensions (modality A/B, per-size, per-category)",
         f"{time.strftime('%Y-%m-%d %H:%M')} | same Tier-1 cache + shipped stack as tier1_results.json",
         "PART M uses COVERAGE scoring (missing/distrusted modality's GT counts as FN) — deliberately "
         "different from the headline trust-aware rule; see module docstring. 95% bootstrap CIs.\n"]

    for pkl in sorted(cdir.glob("*.pkl")):
        if only and pkl.stem not in only:
            continue
        d = pickle.load(open(pkl, "rb"))
        meta, frames = d["meta"], d["frames"]
        name = meta["name"]; t0 = time.time(); res = {}
        if name in PAIRED_AB and meta["has_drones"]:
            res["M_modality_ab"] = part_modality_ab(meta, frames, clfs, verifs)
        if meta["has_drones"]:
            res["SZ_per_size"] = part_per_size(meta, frames, verifs)
        if not meta["has_drones"] and meta["kind"] in KIND_VERIFIER:
            res["CAT_confuser"] = part_per_category(meta, frames, clfs, verifs)
        if not res:
            continue
        res["meta"] = {k: meta[k] for k in ("name", "kind", "rule", "n", "rgb_imgsz", "ir_imgsz") if k in meta}
        res["meta"]["n_source"] = meta.get("n_source", meta["n"])
        results[name] = res
        print(f"  [{name}] n={meta['n']} {time.time()-t0:.1f}s")

        L.append(f"\n## {name}  (n={meta['n']} of {res['meta']['n_source']}, rule={meta['rule']}, "
                 f"imgsz rgb={meta['rgb_imgsz']}/ir={meta['ir_imgsz']})\n")
        if "M_modality_ab" in res:
            L.append("**M — modality A/B (coverage scoring: absent/distrusted side's GT = FN)**\n")
            L.append("| arm | TP | FP | FN | P | R | F1 [95% CI] |\n|---|---|---|---|---|---|---|")
            for c, p in res["M_modality_ab"].items():
                L.append(f"| {c} | {p['TP']} | {p['FP']} | {p['FN']} | {p['precision']} | {p['recall']} | {_f1c(p)} |")
        if "SZ_per_size" in res:
            for mod, blk in res["SZ_per_size"].items():
                L.append(f"\n**SZ — per-size, {mod} (median GT sqrt-area {blk['median_gt_sqrt_area_px']} px, "
                         f"n_gt={blk['n_gt_total']}, filter={VERIFIER_LABELS.get(blk['verifier'], blk['verifier'])})**\n")
                L.append("| bucket | n_gt | bare P | bare R | bare F1 | +filt P | +filt R | +filt F1 |\n"
                         "|---|---|---|---|---|---|---|---|")
                for b in [e[2] for e in SIZE_EDGES]:
                    pb, pf = blk["buckets"]["bare"][b], blk["buckets"]["filt"][b]
                    if pb["n_gt"] == 0 and pb["FP"] == 0 and pf["FP"] == 0:
                        continue
                    def fmt(p):
                        return (f"{p['precision']} | — | —" if p["n_gt"] == 0
                                else f"{p['precision']} | {p['recall']} | {p['f1']}")
                    L.append(f"| {b} | {pb['n_gt']} | {fmt(pb)} | {fmt(pf)} |")
        if "CAT_confuser" in res:
            vk = next(iter(res["CAT_confuser"].values()))["verifier"]
            L.append(f"\n**CAT — per-category fire rates (filter={VERIFIER_LABELS.get(vk, vk)})**\n")
            stages = [s for s in ("bare", "clf[robust8]", "filt_mlp", "clf->filt[robust8]")
                      if s in next(iter(res["CAT_confuser"].values()))]
            L.append("| category | n | " + " | ".join(f"{s} fire [CI]" for s in stages) + " |\n"
                     "|---|---|" + "---|" * len(stages))
            for c, blk in res["CAT_confuser"].items():
                row = [c, str(blk["n"])]
                for s in stages:
                    p = blk[s]; ci = p.get("fire_ci")
                    row.append(f"{p['fire_rate']}" + (f" [{ci[0]}–{ci[1]}]" if ci else ""))
                L.append("| " + " | ".join(row) + " |")

    (outdir / "notes_round1_results.md").write_text("\n".join(L), encoding="utf-8")
    json.dump(results, open(outdir / "notes_round1_results.json", "w"), indent=2, default=float)
    print(f"\nDONE -> {outdir/'notes_round1_results.md'}  +  notes_round1_results.json")


if __name__ == "__main__":
    main()
