"""
thesis_eval/conf_sweep_replay.py — ZERO-GPU detector-confidence sweep from a unified cache.

QUESTION (2026-06-11): does LOWERING the detector conf floor (more recall) + the MLP verifier
beat the production operating points (rgb 0.25 / ir 0.40 / gray 0.40 from gui/fusion_settings.json)?
The Tier-1 cache floor is 0.25, so points below it need the low-conf cache:

  py -u thesis_eval/pipeline_cache_unified.py --conf 0.05 --cache-dir thesis_eval/cache_conf005 --no-patch --target 500  --only antiuav
  py -u thesis_eval/pipeline_cache_unified.py --conf 0.05 --cache-dir thesis_eval/cache_conf005 --no-patch --target 4000 --only selcom_val,rgb_confuser,gray_confuser,ir_confusers,ir_dset_final,rgb_dataset_test,svanstrom_gray

Then:  py -u thesis_eval/conf_sweep_replay.py --cache-dir thesis_eval/cache_conf005
(Also runs on the Tier-1 cache for the t >= 0.25 half-grid: --cache-dir thesis_eval/cache —
 points below a cache's conf floor are CENSORED, never extrapolated.)

Per surface kind:
  solo drone (rgb/ir/gray/rawrgb)  P/R/F1 + bootstrap CI per conf t: bare vs +MLP verifier
                                   (shipped thr rgb 0.25 / ir 0.05 / gray 0.25) + an F1 grid
                                   over verifier thr (incl. the GUI's 0.15).
  confuser (no GT)                 FP count + frame fire-rate per t: bare vs +verifier.
  paired (antiuav)                 trust-aware (NO union) full pipeline per point: bare ->
                                   +filter -> robust8 -> robust8->filter, on three slices:
                                   rgb sweep (ir@0.40), ir sweep (rgb@0.25), diagonal.
                                   robust8 f8 rows recomputed from the conf-masked dets
                                   (pixel-free cols — parity asserted in pipeline_eval_unified).

Figures -> docs/analysis/images/conf_sweep/   tables -> thesis_eval/results/conf_sweep/
"""
from __future__ import annotations
import argparse, json, pickle, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
import sys
for _sub in ("eval", "classifier", "thesis_eval"):
    sys.path.insert(0, str(REPO / _sub))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                   # noqa: E402

from metrics import score_detections, compute_prf, score_trust_aware             # noqa: E402
from pipeline_eval_unified import (                                                # noqa: E402
    load_classifiers, load_verifiers, batch_probs, batch_labels, dets2, dets5, gts,
    boot_f1_ci, boot_rate_ci, recompute_f8, _sum_ta,
    KIND_VERIFIER, RGB_THR_MLP, IR_THR_MLP, GRAY_THR_MLP, VERIFIER_LABELS)

CONF_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]
MLP_GRID  = [0.10, 0.15, 0.25, 0.40, 0.60]          # verifier-thr side grid (0.15 = GUI setting)
DEFAULTS  = {"rgb": 0.25, "ir": 0.40, "gray": 0.40, "rawrgb": 0.40}  # fusion_settings.json
RGB_DEF, IR_DEF = 0.25, 0.40

OUT_DIR = REPO / "thesis_eval" / "results" / "conf_sweep"
FIG_DIR = REPO / "docs" / "analysis" / "images" / "conf_sweep"


def conf_mask(slot, t):
    c = np.asarray(slot["confs"], np.float32)
    return c >= t if len(c) else np.zeros(0, bool)


def grid_for(meta):
    floor = float(meta.get("conf", 0.25))
    g = [t for t in CONF_GRID if t >= floor - 1e-9]
    cen = [t for t in CONF_GRID if t < floor - 1e-9]
    if cen:
        print(f"    [censored below cache floor {floor}: {cen}]")
    return g


# ── solo surfaces (drone or confuser): bare vs +verifier per conf t ──────────────────────────────────
def sweep_solo(meta, frames, verifs):
    kind = meta["kind"]
    slot_key, vkey, vthr = KIND_VERIFIER[kind]
    if vkey not in verifs:
        return {}
    probs = batch_probs(frames, slot_key, verifs[vkey])
    gt_key = "rgb_gt" if slot_key == "rgb" else "ir_gt"
    rule, drones = meta["rule"], meta["has_drones"]
    out = {"verifier": vkey, "vthr": vthr, "default_conf": DEFAULTS[kind], "rows": [], "mlp_grid": {}}
    for t in grid_for(meta):
        if drones:
            cells = {"bare": [], "filt": []}
            for i, fr in enumerate(frames):
                slot, gt = fr[slot_key], gts(fr[gt_key])
                mc = conf_mask(slot, t)
                cells["bare"].append(score_detections(dets2(slot, mc), gt, rule=rule))
                cells["filt"].append(score_detections(dets2(slot, mc & (probs[i] >= vthr)), gt, rule=rule))
            row = {"conf": t}
            for c, rows in cells.items():
                a = np.asarray(rows, np.int64)
                prf = compute_prf(int(a[:, 0].sum()), int(a[:, 1].sum()), int(a[:, 2].sum()))
                ci = boot_f1_ci(a)
                row[c] = {**prf, "f1_ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None}
            out["rows"].append(row)
        else:
            row = {"conf": t}
            for cname, extra in (("bare", None), ("filt", vthr)):
                fps, flags = 0, []
                for i, fr in enumerate(frames):
                    m = conf_mask(fr[slot_key], t)
                    if extra is not None:
                        m = m & (probs[i] >= extra)
                    k = int(m.sum()); fps += k; flags.append(int(k > 0))
                ci = boot_rate_ci(flags)
                row[cname] = {"FP": fps, "fire_rate": round(sum(flags) / max(meta["n"], 1), 4),
                              "fire_ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None}
            out["rows"].append(row)
        # verifier-thr side grid (F1 or FP only — compact)
        g = {}
        for mt in MLP_GRID:
            tp = fp = fn = 0
            for i, fr in enumerate(frames):
                m = conf_mask(fr[slot_key], t) & (probs[i] >= mt)
                if drones:
                    a, b, c = score_detections(dets2(fr[slot_key], m), gts(fr[gt_key]), rule=rule)
                    tp += a; fp += b; fn += c
                else:
                    fp += int(m.sum())
            g[str(mt)] = compute_prf(tp, fp, fn)["f1"] if drones else fp
        out["mlp_grid"][str(t)] = g
    return out


# ── paired (antiuav): full-pipeline sweep, trust-aware scoring ────────────────────────────────────────
def sweep_paired(meta, frames, clfs, verifs):
    rule, is_gray = meta["rule"], meta["is_grayscale"]
    ir_vkey = "aligned_gray" if is_gray else "aligned"
    rgb_probs = batch_probs(frames, "rgb", verifs["mlp_v5"])
    ir_probs = batch_probs(frames, "ir", verifs[ir_vkey])
    ir_vthr = GRAY_THR_MLP if is_gray else IR_THR_MLP
    r8 = clfs.get("robust8")

    g = grid_for(meta)
    points = ([("rgb", t, IR_DEF) for t in g] + [("ir", RGB_DEF, t) for t in g]
              + [("diag", t, t) for t in g])
    seen, uniq = set(), []
    for sl, rt, it in points:
        if (sl, rt, it) not in seen:
            seen.add((sl, rt, it)); uniq.append((sl, rt, it))

    out = {"rgb_default": RGB_DEF, "ir_default": IR_DEF, "ir_verifier": ir_vkey, "rows": []}
    F8, F32 = meta["F8"], meta["F32"]
    F32mat = np.stack([fr["f32_all"] for fr in frames])
    for sl, rt, it in uniq:
        per = {"bare": [], "filt": [], "clf": [], "clf_filt": []}
        f8rows, payload = [], []
        for i, fr in enumerate(frames):
            rgb, ir = fr["rgb"], fr["ir"]
            rgb_g, ir_g = gts(fr["rgb_gt"]), gts(fr["ir_gt"])
            rmc, imc = conf_mask(rgb, rt), conf_mask(ir, it)
            rmf, imf = rmc & (rgb_probs[i] >= RGB_THR_MLP), imc & (ir_probs[i] >= ir_vthr)

            def TA(label, rd, idd):
                return _sum_ta(score_trust_aware(label, rd, idd, rgb_g, ir_g, 1920, 1080, 1920, 1080,
                                                 is_paired=True, rule=rule))
            per["bare"].append(TA(3, dets2(rgb, rmc), dets2(ir, imc)))
            per["filt"].append(TA(3, dets2(rgb, rmf), dets2(ir, imf)))
            if r8:
                f8rows.append(recompute_f8(dets5(rgb, rmc), dets5(ir, imc), is_gray))
                payload.append((dets2(rgb, rmc), dets2(ir, imc), dets2(rgb, rmf), dets2(ir, imf), rgb_g, ir_g))
        if r8:
            labels = batch_labels(r8, np.stack(f8rows), F32mat, F8, F32)
            for (ra, ia, rf, if_, rgb_g, ir_g), L in zip(payload, labels):
                def TA2(label, rd, idd):
                    return _sum_ta(score_trust_aware(int(label), rd, idd, rgb_g, ir_g, 1920, 1080, 1920, 1080,
                                                     is_paired=True, rule=rule))
                per["clf"].append(TA2(L, ra, ia))
                per["clf_filt"].append(TA2(L, rf, if_))
        row = {"slice": sl, "rgb_conf": rt, "ir_conf": it}
        for c, rows in per.items():
            if not rows:
                continue
            a = np.asarray(rows, np.int64)
            prf = compute_prf(int(a[:, 0].sum()), int(a[:, 1].sum()), int(a[:, 2].sum()))
            ci = boot_f1_ci(a)
            row[c] = {**prf, "f1_ci": [round(ci[0], 4), round(ci[1], 4)] if ci else None}
        out["rows"].append(row)
    return out


# ── figures ───────────────────────────────────────────────────────────────────────────────────────────
def fig_solo(name, res, drones):
    rows = res["rows"]
    x = [r["conf"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    if drones:
        ax.plot(x, [r["bare"]["f1"] for r in rows], "o-", label="bare detector")
        ax.plot(x, [r["filt"]["f1"] for r in rows], "s-", label=f"+ MLP verifier @{res['vthr']}")
        lo = [r["filt"]["f1_ci"][0] if r["filt"]["f1_ci"] else r["filt"]["f1"] for r in rows]
        hi = [r["filt"]["f1_ci"][1] if r["filt"]["f1_ci"] else r["filt"]["f1"] for r in rows]
        ax.fill_between(x, lo, hi, alpha=0.15)
        ax.set_ylabel("F1")
    else:
        ax.plot(x, [max(r["bare"]["FP"], 1) for r in rows], "o-", label="bare detector")
        ax.plot(x, [max(r["filt"]["FP"], 1) for r in rows], "s-", label=f"+ MLP verifier @{res['vthr']}")
        ax.set_yscale("log"); ax.set_ylabel("confuser FP (log)")
    ax.axvline(res["default_conf"], color="gray", ls="--", lw=1, label=f"default conf {res['default_conf']}")
    ax.set_xlabel("detector conf threshold"); ax.set_title(f"{name} — conf sweep ({VERIFIER_LABELS.get(res['verifier'], res['verifier'])})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(FIG_DIR / f"{name}_conf_sweep.png", dpi=130); plt.close(fig)


def fig_paired(name, res):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    for ax, sl, xlabel, dflt in ((axes[0], "rgb", "RGB conf (IR @0.40)", RGB_DEF),
                                 (axes[1], "ir", "IR conf (RGB @0.25)", IR_DEF),
                                 (axes[2], "diag", "conf (both)", None)):
        rows = [r for r in res["rows"] if r["slice"] == sl]
        x = [r["rgb_conf" if sl != "ir" else "ir_conf"] for r in rows]
        for cell, lab, mk in (("bare", "bare", "o-"), ("filt", "+filter", "s-"),
                              ("clf", "robust8", "^-"), ("clf_filt", "robust8→filter", "d-")):
            if rows and cell in rows[0]:
                ax.plot(x, [r[cell]["f1"] for r in rows], mk, ms=4, label=lab)
        if dflt:
            ax.axvline(dflt, color="gray", ls="--", lw=1)
        ax.set_xlabel(xlabel); ax.grid(alpha=0.3)
    axes[0].set_ylabel("trust-aware F1"); axes[0].legend(fontsize=8)
    fig.suptitle(f"{name} — full-pipeline conf sweep (per-modality scoring, no union)")
    fig.tight_layout(); fig.savefig(FIG_DIR / f"{name}_pipeline_conf_sweep.png", dpi=130); plt.close(fig)


# ── report ────────────────────────────────────────────────────────────────────────────────────────────
def md_solo(L, name, meta, res):
    drones = meta["has_drones"]
    L.append(f"\n## {name}  (n={meta['n']} of {meta.get('n_source', meta['n'])}, kind={meta['kind']}, "
             f"rule={meta['rule']}, cache_floor={meta.get('conf')}, verifier={VERIFIER_LABELS.get(res['verifier'])}, "
             f"default conf={res['default_conf']})\n")
    if drones:
        L.append("| conf | bare P | bare R | bare F1 | filt P | filt R | filt F1 [95% CI] |")
        L.append("|---|---|---|---|---|---|---|")
        for r in res["rows"]:
            d = " **(default)**" if abs(r["conf"] - res["default_conf"]) < 1e-9 else ""
            ci = r["filt"]["f1_ci"]
            L.append(f"| {r['conf']}{d} | {r['bare']['precision']} | {r['bare']['recall']} | {r['bare']['f1']} "
                     f"| {r['filt']['precision']} | {r['filt']['recall']} | {r['filt']['f1']}"
                     + (f" [{ci[0]}–{ci[1]}]" if ci else "") + " |")
    else:
        L.append("| conf | bare FP | bare fire | filt FP | filt fire [95% CI] |")
        L.append("|---|---|---|---|---|")
        for r in res["rows"]:
            d = " **(default)**" if abs(r["conf"] - res["default_conf"]) < 1e-9 else ""
            ci = r["filt"]["fire_ci"]
            L.append(f"| {r['conf']}{d} | {r['bare']['FP']} | {r['bare']['fire_rate']} "
                     f"| {r['filt']['FP']} | {r['filt']['fire_rate']}"
                     + (f" [{ci[0]}–{ci[1]}]" if ci else "") + " |")
    L.append(f"\n<sub>verifier-thr grid ({'F1' if drones else 'FP'} @ conf × mlp_thr; GUI ships 0.15):</sub>\n")
    L.append("| conf \\ mlp_thr | " + " | ".join(str(m) for m in MLP_GRID) + " |")
    L.append("|---|" + "---|" * len(MLP_GRID))
    for t, g in res["mlp_grid"].items():
        L.append(f"| {t} | " + " | ".join(str(g[str(m)]) for m in MLP_GRID) + " |")


def md_paired(L, name, meta, res):
    L.append(f"\n## {name} — FULL PIPELINE (n={meta['n']} of {meta.get('n_source', meta['n'])}, "
             f"trust-aware per-modality scoring, cache_floor={meta.get('conf')}, "
             f"defaults rgb={res['rgb_default']}/ir={res['ir_default']})\n")
    for sl, title in (("rgb", f"RGB conf sweep (IR fixed @{res['ir_default']})"),
                      ("ir", f"IR conf sweep (RGB fixed @{res['rgb_default']})"),
                      ("diag", "Diagonal (both at t)")):
        rows = [r for r in res["rows"] if r["slice"] == sl]
        if not rows:
            continue
        L.append(f"\n**{title}**\n")
        L.append("| rgb_conf | ir_conf | bare F1 | +filter F1 | robust8 F1 | robust8→filter F1 [95% CI] |")
        L.append("|---|---|---|---|---|---|")
        for r in rows:
            d = " **(default)**" if (r["rgb_conf"] == res["rgb_default"] and r["ir_conf"] == res["ir_default"]) else ""
            cf = r.get("clf_filt", {})
            ci = cf.get("f1_ci")
            L.append(f"| {r['rgb_conf']}{d} | {r['ir_conf']} | {r['bare']['f1']} | {r['filt']['f1']} "
                     f"| {r.get('clf', {}).get('f1', '—')} | {cf.get('f1', '—')}"
                     + (f" [{ci[0]}–{ci[1]}]" if ci else "") + " |")


def main():
    global OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="thesis_eval/cache_conf005")
    ap.add_argument("--only", default="", help="comma list of surfaces")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=str(OUT_DIR), help="results dir (default = committed conf_sweep/)")
    args = ap.parse_args()
    OUT_DIR = Path(args.out) if Path(args.out).is_absolute() else (REPO / args.out)
    cdir = Path(args.cache_dir)
    if not cdir.is_absolute():
        cdir = REPO / cdir
    OUT_DIR.mkdir(parents=True, exist_ok=True); FIG_DIR.mkdir(parents=True, exist_ok=True)
    only = {s.strip() for s in args.only.split(",") if s.strip()}

    print(f"Conf-sweep replay <- {cdir} (zero-GPU)")
    clfs, verifs = load_classifiers(), load_verifiers(args.device)
    pkls = sorted(p for p in cdir.glob("*.pkl") if not only or p.stem in only)
    if not pkls:
        print("  no cache .pkl found — build the low-conf cache first (see module docstring)."); return

    results = {}
    L = ["# Detector-confidence sweep × MLP verifier — does recall+filter beat the defaults?",
         f"{time.strftime('%Y-%m-%d %H:%M')} | cache: {cdir.name} | defaults rgb=0.25 / ir=0.40 / gray=0.40 "
         f"(fusion_settings.json) | verifier thr rgb={RGB_THR_MLP} / ir={IR_THR_MLP} / gray={GRAY_THR_MLP} "
         f"| robust8 tau=0.20",
         "Points below a cache's conf floor are censored. Bootstrap CIs: frame resample, 1000 iters.\n"]

    for pkl in pkls:
        d = pickle.load(open(pkl, "rb")); meta, frames = d["meta"], d["frames"]
        name = meta["name"]; t0 = time.time()
        print(f"  [{name}] kind={meta['kind']} n={meta['n']} floor={meta.get('conf')}")
        if meta["kind"] in ("paired", "grayrgb_paired"):
            res = sweep_paired(meta, frames, clfs, verifs)
            results[name] = {"meta": meta["name"], "paired": res}
            md_paired(L, name, meta, res); fig_paired(name, res)
        else:
            res = sweep_solo(meta, frames, verifs)
            if not res:
                continue
            results[name] = {"meta": meta["name"], "solo": res}
            md_solo(L, name, meta, res); fig_solo(name, res, meta["has_drones"])
        print(f"    done in {time.time()-t0:.1f}s")

    L.append(f"\n---\nFigures: docs/analysis/images/conf_sweep/*.png")
    (OUT_DIR / "conf_sweep_results.md").write_text("\n".join(L), encoding="utf-8")
    json.dump(results, open(OUT_DIR / "conf_sweep_results.json", "w"), indent=2, default=float)
    print(f"\nDONE -> {OUT_DIR / 'conf_sweep_results.md'}  +  conf_sweep_results.json  +  {FIG_DIR}")


if __name__ == "__main__":
    main()
