#!/usr/bin/env python
"""
eval.py - the thesis evaluation front-end.

Prints the numbers behind every dataset and table in the dissertation. By default it reads the frozen
result JSONs (the same source the audit checks), so it needs only the Python standard library and runs on
a fresh clone with no cache and no GPU. Pass --replay to recompute from the detect-once cache instead.

Examples
  py eval.py --list                     # every dataset and the tables it has
  py eval.py --all                      # print everything
  py eval.py svanstrom                  # all tables for one dataset
  py eval.py --rgb rgb_test             # RGB detector, bare, on the RGB test set
  py eval.py --ir  antiuav              # IR detector on Anti-UAV
  py eval.py --pipeline svanstrom       # full pipeline ablation table
  py eval.py --confuser rgb_confuser    # confuser false-alarm table
  py eval.py --dut                      # DUT Anti-UAV held-out test split
  py eval.py --clean-split              # leakage-controlled clean split
  py eval.py --resolution               # Svanstrom resolution sweep
  py eval.py --router-cm                # trust-router held-out confusion matrix
  py eval.py --filter-cm                # confuser-filter held-out confusion matrices
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
R = REPO / "thesis_eval" / "results"
RNR = REPO / "thesis_eval" / "results_noreject"


def load(p):
    p = Path(p)
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


# ---- friendly dataset name -> surface key in tier1_results.json ----
ALIASES = {
    "rgb_test": "rgb_dataset_test", "rgbtest": "rgb_dataset_test",
    "ir_test": "ir_dset_final", "irtest": "ir_dset_final", "ir": "ir_dset_final",
    "selcom": "selcom_val", "svan": "svanstrom", "svanstrom_visible": "svanstrom_rawrgb",
}
def canon(name): return ALIASES.get(name, name)

DETECTOR_ROWS = {"rgb": "ft4/rgb", "ir": "v3b/ir"}


def fmt(cell, keys=("precision", "recall", "f1")):
    out = []
    for k in keys:
        v = cell.get(k)
        if v is not None:
            out.append(f"{k[0].upper()}={v:.4f}")
    for c in ("TP", "FP", "FN"):
        if cell.get(c) is not None:
            out.append(f"{c}={cell[c]}")
    if cell.get("f1_ci"):
        lo, hi = cell["f1_ci"]; out.append(f"F1_CI=[{lo:.3f},{hi:.3f}]")
    return "  ".join(out)


def meta_line(meta):
    if not meta:
        return ""
    bits = []
    if meta.get("n") is not None:
        bits.append(f"frames n={meta['n']}")
    if meta.get("n_gt") is not None:
        bits.append(f"GT drones={meta['n_gt']}")
    if meta.get("rule"):
        bits.append(f"match={meta['rule']}")
    if meta.get("clips") is not None:
        bits.append(f"clips={meta['clips']}")
    return "   (" + ", ".join(bits) + ")" if bits else ""


def print_section(title, section, kind="prf"):
    print(f"\n  {title}")
    if not section:
        print("    (no data)"); return
    for row, cell in section.items():
        if not isinstance(cell, dict):
            continue
        if kind == "confuser":
            fr = cell.get("fire_rate"); fp = cell.get("FP")
            ci = cell.get("fire_ci")
            extra = f"  fire_CI=[{ci[0]:.3f},{ci[1]:.3f}]" if ci else ""
            print(f"    {row:<28} fire_rate={fr:.4f}  FP={fp}{extra}")
        else:
            print(f"    {row:<28} {fmt(cell)}")


# ---- per-dataset tables from tier1_results.json ----
def show_surface(surf, only=None):
    T = load(R / "tier1_results.json")
    if T is None or surf not in T:
        print(f"  unknown dataset '{surf}' (try --list)"); return
    s = T[surf]
    print(f"\n=== {surf} ==={meta_line(s.get('meta'))}")
    secmap = [("A_bare", "Detectors (bare, per modality vs own GT)", "prf"),
              ("B_pipeline", "Full pipeline ablation (P / R / F1)", "prf"),
              ("S4_verifier", "Verifier-only ablation (P / R / F1)", "prf"),
              ("C_confuser", "Confuser false-alarm reduction", "confuser"),
              ("GRAY_SWEEP", "Grayscale P(drone) sweep", "prf")]
    for key, title, kind in secmap:
        if key in s and (only is None or only == key):
            print_section(title, s[key], kind)


def show_detector(surf, modality):
    T = load(R / "tier1_results.json")
    surf = canon(surf)
    if T is None or surf not in T or "A_bare" not in T[surf]:
        print(f"  no detector table for '{surf}'"); return
    row = DETECTOR_ROWS[modality]
    cell = T[surf]["A_bare"].get(row)
    print(f"\n=== {surf}: {modality.upper()} detector ({row}) ==={meta_line(T[surf].get('meta'))}")
    if cell:
        print(f"    {fmt(cell)}")
    else:
        print(f"    (no {row} row; available: {list(T[surf]['A_bare'])})")


# ---- standalone tables (their own JSONs) ----
def show_temporal(which=None):
    V = load(R / "temporal_results.json")
    if V is None:
        print("  temporal_results.json missing"); return
    for surf in ([which] if which else V):
        if surf not in V:
            continue
        print(f"\n=== temporal: {surf} ==={meta_line(V[surf].get('meta'))}")
        for row, cell in V[surf].items():
            if row == "meta" or not isinstance(cell, dict):
                continue
            win = cell.get("window"); fr = cell.get("window_fire")
            if isinstance(win, list) and len(win) >= 3:
                print(f"    {row:<28} window P/R/F1 = {win[0]:.3f}/{win[1]:.3f}/{win[2]:.3f}")
            elif fr is not None:
                print(f"    {row:<28} window_fire={fr:.4f}")


def show_simple_json(path, title):
    d = load(path)
    print(f"\n=== {title} ===")
    if d is None:
        print(f"    {path} missing"); return
    print(json.dumps(d, indent=2)[:4000])


def show_dut():
    T = load(REPO / "thesis_eval/results_dut/tier1_results.json")
    print("\n=== DUT Anti-UAV (held-out test split) ===")
    if T is None:
        print("    results_dut/tier1_results.json missing"); return
    for surf in T:
        s = T[surf]
        print(f"  [{surf}]{meta_line(s.get('meta'))}")
        if "B_pipeline" in s:
            print_section("Full pipeline ablation", s["B_pipeline"])


def show_router_cm():
    d = load(R / "per_model_heldout/router_heldout.json")
    print("\n=== Trust router robust8-nr: held-out confusion matrix ===")
    print(json.dumps(d, indent=2)[:3000] if d else "    missing")


def show_filter_cm():
    d = load(R / "per_model_heldout/filter_heldout_cm.json")
    print("\n=== Confuser filters: held-out confusion matrices ===")
    print(json.dumps(d, indent=2)[:3000] if d else "    missing")


def list_all():
    T = load(R / "tier1_results.json") or {}
    print("Datasets in tier1_results.json (use the name as a positional arg):")
    for surf in T:
        secs = [k for k in T[surf] if k != "meta"]
        print(f"  {surf:<22} -> {', '.join(secs)}")
    print("\nStandalone tables (flags):")
    print("  --dut            DUT Anti-UAV held-out test split")
    print("  --clean-split    leakage-controlled clean split")
    print("  --resolution     Svanstrom resolution sweep")
    print("  --filter-op      confuser-filter operating points")
    print("  --cbam           CBAM IR held-out filter")
    print("  --temporal       segment / video window metrics")
    print("  --router-cm      trust-router held-out confusion matrix")
    print("  --filter-cm      confuser-filter held-out confusion matrices")
    print("\nModality shortcuts:  --rgb <dataset>   --ir <dataset>   --pipeline <dataset>"
          "   --confuser <dataset>   --verifier <dataset>")


def show_all():
    T = load(R / "tier1_results.json") or {}
    for surf in T:
        show_surface(surf)
    show_temporal()
    show_dut()
    show_simple_json(REPO / "runs/clean_split/clean_split_results.json", "Clean split (leakage-controlled)")
    show_simple_json(REPO / "eval/results/svan_resolution_sweep.json", "Svanstrom resolution sweep")
    show_simple_json(REPO / "eval/results/filter_operating_sweep.json", "Filter operating points")
    show_simple_json(REPO / "eval/results/ir_heldout_results.json", "CBAM IR held-out filter")
    show_router_cm()
    show_filter_cm()


def replay(surface):
    """Recompute from the cache via the existing harness (needs the cache + requirements.txt)."""
    import subprocess
    cmd = [sys.executable, str(REPO / "thesis_eval/pipeline_eval_unified.py")]
    if surface:
        cmd += ["--only", canon(surface)]
    print(f"replaying: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=REPO)


def main():
    ap = argparse.ArgumentParser(description="Thesis evaluation front-end (reads frozen result JSONs by default).")
    ap.add_argument("dataset", nargs="?", help="dataset / surface name (see --list)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--rgb", metavar="DATASET", help="RGB detector row for DATASET")
    ap.add_argument("--ir", metavar="DATASET", help="IR detector row for DATASET")
    ap.add_argument("--pipeline", metavar="DATASET", help="pipeline ablation table for DATASET")
    ap.add_argument("--confuser", metavar="DATASET", help="confuser false-alarm table for DATASET")
    ap.add_argument("--verifier", metavar="DATASET", help="verifier-only table for DATASET")
    ap.add_argument("--temporal", nargs="?", const="", metavar="DATASET", help="segment / video window metrics")
    ap.add_argument("--dut", action="store_true")
    ap.add_argument("--clean-split", dest="clean_split", action="store_true")
    ap.add_argument("--resolution", action="store_true")
    ap.add_argument("--filter-op", dest="filter_op", action="store_true")
    ap.add_argument("--cbam", action="store_true")
    ap.add_argument("--router-cm", dest="router_cm", action="store_true")
    ap.add_argument("--filter-cm", dest="filter_cm", action="store_true")
    ap.add_argument("--replay", action="store_true", help="recompute from the cache instead of reading JSON")
    a = ap.parse_args()

    if a.replay:
        replay(a.pipeline or a.dataset)
        return
    if a.list:
        list_all(); return
    if a.all:
        show_all(); return
    if a.rgb:
        show_detector(a.rgb, "rgb"); return
    if a.ir:
        show_detector(a.ir, "ir"); return
    if a.pipeline:
        show_surface(canon(a.pipeline), only="B_pipeline"); return
    if a.confuser:
        show_surface(canon(a.confuser), only="C_confuser"); return
    if a.verifier:
        show_surface(canon(a.verifier), only="S4_verifier"); return
    if a.temporal is not None:
        show_temporal(a.temporal or None); return
    if a.dut:
        show_dut(); return
    if a.clean_split:
        show_simple_json(REPO / "runs/clean_split/clean_split_results.json", "Clean split (leakage-controlled)"); return
    if a.resolution:
        show_simple_json(REPO / "eval/results/svan_resolution_sweep.json", "Svanstrom resolution sweep"); return
    if a.filter_op:
        show_simple_json(REPO / "eval/results/filter_operating_sweep.json", "Filter operating points"); return
    if a.cbam:
        show_simple_json(REPO / "eval/results/ir_heldout_results.json", "CBAM IR held-out filter"); return
    if a.router_cm:
        show_router_cm(); return
    if a.filter_cm:
        show_filter_cm(); return
    if a.dataset:
        show_surface(canon(a.dataset)); return
    ap.print_help()


if __name__ == "__main__":
    main()
