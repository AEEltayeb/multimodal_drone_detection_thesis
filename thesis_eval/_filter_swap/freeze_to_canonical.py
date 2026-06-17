"""freeze_to_canonical.py — promote the v4/thermal-only staging results (_filter_swap/final) into the
canonical thesis evidence dirs, SURGICALLY: for each canonical file, replace only the surfaces it already
has with final's version (same n/stride — only the filter changed), preserving each dir's structure.
Reversible via git. Run from repo root.  py thesis_eval/_filter_swap/freeze_to_canonical.py
"""
import json, shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FINAL = REPO / "thesis_eval/_filter_swap/final"


def surgical(final_file: Path, canon: Path):
    if not final_file.exists():
        print(f"  SKIP (no final): {canon.relative_to(REPO)}"); return
    fin = json.load(open(final_file))
    if canon.exists():
        can = json.load(open(canon))
        updated = [k for k in can if k in fin]
        out = {k: (fin[k] if k in fin else can[k]) for k in can}
    else:                                   # canon missing -> create from final wholesale
        out = fin; updated = list(fin)
        canon.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(canon, "w"), indent=2, default=float)
    print(f"  froze {len(updated):>2} surfaces -> {canon.relative_to(REPO)}")


JOBS = [
    (FINAL / "tier1_results.json",            REPO / "thesis_eval/results/tier1_results.json"),
    (FINAL / "tier1_results.json",            REPO / "thesis_eval/results_noreject/tier1_results.json"),
    (FINAL / "tier1_results.json",            REPO / "thesis_eval/results_clean/tier1_results.json"),
    (FINAL / "tier1_results.json",            REPO / "runs/results_dut/tier1_results.json"),
    (FINAL / "temporal_results.json",         REPO / "thesis_eval/results/temporal_results.json"),
    (FINAL / "temporal_results.json",         REPO / "thesis_eval/results_noreject/temporal_results.json"),
    (FINAL / "notes_round1_results.json",     REPO / "thesis_eval/results/notes_round1_results.json"),
    (FINAL / "notes_round1_results.json",     REPO / "thesis_eval/results_noreject/notes_round1_results.json"),
    (FINAL / "conf_sweep/conf_sweep_results.json", REPO / "thesis_eval/results/conf_sweep/conf_sweep_results.json"),
    (FINAL / "conf_sweep/conf_sweep_results.json", REPO / "runs/conf_sweep/conf_sweep_results.json"),
    (FINAL / "filter_operating_sweep.json",   REPO / "eval/results/filter_operating_sweep.json"),
]

# committed figure (the only binary artifact)
FIG_COPIES = [
    (FINAL / "fig_filter_operating.pdf", REPO / "docs/thesis_working_distilling_overleaf/figures/fig_filter_operating.pdf"),
    (FINAL / "fig_filter_operating.png", REPO / "docs/thesis_working_distilling_overleaf/figures/fig_filter_operating.png"),
]


def main():
    print("FREEZE _filter_swap/final -> canonical evidence dirs (surgical; git-reversible)")
    for f, c in JOBS:
        surgical(f, c)
    for f, c in FIG_COPIES:
        if f.exists():
            c.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(f, c)
            print(f"  froze figure -> {c.relative_to(REPO)}")
    print("DONE. NOTE: runs/clean_split/clean_split_results.json (v3b-solo detector evidence) is "
          "filter-independent and intentionally NOT frozen. Audit will be RED until the .tex cells + "
          "_audit_headline_numbers.py CLAIMED constants are updated to v4 (the edit pass).")


if __name__ == "__main__":
    main()
