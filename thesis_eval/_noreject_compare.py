"""
thesis_eval/_noreject_compare.py — build the classifier-only comparison doc for the no-reject routers.
Reads thesis_eval/results_noreject/tier1_results.json (robust8 vs robust8_nr_drop vs robust8_nr_both,
all surfaces) and emits docs/analysis/2026-06-14_robust8_noreject.md. Zero-GPU, read-only on results.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
J = json.load(open(REPO / "thesis_eval/results_noreject/tier1_results.json"))
OUT = REPO / "docs/analysis/2026-06-14_robust8_noreject.md"
CLFS = ["robust8", "robust8_nr_drop", "robust8_nr_both"]
HDR = ["robust8", "nr_drop", "nr_both"]


def prf(c):
    return f"{c['precision']:.3f}/{c['recall']:.3f}/{c['f1']:.3f}" if c else "—"


def f1(c):
    return f"{c['f1']:.3f}" if c else "—"


def fr(c):
    return f"{c['fire_rate']:.3f}" if c else "—"


drone, dronef, conf, conff = [], [], [], []
mean = {c: [] for c in CLFS}; mean_f = {c: [] for c in CLFS}
for name, r in J.items():
    meta = r["meta"]
    part = r.get("B_pipeline") or r.get("S4_verifier")
    if part and meta.get("has_drones"):
        bare = part.get("bare")
        drone.append([name, meta["kind"], f1(bare)] + [prf(part.get(f"clf[{c}]")) for c in CLFS])
        dronef.append([name, meta["kind"]] + [prf(part.get(f"clf->filt[{c}]")) for c in CLFS])
        for c in CLFS:
            if part.get(f"clf[{c}]"): mean[c].append(part[f"clf[{c}]"]["f1"])
            if part.get(f"clf->filt[{c}]"): mean_f[c].append(part[f"clf->filt[{c}]"]["f1"])
    cc = r.get("C_confuser")
    if cc:
        conf.append([name, fr(cc.get("bare"))] + [fr(cc.get(f"clf[{c}]")) for c in CLFS])
        conff.append([name] + [fr(cc.get(f"clf->filt[{c}]")) for c in CLFS])


def tbl(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out)


def avg(d, c):
    return f"{sum(d[c])/len(d[c]):.3f}" if d[c] else "—"


L = ["# robust8 no-reject (3-class) router — classifier-only comparison",
     "",
     "_2026-06-14. Compares the shipped 4-class **robust8** {reject/rgb/ir/both} against two **no-reject** "
     "3-class variants that must route to rgb/ir/both (reject removed; the verifier does FP rejection):_ "
     "**nr_drop** (reject rows dropped in training) and **nr_both** (reject→both). All three are f8 routers "
     "trained on the same `fusion_dataset_full56.csv` / hyperparams / seq-split; evaluated zero-GPU on every "
     "unified cache. P/R/F1 are per-modality trust-aware (NO union). NOT in the thesis.",
     "",
     "## Held-out training (per-class F1, from the trainer)",
     "| variant | macro-F1 | trust_rgb | trust_ir | both |",
     "|---|---|---|---|---|",
     "| robust8 (4-class, ref) | — | — | — | — |",
     "| nr_drop | 0.941 | 0.854 | 0.980 | 0.990 |",
     "| nr_both | 0.885 | 0.718 | 0.951 | 0.985 |",
     "",
     "## Drone surfaces — classifier ONLY (no verifier)  ·  P/R/F1",
     "_bare = trust-both, no routing. The no-reject routers keep recall high but, lacking reject, do not "
     "suppress confuser FPs at this stage (precision ≈ bare on confuser-rich surfaces)._",
     "",
     tbl(drone, ["surface", "kind", "bare F1", *HDR]),
     "",
     f"_mean clf F1 over drone surfaces — robust8 {avg(mean,'robust8')} · nr_drop {avg(mean,'robust8_nr_drop')} "
     f"· nr_both {avg(mean,'robust8_nr_both')}._",
     "",
     "## Drone surfaces — classifier → verifier (clf→filt)  ·  P/R/F1",
     "_the verifier (mlp_v5 / aligned) now does the FP rejection the no-reject router skipped._",
     "",
     tbl(dronef, ["surface", "kind", *HDR]),
     "",
     f"_mean clf→filt F1 — robust8 {avg(mean_f,'robust8')} · nr_drop {avg(mean_f,'robust8_nr_drop')} "
     f"· nr_both {avg(mean_f,'robust8_nr_both')}._",
     "",
     "## Confuser surfaces — fire rate (lower = better)",
     "**clf only:**", "", tbl(conf, ["surface", "bare", *HDR]), "",
     "**clf→filt (with verifier):**", "", tbl(conff, ["surface", *HDR]), "",
     "## Delivered",
     "- Trainer: `classifier/train_robust8_noreject.py` → `models/routers/robust8_noreject_{drop,both}/model.joblib`",
     "- Results: `thesis_eval/results_noreject/tier1_screening_results.md` + `tier1_results.json`",
     "- Harness: `thesis_eval/pipeline_eval_unified.py` (load_classifiers + batch_labels label_map)",
     "- This doc: `docs/analysis/2026-06-14_robust8_noreject.md`  (NOT in the thesis)",
     ]
OUT.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {OUT}  ({len(drone)} drone surfaces, {len(conf)} confuser surfaces)")
print(f"mean clf F1     robust8 {avg(mean,'robust8')} nr_drop {avg(mean,'robust8_nr_drop')} nr_both {avg(mean,'robust8_nr_both')}")
print(f"mean clf>filt F1 robust8 {avg(mean_f,'robust8')} nr_drop {avg(mean_f,'robust8_nr_drop')} nr_both {avg(mean_f,'robust8_nr_both')}")
