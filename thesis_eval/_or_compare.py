import json
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
J = json.load(open(REPO / "thesis_eval/results_noreject/tier1_results.json"))
ALL = ["robust6", "sa32", "robust8", "robust8_nr_drop", "robust8_nr_both"]
clfm = {k: [] for k in ALL}; filtm = {k: [] for k in ALL}; bare = []; orfilt = []; paired = []; conf = []
for name, r in J.items():
    meta = r["meta"]; part = r.get("B_pipeline") or r.get("S4_verifier")
    if part and meta.get("has_drones"):
        if part.get("bare"): bare.append(part["bare"]["f1"])
        if part.get("filt_mlp"): orfilt.append(part["filt_mlp"]["f1"])
        for k in ALL:
            if part.get(f"clf[{k}]"): clfm[k].append(part[f"clf[{k}]"]["f1"])
            if part.get(f"clf->filt[{k}]"): filtm[k].append(part[f"clf->filt[{k}]"]["f1"])
        if meta["kind"] in ("paired", "grayrgb_paired") and r.get("B_pipeline"):
            p = r["B_pipeline"]
            g = lambda c: f"{p[c]['f1']:.3f}" if p.get(c) else "-"
            paired.append((name, g("bare"), g("filt_mlp"), g("clf->filt[robust8]"), g("clf->filt[robust8_nr_drop]")))
    cc = r.get("C_confuser")
    if cc:
        fr = lambda c: f"{cc[c]['fire_rate']:.3f}" if cc.get(c) else "-"
        conf.append((name, fr("bare"), fr("filt_mlp"), fr("clf->filt[robust8]"), fr("clf->filt[robust8_nr_drop]")))
a = lambda l: f"{sum(l)/len(l):.3f}" if l else "-"
print(f"MEAN drone-surface F1 (n={len(bare)} surfaces):")
print(f"  OR (bare, always trust-both)         clf {a(bare)}")
print(f"  OR + filter (filt_mlp, no router)    ->filt {a(orfilt)}")
for k in ALL:
    print(f"  {k:22} clf {a(clfm[k])}   ->filt {a(filtm[k])}")
print("\nPAIRED drone F1:  surface | OR(bare) | OR+filt | robust8+filt | nr_drop+filt")
for row in paired: print("  " + " | ".join(row))
print("\nCONFUSER fire-rate (lower better): surface | OR(bare) | OR+filt | robust8+filt | nr_drop+filt")
for row in conf: print("  " + " | ".join(row))
