#!/usr/bin/env python3
"""apply_thesis_audit_claims.py — records the `thesis audit` agent-pass verdicts into
knowledge/claims.csv. Each claim from the thesis (via thesis_deliverables.md §4) judged
against the CURRENT (corrected) ledger. Idempotent: skips existing ids.

This is the agent-reasoning half of the audit (the engine did numbers/figures). The
high-value output is the contradicted/unverified claims + suggested rewordings.
"""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import kb  # noqa: E402


def C(id, chapter, claim, verdict, evidence="", conf="high", reword="", notes=""):
    return dict(id=id, chapter=chapter, tex_location="thesis_working.tex", claim_text=claim,
                kind="qualitative" if not any(c.isdigit() for c in claim) else "metric",
                verdict=verdict, evidence=evidence, confidence=conf,
                suggested_rewording=reword, notes=notes)

CLAIMS = [
    C("clm-multistage-98", "Ch7", "Multi-stage pipeline suppresses 98.4% of confuser FPs", "supported",
      "cascade-confuser-collapse;clfzoo_fnfn", notes="ledger says 98.3% (16149->271); 98.4 within rounding"),
    C("clm-baseline-best", "Ch4", "Baseline RGB is the best drone detector (R=0.959)", "supported",
      "rgb_svan_baseline", notes="§3.1/§7 R=0.959-0.961"),
    C("clm-retrainedv2-306", "Ch4", "retrained_v2 collapses drone recall to 0.306", "supported",
      "rgb_svan_retrainedv2"),
    C("clm-imgsz-1280", "Ch3", "imgsz=1280 is required (recall 0.07->0.959)", "partial", "",
      conf="med", reword="cite the grayscale-gap / Svanstrom-resolution evidence explicitly",
      notes="0.07->0.959 baseline-imgsz figure not yet an eval row in KB; supported by memory/§3.1 but add the row"),
    C("clm-scoring-28pp", "Ch6", "Scoring rule (dual vs trust_aware) causes a 28pp F1 swing", "unverified", "",
      conf="low", reword="VERIFY before citing", notes="E_scoring ablation NOT migrated to evals.csv; add the row or soften"),
    C("clm-patch-v2-gt-v4", "Ch4", "Patch verifier v2 > v4 on every metric", "contradicted",
      "patch_v2_svan640;patch_v4_svan640", reword="v4 (0.9331) ~= v2 (0.9311) on Svanstrom F1 - within noise; v2 retained as production (v3 was over-aggressive). Do NOT claim v2 beats v4 on every metric.",
      notes="ledger patch-version-ranking: v4 marginally HIGHER F1 than v2"),
    C("clm-ir-less-halluc", "Ch5", "IR hallucinates less than RGB on confusers (22% vs 53%)", "supported", "",
      notes="§3.3 confuser other: baseline 53% vs IR-gray 22.2%"),
    C("clm-ir-detected-49", "Ch3", "Trust classifier top feature is ir_detected (49%)", "unverified", "",
      conf="low", reword="VERIFY", notes="feature-importance not in KB; check fusion_no_fn_metrics.json"),
    C("clm-rgb-birds-94", "Ch1", "RGB struggles with birds (94% hallucination rate)", "supported", "",
      notes="§3.3 BIRD 94.4%"),
    C("clm-ir-gray-2.4x", "Ch5", "IR on grayscale hallucinates 2.4x less than RGB", "supported", "",
      notes="53/22.2 = 2.4; §3.3"),
    C("clm-ir-v3v4-map", "Ch4", "IR model improved v3->v4 (0.900->0.955) via human review", "contradicted",
      "ir_final_v3;ir_final_v4;ir-version-progression",
      reword="Use test-split F1 0.611->0.765 (P 0.648->0.895) via FP review. The 0.900->0.958 figure is mAP on per-version val splits - explicitly NON-comparable per ledger §4.1; do not cite it.",
      notes="thesis uses the discredited mAP narrative - load-bearing catch"),
    C("clm-ir-gray-generalizes", "Ch5", "IR generalises to grayscale despite never training on it", "supported",
      "ir-grayscale-fallback", conf="med",
      reword="frame as 'matches baseline RGB at ~1/3 confuser fire rate' (NOT 'beats') per corrected §9.4",
      notes="conditional finding; the 'beats every RGB' phrasing is stale (§9.3 superseded)"),
    C("clm-latency-realtime", "Ch8", "Latency is acceptable for real-time deployment", "unsupported",
      "latency-edge-unmeasured", reword="Only verifier-stage latency measured (V5 1.3-2.1ms/det, +1-4% per-frame). Full end-to-end edge latency is UNMEASURED (§10 placeholders) - state as future work or run it.",
      notes="do not claim real-time without the e2e measurement"),
]


def main():
    cur = kb._read("claims"); existing = {r["id"] for r in cur}; added = 0
    for row in CLAIMS:
        if row["id"] in existing:
            continue
        errs = kb._validate_row("claims", row)
        if errs:
            print(f"SKIP {row['id']}: {'; '.join(errs)}"); continue
        cur.append(row); existing.add(row["id"]); added += 1
    kb._write("claims", cur)
    print(f"claims: +{added} ({len(cur)} total)")
    # summary by verdict
    from collections import Counter
    c = Counter(r["verdict"] for r in cur)
    print("verdicts:", dict(c))
    kb._regen_views()


if __name__ == "__main__":
    main()
