"""
mri.report — assemble report.md from the diagnosis + stats + figures.

Follows the project doc convention: a TL;DR verdict first, evidence tables, then
a Delivered section listing absolute artifact paths.
"""
from __future__ import annotations

import json
from pathlib import Path


def _fmt(v, pct=False, nd=3):
    if v is None:
        return "—"
    if pct:
        return f"{v:.1%}"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def write_report(out_dir: Path, detector_path, specs, raws, sep_summary,
                 diag, figures, cv_results=None, mlp_path=None, examples=None):
    out_dir = Path(out_dir)
    rep = out_dir / "report.md"
    lines = []
    A = lines.append

    A(f"# Model MRI — {Path(detector_path).parent.parent.name}\n")
    A(f"**Detector:** `{detector_path}`  ")
    if specs:
        A(f"**Datasets:** {len(specs)} ({sum(s.role=='pos' for s in specs)} pos / "
          f"{sum(s.role=='neg' for s in specs)} neg)\n")
    else:  # resumed from a cached feature corpus — no image-folder specs
        A(f"**Corpus:** resumed feature corpus — {diag.get('n_drone', 0):,} drone / "
          f"{diag.get('n_confuser', 0):,} confuser detections (no image stream)\n")

    # ── Verdict ──────────────────────────────────────────────────────────
    A("## Verdict\n")
    A(f"> **{diag['verdict_text']}**\n")
    A(f"_Evidence: {diag['rationale']}_\n")

    # ── Diagnostic signals ───────────────────────────────────────────────
    A("## Diagnostic signals\n")
    A("| Signal | Value | Meaning |")
    A("|---|---|---|")
    _halluc = diag.get('raw_halluc_rate')
    _hsrc = diag.get('halluc_external')
    _hmean = (f"FP per confuser image (bare detector; {_hsrc})" if _hsrc
              else "FP per confuser image (bare detector)")
    A(f"| Raw hallucination rate | {(_fmt(_halluc, pct=True) if _halluc is not None else 'n/a (feature-only corpus)')} | {_hmean} |")
    if diag.get("raw_drone_prf"):
        p = diag["raw_drone_prf"]
        A(f"| Raw drone P/R/F1 | {p['precision']}/{p['recall']}/{p['f1']} | bare detector on positive sets |")
    A(f"| LDA separability | {_fmt(diag.get('lda_separability'))} | train-set linear split of drone vs confuser |")
    A(f"| Silhouette | {_fmt(diag.get('silhouette'))} | feature-space cluster separation |")
    A(f"| Max ANOVA F | {_fmt(sep_summary.get('max_anova_F'), nd=0)} | strongest single discriminative feature |")
    A(f"| Meta-only max AUROC | {_fmt(diag.get('meta_max_auroc'))} | best metadata feature alone |")
    A(f"| YOLO-feat max AUROC | {_fmt(diag.get('yolo_max_auroc'))} | best learned feature alone |")
    if diag.get("recall_cost") is not None:
        A(f"| Projected FP cut | {_fmt(diag.get('fp_reduction'), pct=True)} | confusers the classifier would reject |")
        A(f"| Recall retention | {_fmt(diag.get('classifier_recall_retention'), pct=True)} | true drones the classifier keeps |")
        _pfr = diag.get('projected_fp_rate')
        A(f"| Projected FP rate | {(_fmt(_pfr, pct=True) if _pfr is not None else 'n/a (needs image scan)')} | hallucination after classifier |")
    A("")

    # ── Per-dataset mining ───────────────────────────────────────────────
    A("## Per-dataset scan\n")
    A("| Dataset | Role | Images | Dets | Drones | Confusers | bare TP/FP/FN |")
    A("|---|---|---|---|---|---|---|")
    for r in raws:
        if not r.get("n_images"):
            continue
        A(f"| {r['name']} | {r.get('role','?')} | {r['n_images']} | {r.get('n_dets',0)} | "
          f"{r.get('mined_drones',0)} | {r.get('mined_confusers',0)} | "
          f"{r.get('tp',0)}/{r.get('fp',0)}/{r.get('fn',0)} |")
    A("")

    # ── Top features ─────────────────────────────────────────────────────
    A("## Top discriminative features (ANOVA F)\n")
    for lbl in sep_summary.get("top_feature_labels", [])[:10]:
        A(f"- {lbl}")
    A("")

    # ── Classifier bench ─────────────────────────────────────────────────
    if cv_results:
        A("## Classifier CV (F1)\n")
        A("| Classifier | feature set | CV F1 |")
        A("|---|---|---|")
        for name, (cv_f1, cv_std, feat) in sorted(
                cv_results.items(), key=lambda kv: -kv[1][0]):
            A(f"| {name} | {feat} | {cv_f1:.4f} ± {cv_std:.4f} |")
        A("")

    # ── Spatial activation examples ──────────────────────────────────────
    if examples:
        A("## Spatial activation examples\n")
        A("One detection per dataset, chosen for highest activation on the top "
          "discriminative neurons. Left: detection crop. Middle: P3 activation "
          "(stride 8, spatial detail). Right: P5 activation (stride 32, semantic depth).\n")
        for ex in examples:
            A(f"**{ex['role']} — {ex['spec']}** (conf={ex['conf']:.2f})")
            A(f"![{ex['role']} {ex['spec']}](images/{Path(ex['path']).name})")
            A("")

    # ── Figures ──────────────────────────────────────────────────────────
    A("## Figures\n")
    for f in figures:
        A(f"![{Path(f).stem}](images/{Path(f).name})")
    A("")

    # ── Delivered ────────────────────────────────────────────────────────
    A("## Delivered\n")
    A(f"- `{(out_dir / 'report.md').resolve()}` — this report")
    A(f"- `{(out_dir / 'features.npz').resolve()}` — extracted feature corpus (X, y, w)")
    A(f"- `{(out_dir / 'stats.json').resolve()}` — all numeric results")
    A(f"- `{(out_dir / 'manifest.json').resolve()}` — CLI args + git SHA")
    if mlp_path:
        A(f"- `{Path(mlp_path).resolve()}` — trained MLP classifier (callable)")
    for f in figures:
        A(f"- `{Path(f).resolve()}`")
    for ex in (examples or []):
        A(f"- `{Path(ex['path']).resolve()}` — {ex['role']} activation example ({ex['spec']})")

    rep.write_text("\n".join(lines), encoding="utf-8")
    return rep
