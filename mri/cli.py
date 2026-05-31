"""
mri.cli — the one entry point. Attach a YOLO model + positive/negative dataset
folders; it images the model's feature space, computes the brain statistics,
renders the plots, optionally trains the confuser MLP, and writes a verdict on
whether the model needs an FP-reduction classifier.

    python -m mri --yolo path/to/best.pt --pos DRONE_DIR --neg CONFUSER_DIR --train-mlp

Output auto-lands in mri/results/<detector>_<timestamp>/ unless --out is given.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
# eval/metrics.py is imported bare ("from metrics import ...") by scan/diagnose.
sys.path.insert(0, str(REPO / "eval"))

from .datasets import parse_dataset_arg, specs_from_config  # noqa: E402
from .extract import FeatureExtractor                        # noqa: E402
from . import scan, stats, plots, diagnose, report           # noqa: E402
from .classifier import (LogRegWrapper, RFWrapper, XGBWrapper, MLPWrapper,  # noqa: E402
                         cross_val_score_f1, save_mlp_artifact)


def _parse_grid(s: str) -> tuple[int, int]:
    h, _, w = s.lower().partition("x")
    return (int(h), int(w or h))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mri", description="Model MRI — detector feature-space diagnosis.")
    p.add_argument("--yolo", required=True, help="Path to YOLO .pt weights.")
    p.add_argument("--pos", nargs="*", default=[],
                   help="Positive (drone) dataset dirs. PATH or PATH:imgsz=..,rule=..,stride=..,max=..")
    p.add_argument("--neg", nargs="*", default=[],
                   help="Negative (confuser) dataset dirs (same inline-spec syntax).")
    p.add_argument("--config", help="YAML with pos:/neg: lists (alternative to --pos/--neg).")
    p.add_argument("--out", help="Output dir (default mri/results/<detector>_<ts>).")
    # Detector / matching
    p.add_argument("--imgsz", type=int, default=640, help="Global YOLO input size.")
    p.add_argument("--conf", type=float, default=0.25, help="Detector confidence threshold.")
    p.add_argument("--iou", type=float, default=0.5, help="IoU match threshold.")
    p.add_argument("--iop", type=float, default=0.5, help="IoP match threshold.")
    p.add_argument("--match-rule", default="iou", choices=["iou", "iop"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--grayscale-input", action="store_true",
                   help="Feed each image as gray-3ch (BGR->gray->3ch) before the "
                        "detector — the grayscale-fallback deployment op. Lets you "
                        "image/train the detector's grayscale-mode feature space.")
    # Feature schema
    p.add_argument("--layers", default="p3,p5", help="FPN maps to pool, e.g. p3,p5 or p3,p4,p5.")
    p.add_argument("--p3-grid", default="2x2"); p.add_argument("--p4-grid", default="1x1")
    p.add_argument("--p5-grid", default="1x1")
    # Sampling
    p.add_argument("--stride", type=int, default=1, help="Global frame stride.")
    p.add_argument("--max-per-source", type=int, default=0, help="Cap dets mined per dataset (0=all).")
    p.add_argument("--quick", action="store_true", help="Smoke preset: stride x5, max 200/source.")
    # Analyses / training
    p.add_argument("--stats", default="pca,lda,anova,heatmap,neurons",
                   help="Comma list: pca,lda,anova,heatmap,neurons.")
    p.add_argument("--examples", action="store_true", default=True,
                   help="Render one spatial activation panel per dataset (live scan only).")
    p.add_argument("--no-examples", dest="examples", action="store_false",
                   help="Skip the per-dataset activation example panels.")
    p.add_argument("--train", action="store_true", help="Train+CV full bench (logreg/rf/xgb/mlp).")
    p.add_argument("--train-mlp", action="store_true", help="Train only the production MLP; save .pt.")
    p.add_argument("--feature-set", default="fused", choices=["meta", "yolo", "fused"])
    p.add_argument("--epochs", type=int, default=120, help="MLP epochs.")
    p.add_argument("--resume", action="store_true", help="Reuse cached features.npz; skip extraction.")
    p.add_argument("--seed", type=int, default=42)
    # Verdict thresholds
    p.add_argument("--fp-rate-thr", type=float, default=0.05)
    p.add_argument("--sep-thr", type=float, default=0.90)
    p.add_argument("--recall-cost-thr", type=float, default=0.10)
    # Held-out deployment eval (honest gate vs the in-pool CV verdict)
    p.add_argument("--holdout-eval", metavar="MLP_WEIGHTS",
                   help="Evaluate this trained verifier (.pt) on the --pos/--neg "
                        "surfaces: per-surface bare vs MLP (+patch) P/R/F1. Runs "
                        "this mode and exits; does not extract/train.")
    p.add_argument("--patch", default="",
                   help="Optional patch-verifier .pt to compare against in --holdout-eval.")
    p.add_argument("--mlp-thr", type=float, default=0.15,
                   help="Keep detection if P(drone) >= this (holdout-eval).")
    p.add_argument("--patch-thr", type=float, default=0.5,
                   help="Keep detection if P(confuser) < this (holdout-eval).")
    return p


def _resolve_out(args) -> Path:
    if args.out:
        return Path(args.out)
    stem = Path(args.yolo).parent.parent.name or Path(args.yolo).stem
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return REPO / "mri" / "results" / f"{stem}_{ts}"


def _build_specs(args):
    g = (args.imgsz, args.stride, args.match_rule, args.max_per_source)
    if args.config:
        import yaml
        cfg = yaml.safe_load(Path(args.config).read_text())
        return specs_from_config(cfg, *g)
    specs = [parse_dataset_arg(a, "pos", *g) for a in args.pos]
    specs += [parse_dataset_arg(a, "neg", *g) for a in args.neg]
    return specs


def _feature_slice(X, schema, feature_set):
    meta_stop = schema.layer_slices()["meta"].stop
    if feature_set == "meta":
        return X[:, :meta_stop]
    if feature_set == "yolo":
        return X[:, meta_stop:]
    return X


def run(args):
    from ultralytics import YOLO

    out_dir = _resolve_out(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    print(f"== Model MRI ==\n  detector: {args.yolo}\n  out: {out_dir}")

    if args.quick:
        args.stride = max(args.stride, 1) * 5
        args.max_per_source = args.max_per_source or 200
        print("  quick mode: stride x5, max 200/source")

    # ── Held-out deployment eval mode (no extraction/training) ───────────
    if args.holdout_eval:
        from . import holdout
        specs = _build_specs(args)
        if not specs:
            print("FATAL: --holdout-eval needs --pos/--neg or --config surfaces.")
            return 1
        return holdout.run_holdout(
            args.yolo, specs, mlp_weights=args.holdout_eval,
            patch_weights=(args.patch or None), conf=args.conf,
            mlp_thr=args.mlp_thr, patch_thr=args.patch_thr,
            device=args.device, out_dir=out_dir,
            grayscale=args.grayscale_input)

    cache = out_dir / "features.npz"
    resume_mode = args.resume and cache.exists()

    specs = _build_specs(args)
    if not specs and not resume_mode:
        print("FATAL: no datasets. Pass --pos/--neg or --config (or --resume a cached run).")
        return 1

    grids = {"p3": _parse_grid(args.p3_grid), "p4": _parse_grid(args.p4_grid),
             "p5": _parse_grid(args.p5_grid)}
    layers = tuple(l.strip() for l in args.layers.split(",") if l.strip())

    if resume_mode:
        print(f"  resume: loading {cache}")
        z = np.load(cache, allow_pickle=True)
        X, y, w = z["X"], z["y"], z["w"]
        raws = json.loads(out_dir.joinpath("features_meta.json").read_text())["raws"]
        # Rebuild schema from saved meta.
        from .extract import FeatureSchema
        sm = json.loads(out_dir.joinpath("features_meta.json").read_text())["schema"]
        schema = FeatureSchema(layers=tuple(sm["layers"]),
                               grids={k: tuple(v) for k, v in sm["grids"].items()},
                               layer_dims=sm["layer_dims"])
        if not specs:
            from .datasets import DatasetSpec
            specs = [DatasetSpec(name=r.get("name", "?"), path=Path("."),
                                 role=r.get("role", "neg")) for r in raws]
        extractor, provenance = None, None
    else:
        model = YOLO(args.yolo)
        extractor = FeatureExtractor(model, layers=layers, grids=grids)
        X, y, w, raws, provenance = scan.collect(
            extractor, specs, conf_thr=args.conf, device=args.device,
            iou_thr=args.iou, iop_thr=args.iop, seed=args.seed,
            grayscale=args.grayscale_input)
        schema = extractor.schema
        np.savez_compressed(cache, X=X, y=y, w=w)
        out_dir.joinpath("features_meta.json").write_text(json.dumps({
            "schema": schema.to_dict(), "raws": raws,
            "n_total": int(len(X)), "n_drone": int((y == 1).sum()),
            "n_confuser": int((y == 0).sum()),
        }, indent=2))

    if len(X) == 0:
        print("FATAL: no features mined. Check dataset paths / labels.")
        return 1
    print(f"  mined {len(X)} detections ({int((y==1).sum())} drone / {int((y==0).sum())} confuser)")

    # ── Statistics ───────────────────────────────────────────────────────
    sep_summary, F, auroc, top = stats.separability_summary(X, y, schema, seed=args.seed)

    # ── Optional classifier training ─────────────────────────────────────
    cv_results, oof, mlp_path = None, None, None
    Xf = _feature_slice(X, schema, args.feature_set)
    if args.train or args.train_mlp:
        print("  training classifier(s)...")
        cv_results = {}
        mlp_kwargs = {"input_dim": Xf.shape[1], "device": args.device, "epochs": args.epochs}
        if args.train:
            for nm, cls, kw in [("logreg", LogRegWrapper, {}), ("rf", RFWrapper, {}),
                                 ("xgb", XGBWrapper, {})]:
                try:
                    f1, sd, _, _ = cross_val_score_f1(cls, kw, Xf, y, sample_weight=w, seed=args.seed)
                    cv_results[nm] = (f1, sd, args.feature_set)
                    print(f"    {nm}: CV F1 {f1:.4f} ± {sd:.4f}")
                except Exception as e:
                    print(f"    {nm}: skipped ({e})")
        f1, sd, best_mlp, oof = cross_val_score_f1(
            MLPWrapper, mlp_kwargs, Xf, y, sample_weight=w, seed=args.seed)
        cv_results["mlp"] = (f1, sd, args.feature_set)
        print(f"    mlp: CV F1 {f1:.4f} ± {sd:.4f}")
        if args.train_mlp or args.train:
            mlp_path = out_dir / "mlp.pt"
            save_mlp_artifact(best_mlp, mlp_path, schema, f1, sd, args.yolo)
            print(f"    saved {mlp_path}")

    # ── Diagnosis ────────────────────────────────────────────────────────
    diag = diagnose.diagnose(
        raws, sep_summary, oof=oof, y=y, threshold=0.5,
        fp_rate_thr=args.fp_rate_thr, sep_thr=args.sep_thr,
        recall_cost_thr=args.recall_cost_thr)
    print(f"\n  VERDICT: {diag['verdict_text']}\n  ({diag['rationale']})\n")

    # ── Plots ────────────────────────────────────────────────────────────
    want = [s.strip() for s in args.stats.split(",") if s.strip()]
    figures = plots.generate_all(X, y, schema, F, top, out_dir / "images",
                                 want=want, diag=diag)

    # ── Per-dataset spatial activation examples ──────────────────────────
    # Needs live feature maps, so only in a full --pos/--neg scan (not --resume).
    example_panels = []
    if args.examples and extractor is not None and provenance:
        from . import examples
        print("  generating per-dataset activation examples...")
        example_panels = examples.generate_examples(
            extractor, provenance, X, y, schema, top, specs,
            out_dir / "images", conf_thr=args.conf, device=args.device)
    elif args.examples and extractor is None:
        print("  (skipping activation examples — needs a live --pos/--neg scan, not --resume)")

    # ── Persist stats.json + manifest + report ───────────────────────────
    out_dir.joinpath("stats.json").write_text(json.dumps({
        "separability": sep_summary, "diagnosis": diag,
        "cv_results": cv_results,
    }, indent=2, default=str))
    out_dir.joinpath("manifest.json").write_text(json.dumps({
        "argv": vars(args), "git_sha": _git_sha(),
        "datasets": [{"name": s.name, "path": str(s.path), "role": s.role,
                      "imgsz": s.imgsz, "rule": s.match_rule} for s in specs],
    }, indent=2, default=str))
    rep = report.write_report(out_dir, args.yolo, specs, raws, sep_summary,
                              diag, figures, cv_results, mlp_path,
                              examples=example_panels)
    print(f"  report: {rep}")

    if extractor is not None:
        extractor.close()
    return 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
