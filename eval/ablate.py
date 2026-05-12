"""ablate.py — Sequential ablation driver.

Reads eval/ablations.yaml, expands factor×level×dataset into concrete eval
commands, gates each run through eval/dryrun.py, then runs the real eval.
On completion, aggregates every cell's metrics CSV into:
  eval/results/_ablation/<timestamp>/master.csv
  eval/results/_ablation/<timestamp>/master.md   (one comparison table per factor)

Usage:
  python eval/ablate.py --matrix eval/ablations.yaml
  python eval/ablate.py --matrix eval/ablations.yaml --factors C_classifier
  python eval/ablate.py --matrix eval/ablations.yaml --skip-dryrun
  python eval/ablate.py --matrix eval/ablations.yaml --aggregate-only \
        --run-dir eval/results/_ablation/2026-05-10T12-00-00
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

EVAL_DIR = Path(__file__).resolve().parent
REPO = EVAL_DIR.parent


def _green(s): return f"\033[32m{s}\033[0m"
def _red(s):   return f"\033[31m{s}\033[0m"
def _yellow(s): return f"\033[33m{s}\033[0m"


def _abs(p: str | Path) -> str:
    """Resolve path relative to repo root."""
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((REPO / pp).resolve())


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if v is not None:
            out[k] = v
    return out


def _build_cmd(
    script: str,
    dataset_name: str,
    dataset_cfg: dict,
    stack: dict,
    output_dir: Path,
) -> list[str]:
    """Build the shell command for one cell."""
    out_str = str(output_dir)

    if script == "eval_pipeline.py":
        cmd = [
            "python", "eval/eval_pipeline.py",
            "--dataset", dataset_name,
            "--stride", str(stack.get("stride", dataset_cfg.get("stride", 1))),
            "--imgsz", str(stack.get("imgsz", 640)),
            "--rgb-conf", str(stack.get("rgb_conf", 0.25)),
            "--ir-conf", str(stack.get("ir_conf", 0.40)),
            "--patch-thr", str(stack.get("patch_thr", 0.70)),
            "--scoring", stack.get("scoring", "trust_aware"),
            "--patch-rgb-weights", _abs(stack["patch_rgb_weights"]),
            "--patch-ir-weights", _abs(stack["patch_ir_weights"]),
            "--classifier-path", _abs(stack["classifier_path"]),
            "--output-dir", out_str,
        ]
        return cmd

    if script == "eval_model.py":
        weights_key = dataset_cfg.get("weights_key", "ir_weights")
        cmd = [
            "python", "eval/eval_model.py",
            "--weights", _abs(stack[weights_key]),
            "--dataset", _abs(REPO / dataset_cfg["root"] / dataset_cfg.get("images", "images")),
            "--stride", str(stack.get("stride", dataset_cfg.get("stride", 1))),
            "--conf", str(stack.get("ir_conf" if weights_key == "ir_weights" else "rgb_conf", 0.25)),
            "--imgsz", str(stack.get("imgsz", 640)),
            "--output-dir", out_str,
        ]
        for extra in dataset_cfg.get("extra_args", []):
            cmd.append(extra)
        return cmd

    if script == "eval_video_temporal.py":
        # stack must include 'video' and 'frame_range'
        cmd = [
            "python", "eval/eval_video_temporal.py",
            "--video", _abs(stack["video"]),
            "--mode", stack.get("mode", "grayscale"),
            "--temporal", stack.get("temporal", "off"),
            "--use-roi-fallback", stack.get("use_roi_fallback", "off"),
            "--cascade", stack.get("cascade", "none"),
            "--imgsz", str(stack.get("imgsz", 640)),
            "--conf", str(stack.get("ir_conf", 0.40)),
            "--output-dir", out_str,
        ]
        if stack.get("frame_range"):
            cmd += ["--frame-range", stack["frame_range"]]
        return cmd

    raise ValueError(f"Unknown script: {script}")


def _run_dryrun(cmd: list[str]) -> bool:
    """Wrap cmd through eval/dryrun.py. True if dry-run passed."""
    full = ["python", "eval/dryrun.py"] + cmd
    print(_yellow(f"[ablate] dryrun: {' '.join(shlex.quote(t) for t in full)}"))
    proc = subprocess.run(full, cwd=str(REPO))
    return proc.returncode == 0


def _run_real(cmd: list[str]) -> bool:
    print(_yellow(f"[ablate] run:    {' '.join(shlex.quote(t) for t in cmd)}"))
    proc = subprocess.run(cmd, cwd=str(REPO))
    return proc.returncode == 0


# ── Cell expansion ───────────────────────────────────────────────────────

def _expand_cells(matrix: dict, only_factors: list[str] | None,
                  only_datasets: list[str] | None = None) -> list[dict]:
    """Yield concrete cells: {factor, level_name, dataset, stack, script}.

    only_datasets: if given, restrict to this subset of dataset names.
    """
    default = matrix["default_stack"]
    datasets = matrix["datasets"]
    cells = []

    for fkey, fdef in matrix["factors"].items():
        if only_factors and fkey not in only_factors:
            continue
        for level in fdef["levels"]:
            level_name = level["name"]
            stack = _merge(default, {k: v for k, v in level.items()
                                     if k not in ("name", "requires_temporal")})
            for ds_name in fdef.get("datasets", []):
                if only_datasets and ds_name not in only_datasets:
                    continue
                ds_cfg = datasets[ds_name]
                script = ds_cfg["script"]

                # Special routing: alert_gate_only requires temporal+video,
                # which the paired pipeline doesn't support. Skip with reason.
                if level.get("requires_temporal") and script == "eval_pipeline.py":
                    cells.append({
                        "factor": fkey, "level": level_name, "dataset": ds_name,
                        "stack": stack, "script": script,
                        "skip_reason": "alert_gate_only requires temporal+video; "
                                       "pipeline-mode datasets do not support it. "
                                       "Run via diagnostic clips instead."
                    })
                    continue

                cells.append({
                    "factor": fkey, "level": level_name, "dataset": ds_name,
                    "stack": _merge(stack, ds_cfg),
                    "script": script,
                    "skip_reason": None,
                })

    return cells


# ── Aggregation ──────────────────────────────────────────────────────────

def _scrape_metrics(cell_dir: Path) -> list[dict]:
    """Read every metrics_iop.csv / *_metrics.csv under cell_dir."""
    rows: list[dict] = []
    for csv_path in cell_dir.rglob("metrics_iop.csv"):
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                r["_source"] = str(csv_path.relative_to(cell_dir))
                rows.append(r)
    # eval_model.py writes <model>_metrics.csv
    for csv_path in cell_dir.rglob("*_metrics.csv"):
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                r["_source"] = str(csv_path.relative_to(cell_dir))
                rows.append(r)
    return rows


def _aggregate(run_dir: Path):
    master_rows = []
    for cell_meta_path in run_dir.rglob("cell.json"):
        meta = json.loads(cell_meta_path.read_text())
        cell_dir = cell_meta_path.parent
        for row in _scrape_metrics(cell_dir):
            master_rows.append({
                "factor": meta["factor"],
                "level": meta["level"],
                "dataset": meta["dataset"],
                "config": row.get("config", ""),
                "TP": row.get("TP", ""),
                "FP": row.get("FP", ""),
                "FN": row.get("FN", ""),
                "precision": row.get("precision", ""),
                "recall": row.get("recall", ""),
                "f1": row.get("f1", ""),
                "manifest": str((cell_dir / "manifest.json").relative_to(run_dir))
                    if (cell_dir / "manifest.json").exists() else "",
            })
    if not master_rows:
        print(_yellow("[ablate] no cells produced metrics — nothing to aggregate"))
        return

    csv_path = run_dir / "master.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(master_rows[0].keys()))
        w.writeheader()
        w.writerows(master_rows)
    print(_green(f"[ablate] master.csv: {csv_path}"))

    # Markdown — one section per factor
    md = [f"# Ablation summary  ({run_dir.name})\n"]
    by_factor: dict[str, list] = {}
    for r in master_rows:
        by_factor.setdefault(r["factor"], []).append(r)
    for factor, rows in sorted(by_factor.items()):
        md.append(f"\n## {factor}\n")
        md.append("| level | dataset | config | TP | FP | FN | P | R | F1 |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        for r in rows:
            md.append(f"| {r['level']} | {r['dataset']} | {r['config']} | "
                      f"{r['TP']} | {r['FP']} | {r['FN']} | "
                      f"{r['precision']} | {r['recall']} | {r['f1']} |")
    md_path = run_dir / "master.md"
    md_path.write_text("\n".join(md))
    print(_green(f"[ablate] master.md:  {md_path}"))


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default="eval/ablations.yaml")
    ap.add_argument("--factors", nargs="*", default=None,
                    help="Subset of factor keys to run (default: all)")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Subset of dataset names to run (default: all)")
    ap.add_argument("--skip-dryrun", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true",
                    help="Skip running cells; just aggregate an existing run dir")
    ap.add_argument("--run-dir", default="",
                    help="When --aggregate-only, path to existing run dir")
    args = ap.parse_args()

    matrix = yaml.safe_load(Path(args.matrix).read_text())

    if args.aggregate_only:
        if not args.run_dir:
            print(_red("--aggregate-only requires --run-dir"))
            sys.exit(2)
        _aggregate(Path(args.run_dir))
        return

    ts = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = REPO / matrix["driver"]["output_root"] / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    print(_green(f"[ablate] run dir: {run_dir}"))

    cells = _expand_cells(matrix, args.factors, args.datasets)
    print(_green(f"[ablate] expanded {len(cells)} cells"))

    n_ok = n_skip = n_fail = 0
    t0 = time.time()
    for i, cell in enumerate(cells, 1):
        cell_dir = run_dir / cell["factor"] / cell["level"] / cell["dataset"]
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "cell.json").write_text(json.dumps({
            "factor": cell["factor"], "level": cell["level"],
            "dataset": cell["dataset"], "script": cell["script"],
            "stack": {k: str(v) for k, v in cell["stack"].items()},
        }, indent=2, default=str))

        header = f"[{i}/{len(cells)}] {cell['factor']}.{cell['level']} on {cell['dataset']}"
        print()
        print(_green("=" * 80))
        print(_green(header))
        print(_green("=" * 80))

        if cell.get("skip_reason"):
            print(_yellow(f"[SKIP] {cell['skip_reason']}"))
            (cell_dir / "SKIPPED.txt").write_text(cell["skip_reason"])
            n_skip += 1
            continue

        ds_cfg = matrix["datasets"][cell["dataset"]]
        try:
            cmd = _build_cmd(
                cell["script"], cell["dataset"], ds_cfg,
                cell["stack"], cell_dir,
            )
        except Exception as e:
            print(_red(f"[ERROR] cmd build: {e}"))
            (cell_dir / "ERROR.txt").write_text(str(e))
            n_fail += 1
            continue

        if not args.skip_dryrun:
            if not _run_dryrun(cmd):
                print(_red("[FAIL] dryrun failed"))
                (cell_dir / "DRYRUN_FAIL.txt").write_text("dryrun returned non-zero")
                n_fail += 1
                if matrix["driver"].get("stop_on_error"):
                    break
                continue

        ok = _run_real(cmd)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            (cell_dir / "RUN_FAIL.txt").write_text("real run returned non-zero")
            if matrix["driver"].get("stop_on_error"):
                break

    elapsed = time.time() - t0
    print()
    print(_green(f"[ablate] done: {n_ok} ok / {n_skip} skipped / {n_fail} failed "
                 f"in {elapsed/60:.1f} min"))
    _aggregate(run_dir)


if __name__ == "__main__":
    main()
