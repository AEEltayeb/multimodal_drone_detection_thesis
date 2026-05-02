"""
cross_eval_all.py — Run every IR model on every IR dataset test split.

Builds a generalization matrix showing how each model performs on data
it wasn't trained on, plus FPPI on Svanström negatives.

Usage:
    python scripts/cross_eval_all.py --dry-run     # preview commands
    python scripts/cross_eval_all.py               # run all
    python scripts/cross_eval_all.py --only M3,M5  # run subset of models
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# ── Project root (relative to this script) ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ══════════════════════════════════════════════════════════════
# MODELS — each with weights path (relative to project root) and T*
# ══════════════════════════════════════════════════════════════
MODELS = {
    "M1_GoldV2": {
        "label": "GoldV2",
        "weights": "models/IR_dsetV1_goldV2_300ep/best.pt",
        "threshold": 0.52,
        "config": "runs/IR_FT_goldV2_IRdsetV1_aug0_s0_pilot/config.yaml",
    },
    "M2_dsetV3": {
        "label": "dsetV3",
        "weights": "runs/IR_FT_dsetV3_aug0_s0/weights/best.pt",
        "threshold": 0.30,
        "config": "runs/IR_FT_dsetV3_aug0_s0/config.yaml",
    },
    "M3_dsetV4": {
        "label": "dsetV4",
        "weights": "models/IR_dsetV4_300ep/best.pt",
        "threshold": 0.17,
        "config": "configs/ir_finetune_pub.yaml",
    },
    "M4_dsetV5": {
        "label": "dsetV5",
        "weights": "models/IR_dsetV5_269ep/best.pt",
        "threshold": 0.42,
        "config": "configs/ir_finetune_pub_v5.yaml",
    },
    "M5_dsetV6": {
        "label": "dsetV6",
        "weights": "models/IR_dsetV6_118ep/best.pt",
        "threshold": 0.33,
        "config": "configs/ir_finetune_pub_v6.yaml",
    },
    "M6_Final": {
        "label": "Final",
        "weights": "models/IR_final_cleaned/weights/best.pt",
        "threshold": 0.40,
        "config": "configs/ir_final_cleaned_eval.yaml",
    },
}

# ══════════════════════════════════════════════════════════════
# DATASETS — test splits to evaluate on
# ══════════════════════════════════════════════════════════════
DATASETS = {
    "D1_GoldV2": {
        "label": "GoldV2",
        "yaml": "datasets/IR_dsetV1_gold_v2/IR_dsetV1_gold.yaml",
        "split": "test",
    },
    "D2_dsetV3": {
        "label": "dsetV3",
        "yaml": "datasets/IR_dsetV3/dataset.yaml",
        "split": "test",
    },
    "D3_dsetV4": {
        "label": "dsetV4",
        "yaml": "datasets/IR_dsetV4/dataset.yaml",
        "split": "test",
    },
    "D4_dsetV5": {
        "label": "dsetV5",
        "yaml": "G:/drone/IR_dsetV5/dataset.yaml",
        "split": "test",
    },
    "D5_dsetV6": {
        "label": "dsetV6",
        "yaml": "G:/drone/IR_dsetV6/dataset.yaml",
        "split": "test",
    },
    "D_Final": {
        "label": "Final",
        "yaml": "G:/drone/IR_dset_final/dataset.yaml",
        "split": "test",
    },
    "NEG_svanstrom": {
        "label": "Svanström",
        "yaml": "G:/drone/IR_video_ir_dataset/dataset.yaml",
        "split": "test",
    },
}

# Which (model, dataset) pairs are "native" (already evaluated)
NATIVE_PAIRS = {
    ("M1_GoldV2", "D1_GoldV2"),
    ("M2_dsetV3", "D2_dsetV3"),
    ("M3_dsetV4", "D3_dsetV4"),
    ("M4_dsetV5", "D4_dsetV5"),
    ("M5_dsetV6", "D5_dsetV6"),
    ("M6_Final", "D_Final"),
}


def build_run_name(model_key: str, dataset_key: str) -> str:
    """Generate a unique run name for a cross-eval pair."""
    m_label = MODELS[model_key]["label"]
    d_label = DATASETS[dataset_key]["label"]
    return f"CROSSEVAL_{m_label}_on_{d_label}"


def run_eval(model_key: str, dataset_key: str, dry_run: bool = False) -> dict | None:
    """Run eval.py for one (model, dataset) pair."""
    model = MODELS[model_key]
    dataset = DATASETS[dataset_key]
    run_name = build_run_name(model_key, dataset_key)

    cmd = [
        sys.executable, "scripts/eval.py",
        "--config", str(model["config"]),
        "--split", dataset["split"],
        "--weights", str(model["weights"]),
        "--threshold", str(model["threshold"]),
        "--dataset-yaml", str(dataset["yaml"]),
    ]

    # We need to override the run_name so output goes to a unique directory.
    # eval.py uses cfg["run_name"] from the config to determine output dir.
    # We'll create a tiny temp config that overrides just the run_name.
    out_dir = PROJECT_ROOT / "runs" / run_name

    print(f"\n{'─'*60}")
    print(f"  {model['label']} → {dataset['label']}")
    print(f"  Run:     {run_name}")
    print(f"  Weights: {model['weights']}")
    print(f"  Dataset: {dataset['yaml']} ({dataset['split']})")
    print(f"  T*:      {model['threshold']}")
    print(f"  Output:  {out_dir}")
    print(f"{'─'*60}")

    if dry_run:
        print(f"  [DRY RUN] Would execute: {' '.join(cmd)}")
        return None

    # Create a temp config that inherits from the model's config but overrides run_name
    import yaml
    with open(PROJECT_ROOT / model["config"], "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["run_name"] = run_name
    cfg["run_grade"] = "EXP"

    temp_config = out_dir / "config.yaml"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(temp_config, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    # Update command to use the temp config
    cmd[cmd.index("--config") + 1] = str(temp_config)

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=False)

    if result.returncode != 0:
        print(f"  [ERROR] eval.py exited with code {result.returncode}")
        return None

    # Load the resulting metrics
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


def collect_matrix(skip_native: bool = True) -> dict:
    """Collect all cross-eval results into a matrix from existing run dirs."""
    matrix = {}
    for model_key in MODELS:
        matrix[model_key] = {}
        for dataset_key in DATASETS:
            if skip_native and (model_key, dataset_key) in NATIVE_PAIRS:
                # Load from native run instead
                native_run = MODELS[model_key]["config"].replace("configs/", "runs/").replace(".yaml", "")
                continue  # We'll fill native results separately

            run_name = build_run_name(model_key, dataset_key)
            metrics_path = PROJECT_ROOT / "runs" / run_name / "metrics.json"

            if metrics_path.exists():
                with open(metrics_path) as f:
                    data = json.load(f)
                matrix[model_key][dataset_key] = data.get("test", data.get("dev", {}))
            else:
                matrix[model_key][dataset_key] = None

    return matrix


def print_matrix(matrix: dict):
    """Pretty-print the results matrix."""
    # Header
    d_labels = [DATASETS[dk]["label"] for dk in DATASETS]
    header = f"{'Model':<12}" + "".join(f"{dl:>12}" for dl in d_labels)
    print(f"\n{'='*len(header)}")
    print("  CROSS-EVALUATION MATRIX — F1 Score")
    print(f"{'='*len(header)}")
    print(header)
    print("─" * len(header))

    for model_key in MODELS:
        row = f"{MODELS[model_key]['label']:<12}"
        for dataset_key in DATASETS:
            result = matrix[model_key].get(dataset_key)
            if result and "f1" in result:
                f1 = result["f1"]
                row += f"{f1:>12.4f}"
            else:
                row += f"{'—':>12}"
        print(row)


def main():
    parser = argparse.ArgumentParser(description="Run all cross-evaluations")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview commands without executing")
    parser.add_argument("--only", default=None,
                        help="Comma-separated list of model keys to run (e.g., M1_GoldV2,M3_dsetV4)")
    parser.add_argument("--skip-native", action="store_true", default=True,
                        help="Skip native (model trained on this dataset) evaluations")
    parser.add_argument("--include-native", action="store_true",
                        help="Include native evaluations too")
    parser.add_argument("--collect-only", action="store_true",
                        help="Just collect and display existing results")
    args = parser.parse_args()

    if args.include_native:
        args.skip_native = False

    if args.collect_only:
        matrix = collect_matrix(skip_native=False)
        print_matrix(matrix)
        # Save matrix
        out_path = PROJECT_ROOT / "runs" / "cross_eval_matrix.json"
        save_data = {}
        for mk in matrix:
            save_data[MODELS[mk]["label"]] = {}
            for dk in matrix[mk]:
                if matrix[mk][dk]:
                    save_data[MODELS[mk]["label"]][DATASETS[dk]["label"]] = matrix[mk][dk]
        with open(out_path, "w") as f:
            json.dump(save_data, f, indent=2)
        print(f"\nMatrix saved to: {out_path}")
        return

    # Determine which models to run
    model_keys = list(MODELS.keys())
    if args.only:
        model_keys = [k.strip() for k in args.only.split(",")]
        for k in model_keys:
            if k not in MODELS:
                print(f"[ERROR] Unknown model key: {k}")
                print(f"  Available: {list(MODELS.keys())}")
                sys.exit(1)

    # Count total evals
    total = 0
    pairs = []
    for mk in model_keys:
        for dk in DATASETS:
            if args.skip_native and (mk, dk) in NATIVE_PAIRS:
                continue
            pairs.append((mk, dk))
            total += 1

    print(f"\n{'='*60}")
    print(f"  CROSS-EVALUATION: {total} evaluations to run")
    print(f"  Models:   {[MODELS[k]['label'] for k in model_keys]}")
    print(f"  Datasets: {[DATASETS[k]['label'] for k in DATASETS]}")
    print(f"{'='*60}")

    # Run all evaluations
    results = {}
    for i, (mk, dk) in enumerate(pairs, 1):
        print(f"\n[{i}/{total}]", end="")
        result = run_eval(mk, dk, dry_run=args.dry_run)
        if result:
            results[(mk, dk)] = result

    # Print summary
    if not args.dry_run:
        print(f"\n\n{'='*60}")
        print(f"  COMPLETED: {len(results)}/{total} evaluations")
        print(f"{'='*60}")

        matrix = collect_matrix(skip_native=False)
        print_matrix(matrix)


if __name__ == "__main__":
    main()
