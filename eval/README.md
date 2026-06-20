# eval/

The evaluation library and standalone analyses used by the thesis. The main eval harness lives in
[`thesis_eval/`](../thesis_eval/README.md); this directory holds the shared pieces it imports plus a few
self-contained sweeps.

## Shared library (imported by the harness)

| Module | Role |
|---|---|
| `metrics.py` | Scoring: precision / recall / F1, IoU and IoP matching, the per-modality trust-aware rule |
| `eval_v4_vs_patch.py` | The MLP confuser-filter inference wrapper |
| `compare_routing_pipeline.py` | Trust-router feature vector and routing comparison |
| `datasets.py` | Dataset and ground-truth iteration helpers |
| `distill_v5_p3p5_ft4.py` | The P3 and P5 feature extractor behind the RGB confuser filter |
| `reporting.py` | Result table and CSV writers |

These are imported by name through `sys.path`; they are kept flat so that wiring stays intact.

## Standalone analyses (each writes a frozen JSON the audit checks)

| Script | Output |
|---|---|
| `svan_resolution_sweep.py` | `results/svan_resolution_sweep.json` (Svanstrom resolution sweep) |
| `filter_operating_sweep.py` | `results/filter_operating_sweep.json` (filter operating points) |
| `eval_ir_heldout.py` | `results/ir_heldout_results.json` (CBAM IR held-out filter) |
| `eval_birdtest_heldout.py` | RGB filter false-fire rate on a held-out bird test split |
| `run_manifest.py` | Evaluation run manifests |

## Note on `eval/results/`

`eval/results/` is excluded from the repo because it is large and regenerable, with one exception: the few
result files the thesis cites are force-tracked so the audit and the source-tracing table in the root
README resolve on a fresh clone.
