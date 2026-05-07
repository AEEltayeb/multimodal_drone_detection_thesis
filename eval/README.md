# eval/ — Unified Evaluation Tools

## Quick Start

```bash
# Pipeline evaluation (all configs, all datasets)
python eval/eval_pipeline.py --dataset both --plot

# Just YOLO detection accuracy
python eval/eval_pipeline.py --yolo-only --dataset antiuav --stride 3

# Just classifier
python eval/eval_pipeline.py --classifier-only --dataset antiuav

# Just filter
python eval/eval_pipeline.py --filter-only --dataset svanstrom

# YouTube OOD filter test
python eval/eval_pipeline.py --dataset youtube_ir --stride 3

# Full evaluation suite with plots
python eval/eval_pipeline.py --preset full

# Model benchmarking
python eval/eval_model.py --weights runs/best.pt --dataset G:/drone/test --plot
python eval/eval_model.py --weights a.pt b.pt --dataset G:/drone/test
python eval/eval_model.py --weights best.pt --dataset path --per-source --conf-sweep 0.1,0.2,0.3

# Cache YOLO detections (run once, evaluate many times)
python eval/cache_inference.py --dataset both --resume
```

## Files

| File | Purpose |
|---|---|
| `eval_pipeline.py` | Full pipeline eval (YOLO + classifier + filter) |
| `eval_model.py` | Raw YOLO model benchmarking |
| `cache_inference.py` | Pre-cache YOLO detections |
| `metrics.py` | Shared: IoU/IoP matching, TP/FP/FN, size buckets |
| `datasets.py` | Shared: dataset loaders |
| `reporting.py` | Shared: CSV/JSON output, console tables, plotting |
| `config.yaml` | Central config: dataset paths, model weights, defaults |
| `PURPOSE.md` | Design rationale and metric definitions |

## Config

Edit `config.yaml` to point to your dataset paths and model weights.
Legacy caches from `classifier/runs/` are auto-discoverable.

See `PURPOSE.md` for full documentation on metrics and output format.
