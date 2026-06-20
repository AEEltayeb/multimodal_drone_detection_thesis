# thesis_eval/

The evaluation harness that produces the thesis numbers. It works in two steps: detect once, then replay
many times with no detector and no GPU.

## How it works

1. **Build the cache (GPU, once).** `pipeline_cache_unified.py` runs the RGB detector (`ft4`) and the IR
   detector (`v3b`) over each evaluation surface and stores, per detection, its box, confidence, the
   verifier feature vector, and the patch score, plus the per-frame trust-router features and the
   per-modality ground truth. The cache lands in `thesis_eval/cache/` (excluded from the repo; it is
   regenerable and large).

   ```
   py thesis_eval/pipeline_cache_unified.py
   ```

2. **Replay (CPU, seconds).** `pipeline_eval_unified.py` reads the cache and replays the whole pipeline
   (router, filters, scoring) to produce the result tables, with bootstrap confidence intervals. No
   detector forward pass.

   ```
   py thesis_eval/pipeline_eval_unified.py                 # all surfaces
   py thesis_eval/pipeline_eval_unified.py --only svanstrom  # one surface
   ```

The frozen outputs of step 2 live in `thesis_eval/results/` and are what the [audit](../audit/README.md)
checks. The default model paths are the production stack, so a replay reproduces the committed numbers
without any environment overrides.

## Held-out evaluations

These score the individual models on data held out of their training, and back the per-model evidence in
the thesis:

| Script | Produces |
|---|---|
| `eval_router_heldout.py` | Trust-router 3-class confusion matrix on a grouped held-out split |
| `eval_filter_heldout_cm.py` | Confuser-filter confusion matrices (RGB and IR) on held-out drones and confusers |
| `leakage_controlled_replay.py` | The clean-split numbers (sequences with zero IR-training overlap) |

## Note on layout

The harness keeps its modules flat here and imports a shared evaluation library from `eval/` and
`classifier/` by way of `sys.path`. That wiring is deliberately left intact: it is dense and load-bearing,
so the publish reorganization documented it rather than rewiring it.
