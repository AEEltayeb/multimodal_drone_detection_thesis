"""run_email_recompute.py - overnight orchestrator for the email-table recompute.

Runs, sequentially (no GPU overlap, offline, resumable):
  1. Phase A   pipeline_cache_paired.py  --surface both   (GPU; self-gates; sharded resume)
  2. Phase B   pipeline_eval_paired.py                     (CPU; robust6 + V5 MLP; 7 configs)
  3. Domain 3  eval_youtube_ir_filter.py --mlp             (GPU; v3b + mlp_aligned)
  4. Domain 3  eval_youtube_ir_filter.py        (patch)    (GPU; OLD baseline, same 14 clips)

Each step logs to mri/results/email_recompute_<ts>/<step>.log. Step 1 self-GPU-gates
(waits for any running job to free the GPU), so it is safe to launch while another job
finishes. Steps 2-4 run after, so no OOM overlap.

  py -u eval/run_email_recompute.py
"""
from __future__ import annotations
import os, subprocess, sys, time
import datetime as dt
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
LOG = REPO / "mri" / "results" / f"email_recompute_{dt.datetime.now():%Y%m%d_%H%M%S}"
LOG.mkdir(parents=True, exist_ok=True)

STEPS = [
    ("1_phaseA_cache",   [PY, "-u", "eval/pipeline_cache_paired.py", "--surface", "both"]),
    ("2_phaseB_eval",    [PY, "-u", "eval/pipeline_eval_paired.py"]),
    # domain-3 (eval_youtube_ir_filter --mlp + patch) already valid from the v1 run
    # (it scores live in f32, no float16 caching) -> not re-run here.
]


def log(m):
    line = f"[{dt.datetime.now():%H:%M:%S}] {m}"
    print(line, flush=True)
    (LOG / "run.log").open("a", encoding="utf-8").write(line + "\n")


def main():
    log(f"== run_email_recompute ==  log={LOG}")
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}
    for name, args in STEPS:
        log(f"START {name}: {' '.join(args)}")
        t0 = time.time()
        with open(LOG / f"{name}.log", "w", encoding="utf-8") as lf:
            rc = subprocess.run(args, cwd=str(REPO), env=env,
                                stdout=lf, stderr=subprocess.STDOUT).returncode
        log(f"DONE {name}: rc={rc} ({(time.time()-t0)/60:.1f} min)")
        if rc != 0:
            log(f"  !! {name} returned {rc} — see {name}.log; continuing")
    log("== all steps done ==")
    log(f"Outputs: eval/results/_email_recompute/comparison_*.md  + "
        f"classifier/runs/eval_youtube_ir{{_mlp,}}/category_summary.csv")


if __name__ == "__main__":
    main()
