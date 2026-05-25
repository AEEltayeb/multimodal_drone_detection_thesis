"""Patch a Ultralytics last.pt to allow resume past an early-stop.

Ultralytics sets ckpt['epoch'] = -1 when training "finishes" (incl. via early-stop),
which makes resume=True fail the start_epoch>0 assertion. This script rewrites
that field to the last completed epoch so resume picks up at the next one.

Usage:
    python "RGB model/unfinish_ckpt.py" \
        "RGB model/Yolo26n_selcom_mixed_ft3_960/weights/last.pt" \
        --last-completed 9
"""

import argparse
from pathlib import Path

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=str)
    ap.add_argument("--last-completed", type=int, required=True,
                    help="1-indexed epoch number that was last completed (per results.csv)")
    args = ap.parse_args()

    p = Path(args.ckpt)
    if not p.exists():
        raise SystemExit(f"checkpoint not found: {p}")

    ckpt = torch.load(str(p), weights_only=False, map_location="cpu")
    old_epoch = ckpt.get("epoch")
    # Ultralytics: start_epoch = ckpt["epoch"] + 1, so to resume at epoch N (1-indexed)
    # we need ckpt["epoch"] = N - 1 (since start_epoch increments to N).
    new_epoch = args.last_completed - 1
    ckpt["epoch"] = new_epoch

    # Clear stopper state if present so patience counts from here
    if "best_fitness" in ckpt and ckpt.get("best_fitness") is None:
        pass  # leave alone
    # Some Ultralytics versions stash a stopper dict; reset it if present
    for k in ("stopper", "early_stopping"):
        if k in ckpt:
            ckpt.pop(k, None)

    backup = p.with_suffix(p.suffix + ".prefix")
    if not backup.exists():
        backup.write_bytes(p.read_bytes())
        print(f"backup -> {backup}")

    torch.save(ckpt, str(p))
    print(f"patched {p}: epoch {old_epoch} -> {new_epoch}  (resume will start at epoch {new_epoch+1})")


if __name__ == "__main__":
    main()
