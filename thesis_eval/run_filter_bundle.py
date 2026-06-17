"""
run_filter_bundle.py — FILTER SWAP KIT (Phase 1).  Evaluate any confuser-filter pair through the
thesis evidence harness, zero-GPU, without touching committed results or weights.

Every filter-dependent replay in thesis_eval/ imports load_verifiers() from pipeline_eval_unified,
so a single set of THESIS_* env vars repoints the RGB + IR(+gray) filters everywhere at once. This
driver sets them, runs the decision-relevant replays into a TAGGED scratch dir, records a manifest
(weight SHA-256 + thresholds + git SHA + timing) for traceability, and (optionally) auto-diffs the
tagged run against a shipped baseline via diff_filters.py.

It does NOT copy weights into models/verifiers/ and does NOT write any committed results dir — that is
promotion/integration (Phase 3, user-gated). Defaults reproduce the SHIPPED stack exactly.

Examples
--------
# shipped baseline (no overrides) -> _filter_swap/shipped/
py -u thesis_eval/run_filter_bundle.py --tag shipped

# a candidate pair -> _filter_swap/v2/ , then diff vs shipped/
py -u thesis_eval/run_filter_bundle.py --tag v2 \
   --rgb eval/results/_v5_balanced_v2/classifiers/mlp_v5_balanced_v2.pt \
   --ir  mri/results/ir_aligned_balanced/classifiers/mlp_aligned.pt \
   --ir-gray mri/results/ir_aligned_balanced/classifiers/mlp_aligned_gray.pt \
   --ir-thr 0.01 --diff-against shipped
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SWAP_ROOT = REPO / "thesis_eval" / "_filter_swap"

# Shipped defaults (None => harness default = the committed weight / threshold).
SHIPPED = {"rgb": None, "ir": None, "ir_gray": None, "rgb_thr": "0.25", "ir_thr": "0.05", "gray_thr": "0.25"}

# Decision-relevant, --out-aware, zero-GPU replays. (filter_operating_sweep writes the COMMITTED figure
# dir with no --out, so it is an INTEGRATION-time step, not a candidate step — excluded here on purpose.)
JOBS = [
    ("pipeline", ["py", "-u", "thesis_eval/pipeline_eval_unified.py", "--out", "{out}", "{only}"]),
    ("temporal", ["py", "-u", "thesis_eval/temporal_replay.py", "--out", "{out}"]),
]


def sha256(p: Path) -> str:
    if not p or not p.exists():
        return "MISSING"
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()[:16]


def git_sha() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def resolve(p: str | None) -> Path | None:
    if not p:
        return None
    q = Path(p)
    return q if q.is_absolute() else (REPO / q)


def main():
    ap = argparse.ArgumentParser(description="Filter swap kit — evaluate a filter pair, zero-GPU.")
    ap.add_argument("--tag", required=True, help="bundle name -> thesis_eval/_filter_swap/<tag>/")
    ap.add_argument("--rgb", default=None, help="RGB filter weight (default: shipped mlp_v5.pt)")
    ap.add_argument("--ir", default=None, help="IR thermal filter weight (default: shipped mlp_aligned.pt)")
    ap.add_argument("--ir-gray", default=None, help="IR grayscale filter weight (default: shipped)")
    ap.add_argument("--rgb-thr", default=SHIPPED["rgb_thr"])
    ap.add_argument("--ir-thr", default=SHIPPED["ir_thr"])
    ap.add_argument("--gray-thr", default=SHIPPED["gray_thr"])
    ap.add_argument("--diff-against", default=None, help="tag of a prior bundle to diff against (e.g. shipped)")
    ap.add_argument("--only", default="", help="comma list of surfaces (smoke-test; forwarded to pipeline)")
    args = ap.parse_args()

    out = SWAP_ROOT / args.tag
    out.mkdir(parents=True, exist_ok=True)

    rgb, ir, ir_gray = resolve(args.rgb), resolve(args.ir), resolve(args.ir_gray)
    env = dict(os.environ)
    if rgb:     env["THESIS_MLP_V5"] = str(rgb)
    if ir:      env["THESIS_ALIGNED"] = str(ir)
    if ir_gray: env["THESIS_ALIGNED_GRAY"] = str(ir_gray)
    env["THESIS_RGB_THR_MLP"] = args.rgb_thr
    env["THESIS_IR_THR_MLP"] = args.ir_thr
    env["THESIS_GRAY_THR_MLP"] = args.gray_thr

    print(f"== filter bundle '{args.tag}' -> {out}")
    print(f"   RGB  = {rgb or '(shipped mlp_v5.pt)'}  @ {args.rgb_thr}")
    print(f"   IR   = {ir or '(shipped mlp_aligned.pt)'}  @ {args.ir_thr}")
    print(f"   GRAY = {ir_gray or '(shipped mlp_aligned_gray.pt)'}  @ {args.gray_thr}")

    job_log = []
    for name, tmpl in JOBS:
        cmd = [a.replace("{out}", str(out)) for a in tmpl]
        # resolve the {only} slot: pipeline gets "--only <list>" or nothing; others drop it
        if "{only}" in cmd:
            i = cmd.index("{only}")
            cmd[i:i + 1] = (["--only", args.only] if args.only else [])
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=str(REPO), env=env).returncode
        dt = round(time.time() - t0, 1)
        job_log.append({"job": name, "cmd": " ".join(cmd), "rc": rc, "seconds": dt})
        print(f"   [{name}] rc={rc} {dt}s")

    manifest = {
        "tag": args.tag,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_sha": git_sha(),
        "weights": {
            "rgb":     {"path": str(rgb) if rgb else "shipped", "sha256_16": sha256(rgb)},
            "ir":      {"path": str(ir) if ir else "shipped", "sha256_16": sha256(ir)},
            "ir_gray": {"path": str(ir_gray) if ir_gray else "shipped", "sha256_16": sha256(ir_gray)},
        },
        "thresholds": {"rgb": args.rgb_thr, "ir": args.ir_thr, "gray": args.gray_thr},
        "jobs": job_log,
    }
    (out / "swap_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"   manifest -> {out/'swap_manifest.json'}")

    if args.diff_against:
        base = SWAP_ROOT / args.diff_against
        if (base / "tier1_results.json").exists() and (out / "tier1_results.json").exists():
            print(f"\n== diff '{args.diff_against}' (baseline) vs '{args.tag}' (candidate)")
            subprocess.run(["py", str(REPO / "thesis_eval" / "results" / "_filter_ab" / "diff_filters.py"),
                            "--shipped", str(base / "tier1_results.json"),
                            "--candidate", str(out / "tier1_results.json")], cwd=str(REPO))
        else:
            print(f"   [diff skipped: need tier1_results.json in both {base} and {out}]")


if __name__ == "__main__":
    main()
