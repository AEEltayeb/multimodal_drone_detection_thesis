"""run_afk_pipeline.py — end-to-end AFK pipeline.

Steps (executed sequentially, each must succeed before the next):
  0. Wait for any heavy python.exe (>50% CPU sustained) other than us
  1. Reuse Lean-13 YOLO caches by copying to lean19/ output dir
  2. Generate Lean-19 dataset (19 features) with yt confusers included
  3. Train Lean-19 classifier
  4. Re-train Lean-13_yt and Lean-10_yt on the yt-augmented dataset
     (by selecting the 13/10-feature subset from the Lean-19 CSV)
  5. Run 5-way eval (lean10, lean13, lean19, 32feat, 40feat) on
     100-frames-per-source test set
  6. Write a results summary to docs/analysis/<date>_lean19_afk_results.md

All commands log to logs/afk_pipeline_<timestamp>.log.
Safe to re-run: each step skips if its primary output already exists.
"""
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import date, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent
LOG_DIR = REPO / "logs"
LOG_DIR.mkdir(exist_ok=True)
TS = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOG_DIR / f"afk_pipeline_{TS}.log"

RGB_WEIGHTS = "RGB model/Yolo26n_selcom_mixed_ft2_1280/weights/best.pt"
IR_WEIGHTS = "runs/corrective_finetune/finetune_v3b/weights/best.pt"
AUV_ROOT = "G:/drone/Anti-UAV-RGBT_yolo_converted/test"
SVAN_ROOT = "G:/drone/svanstrom_paired"

LEAN13_DIR = REPO / "classifier/fusion_models/lean13"
LEAN10_DIR = REPO / "classifier/fusion_models/lean10"
LEAN19_DIR = REPO / "classifier/fusion_models/lean19"
LEAN13_YT_DIR = REPO / "classifier/fusion_models/lean13_yt"
LEAN10_YT_DIR = REPO / "classifier/fusion_models/lean10_yt"

CLF_32 = REPO / "classifier/fusion_models/retrained_v2_32feat/model.joblib"
CLF_40 = REPO / "classifier/fusion_models/control_v3more_40feat/model.joblib"


# ---- Utility ----

def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run(cmd_args, step_name, cwd=None):
    # Force unbuffered python so progress lines flush in real time
    if cmd_args and "python" in str(cmd_args[0]).lower() and "-u" not in cmd_args:
        cmd_args = [cmd_args[0], "-u"] + list(cmd_args[1:])
    env = os.environ.copy(); env["PYTHONUNBUFFERED"] = "1"
    log(f"\n=== {step_name} ===")
    log(f"$ {' '.join(str(a) for a in cmd_args)}")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd_args, cwd=cwd or REPO, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", env=env,
        )
        last = []
        for line in proc.stdout:
            f.write(line); f.flush()
            last.append(line.rstrip())
            if len(last) > 200: last.pop(0)
        rc = proc.wait()
    if rc != 0:
        log(f"!!! step '{step_name}' FAILED (exit {rc}); last 30 lines:")
        for l in last[-30:]: log(f"   {l}")
        raise SystemExit(rc)
    log(f"OK {step_name}")
    return last


def wait_for_idle(threshold_cpu=60, idle_secs=20, poll_secs=15, timeout_min=120):
    """Wait until no foreign python.exe is sustained-CPU-busy."""
    log(f"\n=== Step 0: wait for other python processes (CPU < {threshold_cpu}% sustained {idle_secs}s) ===")
    my_pid = os.getpid()
    deadline = time.time() + timeout_min * 60
    idle_since = None
    while time.time() < deadline:
        try:
            out = subprocess.check_output(
                ["powershell", "-NoProfile", "-Command",
                 "Get-Process python -ErrorAction SilentlyContinue | "
                 "Select-Object Id,CPU | ConvertTo-Csv -NoTypeInformation"],
                text=True, encoding="utf-8")
        except Exception as e:
            log(f"  [warn] process probe failed: {e}; assuming idle"); return
        rows = [r for r in out.splitlines() if r and not r.startswith('"Id"')]
        busy = []
        for r in rows:
            m = re.match(r'"(\d+)","?([\d\.]+)"?', r)
            if not m: continue
            pid, cpu = int(m.group(1)), float(m.group(2))
            if pid == my_pid: continue
            busy.append((pid, cpu))
        # heuristic: if any foreign python has CPU > threshold, consider busy
        active = [b for b in busy if b[1] > threshold_cpu]
        if not active:
            if idle_since is None:
                idle_since = time.time()
            elif time.time() - idle_since >= idle_secs:
                log("  GPU/CPU idle — proceeding."); return
        else:
            idle_since = None
            log(f"  waiting on python: {active}")
        time.sleep(poll_secs)
    log("  [warn] timeout waiting; proceeding anyway")


# ---- Steps ----

def step1_reuse_caches():
    LEAN19_DIR.mkdir(parents=True, exist_ok=True)
    for fn in ("cache_antiuav.json", "cache_svanstrom.json"):
        src = LEAN13_DIR / fn
        dst = LEAN19_DIR / fn
        if dst.exists():
            log(f"  cache already present: {dst.name}"); continue
        if src.exists():
            shutil.copy2(src, dst)
            log(f"  copied {src.name} -> lean19/")
        else:
            log(f"  [warn] {src.name} not found in lean13/; YOLO will rerun")


def step2_generate_lean19():
    csv = LEAN19_DIR / "fusion_dataset_lean19.csv"
    if csv.exists():
        log(f"  dataset exists ({csv.stat().st_size:,} bytes); skipping"); return
    run([sys.executable, "classifier/generate_lean19_data.py",
         "--rgb-weights", RGB_WEIGHTS, "--ir-weights", IR_WEIGHTS,
         "--auv-root", AUV_ROOT, "--svan-root", SVAN_ROOT,
         "--auv-stride", "2", "--svan-stride", "2", "--neg-keep", "0.20",
         "--conf", "0.25",
         "--auv-imgsz", "640", "--svan-imgsz", "1280", "--ir-imgsz", "640",
         "--video-rgb-cache-tag", "selcom_1280_sz1280",
         "--include-yt", "--yt-stride", "3",
         "--output-dir", "classifier/fusion_models/lean19"],
        "Step 2: Generate Lean-19 dataset (+yt)")


def step3_train_lean19():
    if (LEAN19_DIR / "model.joblib").exists():
        log("  Lean-19 model exists; skipping"); return
    run([sys.executable, "classifier/train_lean19_classifier.py"],
        "Step 3: Train Lean-19")


def step4_train_yt_variants():
    """Train Lean-13_yt and Lean-10_yt on the yt-augmented dataset
    by selecting the 13/10 subset of the 19-feature CSV."""
    if (LEAN13_YT_DIR / "model.joblib").exists() and (LEAN10_YT_DIR / "model.joblib").exists():
        log("  Lean-13_yt and Lean-10_yt exist; skipping"); return

    # Write a small inline trainer that supports 13/10 subset selection.
    inline = REPO / "classifier" / "_inline_train_yt_subsets.py"
    inline.write_text("""
import json, re
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from xgboost import XGBClassifier

REPO = Path(__file__).resolve().parent.parent
CSV = REPO / "classifier/fusion_models/lean19/fusion_dataset_lean19.csv"
F13 = ["rgb_max_conf","ir_max_conf","rgb_best_log_bbox_area","ir_best_log_bbox_area",
       "rgb_best_aspect_ratio","ir_best_aspect_ratio","rgb_best_pos_y","ir_best_pos_y",
       "rgb_best_local_contrast","ir_best_local_contrast","rgb_img_mean","ir_img_mean","rgb_img_std"]
F10 = F13[:10]
SEQ_RE = re.compile(r"^(.+?)(?:_f\\d+|_frame\\d+|_\\d{4,})(?:_visible|_infrared)?$", re.I)

def seq_id(stem, source):
    m = SEQ_RE.match(str(stem)); base = m.group(1).rstrip("_") if m else str(stem)
    return f"{source}::{base}"

def train(df, feats, out, tag):
    df = df.copy()
    df["sequence_id"] = df.apply(lambda r: seq_id(r["stem"], r["source"]), axis=1)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(gss.split(df, df["trust_label"], groups=df["sequence_id"].values))
    Xtr, Xte = df.iloc[tr][feats].values, df.iloc[te][feats].values
    ytr, yte = df.iloc[tr]["trust_label"].values, df.iloc[te]["trust_label"].values
    src = df.iloc[te]["source"].values
    print(f"[{tag}] feats={len(feats)} train={len(Xtr):,} test={len(Xte):,}")
    m = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                       subsample=0.8, colsample_bytree=0.8,
                       objective="multi:softprob", num_class=4,
                       eval_metric="mlogloss", tree_method="hist",
                       random_state=42, n_jobs=-1)
    m.fit(Xtr, ytr, verbose=False)
    p = m.predict(Xte)
    acc = float(accuracy_score(yte, p)); f1m = float(f1_score(yte, p, average="macro", zero_division=0))
    print(f"[{tag}] acc={acc:.4f} f1m={f1m:.4f}")
    per = {}
    for d in np.unique(src):
        sm = src == d
        per[d] = {"n": int(sm.sum()), "acc": float(accuracy_score(yte[sm], p[sm])),
                  "f1_macro": float(f1_score(yte[sm], p[sm], average="macro", zero_division=0))}
    imp = dict(zip(feats, m.feature_importances_.tolist()))
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": m, "features": feats}, out/"model.joblib")
    json.dump({"tag": tag, "n_features": len(feats), "features": feats,
               "accuracy": acc, "f1_macro": f1m, "per_dataset": per,
               "feature_importance": imp}, open(out/"metrics.json", "w"), indent=2)
    print(f"  saved -> {out}")

df = pd.read_csv(CSV)
print(f"Loaded {len(df):,} rows from {CSV.name}")
train(df, F13, REPO/"classifier/fusion_models/lean13_yt", "lean13_yt")
train(df, F10, REPO/"classifier/fusion_models/lean10_yt", "lean10_yt")
""", encoding="utf-8")
    run([sys.executable, str(inline)], "Step 4: Train Lean-13_yt and Lean-10_yt subsets")


def step5_eval_all():
    md = REPO / "docs/analysis" / f"{date.today().isoformat()}_classifier_3way_eval.md"
    csv = REPO / "docs/analysis/full_pipeline_ablations/csv/classifier_3way.csv"
    run([sys.executable, "eval/eval_classifier_3way.py",
         "--n-per-source", "100",
         "--rgb-weights", RGB_WEIGHTS, "--ir-weights", IR_WEIGHTS,
         "--auv-root", AUV_ROOT, "--svan-root", SVAN_ROOT,
         "--auv-imgsz", "640", "--svan-imgsz", "1280", "--ir-imgsz", "640",
         "--conf", "0.25",
         "--video-rgb-cache-tag", "selcom_1280_sz1280",
         "--clf-10", "classifier/fusion_models/lean10/model.joblib",
         "--clf-13", "classifier/fusion_models/lean13/model.joblib",
         "--clf-19", "classifier/fusion_models/lean19/model.joblib",
         "--clf-32", str(CLF_32),
         "--clf-40", str(CLF_40),
         "--auv-cache",  "classifier/fusion_models/lean19/cache_antiuav.json",
         "--svan-cache", "classifier/fusion_models/lean19/cache_svanstrom.json"],
        "Step 5: 5-way eval (lean10, lean13, lean19, 32feat, 40feat)")
    log(f"  MD: {md}")
    log(f"  CSV: {csv}")


def step6_summary():
    """Write a one-page summary of the AFK run for the human."""
    import json
    out = REPO / "docs/analysis" / f"{date.today().isoformat()}_lean19_afk_summary.md"
    summary = [f"# AFK pipeline summary - {date.today().isoformat()}\n"]
    summary.append("## Models trained\n")
    for name, d in [("Lean-10 (no yt)", LEAN10_DIR), ("Lean-13 (no yt)", LEAN13_DIR),
                    ("Lean-19 (with yt)", LEAN19_DIR),
                    ("Lean-13_yt", LEAN13_YT_DIR), ("Lean-10_yt", LEAN10_YT_DIR)]:
        mj = d / "metrics.json"
        if mj.exists():
            j = json.load(open(mj))
            summary.append(f"- **{name}**: n_features={j['n_features']}, "
                          f"acc={j['accuracy']:.4f}, F1m={j['f1_macro']:.4f}, "
                          f"feature_importance top-3: " +
                          ", ".join(f"`{k}`={v:.3f}"
                                    for k, v in sorted(j['feature_importance'].items(),
                                                       key=lambda x: x[1], reverse=True)[:3]))
        else:
            summary.append(f"- **{name}**: NOT TRAINED")
    summary.append("\n## Per-source breakdown (from each model's own held-out split)\n")
    summary.append("See each `metrics.json` `per_dataset` field.\n")
    summary.append("\n## Headline 5-way eval (300-frame unified test set)\n")
    eval_md = REPO / "docs/analysis" / f"{date.today().isoformat()}_classifier_3way_eval.md"
    if eval_md.exists():
        summary.append(f"Full report: `{eval_md.relative_to(REPO).as_posix()}`\n")
    summary.append(f"\n## Log\n\n`{LOG_FILE.relative_to(REPO).as_posix()}`\n")
    out.write_text("\n".join(summary), encoding="utf-8")
    log(f"  summary -> {out}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-wait", action="store_true",
                    help="Skip step 0 (wait for idle python processes)")
    args = ap.parse_args()
    log(f"AFK pipeline starting; log -> {LOG_FILE}")
    try:
        if not args.no_wait:
            wait_for_idle()
        else:
            log("Step 0 skipped via --no-wait")
        step1_reuse_caches()
        step2_generate_lean19()
        step3_train_lean19()
        step4_train_yt_variants()
        step5_eval_all()
        step6_summary()
        log("\n=== ALL STEPS COMPLETE ===")
    except SystemExit as e:
        log(f"\n=== PIPELINE ABORTED (exit {e.code}) ===")
        sys.exit(e.code)
    except Exception as e:
        log(f"\n=== PIPELINE CRASHED: {e!r} ===")
        raise


if __name__ == "__main__":
    main()
