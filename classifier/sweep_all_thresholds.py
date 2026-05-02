"""
sweep_all_thresholds.py — Universal threshold & settings sweep.

Phase 0 (infer):     Fresh YOLO + filter + classifier on 1/10th consecutive frames.
Phase A (per-frame): Sweep conf/filter thresholds on Phase 0 cache.
Phase B (temporal):  Simulate N-of-M temporal logic on Phase 0 cache.
Phase C (youtube):   Sweep conf/filter on YouTube confuser videos (GPU).
plots:               Generate thesis-quality plots from CSVs.

Usage:
    python classifier/sweep_all_thresholds.py infer
    python classifier/sweep_all_thresholds.py per-frame
    python classifier/sweep_all_thresholds.py temporal
    python classifier/sweep_all_thresholds.py youtube
    python classifier/sweep_all_thresholds.py plots
    python classifier/sweep_all_thresholds.py all
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from collections import defaultdict
from pathlib import Path

import cv2, joblib, numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO / "ir_gui"))
sys.path.insert(0, str(SCRIPT_DIR))

OUT_ROOT = SCRIPT_DIR / "runs" / "sweep_results"

# ── Dataset paths ────────────────────────────────────────────────────
DATASETS = {
    "antiuav": {
        "rgb_img": Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/images"),
        "ir_img":  Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/images"),
        "rgb_lbl": Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/RGB/labels"),
        "ir_lbl":  Path("G:/drone/Anti-UAV-RGBT_yolo_converted/test/IR/labels"),
    },
    "svanstrom": {
        "rgb_img": Path("G:/drone/svanstrom_paired/RGB/images"),
        "ir_img":  Path("G:/drone/svanstrom_paired/IR/images"),
        "rgb_lbl": Path("G:/drone/svanstrom_paired/RGB/labels"),
        "ir_lbl":  Path("G:/drone/svanstrom_paired/IR/labels"),
    },
}

# Model paths
RGB_WEIGHTS = REPO / "RGB model" / "Yolo26n_hardneg" / "weights" / "best.pt"
IR_WEIGHTS  = REPO / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt"
CLF_PATH    = SCRIPT_DIR / "runs" / "reliability" / "fusion" / "fusion_no_fn_model.joblib"
PATCH_RGB   = SCRIPT_DIR / "runs" / "patches" / "confuser_filter4_rgb.pt"
PATCH_IR    = SCRIPT_DIR / "runs" / "patches" / "confuser_filter4_ir.pt"

SVAN_CATS = ("AIRPLANE", "BIRD", "DRONE", "HELICOPTER")
MIN_CONF = 0.10  # run YOLO at low conf to capture all possible dets


def svan_category(key: str) -> str:
    for c in SVAN_CATS:
        if f"_{c}_" in key:
            return c
    return "OTHER"


def read_yolo_labels(path: Path, w: int, h: int) -> list:
    boxes = []
    if not path.exists():
        return boxes
    for ln in path.read_text().splitlines():
        p = ln.strip().split()
        if len(p) < 5 or p[0] != "0":
            continue
        cx, cy, bw, bh = map(float, p[1:5])
        boxes.append((
            (cx - bw/2)*w, (cy - bh/2)*h,
            (cx + bw/2)*w, (cy + bh/2)*h,
        ))
    return boxes


def iou_iop(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0., ix2 - ix1), max(0., iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0., 0.
    aa = (a[2]-a[0]) * (a[3]-a[1])
    bb = (b[2]-b[0]) * (b[3]-b[1])
    u = aa + bb - inter
    return (inter/u if u > 0 else 0.), (inter/aa if aa > 0 else 0.)


def find_paired_frames(ds: dict) -> list[tuple[str, Path, Path, Path, Path]]:
    """Return list of (key, rgb_img, ir_img, rgb_lbl, ir_lbl) that exist in both."""
    is_antiuav = "Anti-UAV" in str(ds["rgb_img"])
    rgb_suffix = "_visible" if is_antiuav else "_visible"
    ir_suffix = "_infrared" if is_antiuav else "_infrared"

    def make_key(p: Path, suffix: str) -> str:
        return p.stem.replace(suffix, "")

    rgb_map = {make_key(p, rgb_suffix): p for p in ds["rgb_img"].glob("*.*")
               if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")}
    ir_map = {make_key(p, ir_suffix): p for p in ds["ir_img"].glob("*.*")
              if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")}

    common_keys = sorted(set(rgb_map.keys()) & set(ir_map.keys()))
    
    pairs = []
    for k in common_keys:
        r_img = rgb_map[k]
        i_img = ir_map[k]
        r_lbl = ds["rgb_lbl"] / f"{r_img.stem}.txt"
        i_lbl = ds["ir_lbl"] / f"{i_img.stem}.txt"
        pairs.append((k, r_img, i_img, r_lbl, i_lbl))
    return pairs


# ── Phase 0: Fresh inference ─────────────────────────────────────────

def run_infer(datasets_to_run: list[str], fraction: float):
    """Run YOLO + patch verifier + classifier on consecutive 1/N frames."""
    from ultralytics import YOLO
    from patch_verifier import PatchVerifier
    from fusion.features import compute_global_features, compute_target_features, TARGET_NAMES

    out_dir = OUT_ROOT / "cache"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading models...")
    rgb_model = YOLO(str(RGB_WEIGHTS))
    ir_model = YOLO(str(IR_WEIGHTS))
    clf_bundle = joblib.load(CLF_PATH)
    clf_model = clf_bundle["model"]
    feat_cols = clf_bundle["features"]
    patch_rgb = PatchVerifier(str(PATCH_RGB))
    patch_ir = PatchVerifier(str(PATCH_IR))
    print("  All models loaded.")

    def run_yolo(model, frame):
        res = model.predict(frame, conf=MIN_CONF, iou=0.45, imgsz=640,
                            verbose=False, device=0, max_det=300)[0]
        dets = []
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for i in range(len(confs)):
                dets.append(((float(xyxy[i,0]), float(xyxy[i,1]),
                              float(xyxy[i,2]), float(xyxy[i,3])),
                             float(confs[i])))
        return dets

    def build_features(rgb_dets, ir_dets, rgb_gray, ir_gray):
        feats = {}
        for prefix, dets in [("rgb", rgb_dets), ("ir", ir_dets)]:
            confs = [c for _, c in dets]
            n = len(confs)
            if n == 0:
                feats.update({f"{prefix}_n_dets": 0, f"{prefix}_max_conf": 0.,
                              f"{prefix}_mean_conf": 0., f"{prefix}_detected": 0})
            else:
                feats.update({f"{prefix}_n_dets": n,
                              f"{prefix}_max_conf": round(max(confs), 6),
                              f"{prefix}_mean_conf": round(float(np.mean(confs)), 6),
                              f"{prefix}_detected": 1})
        rh, rw = rgb_gray.shape[:2]
        ih, iw = ir_gray.shape[:2]
        g_rgb = compute_global_features(rgb_gray)
        g_ir = compute_global_features(ir_gray)
        feats.update({f"rgb_{k}": v for k, v in g_rgb.items()})
        feats.update({f"ir_{k}": v for k, v in g_ir.items()})
        for prefix, dets, gray, gw, gh in [
            ("rgb", rgb_dets, rgb_gray, rw, rh),
            ("ir",  ir_dets,  ir_gray,  iw, ih)]:
            if not dets:
                feats.update({f"{prefix}_best_{k}": 0. for k in TARGET_NAMES})
            else:
                best_box = max(dets, key=lambda d: d[1])[0]
                tf = compute_target_features(gray, best_box, gw, gh)
                feats.update({f"{prefix}_best_{k}": v for k, v in tf.items()})
        rd, id_ = len(rgb_dets) > 0, len(ir_dets) > 0
        feats["both_detect"] = int(rd and id_)
        feats["neither_detect"] = int(not rd and not id_)
        feats["rgb_only_detect"] = int(rd and not id_)
        feats["ir_only_detect"] = int(not rd and id_)
        return feats

    for ds_name in datasets_to_run:
        ds = DATASETS[ds_name]
        if not ds["rgb_img"].exists():
            print(f"  [SKIP] {ds_name}: {ds['rgb_img']} not found")
            continue

        pairs = find_paired_frames(ds)
        n = len(pairs)
        take = max(1, int(n * fraction))
        start = (n - take) // 2
        pairs = pairs[start:start + take]
        print(f"\n[{ds_name}] {n:,} total paired frames → {len(pairs):,} consecutive ({fraction:.0%})")

        cache_path = out_dir / f"{ds_name}.jsonl"
        t0 = time.time()

        with cache_path.open("w") as fh:
            for fi, (stem, rgb_path, ir_path, rgb_lbl_path, ir_lbl_path) in enumerate(pairs):
                rgb_img = cv2.imread(str(rgb_path))
                ir_img = cv2.imread(str(ir_path))
                if rgb_img is None or ir_img is None:
                    continue

                rh, rw = rgb_img.shape[:2]
                ih, iw = ir_img.shape[:2]

                # YOLO inference
                rgb_dets = run_yolo(rgb_model, rgb_img)
                ir_dets = run_yolo(ir_model, ir_img)

                # Patch verifier on all dets
                rgb_filt_p = (patch_rgb.predict_boxes(
                    rgb_img, [d[0] for d in rgb_dets]).tolist()
                    if rgb_dets else [])
                ir_filt_p = (patch_ir.predict_boxes(
                    ir_img, [d[0] for d in ir_dets]).tolist()
                    if ir_dets else [])

                # Classifier (at a reference conf to get labels)
                rgb_gray = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
                ir_gray = cv2.cvtColor(ir_img, cv2.COLOR_BGR2GRAY)
                feats = build_features(rgb_dets, ir_dets, rgb_gray, ir_gray)
                x = np.array([[feats.get(c, 0) for c in feat_cols]], dtype=np.float32)
                clf_label = int(clf_model.predict(x)[0])

                # GT
                rgb_gt = read_yolo_labels(rgb_lbl_path, rw, rh)
                ir_gt = read_yolo_labels(ir_lbl_path, iw, ih)

                # Match each det against GT
                def match_dets(dets, gts):
                    used_u, used_p = set(), set()
                    matches = []
                    for (box, conf) in dets:
                        best_iu = best_ip = 0.
                        bi_u = bi_p = -1
                        for gi, g in enumerate(gts):
                            iu, ip = iou_iop(box, g)
                            if iu > best_iu: best_iu, bi_u = iu, gi
                            if ip > best_ip: best_ip, bi_p = ip, gi
                        m_u = int(best_iu >= 0.5 and bi_u not in used_u)
                        m_p = int(best_ip >= 0.5 and bi_p not in used_p)
                        if m_u: used_u.add(bi_u)
                        if m_p: used_p.add(bi_p)
                        matches.append((m_u, m_p))
                    return matches

                # Build per-det records: [conf, filter_prob, match_iou, match_iop, box]
                def build_det_records(dets, filt_probs, gts):
                    used_u, used_p = set(), set()
                    records = []
                    for i, (box, conf) in enumerate(dets):
                        fp = filt_probs[i] if i < len(filt_probs) else 0.
                        best_iu = best_ip = 0.
                        bi_u = bi_p = -1
                        for gi, g in enumerate(gts):
                            iu, ip = iou_iop(box, g)
                            if iu > best_iu: best_iu, bi_u = iu, gi
                            if ip > best_ip: best_ip, bi_p = ip, gi
                        m_u = int(best_iu >= 0.5 and bi_u not in used_u)
                        m_p = int(best_ip >= 0.5 and bi_p not in used_p)
                        if m_u: used_u.add(bi_u)
                        if m_p: used_p.add(bi_p)
                        records.append([round(conf, 4), round(fp, 4), m_u, m_p,
                                        [round(x, 1) for x in box]])
                    return records

                rec = {
                    "key": stem,
                    "rgb": build_det_records(rgb_dets, rgb_filt_p, rgb_gt),
                    "ir": build_det_records(ir_dets, ir_filt_p, ir_gt),
                    "rgb_n_gt": len(rgb_gt),
                    "ir_n_gt": len(ir_gt),
                    "clf_label": clf_label,
                }
                fh.write(json.dumps(rec) + "\n")

                if (fi + 1) % 200 == 0:
                    elapsed = time.time() - t0
                    fps = (fi + 1) / elapsed
                    eta = (len(pairs) - fi - 1) / fps
                    print(f"  [{ds_name}] {fi+1}/{len(pairs)}  "
                          f"{fps:.1f} fps  ETA {eta/60:.1f}min")

        elapsed = time.time() - t0
        print(f"  [{ds_name}] Done: {len(pairs)} frames in {elapsed:.0f}s "
              f"→ {cache_path.name}")


# ── Placeholder for Phase A/B/C (next steps) ────────────────────────

def load_cache(ds_name: str) -> list[dict]:
    path = OUT_ROOT / "cache" / f"{ds_name}.jsonl"
    if not path.exists():
        print(f"  [SKIP] {path} not found — run 'infer' first"); return []
    recs = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    print(f"  Loaded {path.name}: {len(recs):,} frames")
    return recs


def run_phase_a(datasets: list[str], fraction: float):
    """Sweep conf × filter thresholds on cached inference data."""
    out_dir = OUT_ROOT / "phase_a"
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_confs = np.arange(0.15, 0.65, 0.05)
    ir_confs = np.arange(0.15, 0.65, 0.05)
    patch_thrs = np.arange(0.40, 1.00, 0.05)
    configs = ["ir_only", "rgb_only", "ir_filter", "rgb_filter"]

    for ds_name in datasets:
        records = load_cache(ds_name)
        if not records: continue

        results = []
        t0 = time.time()
        for rgb_c in rgb_confs:
            for ir_c in ir_confs:
                for p_thr in patch_thrs:
                    for config in configs:
                        for rule_idx, rule in enumerate(["iou", "iop"]):
                            mi = 2 + rule_idx  # match index in det record
                            tp = fp = fn = 0
                            fp_cats = defaultdict(int)
                            for rec in records:
                                cat = svan_category(rec["key"]) if ds_name == "svanstrom" else "OTHER"
                                if config.startswith("ir"):
                                    dets, n_gt = rec["ir"], rec["ir_n_gt"]
                                    ct = ir_c
                                else:
                                    dets, n_gt = rec["rgb"], rec["rgb_n_gt"]
                                    ct = rgb_c
                                use_flt = "filter" in config
                                surv = [d for d in dets if d[0] >= ct]
                                if use_flt:
                                    surv = [d for d in surv if d[1] < p_thr]
                                ft = sum(1 for d in surv if d[mi])
                                ff = len(surv) - ft
                                tp += ft; fp += ff
                                fn += max(0, n_gt - ft)
                                fp_cats[cat] += ff
                            prec = tp/(tp+fp) if tp+fp else 0.
                            rec_ = tp/(tp+fn) if tp+fn else 0.
                            f1 = 2*prec*rec_/(prec+rec_) if prec+rec_ else 0.
                            row = {"dataset": ds_name, "config": config, "rule": rule,
                                   "rgb_conf": round(rgb_c,2), "ir_conf": round(ir_c,2),
                                   "patch_thr": round(p_thr,2),
                                   "TP": tp, "FP": fp, "FN": fn,
                                   "precision": round(prec,4), "recall": round(rec_,4),
                                   "f1": round(f1,4)}
                            if ds_name == "svanstrom":
                                for c in (*SVAN_CATS, "OTHER"):
                                    row[f"fp_{c}"] = fp_cats.get(c, 0)
                            results.append(row)

        csv_path = out_dir / f"{ds_name}_sweep.csv"
        if results:
            with csv_path.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
                w.writeheader(); w.writerows(results)
            print(f"  Saved: {csv_path.name} ({len(results):,} rows) in {time.time()-t0:.1f}s")

        # Print best per config
        print(f"\n[{ds_name}] BEST F1 (IoP):")
        for config in configs:
            sub = [r for r in results if r["config"]==config and r["rule"]=="iop"]
            if not sub: continue
            b = max(sub, key=lambda r: r["f1"])
            print(f"  {config:<14s} rgb={b['rgb_conf']:.2f} ir={b['ir_conf']:.2f} "
                  f"p={b['patch_thr']:.2f}  P={b['precision']:.4f} R={b['recall']:.4f} F1={b['f1']:.4f}")


def run_phase_b(datasets: list[str], fraction: float,
                rgb_conf: float, ir_conf: float, patch_thr: float):
    """Sweep temporal parameters on sequential cached data."""
    from fusion.temporal import PerModalityTemporalState
    out_dir = OUT_ROOT / "phase_b"
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = [3, 5, 8, 10, 15, 20]
    avg_conf_thrs = [0.0, 0.10, 0.20, 0.30, 0.40]
    cooldowns = [0, 5, 10, 15, 30]
    infer_fps_vals = [5, 10, 15, 20, 30]

    for ds_name in datasets:
        records = load_cache(ds_name)
        if not records: continue

        # Group into sequences by stem prefix (before _fNNNNNN)
        sequences = defaultdict(list)
        for rec in records:
            parts = rec["key"].rsplit("_f", 1)
            seq_id = parts[0] if len(parts) == 2 else rec["key"]
            sequences[seq_id].append(rec)
        for s in sequences: sequences[s].sort(key=lambda r: r["key"])
        print(f"  [{ds_name}] {len(sequences)} sequences")

        results = []
        t0 = time.time()
        for infer_fps in infer_fps_vals:
            stride = max(1, int(round(30.0 / infer_fps)))
            for window in windows:
                for require in range(max(2, window//2), window+1):
                    for avg_thr in avg_conf_thrs:
                        for cooldown in cooldowns:
                            s_alert = s_drone = s_false = 0
                            latencies = []
                            for seq_id, frames in sequences.items():
                                has_drone = any(f["rgb_n_gt"]>0 or f["ir_n_gt"]>0 for f in frames)
                                if has_drone: s_drone += 1
                                st = PerModalityTemporalState(
                                    stride=stride, warning_window=window,
                                    warning_require=require, alert_window=window,
                                    alert_require=require, alert_avg_conf_thresh=avg_thr,
                                    warning_cooldown_frames=cooldown,
                                    alert_cooldown_frames=cooldown)
                                alerted = False; first_gt = None
                                for fi, rec in enumerate(frames):
                                    dets = ([[0,0,10,10,d[0]] for d in rec["ir"] if d[0]>=ir_conf]
                                          + [[0,0,10,10,d[0]] for d in rec["rgb"] if d[0]>=rgb_conf])
                                    has_gt = rec["ir_n_gt"]>0 or rec["rgb_n_gt"]>0
                                    if has_gt and first_gt is None: first_gt = fi
                                    _, alert = st.update(dets, 640, 480)
                                    if alert and not alerted:
                                        alerted = True; s_alert += 1
                                        if has_drone and first_gt is not None:
                                            latencies.append(fi - first_gt)
                                        if not has_drone: s_false += 1
                            det_r = s_alert/s_drone if s_drone else 0.
                            fa_r = s_false/max(1, len(sequences)-s_drone) if len(sequences)>s_drone else 0.
                            results.append({
                                "dataset": ds_name, "infer_fps": infer_fps, "stride": stride,
                                "window": window, "require_hits": require,
                                "avg_conf_thr": avg_thr, "cooldown": cooldown,
                                "detection_rate": round(det_r,4),
                                "false_alert_rate": round(fa_r,4),
                                "median_latency": float(np.median(latencies)) if latencies else -1,
                                "n_seqs": len(sequences), "drone_seqs": s_drone,
                            })
            print(f"  [{ds_name}] fps={infer_fps} done ({len(results)} combos, {time.time()-t0:.0f}s)")

        csv_path = out_dir / f"{ds_name}_temporal.csv"
        if results:
            with csv_path.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
                w.writeheader(); w.writerows(results)
            print(f"  Saved: {csv_path.name} ({len(results):,} rows)")
        # Best with 0 false alerts
        clean = [r for r in results if r["false_alert_rate"]==0]
        if clean:
            b = max(clean, key=lambda r: (r["detection_rate"], -r["median_latency"]))
            print(f"  [{ds_name}] BEST (0 false alerts): det_rate={b['detection_rate']:.4f} "
                  f"window={b['window']} require={b['require_hits']} fps={b['infer_fps']} "
                  f"cooldown={b['cooldown']} latency={b['median_latency']:.0f}")


YT_RGB_DIR = Path(r"D:/Downloads/youtube_classifier_videos")
YT_VIDEOS = {
    "airplane_rgb.mp4": "AIRPLANE", "airplane_rgb_2.mp4": "AIRPLANE",
    "airplane_rgb_3.mp4": "AIRPLANE", "airplane_rgb_compilation.mp4": "AIRPLANE",
    "heli_rgb.mp4": "HELICOPTER", "heli_rgb_2.mp4": "HELICOPTER",
    "bird_rgb.mp4": "BIRD", "birds_flock_rgb.mp4": "BIRD",
}

def run_phase_c(fraction: float):
    """Sweep conf × filter thresholds on YouTube confuser videos (GPU)."""
    from ultralytics import YOLO
    from patch_verifier import PatchVerifier
    out_dir = OUT_ROOT / "phase_c"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not RGB_WEIGHTS.exists():
        print(f"  [FATAL] {RGB_WEIGHTS} not found"); return
    model = YOLO(str(RGB_WEIGHTS))
    verifier = PatchVerifier(str(PATCH_RGB), device="cuda:0") if PATCH_RGB.exists() else None
    print(f"  YOLO: {RGB_WEIGHTS.name}  filter: {PATCH_RGB.name if verifier else 'NONE'}")

    rgb_confs = np.arange(0.15, 0.55, 0.05)
    patch_thrs = np.arange(0.40, 1.00, 0.10)
    results = []

    for vname, category in YT_VIDEOS.items():
        path = YT_RGB_DIR / vname
        if not path.exists():
            print(f"  [skip] {vname}"); continue
        cap = cv2.VideoCapture(str(path))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        take = max(1, int(n_total * fraction))
        start = (n_total - take) // 2
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        print(f"\n  [{category}] {vname}: {n_total} frames → {take} sampled")

        frames_data = []
        for fi in range(take):
            ok, frame = cap.read()
            if not ok: break
            res = model.predict(frame, conf=min(rgb_confs), iou=0.45,
                                imgsz=640, verbose=False, device=0, max_det=300)[0]
            dets = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                boxes = [(float(xyxy[i,0]), float(xyxy[i,1]),
                          float(xyxy[i,2]), float(xyxy[i,3])) for i in range(len(confs))]
                filt_probs = np.zeros(len(confs))
                if verifier and boxes:
                    filt_probs = verifier.predict_boxes(frame, boxes)
                dets = [(float(confs[i]), float(filt_probs[i])) for i in range(len(confs))]
            frames_data.append(dets)
            if (fi+1)%500 == 0: print(f"    {fi+1}/{take} frames...")
        cap.release()

        for rgb_c in rgb_confs:
            for p_thr in patch_thrs:
                det_raw = det_flt = 0
                for dets in frames_data:
                    surv_r = [d for d in dets if d[0] >= rgb_c]
                    surv_f = [d for d in surv_r if d[1] < p_thr]
                    if surv_r: det_raw += 1
                    if surv_f: det_flt += 1
                n = len(frames_data)
                results.append({
                    "video": vname, "category": category, "n_frames": n,
                    "rgb_conf": round(rgb_c,2), "patch_thr": round(p_thr,2),
                    "rgb_only_dets": det_raw, "rgb_only_rate": round(det_raw/max(n,1),4),
                    "rgb_filter_dets": det_flt, "rgb_filter_rate": round(det_flt/max(n,1),4),
                    "filter_suppression": round(1 - det_flt/max(det_raw,1),4) if det_raw>0 else 0.,
                })

    csv_path = out_dir / "youtube_sweep.csv"
    if results:
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            w.writeheader(); w.writerows(results)
        print(f"\n  Saved: {csv_path.name} ({len(results):,} rows)")

        import pandas as pd
        df = pd.DataFrame(results)
        agg = df.groupby(["category", "rgb_conf", "patch_thr"]).agg(
            total_dets=("rgb_filter_dets", "sum"),
            total_frames=("n_frames", "sum")
        ).reset_index()
        agg["rate"] = agg["total_dets"] / agg["total_frames"]
        print(f"\n  AGGREGATE any-det rate by category (lower = better):")
        for cat in sorted(df["category"].unique()):
            sub = agg[agg["category"]==cat]
            b = sub.loc[sub["rate"].idxmin()]
            print(f"    {cat}: best rate={b['rate']:.4f} at "
                  f"rgb={b['rgb_conf']:.2f}, p={b['patch_thr']:.2f}")

def generate_plots():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    plot_dir = OUT_ROOT / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase A ──
    for ds in ("antiuav", "svanstrom"):
        cp = OUT_ROOT / "phase_a" / f"{ds}_sweep.csv"
        if not cp.exists(): continue
        iop = pd.read_csv(cp)
        iop = iop[iop["rule"]=="iop"]
        
        # F1 Heatmap
        for config in iop["config"].unique():
            sub = iop[iop["config"]==config]
            best = sub.loc[sub.groupby(["rgb_conf", "ir_conf"])["f1"].idxmax()]
            pivot = best.pivot_table(index="ir_conf", columns="rgb_conf", values="f1", aggfunc="max")
            fig, ax = plt.subplots(figsize=(8,6))
            im = ax.imshow(pivot.values, aspect="auto", cmap="viridis", origin="lower")
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns], rotation=45)
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
            ax.set_xlabel("RGB Conf"); ax.set_ylabel("IR Conf")
            ax.set_title(f"{ds} — {config} F1 (IoP, best patch_thr)")
            plt.colorbar(im, ax=ax, label="F1")
            fig.tight_layout(); fig.savefig(plot_dir / f"{ds}_{config}_f1_heatmap.png", dpi=150)
            plt.close(fig)

        # PR Pareto
        fig, ax = plt.subplots(figsize=(8,6))
        colors = {"ir_only":"#44ddff", "rgb_only":"#44ff88", "ir_filter":"#0088aa", "rgb_filter":"#008844"}
        for config in iop["config"].unique():
            sub = iop[iop["config"]==config]
            ax.scatter(sub["recall"], sub["precision"], alpha=0.1, s=5, color=colors.get(config, "gray"))
            best = sub.sort_values("recall")
            pareto_p, pareto_r, max_p = [], [], 0
            for _, row in best.iterrows():
                if row["precision"] >= max_p:
                    max_p = row["precision"]
                    pareto_p.append(row["precision"])
                    pareto_r.append(row["recall"])
            ax.plot(pareto_r, pareto_p, "-", lw=2, label=config, color=colors.get(config, "gray"))
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title(f"{ds} — PR Pareto Front (IoP)"); ax.legend(); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(plot_dir / f"{ds}_pr_pareto.png", dpi=150)
        plt.close(fig)

    # ── Phase B ──
    for ds in ("antiuav", "svanstrom"):
        cp = OUT_ROOT / "phase_b" / f"{ds}_temporal.csv"
        if not cp.exists(): continue
        df = pd.read_csv(cp)
        valid = df[df["median_latency"] >= 0]
        fig, ax = plt.subplots(figsize=(8,6))
        sc = ax.scatter(valid["median_latency"], valid["detection_rate"],
                        c=valid["window"], cmap="plasma", s=10, alpha=0.5)
        ax.set_xlabel("Median Alert Latency (frames)"); ax.set_ylabel("Sequence Detection Rate")
        ax.set_title(f"{ds} — Temporal: Detection Rate vs Latency")
        plt.colorbar(sc, ax=ax, label="Window Size"); ax.grid(alpha=0.3)
        fig.tight_layout(); fig.savefig(plot_dir / f"{ds}_temporal_det_vs_latency.png", dpi=150)
        plt.close(fig)

    # ── Phase C ──
    cp = OUT_ROOT / "phase_c" / "youtube_sweep.csv"
    if cp.exists():
        df = pd.read_csv(cp)
        fig, axes = plt.subplots(1, 3, figsize=(15,5))
        for i, cat in enumerate(["AIRPLANE", "HELICOPTER", "BIRD"]):
            ax = axes[i]
            sub = df[df["category"]==cat]
            for pthr in sorted(sub["patch_thr"].unique()):
                s = sub[sub["patch_thr"]==pthr]
                agg = s.groupby("rgb_conf")["rgb_filter_rate"].mean()
                ax.plot(agg.index, agg.values, "o-", label=f"p={pthr:.2f}", markersize=4)
            ax.set_xlabel("RGB Conf"); ax.set_ylabel("Detection Rate (lower=better)")
            ax.set_title(cat); ax.legend(fontsize=7); ax.grid(alpha=0.3)
        fig.suptitle("YouTube Confusers — Filter Suppression by Threshold")
        fig.tight_layout(); fig.savefig(plot_dir / "youtube_confuser_rates.png", dpi=150)
        plt.close(fig)

    print(f"  Plots saved to {plot_dir}")


# ── Phase D: Temporal simulation on YouTube confusers ────────────────

def run_phase_d():
    """Sweep temporal settings on cached YouTube per-frame data."""
    from fusion.temporal import PerModalityTemporalState
    yt_csv = OUT_ROOT / "phase_c" / "youtube_sweep.csv"
    if not yt_csv.exists():
        print("  [SKIP] Run 'youtube' first to generate phase_c data"); return

    import pandas as pd
    out_dir = OUT_ROOT / "phase_d"
    out_dir.mkdir(parents=True, exist_ok=True)

    # We need per-frame data, not the aggregated sweep CSV.
    # Re-run YOLO on YouTube at best operating point and cache per-frame hits.
    # Actually, Phase C already ran YOLO but only saved aggregated results.
    # We need to re-run with per-frame caching. Let's do it.
    from ultralytics import YOLO
    from patch_verifier import PatchVerifier

    if not RGB_WEIGHTS.exists():
        print(f"  [FATAL] {RGB_WEIGHTS} not found"); return
    model = YOLO(str(RGB_WEIGHTS))
    verifier = PatchVerifier(str(PATCH_RGB), device="cuda:0") if PATCH_RGB.exists() else None

    rgb_confs = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    patch_thrs = [0.40, 0.55, 0.70, 0.85, 1.00]
    windows = [3, 5, 8, 10, 15, 20]
    cooldowns = [0, 5, 10, 15]

    results = []

    for vname, category in YT_VIDEOS.items():
        path = YT_RGB_DIR / vname
        if not path.exists():
            print(f"  [skip] {vname}"); continue
        cap = cv2.VideoCapture(str(path))
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"\n  [{category}] {vname}: {n_total} frames (ALL)")

        # Cache per-frame: (conf, filter_prob) per detection
        frames_data = []
        for fi in range(n_total):
            ok, frame = cap.read()
            if not ok: break
            res = model.predict(frame, conf=min(rgb_confs), iou=0.45,
                                imgsz=640, verbose=False, device=0, max_det=300)[0]
            dets = []
            if res.boxes is not None and len(res.boxes) > 0:
                xyxy = res.boxes.xyxy.cpu().numpy()
                confs = res.boxes.conf.cpu().numpy()
                boxes = [(float(xyxy[i,0]), float(xyxy[i,1]),
                          float(xyxy[i,2]), float(xyxy[i,3])) for i in range(len(confs))]
                filt_probs = np.zeros(len(confs))
                if verifier and boxes:
                    filt_probs = verifier.predict_boxes(frame, boxes)
                dets = [(float(confs[i]), float(filt_probs[i]),
                         float(xyxy[i,0]), float(xyxy[i,1]),
                         float(xyxy[i,2]), float(xyxy[i,3]))
                        for i in range(len(confs))]
            frames_data.append(dets)
            if (fi+1) % 1000 == 0:
                print(f"    {fi+1}/{n_total} frames...")
        cap.release()
        print(f"    Cached {len(frames_data)} frames")

        # Sweep temporal settings on cached per-frame data
        for rgb_c in rgb_confs:
            for p_thr in patch_thrs:
                for window in windows:
                    for require in range(max(2, window//2), window+1):
                        for cooldown in cooldowns:
                            st = PerModalityTemporalState(
                                stride=1, warning_window=window,
                                warning_require=require, alert_window=window,
                                alert_require=require,
                                warning_cooldown_frames=cooldown,
                                alert_cooldown_frames=cooldown)
                            n_alerts = 0
                            for dets in frames_data:
                                surv = [d for d in dets if d[0] >= rgb_c and d[1] < p_thr]
                                sim_dets = [[d[2], d[3], d[4], d[5], d[0]]
                                            for d in surv]
                                _, alert = st.update(sim_dets, 640, 480)
                                if alert:
                                    n_alerts += 1
                            results.append({
                                "video": vname, "category": category,
                                "n_frames": len(frames_data),
                                "rgb_conf": round(rgb_c, 2),
                                "patch_thr": round(p_thr, 2),
                                "window": window, "require": require,
                                "cooldown": cooldown,
                                "false_alerts": n_alerts,
                            })

    csv_path = out_dir / "youtube_temporal.csv"
    if results:
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            w.writeheader(); w.writerows(results)
        print(f"\n  Saved: {csv_path.name} ({len(results):,} rows)")

        # Find settings with 0 false alerts across ALL videos
        df = pd.DataFrame(results)
        agg = df.groupby(["rgb_conf", "patch_thr", "window", "require", "cooldown"]).agg(
            total_alerts=("false_alerts", "sum"),
            videos=("video", "nunique")
        ).reset_index()
        clean = agg[agg["total_alerts"] == 0]
        if len(clean) > 0:
            # Among 0-alert configs, prefer smallest window (fastest reaction)
            best = clean.sort_values(["window", "require", "rgb_conf"]).iloc[0]
            print(f"\n  BEST 0-alert config: rgb={best['rgb_conf']:.2f} "
                  f"patch={best['patch_thr']:.2f} window={best['window']} "
                  f"require={best['require']} cooldown={best['cooldown']} "
                  f"(across {best['videos']} videos)")
        else:
            # Find minimum total alerts
            best = agg.loc[agg["total_alerts"].idxmin()]
            print(f"\n  WARNING: No 0-alert config found! Best: {best['total_alerts']} "
                  f"alerts at rgb={best['rgb_conf']:.2f} patch={best['patch_thr']:.2f} "
                  f"window={best['window']} require={best['require']}")

        # Per-category breakdown at the best config
        if len(clean) > 0:
            best_cfg = clean.sort_values(["window", "require", "rgb_conf"]).iloc[0]
            sub = df[(df["rgb_conf"]==best_cfg["rgb_conf"]) &
                     (df["patch_thr"]==best_cfg["patch_thr"]) &
                     (df["window"]==best_cfg["window"]) &
                     (df["require"]==best_cfg["require"]) &
                     (df["cooldown"]==best_cfg["cooldown"])]
            print(f"\n  Per-video at best config:")
            for _, row in sub.iterrows():
                print(f"    {row['video']}: {row['false_alerts']} false alerts")

    print("\nPhase D done.")


# ── Cross-dataset Pareto ─────────────────────────────────────────────

def run_pareto():
    """Find settings that work across ALL datasets simultaneously."""
    import pandas as pd
    out_dir = OUT_ROOT / "pareto"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load Phase A results
    dfs_a = {}
    for ds in ("antiuav", "svanstrom"):
        p = OUT_ROOT / "phase_a" / f"{ds}_sweep.csv"
        if p.exists():
            dfs_a[ds] = pd.read_csv(p)

    # Load Phase D results (YouTube temporal)
    yt_temporal = OUT_ROOT / "phase_d" / "youtube_temporal.csv"
    df_d = pd.read_csv(yt_temporal) if yt_temporal.exists() else None

    if not dfs_a:
        print("  [SKIP] No Phase A data found"); return

    # For each threshold combo, compute cross-dataset metrics
    results = []
    configs = ["ir_only", "ir_filter"]  # IR-dominant since RGB fails on Svanström

    for config in configs:
        for ds_name, df in dfs_a.items():
            sub = df[(df["config"] == config) & (df["rule"] == "iop")]
            if sub.empty: continue

            for _, row in sub.iterrows():
                r = {
                    "dataset": ds_name, "config": config,
                    "rgb_conf": row["rgb_conf"], "ir_conf": row["ir_conf"],
                    "patch_thr": row["patch_thr"],
                    "f1": row["f1"], "precision": row["precision"],
                    "recall": row["recall"], "FP": row["FP"],
                }
                results.append(r)

    if not results:
        print("  [SKIP] No results to analyze"); return

    df_all = pd.DataFrame(results)

    # Find configs where BOTH datasets have F1 >= threshold
    print("\n  Cross-dataset Pareto (IR configs, IoP):")
    print("  " + "-" * 70)

    for min_f1 in [0.95, 0.94, 0.93, 0.92]:
        # Group by threshold combo, check both datasets pass
        combos = df_all.groupby(["config", "ir_conf", "patch_thr"]).agg(
            min_f1=("f1", "min"),
            mean_f1=("f1", "mean"),
            total_fp=("FP", "sum"),
            n_datasets=("dataset", "nunique")
        ).reset_index()
        valid = combos[(combos["min_f1"] >= min_f1) & (combos["n_datasets"] >= len(dfs_a))]
        if len(valid) > 0:
            best = valid.sort_values("mean_f1", ascending=False).iloc[0]
            print(f"  min_f1>={min_f1:.2f}: {best['config']} ir={best['ir_conf']:.2f} "
                  f"patch={best['patch_thr']:.2f} → mean_f1={best['mean_f1']:.4f} "
                  f"min_f1={best['min_f1']:.4f} total_fp={best['total_fp']}")
        else:
            print(f"  min_f1>={min_f1:.2f}: No config passes both datasets")

    # If Phase D exists, find intersection with YouTube 0-alert
    if df_d is not None:
        yt_agg = df_d.groupby(["rgb_conf", "patch_thr", "window", "require", "cooldown"]).agg(
            total_alerts=("false_alerts", "sum")
        ).reset_index()
        yt_clean = yt_agg[yt_agg["total_alerts"] == 0]
        if len(yt_clean) > 0:
            yt_patch_vals = set(yt_clean["patch_thr"].unique())
            print(f"\n  YouTube 0-alert patch_thr values: {sorted(yt_patch_vals)}")
            print(f"  YouTube 0-alert min window: "
                  f"{yt_clean.sort_values('window').iloc[0]['window']}")

    # Save
    csv_path = out_dir / "cross_dataset.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path.name}")


def main():
    ap = argparse.ArgumentParser(description="Universal threshold sweep")
    ap.add_argument("phase", choices=["infer", "per-frame", "temporal",
                                       "youtube", "youtube-temporal",
                                       "pareto", "plots", "all"])
    ap.add_argument("--dataset", choices=["antiuav", "svanstrom", "both"],
                    default="both")
    ap.add_argument("--fraction", type=float, default=0.05)
    ap.add_argument("--rgb-conf", type=float, default=0.30)
    ap.add_argument("--ir-conf", type=float, default=0.40)
    ap.add_argument("--patch-thr", type=float, default=0.70)
    args = ap.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    ds_list = ["antiuav", "svanstrom"] if args.dataset == "both" else [args.dataset]

    if args.phase in ("infer", "all"):
        print("=" * 60)
        print("PHASE 0: Fresh inference")
        print("=" * 60)
        run_infer(ds_list, args.fraction)

    if args.phase in ("per-frame", "all"):
        print("\n" + "=" * 60)
        print("PHASE A: Per-frame threshold sweep")
        print("=" * 60)
        run_phase_a(ds_list, args.fraction)

    if args.phase in ("temporal", "all"):
        print("\n" + "=" * 60)
        print("PHASE B: Temporal simulation sweep")
        print("=" * 60)
        run_phase_b(ds_list, args.fraction,
                    args.rgb_conf, args.ir_conf, args.patch_thr)

    if args.phase in ("youtube", "all"):
        print("\n" + "=" * 60)
        print("PHASE C: YouTube confuser video sweep")
        print("=" * 60)
        run_phase_c(args.fraction)

    if args.phase in ("youtube-temporal", "all"):
        print("\n" + "=" * 60)
        print("PHASE D: YouTube temporal simulation")
        print("=" * 60)
        run_phase_d()

    if args.phase in ("pareto", "all"):
        print("\n" + "=" * 60)
        print("Cross-dataset Pareto analysis")
        print("=" * 60)
        run_pareto()

    if args.phase in ("plots", "all"):
        print("\n" + "=" * 60)
        print("Generating plots")
        print("=" * 60)
        generate_plots()


if __name__ == "__main__":
    main()

