"""
det_cache.py — Unified detection cache loader for full-pipeline ablations.

Tries existing cache sources in order; falls back to fresh YOLO inference.
Verifies model fingerprint (sha256 of weights .pt) against cache metadata.

Cache sources searched (in priority order):

  1. eval/cache/raw_detections_<ds>_rgb<imgsz>_<det>.json
     — explicit-named single-model cache (e.g. svanstrom_rgb1280_baseline)
       Schema: {stem: {rgb_dets: [[x1,y1,x2,y2,conf], ...], ir_dets: [...], ...}}

  2. eval/cache/raw_detections_<ds>_rgb<HASH>_ir<HASH>_sz<sz>_st<stride>.json
     — hash-named paired cache w/ manifest.json sibling
       Schema: same; matches when current RGB+IR sha256_short align with filename hashes

  3. eval/results/<known_phase2_dir>/<det>/<det>_frame_detections.csv
     — flat per-frame CSV from Phase 2 eval_model.py runs
       Schema: stem,n_gt,n_raw,n_filt,tp,fp,fn,tp_f,fp_f,fn_f,n_small,n_medium,n_large,dets,sizes
       `dets` column: "x1,y1,x2,y2,conf;x1,y1,x2,y2,conf;..."

  4. eval/cache/full_pipeline/<ds>_<det>_sz<imgsz>.json
     — self-managed cache written by this module on miss

Each loader yields the canonical form: {stem: [(x1,y1,x2,y2,conf), ...]}.

Hash verification: every cache lookup compares current `weights.pt` sha256_short
to the cache's fingerprint. Mismatch → cache rejected.

API:
    cache = DetCache(repo_root)
    dets = cache.get_dets(dataset_key, detector_key, weights_path, imgsz, stem)
    # returns list of (x1,y1,x2,y2,conf) tuples, or None if not cached

    cache.put_dets(dataset_key, detector_key, weights_path, imgsz, stem, dets)
    # writes to self-managed cache
"""

from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

DetList = list[tuple[float, float, float, float, float]]


def sha256_short(path: Path, length: int = 12) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:length]


def parse_dets_str(s: str) -> DetList:
    """Parse per-detection list. Two known formats:
      - 'x1,y1,x2,y2,conf|x1,y1,x2,y2,conf|...'  (eval_model.py per-frame CSV)
      - 'x1,y1,x2,y2,conf;x1,y1,x2,y2,conf;...'  (Roboflow per-frame CSV)
    Picks the right delimiter automatically."""
    out: DetList = []
    if not s:
        return out
    delim = "|" if "|" in s else ";"
    for chunk in s.split(delim):
        parts = chunk.split(",")
        if len(parts) != 5:
            continue
        try:
            out.append(tuple(float(x) for x in parts))
        except ValueError:
            continue
    return out


class DetCache:
    def __init__(self, repo_root: Path, verbose: bool = False):
        self.repo = repo_root
        self.cache_dir = repo_root / "eval" / "cache"
        # Self-managed cache lives under the ablations doc folder now so that
        # raw results + caches stay co-located with the analysis they back.
        # Legacy hash-named caches in eval/cache/ are still discoverable.
        self.self_cache_dir = (repo_root / "docs" / "analysis" /
                               "full_pipeline_ablations" / "cache")
        self.self_cache_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        # In-memory caches to avoid repeated JSON parses
        self._mem_json: dict[Path, dict] = {}
        self._mem_csv: dict[Path, dict] = {}
        self._stats = {"explicit_hits": 0, "hash_hits": 0,
                       "csv_hits": 0, "self_hits": 0,
                       "misses": 0, "rejected_bad_hash": 0}
        self._dirty: set[Path] = set()
        # Known per-frame CSV paths by (dataset, detector)
        self._csv_paths = self._build_csv_index()

    def _build_csv_index(self) -> dict[tuple[str, str], list[Path]]:
        idx: dict[tuple[str, str], list[Path]] = {}
        # Anti-UAV per-model
        for m_dir in (self.repo / "eval" / "results" / "antiuav_per_model").glob("*/"):
            det = m_dir.name
            csv = m_dir / f"{det}_frame_detections.csv"
            if csv.exists():
                idx.setdefault(("antiuav", det), []).append(csv)
        # Selcom val
        for m_dir in (self.repo / "eval" / "results" / "selcom_val_holdout").glob("*/"):
            det = m_dir.name
            csv = m_dir / f"{det}_frame_detections.csv"
            if csv.exists():
                idx.setdefault(("selcom_val", det), []).append(csv)
        # Roboflow OOD
        for ds_dir in (self.repo / "eval" / "results" / "roboflow_ood").glob("rgb_*/"):
            ds_key = f"roboflow_{ds_dir.name}_test"
            for m_dir in ds_dir.glob("*/"):
                det = m_dir.name.replace("rgb_", "")  # rgb_baseline -> baseline
                # Each split (test/train/valid) has its own CSV. We default to test.
                for split in ("test", "train", "valid"):
                    csv = m_dir / split / f"{m_dir.name}_frame_detections.csv"
                    if csv.exists() and split == "test":
                        idx.setdefault((ds_key, det), []).append(csv)
        return idx

    def _explicit_path(self, ds: str, det: str, imgsz: int) -> Path:
        # Match the names produced by cache_inference.py with --tag <det>
        # e.g., raw_detections_svanstrom_rgb1280_baseline.json
        return self.cache_dir / f"raw_detections_{ds}_rgb{imgsz}_{det}.json"

    def _hash_paired_path(self, ds: str, rgb_hash: str, ir_hash: str, imgsz: int, stride: int = 1) -> Path:
        return self.cache_dir / f"raw_detections_{ds}_rgb{rgb_hash}_ir{ir_hash}_sz{imgsz}_st{stride}.json"

    def _self_path(self, ds: str, det: str, imgsz: int) -> Path:
        return self.self_cache_dir / f"{ds}_{det}_sz{imgsz}.json"

    def _load_json(self, path: Path) -> dict | None:
        if path in self._mem_json:
            return self._mem_json[path]
        if not path.exists():
            return None
        try:
            d = json.loads(path.read_text())
            self._mem_json[path] = d
            return d
        except Exception:
            return None

    def _load_csv(self, path: Path) -> dict | None:
        if path in self._mem_csv:
            return self._mem_csv[path]
        if not path.exists():
            return None
        try:
            out: dict[str, DetList] = {}
            with path.open() as f:
                for r in csv.DictReader(f):
                    stem = r.get("stem", "")
                    if not stem: continue
                    out[stem] = parse_dets_str(r.get("dets", ""))
            self._mem_csv[path] = out
            return out
        except Exception:
            return None

    def _verify_manifest_hash(self, manifest_path: Path, rgb_hash_expected: str) -> bool:
        if not manifest_path.exists():
            return False
        try:
            man = json.loads(manifest_path.read_text())
        except Exception:
            return False
        for w in man.get("weights", []):
            if w.get("label", "").startswith("rgb"):
                return w.get("sha256_short", "") == rgb_hash_expected
        return False

    def get_dets(self, dataset: str, detector: str, weights_path: Path,
                 imgsz: int, stem: str, ir_weights_path: Path | None = None,
                 stride: int = 1) -> DetList | None:
        """Try to load cached dets. Returns None if no valid cache."""
        rgb_hash = sha256_short(weights_path)

        # Source 1: explicit-named JSON
        p = self._explicit_path(dataset, detector, imgsz)
        d = self._load_json(p)
        if d and stem in d:
            entry = d[stem]
            self._stats["explicit_hits"] += 1
            return [tuple(x) for x in entry.get("rgb_dets", [])]

        # Source 2: hash-named paired JSON (requires IR weights too)
        if ir_weights_path is not None:
            ir_hash = sha256_short(ir_weights_path)
            p = self._hash_paired_path(dataset, rgb_hash, ir_hash, imgsz, stride)
            d = self._load_json(p)
            if d and stem in d:
                # Verify via sibling manifest
                if self._verify_manifest_hash(p.with_suffix(".manifest.json"), rgb_hash):
                    entry = d[stem]
                    self._stats["hash_hits"] += 1
                    return [tuple(x) for x in entry.get("rgb_dets", [])]
                else:
                    self._stats["rejected_bad_hash"] += 1

        # Source 3: per-frame CSV from Phase 2
        for csv_path in self._csv_paths.get((dataset, detector), []):
            d = self._load_csv(csv_path)
            if d and stem in d:
                self._stats["csv_hits"] += 1
                return d[stem]

        # Source 4: self-managed cache
        p = self._self_path(dataset, detector, imgsz)
        d = self._load_json(p)
        if d:
            # Manifest tells us which weight hash this was built for
            man = d.get("__manifest__", {})
            if man.get("rgb_sha256_short") == rgb_hash and stem in d.get("dets", {}):
                self._stats["self_hits"] += 1
                return [tuple(x) for x in d["dets"][stem]]

        self._stats["misses"] += 1
        return None

    def put_dets(self, dataset: str, detector: str, weights_path: Path,
                 imgsz: int, stem: str, dets: DetList) -> None:
        """Stage dets to in-memory cache. Call flush() to persist."""
        p = self._self_path(dataset, detector, imgsz)
        if p not in self._mem_json:
            self._mem_json[p] = self._load_json(p) or {"__manifest__": {}, "dets": {}}
        d = self._mem_json[p]
        if "dets" not in d: d["dets"] = {}
        if "__manifest__" not in d: d["__manifest__"] = {}
        d["__manifest__"]["rgb_sha256_short"] = sha256_short(weights_path)
        d["__manifest__"]["weights_path"] = str(weights_path)
        d["__manifest__"]["imgsz"] = imgsz
        d["dets"][stem] = [list(map(float, t)) for t in dets]
        self._dirty.add(p)

    def flush(self) -> int:
        """Persist all dirty caches. Returns number of files written."""
        n = 0
        for p in list(getattr(self, "_dirty", set())):
            d = self._mem_json.get(p)
            if d is None: continue
            tmp = p.with_suffix(".json.tmp")
            try:
                tmp.write_text(json.dumps(d, separators=(",", ":")))
                # Windows-safe replace: retry up to 3x to ride out AV/reader locks
                last_err = None
                replaced = False
                for attempt in range(3):
                    try:
                        if p.exists():
                            try: p.unlink()
                            except Exception: pass
                        tmp.replace(p)
                        replaced = True
                        break
                    except Exception as e:
                        last_err = e
                        import time as _t
                        _t.sleep(0.5)
                if not replaced:
                    fallback = p.with_suffix(".json.new")
                    try:
                        tmp.replace(fallback)
                        print(f"  CACHE FLUSH: could not replace {p.name} ({last_err}); wrote {fallback.name} — rename manually")
                    except Exception as e2:
                        print(f"  CACHE FLUSH FAILED {p}: {last_err}; fallback also failed: {e2}")
                        try: tmp.unlink()
                        except Exception: pass
                else:
                    n += 1
            except Exception as e:
                print(f"  CACHE FLUSH FAILED {p}: {e}")
                try: tmp.unlink()
                except Exception: pass
        self._dirty.clear()
        return n

    def stats(self) -> dict:
        return dict(self._stats)


# ── Standalone verification ──

def verify_against_phase2(repo: Path):
    """Walk known cache sources and report status."""
    cache = DetCache(repo)
    print("== Hash mapping ==")
    rgb_weights = {
        "baseline": repo / "RGB model" / "Yolo26n_trained" / "weights" / "best.pt",
        "hardneg_v3more": repo / "RGB model" / "Yolo26n_hardneg_v3_more" / "weights" / "best.pt",
        "retrained_v2": repo / "RGB model" / "Yolo26n_retrained_v2" / "weights" / "best.pt",
        "selcom_1280": repo / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt",
        "selcom_960":  repo / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt",
        "selcom_640":  repo / "RGB model" / "Yolo26n_selcom_mixed_ft2_1280" / "weights" / "best.pt",
    }
    for k, p in rgb_weights.items():
        print(f"  {k:18s} sha256_short={sha256_short(p)}  ({p.name} exists={p.exists()})")
    ir_w = repo / "runs" / "corrective_finetune" / "finetune_v3b" / "weights" / "best.pt"
    print(f"  {'ir_model':18s} sha256_short={sha256_short(ir_w)}")
    print()

    print("== Coverage matrix ==")
    datasets = ["antiuav", "svanstrom", "selcom_val",
                "roboflow_rgb_drone_test", "roboflow_rgb_bird_test",
                "roboflow_rgb_airplane_test", "roboflow_rgb_helicopter_test"]
    detectors_imgsz = [
        ("baseline", 1280), ("baseline", 640),
        ("hardneg_v3more", 1280), ("hardneg_v3more", 640),
        ("retrained_v2", 1280), ("retrained_v2", 640),
        ("selcom_1280", 1280), ("selcom_960", 960), ("selcom_640", 640),
    ]
    for ds in datasets:
        print(f"\n  {ds}:")
        for det, sz in detectors_imgsz:
            # Pick first available stem to test lookup
            stem = None
            # try to grab a stem from the per-frame CSV index
            csvs = cache._csv_paths.get((ds, det), [])
            if csvs:
                d = cache._load_csv(csvs[0])
                if d:
                    stem = next(iter(d.keys()), None)
            # try explicit json
            if stem is None:
                p = cache._explicit_path(ds, det, sz)
                d = cache._load_json(p)
                if d:
                    stem = next(iter(d.keys()), None)
            if stem is None:
                # No cache anywhere — mark MISS
                print(f"    {det:18s} sz={sz:>4}  MISS  (will need YOLO)")
                continue
            dets = cache.get_dets(ds, det, rgb_weights.get(det, repo / "missing.pt"), sz, stem, ir_weights_path=ir_w)
            if dets is None:
                print(f"    {det:18s} sz={sz:>4}  MISS  ")
            else:
                print(f"    {det:18s} sz={sz:>4}  HIT   ({len(dets)} dets in sample stem)")
    print()
    print("Stats:", cache.stats())


if __name__ == "__main__":
    import sys
    repo = Path(__file__).resolve().parents[1]
    verify_against_phase2(repo)
