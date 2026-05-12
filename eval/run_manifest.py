"""run_manifest.py — Provenance helpers for eval scripts.

Every eval entry-point should write a `manifest.json` to its --output-dir at
the *start* of the run (before any heavy work) so that even crashed runs leave
a record. The manifest captures:

  * git commit + dirty flag
  * timestamp + python/torch/ultralytics/cuda versions
  * the parsed argparse Namespace (as dict)
  * resolved paths and short sha256 hashes for: rgb_weights, ir_weights,
    classifier_path, patch_rgb_weights, patch_ir_weights, every cache file
    actually loaded
  * resolved dataset roots
  * caller-supplied "extra" dict (e.g. cache-rebuild reason)

Also exposes weights_short_hash(path) — the same 12-char sha256 prefix used
to auto-tag YOLO cache filenames so different model weights don't silently
share a cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_HASH_LEN = 12


def _short_sha256(path: Path) -> str | None:
    """sha256 of file bytes, hex, first 12 chars. None if file missing."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()[:_HASH_LEN]
    except OSError:
        return None


def weights_short_hash(path: str | Path | None) -> str:
    """Public alias used by cache_inference.py for cache-filename tagging.

    Returns 'missing' if the path doesn't exist (so callers can still build a
    deterministic tag — they'll get a cache-miss downstream).
    """
    if not path:
        return "missing"
    p = Path(path)
    if not p.is_file():
        return "missing"
    return _short_sha256(p) or "missing"


def _git_commit(repo: Path) -> tuple[str, bool]:
    """Return (commit_sha, dirty). ('unknown', False) if not a git repo."""
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo), stderr=subprocess.DEVNULL, text=True,
        ).strip()
        dirty_out = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(repo), stderr=subprocess.DEVNULL, text=True,
        )
        return sha, bool(dirty_out.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown", False


def _safe_version(import_name: str) -> str:
    """Best-effort version probe; never raises."""
    try:
        mod = __import__(import_name)
        return getattr(mod, "__version__", "unknown")
    except Exception:
        return "unavailable"


def _cuda_status() -> dict[str, Any]:
    try:
        import torch
        return {
            "available": bool(torch.cuda.is_available()),
            "device_count": int(torch.cuda.device_count()),
            "device_name": (torch.cuda.get_device_name(0)
                            if torch.cuda.is_available() else None),
        }
    except Exception:
        return {"available": False, "device_count": 0, "device_name": None}


def _hash_block(label: str, path: str | Path | None) -> dict[str, Any]:
    """Return {label, path, sha256_short, exists} for a weights/cache file."""
    if not path:
        return {"label": label, "path": None, "sha256_short": None, "exists": False}
    p = Path(path)
    return {
        "label": label,
        "path": str(p),
        "sha256_short": _short_sha256(p) if p.is_file() else None,
        "exists": p.is_file(),
        "size_bytes": p.stat().st_size if p.is_file() else None,
    }


def manifest_dict(
    args: Any,
    cfg: dict[str, Any] | None = None,
    repo_root: Path | None = None,
    weights_paths: dict[str, str | Path | None] | None = None,
    cache_paths: list[str | Path] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the provenance manifest dict.

    Args:
        args:           argparse Namespace from the calling eval script.
        cfg:            loaded eval/config.yaml (for dataset paths). Optional.
        repo_root:      repo root for git-commit lookup. Defaults to two levels
                        up from this file.
        weights_paths:  dict of label -> path for weights to hash. Common keys:
                        "rgb_weights", "ir_weights", "classifier_path",
                        "patch_rgb_weights", "patch_ir_weights".
        cache_paths:    list of cache files actually loaded by the run.
        extra:          arbitrary additional fields.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[1]

    commit, dirty = _git_commit(repo_root)

    args_dict: dict[str, Any]
    try:
        args_dict = {k: v for k, v in vars(args).items()}
    except TypeError:
        args_dict = {"_repr": repr(args)}

    weights_block = []
    for label, p in (weights_paths or {}).items():
        weights_block.append(_hash_block(label, p))

    cache_block = []
    for p in (cache_paths or []):
        cache_block.append(_hash_block("cache", p))

    dataset_roots: dict[str, str] = {}
    if cfg and isinstance(cfg.get("datasets"), dict):
        for name, ds in cfg["datasets"].items():
            if isinstance(ds, dict) and "root" in ds:
                dataset_roots[name] = str(ds["root"])

    return {
        "schema_version": 1,
        "timestamp_iso": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": commit,
            "dirty": dirty,
            "repo_root": str(repo_root),
        },
        "env": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "torch": _safe_version("torch"),
            "ultralytics": _safe_version("ultralytics"),
            "numpy": _safe_version("numpy"),
            "opencv": _safe_version("cv2"),
            "cuda": _cuda_status(),
        },
        "args": args_dict,
        "weights": weights_block,
        "caches": cache_block,
        "datasets": dataset_roots,
        "extra": extra or {},
    }


def write_manifest(
    out_dir: str | Path,
    args: Any,
    cfg: dict[str, Any] | None = None,
    weights_paths: dict[str, str | Path | None] | None = None,
    cache_paths: list[str | Path] | None = None,
    extra: dict[str, Any] | None = None,
    filename: str = "manifest.json",
) -> Path:
    """Write the manifest to out_dir/manifest.json. Creates out_dir if needed.
    Returns the path written. Caller is encouraged to do this BEFORE heavy
    work so a crashed run still leaves a record.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    m = manifest_dict(
        args=args, cfg=cfg, weights_paths=weights_paths,
        cache_paths=cache_paths, extra=extra,
    )
    path = out / filename
    path.write_text(json.dumps(m, indent=2, default=str))
    return path


def cache_identity_tag(
    rgb_weights: str | Path | None,
    ir_weights: str | Path | None,
    imgsz: int,
    stride: int,
    conf: float | None = None,
) -> str:
    """Build the auto-tag for a YOLO detection cache filename.

    Format: rgb<rh>_ir<ih>_sz<imgsz>_st<stride>[_c<conf>]
    where rh/ih are 12-char weight-file hashes (or 'missing').
    """
    rh = weights_short_hash(rgb_weights)
    ih = weights_short_hash(ir_weights)
    parts = [f"rgb{rh}", f"ir{ih}", f"sz{imgsz}", f"st{stride}"]
    if conf is not None:
        parts.append(f"c{conf:.3f}")
    return "_".join(parts)


if __name__ == "__main__":
    # Smoke test — print a manifest for the current process.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    m = manifest_dict(args=args)
    print(json.dumps(m, indent=2, default=str))
