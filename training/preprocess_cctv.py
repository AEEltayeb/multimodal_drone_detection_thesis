"""
preprocess_cctv.py — Named OpenCV preprocessing transforms for CCTV-to-phone bridging.

All transforms operate on BGR uint8 ndarrays and return BGR uint8 ndarrays.
All are deterministic and run in 1–10 ms per frame.

Usage:
    from preprocess_cctv import apply, VARIANT_NAMES
    frame_out = apply(frame_bgr, "clahe_unsharp_2_10")
"""

from __future__ import annotations
import cv2
import numpy as np


# ── Primitives ────────────────────────────────────────────────────────────────

def _unsharp(bgr: np.ndarray, amount: float, sigma: float) -> np.ndarray:
    blur = cv2.GaussianBlur(bgr, (0, 0), sigma)
    return cv2.addWeighted(bgr, 1.0 + amount, blur, -amount, 0)


def _clahe(bgr: np.ndarray, clip: float, tile: int) -> np.ndarray:
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


def _gamma(bgr: np.ndarray, g: float) -> np.ndarray:
    lut = (np.arange(256, dtype=np.float32) / 255.0) ** (1.0 / g)
    lut = (lut * 255).clip(0, 255).astype(np.uint8)
    return cv2.LUT(bgr, lut)


def _saturation(bgr: np.ndarray, scale: float) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _bilateral(bgr: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(bgr, d=5, sigmaColor=50, sigmaSpace=50)


def _nlm(bgr: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(bgr, None, h=5, hColor=5,
                                            templateWindowSize=7, searchWindowSize=21)


def _laplacian(bgr: np.ndarray) -> np.ndarray:
    lap = cv2.Laplacian(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
    lap = np.clip(np.abs(lap), 0, 255).astype(np.uint8)
    lap_bgr = cv2.cvtColor(lap, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(bgr, 1.0, lap_bgr, 0.5, 0)


# ── Named transforms ──────────────────────────────────────────────────────────

def _none(f):             return f
def _unsharp_05(f):       return _unsharp(f, 0.5, 1.0)
def _unsharp_10(f):       return _unsharp(f, 1.0, 1.0)
def _unsharp_15(f):       return _unsharp(f, 1.5, 1.0)
def _laplacian_fn(f):     return _laplacian(f)
def _clahe_2_8(f):        return _clahe(f, 2.0, 8)
def _clahe_3_8(f):        return _clahe(f, 3.0, 8)
def _clahe_4_8(f):        return _clahe(f, 4.0, 8)
def _clahe_2_16(f):       return _clahe(f, 2.0, 16)
def _c2u10(f):            return _unsharp(_clahe(f, 2.0, 8), 1.0, 1.0)
def _c3u10(f):            return _unsharp(_clahe(f, 3.0, 8), 1.0, 1.0)
def _c4u15(f):            return _unsharp(_clahe(f, 4.0, 8), 1.5, 1.0)
def _bil_u(f):            return _unsharp(_bilateral(f), 1.0, 1.0)
def _nlm_u(f):            return _unsharp(_nlm(f), 1.0, 1.0)
def _g08cu(f):            return _unsharp(_clahe(_gamma(f, 0.8), 2.0, 8), 1.0, 1.0)
def _g12cu(f):            return _unsharp(_clahe(_gamma(f, 1.2), 2.0, 8), 1.0, 1.0)
def _full(f):             return _gamma(_unsharp(_clahe(_bilateral(f), 2.0, 8), 1.0, 1.0), 0.85)
def _sat13(f):            return _saturation(f, 1.3)
def _sat_cu(f):           return _saturation(_unsharp(_clahe(f, 2.0, 8), 1.0, 1.0), 1.3)


# Ordered by expected impact (best guesses first; saturation last / low expectation)
VARIANTS: list[tuple[str, callable]] = [
    ("none",                    _none),
    ("clahe_unsharp_2_10",      _c2u10),
    ("clahe_unsharp_3_10",      _c3u10),
    ("clahe_unsharp_4_15",      _c4u15),
    ("unsharp_10",              _unsharp_10),
    ("unsharp_05",              _unsharp_05),
    ("unsharp_15",              _unsharp_15),
    ("clahe_2_8",               _clahe_2_8),
    ("clahe_3_8",               _clahe_3_8),
    ("clahe_4_8",               _clahe_4_8),
    ("clahe_2_16",              _clahe_2_16),
    ("gamma_08_clahe_unsharp",  _g08cu),
    ("gamma_12_clahe_unsharp",  _g12cu),
    ("full_kitchen_sink",       _full),
    ("bilateral_unsharp",       _bil_u),
    ("nlm_unsharp",             _nlm_u),
    ("laplacian",               _laplacian_fn),
    ("saturation_13",           _sat13),
    ("saturation_clahe_unsharp",_sat_cu),
]

VARIANT_NAMES = [name for name, _ in VARIANTS]
_DISPATCH = {name: fn for name, fn in VARIANTS}


def apply(bgr: np.ndarray, name: str) -> np.ndarray:
    fn = _DISPATCH.get(name)
    if fn is None:
        raise ValueError(f"Unknown preprocessing variant: {name!r}. "
                         f"Available: {VARIANT_NAMES}")
    return fn(bgr)
