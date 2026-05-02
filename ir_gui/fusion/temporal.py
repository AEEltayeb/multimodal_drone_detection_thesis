"""
Temporal logic for fusion mode — shared by fusion_app.py (Tkinter) and api.py (web).

Contains:
  - TemporalContinuity: spatial overlap gate (verbatim from detection.py)
  - PerModalityTemporalState: N-of-M warning/alert windows + cooldowns + ROI propagation
  - Drawing helpers: draw_box, draw_detections, draw_temporal_overlays, overlay_text_big
"""

import cv2
import numpy as np
from collections import deque


# ── HELPER FUNCTIONS ─────────────────────────────────────────────────

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _box_area(b):
    return max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))


def _center_of(b):
    return ((float(b[0]) + float(b[2])) / 2.0, (float(b[1]) + float(b[3])) / 2.0)


def _expand_box_from_center(b, factor, W, H):
    cx, cy = _center_of(b)
    w = float(b[2]) - float(b[0])
    h = float(b[3]) - float(b[1])
    nw = w * float(factor)
    nh = h * float(factor)
    x1 = int(cx - nw / 2)
    y1 = int(cy - nh / 2)
    x2 = int(cx + nw / 2)
    y2 = int(cy + nh / 2)
    return (_clamp(x1, 0, W), _clamp(y1, 0, H), _clamp(x2, 0, W), _clamp(y2, 0, H))


# ── TEMPORAL CONTINUITY (spatial overlap gate — from detection.py) ───

class TemporalContinuity:
    def __init__(self, stride: int = 1):
        self.stride = max(1, int(stride))
        self.prev_gate = None
        self.prev_had_det = False

    def reset(self):
        self.prev_gate = None
        self.prev_had_det = False

    def expansion_factor(self):
        return float(_clamp(2.0 + 0.15 * (self.stride - 1), 2.0, 6.0))

    def build_gate(self, det_box, W, H):
        return _expand_box_from_center(det_box, self.expansion_factor(), W, H)

    def accept(self, det_box, W, H):
        curr_gate = self.build_gate(det_box, W, H)
        if not self.prev_had_det or self.prev_gate is None:
            self.prev_gate = curr_gate
            self.prev_had_det = True
            return True, curr_gate, None, True
        a = self.prev_gate
        b = curr_gate
        ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
        ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
        inter = (ix2 - ix1) > 0 and (iy2 - iy1) > 0
        accepted = bool(inter)
        self.prev_gate = curr_gate
        self.prev_had_det = True
        return accepted, curr_gate, a, False


# ── PER-MODALITY TEMPORAL STATE ──────────────────────────────────────

class PerModalityTemporalState:
    def __init__(self, stride, warning_window=10, warning_require=9,
                 alert_window=10, alert_require=9,
                 alert_avg_conf_thresh=0.0,
                 warning_cooldown_frames=15, alert_cooldown_frames=15,
                 roi_ttl=5, roi_expand=1.5):
        self.stride = stride
        self.tc = TemporalContinuity(stride=stride)

        self.win_warning = deque(maxlen=warning_window)
        self.warning_active = False
        self.warn_cooldown_max = warning_cooldown_frames
        self.warn_cooldown_left = 0
        self.warning_require = warning_require
        self.count_warning_events = 0

        self.win_alert = deque(maxlen=alert_window)
        self.alert_conf_window = deque(maxlen=alert_window)
        self.alert_active = False
        self.alert_cooldown_max = alert_cooldown_frames
        self.alert_cooldown_left = 0
        self.alert_require = alert_require
        self.alert_avg_conf_thresh = alert_avg_conf_thresh
        self.count_alert_events = 0

        # Confuser filter alert-gate: track recent confuser probs
        self.confuser_history = deque(maxlen=alert_window)
        self.confuser_suppressed = False  # True when alert was blocked by confuser

        self.last_roi = None
        self.roi_age = 0
        self.roi_ttl = roi_ttl
        self.roi_expand = roi_expand

        # Last inference cache
        self.last_dets = []
        self.last_dets_sources = []
        self.last_trust = None
        self.last_trust_prob = None

        # Gate visualization
        self.last_gate_prev = None
        self.last_gate_curr = None
        self.last_gate_accepted = None

        # TROI visualization
        self.last_troi_rois = []

    def reset(self):
        self.tc.reset()
        self.win_warning.clear()
        self.win_alert.clear()
        self.alert_conf_window.clear()
        self.warning_active = False
        self.alert_active = False
        self.warn_cooldown_left = 0
        self.alert_cooldown_left = 0
        self.confuser_history.clear()
        self.confuser_suppressed = False
        self.last_roi = None
        self.roi_age = 0
        self.last_dets = []
        self.last_dets_sources = []
        self.last_trust = None
        self.last_trust_prob = None
        self.last_gate_prev = None
        self.last_gate_curr = None
        self.last_gate_accepted = None
        self.last_troi_rois = []

    def add_confuser_prob(self, max_prob):
        """Record the max confuser probability for this frame (None if not run)."""
        self.confuser_history.append(max_prob)

    def should_suppress_alert(self, threshold=0.70, mode="primary_only"):
        """Check if recent confuser probs warrant suppressing an alert.

        Modes:
          - 'primary_only':    30% vote (min 2) of frames above threshold.
          - 'primary_and_avg': primary OR average P >= threshold*0.7 (legacy).
          - 'any_above':       any single frame in history exceeds threshold.
        """
        if not self.confuser_history:
            return False
        valid_probs = [p for p in self.confuser_history if p is not None]
        total_valid = len(valid_probs)
        if total_valid == 0:
            return False
        flagged = sum(1 for p in valid_probs if p >= threshold)

        if mode == "any_above":
            return flagged >= 1

        # Primary: at least 30% (min 2) of valid frames are confidently confuser
        min_required = max(2, int(total_valid * 0.3))
        if flagged >= min_required:
            return True

        # Secondary (legacy): average P(confuser) across window is high
        if mode == "primary_and_avg":
            avg_p = sum(valid_probs) / total_valid
            if avg_p >= threshold * 0.7:
                return True

        return False

    def update(self, dets, frame_w, frame_h, confuser_threshold=None,
               confuser_suppress_mode="primary_only"):
        """Update temporal window. If confuser_threshold is set, alerts are
        suppressed (but window NOT reset) when recent crops exceed that
        confuser probability (behaviour depends on confuser_suppress_mode)."""
        detected = len(dets) > 0
        best_conf = max(d[4] for d in dets) if detected else 0.0

        rep_box = None
        if detected:
            best = max(dets, key=lambda d: d[4])
            rep_box = (best[0], best[1], best[2], best[3])

        # Cooldown handling
        if self.warn_cooldown_left > 0:
            self.warn_cooldown_left -= 1
            self.win_warning.clear()
            self.warning_active = False

        if self.alert_cooldown_left > 0:
            self.alert_cooldown_left -= 1
            self.tc.reset()
            self.win_alert.clear()
            self.alert_conf_window.clear()
            self.alert_active = False

        warn_hit = False
        alert_hit = False

        if rep_box is None:
            self.tc.reset()
            # Clear gate visualization (no spatial reference)
            self.last_gate_prev = None
            self.last_gate_curr = None
            self.last_gate_accepted = None
            # Don't clear windows — let N-of-M handle misses naturally.
            # warn_hit and alert_hit stay False → 0 appended below.
        else:
            ok_w, curr_gate, prev_gate, _ = self.tc.accept(rep_box, frame_w, frame_h)
            warn_hit = bool(ok_w) if self.warn_cooldown_left <= 0 else False
            self.last_gate_prev = prev_gate
            self.last_gate_curr = curr_gate
            self.last_gate_accepted = bool(ok_w)
            alert_hit = bool(ok_w) if self.alert_cooldown_left <= 0 else False

        if self.warn_cooldown_left <= 0:
            self.win_warning.append(1 if warn_hit else 0)
        if self.alert_cooldown_left <= 0:
            self.win_alert.append(1 if alert_hit else 0)
            self.alert_conf_window.append(float(best_conf) if alert_hit else 0.0)

        warning_active = (
            len(self.win_warning) == self.win_warning.maxlen
            and sum(self.win_warning) >= self.warning_require
        ) if self.warn_cooldown_left <= 0 else False

        alert_active_pre = (
            len(self.win_alert) == self.win_alert.maxlen
            and sum(self.win_alert) >= self.alert_require
        ) if self.alert_cooldown_left <= 0 else False

        if alert_active_pre and self.alert_avg_conf_thresh > 0:
            hit_confs = [c for c, h in zip(self.alert_conf_window, self.win_alert) if h]
            avg_conf = (sum(hit_confs) / len(hit_confs)) if hit_confs else 0.0
            if avg_conf < self.alert_avg_conf_thresh:
                alert_active_pre = False

        # Confuser filter alert-gate: suppress alert if majority of
        # recent frames had high confuser probability
        self.confuser_suppressed = False
        if alert_active_pre and confuser_threshold is not None:
            if self.should_suppress_alert(confuser_threshold, mode=confuser_suppress_mode):
                alert_active_pre = False
                self.confuser_suppressed = True

        alert_active = bool(alert_active_pre)

        if warning_active and not self.warning_active:
            self.count_warning_events += 1
            if self.warn_cooldown_max > 0:
                self.warn_cooldown_left = self.warn_cooldown_max
                self.win_warning.clear()
        if alert_active and not self.alert_active:
            self.count_alert_events += 1
            if self.alert_cooldown_max > 0:
                self.alert_cooldown_left = self.alert_cooldown_max
                self.win_alert.clear()
                self.alert_conf_window.clear()

        self.warning_active = bool(warning_active)
        self.alert_active = bool(alert_active)

        # ROI propagation
        if detected:
            best = max(dets, key=lambda d: d[4])
            self.last_roi = tuple(best[:4])
            self.roi_age = 0
        else:
            self.roi_age += 1
            if self.roi_age > self.roi_ttl:
                self.last_roi = None

        self.last_dets = list(dets)
        return self.warning_active, self.alert_active

    def get_window_info(self):
        w_hits = sum(self.win_warning) if self.win_warning else 0
        w_size = self.win_warning.maxlen or 0
        hit_confs = [c for c in self.alert_conf_window if c > 0]
        avg_conf = float(np.mean(hit_confs)) if hit_confs else 0.0
        return w_hits, w_size, avg_conf

    def get_roi_crop(self, frame, img_w, img_h):
        if self.last_roi is None or self.roi_age == 0:
            return None
        x1, y1, x2, y2 = self.last_roi
        bw, bh = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        nw = max(bw * self.roi_expand, 128)
        nh = max(bh * self.roi_expand, 128)
        rx1 = int(max(0, cx - nw / 2))
        ry1 = int(max(0, cy - nh / 2))
        rx2 = int(min(img_w, cx + nw / 2))
        ry2 = int(min(img_h, cy + nh / 2))
        if rx2 - rx1 < 32 or ry2 - ry1 < 32:
            return None
        return frame[ry1:ry2, rx1:rx2], (rx1, ry1)

    @staticmethod
    def remap_dets(dets, offset):
        ox, oy = offset
        return [[d[0] + ox, d[1] + oy, d[2] + ox, d[3] + oy, d[4]] for d in dets]


# ── DRAWING HELPERS ──────────────────────────────────────────────────

def draw_box(img, b, color, thickness=2, label=None, anchor="tl"):
    """Draw rectangle + label outside box (from detection.py)."""
    x1, y1, x2, y2 = map(int, b)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if not label:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thick = 0.5, 1
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick)
    H, W = img.shape[:2]
    pad = 2
    if anchor == "tr":
        tx, ty = x2 - tw - pad, y1 - pad
    elif anchor == "bl":
        tx, ty = x1 + pad, y2 + th + pad
    elif anchor == "br":
        tx, ty = x2 - tw - pad, y2 + th + pad
    else:
        tx, ty = x1 + pad, y1 - pad
    tx = int(_clamp(tx, 0, max(0, W - tw - 1)))
    if ty - th < 0:
        ty = y2 + th + pad
    if ty >= H:
        ty = y1 - pad
    ty = int(_clamp(ty, th + 1, H - 2))
    cv2.putText(img, label, (tx, ty), font, scale, color, thick, cv2.LINE_AA)


def draw_detections(frame, dets, color, label_prefix="", trusted=None,
                    sources=None, show_source_tags=False):
    """Draw detection boxes with optional source coloring."""
    for i, d in enumerate(dets):
        x1, y1, x2, y2, conf = int(d[0]), int(d[1]), int(d[2]), int(d[3]), d[4]
        src = sources[i] if sources and i < len(sources) else "full"

        if trusted is not None:
            c = (0, 255, 0) if trusted[i] else (0, 0, 255)
        elif show_source_tags and sources:
            c = (255, 0, 255) if src == "troi" else (0, 255, 0)
        else:
            c = color

        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        tag = ""
        if show_source_tags and sources:
            tag = "TROI " if src == "troi" else "FULL "
        label = f"{label_prefix}{tag}{conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), c, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_temporal_overlays(frame, temporal, settings):
    """Draw TROI ROI regions and temporal continuity gate boxes."""
    if temporal is None:
        return

    if settings.get("show_troi", False) and temporal.last_troi_rois:
        for roi in temporal.last_troi_rois:
            draw_box(frame, roi, (255, 0, 255), 1, label="TemporalROI", anchor="bl")

    if settings.get("show_gate", False):
        if temporal.last_gate_prev is not None:
            draw_box(frame, temporal.last_gate_prev, (0, 183, 235), 1,
                     label="TCPrev", anchor="bl")
        if temporal.last_gate_curr is not None:
            draw_box(frame, temporal.last_gate_curr, (0, 255, 255), 1,
                     label="TCCurr", anchor="br")


def overlay_text_big(img, lines):
    """Big-font overlay, matching detection.py's overlay_text_big."""
    h, w = img.shape[:2]
    scale = max(0.6, h / 900.0 * 1.35)
    thickness_bg = max(3, int(h / 900.0 * 6))
    thickness_fg = max(1, int(h / 900.0 * 2))
    x = int(w * 0.02)
    line_h = int(h * 0.055)
    y = line_h + int(h * 0.015)
    for i, t in enumerate(lines):
        max_chars = int(w / (scale * 14))
        t = t[:max_chars]
        yy = y + i * line_h
        cv2.putText(img, t, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, float(scale),
                    (0, 0, 0), thickness_bg, cv2.LINE_AA)
        cv2.putText(img, t, (x, yy), cv2.FONT_HERSHEY_SIMPLEX, float(scale),
                    (0, 255, 0), thickness_fg, cv2.LINE_AA)


def build_overlay_lines(temporal, settings, prefix=""):
    """Build overlay lines matching detection.py's windows_big format."""
    if temporal is None:
        return []

    w_hits = sum(temporal.win_warning)
    w_len = len(temporal.win_warning)
    w_need = settings["warning_require_hits"]
    w_win = settings["warning_window_frames"]
    w_events = temporal.count_warning_events

    a_hits = sum(temporal.win_alert)
    a_len = len(temporal.win_alert)
    a_need = settings["alert_require_hits"]
    a_win = settings["alert_window_frames"]
    a_events = temporal.count_alert_events

    avg_conf_str = ""
    if float(settings.get("alert_avg_conf_threshold", 0)) > 0:
        hit_confs = [c for c, h in zip(temporal.alert_conf_window, temporal.win_alert) if h]
        avg_conf = (sum(hit_confs) / len(hit_confs)) if hit_confs else 0.0
        avg_conf_str = f"  avgConf={avg_conf:.2f}"

    suppressed_str = ""
    if temporal.confuser_suppressed:
        suppressed_str = " [CONFUSER SUPPRESSED]"

    return [
        f"{prefix}WARNING hits {w_hits}/{w_len}  need {w_need}/{w_win}  events={w_events}",
        f"{prefix}ALERT   hits {a_hits}/{a_len}  need {a_need}/{a_win}{avg_conf_str}  events={a_events}{suppressed_str}",
    ]
