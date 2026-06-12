"""Session analytics — tracks per-video detection stats and builds visual charts."""
import flet as ft
from .widgets import card, section_title, CENTER
from .theme import TRUST_LABELS

P = ft.Padding


class SessionStats:
    """Accumulates stats during a detection session."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.video_name = ""
        self.mode = ""
        self.total_frames = 0
        self.processed_frames = 0
        self.rgb_detections = 0
        self.ir_detections = 0
        self.trust_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        self.confuser_vetoes = 0
        self.alerts = 0
        self.warnings = 0
        self.fps_samples = []
        self.confidence_samples = []

    def record_frame(self, stats: dict):
        self.processed_frames += 1
        fps = stats.get("fps", 0)
        if fps:
            self.fps_samples.append(fps)
        trust = stats.get("trust")
        if trust is not None and trust in self.trust_counts:
            self.trust_counts[trust] += 1
        if stats.get("alert"):
            self.alerts += 1
        if stats.get("warning"):
            self.warnings += 1
        tp = stats.get("trust_prob")
        if tp:
            self.confidence_samples.append(tp)

    @property
    def avg_fps(self):
        return sum(self.fps_samples) / max(len(self.fps_samples), 1)

    @property
    def avg_confidence(self):
        return sum(self.confidence_samples) / max(len(self.confidence_samples), 1)

    @property
    def total_trust_frames(self):
        return sum(self.trust_counts.values())


def _bar(value, max_val, color, width=200, height=20):
    """Horizontal bar using Container width ratio."""
    pct = value / max(max_val, 1)
    return ft.Container(
        content=ft.Container(
            width=max(4, width * pct), height=height,
            bgcolor=color, border_radius=4,
        ),
        width=width, height=height, bgcolor=color + "20",
        border_radius=4, clip_behavior=ft.ClipBehavior.HARD_EDGE,
    )


def _pie_segment(label, value, total, color, t):
    """Row representing one pie segment (visual legend + bar)."""
    pct = value / max(total, 1) * 100
    return ft.Row([
        ft.Container(width=12, height=12, border_radius=3, bgcolor=color),
        ft.Text(label, size=12, color=t["text"], width=120),
        _bar(value, total, color, width=120, height=14),
        ft.Text(f"{int(value)}", size=12, weight=ft.FontWeight.W_700, color=t["text"]),
        ft.Text(f"({pct:.0f}%)", size=11, color=t["text_dim"]),
    ], spacing=8, vertical_alignment=ft.CrossAxisAlignment.CENTER)


def build_analytics_view(stats: SessionStats, t: dict):
    """Build the full analytics page content."""
    if stats.processed_frames == 0:
        return ft.Container(
            content=ft.Column([
                ft.Icon(ft.Icons.ANALYTICS_ROUNDED, size=64, color=t["text_muted"]),
                ft.Text("No detection data yet", size=18, weight=ft.FontWeight.W_600,
                        color=t["text_muted"]),
                ft.Text("Run a video through detection to see analytics",
                        size=13, color=t["text_dim"]),
            ], alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER, spacing=8),
            expand=True, alignment=CENTER,
        )

    total_trust = stats.total_trust_frames
    trust_colors = {0: "#ff6b6b", 1: "#00b894", 2: "#00cec9", 3: "#00e676"}

    # ── Session Overview card ────────────────────────
    overview = card(ft.Column([
        section_title("Session Overview", t, ft.Icons.INFO_ROUNDED),
        ft.Divider(height=1, color=t["divider"]),
        ft.Row([
            ft.Column([
                ft.Text("Video", size=11, color=t["text_dim"]),
                ft.Text(stats.video_name or "—", size=14,
                        weight=ft.FontWeight.W_600, color=t["text"]),
            ]),
            ft.Column([
                ft.Text("Mode", size=11, color=t["text_dim"]),
                ft.Text(stats.mode or "—", size=14,
                        weight=ft.FontWeight.W_600, color=t["text"]),
            ]),
            ft.Column([
                ft.Text("Frames", size=11, color=t["text_dim"]),
                ft.Text(str(stats.processed_frames), size=14,
                        weight=ft.FontWeight.W_600, color=t["text"]),
            ]),
            ft.Column([
                ft.Text("Avg FPS", size=11, color=t["text_dim"]),
                ft.Text(f"{stats.avg_fps:.1f}", size=14,
                        weight=ft.FontWeight.W_600, color=t["cyan"]),
            ]),
        ], spacing=40),
    ], spacing=10), t, padding=20)

    # ── Fusion Classifier Distribution ───────────────
    trust_chart = card(ft.Column([
        section_title("Fusion Classifier — Trust Distribution", t,
                      ft.Icons.PIE_CHART_ROUNDED),
        ft.Divider(height=1, color=t["divider"]),
        ft.Column([
            _pie_segment(TRUST_LABELS[k], v, total_trust, trust_colors[k], t)
            for k, v in stats.trust_counts.items()
        ], spacing=8),
        ft.Container(height=4),
        ft.Row([
            ft.Text("Dominant:", size=12, color=t["text_dim"]),
            ft.Text(
                TRUST_LABELS[max(stats.trust_counts, key=stats.trust_counts.get)],
                size=13, weight=ft.FontWeight.W_700,
                color=trust_colors[max(stats.trust_counts, key=stats.trust_counts.get)]),
        ], spacing=8),
    ], spacing=10), t, padding=20)

    # ── Detection Performance ────────────────────────
    perf_items = [
        ("Avg Confidence", f"{stats.avg_confidence * 100:.1f}%", t["accent"]),
        ("Alerts Triggered", str(stats.alerts), t["red"]),
        ("Warnings", str(stats.warnings), t["amber"]),
        ("Confuser Vetoes", str(stats.confuser_vetoes), "#e17055"),
    ]
    perf_cards = []
    for label, val, color in perf_items:
        perf_cards.append(card(ft.Column([
            ft.Text(label, size=11, color=t["text_dim"]),
            ft.Text(val, size=28, weight=ft.FontWeight.W_700, color=color),
        ], spacing=4, horizontal_alignment=ft.CrossAxisAlignment.CENTER),
            t, padding=16, elevated=True))

    perf_row = ft.Row(perf_cards, spacing=10, wrap=True)

    # ── Model Breakdown ──────────────────────────────
    model_card = card(ft.Column([
        section_title("Model Breakdown", t, ft.Icons.MODEL_TRAINING_ROUNDED),
        ft.Divider(height=1, color=t["divider"]),
        _model_row("RGB YOLO", t["green"],
                   f"{stats.rgb_detections} detections", t),
        _model_row("IR YOLO", t["cyan"],
                   f"{stats.ir_detections} detections", t),
        _model_row("Fusion Classifier", t["accent"],
                   f"{total_trust} decisions", t),
        _model_row("Confuser Filter", "#e17055",
                   f"{stats.confuser_vetoes} vetoes", t),
    ], spacing=10), t, padding=20)

    return ft.Column([
        ft.Text("Analytics", size=22, weight=ft.FontWeight.W_700, color=t["text"]),
        ft.Text(f"Session data for: {stats.video_name or 'No video'}",
                size=13, color=t["text_dim"]),
        ft.Container(height=8),
        overview,
        ft.Row([trust_chart, ft.Column([perf_row, model_card], spacing=10, expand=True)],
               spacing=12, vertical_alignment=ft.CrossAxisAlignment.START),
    ], spacing=10, scroll=ft.ScrollMode.AUTO, expand=True)


def _model_row(name, color, detail, t):
    return ft.Row([
        ft.Container(width=4, height=36, border_radius=2, bgcolor=color),
        ft.Column([
            ft.Text(name, size=13, weight=ft.FontWeight.W_600, color=t["text"]),
            ft.Text(detail, size=11, color=t["text_dim"]),
        ], spacing=2),
    ], spacing=10)
