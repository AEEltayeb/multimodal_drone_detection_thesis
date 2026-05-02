"""Theme constants for the Flet drone detection app."""
import flet as ft

# ── Color palette ─────────────────────────────────────────
DARK = {
    "bg": "#0f1117",
    "surface": "#1a1d27",
    "card": "#1e2230",
    "card_border": "#2a2e3d",
    "card_hover": "#252a3a",
    "text": "#e8eaf0",
    "text_dim": "#8b8fa3",
    "text_muted": "#5c6078",
    "accent": "#6c5ce7",
    "accent_dim": "#5a4bd1",
    "cyan": "#00cec9",
    "green": "#00b894",
    "amber": "#fdcb6e",
    "red": "#e74c3c",
    "blue": "#74b9ff",
    "input_bg": "#151821",
    "input_border": "#2a2e3d",
    "divider": "#252838",
    "logo": "logo.png",
}

LIGHT = {
    "bg": "#f5f6fa",
    "surface": "#ffffff",
    "card": "#ffffff",
    "card_border": "#e0e3eb",
    "card_hover": "#f0f1f5",
    "text": "#1a1d27",
    "text_dim": "#6b7085",
    "text_muted": "#9ca0b3",
    "accent": "#6c5ce7",
    "accent_dim": "#5a4bd1",
    "cyan": "#0097a7",
    "green": "#00897b",
    "amber": "#f57f17",
    "red": "#d32f2f",
    "blue": "#1976d2",
    "input_bg": "#f0f1f5",
    "input_border": "#d5d8e3",
    "divider": "#e8eaf0",
    "logo": "logo-black.png",
}

TRUST_COLORS = {0: "#e74c3c", 1: "#00b894", 2: "#00cec9", 3: "#00e676"}
TRUST_LABELS = {0: "REJECT BOTH", 1: "TRUST RGB", 2: "TRUST IR", 3: "TRUST BOTH"}


def card(content, theme, padding=16, border_radius=14, expand=False, **kw):
    return ft.Container(
        content=content,
        bgcolor=theme["card"],
        border=ft.border.all(1, theme["card_border"]),
        border_radius=border_radius,
        padding=padding,
        expand=expand,
        shadow=ft.BoxShadow(
            spread_radius=0, blur_radius=12, color="#00000018",
            offset=ft.Offset(0, 4),
        ),
        **kw,
    )


def pill_button(text, on_click, active, theme, accent=None):
    ac = accent or theme["accent"]
    return ft.Container(
        content=ft.Text(text, size=12, weight=ft.FontWeight.W_600,
                        color="#ffffff" if active else theme["text_dim"]),
        bgcolor=ac if active else "transparent",
        border_radius=8,
        padding=ft.padding.symmetric(12, 8),
        on_click=on_click,
        ink=True,
        animate=ft.animation.Animation(200, ft.AnimationCurve.EASE_IN_OUT),
    )


def icon_btn(icon, tooltip, on_click, theme, size=18):
    return ft.IconButton(
        icon=icon, icon_size=size, tooltip=tooltip,
        icon_color=theme["text_dim"],
        style=ft.ButtonStyle(
            bgcolor=theme["surface"],
            shape=ft.RoundedRectangleBorder(radius=8),
            side=ft.BorderSide(1, theme["card_border"]),
        ),
        on_click=on_click,
    )
