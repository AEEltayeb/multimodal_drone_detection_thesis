"""Reusable UI components — Flet 0.84 compatible."""
import flet as ft

CENTER = ft.Alignment(0, 0)


def card(content, t, padding=20, br=14, expand=False, elevated=False, **kw):
    """Card with subtle glass-edge border."""
    bg = t["card_elevated"] if elevated else t["card"]
    return ft.Container(
        content=content, bgcolor=bg,
        border=ft.Border.all(1, t["card_border"]),
        border_radius=br, padding=padding, expand=expand,
        shadow=ft.BoxShadow(
            spread_radius=0, blur_radius=16 if elevated else 6,
            color="#00000035" if elevated else "#00000015",
            offset=ft.Offset(0, 4 if elevated else 2)),
        **kw,
    )


def status_chip(label, color, t):
    """Colored status pill."""
    return ft.Container(
        content=ft.Row([
            ft.Container(width=7, height=7, border_radius=4, bgcolor=color),
            ft.Text(label, size=11, weight=ft.FontWeight.W_600, color=color),
        ], spacing=6),
        bgcolor=color + "18", border=ft.Border.all(1, color + "40"),
        border_radius=20, padding=ft.Padding(left=10, right=10, top=5, bottom=5),
    )


def section_title(text, t, icon=None):
    items = []
    if icon:
        items.append(ft.Icon(icon, size=16, color=t["accent"]))
    items.append(ft.Text(text, size=13, weight=ft.FontWeight.W_700, color=t["text"]))
    return ft.Row(items, spacing=6)


def sidebar_item(icon, label, t, active=False, on_click=None):
    """Sidebar navigation item — active uses accent purple."""
    accent = t["accent"]
    return ft.Container(
        content=ft.Row([
            ft.Icon(icon, size=20, color=accent if active else t["text_dim"]),
            ft.Text(label, size=13,
                    weight=ft.FontWeight.W_600 if active else ft.FontWeight.W_400,
                    color=t["text"] if active else t["text_dim"]),
        ], spacing=12),
        bgcolor=t["sidebar_active"] if active else "transparent",
        border=ft.Border.all(1, t["logo_blue"]) if active else None,
        border_radius=10, padding=ft.Padding(14, 10, 14, 10),
        on_click=on_click, ink=True,
    )
