import reflex as rx
from typing import Any

COLORS = {
    "muted_blue": "blue",
    "text_main": "white",
    "bg_card": "black",
    "neon_blue": "cyan",
}
neon_text_style = {"color": "cyan"}
metric_card_style = {"background": "black"}


def metric_card(label: str, value: Any, subvalue: str = None):
    # If value is not a component, wrap it in rx.text with styling
    if isinstance(value, (str, int, float, rx.Var)):
        value_display = rx.text(value, font_size="1.5em", style=neon_text_style)
    else:
        # It's likely a component (like rx.cond)
        value_display = value

    return rx.box(
        rx.text(label, font_size="0.8em", color=COLORS["muted_blue"]),
        value_display,
        rx.cond(
            subvalue is not None,
            rx.text(subvalue, font_size="0.8em", color=COLORS["text_main"]),
            rx.box(),
        ),
        style=metric_card_style,
        width="100%",
    )


class State(rx.State):
    connected: bool = False
    market_ids: list[str] = []


def index():
    return rx.box(
        rx.grid(
            metric_card("TEST", rx.cond(State.connected, rx.text("A"), rx.text("B"))),
            columns="4",
        )
    )


app = rx.App()
app.add_page(index)
print("SUCCESS: App initialized")
