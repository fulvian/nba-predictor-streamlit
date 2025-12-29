import reflex as rx
from typing import Any
from .state import State
from .styles import (
    COLORS,
    FONTS,
    BASE_STYLE,
    GLASS_CARD,
    GLASS_CARD_HOVER,
    METRIC_CARD,
    DATA_ROW,
    STATUS_DOT_LIVE,
    STATUS_DOT_OFFLINE,
    BUTTON_PRIMARY,
    BUTTON_DANGER,
    KEYFRAMES_CSS,
    neon_text_style,
    table_header_style,
)

# --- Components ---


def metric_card(
    label: str, value: Any, trend: rx.Var = None, prefix: str = "", suffix: str = ""
):
    """Premium HUD Metric Card with Glassmorphism, Trend, and Glow."""
    return rx.vstack(
        rx.hstack(
            rx.text(
                label,
                font_size="0.7em",
                color=COLORS["muted_teal"],
                font_weight="600",
                letter_spacing="0.08em",
                text_transform="uppercase",
                font_family=FONTS["data"],
            ),
            rx.spacer(),
            rx.cond(
                trend >= 0,
                rx.box(
                    rx.text("▲", font_size="0.65em"),
                    color=COLORS["profit"],
                    text_shadow=f"0 0 8px {COLORS['glow_profit']}",
                ),
                rx.box(
                    rx.text("▼", font_size="0.65em"),
                    color=COLORS["loss"],
                    text_shadow=f"0 0 8px {COLORS['glow_loss']}",
                ),
            )
            if trend is not None
            else rx.spacer(),
            align_items="center",
            width="100%",
        ),
        rx.text(
            prefix,
            value,
            suffix,
            font_size="2.2em",
            font_family=FONTS["data"],
            color=COLORS["text_bright"],
            font_weight="700",
            letter_spacing="-0.02em",
            text_shadow=f"0 0 20px {COLORS['glow_cyan']}",
        ),
        spacing="0.6em",
        align_items="flex-start",
        style=METRIC_CARD,
    )


def market_row(market: rx.Var):
    """High-density Market Scanner Row with hover glow."""
    volume_percentage = (market["volume"].to(float) / 100000.0) * 100.0

    return rx.box(
        rx.grid(
            # Event Info
            rx.vstack(
                rx.text(
                    market["market_name"],
                    color=COLORS["text_bright"],
                    font_weight="600",
                    font_size="0.85em",
                    overflow="hidden",
                    white_space="nowrap",
                    text_overflow="ellipsis",
                    max_width="220px",
                ),
                rx.text(
                    market["market_id"],
                    font_size="0.6em",
                    color=COLORS["text_dim"],
                    font_family=FONTS["data"],
                ),
                align_items="flex-start",
                spacing="0.15em",
            ),
            # Status pulsing dot
            rx.hstack(
                rx.cond(
                    market["status"] == "OPEN",
                    rx.box(style=STATUS_DOT_LIVE),
                    rx.box(style=STATUS_DOT_OFFLINE),
                ),
                rx.text(
                    market["status"],
                    font_size="0.65em",
                    color=COLORS["text_main"],
                    font_family=FONTS["data"],
                ),
                spacing="0.5em",
                align_items="center",
            ),
            # Volume bar
            rx.vstack(
                rx.text(
                    "€",
                    market["volume"],
                    font_size="0.75em",
                    font_family=FONTS["data"],
                    color=COLORS["neon_cyan"],
                    font_weight="500",
                ),
                rx.box(
                    rx.box(
                        width=f"{volume_percentage}%",
                        height="3px",
                        background=f"linear-gradient(90deg, {COLORS['neon_cyan']}, {COLORS['muted_teal']})",
                        border_radius="2px",
                    ),
                    width="100%",
                    height="3px",
                    background_color=f"hsla(180, 50%, 50%, 0.15)",
                    border_radius="2px",
                ),
                align_items="flex-end",
                spacing="0.25em",
            ),
            template_columns="2.5fr 1fr 1.5fr",
            width="100%",
            align_items="center",
            gap="1em",
        ),
        style=DATA_ROW,
        cursor="pointer",
        on_click=lambda: State.set_selected_market_id(market["market_id"]),
    )


def odds_row(runner: rx.Var):
    """Tactical Detail: Order Book Row with gradient fills."""
    return rx.grid(
        rx.text(
            runner["runner_name"],
            color=COLORS["text_bright"],
            font_weight="600",
            font_size="0.8em",
            overflow="hidden",
            white_space="nowrap",
            text_overflow="ellipsis",
        ),
        # BACK (Blue Gradient)
        rx.vstack(
            rx.text(
                runner["back_price"],
                font_weight="700",
                color="white",
                font_size="1em",
                font_family=FONTS["data"],
            ),
            rx.text(
                "€",
                runner["back_size"],
                font_size="0.55em",
                color="hsla(0, 0%, 100%, 0.75)",
                font_family=FONTS["data"],
            ),
            background=f"linear-gradient(135deg, {COLORS['back_blue']}, hsl(210, 80%, 40%))",
            padding="0.5em 0.75em",
            border_radius="8px",
            align_items="center",
            spacing="0.1em",
            box_shadow=f"0 4px 12px hsla(210, 100%, 50%, 0.3)",
        ),
        # LAY (Pink Gradient)
        rx.vstack(
            rx.text(
                runner["lay_price"],
                font_weight="700",
                color=COLORS["bg_main"],
                font_size="1em",
                font_family=FONTS["data"],
            ),
            rx.text(
                "€",
                runner["lay_size"],
                font_size="0.55em",
                color="hsla(0, 0%, 0%, 0.6)",
                font_family=FONTS["data"],
            ),
            background=f"linear-gradient(135deg, {COLORS['lay_pink']}, hsl(340, 70%, 55%))",
            padding="0.5em 0.75em",
            border_radius="8px",
            align_items="center",
            spacing="0.1em",
            box_shadow=f"0 4px 12px hsla(340, 90%, 65%, 0.3)",
        ),
        rx.text(
            "€",
            runner["total_matched"],
            color=COLORS["muted_teal"],
            font_family=FONTS["data"],
            font_size="0.75em",
        ),
        template_columns="2.2fr 1fr 1fr 1fr",
        gap="0.75em",
        padding="0.6em 0",
        align_items="center",
        border_bottom=f"1px solid hsla(180, 50%, 50%, 0.08)",
    )


def anomaly_card(anomaly: rx.Var):
    """Component: Live Anomaly Signal Card."""
    # Convert severity to color
    color = rx.cond(
        anomaly["severity"] == "CRITICAL",
        COLORS["loss"],
        rx.cond(anomaly["severity"] == "HIGH", COLORS["warning"], COLORS["neon_cyan"]),
    )

    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.badge(
                    anomaly["type"],
                    variant="outline",
                    color_scheme=rx.cond(
                        anomaly["severity"] == "CRITICAL", "red", "cyan"
                    ),
                    font_size="0.6em",
                ),
                rx.spacer(),
                rx.text(
                    "EV: ",
                    rx.text(
                        (anomaly["ev"].to(float) * 100).to_string(),
                        "%",
                        font_weight="bold",
                    ),
                    font_size="0.7em",
                    color=COLORS["profit"],
                ),
                width="100%",
                message=anomaly["details"],
            ),
            rx.text(
                anomaly["details"],
                font_size="0.8em",
                color=COLORS["text_bright"],
                font_weight="600",
            ),
            rx.text(
                anomaly["market_id"],
                font_size="0.6em",
                color=COLORS["text_dim"],
            ),
            spacing="0.3em",
            align_items="flex-start",
        ),
        style=GLASS_CARD,
        border_left=f"3px solid {color}",
        padding="0.8em",
        margin_bottom="0.5em",
    )


def trading_log_row(trade: rx.Var):
    """Component: Row in trading log."""
    return rx.hstack(
        rx.text(
            trade["action"],
            font_weight="bold",
            color=rx.cond(
                trade["action"] == "BACK", COLORS["back_blue"], COLORS["lay_pink"]
            ),
        ),
        rx.text(trade["runner_name"], flex="1", font_size="0.8em"),
        rx.text("@", trade["price"], font_weight="bold"),
        rx.text("€", trade["stake"], color=COLORS["text_dim"]),
        font_family=FONTS["data"],
        font_size="0.75em",
        width="100%",
        padding="0.3em 0",
        border_bottom=f"1px solid {COLORS['glass_border']}",
    )


def index() -> rx.Component:
    return rx.box(
        # Meta: Load fonts and inject keyframes
        rx.html(
            '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">'
        ),
        rx.html(KEYFRAMES_CSS),
        # Scanline overlay container
        rx.box(
            class_name="scanlines",
            position="fixed",
            top="0",
            left="0",
            right="0",
            bottom="0",
            pointer_events="none",
            z_index="1000",
        ),
        # Main Content
        rx.box(
            # Header / Logo Area
            rx.hstack(
                rx.heading("PROJECT NEON", style=neon_text_style, size="lg"),
                rx.spacer(),
                rx.hstack(
                    rx.badge(
                        "ENGINE: V2.0",
                        color_scheme="cyan",
                        variant="soft",
                        font_family=FONTS["data"],
                    ),
                    rx.cond(
                        State.connected,
                        rx.badge(
                            "● LIVE",
                            color_scheme="green",
                            variant="solid",
                            font_family=FONTS["data"],
                        ),
                        rx.badge(
                            "○ OFFLINE",
                            color_scheme="red",
                            variant="outline",
                            font_family=FONTS["data"],
                        ),
                    ),
                    spacing="0.5em",
                ),
                width="100%",
                margin_bottom="2em",
                padding="0 0.5em",
                align_items="center",
            ),
            # --- HUD Metrics (500ms TICK) ---
            rx.grid(
                metric_card("MARKETS LIVE", State.market_ids.length(), State.pnl_trend),
                metric_card(
                    "DAILY P&L",
                    State.global_stats["daily_pnl"],
                    State.pnl_trend,
                    prefix="€",
                ),
                metric_card(
                    "OPEN POSITIONS",
                    State.global_stats["open_positions"],
                    rx.Var.create(0),
                ),
                metric_card(
                    "SESSION ROI",
                    State.global_stats["avg_odds"],
                    rx.Var.create(0),
                    suffix="%",
                ),
                columns="4",
                gap="1.25em",
                margin_bottom="2em",
            ),
            # --- MAIN PANELS ---
            rx.grid(
                # LEFT: Market Scanner (2s TICK)
                rx.vstack(
                    rx.hstack(
                        rx.heading(
                            "MARKET SCANNER",
                            color=COLORS["neon_cyan"],
                            size="md",
                            font_family=FONTS["data"],
                            letter_spacing="0.05em",
                        ),
                        rx.spacer(),
                        rx.button(
                            "GO LIVE",
                            on_click=State.start_monitoring,
                            size="sm",
                            style=BUTTON_PRIMARY,
                        ),
                        rx.button(
                            "HALT",
                            on_click=State.stop_monitoring,
                            size="sm",
                            style=BUTTON_DANGER,
                        ),
                        width="100%",
                        margin_bottom="1em",
                        align_items="center",
                        spacing="0.75em",
                    ),
                    # Column Headers
                    rx.grid(
                        rx.text("ACTIVE EVENTS", style=table_header_style),
                        rx.text("STATUS", style=table_header_style),
                        rx.text("LIQUIDITY", style=table_header_style),
                        template_columns="2.5fr 1fr 1.5fr",
                        width="100%",
                        padding="0 0.5em",
                    ),
                    # Rows
                    rx.scroll_area(
                        rx.vstack(
                            rx.foreach(State.dashboard_grid, market_row),
                            width="100%",
                            spacing="0",
                        ),
                        height="60vh",
                        style=GLASS_CARD,
                    ),
                    width="100%",
                    align_items="flex-start",
                ),
                # RIGHT: Tactical Detail
                rx.vstack(
                    # Selector
                    rx.select(
                        State.market_ids,
                        placeholder="SELECT TARGET MARKET...",
                        on_change=State.set_selected_market_id,
                        width="100%",
                        bg=COLORS["bg_card"],
                        color=COLORS["text_bright"],
                        border=f"1px solid {COLORS['glass_border']}",
                        border_radius="8px",
                        font_family=FONTS["data"],
                        margin_bottom="1em",
                    ),
                    # Order Book
                    rx.vstack(
                        rx.heading(
                            "TACTICAL VIEW",
                            size="sm",
                            color=COLORS["neon_cyan"],
                            font_family=FONTS["data"],
                            margin_bottom="0.75em",
                        ),
                        rx.grid(
                            rx.text("PARTICIPANT", style=table_header_style),
                            rx.text("BACK", style=table_header_style),
                            rx.text("LAY", style=table_header_style),
                            rx.text("MATCHED", style=table_header_style),
                            template_columns="2.2fr 1fr 1fr 1fr",
                            width="100%",
                        ),
                        rx.scroll_area(
                            rx.vstack(
                                rx.foreach(State.focused_odds, odds_row),
                                width="100%",
                                spacing="0",
                            ),
                            height="35vh",
                        ),
                        style=GLASS_CARD,
                        width="100%",
                        align_items="flex-start",
                    ),
                    # System Telemetry
                    rx.vstack(
                        rx.heading(
                            "SYSTEM TELEMETRY",
                            size="xs",
                            color=COLORS["muted_teal"],
                            font_family=FONTS["data"],
                            letter_spacing="0.1em",
                        ),
                        rx.scroll_area(
                            rx.cond(
                                State.alerts.length() > 0,
                                rx.vstack(
                                    rx.foreach(
                                        State.alerts,
                                        lambda log: rx.text(
                                            log,
                                            font_family=FONTS["data"],
                                            font_size="0.7em",
                                            color=COLORS["neon_cyan"],
                                        ),
                                    ),
                                    width="100%",
                                    spacing="0.25em",
                                ),
                                rx.text(
                                    "AWAITING DATA STREAM...",
                                    font_family=FONTS["data"],
                                    font_size="0.7em",
                                    color=COLORS["text_dim"],
                                    font_style="italic",
                                ),
                            ),
                            height="12vh",
                        ),
                        background_color="hsla(0, 0%, 0%, 0.5)",
                        backdrop_filter="blur(10px)",
                        padding="1em",
                        border=f"1px solid {COLORS['glass_border']}",
                        width="100%",
                        border_radius="12px",
                    ),
                    width="100%",
                    spacing="1em",
                ),
                # LOAD SYSTEM PANEL (NEW)
                rx.vstack(
                    rx.hstack(
                        rx.heading(
                            "LOAD SYSTEM",
                            color=COLORS["warning"],
                            size="md",
                            font_family=FONTS["data"],
                        ),
                        rx.spacer(),
                        rx.cond(
                            State.load_system_enabled,
                            rx.button(
                                "ACTIVE",
                                on_click=State.toggle_load_system,
                                style=BUTTON_PRIMARY,
                                color_scheme="green",
                            ),
                            rx.button(
                                "ENABLE",
                                on_click=State.toggle_load_system,
                                style=BUTTON_PRIMARY,
                            ),
                        ),
                        width="100%",
                        align_items="center",
                    ),
                    # Stats Row
                    rx.grid(
                        metric_card(
                            "ANOMALIES", State.load_stats["anomalies_detected"]
                        ),
                        metric_card("E.V. TRADES", State.load_stats["active_bets"]),
                        columns="2",
                        gap="1em",
                        width="100%",
                    ),
                    # Anomalies Feed
                    rx.heading(
                        "LIVE ANOMALIES",
                        size="xs",
                        color=COLORS["text_dim"],
                        margin_top="1em",
                    ),
                    rx.scroll_area(
                        rx.vstack(
                            rx.foreach(State.active_anomalies, anomaly_card),
                            width="100%",
                        ),
                        height="30vh",
                        style=GLASS_CARD,
                    ),
                    # Trading Log
                    rx.heading(
                        "EXECUTION LOG",
                        size="xs",
                        color=COLORS["text_dim"],
                        margin_top="1em",
                    ),
                    rx.scroll_area(
                        rx.vstack(
                            rx.foreach(State.recent_trades, trading_log_row),
                            width="100%",
                        ),
                        height="20vh",
                        style=GLASS_CARD,
                    ),
                    width="100%",
                ),
                template_columns="1.3fr 1fr 1fr",
                gap="1.5em",
                width="100%",
            ),
            padding="2em",
            position="relative",
            z_index="1",
        ),
        background_color=COLORS["bg_main"],
        background_image=COLORS["bg_gradient"],
        min_height="100vh",
        on_mount=State.on_load,
    )


app = rx.App(style=BASE_STYLE)
app.add_page(index)
