import reflex as rx

# --- Premium Cyberpunk / Bloomberg Palette (HSL-based) ---
COLORS = {
    # Backgrounds
    "bg_main": "#080A0F",  # Deeper, more immersive dark
    "bg_card": "hsla(210, 25%, 12%, 0.75)",  # Richer glass base
    "bg_gradient": "radial-gradient(ellipse at 20% 0%, hsla(180, 100%, 25%, 0.15), transparent 50%), radial-gradient(ellipse at 80% 100%, hsla(280, 100%, 20%, 0.1), transparent 50%)",
    # Text
    "text_main": "#A0AEC0",  # Softer gray for readability
    "text_bright": "#F7FAFC",  # Almost white
    "text_dim": "#4A5568",  # For tertiary info
    # Accents
    "neon_cyan": "#00F0FF",  # More vibrant cyan
    "muted_teal": "#38B2AC",  # Balanced teal
    # Trading Colors (HSL for dynamic manipulation)
    "back_blue": "hsl(210, 100%, 50%)",  # Vivid Blue for BACK
    "lay_pink": "hsl(340, 90%, 65%)",  # Coral Pink for LAY
    "profit": "hsl(160, 100%, 50%)",  # Neon Green
    "loss": "hsl(0, 100%, 60%)",  # Bright Red
    "warning": "hsl(40, 100%, 60%)",  # Alert Orange
    # Glassmorphism Borders
    "glass_border": "hsla(180, 100%, 70%, 0.2)",
    "glass_border_hover": "hsla(180, 100%, 70%, 0.45)",
    # Glow Effects
    "glow_cyan": "hsla(180, 100%, 50%, 0.4)",
    "glow_profit": "hsla(160, 100%, 50%, 0.5)",
    "glow_loss": "hsla(0, 100%, 60%, 0.5)",
}

# --- Typography ---
FONTS = {
    "ui": "'Inter', -apple-system, BlinkMacSystemFont, sans-serif",
    "data": "'JetBrains Mono', 'SF Mono', 'Fira Code', monospace",
}

# --- Global Base Style ---
BASE_STYLE = {
    "background_color": COLORS["bg_main"],
    "background_image": COLORS["bg_gradient"],
    "color": COLORS["text_main"],
    "font_family": FONTS["ui"],
    "min_height": "100vh",
}

# --- Premium Glassmorphism Card ---
GLASS_CARD = {
    "background": COLORS["bg_card"],
    "backdrop_filter": "blur(20px) saturate(180%)",
    "-webkit-backdrop-filter": "blur(20px) saturate(180%)",
    "border": f"1px solid {COLORS['glass_border']}",
    "border_radius": "16px",
    "padding": "1.5em",
    "box_shadow": f"0 8px 32px 0 rgba(0, 0, 0, 0.4), inset 0 1px 0 0 hsla(0, 0%, 100%, 0.05)",
    "transition": "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
}

# --- Glassmorphism Card with Hover Glow ---
GLASS_CARD_HOVER = {
    **GLASS_CARD,
    "_hover": {
        "transform": "translateY(-4px) scale(1.01)",
        "box_shadow": f"0 16px 48px {COLORS['glow_cyan']}, inset 0 1px 0 0 hsla(0, 0%, 100%, 0.1)",
        "border": f"1px solid {COLORS['glass_border_hover']}",
    },
}

# --- Metric Card Style (Prominent Values) ---
METRIC_CARD = {
    **GLASS_CARD_HOVER,
    "padding": "1.25em 1.5em",
    "min_width": "180px",
}

# --- Neon Text Glow ---
neon_text_style = {
    "color": COLORS["neon_cyan"],
    "font_weight": "700",
    "text_shadow": f"0 0 8px {COLORS['glow_cyan']}, 0 0 20px {COLORS['glow_cyan']}",
    "font_family": FONTS["data"],
    "letter_spacing": "0.05em",
}

# --- Table Header Style ---
table_header_style = {
    "color": COLORS["muted_teal"],
    "font_size": "0.7em",
    "font_weight": "600",
    "text_transform": "uppercase",
    "letter_spacing": "0.08em",
    "padding": "0.75em 0.5em",
    "border_bottom": f"1px solid {COLORS['glass_border']}",
    "font_family": FONTS["data"],
}

# --- Data Row Style (High Density) ---
DATA_ROW = {
    "padding": "0.75em 0.5em",
    "border_bottom": f"1px solid hsla(180, 50%, 50%, 0.08)",
    "transition": "background-color 0.2s ease",
    "_hover": {
        "background_color": "hsla(180, 100%, 50%, 0.05)",
    },
}

# --- Status Dot Animation (Pulse) ---
STATUS_DOT_LIVE = {
    "width": "10px",
    "height": "10px",
    "border_radius": "50%",
    "background_color": COLORS["profit"],
    "box_shadow": f"0 0 8px {COLORS['glow_profit']}, 0 0 16px {COLORS['glow_profit']}",
    "animation": "pulse 2s infinite",
}

STATUS_DOT_OFFLINE = {
    "width": "10px",
    "height": "10px",
    "border_radius": "50%",
    "background_color": COLORS["loss"],
    "box_shadow": f"0 0 8px {COLORS['glow_loss']}",
}

# --- Keyframes CSS (to be injected via rx.html) ---
KEYFRAMES_CSS = """
<style>
@keyframes pulse {
    0%, 100% { transform: scale(1); opacity: 1; }
    50% { transform: scale(1.15); opacity: 0.8; }
}

@keyframes glow {
    0%, 100% { box-shadow: 0 0 10px hsla(180, 100%, 50%, 0.3); }
    50% { box-shadow: 0 0 25px hsla(180, 100%, 50%, 0.6); }
}

@keyframes scanline {
    0% { background-position: 0 0; }
    100% { background-position: 0 100px; }
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Scanline Overlay */
.scanlines::before {
    content: '';
    position: absolute;
    inset: 0;
    background: repeating-linear-gradient(
        0deg,
        transparent,
        transparent 2px,
        rgba(0, 240, 255, 0.02) 2px,
        rgba(0, 240, 255, 0.02) 4px
    );
    pointer-events: none;
    animation: scanline 8s linear infinite;
}
</style>
"""

# --- Button Styles ---
BUTTON_PRIMARY = {
    "bg": COLORS["neon_cyan"],
    "color": COLORS["bg_main"],
    "font_weight": "600",
    "font_family": FONTS["data"],
    "letter_spacing": "0.05em",
    "border_radius": "8px",
    "padding": "0.75em 1.5em",
    "box_shadow": f"0 4px 15px {COLORS['glow_cyan']}",
    "transition": "all 0.2s ease",
    "_hover": {
        "transform": "translateY(-2px)",
        "box_shadow": f"0 6px 20px {COLORS['glow_cyan']}",
        "opacity": "0.9",
    },
    "_active": {
        "transform": "translateY(0)",
    },
}

BUTTON_DANGER = {
    "bg": "transparent",
    "color": COLORS["loss"],
    "font_weight": "600",
    "font_family": FONTS["data"],
    "letter_spacing": "0.05em",
    "border": f"1px solid {COLORS['loss']}",
    "border_radius": "8px",
    "padding": "0.75em 1.5em",
    "transition": "all 0.2s ease",
    "_hover": {
        "bg": f"hsla(0, 100%, 60%, 0.15)",
        "box_shadow": f"0 4px 15px {COLORS['glow_loss']}",
    },
}
