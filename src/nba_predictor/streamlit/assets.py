"""
Assets for the NBA Predictor Dashboard (Anthropic Style)
Contains manually crafted, high-fidelity SVG strings for perfect clarity and consistency.
Style: Minimalist line art, Stroke #191919, Stroke-Width 1.5px, Transparent Fill.
"""

# --- Style Constants ---
STROKE_COLOR = "#191919"
STROKE_WIDTH = "1.5"
FILL_COLOR = "none"
SVG_HEADER = f'width="24" height="24" viewBox="0 0 24 24" fill="{FILL_COLOR}" stroke="{STROKE_COLOR}" stroke-width="{STROKE_WIDTH}" stroke-linecap="round" stroke-linejoin="round"'

# --- Navigation & Core Icons ---

# Logo: A stylized, clean basketball hoop and net
ICON_LOGO_NBA = f"""<svg {SVG_HEADER}><circle cx="12" cy="12" r="10"></circle><path d="M19.13 5.09A5 5 0 0 1 12 2c-1.4 0-2.8.6-4 1.5"></path><path d="M21.17 13.42a6 6 0 0 0-3.6-1.42 6 6 0 0 0-3.84 1.23"></path><path d="M11.39 17.99a5 5 0 0 1-5-5a5 5 0 0 1 1.64-3.9"></path><path d="M4.4 14.1a10 10 0 0 0 4.78 6.3"></path><path d="M19.13 18.91a10 10 0 0 0 2.04-6.91"></path><path d="M2.83 8.58a10 10 0 0 0 1.57 5.52"></path></svg>"""

ICON_NAV_HOME = f"""<svg {SVG_HEADER}><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>"""

ICON_NAV_CALENDAR = f"""<svg {SVG_HEADER}><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>"""

ICON_NAV_BRAIN = f"""<svg {SVG_HEADER}><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"></path><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z"></path></svg>"""

ICON_NAV_CHART = f"""<svg {SVG_HEADER}><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>"""

ICON_NAV_TRADE = f"""<svg {SVG_HEADER}><path d="M21 12V7H5a2 2 0 0 1 0-4h14v4"></path><path d="M3 5v14a2 2 0 0 0 2 2h16v-5"></path><path d="M18 12a2 2 0 0 0 0 4h4v-4Z"></path></svg>"""

ICON_NAV_PORTFOLIO = f"""<svg {SVG_HEADER}><rect x="2" y="7" width="20" height="14" rx="2" ry="2"></rect><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"></path></svg>"""

ICON_NAV_WALLET = f"""<svg {SVG_HEADER}><path d="M21 12V7H5a2 2 0 0 1 0-4h14v4"></path><path d="M3 5v14a2 2 0 0 0 2 2h16v-5"></path><path d="M18 12a2 2 0 0 0 0 4h4v-4Z"></path></svg>"""

# --- Domain Specific Icons ---

ICON_BASKETBALL = ICON_LOGO_NBA

ICON_ANALYTICS = f"""<svg {SVG_HEADER}><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>"""

ICON_BETTING = f"""<svg {SVG_HEADER}><circle cx="12" cy="12" r="10"></circle><path d="M16 8h-6a2 2 0 1 0 0 4h4a2 2 0 1 1 0 4H8"></path><line x1="12" y1="18" x2="12" y2="22"></line><line x1="12" y1="2" x2="12" y2="6"></line></svg>"""

ICON_TARGET = f"""<svg {SVG_HEADER}><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>"""

ICON_CHECK_CIRCLE = f"""<svg {SVG_HEADER}><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>"""

ICON_LIGHTBULB = f"""<svg {SVG_HEADER}><line x1="9" y1="18" x2="15" y2="18"></line><line x1="10" y1="22" x2="14" y2="22"></line><path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 16.5 8 4.5 4.5 0 0 0 12 3.5 4.5 4.5 0 0 0 7.5 8c0 1.54.81 2.9 2.08 3.61.55.32.89.91 1.03 1.54"></path></svg>"""

ICON_TRASH = f"""<svg {SVG_HEADER}><path d="M3 6h18"></path><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>"""

ICON_REFRESH = f"""<svg {SVG_HEADER}><path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"></path><path d="M21 3v5h-5"></path><path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"></path><path d="M8 16H3v5"></path></svg>"""

ICON_SEARCH = f"""<svg {SVG_HEADER}><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>"""

ICON_ARROW_LEFT = f"""<svg {SVG_HEADER}><line x1="19" y1="12" x2="5" y2="12"></line><polyline points="12 19 5 12 12 5"></polyline></svg>"""

ICON_ARROW_RIGHT = f"""<svg {SVG_HEADER}><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>"""

ICON_SEARCH_SMALL = f"""<svg {SVG_HEADER.replace('width="24"', 'width="16"').replace('height="24"', 'height="16"')}><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>"""

ICON_TOAST_SUCCESS = f"""<svg {SVG_HEADER} stroke="#52A878"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>"""

ICON_TOAST_WARNING = f"""<svg {SVG_HEADER} stroke="#D97706"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>"""

ICON_TOAST_ERROR = f"""<svg {SVG_HEADER} stroke="#EF4444"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>"""

# --- Aliases for backward compatibility or specific contexts ---
ICON_HOME = ICON_NAV_HOME
ICON_WALLET = ICON_NAV_WALLET
ICON_PORTFOLIO = ICON_NAV_PORTFOLIO
ICON_CALENDAR = ICON_NAV_CALENDAR
ICON_CLOCK = f"""<svg {SVG_HEADER}><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>"""
ICON_BRAIN = ICON_NAV_BRAIN
ICON_CHART_BAR = ICON_NAV_CHART

# --- New Action Icons (Anthropic Style - Lucide Based) ---
# All icons use standard 24x24 viewbox, stroke width 1.5, and current stroke color

ICON_ANALYZE = (
    f"""<svg {SVG_HEADER}><path d="M22 12h-4l-3 9L9 3l-3 9H2"></path></svg>"""
)

ICON_WAND = f"""<svg {SVG_HEADER}><path d="M15 4V2"></path><path d="M15 16v-2"></path><path d="M8 9h2"></path><path d="M20 9h2"></path><path d="M17.8 11.8 19 13"></path><path d="M15 9h0"></path><path d="M17.8 6.2 19 5"></path><path d="M3 21l9-9"></path><path d="M12.2 6.2 11 5"></path></svg>"""

ICON_CHECK = (
    f"""<svg {SVG_HEADER}><polyline points="20 6 9 17 4 12"></polyline></svg>"""
)

ICON_BAR_CHART = ICON_NAV_CHART

ICON_PLAY = (
    f"""<svg {SVG_HEADER}><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>"""
)

ICON_SUMMARY = f"""<svg {SVG_HEADER}><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect><path d="M8 12h8"></path><path d="M8 16h8"></path></svg>"""

ICON_FORCE_UPDATE = ICON_REFRESH
ICON_CLIPBOARD = ICON_SUMMARY  # Alias

ICON_TRASH = f"""<svg {SVG_HEADER}><path d="M3 6h18"></path><path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"></path><path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>"""

ICON_HOURGLASS = ICON_CLOCK
ICON_SCROLL = ICON_SUMMARY
