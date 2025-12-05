import base64
import os

# Paths
base_path = "/Users/fulvioventura/.gemini/antigravity/brain/f0a8ab2d-bf79-4ebe-bd70-5dcf18c67268/"
output_file = "src/nba_predictor/streamlit/assets.py"

# Generated Images Map
image_files = {
    "ICON_LOGO_NBA": "icon_logo_nba_1764865362563.png",
    "ICON_BASKETBALL": "icon_basketball_minimal_1764865012067.png",
    "ICON_ANALYTICS": "icon_analytics_minimal_1764865026170.png",
    "ICON_BETTING": "icon_betting_minimal_1764865042613.png",
}

# Manual SVGs (Anthropic Style: Stroke #191919, Stroke-width 1.5, Fill none)
svg_style = 'width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#191919" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"'

svgs = {
    "ICON_HOME": f'<svg {svg_style}><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>',
    "ICON_WALLET": f'<svg {svg_style}><path d="M20 12V8H6a2 2 0 0 1-2-2c0-1.1.9-2 2-2h12v4"></path><path d="M4 6v12a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-6a2 2 0 0 0-2-2H6.9c-.98 0-2 1.22-2 2.5S5.9 14 6.9 14h9"></path></svg>',
    "ICON_PORTFOLIO": f'<svg {svg_style}><rect x="2" y="7" width="20" height="14" rx="2" ry="2"></rect><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"></path></svg>',
    "ICON_CALENDAR": f'<svg {svg_style}><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>',
    "ICON_REFRESH": f'<svg {svg_style}><path d="M23 4v6h-6"></path><path d="M1 20v-6h6"></path><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>',
    "ICON_CLOCK": f'<svg {svg_style}><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>',
    "ICON_SEARCH": f'<svg {svg_style}><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>',
    "ICON_BRAIN": f'<svg {svg_style}><path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"></path><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z"></path></svg>',
    "ICON_ARROW_LEFT": f'<svg {svg_style}><polyline points="15 18 9 12 15 6"></polyline></svg>',
    "ICON_ARROW_RIGHT": f'<svg {svg_style}><polyline points="9 18 15 12 9 6"></polyline></svg>',
    "ICON_CHART_BAR": f'<svg {svg_style}><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>',
    "ICON_TARGET": f'<svg {svg_style}><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>',
    "ICON_LIGHTBULB": f'<svg {svg_style}><line x1="9" y1="18" x2="15" y2="18"></line><line x1="10" y1="22" x2="14" y2="22"></line><path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 16.5 8 4.5 4.5 0 0 0 12 3.5 4.5 4.5 0 0 0 7.5 8c0 1.54.81 2.9 2.08 3.61.55.32.89.91 1.03 1.54"></path><path d="M12 2v2"></path><path d="M12 20v2"></path><path d="M20 12h2"></path><path d="M2 12h2"></path></svg>',
    "ICON_CLIPBOARD": f'<svg {svg_style}><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2"></path><rect x="8" y="2" width="8" height="4" rx="1" ry="1"></rect></svg>',
    "ICON_CHECK_CIRCLE": f'<svg {svg_style}><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>',
    "ICON_HOURGLASS": f'<svg {svg_style}><path d="M5 22h14"></path><path d="M5 2h14"></path><path d="M17 22v-4.172a2 2 0 0 0-.586-1.414L12 12l-4.414 4.414A2 2 0 0 0 7 17.828V22"></path><path d="M7 2v4.172a2 2 0 0 0 .586 1.414L12 12l4.414-4.414A2 2 0 0 0 17 6.172V2"></path></svg>',
    "ICON_SCROLL": f'<svg {svg_style}><path d="M19 20H5a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h11a5 5 0 0 1 5 5v11h-2"></path><path d="M17 21v-8a2 2 0 0 0-2-2H5"></path></svg>',
    "ICON_TRASH": f'<svg {svg_style}><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>',
}

with open(output_file, "w") as f:
    f.write(
        '"""\nAssets for the NBA Predictor Dashboard (Anthropic Style)\nContains Base64 encoded images and SVG strings.\n"""\n\n'
    )

    # Write Base64 Images
    for name, filename in image_files.items():
        try:
            with open(os.path.join(base_path, filename), "rb") as img:
                b64 = base64.b64encode(img.read()).decode("utf-8")
                f.write(f'{name} = "data:image/png;base64,{b64}"\n')
        except Exception as e:
            f.write(f'# Error reading {filename}: {e}\n{name} = ""\n')

    f.write("\n# --- SVG Icons ---\n")
    for name, svg in svgs.items():
        f.write(f"{name} = '{svg}'\n")

print(f"Successfully generated {output_file}")
