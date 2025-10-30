#!/usr/bin/env python3
"""
Sistema di styling professionale per la dashboard NBA
Fornisce colori, layout e componenti visivi coerenti
"""

import streamlit as st
from typing import Dict, Any, Optional
import pandas as pd

class NBAStylingSystem:
    """Sistema di styling centrale per l'applicazione NBA"""

    # Palette colori professionale NBA-inspired
    COLORS = {
        # Colori primari (tema NBA)
        'primary': '#1D428A',        # Blu NBA ufficiale
        'secondary': '#C8102E',      # Rosso NBA
        'accent': '#FFB81C',         # Oro NBA
        'dark': '#0A0E27',           # Blu scuro
        'light': '#F8F9FA',          # Grigio chiaro

        # Colori di stato
        'success': '#28A745',        # Verde successo
        'warning': '#FFC107',        # Giallo attenzione
        'danger': '#DC3545',         # Rosso pericolo
        'info': '#17A2B8',           # Blu informazione

        # Colori per betting
        'value_bet': '#28A745',      # Verde per value bet
        'no_value': '#6C757D',       # Grigio per no value
        'high_edge': '#FF6B35',      # Arancione per edge alto
        'medium_edge': '#FFB81C',    # Oro per edge medio
        'low_edge': '#17A2B8',       # Blu per edge basso

        # Gradienti
        'gradient_primary': 'linear-gradient(135deg, #1D428A 0%, #2E5CB8 100%)',
        'gradient_success': 'linear-gradient(135deg, #28A745 0%, #34CE57 100%)',
        'gradient_warning': 'linear-gradient(135deg, #FFC107 0%, #FFD23F 100%)',
        'gradient_danger': 'linear-gradient(135deg, #DC3545 0%, #E4606D 100%)',
    }

    # Spaziature coerenti
    SPACING = {
        'xs': '0.25rem',
        'sm': '0.5rem',
        'md': '1rem',
        'lg': '1.5rem',
        'xl': '2rem',
        'xxl': '3rem'
    }

    # Border radius
    BORDER_RADIUS = {
        'sm': '4px',
        'md': '8px',
        'lg': '12px',
        'xl': '16px',
        'full': '50%'
    }

    # Shadow effects
    SHADOWS = {
        'sm': '0 1px 3px rgba(0,0,0,0.12), 0 1px 2px rgba(0,0,0,0.24)',
        'md': '0 4px 6px rgba(0,0,0,0.1), 0 2px 4px rgba(0,0,0,0.06)',
        'lg': '0 10px 25px rgba(0,0,0,0.15), 0 6px 10px rgba(0,0,0,0.08)',
        'xl': '0 20px 40px rgba(0,0,0,0.2), 0 10px 20px rgba(0,0,0,0.1)'
    }

    @staticmethod
    def get_css_injection() -> str:
        """Restituisce il CSS completo da iniettare nell'applicazione"""
        return f"""
        <style>
        /* Import font professional */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

        /* Variabili CSS */
        :root {{
            --primary-color: {NBAStylingSystem.COLORS['primary']};
            --secondary-color: {NBAStylingSystem.COLORS['secondary']};
            --accent-color: {NBAStylingSystem.COLORS['accent']};
            --dark-color: {NBAStylingSystem.COLORS['dark']};
            --light-color: {NBAStylingSystem.COLORS['light']};
            --success-color: {NBAStylingSystem.COLORS['success']};
            --warning-color: {NBAStylingSystem.COLORS['warning']};
            --danger-color: {NBAStylingSystem.COLORS['danger']};
            --info-color: {NBAStylingSystem.COLORS['info']};
        }}

        /* Stili globali */
        .stApp {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        }}

        /* Header styles */
        .main-header {{
            background: var(--gradient-primary);
            color: white;
            padding: {NBAStylingSystem.SPACING['lg']} {NBAStylingSystem.SPACING['xl']};
            border-radius: {NBAStylingSystem.BORDER_RADIUS['lg']};
            margin-bottom: {NBAStylingSystem.SPACING['lg']};
            box-shadow: {NBAStylingSystem.SHADOWS['md']};
            text-align: center;
        }}

        /* Card styles */
        .metric-card {{
            background: white;
            border-radius: {NBAStylingSystem.BORDER_RADIUS['lg']};
            padding: {NBAStylingSystem.SPACING['lg']};
            box-shadow: {NBAStylingSystem.SHADOWS['md']};
            border-left: 4px solid var(--primary-color);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }}

        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: {NBAStylingSystem.SHADOWS['lg']};
        }}

        .success-card {{
            border-left-color: var(--success-color);
            background: linear-gradient(135deg, #ffffff 0%, #f8fff9 100%);
        }}

        .warning-card {{
            border-left-color: var(--warning-color);
            background: linear-gradient(135deg, #ffffff 0%, #fffdf5 100%);
        }}

        .danger-card {{
            border-left-color: var(--danger-color);
            background: linear-gradient(135deg, #ffffff 0%, #fff5f5 100%);
        }}

        .info-card {{
            border-left-color: var(--info-color);
            background: linear-gradient(135deg, #ffffff 0%, #f5fdff 100%);
        }}

        /* Button styles */
        .btn-primary {{
            background: var(--gradient-primary);
            color: white;
            border: none;
            padding: {NBAStylingSystem.SPACING['md']} {NBAStylingSystem.SPACING['lg']};
            border-radius: {NBAStylingSystem.BORDER_RADIUS['md']};
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: {NBAStylingSystem.SHADOWS['sm']};
        }}

        .btn-primary:hover {{
            transform: translateY(-1px);
            box-shadow: {NBAStylingSystem.SHADOWS['md']};
        }}

        /* Status indicators */
        .status-badge {{
            display: inline-block;
            padding: {NBAStylingSystem.SPACING['xs']} {NBAStylingSystem.SPACING['sm']};
            border-radius: {NBAStylingSystem.BORDER_RADIUS['full']};
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .status-success {{
            background: var(--success-color);
            color: white;
        }}

        .status-warning {{
            background: var(--warning-color);
            color: var(--dark-color);
        }}

        .status-danger {{
            background: var(--danger-color);
            color: white;
        }}

        .status-info {{
            background: var(--info-color);
            color: white;
        }}

        /* Value bet indicator */
        .value-indicator {{
            padding: {NBAStylingSystem.SPACING['sm']};
            border-radius: {NBAStylingSystem.BORDER_RADIUS['md']};
            font-weight: 600;
            text-align: center;
            margin: {NBAStylingSystem.SPACING['sm']} 0;
        }}

        .value-strong {{
            background: linear-gradient(135deg, #28A745 0%, #20c997 100%);
            color: white;
            box-shadow: {NBAStylingSystem.SHADOWS['md']};
        }}

        .value-moderate {{
            background: linear-gradient(135deg, #FFB81C 0%, #FFC107 100%);
            color: var(--dark-color);
            box-shadow: {NBAStylingSystem.SHADOWS['sm']};
        }}

        .value-weak {{
            background: linear-gradient(135deg, #17A2B8 0%, #20c997 100%);
            color: white;
            box-shadow: {NBAStylingSystem.SHADOWS['sm']};
        }}

        /* Table improvements */
        .dataframe {{
            border-radius: {NBAStylingSystem.BORDER_RADIUS['lg']};
            overflow: hidden;
            box-shadow: {NBAStylingSystem.SHADOWS['md']};
        }}

        .dataframe th {{
            background: var(--primary-color);
            color: white;
            font-weight: 600;
            padding: {NBAStylingSystem.SPACING['md']};
        }}

        .dataframe td {{
            padding: {NBAStylingSystem.SPACING['sm']} {NBAStylingSystem.SPACING['md']};
            border-bottom: 1px solid #e9ecef;
        }}

        .dataframe tr:hover {{
            background: #f8f9fa;
        }}

        /* Step indicators */
        .step-indicator {{
            display: flex;
            align-items: center;
            justify-content: center;
            width: 40px;
            height: 40px;
            border-radius: {NBAStylingSystem.BORDER_RADIUS['full']};
            font-weight: 700;
            font-size: 1.1rem;
            margin-right: {NBAStylingSystem.SPACING['md']};
        }}

        .step-completed {{
            background: var(--success-color);
            color: white;
        }}

        .step-active {{
            background: var(--primary-color);
            color: white;
            box-shadow: {NBAStylingSystem.SHADOWS['md']};
        }}

        .step-inactive {{
            background: #e9ecef;
            color: #6c757d;
        }}

        /* Odds display */
        .odds-display {{
            font-family: 'Courier New', monospace;
            font-weight: 700;
            font-size: 1.1rem;
            padding: {NBAStylingSystem.SPACING['xs']} {NBAStylingSystem.SPACING['sm']};
            border-radius: {NBAStylingSystem.BORDER_RADIUS['sm']};
            background: #f8f9fa;
        }}

        .odds-positive {{
            color: var(--success-color);
            background: #f8fff9;
        }}

        .odds-negative {{
            color: var(--danger-color);
            background: #fff5f5;
        }}

        /* Team scores */
        .team-score {{
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--primary-color);
        }}

        .team-name {{
            font-weight: 600;
            color: var(--dark-color);
            margin-bottom: {NBAStylingSystem.SPACING['xs']};
        }}

        /* Animation classes */
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}

        .animate-slide-in {{
            animation: slideIn 0.5s ease-out;
        }}

        @keyframes pulse {{
            0% {{ transform: scale(1); }}
            50% {{ transform: scale(1.05); }}
            100% {{ transform: scale(1); }}
        }}

        .animate-pulse {{
            animation: pulse 2s infinite;
        }}

        /* Loading states */
        .loading-overlay {{
            background: rgba(255, 255, 255, 0.9);
            backdrop-filter: blur(4px);
        }}

        /* Responsive design */
        @media (max-width: 768px) {{
            .main-header {{
                padding: {NBAStylingSystem.SPACING['md']} {NBAStylingSystem.SPACING['lg']};
            }}

            .metric-card {{
                padding: {NBAStylingSystem.SPACING['md']};
            }}
        }}
        </style>
        """

    @staticmethod
    def inject_css():
        """Inietta il CSS nell'applicazione Streamlit"""
        st.markdown(NBAStylingSystem.get_css_injection(), unsafe_allow_html=True)

    @staticmethod
    def create_metric_card(title: str, value: str, delta: Optional[str] = None,
                         card_type: str = 'default', icon: Optional[str] = None) -> str:
        """Crea una card metrica stilizzata"""
        card_class = f"metric-card {card_type}-card animate-slide-in"

        icon_html = f"<div style='font-size: 2rem; margin-bottom: 0.5rem;'>{icon}</div>" if icon else ""
        delta_html = f"<div style='color: var(--success-color); font-weight: 600; margin-top: 0.5rem;'>{delta}</div>" if delta else ""

        return f"""
        <div class="{card_class}">
            {icon_html}
            <h3 style="margin: 0; color: #6c757d; font-size: 0.9rem; font-weight: 500;">{title}</h3>
            <div style="font-size: 1.8rem; font-weight: 700; color: var(--dark-color); margin: 0.25rem 0;">{value}</div>
            {delta_html}
        </div>
        """

    @staticmethod
    def create_status_badge(status: str, text: str) -> str:
        """Crea un badge di stato"""
        return f'<span class="status-badge status-{status}">{text}</span>'

    @staticmethod
    def create_value_indicator(edge: float, threshold_strong: float = 5.0,
                             threshold_moderate: float = 2.0) -> str:
        """Crea un indicatore visivo per value bet"""
        if edge >= threshold_strong:
            indicator_class = "value-indicator value-strong"
            text = f"🔥 STRONG VALUE ({edge:+.1f}%)"
        elif edge >= threshold_moderate:
            indicator_class = "value-indicator value-moderate"
            text = f"⭐ MODERATE VALUE ({edge:+.1f}%)"
        else:
            indicator_class = "value-indicator value-weak"
            text = f"💡 WEAK VALUE ({edge:+.1f}%)"

        return f'<div class="{indicator_class}">{text}</div>'

    @staticmethod
    def format_odds_display(odds: float) -> str:
        """Formatta le quote con stile appropriato"""
        odds_class = "odds-positive" if odds > 2.0 else "odds-negative"
        return f'<span class="odds-display {odds_class}">{odds:.2f}</span>'

    @staticmethod
    def create_step_indicator(step: int, current_step: int, title: str) -> str:
        """Crea un indicatore di step per il workflow"""
        if step < current_step:
            status_class = "step-completed"
            icon = "✓"
        elif step == current_step:
            status_class = "step-active animate-pulse"
            icon = str(step)
        else:
            status_class = "step-inactive"
            icon = str(step)

        return f"""
        <div style="display: flex; align-items: center; margin: 0.5rem 0;">
            <div class="step-indicator {status_class}">{icon}</div>
            <div style="font-weight: 600; color: var(--dark-color);">{title}</div>
        </div>
        """

    @staticmethod
    def highlight_dataframe(df: pd.DataFrame, color_columns: list = None) -> pd.DataFrame:
        """Applica evidenziazione alle colonne specificate del DataFrame"""
        if color_columns is None:
            color_columns = []

        def highlight_value(val, col):
            if col not in color_columns:
                return ''

            # Logica di evidenziazione basata sul tipo di colonna
            if 'edge' in col.lower():
                if pd.notna(val) and val > 5.0:
                    return 'background-color: #d4edda; color: #155724; font-weight: 600;'
                elif pd.notna(val) and val > 2.0:
                    return 'background-color: #fff3cd; color: #856404; font-weight: 600;'
            elif 'probability' in col.lower():
                if pd.notna(val) and val > 0.6:
                    return 'background-color: #d1ecf1; color: #0c5460; font-weight: 600;'
            elif 'stake' in col.lower():
                if pd.notna(val) and val > 50:
                    return 'background-color: #f8d7da; color: #721c24; font-weight: 600;'

            return ''

        styled_df = df.style.apply(lambda x: [highlight_value(val, col) for col, val in zip(x.index, x)], axis=1)
        return styled_df

# Funzioni helper per uso rapido
def apply_styling():
    """Applica il styling system all'applicazione"""
    NBAStylingSystem.inject_css()

def create_hero_header(title: str, subtitle: str = None) -> str:
    """Crea un header eroe professionale"""
    subtitle_html = f"<p style='margin: 0.5rem 0; opacity: 0.9; font-size: 1.1rem;'>{subtitle}</p>" if subtitle else ""

    return f"""
    <div class="main-header animate-slide-in">
        <h1 style="margin: 0; font-size: 2.5rem; font-weight: 700;">🏀 {title}</h1>
        {subtitle_html}
    </div>
    """

def create_section_header(title: str, icon: str = None, description: str = None) -> str:
    """Crea un header di sezione"""
    icon_html = f"{icon} " if icon else ""
    desc_html = f"<p style='margin: 0.5rem 0; color: #6c757d;'>{description}</p>" if description else ""

    return f"""
    <div style="margin: 2rem 0 1rem 0; border-bottom: 2px solid var(--primary-color); padding-bottom: 0.5rem;">
        <h2 style="margin: 0; color: var(--primary-color); font-weight: 600; font-size: 1.5rem;">
            {icon_html}{title}
        </h2>
        {desc_html}
    </div>
    """