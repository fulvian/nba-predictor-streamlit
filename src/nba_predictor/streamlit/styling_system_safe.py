#!/usr/bin/env python3
"""
Sistema di styling sicuro e compatibile per la dashboard NBA
Versione semplificata per evitare conflitti CSS
"""

import streamlit as st
from typing import Dict, Any, Optional

class NBAStylingSafe:
    """Sistema di styling sicuro e minimale"""

    # Colori base sicuri
    COLORS = {
        'primary': '#1f77b4',
        'success': '#2ca02c',
        'warning': '#ff7f0e',
        'danger': '#d62728',
        'info': '#17a2b8',
        'light': '#f8f9fa',
        'dark': '#343a40'
    }

    @staticmethod
    def get_safe_css() -> str:
        """CSS minimale e sicuro che non crea conflitti"""
        return """
        <style>
        /* Stili base sicuri */
        .safe-metric-card {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }

        .safe-header {
            background: linear-gradient(135deg, #1f77b4 0%, #2e5cb8 100%);
            color: white;
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 1rem;
            text-align: center;
        }

        .safe-section-header {
            color: #343a40;
            border-bottom: 2px solid #1f77b4;
            padding-bottom: 0.5rem;
            margin: 1.5rem 0 1rem 0;
        }

        .safe-value-strong {
            background: #d4edda;
            color: #155724;
            padding: 1rem;
            border-radius: 8px;
            font-weight: 600;
            text-align: center;
            margin: 1rem 0;
        }

        .safe-value-moderate {
            background: #fff3cd;
            color: #856404;
            padding: 1rem;
            border-radius: 8px;
            font-weight: 600;
            text-align: center;
            margin: 1rem 0;
        }

        .safe-value-weak {
            background: #d1ecf1;
            color: #0c5460;
            padding: 1rem;
            border-radius: 8px;
            font-weight: 600;
            text-align: center;
            margin: 1rem 0;
        }

        .safe-metric {
            font-size: 1.5rem;
            font-weight: 700;
            color: #1f77b4;
        }

        .safe-metric-label {
            font-size: 0.9rem;
            color: #6c757d;
            margin-bottom: 0.5rem;
        }

        .safe-step-indicator {
            display: flex;
            align-items: center;
            margin: 0.5rem 0;
        }

        .safe-step-circle {
            width: 30px;
            height: 30px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 600;
            margin-right: 0.75rem;
            color: white;
            font-size: 0.9rem;
        }

        .safe-step-completed {
            background: #2ca02c;
        }

        .safe-step-active {
            background: #1f77b4;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }

        .safe-step-pending {
            background: #6c757d;
        }

        /* Media queries per responsività */
        @media (max-width: 768px) {
            .safe-header {
                padding: 1rem;
            }

            .safe-metric-card {
                padding: 0.75rem;
                margin: 0.25rem 0;
            }
        }
        </style>
        """

    @staticmethod
    def inject_safe_css():
        """Inietta CSS sicuro nell'applicazione"""
        st.markdown(NBAStylingSafe.get_safe_css(), unsafe_allow_html=True)

    @staticmethod
    def create_safe_header(title: str, subtitle: str = None) -> str:
        """Crea header sicuro"""
        subtitle_html = f"<p style='margin: 0.5rem 0; opacity: 0.9;'>{subtitle}</p>" if subtitle else ""
        return f"""
        <div class="safe-header">
            <h1 style="margin: 0; font-size: 2rem; font-weight: 700;">🏀 {title}</h1>
            {subtitle_html}
        </div>
        """

    @staticmethod
    def create_safe_metric_card(title: str, value: str, delta: str = None) -> str:
        """Crea card metrica sicura"""
        delta_html = f"<div style='color: #2ca02c; font-weight: 600; margin-top: 0.5rem; font-size: 0.9rem;'>{delta}</div>" if delta else ""
        return f"""
        <div class="safe-metric-card">
            <div class="safe-metric-label">{title}</div>
            <div class="safe-metric">{value}</div>
            {delta_html}
        </div>
        """

    @staticmethod
    def create_safe_section_header(title: str, description: str = None) -> str:
        """Crea header sezione sicuro"""
        desc_html = f"<p style='margin: 0.5rem 0; color: #6c757d; font-size: 0.9rem;'>{description}</p>" if description else ""
        return f"""
        <div class="safe-section-header">
            <h2 style="margin: 0; font-size: 1.3rem; font-weight: 600;">{title}</h2>
            {desc_html}
        </div>
        """

    @staticmethod
    def create_safe_value_indicator(edge: float) -> str:
        """Crea indicatore value sicuro"""
        if edge >= 5.0:
            return f'<div class="safe-value-strong">🔥 STRONG VALUE ({edge:+.1f}%)</div>'
        elif edge >= 2.0:
            return f'<div class="safe-value-moderate">⭐ MODERATE VALUE ({edge:+.1f}%)</div>'
        else:
            return f'<div class="safe-value-weak">💡 WEAK VALUE ({edge:+.1f}%)</div>'

    @staticmethod
    def create_safe_step_indicator(step: int, current_step: int, title: str) -> str:
        """Crea indicatore step sicuro"""
        if step < current_step:
            step_class = "safe-step-completed"
            step_text = "✓"
        elif step == current_step:
            step_class = "safe-step-active"
            step_text = str(step)
        else:
            step_class = "safe-step-pending"
            step_text = str(step)

        return f"""
        <div class="safe-step-indicator">
            <div class="safe-step-circle {step_class}">{step_text}</div>
            <div style="font-weight: 600; color: #343a40;">{title}</div>
        </div>
        """

# Funzioni helper per uso rapido
def apply_safe_styling():
    """Applica styling sicuro all'applicazione"""
    NBAStylingSafe.inject_safe_css()

def create_safe_hero_header(title: str, subtitle: str = None) -> str:
    """Crea header eroe sicuro"""
    return NBAStylingSafe.create_safe_header(title, subtitle)

def create_safe_section_header(title: str, description: str = None) -> str:
    """Crea header sezione sicuro"""
    return NBAStylingSafe.create_safe_section_header(title, description)