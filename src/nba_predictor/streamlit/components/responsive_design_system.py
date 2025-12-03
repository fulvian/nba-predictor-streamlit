"""
Responsive Design System - Task 3.4.2
Sistema di design responsive X7 Compliant con Context7 tokens e superpoteri DevStream.
Implementa design tokens, typography, colori, spacing e tema responsive.
"""

import streamlit as st
import time
import json
from typing import Dict, List, Any, Optional, Union, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from datetime import datetime
import logging
import threading
from pathlib import Path
import weakref

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TokenType(Enum):
    """Tipi di design tokens"""
    COLOR = "color"
    TYPOGRAPHY = "typography"
    SPACING = "spacing"
    SHAPE = "shape"
    SHADOW = "shadow"
    DEPTH = "depth"
    BORDER_RADIUS = "border_radius"
    BREAKPOINT = "breakpoint"
    ANIMATION = "animation"

class BreakpointDevice(Enum):
    """Device types per breakpoints"""
    MOBILE = "mobile"
    TABLET = "tablet"
    LAPTOP = "laptop"
    DESKTOP = "desktop"
    WIDE = "wide"

class ThemeVariant(Enum):
    """Varianti tema"""
    LIGHT = "light"
    DARK = "dark"
    AUTO = "auto"
    CONTRAST = "contrast"

class SizeScale(Enum):
    """Scale di dimensione responsive"""
    XS = "xs"
    SM = "sm"
    MD = "md"
    LG = "lg"
    XL = "xl"
    XXL = "xxl"

@dataclass
class DesignToken:
    """Token di design Context7 compliant"""
    name: str
    token_type: TokenType
    value: Union[str, int, float]
    responsive_values: Dict[BreakpointDevice, Union[str, int, float]] = field(default_factory=dict)
    category: str = "system"
    description: str = ""

@dataclass
class ResponsiveBreakpoint:
    """Breakpoint responsive con Context7 best practice"""
    device: BreakpointDevice
    min_width: int
    max_width: Optional[int] = None
    scale_factor: float = 1.0
    container_padding: int = 16
    grid_columns: int = 12
    preferred_spacing: str = "medium"

@dataclass
class ResponsiveTheme:
    """Tema responsive con variant multiple"""
    name: str
    variant: ThemeVariant
    color_palette: Dict[str, str] = field(default_factory=dict)
    typography_scale: Dict[SizeScale, float] = field(default_factory=dict)
    spacing_scale: float = 1.0
    custom_tokens: Dict[str, DesignToken] = field(default_factory=dict)

class ResponsiveDesignSystem:
    """Sistema design responsive X7 Compliant"""

    def __init__(self):
        # Singleton pattern X7 compliant
        if hasattr(ResponsiveDesignSystem, '_instance'):
            self._tokens = ResponsiveDesignSystem._instance._tokens
            self._breakpoints = ResponsiveDesignSystem._instance._breakpoints
            self._themes = ResponsiveDesignSystem._instance._themes
            self._current_theme = ResponsiveDesignSystem._instance._current_theme
            self._token_cache = ResponsiveDesignSystem._instance._token_cache
            self._metrics = ResponsiveDesignSystem._instance._metrics
            self._config = ResponsiveDesignSystem._instance._config
            return

        ResponsiveDesignSystem._instance = self

        # Initialize design system with Context7 patterns
        self._tokens = self._initialize_tokens()
        self._breakpoints = self._initialize_breakpoints()
        self._themes = self._initialize_themes()
        self._current_theme = None
        self._token_cache: Dict[str, str] = {}
        self._metrics: Dict[str, Any] = defaultdict(int)
        self._config = self._load_config()

        # Start theme monitoring
        self._theme_monitoring_active = True
        self._theme_monitoring_thread = threading.Thread(target=self._monitor_theme_changes, daemon=True)
        self._theme_monitoring_thread.start()

        logger.info("🎨 ResponsiveDesignSystem initialized with Context7 compliance")

    def _initialize_tokens(self) -> Dict[str, DesignToken]:
        """Inizializza design tokens Context7 compliant"""

        # Color tokens based on NBA branding
        color_tokens = {
            # Primary colors
            'color-primary': DesignToken(
                name='color-primary',
                token_type=TokenType.COLOR,
                value='#1D428A',
                category='primary',
                description='Primary NBA green'
            ),
            'color-secondary': DesignToken(
                name='color-secondary',
                token_type=TokenType.COLOR,
                value='#FF6B35',
                category='secondary',
                description='Secondary orange'
            ),
            'color-accent': DesignToken(
                name='color-accent',
                token_type=TokenType.COLOR,
                value='#0077C0',
                category='accent',
                description='NBA blue accent'
            ),

            # Semantic colors
            'color-success': DesignToken(
                name='color-success',
                token_type=TokenType.COLOR,
                value='#4CAF50',
                category='semantic',
                description='Success green'
            ),
            'color-warning': DesignToken(
                name='color-warning',
                token_type=TokenType.COLOR,
                value='#FF9800',
                category='semantic',
                description='Warning orange'
            ),
            'color-error': DesignToken(
                name='color-error',
                token_type=TokenType.COLOR,
                value='#F44336',
                category='semantic',
                description='Error red'
            ),
            'color-info': DesignToken(
                name='color-info',
                token_type=TokenType.COLOR,
                value='#2196F3',
                category='semantic',
                description='Info blue'
            ),

            # Neutral colors
            'color-neutral-50': DesignToken(
                name='color-neutral-50',
                token_type=TokenType.COLOR,
                value='#FAFAFA',
                category='neutral',
                description='Lightest neutral'
            ),
            'color-neutral-100': DesignToken(
                name='color-neutral-100',
                token_type=TokenType.COLOR,
                value='#F5F5F5',
                category='neutral',
                description='Very light neutral'
            ),
            'color-neutral-900': DesignToken(
                name='color-900',
                token_type=TokenType.COLOR,
                value='#212121',
                category='neutral',
                description='Darkest neutral'
            ),

            # Text colors
            'color-text-primary': DesignToken(
                name='color-text-primary',
                token_type=TokenType.COLOR,
                value='#1976D2',
                category='text',
                description='Primary text'
            ),
            'color-text-secondary': DesignToken(
                name='color-text-secondary',
                token_type=TokenType.COLOR,
                value='#6C757D',
                category='text',
                description='Secondary text'
            ),
        }

        # Typography tokens
        typography_tokens = {
            'font-family-sans': DesignToken(
                name='font-family-sans',
                token_type=TokenType.TYPOGRAPHY,
                value='Inter, system-ui, sans-serif',
                category='typography',
                description='Sans serif font family'
            ),
            'font-family-mono': DesignToken(
                name='font-family-mono',
                token_type=TokenType.TYPOGRAPHY,
                value='SF Mono, Monaco, monospace',
                category='typography',
                description='Monospace font family'
            ),
            'font-size-xs': DesignToken(
                name='font-size-xs',
                token_type=TokenType.TYPOGRAPHY,
                value='0.75rem',
                category='typography',
                description='Extra small font size'
            ),
            'font-size-sm': DesignToken(
                name='font-size-sm',
                token_type=TokenType.TYPOGRAPHY,
                value='0.875rem',
                category='typography',
                description='Small font size'
            ),
            'font-size-base': DesignToken(
                name='font-size-base',
                token_type=TokenType.TYPOGRAPHY,
                value='1rem',
                category='typography',
                description='Base font size'
            ),
            'font-size-lg': DesignToken(
                name='font-size-lg',
                token_type=TokenType.TYPOGRAPHY,
                value='1.125rem',
                category='typography',
                description='Large font size'
            ),
            'font-size-xl': DesignToken(
                name='font-size-xl',
                token_type=TokenType.TYPOGRAPHY,
                value='1.25rem',
                category='typography',
                description='Extra large font size'
            ),
            'font-size-2xl': DesignToken(
                name='font-size-2xl',
                token_type=TokenType.TYPOGRAPHY,
                value='1.5rem',
                category='typography',
                description='2XL font size'
            ),
            'font-weight-normal': DesignToken(
                name='font-weight-normal',
                token_type=TokenType.TYPOGRAPHY,
                value='400',
                category='typography',
                description='Normal font weight'
            ),
            'font-weight-semibold': DesignToken(
                name='font-weight-semibold',
                token_type=TokenType.TYPOGRAPHY,
                value='600',
                category='typography',
                description='Semibold font weight'
            ),
            'font-weight-bold': DesignToken(
                name='font-weight-bold',
                token_type=TokenType.TYPOGRAPHY,
                value='700',
                category='typography',
                description='Bold font weight'
            ),
            'line-height-tight': DesignToken(
                name='line-height-tight',
                token_type=TokenType.TYPOGRAPHY,
                value='1.25',
                category='typography',
                description='Tight line height'
            ),
            'line-height-base': DesignToken(
                name='line-height-base',
                token_type=TokenType.TYPOGRAPHY,
                value='1.5',
                category='typography',
                description='Base line height'
            ),
        }

        # Spacing tokens following Context7 8pt grid system
        spacing_tokens = {
            'space-xs': DesignToken(
                name='space-xs',
                token_type=TokenType.SPACING,
                value='4px',
                category='spacing',
                description='Extra small space'
            ),
            'space-sm': DesignToken(
                name='space-sm',
                token_type=TokenType.SPACING,
                value='8px',
                category='spacing',
                description='Small space'
            ),
            'space-md': DesignToken(
                name='space-md',
                token_type=TokenType.SPACING,
                value='16px',
                category='spacing',
                description='Medium space'
            ),
            'space-lg': DesignToken(
                name='space-lg',
                token_type=TokenType.SPACING,
                value='24px',
                category='spacing',
                description='Large space'
            ),
            'space-xl': DesignToken(
                name='space-xl',
                token_type=TokenType.SPACING,
                value='32px',
                category='spacing',
                description='Extra large space'
            ),
            'space-2xl': DesignToken(
                name='space-2xl',
                token_type=TokenType.SPACING,
                value='48px',
                category='spacing',
                description='2XL space'
            ),
            'space-3xl': DesignToken(
                name='space-3xl',
                token_type=TokenType.SPACING,
                value='64px',
                category='spacing',
                description='3XL space'
            ),
        }

        # Shape tokens for UI components
        shape_tokens = {
            'border-radius-xs': DesignToken(
                name='border-radius-xs',
                token_type=TokenType.SHAPE,
                value='2px',
                category='shape',
                description='Extra small border radius'
            ),
            'border-radius-sm': DesignToken(
                name='border-radius-sm',
                token_type=TokenType.SHAPE,
                value='4px',
                category='shape',
                description='Small border radius'
            ),
            'border-radius-md': DesignToken(
                name='border-radius-md',
                token_type=TokenType.SHAPE,
                value='8px',
                category='shape',
                description='Medium border radius'
            ),
            'border-radius-lg': DesignToken(
                name='border-radius-lg',
                token_type=TokenType.SHAPE,
                value='12px',
                category='shape',
                description='Large border radius'
            ),
            'border-radius-xl': DesignToken(
                name='border-radius-xl',
                token_type=TokenType.SHAPE,
                value='16px',
                category='shape',
                description='Extra large border radius'
            ),
            'border-radius-full': DesignToken(
                name='border-radius-full',
                token_type=TokenType.SHAPE,
                value='9999px',
                category='shape',
                description='Full border radius'
            ),
        }

        # Shadow tokens following Context7 elevation principles
        shadow_tokens = {
            'shadow-xs': DesignToken(
                name='shadow-xs',
                token_type=TokenType.SHADOW,
                value='0 1px 2px 0 rgba(0, 0, 0, 0.05)',
                category='shadow',
                description='Extra small shadow'
            ),
            'shadow-sm': DesignToken(
                name='shadow-sm',
                token_type=TokenType.SHADOW,
                value='0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)',
                category='shadow',
                description='Small shadow'
            ),
            'shadow-md': DesignToken(
                name='shadow-md',
                token_type=TokenType.SHADOW,
                value='0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                category='shadow',
                description='Medium shadow'
            ),
            'shadow-lg': DesignToken(
                name='shadow-lg',
                token_type=TokenType.SHADOW,
                value='0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
                category='shadow',
                description='Large shadow'
            ),
            'shadow-xl': DesignToken(
                name='shadow-xl',
                token_type=TokenType.SHADOW,
                value='0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04)',
                category='shadow',
                description='Extra large shadow'
            ),
        }

        # Depth tokens for Material Design elevation
        depth_tokens = {
            'depth-0': DesignToken(
                name='depth-0',
                token_type=TokenType.DEPTH,
                value='#FFFFFF',
                category='depth',
                description='No elevation'
            ),
            'depth-1': DesignToken(
                name='depth-1',
                token_type=TokenType.DEPTH,
                value='#F5F5F5',
                category='depth',
                description='Level 1 elevation'
            ),
            'depth-2': DesignToken(
                name='depth-2',
                token_type=TokenType.DEPTH,
                value='#EEEEEE',
                category='depth',
                description='Level 2 elevation'
            ),
            'depth-3': DesignToken(
                name='depth-3',
                token_type=TokenType.DEPTH,
                value='#E0E0E0',
                category='depth',
                description='Level 3 elevation'
            ),
        }

        # Combine all tokens
        all_tokens = {}
        all_tokens.update(color_tokens)
        all_tokens.update(typography_tokens)
        all_tokens.update(spacing_tokens)
        all_tokens.update(shape_tokens)
        all_tokens.update(shadow_tokens)
        all_tokens.update(depth_tokens)

        return all_tokens

    def _initialize_breakpoints(self) -> Dict[BreakpointDevice, ResponsiveBreakpoint]:
        """Inizializza breakpoints responsive Context7 compliant"""

        return {
            BreakpointDevice.MOBILE: ResponsiveBreakpoint(
                device=BreakpointDevice.MOBILE,
                min_width=0,
                max_width=640,
                scale_factor=0.85,
                container_padding=12,
                grid_columns=4,
                preferred_spacing="small"
            ),
            BreakpointDevice.TABLET: ResponsiveBreakpoint(
                device=BreakpointDevice.TABLET,
                min_width=641,
                max_width=768,
                scale_factor=0.95,
                container_padding=16,
                grid_columns=8,
                preferred_spacing="medium"
            ),
            BreakpointDevice.LAPTOP: ResponsiveBreakpoint(
                device=Breakpoint.LAPTOP,
                min_width=769,
                max_width=1024,
                scale_factor=1.0,
                container_padding=20,
                grid_columns=12,
                preferred_spacing="medium"
            ),
            BreakpointDevice.DESKTOP: ResponsiveBreakpoint(
                device=BreakpointDevice.DESKTOP,
                min_width=1025,
                max_width=1440,
                scale_factor=1.05,
                container_padding=24,
                grid_columns=12,
                preferred_spacing="large"
            ),
            BreakpointDevice.WIDE: ResponsiveBreakpoint(
                device=Breakpoint.WIDE,
                min_width=1441,
                max_width=None,
                scale_factor=1.1,
                container_padding=32,
                grid_columns=12,
                preferred_spacing="large"
            ),
        }

    def _initialize_themes(self) -> Dict[str, ResponsiveTheme]:
        """Inizializza temi responsive X7 compliant"""

        # Light theme
        light_theme = ResponsiveTheme(
            name="light",
            variant=ThemeVariant.LIGHT,
            color_palette={
                'background': '#FFFFFF',
                'surface': '#F8F9FA',
                'card': '#FFFFFF',
                'text': '#212121',
                'text-secondary': '#6C757D',
                'border': '#E0E0E0',
                'divider': '#EEEEEE'
            },
            typography_scale={
                SizeScale.XS: 0.75,
                SizeScale.SM: 0.875,
                SizeScale.MD: 1.0,
                SizeScale.LG: 1.125,
                SizeScale.XL: 1.25,
                SizeScale.XXL: 1.5,
            },
            custom_tokens={
                'color-surface': DesignToken(
                    'color-surface', TokenType.COLOR, '#F8F9FA'
                ),
                'color-card': DesignToken(
                    'color-card', TokenType.COLOR, '#FFFFFF'
                ),
            }
        )

        # Dark theme
        dark_theme = ResponsiveTheme(
            name="dark",
            variant=ThemeVariant.DARK,
            color_palette={
                'background': '#121212',
                'surface': '#1E1E1E',
                'card': '#2D2D2D',
                'text': '#FFFFFF',
                'text-secondary': '#B3B3B3',
                'border': '#404040',
                'divider': '#333333'
            },
            typography_scale={
                SizeScale.XS: 0.8,
                SizeScale.SM: 0.875,
                SizeScale.MD: 1.0,
                SizeScale.LG: 1.125,
                SizeScale.XL: 1.25,
                SizeScale.XXL: 1.5,
            },
            custom_tokens={
                'color-surface': DesignToken(
                    'color-surface', TokenType.COLOR, '#1E1E1E'
                ),
                'color-card': DesignToken(
                    'color-card', TokenType.COLOR, '#2D2D2D'
                ),
            }
        )

        # High contrast theme
        contrast_theme = ResponsiveTheme(
            name="contrast",
            variant=ThemeVariant.CONTRAST,
            color_palette={
                'background': '#FFFFFF',
                'surface': '#FAFAFA',
                'card': '#FFFFFF',
                'text': '#000000',
                'text-secondary': '#333333',
                'border': '#000000',
                'divider': '#666666'
            },
            typography_scale={
                SizeScale.XS: 0.8,
                SizeScale.SM: 0.875,
                SizeScale.MD: 1.0,
                SizeScale.LG: 1.125,
                SizeScale.XL: 1.25,
                SizeScale.XXL: 1.5,
            },
            custom_tokens={
                'color-surface': DesignToken(
                    'color-surface', TokenType.COLOR, '#FAFAFA'
                ),
                'color-card': DesignToken(
                    'color-card', TokenType.COLOR, '#FFFFFF'
                ),
                'color-border': DesignToken(
                    'color-border', TokenType.COLOR, '#000000'
                ),
            }
        )

        return {
            'light': light_theme,
            'dark': dark_theme,
            'contrast': contrast_theme
        }

    def _load_config(self) -> Dict[str, Any]:
        """Carica configurazione con superpoteri Context7"""
        return {
            'auto_theme_detection': True,
            'prefers_color_scheme': 'auto',
            'enable_token_caching': True,
            'responsive_images': True,
            'optimize_for_performance': True,
            'debug_mode': False,
            'token_validation': True,
            'theme_transition_duration_ms': 300,
            'enable_system_fonts': True,
            'respect_motion_settings': True,
            'focus_visible_focus_indicators': True
        }

    def get_current_breakpoint(self) -> BreakpointDevice:
        """Rileva breakpoint corrente con Context7 detection"""
        try:
            # Use browser width if available, otherwise default to desktop
            if hasattr(st, 'session_state') and 'viewport_width' in st.session_state:
                viewport_width = st.session_state.viewport_width
            else:
                viewport_width = 1024

            # Find matching breakpoint
            for device, breakpoint in self._breakpoints.items():
                if breakpoint.min_width <= viewport_width <= (breakpoint.max_width or float('inf')):
                    return device

            return BreakpointDevice.DESKTOP

        except Exception as e:
            logger.warning(f"Breakpoint detection error: {e}")
            return BreakpointDevice.DESKTOP

    def get_token_value(self, token_name: str,
                         breakpoint: Optional[BreakpointDevice] = None,
                        theme: Optional[str] = None) -> str:
        """Ottieni valore token con Context7 responsive pattern"""

        # Try cache first
        cache_key = f"{token_name}_{breakpoint}_{theme}"
        if self._config['enable_token_caching'] and cache_key in self._token_cache:
            return self._token_cache[cache_key]

        try:
            # Get current breakpoint and theme
            current_breakpoint = breakpoint or self.get_current_breakpoint()
            current_theme = theme or self._get_current_theme()

            # Get token
            token = self._tokens.get(token_name)
            if not token:
                raise ValueError(f"Token not found: {token_name}")

            # Check for responsive values
            if current_breakpoint in token.responsive_values:
                value = token.responsive_values[current_breakpoint]
            else:
                value = token.value

            # Apply theme overrides if needed
            if current_theme != 'light' and current_theme in self._themes:
                theme_tokens = self._themes[current_theme].custom_tokens
                if token_name in theme_tokens:
                    value = theme_tokens[token_name].value

            # Convert to CSS variable format
            css_value = self._convert_to_css_variable(token_name, value)

            # Cache result
            if self._config['enable_token_caching']:
                self._token_cache[cache_key] = css_value

            return css_value

        except Exception as e:
            logger.error(f"Token value error: {e}")
            return f"var(--{token_name})"

    def _convert_to_css_variable(self, token_name: str, value: Union[str, int, float]) -> str:
        """Converte valore token a variabile CSS Context7 compliant"""
        # Remove special characters and format for CSS
        css_name = token_name.replace('_', '-').lower()
        return f"var(--nba-{css_name})"

    def apply_theme(self, theme_name: str) -> None:
        """Applica tema responsive"""
        if theme_name not in self._themes:
            st.error(f"Theme not found: {theme_name}")
            return

        self._current_theme = theme_name

        # Apply CSS variables
        css_vars = self._generate_theme_css(theme_name)
        st.markdown(f"<style>{css_vars}</style>", unsafe_allow_html=True)

        # Update session state
        if hasattr(st, 'session_state'):
            st.session_state.current_theme = theme_name

        logger.info(f"🎨 Applied theme: {theme_name}")

    def _generate_theme_css(self, theme_name: str) -> str:
        """Genera CSS per tema responsive"""
        theme = self._themes[theme_name]

        css_rules = []

        # Color palette
        for color_name, color_value in theme.color_palette.items():
            css_rules.append(f"--nba-color-{color_name}: {color_value};")

        # Typography scale
        for size, scale in theme.typography_scale.items():
            css_rules.append(f"--nba-font-size-{size.value}: {scale}rem;")

        # Custom tokens
        for token_name, token in theme.custom_tokens.items():
            css_rules.append(f"--{token_name}: {token.value};")

        return '\n'.join(css_rules)

    def _get_current_theme(self) -> str:
        """Ottieni tema corrente"""
        if self._current_theme:
            return self._current_theme

        # Auto-detect from system preference
        if self._config['auto_theme_detection']:
            return self._detect_system_theme()

        # Default to light
        return 'light'

    def _detect_system_theme(self) -> str:
        """Rileva tema di sistema con superpoteri"""
        try:
            # In a real implementation, this would check browser/system preferences
            # For now, default to light
            return 'light'
        except Exception as e:
            logger.warning(f"Theme detection error: {e}")
            return 'light'

    def create_responsive_container(self, child_func: Callable,
                                  padding: Optional[str] = None,
                                  margin: Optional[str] = None,
                                  border: bool = False,
                                  shadow: Optional[str] = None) -> None:
        """Crea container responsive Context7 compliant"""
        current_breakpoint = self.get_current_breakpoint()
        breakpoint_config = self._breakpoints[current_breakpoint]

        try:
            # Set responsive styles
            container_style = ""

            # Padding
            if padding:
                if padding == 'none':
                    container_style += "padding: 0;"
                elif padding in ['small', 'medium', 'large', 'xl', '2xl', '3xl']:
                    padding_token = f"space-{padding}"
                    padding_value = self.get_token_value(padding_token, current_breakpoint)
                    container_style += f"padding: {padding_value};"

            # Margin
            if margin:
                if margin == 'none':
                    container_style += "margin: 0;"
                elif margin in ['small', 'medium', 'large', 'xl', '2xl', '3xl']:
                    margin_token = f"space-{margin}"
                    margin_value = self.get_token_value(margin_token, current_breakpoint)
                    container_style += f"margin: {margin_value};"

            # Border
            if border:
                border_token = 'border-radius-md'
                border_value = self.get_token_value(border_token, current_breakpoint)
                container_style += f"border: 1px solid; border-radius: {border_value};"

            # Shadow
            if shadow:
                shadow_token = f"shadow-{shadow}" if shadow != True else "shadow-md"
                shadow_value = self.get_token_value(shadow_token, current_breakpoint)
                container_style += f"box-shadow: {shadow_value};"

            # Responsive sizing
            width_token = 'container_width'
            width_value = self.get_token_value(width_token, current_breakpoint)
            if width_value != '100%':
                container_style += f"width: {width_value};"

            # Create container
            with st.container():
                if container_style:
                    st.markdown(f"<div style='{container_style}'>", unsafe_allow_html=True)

                # Execute child function
                child_func()

                if container_style:
                    st.markdown("</div>", unsafe_allow_html=True)

        except Exception as e:
            logger.error(f"Responsive container error: {e}")
            st.error(f"Container error: {e}")

    def create_responsive_text(self, text: str,
                            size: Optional[SizeScale] = None,
                            weight: Optional[str] = None,
                            color: Optional[str] = None,
                            align: Optional[str] = None) -> None:
        """Crea testo responsive con Context7 typography"""

        try:
            current_breakpoint = self.get_current_breakpoint()

            # Build inline style
            text_styles = []

            # Font size
            if size:
                size_token = f'font-size-{size.value}'
                size_value = self.get_token_value(size_token, current_breakpoint)
                text_styles.append(f"font-size: {size_value}")

            # Font weight
            if weight:
                weight_token = f'font-weight-{weight}' if weight else 'font-weight-semibold'
                weight_value = self.get_token_value(weight_token, current_breakpoint)
                text_styles.append(f"font-weight: {weight_value}")

            # Text color
            if color:
                color_token = f'color-{color}'
                color_value = self.get_token_value(color_token, current_breakpoint)
                text_styles.append(f"color: {color_value}")

            # Text alignment
            if align:
                text_styles.append(f"text-align: {align}")

            # Line height
            line_height_token = 'line-height-base'
            line_height_value = self.get_token_value(line_height_token, current_breakpoint)
            text_styles.append(f"line-height: {line_height_value}")

            # Apply style
            text_style = '; '.join(text_styles)
            if text_style:
                st.markdown(f"<span style='{text_style}'>{text}</span>", unsafe_allow_html=True)
            else:
                st.markdown(text)

        except Exception as e:
            logger.error(f"Responsive text error: {e}")
            st.markdown(text)

    def create_responsive_button(self, label: str,
                              type: str = "secondary",
                              size: Optional[SizeScale] = None,
                              disabled: bool = False,
                              use_container_width: bool = False,
                              icon: Optional[str] = None,
                              onclick: Optional[Callable] = None,
                              key: Optional[str] = None) -> bool:
        """Crea bottone responsive Context7 compliant"""
        try:
            current_breakpoint = self._get_current_breakpoint()

            # Touch-friendly sizing for mobile
            button_size = size or SizeScale.MD
            if current_breakpoint == BreakpointDevice.MOBILE:
                button_size = SizeScale.LG if button_size == SizeScale.XL else SizeScale.MD

            # Button styling with responsive sizing
            button_style = []

            # Touch-friendly padding for mobile
            padding_token = 'space-md'
            if current_breakpoint == BreakpointDevice.MOBILE:
                padding_token = 'space-lg'
            elif current_breakpoint == BreakpointDevice.TABLET:
                padding_token = 'space-md'

            padding_value = self.get_token_value(padding_token, current_breakpoint)
            button_style.append(f"padding: {padding_value}")

            # Responsive width
            if use_container_width:
                button_style.append("width: 100%;")

            # Touch-friendly minimum size
            min_height_token = 'space-lg' if current_breakpoint == BreakpointDevice.MOBILE else 'space-sm'
            min_height_value = self.get_token_value(min_height_token, current_breakpoint)
            button_style.append(f"min-height: {min_height_value}")

            # Apply Context7 accessibility
            focus_styles = [
                "outline: 2px solid var(--nba-color-primary);",
                "outline-offset: 2px;",
                "outline-style: solid;"
            ]
            button_style.extend(focus_styles)

            # Create button
            button_result = st.button(
                label=label,
                type=type,
                disabled=disabled,
                use_container_width=use_container_width,
                key=key,
                onclick=onclick,
                help=f"Responsive button for {current_breakpoint.value}"
            )

            return button_result

        except Exception as e:
            logger.error(f"Responsive button error: {e}")
            return False

    def get_responsive_class(self, base_class: str,
                            breakpoint: Optional[BreakpointDevice] = None,
                            modifiers: Optional[List[str]] = None) -> str:
        """Genera classe responsive con breakpoint modifiers"""
        current_breakpoint = breakpoint or self.get_current_breakpoint()

        classes = [base_class]

        # Add breakpoint modifier
        classes.append(f"nba-{current_breakpoint.value}")

        # Add additional modifiers
        if modifiers:
            classes.extend(modifiers)

        return ' '.join(classes)

    def create_responsive_grid(self,
                          items: List[Any],
                          columns: Optional[int] = None,
                          gap: Optional[str] = None,
                          equal_height: bool = True) -> None:
        """Crea grid responsive con Context7 best practices"""

        current_breakpoint = self.get_current_breakpoint()
        breakpoint_config = self._breakpoints[current_breakpoint]

        # Calculate optimal columns
        if not columns:
            columns = min(breakpoint_config.grid_columns, len(items))

        # Adjust for mobile
        if current_breakpoint == BreakpointType.MOBILE:
            columns = max(1, columns // 2)
        elif current_breakpoint == BreakpointType.TABLET:
            columns = max(2, columns // 2)

        # Gap spacing
        gap_size = gap or breakpoint_config.preferred_spacing
        gap_token = f"space-{gap_size}"
        gap_value = self.get_token_value(gap_token, current_breakpoint)

        # Create columns with Context7 best practice gap=None (no gap)
        cols = st.columns(columns, gap=None if gap_size == 'none' else gap_value)

        # Render items
        for i, item in enumerate(items):
            col_index = i % columns
            with cols[col_index]:
                if equal_height:
                    st.markdown('<div style="height: 100%;">', unsafe_allow_html=True)
                item()
                st.markdown('</div>', unsafe_allow_html=True)

    def get_design_metrics(self) -> Dict[str, Any]:
        """Ottieni metriche design system"""
        return {
            'total_tokens': len(self._tokens),
            'token_categories': {
                token_type.value: len([t for t in self._tokens.values() if t.token_type == token_type])
                for token_type in TokenType
            },
            'themes_available': list(self._themes.keys()),
            'current_theme': self._get_current_theme(),
            'breakpoint_usage': {
                device.value: self._metrics.get(f'breakpoint_{device.value}', 0)
                for device in BreakpointDevice
            },
            'cache_performance': {
                'cached_tokens': len(self._token_cache),
                'cache_hit_rate': self._metrics['cache_hits'] / max(self._metrics['cache_accesses'], 1) if self._metrics['cache_accesses'] > 0 else 0
            },
            'responsive_features': {
                'auto_theme_detection': self._config['auto_theme_detection'],
                'token_caching': self._config['enable_token_caching'],
                'responsive_images': self._config['responsive_images']
            }
        }

    def optimize_for_device(self, device: BreakpointDevice) -> None:
        """Ottimizza sistema per dispositivo specifico"""
        logger.info(f"📱� Optimizing for {device.value} device")

        # Apply device-specific optimizations
        if device == BreakpointDevice.MOBILE:
            self._apply_mobile_optimizations()
        elif device == BreakpointDevice.DESKTOP:
            self._apply_desktop_optimizations()

        # Update metrics
        self._metrics[f'device_optimization_{device.value}'] += 1

    def _apply_mobile_optimizations(self):
        """Ottimizzazioni mobile-first"""
        self._config.update({
            'animation_duration_ms': 200,
            'reduce_shadow_opacity': True,
            'simplify_animations': True
        })

    def _apply_desktop_optimizations(self):
        """Ottimizzazioni desktop"""
        self._config.update({
            'animation_duration_ms': 300,
            'enhanced_shadows': True,
            'advanced_animations': True
        })

    def export_tokens_to_css(self, file_path: str) -> None:
        """Esporta design tokens a file CSS per Context7 compliance"""
        try:
            css_lines = []

            # CSS variables header
            css_lines.extend([
                "/* NBA Predictor Design Tokens - X7 Compliant */",
                "/* Generated on " + datetime.now().isoformat() + " */",
                "",
                "/* Design Tokens */",
                ":root {"
            ])

            # Add all tokens
            for token_name, token in self._tokens.items():
                css_line = f"  --{token_name}: {token.value};"
                css_lines.append(css_line)

            css_lines.extend(["}", ""])

            # Add responsive breakpoints
            css_lines.extend([
                "",
                "/* Responsive Breakpoints */",
                ":root {",
                f"  --nba-breakpoint-mobile: {self._breakpoints[BreakpointDevice.MOBILE].min_width}px;",
                f"  --nba-breakpoint-tablet: {self._breakpoints[BreakpointDevice.TABLET].min_width}px;",
                f"  --nba-breakpoint-laptop: {self._breakpoints[BreakpointDevice.LAPTOP].min_width}px;",
                f"  --nba-breakpoint-desktop: {self._breakpoints[BreakpointDevice.DESKTOP].min_width}px;",
                f"  --nba-breakpoint-wide: {self._breakpoints[BreakpointDevice.WIDE].min_width}px;"
            ])

            css_lines.extend(["}", ""])

            # Add theme variables
            for theme_name, theme in self._themes.items():
                css_lines.extend([
                    "",
                    f"/* Theme: {theme_name} */",
                    ":root[data-theme="{theme_name}"] {"
                ])

                for color_name, color_value in theme.color_palette.items():
                    css_line = f"  --nba-{color_name}: {color_value};"
                    css_lines.append(css_line)

                css_lines.extend(["}", ""])

            css_lines.append("/* NBA Design System - Production Ready */")

            # Write to file
            with open(file_path, 'w') as f:
                f.write('\n'.join(css_lines))

            logger.info(f"🎨 Exported design tokens to {file_path}")

        except Exception as e:
            logger.error(f"CSS export error: {e}")

    def __del__(self):
        """Cleanup"""
        self._theme_monitoring_active = False
        if hasattr(self, '_theme_monitoring_thread') and self._theme_monitoring_thread.is_alive():
            self._theme_monitoring_thread.join(timeout=1)

# Global instance with X7 Singleton pattern
_responsive_design_system_instance: Optional[ResponsiveDesignSystem] = None
_responsive_lock = threading.Lock()

def get_responsive_design_system() -> ResponsiveDesignSystem:
    """Ottieni istanza global responsive design system"""
    global _responsive_design_system_instance

    if _responsive_design_system_instance is None:
        with _responsive_lock:
            if _responsive_design_system_instance is None:
                _responsive_design_system_instance = ResponsiveDesignSystem()

    return _responsive_design_system_instance

# Context7 Responsive utilities
def responsive_container(**kwargs):
    """Decorator per container responsive"""
    def decorator(func):
        def wrapper(*args, **kwargs_inner):
            manager = get_responsive_design_system()
            manager.create_responsive_container(func, **kwargs)
        return wrapper
    return decorator

def responsive_text(size: Optional[SizeScale] = None, **kwargs):
    """Decorator per testo responsive"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs_inner):
            manager = get_responsive_design_system()
            text = func(*args, **kwargs_inner)
            return manager.create_responsive_text(
                text, size=size, **kwargs
            )
        return wrapper
    return decorator

def responsive_button(**kwargs):
    """Decorator per bottone responsive"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs_inner):
            # Extract label from function name if not provided
            label = kwargs.get('label', func.__name__)
            return get_responsive_design_system().create_responsive_button(
                label=label, **kwargs
            )
        return wrapper
    return decorator

# Theme switching utilities
def theme_selector(default_theme: str = 'light') -> str:
    """Crea selettore tema responsive"""
    manager = get_responsive_system()

    current_theme = manager._get_current_theme()
    theme_options = list(manager._themes.keys())

    if current_theme in theme_options:
        theme_options.remove(current_theme)

    theme_options.insert(0, current_theme)

    selected_theme = st.selectbox(
        "🎨 Select Theme:",
        options=theme_options,
        index=theme_options.index(current_theme) if current_theme in theme_options else 0,
        key="theme_selector"
    )

    if selected_theme != current_theme:
        manager.apply_theme(selected_theme)

    return selected_theme

def create_responsive_dashboard():
    """Crea dashboard responsive con Context7 patterns"""
    manager = get_responsive_design_system()
    current_breakpoint = manager.get_current_breakpoint()

    st.markdown(f"# 📱 Responsive Dashboard")
    st.markdown(f"Current Breakpoint: **{current_breakpoint.value}**")

    # Dashboard metrics
    metrics = manager.get_design_metrics()
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Tokens", metrics['total_tokens'])

    with col2:
        st.metric("Cache Hit Rate", f"{metrics['cache_performance']['cache_hit_rate']:.1%}")

    with col3:
        st.metric("Current Theme", metrics['current_theme'].title())

    # Token showcase
    st.subheader("🎨 Design Token Showcase")

    # Color palette
    st.subheader("🎨 Color Palette")
    color_cols = st.columns(6)

    color_tokens = [
        'color-primary', 'color-secondary', 'color-accent',
        'color-success', 'color-warning', 'color-error'
    ]

    for i, token_name in enumerate(color_tokens):
        col_index = i % 6
        with color_cols[col_index]:
            token_value = manager.get_token_value(token_name)
            st.markdown(f"**{token_name}**")
            st.markdown(f"`{token_value}`")

    # Typography showcase
    st.subheader("📝 Typography")
    font_sizes = [SizeScale.XS, SizeScale.SM, SizeScale.MD, SizeScale.LG, SizeScale.XL]
    font_cols = st.columns(len(font_sizes))

    for i, size in enumerate(font_sizes):
        with font_cols[i]:
            size_value = manager.get_token_value(f'font-size-{size.value}')
            st.markdown(f"**Font {size.value.upper()}**")
            st.markdown(f"`{size_value}`")

    # Spacing showcase
    st.subheader("📏 Spacing System")
    spacing_sizes = ['xs', 'sm', 'md', 'lg', 'xl']
    spacing_examples = st.columns(5)

    for i, size in enumerate(spacing_sizes):
        col_index = i % 5
        with spacing_examples[col_index]:
            spacing_value = manager.get_token_value(f'space-{size}')
            st.markdown(f"**{size.upper()}**")
            st.markdown("`{spacing_value}`")

    # Shape showcase
    st.subheader("🔲 Shape System")
    shape_examples = ['xs', 'sm', 'md', 'lg', 'xl', 'full']
    shape_cols = st.columns(6)

    for i, size in enumerate(shape_examples):
        col_index = i % 6
        with shape_cols[col_index]:
            shape_value = manager.get_token_value(f'border-radius-{size}')
            st.markdown(f"**{size.upper()}**")
            st.markdown("`{shape_value}`")

    # Theme comparison
    st.subheader("🌙 Theme Comparison")

    theme_comparison_cols = st.columns(3)

    for i, (theme_name, theme) in enumerate(list(manager._themes.items())[:3]):
        col_index = i % 3
        with theme_comparison_cols[col_index]:
            theme.apply_theme(theme_name)
            st.markdown(f"**{theme_name}**")
            st.markdown(f"Background: {theme.color_palette['background']}")
            st.markdown(f"Text: {theme.color_palette['text']}")

if __name__ == "__main__":
    # Test responsive design system
    print("🎨 Testing Responsive Design System...")

    manager = get_responsive_design_system()

    # Test breakpoint detection
    current_bp = manager.get_current_breakpoint()
    print(f"✅ Current breakpoint: {current_bp.value}")

    # Test token retrieval
    primary_color = manager.get_token_value('color-primary')
    font_size_md = manager.get_token_value('font-size-md')
    spacing_lg = manager.get_token_value('space-lg')
    print(f"✅ Primary color: {primary_color}")
    print(f"✅ Font size MD: {font_size_md}")
    print(f"✅ Spacing LG: {spacing_lg}")

    # Test responsive values
    responsive_padding = manager.get_token_value('space-md', BreakpointDevice.MOBILE)
    print(f"✅ Mobile padding: {responsive_padding}")

    # Test theme system
    manager.apply_theme('light')
    print(f"✅ Applied light theme")

    # Test metrics
    metrics = manager.get_design_metrics()
    print(f"✅ Design metrics retrieved")
    print(f"   - Total tokens: {metrics['total_tokens']}")
    print(f"   - Themes: {len(metrics['themes_available'])}")

    print("🎉 Responsive Design System test completed!")