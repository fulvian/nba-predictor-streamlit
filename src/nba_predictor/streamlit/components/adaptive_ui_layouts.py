"""
Adaptive UI Layouts - Task 3.4.1
Sistema di layout adattivo con Context7 compliance e superpoteri DevStream.
Implementa layout responsive, breakpoints intelligenti e componenti modulari.
"""

import streamlit as st
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json
from datetime import datetime
import logging
from functools import wraps
import weakref

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BreakpointType(Enum):
    """Tipi di breakpoints responsive"""
    MOBILE = "mobile"
    TABLET = "tablet"
    DESKTOP = "desktop"
    WIDE = "wide"

class LayoutPattern(Enum):
    """Pattern di layout X7 Compliant"""
    GRID = "grid"
    CARDS = "cards"
    LIST = "list"
    DASHBOARD = "dashboard"
    SIDEBAR = "sidebar"
    STACK = "stack"

class ViewMode(Enum):
    """Modalità di visualizzazione"""
    COMPACT = "compact"
    COMFORTABLE = "comfortable"
    SPACIOUS = "spacious"
    AUTO = "auto"

@dataclass
class BreakpointConfig:
    """Configurazione breakpoint"""
    name: BreakpointType
    min_width: int
    max_width: Optional[int] = None
    columns: int = 12
    gutters: str = "medium"
    component_size: str = "normal"

@dataclass
class LayoutConstraints:
    """Vincoli di layout"""
    min_columns: int = 1
    max_columns: int = 12
    preferred_column_width: int = 200
    max_column_width: int = 400
    min_gap: int = 8
    max_gap: int = 32
    aspect_ratios: List[str] = field(default_factory=lambda: ["16:9", "4:3", "1:1"])

@dataclass
class AdaptiveComponent:
    """Componente adattivo"""
    component_id: str
    widget_factory: Callable
    min_width: int = 200
    max_width: int = 600
    preferred_columns: int = 1
    flexible: bool = True
    priority: int = 5
    collapse_on_mobile: bool = False
    expand_on_wide: bool = False

class AdaptiveLayoutManager:
    """Manager layout adattivo X7 Compliant"""

    def __init__(self):
        # Singleton pattern X7 Compliant
        if hasattr(AdaptiveLayoutManager, '_instance'):
            self._breakpoints = AdaptiveLayoutManager._instance._breakpoints
            self._components = AdaptiveLayoutManager._instance._components
            self._layout_history = AdaptiveLayoutManager._instance._layout_history
            self._metrics = AdaptiveLayoutManager._instance._metrics
            self._config = AdaptiveLayoutManager._instance._config
            self._callbacks = AdaptiveLayoutManager._instance._callbacks
            return

        AdaptiveLayoutManager._instance = self
        self._breakpoints = self._initialize_breakpoints()
        self._components: Dict[str, AdaptiveComponent] = {}
        self._layout_history: List[Dict[str, Any]] = []
        self._metrics: Dict[str, Any] = defaultdict(int)
        self._config = self._load_config()
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)

        # Start monitoring thread
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._monitor_layout_changes, daemon=True)
        self._monitoring_thread.start()

        logger.info("🚀 AdaptiveLayoutManager initialized with X7 compliance")

    def _initialize_breakpoints(self) -> Dict[BreakpointType, BreakpointConfig]:
        """Inizializza breakpoints responsive Context7 compliant"""
        return {
            BreakpointType.MOBILE: BreakpointConfig(
                name=BreakpointType.MOBILE,
                min_width=0,
                max_width=640,
                columns=4,
                gutters="small",
                component_size="small"
            ),
            BreakpointType.TABLET: BreakpointConfig(
                name=BreakpointType.TABLET,
                min_width=641,
                max_width=1024,
                columns=8,
                gutters="medium",
                component_size="normal"
            ),
            BreakpointType.DESKTOP: BreakpointConfig(
                name=BreakpointType.DESKTOP,
                min_width=1025,
                max_width=1440,
                columns=12,
                gutters="medium",
                component_size="normal"
            ),
            BreakpointType.WIDE: BreakpointConfig(
                name=BreakpointType.WIDE,
                min_width=1441,
                max_width=None,
                columns=12,
                gutters="large",
                component_size="large"
            )
        }

    def _load_config(self) -> Dict[str, Any]:
        """Carica configurazione con superpoteri Context7"""
        return {
            'enable_auto_resizing': True,
            'animation_duration_ms': 300,
            'responsive_images': True,
            'touch_friendly_buttons': True,
            'accessibility_first': True,
            'performance_mode': 'balanced',
            'cache_layouts': True,
            'breakpoint_detection': 'automatic',
            'theme_adaptation': True
        }

    def get_current_breakpoint(self) -> BreakpointType:
        """Rileva breakpoint corrente con Context7 detection"""
        try:
            # Get viewport width from session state or use Streamlit's detection
            viewport_width = self._get_viewport_width()

            for breakpoint_type, config in self._breakpoints.items():
                if config.min_width <= viewport_width <= (config.max_width or float('inf')):
                    return breakpoint_type

            return BreakpointType.DESKTOP

        except Exception as e:
            logger.warning(f"Breakpoint detection error: {e}")
            return BreakpointType.DESKTOP

    def _get_viewport_width(self) -> int:
        """Ottieni viewport width con fallback intelligente"""
        try:
            # Try to get from session state
            if hasattr(st, 'session_state') and 'viewport_width' in st.session_state:
                return st.session_state.viewport_width

            # Use Streamlit's built-in responsive detection
            # Since we can't directly get viewport, use content area width
            return 1024  # Default desktop width

        except Exception:
            return 1024

    def register_component(self, component: AdaptiveComponent):
        """Registra componente adattivo"""
        self._components[component.component_id] = component
        self._metrics['components_registered'] += 1

        # Emit event for DevStream integration
        self._emit_layout_event("component_registered", {
            'component_id': component.component_id,
            'min_width': component.min_width,
            'priority': component.priority
        })

        logger.info(f"📱 Component registered: {component.component_id}")

    def create_adaptive_layout(self, component_ids: List[str],
                              pattern: LayoutPattern = LayoutPattern.GRID,
                              constraints: Optional[LayoutConstraints] = None) -> None:
        """Crea layout adattivo X7 Compliant"""
        if constraints is None:
            constraints = LayoutConstraints()

        current_breakpoint = self.get_current_breakpoint()
        breakpoint_config = self._breakpoints[current_breakpoint]

        # Get registered components
        components = [self._components[cid] for cid in component_ids if cid in self._components]
        components.sort(key=lambda x: x.priority)

        if not components:
            st.warning("No components found for adaptive layout")
            return

        # Create layout based on pattern
        layout_result = self._create_pattern_layout(components, pattern, breakpoint_config, constraints)

        # Store layout history
        self._layout_history.append({
            'timestamp': datetime.now(),
            'breakpoint': current_breakpoint.value,
            'pattern': pattern.value,
            'components_count': len(components),
            'layout_type': layout_result['type']
        })

        # Update metrics
        self._metrics['layouts_created'] += 1
        self._metrics[f'layout_{pattern.value}'] += 1
        self._metrics[f'breakpoint_{current_breakpoint.value}'] += 1

    def _create_pattern_layout(self, components: List[AdaptiveComponent],
                              pattern: LayoutPattern,
                              breakpoint_config: BreakpointConfig,
                              constraints: LayoutConstraints) -> Dict[str, Any]:
        """Crea layout specifico pattern con Context7 best practices"""

        if pattern == LayoutPattern.GRID:
            return self._create_grid_layout(components, breakpoint_config, constraints)
        elif pattern == LayoutPattern.CARDS:
            return self._create_cards_layout(components, breakpoint_config, constraints)
        elif pattern == LayoutPattern.LIST:
            return self._create_list_layout(components, breakpoint_config, constraints)
        elif pattern == LayoutPattern.DASHBOARD:
            return self._create_dashboard_layout(components, breakpoint_config, constraints)
        elif pattern == LayoutPattern.SIDEBAR:
            return self._create_sidebar_layout(components, breakpoint_config, constraints)
        elif pattern == LayoutPattern.STACK:
            return self._create_stack_layout(components, breakpoint_config, constraints)
        else:
            return self._create_grid_layout(components, breakpoint_config, constraints)

    def _create_grid_layout(self, components: List[AdaptiveComponent],
                           breakpoint_config: BreakpointConfig,
                           constraints: LayoutConstraints) -> Dict[str, Any]:
        """Crea layout grid responsive Context7 compliant"""

        # Calculate optimal columns based on components and breakpoint
        optimal_columns = self._calculate_optimal_columns(
            components, breakpoint_config, constraints
        )

        # Create columns with no gap for seamless layout (Context7 best practice)
        cols = st.columns(optimal_columns, gap=None)

        rendered_components = 0
        for i, component in enumerate(components):
            col_index = i % optimal_columns
            with cols[col_index]:
                try:
                    # Apply responsive sizing
                    component_size = self._get_responsive_component_size(component, breakpoint_config)

                    # Render component with adaptive container
                    with st.container(height=None, width=None):
                        component.widget_factory(
                            size=component_size,
                            responsive=True,
                            breakpoint=breakpoint_config.name.value
                        )

                    rendered_components += 1

                except Exception as e:
                    st.error(f"Error rendering component {component.component_id}: {e}")
                    logger.error(f"Component render error: {e}")

        return {
            'type': 'grid',
            'columns': optimal_columns,
            'components_rendered': rendered_components,
            'breakpoint': breakpoint_config.name.value
        }

    def _create_cards_layout(self, components: List[AdaptiveComponent],
                            breakpoint_config: BreakpointConfig,
                            constraints: LayoutConstraints) -> Dict[str, Any]:
        """Crea layout cards responsive"""

        # Calculate cards per row
        cards_per_row = self._calculate_cards_per_row(components, breakpoint_config)

        # Create columns with appropriate gap
        gap_size = self._map_gap_to_streamlit(breakpoint_config.gutters)
        cols = st.columns(cards_per_row, gap=gap_size)

        rendered_components = 0
        for i, component in enumerate(components):
            col_index = i % cards_per_row
            with cols[col_index]:
                try:
                    # Create card container
                    with st.container(border=True, height=None):
                        # Card header with component name
                        st.markdown(f"**{component.component_id}**")

                        # Responsive component
                        component_size = self._get_responsive_component_size(component, breakpoint_config)
                        component.widget_factory(
                            size=component_size,
                            card=True,
                            responsive=True
                        )

                    rendered_components += 1

                except Exception as e:
                    st.error(f"Error rendering card {component.component_id}: {e}")
                    logger.error(f"Card render error: {e}")

        return {
            'type': 'cards',
            'cards_per_row': cards_per_row,
            'components_rendered': rendered_components,
            'breakpoint': breakpoint_config.name.value
        }

    def _create_list_layout(self, components: List[AdaptiveComponent],
                          breakpoint_config: BreakpointConfig,
                          constraints: LayoutConstraints) -> Dict[str, Any]:
        """Crea layout list responsive"""

        rendered_components = 0

        for component in components:
            try:
                # Create list item container
                with st.container(border=True):
                    # List item header
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"**{component.component_id}**")

                    with col2:
                        if st.button("⋮", key=f"list_menu_{component.component_id}", help="Options"):
                            st.sidebar.write(f"Options for {component.component_id}")

                    # Component content
                    component.widget_factory(
                        size='compact',
                        list_item=True,
                        responsive=True
                    )

                rendered_components += 1

            except Exception as e:
                st.error(f"Error rendering list item {component.component_id}: {e}")
                logger.error(f"List render error: {e}")

        return {
            'type': 'list',
            'components_rendered': rendered_components,
            'breakpoint': breakpoint_config.name.value
        }

    def _create_dashboard_layout(self, components: List[AdaptiveComponent],
                               breakpoint_config: BreakpointConfig,
                               constraints: LayoutConstraints) -> Dict[str, Any]:
        """Crea layout dashboard responsive"""

        # Dashboard pattern: main content + sidebar on larger screens
        current_breakpoint = self.get_current_breakpoint()

        if current_breakpoint in [BreakpointType.MOBILE, BreakpointType.TABLET]:
            # Mobile/Tablet: single column
            return self._create_single_column_dashboard(components, breakpoint_config)
        else:
            # Desktop/Wide: two-column layout
            return self._create_two_column_dashboard(components, breakpoint_config)

    def _create_single_column_dashboard(self, components: List[AdaptiveComponent],
                                    breakpoint_config: BreakpointConfig) -> Dict[str, Any]:
        """Dashboard a colonna singola per mobile/tablet"""

        rendered_components = 0

        for component in components:
            try:
                # Priority components get full width
                if component.priority <= 3:
                    with st.container(border=True, height=200):
                        st.markdown(f"### {component.component_id}")
                        component.widget_factory(size='full', responsive=True)
                else:
                    # Other components in normal containers
                    with st.container(border=True):
                        st.markdown(f"**{component.component_id}**")
                        component.widget_factory(size='normal', responsive=True)

                rendered_components += 1

            except Exception as e:
                st.error(f"Error rendering dashboard item {component.component_id}: {e}")
                logger.error(f"Dashboard render error: {e}")

        return {
            'type': 'dashboard_single',
            'components_rendered': rendered_components,
            'breakpoint': breakpoint_config.name.value
        }

    def _create_two_column_dashboard(self, components: List[AdaptiveComponent],
                                   breakpoint_config: BreakpointConfig) -> Dict[str, Any]:
        """Dashboard a due colonne per desktop/wide"""

        # Split components by priority
        high_priority = [c for c in components if c.priority <= 2]
        normal_priority = [c for c in components if 2 < c.priority <= 7]
        low_priority = [c for c in components if c.priority > 7]

        rendered_components = 0

        # Main content area (70% width)
        main_col, sidebar_col = st.columns([7, 3], gap="large")

        with main_col:
            # High priority components
            for component in high_priority[:2]:  # Top 2 high priority
                try:
                    with st.container(border=True, height=250):
                        st.markdown(f"### {component.component_id}")
                        component.widget_factory(size='large', responsive=True)
                    rendered_components += 1
                except Exception as e:
                    logger.error(f"Dashboard main render error: {e}")

            # Normal priority in grid
            if normal_priority:
                normal_cols = st.columns(2, gap="medium")
                for i, component in enumerate(normal_priority[:4]):
                    col_index = i % 2
                    with normal_cols[col_index]:
                        try:
                            with st.container(border=True, height=200):
                                st.markdown(f"**{component.component_id}**")
                                component.widget_factory(size='normal', responsive=True)
                            rendered_components += 1
                        except Exception as e:
                            logger.error(f"Dashboard normal render error: {e}")

        with sidebar_col:
            # Sidebar components
            all_sidebar_components = normal_priority[4:] + low_priority

            for component in all_sidebar_components[:5]:  # Top 5 sidebar
                try:
                    with st.container(border=True, height=150):
                        st.markdown(f"**{component.component_id}**")
                        component.widget_factory(size='compact', responsive=True)
                    rendered_components += 1
                except Exception as e:
                    logger.error(f"Dashboard sidebar render error: {e}")

        return {
            'type': 'dashboard_two_column',
            'components_rendered': rendered_components,
            'breakpoint': breakpoint_config.name.value
        }

    def _create_sidebar_layout(self, components: List[AdaptiveComponent],
                              breakpoint_config: BreakpointConfig,
                              constraints: LayoutConstraints) -> Dict[str, Any]:
        """Crea layout con sidebar responsive"""

        current_breakpoint = self.get_current_breakpoint()

        if current_breakpoint == BreakpointType.MOBILE:
            # Mobile: sidebar becomes top navigation
            return self._create_mobile_sidebar_layout(components, breakpoint_config)
        else:
            # Desktop: traditional sidebar
            return self._create_desktop_sidebar_layout(components, breakpoint_config)

    def _create_mobile_sidebar_layout(self, components: List[AdaptiveComponent],
                                     breakpoint_config: BreakpointConfig) -> Dict[str, Any]:
        """Layout sidebar mobile-friendly"""

        rendered_components = 0

        # Top navigation bar
        with st.container(border=True):
            selected_tab = st.selectbox(
                "Navigation:",
                options=[c.component_id for c in components],
                key="mobile_sidebar_nav"
            )

        # Render selected component
        for component in components:
            if component.component_id == selected_tab:
                try:
                    component.widget_factory(size='full', responsive=True)
                    rendered_components += 1
                except Exception as e:
                    st.error(f"Error rendering mobile sidebar {component.component_id}: {e}")
                    logger.error(f"Mobile sidebar render error: {e}")
                break

        return {
            'type': 'sidebar_mobile',
            'components_rendered': rendered_components,
            'breakpoint': breakpoint_config.name.value
        }

    def _create_desktop_sidebar_layout(self, components: List[AdaptiveComponent],
                                     breakpoint_config: BreakpointConfig) -> Dict[str, Any]:
        """Layout sidebar desktop"""

        # Sidebar (25%) + Main content (75%)
        sidebar_col, main_col = st.columns([1, 3], gap="large")

        rendered_components = 0

        with sidebar_col:
            st.markdown("### 📱 Navigation")

            for component in components:
                try:
                    if st.button(f"🔹 {component.component_id}",
                               key=f"sidebar_{component.component_id}",
                               use_container_width=True):
                        st.session_state[f'selected_component'] = component.component_id
                    rendered_components += 1
                except Exception as e:
                    logger.error(f"Desktop sidebar button error: {e}")

        with main_col:
            selected_id = st.session_state.get('selected_component', components[0].component_id)

            for component in components:
                if component.component_id == selected_id:
                    try:
                        component.widget_factory(size='large', responsive=True)
                        rendered_components += 1
                    except Exception as e:
                        st.error(f"Error rendering main content {component.component_id}: {e}")
                        logger.error(f"Main content render error: {e}")
                    break

        return {
            'type': 'sidebar_desktop',
            'components_rendered': rendered_components,
            'breakpoint': breakpoint_config.name.value
        }

    def _create_stack_layout(self, components: List[AdaptiveComponent],
                           breakpoint_config: BreakpointConfig,
                           constraints: LayoutConstraints) -> Dict[str, Any]:
        """Crea layout stack responsive"""

        rendered_components = 0

        for component in components:
            try:
                # Expandable stack items
                with st.expander(f"📋 {component.component_id}", expanded=component.priority <= 3):
                    component.widget_factory(
                        size=self._get_responsive_component_size(component, breakpoint_config),
                        stack=True,
                        responsive=True
                    )

                rendered_components += 1

            except Exception as e:
                st.error(f"Error rendering stack item {component.component_id}: {e}")
                logger.error(f"Stack render error: {e}")

        return {
            'type': 'stack',
            'components_rendered': rendered_components,
            'breakpoint': breakpoint_config.name.value
        }

    def _calculate_optimal_columns(self, components: List[AdaptiveComponent],
                                  breakpoint_config: BreakpointConfig,
                                  constraints: LayoutConstraints) -> int:
        """Calcola numero ottimale di colonne"""

        available_width = breakpoint_config.max_width or 1200
        total_min_width = sum(c.min_width for c in components)

        # Simple calculation: fit as many columns as possible
        max_columns = min(
            breakpoint_config.columns,
            max(1, available_width // max(c.min_width for c in components)),
            constraints.max_columns
        )

        return max_columns

    def _calculate_cards_per_row(self, components: List[AdaptiveComponent],
                               breakpoint_config: BreakpointConfig) -> int:
        """Calcola carte per riga responsive"""

        current_breakpoint = self.get_current_breakpoint()

        if current_breakpoint == BreakpointType.MOBILE:
            return 1
        elif current_breakpoint == BreakpointType.TABLET:
            return 2
        elif current_breakpoint == BreakpointType.DESKTOP:
            return 3
        else:  # WIDE
            return 4

    def _get_responsive_component_size(self, component: AdaptiveComponent,
                                     breakpoint_config: BreakpointConfig) -> str:
        """Ottieni dimensione componente responsive"""

        # Base size from breakpoint config
        base_size = breakpoint_config.component_size

        # Adjust based on component characteristics
        if component.max_width > 500:
            return 'large'
        elif component.min_width < 200:
            return 'small'
        else:
            return base_size

    def _map_gap_to_streamlit(self, gap: str) -> str:
        """Mappa gap a Streamlit gap values"""
        gap_mapping = {
            'small': 'small',
            'medium': 'medium',
            'large': 'large'
        }
        return gap_mapping.get(gap, 'medium')

    def add_layout_callback(self, event: str, callback: Callable):
        """Aggiunge callback per eventi layout"""
        self._callbacks[event].append(callback)

    def _emit_layout_event(self, event: str, data: Dict[str, Any]):
        """Emetti evento layout"""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error(f"Layout callback error: {e}")

    def _monitor_layout_changes(self):
        """Monitora cambiamenti layout in background"""
        while self._monitoring_active:
            try:
                # Check for layout changes
                current_breakpoint = self.get_current_breakpoint()

                # Update metrics
                self._metrics['layout_checks'] += 1

                # Sleep for monitoring interval
                time.sleep(5)

            except Exception as e:
                logger.error(f"Layout monitoring error: {e}")
                time.sleep(10)

    def get_layout_metrics(self) -> Dict[str, Any]:
        """Ottieni metriche layout"""
        return {
            'total_layouts': len(self._layout_history),
            'components_registered': self._metrics['components_registered'],
            'current_breakpoint': self.get_current_breakpoint().value,
            'breakpoint_usage': {
                bp.value: self._metrics[f'breakpoint_{bp.value}']
                for bp in BreakpointType
            },
            'pattern_usage': {
                pattern.value: self._metrics[f'layout_{pattern.value}']
                for pattern in LayoutPattern
            },
            'latest_layouts': self._layout_history[-5:] if self._layout_history else []
        }

    def optimize_for_performance(self):
        """Ottimizza layout per performance"""
        logger.info("🚀 Optimizing layout for performance")

        # Clear old layout history
        if len(self._layout_history) > 100:
            self._layout_history = self._layout_history[-50:]

        # Update metrics
        self._metrics['performance_optimizations'] += 1

    def __del__(self):
        """Cleanup"""
        self._monitoring_active = False
        if hasattr(self, '_monitoring_thread') and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=1)

# Global instance with X7 Singleton pattern
_layout_manager_instance: Optional[AdaptiveLayoutManager] = None
_layout_lock = threading.Lock()

def get_adaptive_layout_manager() -> AdaptiveLayoutManager:
    """Ottieni istanza global adaptive layout manager"""
    global _layout_manager_instance

    if _layout_manager_instance is None:
        with _layout_lock:
            if _layout_manager_instance is None:
                _layout_manager_instance = AdaptiveLayoutManager()

    return _layout_manager_instance

# Decorators for easy component registration
def adaptive_component(component_id: str, **kwargs):
    """Decorator per registrazione componente adattivo"""
    def decorator(func):
        component = AdaptiveComponent(
            component_id=component_id,
            widget_factory=func,
            **kwargs
        )

        # Auto-register with layout manager
        manager = get_adaptive_layout_manager()
        manager.register_component(component)

        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper
    return decorator

# Context7 Responsive utilities
def responsive_columns(component_ids: List[str],
                     breakpoints: Optional[Dict[str, int]] = None,
                     pattern: LayoutPattern = LayoutPattern.GRID) -> None:
    """Crea colonne responsive con Context7 best practices"""
    manager = get_adaptive_layout_manager()
    manager.create_adaptive_layout(component_ids, pattern=pattern)

def responsive_container(child_func: Callable,
                         responsive_config: Optional[Dict[str, Any]] = None) -> None:
    """Crea container responsive"""
    try:
        with st.container():
            child_func()
    except Exception as e:
        logger.error(f"Responsive container error: {e}")
        st.error(f"Container error: {e}")

if __name__ == "__main__":
    # Test adaptive layout manager
    print("🚀 Testing Adaptive Layout Manager...")

    manager = get_adaptive_layout_manager()

    # Test breakpoint detection
    current_bp = manager.get_current_breakpoint()
    print(f"✅ Current breakpoint: {current_bp.value}")

    # Test metrics
    metrics = manager.get_layout_metrics()
    print(f"✅ Layout metrics retrieved")
    print(f"   - Total layouts: {metrics['total_layouts']}")
    print(f"   - Components registered: {metrics['components_registered']}")
    print(f"   - Current breakpoint: {metrics['current_breakpoint']}")

    print("🎉 Adaptive Layout Manager test completed!")