"""
Accessibility Features System - Context7 Compliant
Task 3.4.3: Accessibility Features Implementation

Provides comprehensive accessibility features for Streamlit components with:
- WCAG 2.1 AA compliance
- Keyboard navigation support
- Screen reader compatibility
- ARIA labels and descriptions
- Focus management
- Motion reduction
- High contrast support
"""

import streamlit as st
import time
import json
import logging
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
from collections import defaultdict
import threading
import queue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AccessibilityLevel(Enum):
    """Accessibility compliance levels"""
    A = "A"
    AA = "AA"
    AAA = "AAA"

class ScreenReaderType(Enum):
    """Supported screen readers"""
    NVDA = "nvda"
    JAWS = "jaws"
    VOICEOVER = "voiceover"
    TALKBACK = "talkback"
    GENERIC = "generic"

class MotionPreference(Enum):
    """User motion preferences"""
    NO_PREFERENCE = "no-preference"
    REDUCE = "reduce"

class FocusTrapStrategy(Enum):
    """Focus trap strategies"""
    TAB_TRAP = "tab-trap"
    SCOPE_LOCK = "scope-lock"
    MODAL_LOCK = "modal-lock"

@dataclass
class AccessibilityConfig:
    """Accessibility configuration"""
    level: AccessibilityLevel = AccessibilityLevel.AA
    enable_keyboard_navigation: bool = True
    enable_screen_reader_support: bool = True
    enable_high_contrast: bool = False
    enable_large_text: bool = False
    enable_reduced_motion: bool = False
    enable_focus_indicators: bool = True
    enable_aria_labels: bool = True
    enable_descriptions: bool = True
    auto_detect_preferences: bool = True

@dataclass
class FocusElement:
    """Focusable element information"""
    element_id: str
    selector: str
    index: int = 0
    group: Optional[str] = None
    skip: bool = False
    trap: bool = False

@dataclass
class AriaLabel:
    """ARIA label configuration"""
    element_id: str
    label: str
    description: Optional[str] = None
    live_region: Optional[str] = None
    atomic: bool = False
    relevant: str = "additions text"
    busy: bool = False

@dataclass
class KeyboardNavigation:
    """Keyboard navigation configuration"""
    key: str
    action: str
    target: str
    description: str
    prevent_default: bool = True
    stop_propagation: bool = True

class AccessibilityFeaturesManager:
    """
    Main accessibility features manager
    Provides comprehensive accessibility support with Context7 compliance
    """

    def __init__(self, config: Optional[AccessibilityConfig] = None):
        self.config = config or AccessibilityConfig()
        self._focus_elements: Dict[str, FocusElement] = {}
        self._aria_labels: Dict[str, AriaLabel] = {}
        self._keyboard_handlers: Dict[str, List[KeyboardNavigation]] = {}
        self._screen_reader_config: Dict[str, Any] = {}
        self._user_preferences: Dict[str, Any] = {}
        self._focus_history: List[str] = []
        self._current_focus: Optional[str] = None
        self._initialized = False

        # Thread safety
        self._lock = threading.RLock()

        # Initialize features
        self._initialize_accessibility_features()

    def _initialize_accessibility_features(self):
        """Initialize accessibility features"""
        try:
            with self._lock:
                # Detect user preferences if enabled
                if self.config.auto_detect_preferences:
                    self._detect_user_preferences()

                # Initialize screen reader support
                if self.config.enable_screen_reader_support:
                    self._initialize_screen_reader_support()

                # Initialize keyboard navigation
                if self.config.enable_keyboard_navigation:
                    self._initialize_keyboard_navigation()

                # Initialize ARIA support
                if self.config.enable_aria_labels:
                    self._initialize_aria_support()

                # Initialize motion preferences
                self._initialize_motion_preferences()

                # Initialize focus management
                self._initialize_focus_management()

                self._initialized = True

                logger.info("🚀 Accessibility Features Manager initialized with Context7 compliance")
                logger.info(f"   - Compliance Level: {self.config.level.value}")
                logger.info(f"   - Keyboard Navigation: {self.config.enable_keyboard_navigation}")
                logger.info(f"   - Screen Reader Support: {self.config.enable_screen_reader_support}")
                logger.info(f"   - High Contrast: {self.config.enable_high_contrast}")
                logger.info(f"   - Reduced Motion: {self.config.enable_reduced_motion}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize accessibility features: {e}")
            raise

    def _detect_user_preferences(self):
        """Detect user accessibility preferences"""
        try:
            # Detect reduced motion preference
            reduced_motion = self._check_reduced_motion_preference()
            self._user_preferences['reduced_motion'] = reduced_motion

            # Detect high contrast preference
            high_contrast = self._check_high_contrast_preference()
            self._user_preferences['high_contrast'] = high_contrast

            # Detect large text preference
            large_text = self._check_large_text_preference()
            self._user_preferences['large_text'] = large_text

            # Apply detected preferences
            if reduced_motion:
                self.config.enable_reduced_motion = True
            if high_contrast:
                self.config.enable_high_contrast = True
            if large_text:
                self.config.enable_large_text = True

            logger.info(f"   - Detected preferences: {self._user_preferences}")

        except Exception as e:
            logger.warning(f"⚠️ Could not detect user preferences: {e}")

    def _check_reduced_motion_preference(self) -> bool:
        """Check if user prefers reduced motion"""
        # In a real implementation, this would check system preferences
        # For Streamlit, we'll use a session state approach
        return st.session_state.get('prefers_reduced_motion', False)

    def _check_high_contrast_preference(self) -> bool:
        """Check if user prefers high contrast"""
        return st.session_state.get('prefers_high_contrast', False)

    def _check_large_text_preference(self) -> bool:
        """Check if user prefers large text"""
        return st.session_state.get('prefers_large_text', False)

    def _initialize_screen_reader_support(self):
        """Initialize screen reader support"""
        try:
            # Detect screen reader
            screen_reader = self._detect_screen_reader()
            self._screen_reader_config = {
                'type': screen_reader,
                'enabled': True,
                'announcements': [],
                'live_regions': {}
            }

            # Setup screen reader specific features
            if screen_reader != ScreenReaderType.GENERIC:
                self._setup_screen_reader_specific_features(screen_reader)

            logger.info(f"   - Screen Reader: {screen_reader.value}")

        except Exception as e:
            logger.warning(f"⚠️ Could not initialize screen reader support: {e}")

    def _detect_screen_reader(self) -> ScreenReaderType:
        """Detect active screen reader"""
        # In a real implementation, this would detect the actual screen reader
        # For now, we'll use a generic approach
        return ScreenReaderType.GENERIC

    def _setup_screen_reader_specific_features(self, screen_reader: ScreenReaderType):
        """Setup features specific to screen reader type"""
        try:
            if screen_reader == ScreenReaderType.NVDA:
                # NVDA specific configurations
                self._screen_reader_config['nvda_mode'] = True
                self._screen_reader_config['virtual_buffer'] = True

            elif screen_reader == ScreenReaderType.JAWS:
                # JAWS specific configurations
                self._screen_reader_config['jaws_mode'] = True
                self._screen_reader_config['forms_mode'] = True

            elif screen_reader == ScreenReaderType.VOICEOVER:
                # VoiceOver specific configurations
                self._screen_reader_config['voiceover_mode'] = True
                self._screen_reader_config['rotor_enabled'] = True

        except Exception as e:
            logger.warning(f"⚠️ Could not setup screen reader specific features: {e}")

    def _initialize_keyboard_navigation(self):
        """Initialize keyboard navigation"""
        try:
            # Setup common keyboard shortcuts
            default_shortcuts = [
                KeyboardNavigation(
                    key="Tab",
                    action="focus_next",
                    target="focusable",
                    description="Navigate to next focusable element"
                ),
                KeyboardNavigation(
                    key="Shift+Tab",
                    action="focus_previous",
                    target="focusable",
                    description="Navigate to previous focusable element"
                ),
                KeyboardNavigation(
                    key="Enter",
                    action="activate",
                    target="focused",
                    description="Activate focused element"
                ),
                KeyboardNavigation(
                    key="Space",
                    action="select",
                    target="focused",
                    description="Select focused element"
                ),
                KeyboardNavigation(
                    key="Escape",
                    action="close",
                    target="modal",
                    description="Close modal or cancel action"
                )
            ]

            self._keyboard_handlers['global'] = default_shortcuts

            logger.info(f"   - Keyboard shortcuts: {len(default_shortcuts)} configured")

        except Exception as e:
            logger.warning(f"⚠️ Could not initialize keyboard navigation: {e}")

    def _initialize_aria_support(self):
        """Initialize ARIA support"""
        try:
            # Setup common ARIA labels
            common_labels = [
                AriaLabel(
                    element_id="main_navigation",
                    label="Main Navigation",
                    description="Primary navigation menu for the application"
                ),
                AriaLabel(
                    element_id="content_area",
                    label="Main Content",
                    live_region="polite"
                ),
                AriaLabel(
                    element_id="betting_form",
                    label="Betting Form",
                    description="Form to place NBA bets"
                ),
                AriaLabel(
                    element_id="predictions_display",
                    label="Predictions",
                    description="NBA game predictions and analysis"
                )
            ]

            for label in common_labels:
                self._aria_labels[label.element_id] = label

            logger.info(f"   - ARIA labels: {len(common_labels)} configured")

        except Exception as e:
            logger.warning(f"⚠️ Could not initialize ARIA support: {e}")

    def _initialize_motion_preferences(self):
        """Initialize motion preferences"""
        try:
            # Apply reduced motion if enabled
            if self.config.enable_reduced_motion:
                st.markdown("""
                <style>
                *, *::before, *::after {
                    animation-duration: 0.01ms !important;
                    animation-iteration-count: 1 !important;
                    transition-duration: 0.01ms !important;
                }
                </style>
                """, unsafe_allow_html=True)

            logger.info(f"   - Reduced motion: {'enabled' if self.config.enable_reduced_motion else 'disabled'}")

        except Exception as e:
            logger.warning(f"⚠️ Could not initialize motion preferences: {e}")

    def _initialize_focus_management(self):
        """Initialize focus management"""
        try:
            # Apply focus styles if enabled
            if self.config.enable_focus_indicators:
                st.markdown("""
                <style>
                :focus {
                    outline: 3px solid #0066cc !important;
                    outline-offset: 2px !important;
                }

                :focus:not(:focus-visible) {
                    outline: none !important;
                }

                :focus-visible {
                    outline: 3px solid #0066cc !important;
                    outline-offset: 2px !important;
                }

                .skip-link {
                    position: absolute;
                    top: -40px;
                    left: 6px;
                    background: #000;
                    color: #fff;
                    padding: 8px;
                    text-decoration: none;
                    border-radius: 4px;
                    z-index: 10000;
                }

                .skip-link:focus {
                    top: 6px;
                }
                </style>
                """, unsafe_allow_html=True)

            logger.info(f"   - Focus indicators: {'enabled' if self.config.enable_focus_indicators else 'disabled'}")

        except Exception as e:
            logger.warning(f"⚠️ Could not initialize focus management: {e}")

    def register_focusable_element(self,
                                 element_id: str,
                                 selector: str,
                                 group: Optional[str] = None,
                                 index: int = 0,
                                 skip: bool = False,
                                 trap: bool = False):
        """Register a focusable element"""
        try:
            element = FocusElement(
                element_id=element_id,
                selector=selector,
                index=index,
                group=group,
                skip=skip,
                trap=trap
            )

            self._focus_elements[element_id] = element

            logger.debug(f"   - Registered focusable element: {element_id}")

        except Exception as e:
            logger.error(f"❌ Error registering focusable element {element_id}: {e}")

    def add_aria_label(self,
                      element_id: str,
                      label: str,
                      description: Optional[str] = None,
                      live_region: Optional[str] = None,
                      atomic: bool = False,
                      relevant: str = "additions text",
                      busy: bool = False):
        """Add ARIA label to element"""
        try:
            aria_label = AriaLabel(
                element_id=element_id,
                label=label,
                description=description,
                live_region=live_region,
                atomic=atomic,
                relevant=relevant,
                busy=busy
            )

            self._aria_labels[element_id] = aria_label

            logger.debug(f"   - Added ARIA label for: {element_id}")

        except Exception as e:
            logger.error(f"❌ Error adding ARIA label for {element_id}: {e}")

    def add_keyboard_handler(self,
                           component_id: str,
                           key: str,
                           action: str,
                           target: str,
                           description: str,
                           prevent_default: bool = True,
                           stop_propagation: bool = True):
        """Add keyboard handler"""
        try:
            handler = KeyboardNavigation(
                key=key,
                action=action,
                target=target,
                description=description,
                prevent_default=prevent_default,
                stop_propagation=stop_propagation
            )

            if component_id not in self._keyboard_handlers:
                self._keyboard_handlers[component_id] = []

            self._keyboard_handlers[component_id].append(handler)

            logger.debug(f"   - Added keyboard handler for {component_id}: {key} -> {action}")

        except Exception as e:
            logger.error(f"❌ Error adding keyboard handler for {component_id}: {e}")

    def announce_to_screen_reader(self,
                                 message: str,
                                 priority: str = "polite",
                                 timeout: Optional[int] = None):
        """Announce message to screen reader"""
        try:
            if not self.config.enable_screen_reader_support:
                return

            # Add announcement to queue
            announcement = {
                'message': message,
                'priority': priority,
                'timestamp': time.time(),
                'timeout': timeout
            }

            self._screen_reader_config['announcements'].append(announcement)

            # Render live region for announcement
            announcement_id = f"announcement_{int(time.time() * 1000)}"

            st.markdown(f"""
            <div id="{announcement_id}"
                 role="status"
                 aria-live="{priority}"
                 aria-atomic="true"
                 class="sr-only"
                 style="position: absolute; left: -10000px; width: 1px; height: 1px; overflow: hidden;">
                {message}
            </div>
            """, unsafe_allow_html=True)

            logger.debug(f"   - Screen reader announcement: {message}")

        except Exception as e:
            logger.error(f"❌ Error announcing to screen reader: {e}")

    def create_skip_links(self, links: List[Tuple[str, str]]):
        """Create skip links for keyboard navigation"""
        try:
            if not self.config.enable_keyboard_navigation:
                return

            skip_links_html = []
            for link_id, link_text in links:
                skip_links_html.append(f'<a href="#{link_id}" class="skip-link">{link_text}</a>')

            st.markdown("".join(skip_links_html), unsafe_allow_html=True)

            logger.debug(f"   - Created skip links: {len(links)}")

        except Exception as e:
            logger.error(f"❌ Error creating skip links: {e}")

    def create_accessible_heading(self,
                                text: str,
                                level: int = 1,
                                element_id: Optional[str] = None):
        """Create accessible heading"""
        try:
            element_id = element_id or f"heading_{int(time.time() * 1000)}"

            # Add ARIA label if enabled
            if self.config.enable_aria_labels:
                self.add_aria_label(element_id, text)

            # Render heading
            if level == 1:
                st.markdown(f'<h1 id="{element_id}">{text}</h1>', unsafe_allow_html=True)
            elif level == 2:
                st.markdown(f'<h2 id="{element_id}">{text}</h2>', unsafe_allow_html=True)
            elif level == 3:
                st.markdown(f'<h3 id="{element_id}">{text}</h3>', unsafe_allow_html=True)
            elif level == 4:
                st.markdown(f'<h4 id="{element_id}">{text}</h4>', unsafe_allow_html=True)
            elif level == 5:
                st.markdown(f'<h5 id="{element_id}">{text}</h5>', unsafe_allow_html=True)
            elif level == 6:
                st.markdown(f'<h6 id="{element_id}">{text}</h6>', unsafe_allow_html=True)

            logger.debug(f"   - Created accessible heading level {level}: {text}")

        except Exception as e:
            logger.error(f"❌ Error creating accessible heading: {e}")
            st.markdown(f"{'#' * level} {text}")

    def create_accessible_button(self,
                               label: str,
                               key: str,
                               help_text: Optional[str] = None,
                               disabled: bool = False):
        """Create accessible button"""
        try:
            element_id = f"button_{key}"

            # Add ARIA label and description
            if self.config.enable_aria_labels:
                self.add_aria_label(
                    element_id=element_id,
                    label=label,
                    description=help_text
                )

            # Create button with accessibility features
            button = st.button(
                label,
                key=key,
                help=help_text,
                disabled=disabled
            )

            # Register as focusable element
            if self.config.enable_keyboard_navigation:
                self.register_focusable_element(
                    element_id=element_id,
                    selector=f"[data-testid='stButton'][key='{key}']"
                )

            return button

        except Exception as e:
            logger.error(f"❌ Error creating accessible button: {e}")
            return st.button(label, key=key, help=help_text, disabled=disabled)

    def create_accessible_form(self,
                             form_title: str,
                             form_fields: List[Dict[str, Any]],
                             submit_label: str = "Submit",
                             form_key: str = "accessible_form"):
        """Create accessible form"""
        try:
            form_id = f"form_{form_key}"

            # Create form container with proper labeling
            with st.form(key=form_key):
                # Form title
                self.create_accessible_heading(form_title, level=2, element_id=f"{form_id}_title")

                # Form description for screen readers
                if self.config.enable_aria_labels:
                    self.announce_to_screen_reader(f"Form: {form_title}")

                # Create fields
                field_values = {}

                for field in form_fields:
                    field_type = field.get('type', 'text')
                    field_key = field.get('key')
                    field_label = field.get('label')
                    field_help = field.get('help')
                    field_required = field.get('required', False)
                    field_options = field.get('options', [])

                    field_id = f"{form_id}_{field_key}"

                    # Field label with required indicator
                    label_text = field_label
                    if field_required:
                        label_text += " *"

                    # Create field based on type
                    if field_type == 'text':
                        value = st.text_input(
                            label=label_text,
                            key=field_key,
                            help=field_help
                        )

                    elif field_type == 'number':
                        value = st.number_input(
                            label=label_text,
                            key=field_key,
                            help=field_help
                        )

                    elif field_type == 'select':
                        value = st.selectbox(
                            label=label_text,
                            options=field_options,
                            key=field_key,
                            help=field_help
                        )

                    elif field_type == 'multiselect':
                        value = st.multiselect(
                            label=label_text,
                            options=field_options,
                            key=field_key,
                            help=field_help
                        )

                    elif field_type == 'checkbox':
                        value = st.checkbox(
                            label=label_text,
                            key=field_key,
                            help=field_help
                        )

                    else:
                        value = st.text_input(
                            label=label_text,
                            key=field_key,
                            help=field_help
                        )

                    field_values[field_key] = value

                    # Add ARIA label if enabled
                    if self.config.enable_aria_labels:
                        self.add_aria_label(
                            element_id=field_id,
                            label=label_text,
                            description=field_help
                        )

                # Submit button
                submitted = self.create_accessible_button(
                    label=submit_label,
                    key=f"{form_key}_submit"
                )

                if submitted:
                    # Announce form submission
                    if self.config.enable_screen_reader_support:
                        self.announce_to_screen_reader("Form submitted successfully")

                    return field_values

                return None

        except Exception as e:
            logger.error(f"❌ Error creating accessible form: {e}")
            return None

    def create_accessible_table(self,
                              data: List[Dict[str, Any]],
                              title: str,
                              caption: Optional[str] = None):
        """Create accessible data table"""
        try:
            table_id = f"table_{int(time.time() * 1000)}"

            # Table title
            self.create_accessible_heading(title, level=3, element_id=f"{table_id}_title")

            # Table caption for accessibility
            if caption and self.config.enable_aria_labels:
                st.markdown(f'<p id="{table_id}_caption" class="table-caption">{caption}</p>',
                          unsafe_allow_html=True)
                self.add_aria_label(
                    element_id=table_id,
                    label=title,
                    description=caption
                )

            # Convert data to DataFrame for display
            import pandas as pd
            df = pd.DataFrame(data)

            # Display table with accessibility features
            st.dataframe(df, use_container_width=True)

            # Announce table content to screen readers
            if self.config.enable_screen_reader_support:
                rows_text = f"{len(data)} rows" if data else "No data"
                cols_text = f"{len(data[0])} columns" if data else "No columns"
                self.announce_to_screen_reader(f"Table {title}: {rows_text}, {cols_text}")

            logger.debug(f"   - Created accessible table: {title}")

        except Exception as e:
            logger.error(f"❌ Error creating accessible table: {e}")
            # Fallback to simple display
            st.title(title)
            if caption:
                st.caption(caption)
            st.json(data)

    def create_accessible_chart(self,
                              chart_data: Any,
                              chart_type: str,
                              title: str,
                              description: Optional[str] = None):
        """Create accessible chart"""
        try:
            chart_id = f"chart_{int(time.time() * 1000)}"

            # Chart title
            self.create_accessible_heading(title, level=3, element_id=f"{chart_id}_title")

            # Chart description
            if description:
                st.markdown(f'<p id="{chart_id}_desc">{description}</p>', unsafe_allow_html=True)

            # Create chart based on type
            if chart_type == 'line':
                st.line_chart(chart_data)
            elif chart_type == 'bar':
                st.bar_chart(chart_data)
            elif chart_type == 'area':
                st.area_chart(chart_data)
            else:
                st.line_chart(chart_data)

            # Add data table for screen readers
            if self.config.enable_screen_reader_support:
                with st.expander("View Chart Data (Accessible)", expanded=False):
                    if hasattr(chart_data, 'to_dict'):
                        st.json(chart_data.to_dict())
                    else:
                        st.json(chart_data)

                self.announce_to_screen_reader(f"Chart {title} displayed. Data available in expanded section.")

            # Add ARIA label
            if self.config.enable_aria_labels:
                self.add_aria_label(
                    element_id=chart_id,
                    label=title,
                    description=description
                )

            logger.debug(f"   - Created accessible chart: {title}")

        except Exception as e:
            logger.error(f"❌ Error creating accessible chart: {e}")
            # Fallback to data display
            st.title(title)
            if description:
                st.write(description)
            st.dataframe(chart_data)

    def get_accessibility_info(self) -> Dict[str, Any]:
        """Get current accessibility configuration info"""
        try:
            info = {
                'initialized': self._initialized,
                'compliance_level': self.config.level.value,
                'features_enabled': {
                    'keyboard_navigation': self.config.enable_keyboard_navigation,
                    'screen_reader_support': self.config.enable_screen_reader_support,
                    'high_contrast': self.config.enable_high_contrast,
                    'large_text': self.config.enable_large_text,
                    'reduced_motion': self.config.enable_reduced_motion,
                    'focus_indicators': self.config.enable_focus_indicators,
                    'aria_labels': self.config.enable_aria_labels,
                    'descriptions': self.config.enable_descriptions
                },
                'user_preferences': self._user_preferences,
                'focus_elements_count': len(self._focus_elements),
                'aria_labels_count': len(self._aria_labels),
                'keyboard_handlers_count': len(self._keyboard_handlers),
                'screen_reader': self._screen_reader_config.get('type', 'none')
            }

            return info

        except Exception as e:
            logger.error(f"❌ Error getting accessibility info: {e}")
            return {}

# Global accessibility manager instance
_accessibility_manager: Optional[AccessibilityFeaturesManager] = None

def get_accessibility_manager(config: Optional[AccessibilityConfig] = None) -> AccessibilityFeaturesManager:
    """Get or create the global accessibility manager instance"""
    global _accessibility_manager

    if _accessibility_manager is None:
        _accessibility_manager = AccessibilityFeaturesManager(config)

    return _accessibility_manager

def init_accessibility_features(config: Optional[AccessibilityConfig] = None) -> AccessibilityFeaturesManager:
    """Initialize accessibility features (alias for get_accessibility_manager)"""
    return get_accessibility_manager(config)

# Context7 compliant utility functions
def create_accessible_section(title: str,
                            content_func: Callable,
                            section_id: Optional[str] = None,
                            level: int = 2):
    """Create accessible content section"""
    try:
        accessibility_manager = get_accessibility_manager()
        section_id = section_id or f"section_{int(time.time() * 1000)}"

        # Section heading
        accessibility_manager.create_accessible_heading(title, level=level, element_id=section_id)

        # Section content
        with st.container():
            content_func()

    except Exception as e:
        logger.error(f"❌ Error creating accessible section: {e}")
        # Fallback
        st.markdown(f"{'#' * level} {title}")
        content_func()

def create_accessible_alert(message: str,
                          alert_type: str = "info",
                          dismissible: bool = True):
    """Create accessible alert/notification"""
    try:
        accessibility_manager = get_accessibility_manager()

        alert_id = f"alert_{int(time.time() * 1000)}"

        # Determine alert role and aria-live based on type
        role_map = {
            "error": ("alert", "assertive"),
            "warning": ("alert", "polite"),
            "success": ("status", "polite"),
            "info": ("status", "polite")
        }

        role, live_region = role_map.get(alert_type, ("status", "polite"))

        # Display alert using Streamlit
        if alert_type == "error":
            st.error(message)
        elif alert_type == "warning":
            st.warning(message)
        elif alert_type == "success":
            st.success(message)
        else:
            st.info(message)

        # Announce to screen reader
        if accessibility_manager.config.enable_screen_reader_support:
            accessibility_manager.announce_to_screen_reader(message, live_region)

        # Add ARIA attributes
        if accessibility_manager.config.enable_aria_labels:
            st.markdown(f"""
            <div id="{alert_id}"
                 role="{role}"
                 aria-live="{live_region}"
                 aria-atomic="true"
                 class="sr-only"
                 style="position: absolute; left: -10000px; width: 1px; height: 1px; overflow: hidden;">
                {message}
            </div>
            """, unsafe_allow_html=True)

    except Exception as e:
        logger.error(f"❌ Error creating accessible alert: {e}")
        # Fallback
        if alert_type == "error":
            st.error(message)
        elif alert_type == "warning":
            st.warning(message)
        elif alert_type == "success":
            st.success(message)
        else:
            st.info(message)

def create_accessible_loading(text: str = "Loading..."):
    """Create accessible loading indicator"""
    try:
        accessibility_manager = get_accessibility_manager()

        loading_id = f"loading_{int(time.time() * 1000)}"

        # Show loading spinner
        with st.spinner(text):
            # Announce loading state
            if accessibility_manager.config.enable_screen_reader_support:
                accessibility_manager.announce_to_screen_reader(text, "assertive")

            # Add ARIA live region for loading status
            if accessibility_manager.config.enable_aria_labels:
                st.markdown(f"""
                <div id="{loading_id}"
                     role="status"
                     aria-live="polite"
                     aria-busy="true"
                     class="sr-only"
                     style="position: absolute; left: -10000px; width: 1px; height: 1px; overflow: hidden;">
                    {text}
                </div>
                """, unsafe_allow_html=True)

            yield  # Yield control back to caller

        # Announce completion
        if accessibility_manager.config.enable_screen_reader_support:
            accessibility_manager.announce_to_screen_reader("Loading completed", "polite")

        # Update ARIA attributes
        if accessibility_manager.config.enable_aria_labels:
            st.markdown(f"""
            <script>
                var loadingElement = document.getElementById('{loading_id}');
                if (loadingElement) {{
                    loadingElement.setAttribute('aria-busy', 'false');
                    loadingElement.textContent = 'Loading completed';
                }}
            </script>
            """, unsafe_allow_html=True)

    except Exception as e:
        logger.error(f"❌ Error creating accessible loading: {e}")
        # Fallback
        with st.spinner(text):
            yield