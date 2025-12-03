"""
Progressive Web App Features System - Context7 Compliant
Task 3.4.4: Progressive Web App Features Implementation

Provides comprehensive PWA features for Streamlit applications with:
- Service Worker management
- Offline functionality
- App manifest generation
- Cache management
- Push notifications
- Background sync
- Install prompts
- Performance monitoring
"""

import streamlit as st
import json
import time
import logging
import hashlib
import base64
from typing import Dict, List, Any, Optional, Callable, Tuple
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import threading
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CacheStrategy(Enum):
    """Caching strategies for PWA"""
    CACHE_FIRST = "cache_first"
    NETWORK_FIRST = "network_first"
    CACHE_ONLY = "cache_only"
    NETWORK_ONLY = "network_only"
    STALE_WHILE_REVALIDATE = "stale_while_revalidate"

class NotificationPermission(Enum):
    """Notification permission levels"""
    DEFAULT = "default"
    GRANTED = "granted"
    DENIED = "denied"

class InstallStatus(Enum):
    """PWA install status"""
    NOT_INSTALLED = "not_installed"
    INSTALLABLE = "installable"
    INSTALLED = "installed"
    NOT_SUPPORTED = "not_supported"

@dataclass
class PWAConfig:
    """PWA configuration"""
    app_name: str = "NBA Predictor"
    app_short_name: str = "NBA-Predictor"
    app_description: str = "NBA Game Prediction and Betting Analysis"
    app_version: str = "1.0.0"
    theme_color: str = "#1f77b4"
    background_color: str = "#ffffff"
    display: str = "standalone"  # standalone, fullscreen, minimal-ui, browser
    orientation: str = "portrait"  # any, natural, landscape, portrait
    start_url: str = "/"
    scope: str = "/"
    icons: List[Dict[str, Any]] = field(default_factory=list)
    categories: List[str] = field(default_factory=lambda: ["sports", "productivity"])
    lang: str = "en-US"
    dir: str = "ltr"
    prefer_related_applications: bool = False

@dataclass
class CacheConfig:
    """Cache configuration"""
    strategy: CacheStrategy = CacheStrategy.CACHE_FIRST
    max_age: int = 3600  # 1 hour
    max_size: int = 50 * 1024 * 1024  # 50MB
    version: str = "v1"
    cache_name: str = "nba_predictor_cache"

@dataclass
class ServiceWorkerConfig:
    """Service worker configuration"""
    enabled: bool = True
    skip_waiting: bool = True
    clients_claim: bool = True
    precache_files: List[str] = field(default_factory=list)
    runtime_cache: List[Dict[str, Any]] = field(default_factory=list)

class PWAFeaturesManager:
    """
    Main PWA features manager
    Provides comprehensive PWA functionality with Context7 compliance
    """

    def __init__(self,
                 pwa_config: Optional[PWAConfig] = None,
                 cache_config: Optional[CacheConfig] = None,
                 sw_config: Optional[ServiceWorkerConfig] = None):
        self.pwa_config = pwa_config or PWAConfig()
        self.cache_config = cache_config or CacheConfig()
        self.sw_config = sw_config or ServiceWorkerConfig()

        # PWA state
        self._is_offline = False
        self._install_status = InstallStatus.NOT_INSTALLED
        self._notification_permission = NotificationPermission.DEFAULT
        self._cached_resources: Dict[str, Dict[str, Any]] = {}
        self._background_sync_queue: List[Dict[str, Any]] = []

        # Performance metrics
        self._performance_metrics: Dict[str, Any] = {
            'load_times': [],
            'cache_hits': 0,
            'cache_misses': 0,
            'network_requests': 0,
            'service_worker_messages': 0
        }

        # Thread safety
        self._lock = threading.RLock()

        # Initialize PWA features
        self._initialize_pwa_features()

    def _initialize_pwa_features(self):
        """Initialize PWA features"""
        try:
            with self._lock:
                # Initialize service worker
                if self.sw_config.enabled:
                    self._initialize_service_worker()

                # Initialize caching
                self._initialize_caching()

                # Initialize app manifest
                self._initialize_app_manifest()

                # Initialize install detection
                self._initialize_install_detection()

                # Initialize notifications
                self._initialize_notifications()

                # Initialize offline detection
                self._initialize_offline_detection()

                logger.info("🚀 PWA Features Manager initialized with Context7 compliance")
                logger.info(f"   - App: {self.pwa_config.app_name} v{self.pwa_config.app_version}")
                logger.info(f"   - Service Worker: {'enabled' if self.sw_config.enabled else 'disabled'}")
                logger.info(f"   - Cache Strategy: {self.cache_config.strategy.value}")
                logger.info(f"   - Install Status: {self._install_status.value}")

        except Exception as e:
            logger.error(f"❌ Failed to initialize PWA features: {e}")
            raise

    def _initialize_service_worker(self):
        """Initialize service worker"""
        try:
            # Generate service worker script
            sw_script = self._generate_service_worker_script()

            # Cache service worker for Streamlit injection
            st.session_state['service_worker_script'] = sw_script

            # Set up service worker communication
            st.session_state['service_worker_ready'] = False

            logger.debug(f"   - Service worker script generated ({len(sw_script)} chars)")

        except Exception as e:
            logger.warning(f"⚠️ Could not initialize service worker: {e}")

    def _generate_service_worker_script(self) -> str:
        """Generate service worker script"""
        script = f'''
// NBA Predictor Service Worker - Generated {datetime.now().isoformat()}
const CACHE_NAME = '{self.cache_config.cache_name}_{self.cache_config.version}';
const RUNTIME_CACHE = 'nba_predictor_runtime';

// Files to precache
const PRECACHE_URLS = {json.dumps(self.sw_config.precache_files)};

// Install event
self.addEventListener('install', event => {{
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then(cache => {{
                console.log('Service Worker: Precaching files');
                return cache.addAll(PRECACHE_URLS);
            }})
            .then(() => self.skipWaiting())
    );
}});

// Activate event
self.addEventListener('activate', event => {{
    event.waitUntil(
        caches.keys().then(cacheNames => {{
            return Promise.all(
                cacheNames.map(cacheName => {{
                    if (cacheName !== CACHE_NAME) {{
                        console.log('Service Worker: Deleting old cache', cacheName);
                        return caches.delete(cacheName);
                    }}
                }})
            );
        }}).then(() => self.clients.claim())
    );
}});

// Fetch event
self.addEventListener('fetch', event => {{
    const url = new URL(event.request.url);

    // Skip non-GET requests
    if (event.request.method !== 'GET') return;

    // Handle different cache strategies
    if (url.origin === location.origin) {{
        // Same origin - use cache strategy
        event.respondWith(
            caches.match(event.request).then(response => {{
                {self._get_cache_strategy_logic()}
            }})
        );
    }} else {{
        // Cross-origin - network first
        event.respondWith(
            fetch(event.request).catch(() => {{
                // Fallback to cache if available
                return caches.match(event.request);
            }})
        );
    }}
}});

// Background sync
self.addEventListener('sync', event => {{
    if (event.tag === 'background-sync') {{
        event.waitUntil(doBackgroundSync());
    }}
}});

// Push notifications
self.addEventListener('push', event => {{
    const options = {{
        body: event.data ? event.data.text() : 'NBA Predictor Update',
        icon: '/icons/icon-192x192.png',
        badge: '/icons/badge-72x72.png',
        tag: 'nba-predictor',
        renotify: true
    }};

    event.waitUntil(
        self.registration.showNotification('NBA Predictor', options)
    );
}});

// Notification click
self.addEventListener('notificationclick', event => {{
    event.notification.close();
    event.waitUntil(
        clients.openWindow('/')
    );
}});

// Cache strategy helper functions
function isCacheValid(response, maxAge) {{
    if (!response || !response.headers) return false;

    const dateHeader = response.headers.get('date');
    if (!dateHeader) return false;

    const responseTime = new Date(dateHeader).getTime();
    const now = Date.now();
    return (now - responseTime) < (maxAge * 1000);
}}

async function doBackgroundSync() {{
    // Handle background sync operations
    console.log('Background sync triggered');
}}
'''
        return script

    def _get_cache_strategy_logic(self) -> str:
        """Get cache strategy logic for service worker"""
        strategy = self.cache_config.strategy
        max_age = self.cache_config.max_age

        if strategy == CacheStrategy.CACHE_FIRST:
            return f'''
                if (response && isCacheValid(response, {max_age})) {{
                    console.log('Cache hit:', event.request.url);
                    return response;
                }} else {{
                    console.log('Cache miss, fetching:', event.request.url);
                    return fetch(event.request).then(fetchResponse => {{
                        // Cache the new response
                        if (fetchResponse.status === 200) {{
                            const responseClone = fetchResponse.clone();
                            caches.open(CACHE_NAME).then(cache => {{
                                cache.put(event.request, responseClone);
                            }});
                        }}
                        return fetchResponse;
                    }});
                }}
            '''
        elif strategy == CacheStrategy.NETWORK_FIRST:
            return f'''
                return fetch(event.request)
                    .then(fetchResponse => {{
                        // Cache successful network responses
                        if (fetchResponse.status === 200) {{
                            const responseClone = fetchResponse.clone();
                            caches.open(CACHE_NAME).then(cache => {{
                                cache.put(event.request, responseClone);
                            }});
                        }}
                        return fetchResponse;
                    }})
                    .catch(() => {{
                        // Network failed, try cache
                        console.log('Network failed, trying cache:', event.request.url);
                        return response;
                    }});
            '''
        elif strategy == CacheStrategy.STALE_WHILE_REVALIDATE:
            return f'''
                const networkPromise = fetch(event.request).then(fetchResponse => {{
                    if (fetchResponse.status === 200) {{
                        const responseClone = fetchResponse.clone();
                        caches.open(CACHE_NAME).then(cache => {{
                            cache.put(event.request, responseClone);
                        }});
                    }}
                    return fetchResponse;
                }}).catch(() => response);

                return response || networkPromise;
            '''
        else:
            return 'return fetch(event.request);'

    def _initialize_caching(self):
        """Initialize caching system"""
        try:
            # Set up runtime cache configurations
            runtime_cache_config = [
                {
                    'urlPattern': '/api/*',
                    'handler': 'NetworkFirst',
                    'options': {
                        'cacheName': 'api-cache',
                        'expiration': {
                            'maxEntries': 100,
                            'maxAgeSeconds': self.cache_config.max_age
                        }
                    }
                },
                {
                    'urlPattern': 'https://cdn.jsdelivr.net/*',
                    'handler': 'CacheFirst',
                    'options': {
                        'cacheName': 'cdn-cache',
                        'expiration': {
                            'maxEntries': 50,
                            'maxAgeSeconds': self.cache_config.max_age * 24 * 7  # 1 week
                        }
                    }
                }
            ]

            self.sw_config.runtime_cache = runtime_cache_config

            logger.debug(f"   - Runtime cache strategies: {len(runtime_cache_config)}")

        except Exception as e:
            logger.warning(f"⚠️ Could not initialize caching: {e}")

    def _initialize_app_manifest(self):
        """Initialize app manifest"""
        try:
            # Generate default icons if not provided
            if not self.pwa_config.icons:
                self.pwa_config.icons = [
                    {
                        'src': '/icons/icon-72x72.png',
                        'sizes': '72x72',
                        'type': 'image/png'
                    },
                    {
                        'src': '/icons/icon-96x96.png',
                        'sizes': '96x96',
                        'type': 'image/png'
                    },
                    {
                        'src': '/icons/icon-128x128.png',
                        'sizes': '128x128',
                        'type': 'image/png'
                    },
                    {
                        'src': '/icons/icon-144x144.png',
                        'sizes': '144x144',
                        'type': 'image/png'
                    },
                    {
                        'src': '/icons/icon-152x152.png',
                        'sizes': '152x152',
                        'type': 'image/png'
                    },
                    {
                        'src': '/icons/icon-192x192.png',
                        'sizes': '192x192',
                        'type': 'image/png',
                        'purpose': 'any maskable'
                    },
                    {
                        'src': '/icons/icon-384x384.png',
                        'sizes': '384x384',
                        'type': 'image/png'
                    },
                    {
                        'src': '/icons/icon-512x512.png',
                        'sizes': '512x512',
                        'type': 'image/png',
                        'purpose': 'any maskable'
                    }
                ]

            # Generate manifest JSON
            manifest = self._generate_app_manifest()

            # Store manifest for injection
            st.session_state['app_manifest'] = manifest

            logger.debug(f"   - App manifest generated ({len(manifest)} chars)")

        except Exception as e:
            logger.warning(f"⚠️ Could not initialize app manifest: {e}")

    def _generate_app_manifest(self) -> str:
        """Generate app manifest JSON"""
        manifest = {
            'name': self.pwa_config.app_name,
            'short_name': self.pwa_config.app_short_name,
            'description': self.pwa_config.app_description,
            'version': self.pwa_config.app_version,
            'theme_color': self.pwa_config.theme_color,
            'background_color': self.pwa_config.background_color,
            'display': self.pwa_config.display,
            'orientation': self.pwa_config.orientation,
            'start_url': self.pwa_config.start_url,
            'scope': self.pwa_config.scope,
            'icons': self.pwa_config.icons,
            'categories': self.pwa_config.categories,
            'lang': self.pwa_config.lang,
            'dir': self.pwa_config.dir,
            'prefer_related_applications': self.pwa_config.prefer_related_applications
        }

        return json.dumps(manifest, indent=2)

    def _initialize_install_detection(self):
        """Initialize PWA install detection"""
        try:
            # Detect if app is already installed
            install_detection_script = '''
            // PWA Install Detection
            (function() {
                let deferredPrompt;
                let installButton = null;

                // Listen for beforeinstallprompt event
                window.addEventListener('beforeinstallprompt', (e) => {
                    e.preventDefault();
                    deferredPrompt = e;

                    // Show install button or UI
                    window.dispatchEvent(new CustomEvent('pwa-installable', {
                        detail: { canInstall: true }
                    }));
                });

                // Listen for appinstalled event
                window.addEventListener('appinstalled', () => {
                    deferredPrompt = null;
                    window.dispatchEvent(new CustomEvent('pwa-installed', {
                        detail: { installed: true }
                    }));
                });

                // Check if app is already installed
                if (window.matchMedia('(display-mode: standalone)').matches) {
                    window.dispatchEvent(new CustomEvent('pwa-already-installed', {
                        detail: { installed: true }
                    }));
                }

                // Expose install function
                window.installPWA = async function() {
                    if (!deferredPrompt) {
                        console.log('PWA install prompt not available');
                        return false;
                    }

                    deferredPrompt.prompt();
                    const { outcome } = await deferredPrompt.userChoice;
                    deferredPrompt = null;

                    return outcome === 'accepted';
                };
            })();
            '''

            st.session_state['install_detection_script'] = install_detection_script

            logger.debug("   - Install detection initialized")

        except Exception as e:
            logger.warning(f"⚠️ Could not initialize install detection: {e}")

    def _initialize_notifications(self):
        """Initialize push notifications"""
        try:
            # Initialize notification system
            notification_script = '''
            // PWA Notifications System
            (function() {
                window.PWANotifications = {
                    async requestPermission() {
                        if (!('Notification' in window)) {
                            console.log('This browser does not support notifications');
                            return 'denied';
                        }

                        if (Notification.permission === 'default') {
                            const permission = await Notification.requestPermission();
                            window.dispatchEvent(new CustomEvent('notification-permission-changed', {
                                detail: { permission }
                            }));
                            return permission;
                        }

                        return Notification.permission;
                    },

                    async subscribeToPush() {
                        if (!('serviceWorker' in navigator) || !('PushManager' in window)) {
                            console.log('Push messaging is not supported');
                            return null;
                        }

                        try {
                            const registration = await navigator.serviceWorker.ready;
                            const subscription = await registration.pushManager.subscribe({
                                userVisibleOnly: true,
                                applicationServerKey: this.urlBase64ToUint8Array('YOUR_VAPID_PUBLIC_KEY')
                            });

                            window.dispatchEvent(new CustomEvent('push-subscription-created', {
                                detail: { subscription }
                            }));

                            return subscription;
                        } catch (error) {
                            console.error('Failed to subscribe to push notifications:', error);
                            return null;
                        }
                    },

                    showNotification(title, options = {}) {
                        if (Notification.permission === 'granted') {
                            return new Notification(title, {
                                icon: '/icons/icon-192x192.png',
                                badge: '/icons/badge-72x72.png',
                                tag: 'nba-predictor',
                                renotify: true,
                                ...options
                            });
                        }
                    },

                    urlBase64ToUint8Array(base64String) {
                        const padding = '='.repeat((4 - base64String.length % 4) % 4);
                        const base64 = (base64String + padding)
                            .replace(/-/g, '+')
                            .replace(/_/g, '/');

                        const rawData = window.atob(base64);
                        const outputArray = new Uint8Array(rawData.length);

                        for (let i = 0; i < rawData.length; ++i) {
                            outputArray[i] = rawData.charCodeAt(i);
                        }

                        return outputArray;
                    }
                };
            })();
            '''

            st.session_state['notification_script'] = notification_script

            logger.debug("   - Notifications system initialized")

        except Exception as e:
            logger.warning(f"⚠️ Could not initialize notifications: {e}")

    def _initialize_offline_detection(self):
        """Initialize offline detection"""
        try:
            offline_detection_script = '''
            // Offline Detection
            (function() {
                let isOnline = navigator.onLine;

                function updateOnlineStatus() {
                    isOnline = navigator.onLine;
                    window.dispatchEvent(new CustomEvent('online-status-changed', {
                        detail: { online: isOnline }
                    }));

                    if (!isOnline) {
                        document.body.classList.add('offline');
                    } else {
                        document.body.classList.remove('offline');
                    }
                }

                window.addEventListener('online', updateOnlineStatus);
                window.addEventListener('offline', updateOnlineStatus);

                // Initial status
                updateOnlineStatus();
            })();
            '''

            st.session_state['offline_detection_script'] = offline_detection_script

            logger.debug("   - Offline detection initialized")

        except Exception as e:
            logger.warning(f"⚠️ Could not initialize offline detection: {e}")

    def inject_pwa_meta_tags(self):
        """Inject PWA meta tags into Streamlit app"""
        try:
            meta_tags = f"""
            <!-- PWA Meta Tags -->
            <meta name="theme-color" content="{self.pwa_config.theme_color}">
            <meta name="apple-mobile-web-app-capable" content="yes">
            <meta name="apple-mobile-web-app-status-bar-style" content="default">
            <meta name="apple-mobile-web-app-title" content="{self.pwa_config.app_short_name}">
            <meta name="application-name" content="{self.pwa_config.app_name}">
            <meta name="description" content="{self.pwa_config.app_description}">
            <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">

            <!-- Apple Icons -->
            <link rel="apple-touch-icon" href="/icons/icon-152x152.png">
            <link rel="apple-touch-icon" sizes="152x152" href="/icons/icon-152x152.png">
            <link rel="apple-touch-icon" sizes="167x167" href="/icons/icon-167x167.png">
            <link rel="apple-touch-icon" sizes="180x180" href="/icons/icon-180x180.png">

            <!-- Favicon -->
            <link rel="icon" type="image/png" sizes="32x32" href="/icons/icon-32x32.png">
            <link rel="icon" type="image/png" sizes="16x16" href="/icons/icon-16x16.png">
            <link rel="shortcut icon" href="/icons/icon-192x192.png">

            <!-- Manifest -->
            <link rel="manifest" href="/manifest.json">
            """

            st.markdown(meta_tags, unsafe_allow_html=True)

            logger.debug("   - PWA meta tags injected")

        except Exception as e:
            logger.warning(f"⚠️ Could not inject PWA meta tags: {e}")

    def inject_service_worker(self):
        """Inject service worker registration"""
        try:
            if not self.sw_config.enabled:
                return

            sw_registration_script = f'''
            <!-- Service Worker Registration -->
            <script>
            if ('serviceWorker' in navigator) {{
                window.addEventListener('load', () => {{
                    navigator.serviceWorker.register('/sw.js', {{
                        scope: '{self.pwa_config.scope}'
                    }})
                    .then(registration => {{
                        console.log('Service Worker registered:', registration.scope);
                        window.dispatchEvent(new CustomEvent('service-worker-registered', {{
                            detail: {{ registration }}
                        }}));
                    }})
                    .catch(error => {{
                        console.error('Service Worker registration failed:', error);
                        window.dispatchEvent(new CustomEvent('service-worker-registration-failed', {{
                            detail: {{ error }}
                        }}));
                    }});
                }});
            }} else {{
                console.log('Service Worker is not supported');
            }}
            </script>
            '''

            st.markdown(sw_registration_script, unsafe_allow_html=True)

            logger.debug("   - Service worker registration injected")

        except Exception as e:
            logger.warning(f"⚠️ Could not inject service worker: {e}")

    def inject_pwa_scripts(self):
        """Inject all PWA scripts"""
        try:
            scripts = []

            # Service worker registration
            if self.sw_config.enabled:
                scripts.append(st.session_state.get('service_worker_script', ''))

            # Install detection
            scripts.append(st.session_state.get('install_detection_script', ''))

            # Notifications
            scripts.append(st.session_state.get('notification_script', ''))

            # Offline detection
            scripts.append(st.session_state.get('offline_detection_script', ''))

            # Combine all scripts
            combined_script = f'''
            <script>
            {"".join(scripts)}

            // PWA Ready Event
            window.addEventListener('load', () => {{
                window.dispatchEvent(new CustomEvent('pwa-ready', {{
                    detail: {{
                        version: '{self.pwa_config.app_version}',
                        features: {{
                            serviceWorker: {str(self.sw_config.enabled).lower()},
                            notifications: true,
                            offline: true,
                            installable: true
                        }}
                    }}
                }}));
            }});
            </script>
            '''

            st.markdown(combined_script, unsafe_allow_html=True)

            logger.debug("   - PWA scripts injected")

        except Exception as e:
            logger.warning(f"⚠️ Could not inject PWA scripts: {e}")

    def create_install_prompt(self, button_text: str = "Install App"):
        """Create PWA install prompt"""
        try:
            install_prompt_html = f'''
            <div id="pwa-install-prompt" style="display: none;">
                <button id="pwa-install-button" class="pwa-install-btn">
                    📱 {button_text}
                </button>
            </div>

            <script>
            // Handle PWA install button
            let installButton = document.getElementById('pwa-install-button');
            let installPrompt = document.getElementById('pwa-install-prompt');

            // Show install prompt when app is installable
            window.addEventListener('pwa-installable', (e) => {{
                if (installPrompt) {{
                    installPrompt.style.display = 'block';
                }}
            }});

            // Hide install prompt when app is installed
            window.addEventListener('pwa-installed', () => {{
                if (installPrompt) {{
                    installPrompt.style.display = 'none';
                }}
            }});

            // Handle install button click
            if (installButton) {{
                installButton.addEventListener('click', async () => {{
                    if (window.installPWA) {{
                        const installed = await window.installPWA();
                        if (installed) {{
                            console.log('PWA installed successfully');
                        }}
                    }}
                }});
            }}
            </script>

            <style>
            .pwa-install-btn {{
                background: {self.pwa_config.theme_color};
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 8px;
                font-weight: 600;
                cursor: pointer;
                display: inline-flex;
                align-items: center;
                gap: 8px;
                transition: all 0.3s ease;
            }}

            .pwa-install-btn:hover {{
                opacity: 0.9;
                transform: translateY(-1px);
            }}

            .pwa-install-btn:active {{
                transform: translateY(0);
            }}
            </style>
            '''

            st.markdown(install_prompt_html, unsafe_allow_html=True)

            logger.debug("   - Install prompt created")

        except Exception as e:
            logger.warning(f"⚠️ Could not create install prompt: {e}")

    def create_offline_indicator(self):
        """Create offline status indicator"""
        try:
            offline_indicator_html = '''
            <div id="offline-indicator" style="display: none;">
                <span class="offline-icon">📡</span>
                <span class="offline-text">You're offline</span>
            </div>

            <script>
            // Handle offline status
            let offlineIndicator = document.getElementById('offline-indicator');

            window.addEventListener('online-status-changed', (e) => {{
                if (e.detail.online) {{
                    if (offlineIndicator) {{
                        offlineIndicator.style.display = 'none';
                    }}
                }} else {{
                    if (offlineIndicator) {{
                        offlineIndicator.style.display = 'flex';
                    }}
                }}
            }});
            </script>

            <style>
            #offline-indicator {{
                position: fixed;
                top: 20px;
                right: 20px;
                background: #ff6b6b;
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                display: flex;
                align-items: center;
                gap: 8px;
                font-size: 14px;
                font-weight: 600;
                z-index: 10000;
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
                animation: slideIn 0.3s ease;
            }}

            .offline-icon {{
                animation: pulse 2s infinite;
            }}

            @keyframes slideIn {{
                from {{
                    transform: translateX(100%);
                    opacity: 0;
                }}
                to {{
                    transform: translateX(0);
                    opacity: 1;
                }}
            }}

            @keyframes pulse {{
                0%, 100% {{
                    opacity: 1;
                }}
                50% {{
                    opacity: 0.5;
                }}
            }}

            .offline {{
                position: relative;
            }}

            .offline::after {{
                content: '';
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: rgba(255, 107, 107, 0.1);
                z-index: 9999;
                pointer-events: none;
            }}
            </style>
            '''

            st.markdown(offline_indicator_html, unsafe_allow_html=True)

            logger.debug("   - Offline indicator created")

        except Exception as e:
            logger.warning(f"⚠️ Could not create offline indicator: {e}")

    def cache_resource(self, url: str, data: Any, content_type: str = "application/json"):
        """Cache a resource"""
        try:
            with self._lock:
                cache_entry = {
                    'url': url,
                    'data': data,
                    'content_type': content_type,
                    'timestamp': time.time(),
                    'size': len(str(data))
                }

                self._cached_resources[url] = cache_entry

                # Update performance metrics
                self._performance_metrics['cache_hits'] += 1

                logger.debug(f"   - Cached resource: {url} ({cache_entry['size']} bytes)")

        except Exception as e:
            logger.error(f"❌ Error caching resource {url}: {e}")

    def get_cached_resource(self, url: str) -> Optional[Any]:
        """Get cached resource"""
        try:
            with self._lock:
                if url not in self._cached_resources:
                    self._performance_metrics['cache_misses'] += 1
                    return None

                cache_entry = self._cached_resources[url]

                # Check if cache is expired
                age = time.time() - cache_entry['timestamp']
                if age > self.cache_config.max_age:
                    del self._cached_resources[url]
                    self._performance_metrics['cache_misses'] += 1
                    return None

                self._performance_metrics['cache_hits'] += 1
                return cache_entry['data']

        except Exception as e:
            logger.error(f"❌ Error getting cached resource {url}: {e}")
            return None

    def schedule_background_sync(self, data: Dict[str, Any], tag: str = "default"):
        """Schedule data for background sync"""
        try:
            with self._lock:
                sync_item = {
                    'data': data,
                    'tag': tag,
                    'timestamp': time.time(),
                    'id': hashlib.md5(f"{tag}_{time.time()}".encode()).hexdigest()
                }

                self._background_sync_queue.append(sync_item)

                logger.debug(f"   - Scheduled background sync: {tag}")

        except Exception as e:
            logger.error(f"❌ Error scheduling background sync: {e}")

    def send_push_notification(self, title: str, body: str, options: Optional[Dict[str, Any]] = None):
        """Send push notification"""
        try:
            notification_options = {
                'body': body,
                'icon': '/icons/icon-192x192.png',
                'badge': '/icons/badge-72x72.png',
                'tag': 'nba-predictor',
                'renotify': True,
                'requireInteraction': False,
            }

            # Merge additional options if provided
            if options:
                notification_options.update(options)

            # In browser environment
            notification_script = f'''
            <script>
            if (window.PWANotifications && Notification.permission === 'granted') {{
                window.PWANotifications.showNotification('{title}', {json.dumps(notification_options)});
            }}
            </script>
            '''

            st.markdown(notification_script, unsafe_allow_html=True)

            logger.debug(f"   - Push notification sent: {title}")

        except Exception as e:
            logger.error(f"❌ Error sending push notification: {e}")

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get PWA performance metrics"""
        try:
            with self._lock:
                metrics = self._performance_metrics.copy()

                # Calculate additional metrics
                total_requests = metrics['cache_hits'] + metrics['cache_misses']
                cache_hit_rate = (metrics['cache_hits'] / total_requests) if total_requests > 0 else 0

                # Add derived metrics
                metrics.update({
                    'cache_hit_rate': cache_hit_rate,
                    'total_requests': total_requests,
                    'cache_size': sum(entry['size'] for entry in self._cached_resources.values()),
                    'background_sync_queue_size': len(self._background_sync_queue),
                    'offline': self._is_offline,
                    'install_status': self._install_status.value,
                    'notification_permission': self._notification_permission.value
                })

                return metrics

        except Exception as e:
            logger.error(f"❌ Error getting performance metrics: {e}")
            return {}

    def get_pwa_info(self) -> Dict[str, Any]:
        """Get comprehensive PWA information"""
        try:
            info = {
                'app_info': {
                    'name': self.pwa_config.app_name,
                    'short_name': self.pwa_config.app_short_name,
                    'version': self.pwa_config.app_version,
                    'description': self.pwa_config.app_description
                },
                'features': {
                    'service_worker': self.sw_config.enabled,
                    'caching': True,
                    'notifications': True,
                    'offline_support': True,
                    'install_prompt': True,
                    'background_sync': True
                },
                'configuration': {
                    'cache_strategy': self.cache_config.strategy.value,
                    'cache_max_age': self.cache_config.max_age,
                    'cache_max_size': self.cache_config.max_size,
                    'display_mode': self.pwa_config.display,
                    'theme_color': self.pwa_config.theme_color
                },
                'status': {
                    'install_status': self._install_status.value,
                    'notification_permission': self._notification_permission.value,
                    'offline': self._is_offline,
                    'service_worker_ready': st.session_state.get('service_worker_ready', False)
                },
                'resources': {
                    'cached_items': len(self._cached_resources),
                    'background_sync_queue': len(self._background_sync_queue)
                }
            }

            return info

        except Exception as e:
            logger.error(f"❌ Error getting PWA info: {e}")
            return {}

# Global PWA manager instance
_pwa_manager: Optional[PWAFeaturesManager] = None

def get_pwa_manager(pwa_config: Optional[PWAConfig] = None,
                   cache_config: Optional[CacheConfig] = None,
                   sw_config: Optional[ServiceWorkerConfig] = None) -> PWAFeaturesManager:
    """Get or create the global PWA manager instance"""
    global _pwa_manager

    if _pwa_manager is None:
        _pwa_manager = PWAFeaturesManager(pwa_config, cache_config, sw_config)

    return _pwa_manager

def init_pwa_features(pwa_config: Optional[PWAConfig] = None,
                     cache_config: Optional[CacheConfig] = None,
                     sw_config: Optional[ServiceWorkerConfig] = None) -> PWAFeaturesManager:
    """Initialize PWA features (alias for get_pwa_manager)"""
    return get_pwa_manager(pwa_config, cache_config, sw_config)

# Context7 compliant utility functions
def create_pwa_ready_page():
    """Create a PWA-ready Streamlit page"""
    try:
        pwa_manager = get_pwa_manager()

        # Inject PWA components
        pwa_manager.inject_pwa_meta_tags()
        pwa_manager.inject_service_worker()
        pwa_manager.inject_pwa_scripts()
        pwa_manager.create_install_prompt()
        pwa_manager.create_offline_indicator()

        # Add PWA status info
        if st.sidebar.checkbox("📱 PWA Status", False):
            st.sidebar.markdown("### Progressive Web App Features")

            pwa_info = pwa_manager.get_pwa_info()

            # App info
            with st.sidebar.expander("📋 App Information", expanded=False):
                st.json(pwa_info['app_info'])

            # Features status
            with st.sidebar.expander("✨ Features Status", expanded=False):
                features_status = {k: "✅" if v else "❌" for k, v in pwa_info['features'].items()}
                for feature, status in features_status.items():
                    st.sidebar.write(f"{status} {feature.replace('_', ' ').title()}")

            # Performance metrics
            with st.sidebar.expander("📊 Performance Metrics", expanded=False):
                metrics = pwa_manager.get_performance_metrics()
                st.sidebar.write(f"Cache Hit Rate: {metrics.get('cache_hit_rate', 0):.1%}")
                st.sidebar.write(f"Total Requests: {metrics.get('total_requests', 0)}")
                st.sidebar.write(f"Cache Size: {metrics.get('cache_size', 0) / 1024:.1f} KB")

        logger.info("📱 PWA-ready page components injected")

    except Exception as e:
        logger.error(f"❌ Error creating PWA-ready page: {e}")

def create_install_banner(message: str = "🏀 Install NBA Predictor for the best experience!"):
    """Create PWA install banner"""
    try:
        install_banner_html = f'''
        <div id="pwa-install-banner" style="display: none;">
            <div class="banner-content">
                <span class="banner-message">{message}</span>
                <button id="banner-install-btn" class="banner-install-btn">Install</button>
                <button id="banner-dismiss-btn" class="banner-dismiss-btn">✕</button>
            </div>
        </div>

        <script>
        let installBanner = document.getElementById('pwa-install-banner');
        let bannerInstallBtn = document.getElementById('banner-install-btn');
        let bannerDismissBtn = document.getElementById('banner-dismiss-btn');

        // Show banner when app is installable
        window.addEventListener('pwa-installable', () => {{
            if (installBanner && !localStorage.getItem('pwa-banner-dismissed')) {{
                installBanner.style.display = 'block';
            }}
        }});

        // Hide banner when app is installed
        window.addEventListener('pwa-installed', () => {{
            if (installBanner) {{
                installBanner.style.display = 'none';
            }}
        }});

        // Handle install button click
        if (bannerInstallBtn) {{
            bannerInstallBtn.addEventListener('click', async () => {{
                if (window.installPWA) {{
                    const installed = await window.installPWA();
                    if (installed) {{
                        if (installBanner) {{
                            installBanner.style.display = 'none';
                        }}
                    }}
                }}
            }});
        }}

        // Handle dismiss button click
        if (bannerDismissBtn) {{
            bannerDismissBtn.addEventListener('click', () => {{
                if (installBanner) {{
                    installBanner.style.display = 'none';
                    localStorage.setItem('pwa-banner-dismissed', 'true');
                }}
            }});
        }}
        </script>

        <style>
        #pwa-install-banner {{
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 12px 20px;
            z-index: 10001;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            animation: slideDown 0.5s ease;
        }}

        .banner-content {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            max-width: 1200px;
            margin: 0 auto;
        }}

        .banner-message {{
            font-weight: 600;
            flex: 1;
        }}

        .banner-install-btn {{
            background: white;
            color: #667eea;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            font-weight: 600;
            cursor: pointer;
            margin-right: 12px;
            transition: all 0.3s ease;
        }}

        .banner-install-btn:hover {{
            transform: translateY(-1px);
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }}

        .banner-dismiss-btn {{
            background: none;
            border: none;
            color: white;
            font-size: 18px;
            cursor: pointer;
            padding: 4px 8px;
            border-radius: 4px;
            transition: background-color 0.3s ease;
        }}

        .banner-dismiss-btn:hover {{
            background-color: rgba(255,255,255,0.1);
        }}

        @keyframes slideDown {{
            from {{
                transform: translateY(-100%);
                opacity: 0;
            }}
            to {{
                transform: translateY(0);
                opacity: 1;
            }}
        }}

        /* Adjust main content for banner */
        .main .block-container {{
            margin-top: 80px;
        }}
        </style>
        '''

        st.markdown(install_banner_html, unsafe_allow_html=True)

        logger.debug("   - Install banner created")

    except Exception as e:
        logger.warning(f"⚠️ Could not create install banner: {e}")