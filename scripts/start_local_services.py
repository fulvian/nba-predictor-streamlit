#!/usr/bin/env python3
"""
NBA Predictor Local Services Launcher
启动本地服务的脚本 - Avvia i servizi NBA Predictor in locale

This script starts all NBA Predictor services locally:
- Streamlit Dashboard (port 8501)
- FastAPI REST API (port 8000)
- WebSocket Server (port 8001)
"""

import os
import sys
import subprocess
import time
import signal
import logging
from pathlib import Path
from typing import Dict, List, Optional
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ServiceManager:
    """Manages multiple services as subprocesses."""

    def __init__(self):
        self.services: Dict[str, subprocess.Popen] = {}
        self.running = True

        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, shutting down services...")
        self.running = False
        self.stop_all_services()
        sys.exit(0)

    def start_service(
        self, name: str, command: List[str], cwd: Optional[str] = None
    ) -> bool:
        """Start a service as subprocess."""
        try:
            logger.info(f"Starting {name}...")

            # Set environment variables
            # Set environment variables
            env = os.environ.copy()
            # Get project root (parent of scripts directory)
            project_root = Path(__file__).resolve().parent.parent
            env["PYTHONPATH"] = f"{str(project_root)}:{str(project_root / 'src')}"
            env["NBA_PREDICTOR_ENV"] = "development"

            process = subprocess.Popen(
                command,
                cwd=cwd or Path.cwd(),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
            )

            self.services[name] = process
            logger.info(f"✅ {name} started (PID: {process.pid})")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to start {name}: {e}")
            return False

    def stop_service(self, name: str) -> bool:
        """Stop a specific service."""
        if name not in self.services:
            logger.warning(f"Service {name} not found")
            return False

        try:
            process = self.services[name]
            logger.info(f"Stopping {name} (PID: {process.pid})...")

            # Try graceful shutdown first
            process.terminate()

            # Wait for process to stop
            try:
                process.wait(timeout=10)
                logger.info(f"✅ {name} stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                process.kill()
                process.wait()
                logger.info(f"✅ {name} force killed")

            del self.services[name]
            return True

        except Exception as e:
            logger.error(f"❌ Failed to stop {name}: {e}")
            return False

    def stop_all_services(self):
        """Stop all running services."""
        logger.info("Stopping all services...")
        for name in list(self.services.keys()):
            self.stop_service(name)

    def monitor_services(self):
        """Monitor running services and log their output."""
        while self.running:
            for name, process in list(self.services.items()):
                if process.poll() is not None:
                    # Process has terminated
                    return_code = process.returncode
                    if return_code != 0:
                        logger.error(f"❌ {name} exited with code {return_code}")
                    else:
                        logger.info(f"✅ {name} exited normally")

                    # Read any remaining output
                    if process.stdout:
                        output = process.stdout.read()
                        if output:
                            logger.info(f"{name} output: {output}")

                    del self.services[name]

            # Check if any services are still running
            if not self.services:
                logger.info("All services have stopped")
                break

            time.sleep(1)

    def check_service_health(self, name: str, url: str, timeout: int = 30) -> bool:
        """Check if a service is healthy by making HTTP request."""
        try:
            import requests

            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        logger.info(f"✅ {name} is healthy at {url}")
                        return True
                except requests.exceptions.RequestException:
                    pass

                time.sleep(2)

            logger.warning(f"⚠️ {name} health check timed out")
            return False

        except ImportError:
            logger.warning(f"⚠️ Cannot check {name} health: requests not available")
            return True  # Assume healthy if requests not available


def setup_environment():
    """Setup the development environment."""
    logger.info("🔧 Setting up environment...")

    # Check if virtual environment exists
    venv_paths = [".venv_new", ".venv", "venv"]
    venv_path = None

    for path in venv_paths:
        if Path(path).exists():
            venv_path = Path(path)
            break

    if venv_path:
        logger.info(f"✅ Using virtual environment: {venv_path}")
        # Activate virtual environment
        python_exe = venv_path / "bin" / "python3"
        if not python_exe.exists():
            python_exe = venv_path / "bin" / "python"

        if python_exe.exists():
            return str(python_exe)

    # Fallback to system python if no venv found
    logger.warning("⚠️ No virtual environment found. Using system Python.")
    return sys.executable


def cleanup_existing_processes():
    """Find and kill existing Streamlit/FastAPI processes to ensure clean start."""
    logger.info("🧹 Cleaning up existing processes...")
    try:
        # Kill Streamlit on port 8501
        subprocess.run(["pkill", "-f", "streamlit run.*8501"], capture_output=True)
        # Kill generic streamlit
        subprocess.run(
            ["pkill", "-f", "streamlit run src/nba_predictor"], capture_output=True
        )
        # Kill FastAPI/Uvicorn
        subprocess.run(
            ["pkill", "-f", "uvicorn src.api.nba_prediction_api"], capture_output=True
        )
        time.sleep(1)  # Wait for cleanup
    except Exception as e:
        logger.warning(f"⚠️ Cleanup warning: {e}")


def start_streamlit_dashboard(python_exe: str, service_manager: ServiceManager) -> bool:
    """Start the Streamlit dashboard."""
    logger.info("🚀 Starting Streamlit dashboard...")

    dashboard_path = "src/nba_predictor/streamlit/new_wic_dashboard.py"
    if not Path(dashboard_path).exists():
        logger.error(f"❌ Dashboard file not found: {dashboard_path}")
        return False

    command = [
        python_exe,
        "-m",
        "streamlit",
        "run",
        dashboard_path,
        "--server.port",
        "8501",
        "--server.address",
        "0.0.0.0",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]

    return service_manager.start_service("streamlit", command)


def start_fastapi_server(python_exe: str, service_manager: ServiceManager) -> bool:
    """Start the FastAPI server."""
    logger.info("🚀 Starting FastAPI server...")

    api_path = "src/api/nba_prediction_api.py"
    # Check if API file exists, if not, skip (user might not have it yet or it's elsewhere)
    if not Path(api_path).exists():
        logger.warning(f"⚠️ API file not found: {api_path}. Skipping API start.")
        return False

    command = [
        python_exe,
        "-m",
        "uvicorn",
        "src.api.nba_prediction_api:app",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--reload",
    ]

    return service_manager.start_service("fastapi", command)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Start NBA Predictor local services")
    parser.add_argument(
        "--dashboard-only",
        action="store_true",
        help="Start only the Streamlit dashboard",
    )
    parser.add_argument(
        "--api-only", action="store_true", help="Start only the FastAPI server"
    )
    parser.add_argument(
        "--no-health-check", action="store_true", help="Skip health checks"
    )

    args = parser.parse_args()

    logger.info("🏀 NBA Predictor Local Services Launcher")
    logger.info("=" * 50)

    # 0. Cleanup old processes
    cleanup_existing_processes()

    # Setup environment
    python_exe = setup_environment()
    if not python_exe:
        sys.exit(1)

    # Create service manager
    service_manager = ServiceManager()

    try:
        # Start services based on arguments
        services_started = []

        if not args.api_only:
            if start_streamlit_dashboard(python_exe, service_manager):
                services_started.append("streamlit")
                time.sleep(2)  # Give Streamlit time to start

        if not args.dashboard_only:
            if start_fastapi_server(python_exe, service_manager):
                services_started.append("fastapi")
                time.sleep(2)  # Give FastAPI time to start

        if not services_started:
            logger.error("❌ No services were started. Check logs.")
            sys.exit(1)

        # Health checks
        if not args.no_health_check:
            logger.info("🔍 Performing health checks...")

            if "streamlit" in services_started:
                service_manager.check_service_health(
                    "Streamlit", "http://localhost:8501/_stcore/health"
                )

            if "fastapi" in services_started:
                service_manager.check_service_health(
                    "FastAPI", "http://localhost:8000/health"
                )

        # Print service URLs
        logger.info("🌐 Services are running at:")
        if "streamlit" in services_started:
            logger.info("   📊 Streamlit Dashboard: http://localhost:8501")
        if "fastapi" in services_started:
            logger.info("   🔌 FastAPI Server: http://localhost:8000")
            logger.info("   📚 API Documentation: http://localhost:8000/docs")

        logger.info("=" * 50)
        logger.info("Press Ctrl+C to stop all services")

        # Monitor services
        service_manager.monitor_services()

    except KeyboardInterrupt:
        logger.info("🛑 Received interrupt signal")
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
    finally:
        service_manager.stop_all_services()
        logger.info("👋 All services stopped")


if __name__ == "__main__":
    main()
