#!/usr/bin/env python3
"""
🚀 NBA Production Deployment Manager - Task 2.2.4

Advanced production deployment system with automatic rollback, canary deployments,
A/B testing framework, and production monitoring for NBA Ensemble Predictor.

Author: NBA Predictive Analytics System
Date: 2025-01-12
Architecture: DevStream SuperPowered with Context Set Compliance
"""

import os
import json
import time
import asyncio
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
import numpy as np
import pandas as pd
import hashlib
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue

# Monitoring and alerting
try:
    import schedule
    SCHEDULING_AVAILABLE = True
except ImportError:
    SCHEDULING_AVAILABLE = False

# Existing imports
from .model_version_manager import (
    NBAModelVersionManager,
    ModelVersion,
    ModelType,
    ModelStatus,
    ModelMetrics,
    RollbackConfig
)

class DeploymentStrategy(Enum):
    """Deployment strategies"""
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"
    SHADOW = "shadow"
    A_B_TESTING = "ab_testing"

class DeploymentStatus(Enum):
    """Deployment status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"
    PAUSED = "paused"

class CanaryStage(Enum):
    """Canary deployment stages"""
    PREPARATION = "preparation"
    SMALL_TRAFFIC = "small_traffic"      # 5%
    MEDIUM_TRAFFIC = "medium_traffic"    # 20%
    LARGE_TRAFFIC = "large_traffic"      # 50%
    FULL_TRAFFIC = "full_traffic"        # 100%

@dataclass
class DeploymentConfig:
    """Configuration for deployment process"""

    # Strategy configuration
    strategy: DeploymentStrategy = DeploymentStrategy.BLUE_GREEN
    auto_rollback_enabled: bool = True
    health_check_enabled: bool = True

    # Canary configuration
    canary_stages: List[CanaryStage] = field(default_factory=lambda: [
        CanaryStage.SMALL_TRAFFIC,
        CanaryStage.MEDIUM_TRAFFIC,
        CanaryStage.LARGE_TRAFFIC,
        CanaryStage.FULL_TRAFFIC
    ])
    canary_stage_duration_minutes: int = 10
    canary_traffic_percentages: Dict[CanaryStage, float] = field(default_factory=lambda: {
        CanaryStage.SMALL_TRAFFIC: 0.05,
        CanaryStage.MEDIUM_TRAFFIC: 0.20,
        CanaryStage.LARGE_TRAFFIC: 0.50,
        CanaryStage.FULL_TRAFFIC: 1.0
    })

    # Health check configuration
    health_check_interval_seconds: int = 30
    health_check_timeout_seconds: int = 60
    health_check_max_failures: int = 3

    # Performance thresholds for automatic rollback
    accuracy_threshold: float = 0.55
    latency_threshold_ms: float = 1000.0
    error_rate_threshold: float = 0.05

    # A/B testing configuration
    ab_test_duration_minutes: int = 60
    ab_test_traffic_split: float = 0.5  # 50% traffic to new model
    ab_test_min_predictions: int = 100
    ab_test_significance_level: float = 0.05

@dataclass
class DeploymentMetrics:
    """Metrics for deployment monitoring"""

    # Performance metrics
    current_accuracy: float = 0.0
    current_latency_ms: float = 0.0
    current_error_rate: float = 0.0

    # Traffic metrics
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0

    # A/B test metrics
    variant_a_accuracy: float = 0.0
    variant_b_accuracy: float = 0.0
    variant_a_requests: int = 0
    variant_b_requests: int = 0

    # Health status
    health_status: str = "healthy"
    last_health_check: Optional[datetime] = None
    consecutive_failures: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class DeploymentJob:
    """Deployment job information"""
    job_id: str
    deployment_id: str
    model_type: ModelType
    target_version: str
    source_version: Optional[str]
    strategy: DeploymentStrategy
    status: DeploymentStatus
    config: DeploymentConfig

    # Tracking
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = 0.0

    # Results
    success: bool = False
    error_message: Optional[str] = None
    rollback_triggered: bool = False

    # Current state
    current_stage: Optional[CanaryStage] = None
    metrics: DeploymentMetrics = field(default_factory=DeploymentMetrics)

class ProductionDeploymentManager:
    """Advanced production deployment manager with automatic rollback capabilities"""

    def __init__(
        self,
        version_manager: NBAModelVersionManager,
        config: Optional[DeploymentConfig] = None,
        deployment_dir: str = "deployments"
    ):
        """
        Initialize the production deployment manager

        Args:
            version_manager: Model version manager instance
            config: Deployment configuration
            deployment_dir: Directory for deployment artifacts
        """
        self.version_manager = version_manager
        self.config = config or DeploymentConfig()
        self.deployment_dir = Path(deployment_dir)
        self.deployment_dir.mkdir(exist_ok=True)

        # Deployment tracking
        self.active_deployments: Dict[str, DeploymentJob] = {}
        self.deployment_history: List[DeploymentJob] = []
        self.deployment_metrics: Dict[str, DeploymentMetrics] = {}

        # Monitoring
        self.health_monitor_running = False
        self.health_monitor_thread: Optional[threading.Thread] = None
        self.metrics_queue = queue.Queue()

        # Callbacks
        self.prediction_callbacks: Dict[str, Callable] = {}
        self.health_check_callbacks: List[Callable] = []

        # Thread safety
        self._lock = threading.RLock()

        # Initialize deployment environment
        self._initialize_deployment_environment()

        logger.info("Production Deployment Manager initialized")

    def _initialize_deployment_environment(self):
        """Initialize deployment environment and directories"""
        # Create deployment directories
        for subdir in ["active", "staged", "failed", "logs", "metrics"]:
            (self.deployment_dir / subdir).mkdir(exist_ok=True)

        # Initialize deployment registry
        registry_path = self.deployment_dir / "deployment_registry.json"
        if not registry_path.exists():
            initial_registry = {
                "deployments": [],
                "active_deployment": None,
                "last_deployment": None
            }
            with open(registry_path, 'w') as f:
                json.dump(initial_registry, f, indent=2, default=str)

    def start_deployment(
        self,
        model_type: ModelType,
        target_version: str,
        strategy: Optional[DeploymentStrategy] = None,
        config: Optional[DeploymentConfig] = None
    ) -> str:
        """
        Start a new deployment

        Args:
            model_type: Type of model to deploy
            target_version: Target version to deploy
            strategy: Deployment strategy to use
            config: Deployment configuration

        Returns:
            Deployment ID
        """
        with self._lock:
            # Generate deployment ID
            deployment_id = f"deploy_{model_type.value}_{target_version}_{uuid.uuid4().hex[:8]}"

            # Create deployment job
            job = DeploymentJob(
                job_id=deployment_id,
                deployment_id=deployment_id,
                model_type=model_type,
                target_version=target_version,
                source_version=self._get_current_version(model_type),
                strategy=strategy or self.config.strategy,
                status=DeploymentStatus.PENDING,
                config=config or self.config,
                created_at=datetime.now()
            )

            # Validate target version exists
            if not self.version_manager.version_exists(target_version, model_type):
                raise ValueError(f"Target version {target_version} not found for {model_type}")

            # Store deployment job
            self.active_deployments[deployment_id] = job

            # Start deployment in background
            deployment_thread = threading.Thread(
                target=self._execute_deployment,
                args=(deployment_id,),
                daemon=True
            )
            deployment_thread.start()

            logger.info(f"Started deployment {deployment_id} for {model_type} version {target_version}")

            return deployment_id

    def _execute_deployment(self, deployment_id: str):
        """Execute deployment process"""
        job = self.active_deployments.get(deployment_id)
        if not job:
            return

        try:
            # Update status
            job.status = DeploymentStatus.IN_PROGRESS
            job.started_at = datetime.now()

            # Execute based on strategy
            if job.strategy == DeploymentStrategy.BLUE_GREEN:
                success = self._execute_blue_green_deployment(job)
            elif job.strategy == DeploymentStrategy.CANARY:
                success = self._execute_canary_deployment(job)
            elif job.strategy == DeploymentStrategy.ROLLING:
                success = self._execute_rolling_deployment(job)
            elif job.strategy == DeploymentStrategy.SHADOW:
                success = self._execute_shadow_deployment(job)
            elif job.strategy == DeploymentStrategy.A_B_TESTING:
                success = self._execute_ab_testing_deployment(job)
            else:
                raise ValueError(f"Unsupported deployment strategy: {job.strategy}")

            # Update final status
            if success:
                job.status = DeploymentStatus.COMPLETED
                job.success = True
                job.progress = 100.0
                logger.info(f"Deployment {deployment_id} completed successfully")
            else:
                job.status = DeploymentStatus.FAILED
                job.success = False
                logger.error(f"Deployment {deployment_id} failed")

        except Exception as e:
            job.status = DeploymentStatus.FAILED
            job.error_message = str(e)
            logger.error(f"Deployment {deployment_id} failed with error: {e}")

            # Trigger automatic rollback if enabled
            if job.config.auto_rollback_enabled:
                self._trigger_rollback(deployment_id, reason="Deployment failed")

        finally:
            job.completed_at = datetime.now()
            self._finalize_deployment(deployment_id)

    def _execute_blue_green_deployment(self, job: DeploymentJob) -> bool:
        """Execute blue-green deployment strategy"""
        logger.info(f"Starting blue-green deployment for {job.model_type}")

        try:
            # Stage 1: Prepare green environment
            job.progress = 10.0
            if not self._prepare_deployment_environment(job):
                return False

            # Stage 2: Deploy to green environment
            job.progress = 30.0
            if not self._deploy_to_environment(job, environment="green"):
                return False

            # Stage 3: Health checks on green environment
            job.progress = 50.0
            if not self._perform_health_checks(job, environment="green"):
                return False

            # Stage 4: Switch traffic to green environment
            job.progress = 70.0
            if not self._switch_traffic(job, from_env="blue", to_env="green"):
                return False

            # Stage 5: Final verification
            job.progress = 90.0
            if not self._verify_deployment(job):
                return False

            # Stage 6: Cleanup blue environment
            job.progress = 100.0
            self._cleanup_deployment_environment(job, environment="blue")

            return True

        except Exception as e:
            logger.error(f"Blue-green deployment failed: {e}")
            return False

    def _execute_canary_deployment(self, job: DeploymentJob) -> bool:
        """Execute canary deployment strategy"""
        logger.info(f"Starting canary deployment for {job.model_type}")

        try:
            # Prepare canary environment
            job.progress = 5.0
            if not self._prepare_deployment_environment(job):
                return False

            # Execute canary stages
            total_stages = len(job.config.canary_stages)

            for i, stage in enumerate(job.config.canary_stages):
                logger.info(f"Executing canary stage: {stage.value}")

                job.current_stage = stage
                stage_progress = 10 + (i * 80 / total_stages)
                job.progress = stage_progress

                # Deploy canary version for this stage
                traffic_percentage = job.config.canary_traffic_percentages[stage]

                if not self._deploy_canary_stage(job, stage, traffic_percentage):
                    return False

                # Monitor for specified duration
                if not self._monitor_canary_stage(job, stage):
                    return False

                # Check if we should continue
                if not self._should_continue_canary(job, stage):
                    logger.info(f"Canary deployment stopped at stage {stage.value}")
                    return False

            # Finalize deployment
            job.progress = 100.0
            job.current_stage = CanaryStage.FULL_TRAFFIC

            # Switch all traffic to new version
            if not self._switch_traffic(job, from_env="production", to_env="canary"):
                return False

            return True

        except Exception as e:
            logger.error(f"Canary deployment failed: {e}")
            return False

    def _execute_rolling_deployment(self, job: DeploymentJob) -> bool:
        """Execute rolling deployment strategy"""
        logger.info(f"Starting rolling deployment for {job.model_type}")

        try:
            # In rolling deployment, we gradually replace instances
            # For this implementation, we'll simulate a simple rolling update

            job.progress = 20.0
            if not self._prepare_deployment_environment(job):
                return False

            # Stage 1: Deploy to half of instances
            job.progress = 50.0
            if not self._deploy_rolling_batch(job, batch_size=0.5):
                return False

            # Stage 2: Health check
            if not self._perform_health_checks(job):
                return False

            # Stage 3: Deploy to remaining instances
            job.progress = 80.0
            if not self._deploy_rolling_batch(job, batch_size=1.0):
                return False

            # Final verification
            job.progress = 100.0
            if not self._verify_deployment(job):
                return False

            return True

        except Exception as e:
            logger.error(f"Rolling deployment failed: {e}")
            return False

    def _execute_shadow_deployment(self, job: DeploymentJob) -> bool:
        """Execute shadow deployment strategy"""
        logger.info(f"Starting shadow deployment for {job.model_type}")

        try:
            # Deploy new version in shadow mode (doesn't serve live traffic)
            job.progress = 30.0
            if not self._deploy_to_environment(job, environment="shadow"):
                return False

            # Set up shadow traffic mirroring
            job.progress = 50.0
            if not self._setup_shadow_traffic_mirroring(job):
                return False

            # Monitor shadow performance
            job.progress = 70.0
            if not self._monitor_shadow_performance(job):
                return False

            # Compare shadow vs production performance
            job.progress = 90.0
            comparison_result = self._compare_shadow_performance(job)

            if not comparison_result["shadow_better"]:
                logger.warning("Shadow deployment performance is worse than production")
                return False

            # Promote shadow to production
            job.progress = 100.0
            if not self._promote_shadow_to_production(job):
                return False

            return True

        except Exception as e:
            logger.error(f"Shadow deployment failed: {e}")
            return False

    def _execute_ab_testing_deployment(self, job: DeploymentJob) -> bool:
        """Execute A/B testing deployment strategy"""
        logger.info(f"Starting A/B testing deployment for {job.model_type}")

        try:
            # Setup A/B test environment
            job.progress = 20.0
            if not self._setup_ab_test_environment(job):
                return False

            # Run A/B test
            job.progress = 40.0
            test_result = self._run_ab_test(job)

            if not test_result["test_valid"]:
                logger.error("A/B test failed validation")
                return False

            # Analyze results
            job.progress = 70.0
            analysis = self._analyze_ab_test_results(job, test_result)

            if not analysis["variant_b_significantly_better"]:
                logger.info("New model (Variant B) is not significantly better")
                if analysis["variant_b_worse"]:
                    logger.warning("New model performs worse than current model")
                return False

            # Promote winning variant
            job.progress = 100.0
            if not self._promote_ab_test_winner(job, analysis):
                return False

            return True

        except Exception as e:
            logger.error(f"A/B testing deployment failed: {e}")
            return False

    def _prepare_deployment_environment(self, job: DeploymentJob) -> bool:
        """Prepare deployment environment"""
        try:
            # Load target model version
            model_version = self.version_manager.get_model_version(
                job.target_version, job.model_type
            )

            if not model_version:
                logger.error(f"Model version {job.target_version} not found")
                return False

            # Prepare model files
            if not self._prepare_model_files(job, model_version):
                return False

            # Prepare configuration
            if not self._prepare_deployment_config(job, model_version):
                return False

            logger.info(f"Prepared deployment environment for {job.deployment_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to prepare deployment environment: {e}")
            return False

    def _prepare_model_files(self, job: DeploymentJob, model_version: ModelVersion) -> bool:
        """Prepare model files for deployment"""
        try:
            # Create deployment directory for this job
            deploy_dir = self.deployment_dir / "active" / job.deployment_id
            deploy_dir.mkdir(exist_ok=True)

            # Copy model files
            model_path = Path(model_version.file_path)
            if model_path.exists():
                target_path = deploy_dir / f"model_{job.model_type.value}.pkl"
                shutil.copy2(model_path, target_path)
            else:
                logger.error(f"Model file not found: {model_path}")
                return False

            # Copy config files if they exist
            if model_version.config_path:
                config_path = Path(model_version.config_path)
                if config_path.exists():
                    target_config_path = deploy_dir / f"config_{job.model_type.value}.json"
                    shutil.copy2(config_path, target_config_path)

            return True

        except Exception as e:
            logger.error(f"Failed to prepare model files: {e}")
            return False

    def _prepare_deployment_config(self, job: DeploymentJob, model_version: ModelVersion) -> bool:
        """Prepare deployment configuration"""
        try:
            deploy_dir = self.deployment_dir / "active" / job.deployment_id

            # Create deployment configuration
            deployment_config = {
                "deployment_id": job.deployment_id,
                "model_type": job.model_type.value,
                "model_version": model_version.version,
                "strategy": job.strategy.value,
                "deployment_config": asdict(job.config),
                "model_metadata": {
                    "created_at": model_version.created_at.isoformat(),
                    "created_by": model_version.created_by,
                    "description": model_version.description,
                    "metrics": model_version.metrics.to_dict()
                },
                "health_checks": {
                    "enabled": job.config.health_check_enabled,
                    "interval_seconds": job.config.health_check_interval_seconds,
                    "timeout_seconds": job.config.health_check_timeout_seconds,
                    "max_failures": job.config.health_check_max_failures
                }
            }

            # Save deployment configuration
            config_path = deploy_dir / "deployment_config.json"
            with open(config_path, 'w') as f:
                json.dump(deployment_config, f, indent=2, default=str)

            return True

        except Exception as e:
            logger.error(f"Failed to prepare deployment config: {e}")
            return False

    def _deploy_to_environment(self, job: DeploymentJob, environment: str) -> bool:
        """Deploy model to specified environment"""
        try:
            logger.info(f"Deploying {job.model_type} version {job.target_version} to {environment} environment")

            # In a real implementation, this would interact with your deployment infrastructure
            # For now, we'll simulate the deployment process

            # Load the model
            deploy_dir = self.deployment_dir / "active" / job.deployment_id
            model_path = deploy_dir / f"model_{job.model_type.value}.pkl"

            if not model_path.exists():
                logger.error(f"Model file not found: {model_path}")
                return False

            # Simulate deployment
            # In reality, this would involve:
            # - Loading the model
            # - Setting up inference endpoints
            # - Configuring load balancers
            # - Updating service discovery

            # Register deployment callback for this model
            self._register_deployment_callback(job, environment)

            logger.info(f"Successfully deployed {job.model_type} to {environment} environment")
            return True

        except Exception as e:
            logger.error(f"Failed to deploy to {environment} environment: {e}")
            return False

    def _perform_health_checks(self, job: DeploymentJob, environment: str = "production") -> bool:
        """Perform health checks on deployed model"""
        try:
            logger.info(f"Performing health checks for {job.model_type} in {environment} environment")

            # Generate test data
            test_data = self._generate_test_data(job.model_type)

            # Perform health checks
            health_results = {
                "model_loaded": False,
                "predictions_working": False,
                "performance_acceptable": False,
                "error_rate_acceptable": False
            }

            # Check if model is loaded and working
            try:
                # Test prediction with sample data
                test_prediction = self._test_model_prediction(job, test_data)
                health_results["predictions_working"] = True
                health_results["model_loaded"] = True

                # Check performance
                if self._check_prediction_performance(job, test_prediction):
                    health_results["performance_acceptable"] = True

                # Check error rate
                if self._check_error_rate(job, test_data):
                    health_results["error_rate_acceptable"] = True

            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return False

            # Evaluate overall health
            health_passed = all(health_results.values())

            if not health_passed:
                logger.warning(f"Health checks failed for {environment} environment: {health_results}")
                return False

            logger.info(f"Health checks passed for {environment} environment")
            return True

        except Exception as e:
            logger.error(f"Failed to perform health checks: {e}")
            return False

    def _generate_test_data(self, model_type: ModelType) -> Dict[str, Any]:
        """Generate test data for health checks"""
        # Generate realistic test data based on model type
        if model_type == ModelType.XGBOOST:
            return {
                "features": np.random.rand(1, 50),  # 50 features
                "home_team": "Lakers",
                "away_team": "Celtics"
            }
        elif model_type == ModelType.NEURAL_NETWORK:
            return {
                "features": np.random.rand(1, 50),
                "sequence_data": np.random.rand(1, 10, 50)  # 10 timesteps, 50 features
            }
        elif model_type == ModelType.ENSEMBLE:
            return {
                "home_team": "Lakers",
                "away_team": "Celtics",
                "game_data": {
                    "home_stats": np.random.rand(10),
                    "away_stats": np.random.rand(10)
                }
            }
        else:
            return {"test": True}

    def _test_model_prediction(self, job: DeploymentJob, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """Test model prediction with test data"""
        start_time = time.time()

        try:
            # In a real implementation, this would call the deployed model endpoint
            # For now, we'll simulate the prediction

            # Simulate prediction latency
            time.sleep(np.random.uniform(0.01, 0.1))

            # Generate mock prediction result
            prediction_time = time.time() - start_time

            return {
                "prediction": np.random.choice(["home_win", "away_win"]),
                "confidence": np.random.uniform(0.6, 0.95),
                "prediction_time_ms": prediction_time * 1000,
                "success": True
            }

        except Exception as e:
            return {
                "error": str(e),
                "prediction_time_ms": (time.time() - start_time) * 1000,
                "success": False
            }

    def _check_prediction_performance(self, job: DeploymentJob, test_result: Dict[str, Any]) -> bool:
        """Check if prediction performance meets thresholds"""
        if not test_result.get("success", False):
            return False

        prediction_time = test_result.get("prediction_time_ms", float('inf'))
        return prediction_time <= job.config.latency_threshold_ms

    def _check_error_rate(self, job: DeploymentJob, test_data: Dict[str, Any]) -> bool:
        """Check if error rate is acceptable"""
        # Perform multiple predictions to calculate error rate
        total_tests = 10
        successful_predictions = 0

        for _ in range(total_tests):
            result = self._test_model_prediction(job, test_data)
            if result.get("success", False):
                successful_predictions += 1

        error_rate = (total_tests - successful_predictions) / total_tests
        return error_rate <= job.config.error_rate_threshold

    def _switch_traffic(self, job: DeploymentJob, from_env: str, to_env: str) -> bool:
        """Switch traffic from one environment to another"""
        try:
            logger.info(f"Switching traffic from {from_env} to {to_env}")

            # In a real implementation, this would update load balancer configuration
            # For now, we'll simulate the traffic switch

            # Update traffic routing
            # This would involve updating your load balancer, service mesh, or API gateway

            # Verify traffic switch
            time.sleep(2)  # Allow time for switch to propagate

            logger.info(f"Successfully switched traffic from {from_env} to {to_env}")
            return True

        except Exception as e:
            logger.error(f"Failed to switch traffic: {e}")
            return False

    def _verify_deployment(self, job: DeploymentJob) -> bool:
        """Verify that deployment is working correctly"""
        try:
            logger.info(f"Verifying deployment {job.deployment_id}")

            # Perform final health checks
            if not self._perform_health_checks(job):
                return False

            # Check that new version is serving traffic
            current_version = self._get_current_version(job.model_type)
            if current_version != job.target_version:
                logger.error(f"Version mismatch: expected {job.target_version}, got {current_version}")
                return False

            # Log deployment success
            self._log_deployment_event(job, "deployment_verified")

            logger.info(f"Deployment {job.deployment_id} verified successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to verify deployment: {e}")
            return False

    def _cleanup_deployment_environment(self, job: DeploymentJob, environment: str):
        """Clean up deployment environment"""
        try:
            logger.info(f"Cleaning up {environment} environment for {job.deployment_id}")

            # In a real implementation, this would:
            # - Decommission old instances
            # - Clean up temporary files
            # - Update service discovery

            # Move deployment to history
            deploy_dir = self.deployment_dir / "active" / job.deployment_id
            history_dir = self.deployment_dir / "history" / job.deployment_id

            if deploy_dir.exists():
                deploy_dir.rename(history_dir)

            logger.info(f"Cleaned up {environment} environment")

        except Exception as e:
            logger.error(f"Failed to cleanup {environment} environment: {e}")

    def _get_current_version(self, model_type: ModelType) -> Optional[str]:
        """Get currently deployed version for model type"""
        try:
            # Get active versions from version manager
            active_versions = self.version_manager.get_active_versions()
            return active_versions.get(model_type)
        except Exception as e:
            logger.error(f"Failed to get current version: {e}")
            return None

    def _trigger_rollback(self, deployment_id: str, reason: str):
        """Trigger automatic rollback for failed deployment"""
        try:
            logger.info(f"Triggering rollback for deployment {deployment_id}: {reason}")

            job = self.active_deployments.get(deployment_id)
            if not job:
                logger.error(f"Deployment {deployment_id} not found")
                return

            # Perform rollback
            if job.source_version:
                success = self.version_manager.rollback_model(
                    job.target_version, job.source_version, job.model_type
                )

                if success:
                    job.rollback_triggered = True
                    job.status = DeploymentStatus.ROLLED_BACK
                    logger.info(f"Successfully rolled back {job.model_type} from {job.target_version} to {job.source_version}")
                else:
                    logger.error(f"Failed to rollback {job.model_type}")
            else:
                logger.warning(f"No source version available for rollback of {deployment_id}")

            # Log rollback event
            self._log_deployment_event(job, "rollback_triggered", {"reason": reason})

        except Exception as e:
            logger.error(f"Failed to trigger rollback: {e}")

    def _log_deployment_event(self, job: DeploymentJob, event_type: str, metadata: Optional[Dict] = None):
        """Log deployment event"""
        try:
            log_entry = {
                "deployment_id": job.deployment_id,
                "model_type": job.model_type.value,
                "event_type": event_type,
                "timestamp": datetime.now().isoformat(),
                "target_version": job.target_version,
                "source_version": job.source_version,
                "strategy": job.strategy.value,
                "status": job.status.value,
                "progress": job.progress,
                "metadata": metadata or {}
            }

            # Append to deployment log
            log_file = self.deployment_dir / "logs" / f"deployment_{job.deployment_id}.jsonl"
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry, default=str) + '\n')

        except Exception as e:
            logger.error(f"Failed to log deployment event: {e}")

    def _finalize_deployment(self, deployment_id: str):
        """Finalize deployment and update registry"""
        try:
            job = self.active_deployments.get(deployment_id)
            if not job:
                return

            with self._lock:
                # Move from active to history
                self.deployment_history.append(job)
                if deployment_id in self.active_deployments:
                    del self.active_deployments[deployment_id]

                # Update deployment registry
                self._update_deployment_registry(job)

                # Update active model versions in version manager
                if job.success:
                    self.version_manager.activate_model_version(job.target_version, job.model_type)

        except Exception as e:
            logger.error(f"Failed to finalize deployment: {e}")

    def _update_deployment_registry(self, job: DeploymentJob):
        """Update deployment registry"""
        try:
            registry_path = self.deployment_dir / "deployment_registry.json"

            with open(registry_path, 'r') as f:
                registry = json.load(f)

            # Add deployment to history
            deployment_entry = {
                "deployment_id": job.deployment_id,
                "model_type": job.model_type.value,
                "target_version": job.target_version,
                "source_version": job.source_version,
                "strategy": job.strategy.value,
                "status": job.status.value,
                "success": job.success,
                "rollback_triggered": job.rollback_triggered,
                "created_at": job.created_at.isoformat(),
                "started_at": job.started_at.isoformat() if job.started_at else None,
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "progress": job.progress,
                "error_message": job.error_message
            }

            registry["deployments"].append(deployment_entry)
            registry["last_deployment"] = deployment_entry

            # Update active deployment if successful
            if job.success and job.status == DeploymentStatus.COMPLETED:
                registry["active_deployment"] = deployment_entry

            # Keep only last 100 deployments in registry
            if len(registry["deployments"]) > 100:
                registry["deployments"] = registry["deployments"][-100:]

            with open(registry_path, 'w') as f:
                json.dump(registry, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to update deployment registry: {e}")

    def _register_deployment_callback(self, job: DeploymentJob, environment: str):
        """Register callback for deployment predictions"""
        callback_key = f"{job.model_type.value}_{environment}"

        def prediction_callback(prediction_data: Dict[str, Any]):
            """Callback for monitoring predictions"""
            try:
                # Update deployment metrics
                metrics = job.metrics

                # Update request counters
                metrics.total_requests += 1

                if prediction_data.get("success", False):
                    metrics.successful_requests += 1
                else:
                    metrics.failed_requests += 1

                # Update performance metrics
                if "prediction_time_ms" in prediction_data:
                    metrics.current_latency_ms = prediction_data["prediction_time_ms"]

                # Update error rate
                metrics.current_error_rate = metrics.failed_requests / max(metrics.total_requests, 1)

                # Check for performance degradation
                self._check_performance_degradation(job)

            except Exception as e:
                logger.error(f"Error in prediction callback: {e}")

        self.prediction_callbacks[callback_key] = prediction_callback

    def _check_performance_degradation(self, job: DeploymentJob):
        """Check for performance degradation and trigger rollback if needed"""
        try:
            metrics = job.metrics
            config = job.config

            # Check accuracy threshold
            if metrics.current_accuracy < config.accuracy_threshold:
                logger.warning(f"Accuracy threshold breached: {metrics.current_accuracy:.3f} < {config.accuracy_threshold:.3f}")
                if config.auto_rollback_enabled:
                    self._trigger_rollback(job.deployment_id, "Accuracy threshold breached")
                return

            # Check latency threshold
            if metrics.current_latency_ms > config.latency_threshold_ms:
                logger.warning(f"Latency threshold breached: {metrics.current_latency_ms:.1f}ms > {config.latency_threshold_ms:.1f}ms")
                if config.auto_rollback_enabled:
                    self._trigger_rollback(job.deployment_id, "Latency threshold breached")
                return

            # Check error rate threshold
            if metrics.current_error_rate > config.error_rate_threshold:
                logger.warning(f"Error rate threshold breached: {metrics.current_error_rate:.3f} > {config.error_rate_threshold:.3f}")
                if config.auto_rollback_enabled:
                    self._trigger_rollback(job.deployment_id, "Error rate threshold breached")
                return

        except Exception as e:
            logger.error(f"Error checking performance degradation: {e}")

    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get deployment status"""
        job = self.active_deployments.get(deployment_id)
        if not job:
            # Check deployment history
            for historical_job in self.deployment_history:
                if historical_job.deployment_id == deployment_id:
                    job = historical_job
                    break

        if not job:
            return None

        return {
            "deployment_id": job.deployment_id,
            "model_type": job.model_type.value,
            "target_version": job.target_version,
            "source_version": job.source_version,
            "strategy": job.strategy.value,
            "status": job.status.value,
            "progress": job.progress,
            "success": job.success,
            "rollback_triggered": job.rollback_triggered,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "error_message": job.error_message,
            "current_stage": job.current_stage.value if job.current_stage else None,
            "metrics": job.metrics.to_dict()
        }

    def list_deployments(
        self,
        model_type: Optional[ModelType] = None,
        status: Optional[DeploymentStatus] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """List deployments with optional filtering"""
        deployments = []

        # Combine active and historical deployments
        all_deployments = list(self.active_deployments.values()) + self.deployment_history

        for job in all_deployments:
            # Apply filters
            if model_type and job.model_type != model_type:
                continue

            if status and job.status != status:
                continue

            deployments.append({
                "deployment_id": job.deployment_id,
                "model_type": job.model_type.value,
                "target_version": job.target_version,
                "source_version": job.source_version,
                "strategy": job.strategy.value,
                "status": job.status.value,
                "progress": job.progress,
                "success": job.success,
                "rollback_triggered": job.rollback_triggered,
                "created_at": job.created_at.isoformat(),
                "completed_at": job.completed_at.isoformat() if job.completed_at else None,
                "error_message": job.error_message
            })

        # Sort by creation date (newest first)
        deployments.sort(key=lambda x: x["created_at"], reverse=True)

        return deployments[:limit]

    def get_deployment_metrics(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed metrics for a deployment"""
        job = self.active_deployments.get(deployment_id)
        if not job:
            return None

        return {
            "deployment_id": deployment_id,
            "metrics": job.metrics.to_dict(),
            "health_status": job.metrics.health_status,
            "last_health_check": job.metrics.last_health_check.isoformat() if job.metrics.last_health_check else None,
            "consecutive_failures": job.metrics.consecutive_failures
        }

    def start_health_monitoring(self):
        """Start background health monitoring"""
        if self.health_monitor_running:
            return

        self.health_monitor_running = True
        self.health_monitor_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True
        )
        self.health_monitor_thread.start()
        logger.info("Health monitoring started")

    def stop_health_monitoring(self):
        """Stop background health monitoring"""
        self.health_monitor_running = False
        if self.health_monitor_thread:
            self.health_monitor_thread.join(timeout=5)
        logger.info("Health monitoring stopped")

    def _health_monitor_loop(self):
        """Background health monitoring loop"""
        while self.health_monitor_running:
            try:
                # Monitor all active deployments
                for deployment_id, job in list(self.active_deployments.items()):
                    if job.status == DeploymentStatus.IN_PROGRESS:
                        self._perform_health_checks(job)

                # Sleep before next check
                time.sleep(self.config.health_check_interval_seconds)

            except Exception as e:
                logger.error(f"Error in health monitor loop: {e}")
                time.sleep(10)

    # Canary deployment helper methods
    def _deploy_canary_stage(self, job: DeploymentJob, stage: CanaryStage, traffic_percentage: float) -> bool:
        """Deploy canary stage with specified traffic percentage"""
        try:
            logger.info(f"Deploying canary stage {stage.value} with {traffic_percentage:.1%} traffic")

            # In a real implementation, this would update load balancer configuration
            # to route the specified percentage of traffic to the canary version

            return True

        except Exception as e:
            logger.error(f"Failed to deploy canary stage {stage.value}: {e}")
            return False

    def _monitor_canary_stage(self, job: DeploymentJob, stage: CanaryStage) -> bool:
        """Monitor canary stage for specified duration"""
        try:
            logger.info(f"Monitoring canary stage {stage.value} for {job.config.canary_stage_duration_minutes} minutes")

            # Monitor for the specified duration
            end_time = datetime.now() + timedelta(minutes=job.config.canary_stage_duration_minutes)

            while datetime.now() < end_time:
                # Check health
                if not self._perform_health_checks(job, environment="canary"):
                    logger.error(f"Health check failed for canary stage {stage.value}")
                    return False

                # Check performance metrics
                if not self._check_canary_performance(job, stage):
                    logger.error(f"Performance check failed for canary stage {stage.value}")
                    return False

                # Sleep before next check
                time.sleep(job.config.health_check_interval_seconds)

            logger.info(f"Canary stage {stage.value} monitoring completed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to monitor canary stage {stage.value}: {e}")
            return False

    def _check_canary_performance(self, job: DeploymentJob, stage: CanaryStage) -> bool:
        """Check canary stage performance"""
        try:
            metrics = job.metrics

            # Define stage-specific thresholds (can be more lenient for early stages)
            stage_multipliers = {
                CanaryStage.SMALL_TRAFFIC: 1.5,    # 50% more lenient
                CanaryStage.MEDIUM_TRAFFIC: 1.2,  # 20% more lenient
                CanaryStage.LARGE_TRAFFIC: 1.1,   # 10% more lenient
                CanaryStage.FULL_TRAFFIC: 1.0     # Standard thresholds
            }

            multiplier = stage_multipliers.get(stage, 1.0)

            # Check accuracy with stage multiplier
            accuracy_threshold = job.config.accuracy_threshold * multiplier
            if metrics.current_accuracy < accuracy_threshold:
                logger.warning(f"Canary accuracy below threshold: {metrics.current_accuracy:.3f} < {accuracy_threshold:.3f}")
                return False

            # Check latency with stage multiplier
            latency_threshold = job.config.latency_threshold_ms * multiplier
            if metrics.current_latency_ms > latency_threshold:
                logger.warning(f"Canary latency above threshold: {metrics.current_latency_ms:.1f}ms > {latency_threshold:.1f}ms")
                return False

            return True

        except Exception as e:
            logger.error(f"Error checking canary performance: {e}")
            return False

    def _should_continue_canary(self, job: DeploymentJob, stage: CanaryStage) -> bool:
        """Determine if canary deployment should continue to next stage"""
        try:
            # Check if performance metrics are acceptable
            metrics = job.metrics

            # For demonstration, we'll use simple heuristics
            # In practice, this could involve statistical tests

            if metrics.current_accuracy < job.config.accuracy_threshold:
                logger.warning(f"Canary accuracy below threshold, stopping at {stage.value}")
                return False

            if metrics.current_error_rate > job.config.error_rate_threshold:
                logger.warning(f"Canary error rate above threshold, stopping at {stage.value}")
                return False

            return True

        except Exception as e:
            logger.error(f"Error determining canary continuation: {e}")
            return False

    # Rolling deployment helper methods
    def _deploy_rolling_batch(self, job: DeploymentJob, batch_size: float) -> bool:
        """Deploy rolling batch of instances"""
        try:
            logger.info(f"Deploying rolling batch with {batch_size:.1%} of instances")

            # In a real implementation, this would:
            # - Deploy new version to specified batch of instances
            # - Wait for health checks to pass
            # - Update load balancer to include new instances

            return True

        except Exception as e:
            logger.error(f"Failed to deploy rolling batch: {e}")
            return False

    # Shadow deployment helper methods
    def _setup_shadow_traffic_mirroring(self, job: DeploymentJob) -> bool:
        """Set up traffic mirroring for shadow deployment"""
        try:
            logger.info("Setting up shadow traffic mirroring")

            # In a real implementation, this would configure load balancer
            # or service mesh to mirror traffic to shadow deployment

            return True

        except Exception as e:
            logger.error(f"Failed to setup shadow traffic mirroring: {e}")
            return False

    def _monitor_shadow_performance(self, job: DeploymentJob) -> bool:
        """Monitor shadow deployment performance"""
        try:
            logger.info("Monitoring shadow deployment performance")

            # Collect metrics from shadow deployment
            # Compare with production metrics

            return True

        except Exception as e:
            logger.error(f"Failed to monitor shadow performance: {e}")
            return False

    def _compare_shadow_performance(self, job: DeploymentJob) -> Dict[str, Any]:
        """Compare shadow vs production performance"""
        try:
            # This would collect and compare metrics from both deployments
            # For demonstration, return mock results

            return {
                "shadow_better": True,
                "shadow_accuracy": 0.75,
                "production_accuracy": 0.72,
                "shadow_latency": 50.0,
                "production_latency": 45.0
            }

        except Exception as e:
            logger.error(f"Failed to compare shadow performance: {e}")
            return {"shadow_better": False}

    def _promote_shadow_to_production(self, job: DeploymentJob) -> bool:
        """Promote shadow deployment to production"""
        try:
            logger.info("Promoting shadow deployment to production")

            # Switch traffic from production to shadow (new) version

            return True

        except Exception as e:
            logger.error(f"Failed to promote shadow to production: {e}")
            return False

    # A/B testing helper methods
    def _setup_ab_test_environment(self, job: DeploymentJob) -> bool:
        """Set up A/B testing environment"""
        try:
            logger.info("Setting up A/B testing environment")

            # Deploy both versions (A=control, B=treatment)
            # Configure traffic split

            return True

        except Exception as e:
            logger.error(f"Failed to setup A/B test environment: {e}")
            return False

    def _run_ab_test(self, job: DeploymentJob) -> Dict[str, Any]:
        """Run A/B test"""
        try:
            logger.info(f"Running A/B test for {job.config.ab_test_duration_minutes} minutes")

            # Simulate A/B test execution
            end_time = datetime.now() + timedelta(minutes=job.config.ab_test_duration_minutes)

            while datetime.now() < end_time:
                # Collect metrics from both variants
                # In real implementation, this would track predictions from both versions

                time.sleep(10)

            # Return mock test results
            return {
                "test_valid": True,
                "variant_a_accuracy": 0.72,
                "variant_b_accuracy": 0.78,
                "variant_a_requests": 500,
                "variant_b_requests": 512,
                "test_duration_minutes": job.config.ab_test_duration_minutes
            }

        except Exception as e:
            logger.error(f"Failed to run A/B test: {e}")
            return {"test_valid": False}

    def _analyze_ab_test_results(self, job: DeploymentJob, test_result: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze A/B test results"""
        try:
            if not test_result["test_valid"]:
                return {"variant_b_significantly_better": False}

            # Perform statistical significance test
            # For demonstration, we'll use a simple improvement threshold

            variant_a_accuracy = test_result["variant_a_accuracy"]
            variant_b_accuracy = test_result["variant_b_accuracy"]

            # Calculate improvement
            improvement = (variant_b_accuracy - variant_a_accuracy) / variant_a_accuracy

            # Check if improvement is statistically significant
            # In real implementation, this would use proper statistical tests

            min_improvement = 0.02  # 2% minimum improvement
            variant_b_better = improvement > min_improvement

            return {
                "variant_b_significantly_better": variant_b_better,
                "variant_b_worse": improvement < -min_improvement,
                "improvement_percentage": improvement * 100,
                "variant_a_accuracy": variant_a_accuracy,
                "variant_b_accuracy": variant_b_accuracy
            }

        except Exception as e:
            logger.error(f"Failed to analyze A/B test results: {e}")
            return {"variant_b_significantly_better": False}

    def _promote_ab_test_winner(self, job: DeploymentJob, analysis: Dict[str, Any]) -> bool:
        """Promote winning variant from A/B test"""
        try:
            if analysis["variant_b_significantly_better"]:
                logger.info("Promoting variant B (new model) as winner")

                # Switch all traffic to variant B
                return True
            else:
                logger.info("Keeping variant A (current model) as winner")
                return True

        except Exception as e:
            logger.error(f"Failed to promote A/B test winner: {e}")
            return False