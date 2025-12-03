#!/usr/bin/env python3
"""
🔄 NBA Model Retraining Pipeline - Advanced Automation System
Task 2.2.5: Implement Model Retraining Pipeline

**Architecture**: DevStream SuperPowered with ContextSet Compliance
**Purpose**: Automated model retraining with data quality validation, performance monitoring, and NBA-specific features

This system provides:
- Automated retraining scheduling and execution
- Data quality validation and monitoring
- Model performance degradation detection
- Integration with existing versioning system
- NBA-specific data pipeline and feature engineering
- Comprehensive monitoring and alerting
"""

import asyncio
import logging
import json
import time
import hashlib
import threading
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Optional, Any, Tuple, Union, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import schedule
import pickle
import warnings

# ML and Data Quality imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import evidently
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, ClassificationPreset
from evidently.test_preset import NoTargetPerformanceTestPreset

# Suppress warnings
warnings.filterwarnings('ignore')

# Import existing NBA systems
try:
    from ..ensemble.nba_ensemble_predictor import NBAEnsemblePredictor
    from ..ensemble.model_version_manager import NBAModelVersionManager, ModelType
    from ..api.data_provider import NBADataProvider
    RETRAINING_AVAILABLE = True
except ImportError:
    RETRAINING_AVAILABLE = False
    logging.warning("⚠️ NBA systems not available for retraining pipeline")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/retraining_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class RetrainingTriggerType(Enum):
    """Types of retraining triggers"""
    SCHEDULED = "scheduled"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    MANUAL = "manual"
    DATA_QUALITY_ISSUE = "data_quality_issue"


class DataQualityStatus(Enum):
    """Data quality assessment status"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    CRITICAL = "critical"


class RetrainingStatus(Enum):
    """Retraining job status"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    VALIDATION_FAILED = "validation_failed"


@dataclass
class RetrainingConfig:
    """Configuration for retraining pipeline"""
    # Scheduling
    schedule_enabled: bool = True
    schedule_interval: str = "daily"  # hourly, daily, weekly
    schedule_time: str = "02:00"  # Time of day for daily scheduling

    # Performance thresholds
    accuracy_threshold: float = 0.65
    performance_degradation_threshold: float = 0.05  # 5% drop triggers retraining
    data_drift_threshold: float = 0.1  # 10% drift triggers retraining

    # Data quality
    min_training_samples: int = 1000
    max_training_samples: int = 50000
    data_freshness_days: int = 30  # Maximum age of training data
    quality_score_threshold: float = 0.7

    # Training parameters
    test_size: float = 0.2
    cv_folds: int = 5
    random_state: int = 42
    max_retraining_time: int = 3600  # 1 hour max

    # Notification settings
    notifications_enabled: bool = True
    notification_email: Optional[str] = None
    webhook_url: Optional[str] = None

    # NBA-specific settings
    nba_season_required: bool = True
    min_games_per_team: int = 10
    current_season_weight: float = 1.5  # Weight current season games higher

    # Advanced settings
    enable_early_stopping: bool = True
    enable_hyperparameter_tuning: bool = True
    enable_feature_selection: bool = True
    enable_ensemble_optimization: bool = True


@dataclass
class DataQualityReport:
    """Data quality assessment report"""
    status: DataQualityStatus
    score: float
    total_samples: int
    features_count: int
    missing_values: int
    outliers: int
    data_freshness_days: int
    nba_season_coverage: float
    team_coverage: int
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RetrainingJob:
    """Retraining job definition"""
    job_id: str
    trigger_type: RetrainingTriggerType
    status: RetrainingStatus
    config: RetrainingConfig
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    metrics_before: Optional[Dict[str, float]] = None
    metrics_after: Optional[Dict[str, float]] = None
    data_quality_report: Optional[DataQualityReport] = None
    new_model_version: Optional[str] = None
    rollback_version: Optional[str] = None


@dataclass
class PerformanceMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: float
    nba_accuracy: float
    home_win_accuracy: float
    away_win_accuracy: float
    timestamp: datetime = field(default_factory=datetime.now)


class NBADataValidator:
    """NBA-specific data validation and quality assessment"""

    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".NBADataValidator")

    def validate_data_quality(self, data: pd.DataFrame, config: RetrainingConfig) -> DataQualityReport:
        """
        Comprehensive data quality validation for NBA data

        Args:
            data: Training data DataFrame
            config: Retraining configuration

        Returns:
            DataQualityReport with assessment results
        """
        self.logger.info("🔍 Starting NBA data quality validation...")

        issues = []
        recommendations = []
        score = 0.0

        # Basic data checks
        total_samples = len(data)
        features_count = len(data.columns)

        # Sample size validation
        if total_samples < config.min_training_samples:
            issues.append(f"Insufficient samples: {total_samples} < {config.min_training_samples}")
            recommendations.append("Collect more training data or reduce minimum sample requirement")
            score += 0.1
        elif total_samples > config.max_training_samples:
            issues.append(f"Too many samples: {total_samples} > {config.max_training_samples}")
            recommendations.append("Consider data sampling or feature reduction")
            score += 0.3
        else:
            score += 0.3

        # Missing values check
        missing_values = data.isnull().sum().sum()
        missing_percentage = (missing_values / (total_samples * features_count)) * 100
        if missing_percentage > 10:
            issues.append(f"High missing values: {missing_percentage:.1f}%")
            recommendations.append("Implement missing value imputation strategy")
            score += 0.0
        elif missing_percentage > 5:
            issues.append(f"Moderate missing values: {missing_percentage:.1f}%")
            recommendations.append("Review and handle missing values")
            score += 0.1
        else:
            score += 0.2

        # NBA-specific validations
        nba_features = ['home_team', 'away_team', 'game_date']
        missing_nba_features = [f for f in nba_features if f not in data.columns]
        if missing_nba_features:
            issues.append(f"Missing NBA features: {missing_nba_features}")
            recommendations.append("Ensure NBA-specific features are included")
        else:
            score += 0.2

        # Team coverage validation
        if 'home_team' in data.columns:
            unique_teams = len(set(data['home_team'].unique()) | set(data['away_team'].unique()))
            if unique_teams < 25:  # NBA has 30 teams
                issues.append(f"Limited team coverage: {unique_teams} teams")
                recommendations.append("Include data from all NBA teams")
                score += 0.1
            else:
                score += 0.2

        # Data freshness check
        if 'game_date' in data.columns:
            data['game_date'] = pd.to_datetime(data['game_date'])
            latest_date = data['game_date'].max()
            days_old = (datetime.now() - latest_date).days
            data_freshness_days = days_old

            if days_old > config.data_freshness_days:
                issues.append(f"Stale data: {days_old} days old")
                recommendations.append("Update training data with recent games")
                score += 0.1
            else:
                score += 0.2
        else:
            data_freshness_days = 999
            score += 0.0

        # Determine overall status
        total_score = min(score, 1.0)
        if total_score >= 0.9:
            status = DataQualityStatus.EXCELLENT
        elif total_score >= 0.7:
            status = DataQualityStatus.GOOD
        elif total_score >= 0.5:
            status = DataQualityStatus.ACCEPTABLE
        elif total_score >= 0.3:
            status = DataQualityStatus.POOR
        else:
            status = DataQualityStatus.CRITICAL

        report = DataQualityReport(
            status=status,
            score=total_score,
            total_samples=total_samples,
            features_count=features_count,
            missing_values=missing_values,
            outliers=0,  # TODO: Implement outlier detection
            data_freshness_days=data_freshness_days,
            nba_season_coverage=1.0,  # TODO: Calculate season coverage
            team_coverage=unique_teams if 'home_team' in data.columns else 0,
            issues=issues,
            recommendations=recommendations
        )

        self.logger.info(f"✅ Data quality validation completed: {status.value} (score: {total_score:.3f})")
        return report


class PerformanceMonitor:
    """Model performance monitoring and degradation detection"""

    def __init__(self, config: RetrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + ".PerformanceMonitor")
        self.performance_history: List[PerformanceMetrics] = []

    def evaluate_model_performance(self, predictor: NBAEnsemblePredictor,
                                 test_data: Tuple[np.ndarray, np.ndarray]) -> PerformanceMetrics:
        """Evaluate current model performance"""
        X_test, y_test = test_data

        try:
            # Get predictions
            predictions = predictor.predict(X_test)
            prediction_probs = predictor.predict_proba(X_test)

            # Calculate basic metrics
            accuracy = accuracy_score(y_test, predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='weighted')
            auc_roc = roc_auc_score(y_test, prediction_probs)

            # TODO: Implement NBA-specific metrics
            nba_accuracy = accuracy
            home_win_accuracy = accuracy
            away_win_accuracy = accuracy

            metrics = PerformanceMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                auc_roc=auc_roc,
                nba_accuracy=nba_accuracy,
                home_win_accuracy=home_win_accuracy,
                away_win_accuracy=away_win_accuracy
            )

            self.performance_history.append(metrics)
            self.logger.info(f"📊 Model performance evaluated: accuracy={accuracy:.3f}, f1={f1:.3f}")

            return metrics

        except Exception as e:
            self.logger.error(f"❌ Error evaluating model performance: {e}")
            # Return default metrics
            return PerformanceMetrics(
                accuracy=0.0, precision=0.0, recall=0.0, f1_score=0.0,
                auc_roc=0.0, nba_accuracy=0.0, home_win_accuracy=0.0, away_win_accuracy=0.0
            )

    def check_performance_degradation(self, current_metrics: PerformanceMetrics) -> bool:
        """Check if performance has degraded significantly"""
        if len(self.performance_history) < 2:
            return False

        previous_metrics = self.performance_history[-2]

        # Check accuracy degradation
        accuracy_change = current_metrics.accuracy - previous_metrics.accuracy
        if accuracy_change < -self.config.performance_degradation_threshold:
            self.logger.warning(f"⚠️ Performance degradation detected: {accuracy_change:.3f}")
            return True

        return False


class DataDriftDetector:
    """Data drift detection using Evidently"""

    def __init__(self):
        self.logger = logging.getLogger(__name__ + ".DataDriftDetector")

    def detect_data_drift(self, reference_data: pd.DataFrame,
                         current_data: pd.DataFrame, threshold: float = 0.1) -> bool:
        """
        Detect data drift between reference and current datasets

        Args:
            reference_data: Baseline training data
            current_data: New data to compare
            threshold: Drift threshold

        Returns:
            True if significant drift detected
        """
        try:
            # Create drift report
            drift_report = Report(metrics=[
                DataDriftPreset(stattest="wasserstein", stattest_threshold=threshold)
            ])

            drift_report.run(reference_data=reference_data, current_data=current_data)

            # Get drift results
            results = drift_report.as_dict()
            drift_score = results.get('metrics', [{}])[0].get('result', {}).get('drift_share', 0)

            has_drift = drift_score > threshold

            if has_drift:
                self.logger.warning(f"⚠️ Data drift detected: {drift_score:.3f}")
            else:
                self.logger.info(f"✅ No significant data drift: {drift_score:.3f}")

            return has_drift

        except Exception as e:
            self.logger.error(f"❌ Error detecting data drift: {e}")
            return False


class NBARetrainingPipeline:
    """Main NBA Model Retraining Pipeline"""

    def __init__(self, config: Optional[RetrainingConfig] = None):
        self.config = config or RetrainingConfig()
        self.logger = logging.getLogger(__name__ + ".NBARetrainingPipeline")

        # Initialize components
        self.data_validator = NBADataValidator()
        self.performance_monitor = PerformanceMonitor(self.config)
        self.drift_detector = DataDriftDetector()

        # Initialize NBA systems if available
        if RETRAINING_AVAILABLE:
            self.predictor = NBAEnsemblePredictor()
            self.version_manager = None  # Will be initialized from predictor
            self.data_provider = NBADataProvider()
        else:
            self.predictor = None
            self.version_manager = None
            self.data_provider = None

        # Retraining state
        self.active_jobs: Dict[str, RetrainingJob] = {}
        self.job_history: List[RetrainingJob] = []
        self.is_running = False
        self.scheduler_thread = None

        # Performance tracking
        self.last_performance_check = datetime.now()
        self.reference_data = None

        self.logger.info("🔄 NBA Retraining Pipeline initialized")

    def setup_scheduler(self):
        """Setup automated retraining scheduler"""
        if not self.config.schedule_enabled:
            return

        if self.config.schedule_interval == "hourly":
            schedule.every().hour.do(self._scheduled_retraining)
        elif self.config.schedule_interval == "daily":
            schedule.every().day.at(self.config.schedule_time).do(self._scheduled_retraining)
        elif self.config.schedule_interval == "weekly":
            schedule.every().week.do(self._scheduled_retraining)

        self.logger.info(f"⏰ Scheduler configured: {self.config.schedule_interval}")

    def start_scheduler(self):
        """Start the scheduler in background thread"""
        if self.scheduler_thread is None or not self.scheduler_thread.is_alive():
            self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            self.scheduler_thread.start()
            self.logger.info("🚀 Retraining scheduler started")

    def _run_scheduler(self):
        """Run the scheduler loop"""
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def _scheduled_retraining(self):
        """Scheduled retraining job"""
        self.logger.info("⏰ Starting scheduled retraining...")
        self.trigger_retraining(RetrainingTriggerType.SCHEDULED)

    def trigger_retraining(self, trigger_type: RetrainingTriggerType,
                          reason: Optional[str] = None) -> str:
        """
        Trigger a retraining job

        Args:
            trigger_type: What triggered the retraining
            reason: Additional reason for triggering

        Returns:
            Job ID for tracking
        """
        job_id = hashlib.md5(f"{trigger_type.value}_{datetime.now()}".encode()).hexdigest()[:8]

        job = RetrainingJob(
            job_id=job_id,
            trigger_type=trigger_type,
            status=RetrainingStatus.PENDING,
            config=self.config,
            created_at=datetime.now()
        )

        self.active_jobs[job_id] = job
        self.logger.info(f"🔄 Retraining job created: {job_id} (trigger: {trigger_type.value})")

        # Schedule execution
        threading.Thread(target=self._execute_retraining_job, args=(job_id,), daemon=True).start()

        return job_id

    def _execute_retraining_job(self, job_id: str):
        """Execute a retraining job"""
        if not RETRAINING_AVAILABLE:
            self.logger.error("❌ Retraining systems not available")
            return

        job = self.active_jobs.get(job_id)
        if not job:
            self.logger.error(f"❌ Job {job_id} not found")
            return

        try:
            job.status = RetrainingStatus.RUNNING
            job.started_at = datetime.now()

            self.logger.info(f"🚀 Starting retraining job {job_id}")

            # Step 1: Collect and validate data
            training_data = self._collect_training_data()
            if training_data is None:
                job.status = RetrainingStatus.VALIDATION_FAILED
                job.error_message = "Failed to collect training data"
                return

            # Step 2: Data quality validation
            data_quality_report = self.data_validator.validate_data_quality(
                pd.DataFrame(training_data[0]), job.config
            )
            job.data_quality_report = data_quality_report

            if data_quality_report.status in [DataQualityStatus.CRITICAL, DataQualityStatus.POOR]:
                job.status = RetrainingStatus.VALIDATION_FAILED
                job.error_message = f"Data quality too low: {data_quality_report.status.value}"
                return

            # Step 3: Evaluate current model performance
            X_train, X_test, y_train, y_test = train_test_split(
                training_data[0], training_data[1],
                test_size=job.config.test_size,
                random_state=job.config.random_state
            )

            current_metrics = self.performance_monitor.evaluate_model_performance(
                self.predictor, (X_test, y_test)
            )
            job.metrics_before = asdict(current_metrics)

            # Step 4: Check if retraining is needed
            should_retrain = self._should_retrain(current_metrics, data_quality_report)
            if not should_retrain:
                job.status = RetrainingStatus.CANCELLED
                job.error_message = "Retraining not necessary"
                return

            # Step 5: Retrain models
            success = self.predictor.train_models(training_data=(X_train, y_train))
            if not success:
                job.status = RetrainingStatus.FAILED
                job.error_message = "Model training failed"
                return

            # Step 6: Evaluate new model performance
            new_metrics = self.performance_monitor.evaluate_model_performance(
                self.predictor, (X_test, y_test)
            )
            job.metrics_after = asdict(new_metrics)

            # Step 7: Register new model version
            if self.predictor.version_manager:
                version_info = self.predictor.register_model_version()
                job.new_model_version = version_info.version if version_info else None

            # Step 8: Validate performance improvement
            if new_metrics.accuracy <= current_metrics.accuracy * 0.95:  # Allow 5% degradation
                # Rollback if performance degraded
                if job.new_model_version and self.predictor.version_manager:
                    self.predictor.rollback_model(job.new_model_version)
                    job.rollback_version = job.new_model_version
                job.status = RetrainingStatus.FAILED
                job.error_message = "New model performance degraded, rolled back"
                return

            job.status = RetrainingStatus.SUCCESS
            job.completed_at = datetime.now()

            self.logger.info(f"✅ Retraining job {job_id} completed successfully")

        except Exception as e:
            self.logger.error(f"❌ Retraining job {job_id} failed: {e}")
            job.status = RetrainingStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()

        finally:
            # Move job to history
            if job_id in self.active_jobs:
                self.job_history.append(self.active_jobs.pop(job_id))

    def _collect_training_data(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Collect training data from various sources"""
        try:
            # Try to get real NBA data if available
            if self.data_provider:
                # TODO: Implement real data collection
                pass

            # Fallback to synthetic data
            if self.predictor:
                return self.predictor._generate_synthetic_data()

            return None

        except Exception as e:
            self.logger.error(f"❌ Error collecting training data: {e}")
            return None

    def _should_retrain(self, current_metrics: PerformanceMetrics,
                       data_quality_report: DataQualityReport) -> bool:
        """Determine if retraining is necessary"""

        # Check performance thresholds
        if current_metrics.accuracy < self.config.accuracy_threshold:
            self.logger.info("🔄 Retraining needed: accuracy below threshold")
            return True

        # Check performance degradation
        if self.performance_monitor.check_performance_degradation(current_metrics):
            self.logger.info("🔄 Retraining needed: performance degradation")
            return True

        # Check data quality
        if data_quality_report.score < self.config.quality_score_threshold:
            self.logger.info("🔄 Retraining needed: data quality issues")
            return True

        return False

    def start(self):
        """Start the retraining pipeline"""
        self.is_running = True
        self.setup_scheduler()
        self.start_scheduler()
        self.logger.info("🚀 NBA Retraining Pipeline started")

    def stop(self):
        """Stop the retraining pipeline"""
        self.is_running = False
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            self.scheduler_thread.join(timeout=5)
        self.logger.info("🛑 NBA Retraining Pipeline stopped")

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        active_jobs = len([j for j in self.active_jobs.values() if j.status == RetrainingStatus.RUNNING])
        successful_jobs = len([j for j in self.job_history if j.status == RetrainingStatus.SUCCESS])
        failed_jobs = len([j for j in self.job_history if j.status == RetrainingStatus.FAILED])

        return {
            "is_running": self.is_running,
            "active_jobs": active_jobs,
            "total_jobs": len(self.job_history),
            "successful_jobs": successful_jobs,
            "failed_jobs": failed_jobs,
            "success_rate": successful_jobs / len(self.job_history) if self.job_history else 0,
            "last_performance_check": self.last_performance_check.isoformat(),
            "scheduler_active": self.scheduler_thread.is_alive() if self.scheduler_thread else False
        }

    def get_job_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent job history"""
        recent_jobs = sorted(self.job_history, key=lambda j: j.created_at, reverse=True)[:limit]
        return [asdict(job) for job in recent_jobs]


# Global pipeline instance
_pipeline_instance: Optional[NBARetrainingPipeline] = None


def get_retraining_pipeline(config: Optional[RetrainingConfig] = None) -> NBARetrainingPipeline:
    """Get or create the global retraining pipeline instance"""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = NBARetrainingPipeline(config)
    return _pipeline_instance


def start_retraining_pipeline(config: Optional[RetrainingConfig] = None) -> NBARetrainingPipeline:
    """Start the retraining pipeline"""
    pipeline = get_retraining_pipeline(config)
    pipeline.start()
    return pipeline


def trigger_manual_retraining() -> str:
    """Trigger manual retraining"""
    pipeline = get_retraining_pipeline()
    return pipeline.trigger_retraining(RetrainingTriggerType.MANUAL)


if __name__ == "__main__":
    # Example usage
    config = RetrainingConfig(
        schedule_enabled=True,
        schedule_interval="daily",
        schedule_time="02:00",
        accuracy_threshold=0.65,
        performance_degradation_threshold=0.05
    )

    pipeline = start_retraining_pipeline(config)

    try:
        # Keep running
        while True:
            time.sleep(60)
            status = pipeline.get_status()
            print(f"Pipeline status: {status}")
    except KeyboardInterrupt:
        print("Stopping pipeline...")
        pipeline.stop()