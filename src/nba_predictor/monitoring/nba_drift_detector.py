#!/usr/bin/env python3
"""
🔍 NBA Drift Detection System - Task 2.1.2 Implementation

Sistema di drift detection per feature distributions nel NBA prediction system.
Implementa drift monitoring con Evidently AI per MLIntegrationBridge.

Author: NBA Predictive Analytics System
Date: 2025-01-10
Architecture: DevStream SuperPowered with Context Set Compliance
"""

import logging
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
import pandas as pd
import numpy as np

# Evidently imports per drift detection
from evidently import ColumnMapping
from evidently.report import Report
from evidently.test_suite import TestSuite
from evidently.metrics import DatasetDriftMetric, FeatureDriftMetric
from evidently.test_preset import DataDriftPreset, NoTargetPerformanceTestPreset
from evidently.tests import TestFeatureValueDrift, TestColumnShareOfMissingValues, TestNumberOfRows
from evidently.descriptors import TextDescriptor, NumericDescriptor
from evidently.pipeline.column_mapping import ColumnMapping

from src.nba_predictor.streamlit.components.ml_integration_bridge import MLIntegrationBridge

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class DriftDetectionConfig:
    """Configuration for drift detection system"""
    drift_threshold: float = 0.5  # Evidently default
    confidence_level: float = 0.95
    min_sample_size: int = 50
    reference_data_size: int = 1000
    max_history_size: int = 10000
    monitoring_window_days: int = 30
    drift_check_interval_minutes: int = 60

    # Evidently-specific settings
    stattest: str = "ks"  # Kolmogorov-Smirnov test
    stattest_threshold: float = 0.05  # p-value threshold

    # NBA-specific features
    momentum_features: List[str] = field(default_factory=lambda: [
        "home_team_momentum", "away_team_momentum", "momentum_difference"
    ])
    schedule_features: List[str] = field(default_factory=lambda: [
        "home_team_rest_days", "away_team_rest_days", "rest_advantage",
        "home_team_back_to_back", "away_team_back_to_back"
    ])
    performance_features: List[str] = field(default_factory=lambda: [
        "home_team_win_rate", "away_team_win_rate", "home_team_points_per_game",
        "away_team_points_per_game", "home_team_points_allowed_per_game",
        "away_team_points_allowed_per_game"
    ])

@dataclass
class DriftAlert:
    """Represents a drift detection alert"""
    feature_name: str
    drift_score: float
    drift_detected: bool
    test_type: str
    confidence: float
    reference_stats: Dict[str, float]
    current_stats: Dict[str, float]
    timestamp: datetime
    severity: str = "medium"  # low, medium, high, critical

    @property
    def alert_id(self) -> str:
        """Generate unique alert ID"""
        return f"drift_{self.feature_name}_{self.timestamp.strftime('%Y%m%d_%H%M%S')}"

@dataclass
class ModelDriftStatus:
    """Tracks drift status for a specific model"""
    model_name: str
    last_drift_check: Optional[datetime] = None
    features_drifted: List[str] = field(default_factory=list)
    overall_drift_detected: bool = False
    drift_score: float = 0.0
    confidence_level: float = 0.0
    alerts_history: deque = field(default_factory=lambda: deque(maxlen=100))
    total_drift_alerts: int = 0

    def add_alert(self, alert: DriftAlert):
        """Add a new drift alert"""
        self.alerts_history.append(alert)
        self.total_drift_alerts += 1

        # Update overall status
        if alert.drift_detected:
            if alert.feature_name not in self.features_drifted:
                self.features_drifted.append(alert.feature_name)

            # Update severity based on alert
            self.drift_score = max(self.drift_score, alert.drift_score)

    def get_summary(self) -> Dict[str, Any]:
        """Get drift status summary"""
        recent_alerts = [alert for alert in self.alerts_history
                        if alert.timestamp > datetime.now() - timedelta(hours=24)]

        return {
            "model_name": self.model_name,
            "last_drift_check": self.last_drift_check,
            "features_drifted": self.features_drifted,
            "overall_drift_detected": self.overall_drift_detected,
            "drift_score": self.drift_score,
            "confidence_level": self.confidence_level,
            "total_alerts": self.total_drift_alerts,
            "recent_alerts_24h": len(recent_alerts),
            "severity": self._get_severity_level()
        }

    def _get_severity_level(self) -> str:
        """Determine overall severity level"""
        recent_alerts = [alert for alert in self.alerts_history
                        if alert.timestamp > datetime.now() - timedelta(hours=24)]

        if not recent_alerts:
            return "low"

        critical_count = sum(1 for alert in recent_alerts if alert.severity == "critical")
        high_count = sum(1 for alert in recent_alerts if alert.severity == "high")

        if critical_count > 0:
            return "critical"
        elif high_count > 0:
            return "high"
        elif len(recent_alerts) > 3:
            return "medium"
        else:
            return "low"

class NBADriftDetector:
    """NBA-specific drift detection system using Evidently AI"""

    def __init__(self,
                 bridge: Optional[MLIntegrationBridge] = None,
                 config: Optional[DriftDetectionConfig] = None,
                 enable_background_monitoring: bool = True):
        """Initialize NBA drift detection system"""

        self.bridge = bridge or MLIntegrationBridge()
        self.config = config or DriftDetectionConfig()
        self.enable_background_monitoring = enable_background_monitoring

        # State management
        self.reference_data: Dict[str, pd.DataFrame] = {}
        self.feature_stats_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.model_drift_status: Dict[str, ModelDriftStatus] = {}

        # Drift detection configuration
        self.column_mapping = self._create_nba_column_mapping()

        # Background monitoring
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        logger.info("NBA Drift Detector initialized")

        # Initialize background monitoring if enabled
        if self.enable_background_monitoring:
            self.start_background_monitoring()

    def _create_nba_column_mapping(self) -> ColumnMapping:
        """Create Evidently column mapping for NBA features"""

        # Define categorical and numerical features
        categorical_features = [
            "home_team", "away_team", "season", "game_type"
        ]

        numerical_features = (
            self.config.momentum_features +
            self.config.schedule_features +
            self.config.performance_features
        )

        # Target column (if available)
        target_column = "home_team_win"  # Binary target

        # Prediction column for data profiling
        prediction_column = "home_team_win_probability"

        return ColumnMapping(
            target=target_column,
            prediction=prediction_column,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
            task=None
        )

    def initialize_reference_data(self,
                                 model_name: str,
                                 reference_features: pd.DataFrame,
                                 force_update: bool = False) -> bool:
        """Initialize reference data for drift detection"""

        try:
            # Validate reference data
            if len(reference_features) < self.config.min_sample_size:
                logger.warning(f"Reference data too small for model {model_name}: {len(reference_features)}")
                return False

            # Store reference data
            if model_name not in self.reference_data or force_update:
                self.reference_data[model_name] = reference_features.copy()

                # Initialize drift status
                if model_name not in self.model_drift_status:
                    self.model_drift_status[model_name] = ModelDriftStatus(model_name=model_name)

                logger.info(f"Reference data initialized for model {model_name}: {len(reference_features)} samples")
                return True

            logger.info(f"Reference data already exists for model {model_name}")
            return True

        except Exception as e:
            logger.error(f"Error initializing reference data for {model_name}: {e}")
            return False

    def detect_drift_for_prediction(self,
                                   model_name: str,
                                   input_features: Dict[str, Any],
                                   prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Detect drift for a single prediction"""

        try:
            # Check if reference data exists
            if model_name not in self.reference_data:
                logger.warning(f"No reference data available for model {model_name}")
                return None

            # Convert input features to DataFrame
            current_df = pd.DataFrame([input_features])

            # Perform drift detection
            drift_results = self._perform_evidently_drift_detection(
                model_name,
                self.reference_data[model_name],
                current_df
            )

            # Store current features for future reference
            self._update_feature_history(model_name, input_features)

            # Update drift status
            self._update_drift_status(model_name, drift_results)

            return drift_results

        except Exception as e:
            logger.error(f"Error in drift detection for {model_name}: {e}")
            return None

    def _perform_evidently_drift_detection(self,
                                          model_name: str,
                                          reference_data: pd.DataFrame,
                                          current_data: pd.DataFrame) -> Dict[str, Any]:
        """Perform Evidently-based drift detection"""

        results = {
            "model_name": model_name,
            "timestamp": datetime.now(),
            "overall_drift_detected": False,
            "drift_score": 0.0,
            "features_drifted": [],
            "feature_drift_details": {},
            "data_quality_issues": [],
            "alerts": []
        }

        try:
            # Create DataDriftPreset for comprehensive analysis
            drift_preset = DataDriftPreset(
                stattest=self.config.stattest,
                stattest_threshold=self.config.stattest_threshold,
                features=self.config.momentum_features + self.config.schedule_features + self.config.performance_features
            )

            # Generate drift report
            drift_report = Report(metrics=[DatasetDriftMetric()])
            drift_report.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )

            # Generate test suite for detailed feature analysis
            drift_test_suite = TestSuite(tests=[drift_preset])
            drift_test_suite.run(
                reference_data=reference_data,
                current_data=current_data,
                column_mapping=self.column_mapping
            )

            # Extract drift results
            results.update(self._extract_drift_results(drift_report, drift_test_suite))

        except Exception as e:
            logger.error(f"Error in Evidently drift detection: {e}")
            results["error"] = str(e)

        return results

    def _extract_drift_results(self,
                              report: Report,
                              test_suite: TestSuite) -> Dict[str, Any]:
        """Extract drift results from Evidently report"""

        results = {
            "overall_drift_detected": False,
            "drift_score": 0.0,
            "features_drifted": [],
            "feature_drift_details": {},
            "data_quality_issues": []
        }

        try:
            # Extract dataset drift metric
            for result in report.as_dict()["metrics"]:
                if result["metric_name"] == "DatasetDriftMetric":
                    results["overall_drift_detected"] = result["result"]["drift_share"] > self.config.drift_threshold
                    results["drift_score"] = result["result"]["drift_share"]
                    break

            # Extract feature-level drift
            for test_result in test_suite.as_dict()["tests"]:
                if test_result["name"] == "DataDriftPreset":
                    for feature_test in test_result["parameters"]["tests"]:
                        feature_name = feature_test.get("feature_name", "unknown")

                        if "TestFeatureValueDrift" in str(feature_test):
                            test_info = test_result.get("test_results", {}).get(feature_name, {})

                            if isinstance(test_info, dict) and "drift_score" in test_info:
                                drift_detected = test_info.get("drift_detected", False)
                                drift_score = test_info.get("drift_score", 0.0)

                                if drift_detected:
                                    results["features_drifted"].append(feature_name)

                                results["feature_drift_details"][feature_name] = {
                                    "drift_detected": drift_detected,
                                    "drift_score": drift_score,
                                    "test_statistic": test_info.get("test_statistic", 0.0),
                                    "p_value": test_info.get("p_value", 1.0),
                                    "confidence": 1 - test_info.get("p_value", 1.0)
                                }

            # Create alerts for drifted features
            for feature_name in results["features_drifted"]:
                if feature_name in results["feature_drift_details"]:
                    alert = DriftAlert(
                        feature_name=feature_name,
                        drift_score=results["feature_drift_details"][feature_name]["drift_score"],
                        drift_detected=True,
                        test_type="FeatureValueDrift",
                        confidence=results["feature_drift_details"][feature_name]["confidence"],
                        reference_stats={},
                        current_stats={},
                        timestamp=datetime.now(),
                        severity=self._calculate_drift_severity(results["feature_drift_details"][feature_name]["drift_score"])
                    )
                    results["alerts"].append(alert.__dict__)

        except Exception as e:
            logger.error(f"Error extracting drift results: {e}")
            results["extraction_error"] = str(e)

        return results

    def _calculate_drift_severity(self, drift_score: float) -> str:
        """Calculate drift severity based on drift score"""
        if drift_score > 0.8:
            return "critical"
        elif drift_score > 0.6:
            return "high"
        elif drift_score > 0.4:
            return "medium"
        else:
            return "low"

    def _update_feature_history(self, model_name: str, input_features: Dict[str, Any]):
        """Update feature statistics history"""
        timestamp = datetime.now()

        for feature_name, value in input_features.items():
            if isinstance(value, (int, float)):
                self.feature_stats_history[f"{model_name}_{feature_name}"].append({
                    "timestamp": timestamp,
                    "value": value
                })

    def _update_drift_status(self, model_name: str, drift_results: Dict[str, Any]):
        """Update model drift status"""
        if model_name not in self.model_drift_status:
            self.model_drift_status[model_name] = ModelDriftStatus(model_name=model_name)

        status = self.model_drift_status[model_name]
        status.last_drift_check = datetime.now()
        status.overall_drift_detected = drift_results.get("overall_drift_detected", False)
        status.drift_score = drift_results.get("drift_score", 0.0)

        # Add alerts
        for alert_data in drift_results.get("alerts", []):
            try:
                # Recreate alert object from dict
                alert = DriftAlert(
                    feature_name=alert_data["feature_name"],
                    drift_score=alert_data["drift_score"],
                    drift_detected=alert_data["drift_detected"],
                    test_type=alert_data["test_type"],
                    confidence=alert_data["confidence"],
                    reference_stats=alert_data["reference_stats"],
                    current_stats=alert_data["current_stats"],
                    timestamp=alert_data["timestamp"],
                    severity=alert_data["severity"]
                )
                status.add_alert(alert)
            except Exception as e:
                logger.error(f"Error creating alert from dict: {e}")

    def generate_drift_report(self,
                             model_name: str,
                             days: int = 7) -> Optional[Dict[str, Any]]:
        """Generate comprehensive drift report for a model"""

        try:
            if model_name not in self.model_drift_status:
                logger.warning(f"No drift status available for model {model_name}")
                return None

            status = self.model_drift_status[model_name]
            cutoff_date = datetime.now() - timedelta(days=days)

            # Filter alerts by date
            recent_alerts = [alert for alert in status.alerts_history
                           if alert.timestamp > cutoff_date]

            # Generate feature-specific reports
            feature_reports = {}
            for feature_name in (self.config.momentum_features + self.config.schedule_features + self.config.performance_features):
                feature_key = f"{model_name}_{feature_name}"
                if feature_key in self.feature_stats_history:
                    feature_history = list(self.feature_stats_history[feature_key])

                    if feature_history:
                        values = [point["value"] for point in feature_history]
                        feature_reports[feature_name] = {
                            "total_observations": len(values),
                            "mean": np.mean(values),
                            "std": np.std(values),
                            "min": np.min(values),
                            "max": np.max(values),
                            "recent_trend": self._calculate_trend(values[-10:]),  # Last 10 values
                            "drift_alerts": len([a for a in recent_alerts if a.feature_name == feature_name])
                        }

            return {
                "model_name": model_name,
                "report_period_days": days,
                "generation_time": datetime.now(),
                "drift_status_summary": status.get_summary(),
                "feature_analysis": feature_reports,
                "recent_alerts": [alert.__dict__ for alert in recent_alerts],
                "recommendations": self._generate_drift_recommendations(status, recent_alerts)
            }

        except Exception as e:
            logger.error(f"Error generating drift report for {model_name}: {e}")
            return None

    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate simple trend for recent values"""
        if len(values) < 2:
            return "insufficient_data"

        # Simple linear regression to determine trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]

        if slope > 0.01:
            return "increasing"
        elif slope < -0.01:
            return "decreasing"
        else:
            return "stable"

    def _generate_drift_recommendations(self,
                                       status: ModelDriftStatus,
                                       recent_alerts: List[DriftAlert]) -> List[str]:
        """Generate recommendations based on drift status"""
        recommendations = []

        if status.overall_drift_detected:
            recommendations.append("Consider model retraining with recent data")
            recommendations.append("Review feature engineering pipeline")

        # Check for specific patterns
        critical_features = [alert.feature_name for alert in recent_alerts if alert.severity == "critical"]
        if critical_features:
            recommendations.append(f"Critical drift detected in features: {', '.join(critical_features)}")
            recommendations.append("Investigate data sources for critical features")

        high_frequency_drift = len([alert for alert in recent_alerts if alert.timestamp > datetime.now() - timedelta(hours=1)])
        if high_frequency_drift > 5:
            recommendations.append("High drift frequency detected - review data pipeline")

        # Check momentum features
        momentum_drift = [alert for alert in recent_alerts if "momentum" in alert.feature_name]
        if len(momentum_drift) > 2:
            recommendations.append("Momentum calculations may need recalibration")

        # Check schedule features
        schedule_drift = [alert for alert in recent_alerts if "rest" in alert.feature_name or "back_to_back" in alert.feature_name]
        if len(schedule_drift) > 2:
            recommendations.append("Schedule analysis patterns may have changed")

        if not recommendations:
            recommendations.append("No significant drift detected - continue monitoring")

        return recommendations

    def start_background_monitoring(self):
        """Start background drift monitoring"""
        if not self.enable_background_monitoring:
            return

        if self._monitoring_active:
            logger.warning("Background monitoring already active")
            return

        self._monitoring_active = True
        self._stop_event.clear()
        self._monitoring_thread = threading.Thread(
            target=self._background_monitoring_loop,
            daemon=True,
            name="NBA-Drift-Monitor"
        )
        self._monitoring_thread.start()
        logger.info("Background drift monitoring started")

    def stop_background_monitoring(self):
        """Stop background drift monitoring"""
        if not self._monitoring_active:
            return

        self._monitoring_active = False
        self._stop_event.set()

        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)

        logger.info("Background drift monitoring stopped")

    def _background_monitoring_loop(self):
        """Background monitoring loop"""
        logger.info("Background drift monitoring loop started")

        while self._monitoring_active and not self._stop_event.is_set():
            try:
                # Check drift for all models with reference data
                for model_name in list(self.reference_data.keys()):
                    # Get recent feature statistics
                    recent_features = self._get_recent_features_for_model(model_name)

                    if recent_features is not None and len(recent_features) >= self.config.min_sample_size:
                        # Perform drift check
                        drift_results = self._perform_evidently_drift_detection(
                            model_name,
                            self.reference_data[model_name],
                            recent_features
                        )

                        # Update drift status
                        self._update_drift_status(model_name, drift_results)

                        # Log significant drift
                        if drift_results.get("overall_drift_detected", False):
                            logger.warning(
                                f"Drift detected for model {model_name}: "
                                f"score={drift_results.get('drift_score', 0):.3f}, "
                                f"features={drift_results.get('features_drifted', [])}"
                            )

                # Wait for next check
                self._stop_event.wait(timeout=self.config.drift_check_interval_minutes * 60)

            except Exception as e:
                logger.error(f"Error in background monitoring: {e}")
                self._stop_event.wait(timeout=60)  # Wait 1 minute on error

        logger.info("Background drift monitoring loop stopped")

    def _get_recent_features_for_model(self, model_name: str) -> Optional[pd.DataFrame]:
        """Get recent feature data for model"""
        try:
            # Collect recent features from history
            recent_data = []

            for feature_name in (self.config.momentum_features + self.config.schedule_features + self.config.performance_features):
                feature_key = f"{model_name}_{feature_name}"
                if feature_key in self.feature_stats_history:
                    # Get last 100 observations for this feature
                    history = list(self.feature_stats_history[feature_key])[-100:]

                    for i, point in enumerate(history):
                        if i >= len(recent_data):
                            recent_data.append({"timestamp": point["timestamp"]})
                        recent_data[i][feature_name] = point["value"]

            if recent_data:
                return pd.DataFrame(recent_data)

            return None

        except Exception as e:
            logger.error(f"Error getting recent features for {model_name}: {e}")
            return None

    def get_system_drift_status(self) -> Dict[str, Any]:
        """Get overall drift system status"""

        total_models = len(self.model_drift_status)
        models_with_drift = sum(1 for status in self.model_drift_status.values() if status.overall_drift_detected)
        total_alerts_24h = sum(
            len([alert for alert in status.alerts_history
                if alert.timestamp > datetime.now() - timedelta(hours=24)])
            for status in self.model_drift_status.values()
        )

        return {
            "monitoring_active": self._monitoring_active,
            "total_models_monitored": total_models,
            "models_with_drift": models_with_drift,
            "models_without_drift": total_models - models_with_drift,
            "total_alerts_24h": total_alerts_24h,
            "reference_data_available": list(self.reference_data.keys()),
            "config": {
                "drift_threshold": self.config.drift_threshold,
                "confidence_level": self.config.confidence_level,
                "check_interval_minutes": self.config.drift_check_interval_minutes
            },
            "model_status": {
                name: status.get_summary()
                for name, status in self.model_drift_status.items()
            }
        }

    def cleanup(self):
        """Cleanup drift detector resources"""
        self.stop_background_monitoring()
        self.reference_data.clear()
        self.feature_stats_history.clear()
        self.model_drift_status.clear()
        logger.info("NBA Drift Detector cleanup completed")

# Global drift detector instance
_global_drift_detector: Optional[NBADriftDetector] = None

def get_drift_detector() -> NBADriftDetector:
    """Get global drift detector instance"""
    global _global_drift_detector
    if _global_drift_detector is None:
        _global_drift_detector = NBADriftDetector()
    return _global_drift_detector

def initialize_nba_drift_detection(model_name: str,
                                  reference_features: pd.DataFrame,
                                  config: Optional[DriftDetectionConfig] = None) -> bool:
    """Initialize drift detection for a specific NBA model"""

    try:
        detector = get_drift_detector()

        # Override config if provided
        if config:
            detector.config = config

        return detector.initialize_reference_data(model_name, reference_features)

    except Exception as e:
        logger.error(f"Error initializing NBA drift detection for {model_name}: {e}")
        return False

def detect_model_drift(model_name: str,
                      input_features: Dict[str, Any],
                      prediction: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Detect drift for a single model prediction"""

    try:
        detector = get_drift_detector()
        return detector.detect_drift_for_prediction(model_name, input_features, prediction)

    except Exception as e:
        logger.error(f"Error detecting model drift for {model_name}: {e}")
        return None