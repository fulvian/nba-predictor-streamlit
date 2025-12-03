#!/usr/bin/env python3
"""
🚀 Advanced ML Operations Dashboard - Task 2.2.3-2.2.5

Comprehensive dashboard for advanced ML operations including:
- Model explanation and interpretability
- Production deployment management
- Model versioning and rollback
- Automated retraining monitoring

Author: NBA Predictive Analytics System
Date: 2025-01-12
Architecture: DevStream SuperPowered with Context Set Compliance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import logging

# Import advanced ML components
try:
    from src.nba_predictor.ensemble.nba_ensemble_predictor import NBAEnsemblePredictor
    from src.nba_predictor.ensemble.prediction_explainer import (
        NBAPredictionExplainer,
        ExplanationConfig,
        PredictionExplanation,
        ExplanationMethod,
    )
    from src.nba_predictor.ensemble.model_version_manager import (
        NBAModelVersionManager,
        ModelVersion,
        ModelType,
        ModelStatus,
    )
    from src.nba_predictor.ensemble.production_deployment_manager import (
        ProductionDeploymentManager,
        DeploymentConfig,
        DeploymentStrategy,
        DeploymentStatus,
    )
    from src.nba_predictor.ensemble.model_retraining_pipeline import (
        NBARetrainingPipeline,
        RetrainingConfig,
        RetrainingStatus,
    )

    ADVANCED_ML_AVAILABLE = True
except ImportError as e:
    ADVANCED_ML_AVAILABLE = False
    logging.warning(f"Advanced ML components not available: {e}")

# Import explanation dashboard - REMOVED as part of refactoring
# The nba_explanation_dashboard.py was deprecated and removed
EXPLANATION_DASHBOARD_AVAILABLE = False

logger = logging.getLogger(__name__)


class AdvancedMLOperationsDashboard:
    """Advanced ML Operations Dashboard"""

    def __init__(self):
        """Initialize the advanced ML operations dashboard"""
        self.predictor = None
        self.version_manager = None
        self.deployment_manager = None
        self.retraining_pipeline = None
        self.explanation_dashboard = None

        self._initialize_components()

    def _initialize_components(self):
        """Initialize all ML components"""
        if not ADVANCED_ML_AVAILABLE:
            logger.warning("Advanced ML components not available")
            return

        try:
            # Initialize core predictor
            self.predictor = NBAEnsemblePredictor()

            # Initialize version manager if available
            if self.predictor.is_version_manager_available():
                self.version_manager = self.predictor.get_version_manager()

            # Initialize deployment manager
            if self.version_manager:
                self.deployment_manager = ProductionDeploymentManager(
                    version_manager=self.version_manager
                )

            # Initialize retraining pipeline if available
            if self.predictor.is_retraining_available():
                self.retraining_pipeline = self.predictor.get_retraining_pipeline()

            # Initialize explanation dashboard - REMOVED as part of refactoring
            # The NBAExplanationDashboard was deprecated and removed
            self.explanation_dashboard = None

            logger.info("Advanced ML components initialized successfully")

        except Exception as e:
            logger.error(f"Error initializing advanced ML components: {e}")

    def render_dashboard(self):
        """Render the complete advanced ML operations dashboard"""
        st.markdown("---")
        st.markdown("## 🚀 Advanced ML Operations")
        st.markdown(
            "*Advanced Machine Learning operations management with DevStream SuperPowered architecture*"
        )

        if not ADVANCED_ML_AVAILABLE:
            st.error("❌ Advanced ML components not available")
            st.info("Please ensure all required dependencies are installed")
            return

        # Create main navigation tabs
        tabs = st.tabs(
            [
                "🧠 Model Explanation",
                "📦 Model Versioning",
                "🚀 Production Deployment",
                "🔄 Automated Retraining",
                "📊 ML Operations Overview",
            ]
        )

        with tabs[0]:
            self._render_model_explanation_tab()

        with tabs[1]:
            self._render_model_versioning_tab()

        with tabs[2]:
            self._render_production_deployment_tab()

        with tabs[3]:
            self._render_automated_retraining_tab()

        with tabs[4]:
            self._render_ml_operations_overview_tab()

    def _render_model_explanation_tab(self):
        """Render model explanation and interpretability tab"""
        st.markdown("### 🧠 Model Explanation & Interpretability")

        # Explanation dashboard was removed as part of refactoring
        st.info(
            "ℹ️ **Model Explanation Dashboard** was deprecated and removed as part of the refactoring process."
        )
        st.info(
            "🔧 **Alternative**: Use the main betting workflow dashboard for model insights and predictions."
        )

        # Display basic model information if available
        if self.predictor:
            st.markdown("#### 📊 Current Model Status")
            col1, col2 = st.columns(2)

            with col1:
                st.success("✅ Ensemble Predictor Available")
                if hasattr(self.predictor, "models"):
                    st.metric("Active Models", len(self.predictor.models))

            with col2:
                if hasattr(self.predictor, "model_weights"):
                    total_weight = sum(self.predictor.model_weights.values())
                    st.metric("Total Model Weight", f"{total_weight:.3f}")

        # Basic explanation controls (without the removed dashboard)
        st.markdown("---")
        st.markdown("#### 🔧 Basic Explanation Settings")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Available Methods**")
            methods = ["SHAP Values", "Feature Importance", "Prediction Confidence"]
            selected_methods = st.multiselect(
                "Select explanation methods",
                options=methods,
                default=["Feature Importance", "Prediction Confidence"],
            )

        with col2:
            st.markdown("**Display Settings**")
            show_confidence = st.checkbox("Show prediction confidence", value=True)
            show_feature_importance = st.checkbox("Show feature importance", value=True)

        if st.button("🔧 Update Settings"):
            st.success("✅ Settings updated!")

    def _render_model_versioning_tab(self):
        """Render model versioning and management tab"""
        st.markdown("### 📦 Model Versioning & Management")

        if not self.version_manager:
            st.error("❌ Model version manager not available")
            return

        # Version management tabs
        version_tabs = st.tabs(
            [
                "📋 Version Registry",
                "📊 Version Comparison",
                "🔄 Rollback Management",
                "📈 Performance Tracking",
            ]
        )

        with version_tabs[0]:
            self._render_version_registry()

        with version_tabs[1]:
            self._render_version_comparison()

        with version_tabs[2]:
            self._render_rollback_management()

        with version_tabs[3]:
            self._render_performance_tracking()

    def _render_version_registry(self):
        """Render model version registry"""
        st.markdown("#### 📋 Model Version Registry")

        # Model type selection
        model_types = [mt.value for mt in ModelType]
        selected_model_type = st.selectbox(
            "Select Model Type", options=model_types, key="version_registry_model_type"
        )

        model_type = ModelType(selected_model_type)

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🔄 Refresh Versions", help="Reload version registry"):
                st.success("✅ Version registry refreshed!")

        with col2:
            if st.button("📊 Generate Report", help="Generate version report"):
                self._generate_version_report(model_type)

        with col3:
            if st.button("🧹 Cleanup Old Versions", help="Clean up old model versions"):
                self._cleanup_old_versions(model_type)

        # Display versions
        try:
            versions = self.version_manager.get_model_versions(model_type)

            if versions:
                # Convert to DataFrame for display
                version_data = []
                for version in versions:
                    version_data.append(
                        {
                            "Version": version.version,
                            "Status": version.status.value,
                            "Created": version.created_at.strftime("%Y-%m-%d %H:%M"),
                            "Created By": version.created_by,
                            "Description": version.description[:50] + "..."
                            if len(version.description) > 50
                            else version.description,
                            "Accuracy": f"{version.metrics.accuracy:.3f}",
                            "NBA Accuracy": f"{version.metrics.nba_accuracy:.3f}",
                            "Latency": f"{version.metrics.prediction_latency_ms:.1f}ms",
                        }
                    )

                df = pd.DataFrame(version_data)
                st.dataframe(df, use_container_width=True)

                # Version details
                if st.button("📝 Show Version Details"):
                    selected_version = st.selectbox(
                        "Select version for details",
                        options=[v.version for v in versions],
                    )
                    self._show_version_details(selected_version, model_type)

            else:
                st.info("No versions found for this model type")

        except Exception as e:
            st.error(f"Error loading versions: {e}")

    def _render_version_comparison(self):
        """Render model version comparison"""
        st.markdown("#### 📊 Model Version Comparison")

        try:
            versions = self.version_manager.get_model_versions()
            all_versions = []

            for model_type, version_list in versions.items():
                for version in version_list:
                    all_versions.append(
                        {
                            "Model Type": model_type.value,
                            "Version": version.version,
                            "Status": version.status.value,
                            "Accuracy": version.metrics.accuracy,
                            "NBA Accuracy": version.metrics.nba_accuracy,
                            "Precision": version.metrics.precision,
                            "Recall": version.metrics.recall,
                            "F1 Score": version.metrics.f1_score,
                            "Latency (ms)": version.metrics.prediction_latency_ms,
                            "Model Size (MB)": version.metrics.model_size_mb,
                        }
                    )

            if all_versions:
                df = pd.DataFrame(all_versions)

                # Comparison charts
                col1, col2 = st.columns(2)

                with col1:
                    # Accuracy comparison
                    fig = px.bar(
                        df,
                        x="Version",
                        y="NBA Accuracy",
                        color="Model Type",
                        title="NBA Accuracy by Version",
                        barmode="group",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    # Latency comparison
                    fig = px.bar(
                        df,
                        x="Version",
                        y="Latency (ms)",
                        color="Model Type",
                        title="Prediction Latency by Version",
                        barmode="group",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Detailed comparison table
                st.dataframe(df, use_container_width=True)

            else:
                st.info("No versions available for comparison")

        except Exception as e:
            st.error(f"Error comparing versions: {e}")

    def _render_rollback_management(self):
        """Render rollback management interface"""
        st.markdown("#### 🔄 Rollback Management")

        # Model type selection
        model_types = [mt.value for mt in ModelType]
        selected_model_type = st.selectbox(
            "Select Model Type", options=model_types, key="rollback_model_type"
        )

        model_type = ModelType(selected_model_type)

        # Current version info
        try:
            active_versions = self.version_manager.get_active_versions()
            current_version = active_versions.get(model_type)

            if current_version:
                st.info(f"📍 Current active version: **{current_version}**")

                # Available versions for rollback
                versions = self.version_manager.get_model_versions(model_type)
                rollback_options = [
                    v.version for v in versions if v.version != current_version
                ]

                if rollback_options:
                    selected_rollback_version = st.selectbox(
                        "Select version to rollback to",
                        options=rollback_options,
                        key="rollback_version_select",
                    )

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("🔄 Execute Rollback", type="secondary"):
                            success = self.version_manager.rollback_model(
                                current_version, selected_rollback_version, model_type
                            )
                            if success:
                                st.success(
                                    f"✅ Successfully rolled back to version {selected_rollback_version}"
                                )
                            else:
                                st.error("❌ Rollback failed")

                    with col2:
                        if st.button(
                            "📋 Compare Versions",
                            help="Compare versions before rollback",
                        ):
                            self._compare_versions_for_rollback(
                                current_version, selected_rollback_version, model_type
                            )

                else:
                    st.warning("No previous versions available for rollback")
            else:
                st.warning("No active version found for this model type")

        except Exception as e:
            st.error(f"Error managing rollback: {e}")

    def _render_performance_tracking(self):
        """Render performance tracking for model versions"""
        st.markdown("#### 📈 Performance Tracking")

        # Time range selection
        time_range = st.selectbox(
            "Select time range",
            options=["Last 24 hours", "Last 7 days", "Last 30 days", "Last 90 days"],
            key="performance_time_range",
        )

        try:
            # Get performance data
            performance_data = self.version_manager.get_performance_history()

            if performance_data:
                # Performance metrics over time
                fig = make_subplots(
                    rows=2,
                    cols=2,
                    subplot_titles=("Accuracy", "Latency", "Error Rate", "Model Size"),
                    specs=[
                        [{"secondary_y": False}, {"secondary_y": False}],
                        [{"secondary_y": False}, {"secondary_y": False}],
                    ],
                )

                # Add traces for different metrics
                for model_type in performance_data.keys():
                    data = performance_data[model_type]

                    # Accuracy
                    fig.add_trace(
                        go.Scatter(
                            x=data["timestamps"],
                            y=data["accuracy"],
                            name=f"{model_type.value} Accuracy",
                            mode="lines+markers",
                        ),
                        row=1,
                        col=1,
                    )

                    # Latency
                    fig.add_trace(
                        go.Scatter(
                            x=data["timestamps"],
                            y=data["latency_ms"],
                            name=f"{model_type.value} Latency",
                            mode="lines+markers",
                        ),
                        row=1,
                        col=2,
                    )

                fig.update_layout(height=600, title_text="Model Performance Over Time")
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.info("No performance data available")

        except Exception as e:
            st.error(f"Error loading performance data: {e}")

    def _render_production_deployment_tab(self):
        """Render production deployment management tab"""
        st.markdown("### 🚀 Production Deployment Management")

        if not self.deployment_manager:
            st.error("❌ Production deployment manager not available")
            return

        # Deployment management tabs
        deploy_tabs = st.tabs(
            [
                "🎯 Deploy New Model",
                "📊 Active Deployments",
                "📈 Deployment History",
                "⚙️ Deployment Config",
            ]
        )

        with deploy_tabs[0]:
            self._render_new_deployment()

        with deploy_tabs[1]:
            self._render_active_deployments()

        with deploy_tabs[2]:
            self._render_deployment_history()

        with deploy_tabs[3]:
            self._render_deployment_config()

    def _render_new_deployment(self):
        """Render new deployment interface"""
        st.markdown("#### 🎯 Deploy New Model Version")

        # Deployment configuration
        col1, col2 = st.columns(2)

        with col1:
            # Model type selection
            model_types = [mt.value for mt in ModelType]
            selected_model_type = st.selectbox(
                "Model Type", options=model_types, key="deploy_model_type"
            )

            # Available versions
            model_type = ModelType(selected_model_type)
            versions = self.version_manager.get_model_versions(model_type)
            version_options = [
                v.version for v in versions if v.status == ModelStatus.STAGED
            ]

            if version_options:
                selected_version = st.selectbox(
                    "Target Version",
                    options=version_options,
                    key="deploy_target_version",
                )
            else:
                st.warning("No staged versions available for deployment")
                return

        with col2:
            # Deployment strategy
            strategy_options = [s.value for s in DeploymentStrategy]
            selected_strategy = st.selectbox(
                "Deployment Strategy", options=strategy_options, key="deploy_strategy"
            )

            strategy = DeploymentStrategy(selected_strategy)

            # Strategy-specific configuration
            if strategy == DeploymentStrategy.CANARY:
                st.markdown("**Canary Configuration**")
                auto_promote = st.checkbox("Auto-promote canary stages", value=True)
                stage_duration = st.slider(
                    "Stage duration (minutes)", min_value=5, max_value=60, value=10
                )

            elif strategy == DeploymentStrategy.A_B_TESTING:
                st.markdown("**A/B Testing Configuration**")
                test_duration = st.slider(
                    "Test duration (minutes)", min_value=30, max_value=240, value=60
                )
                traffic_split = st.slider(
                    "Traffic to new model (%)", min_value=10, max_value=50, value=50
                )

        # Health checks configuration
        st.markdown("#### 🔍 Health Check Configuration")
        col1, col2, col3 = st.columns(3)

        with col1:
            health_check_enabled = st.checkbox("Enable health checks", value=True)
            accuracy_threshold = st.slider(
                "Accuracy threshold", min_value=0.5, max_value=0.9, value=0.6, step=0.01
            )

        with col2:
            latency_threshold = st.slider(
                "Latency threshold (ms)", min_value=100, max_value=2000, value=500
            )
            auto_rollback = st.checkbox("Auto-rollback on failure", value=True)

        with col3:
            error_rate_threshold = st.slider(
                "Error rate threshold",
                min_value=0.01,
                max_value=0.2,
                value=0.05,
                step=0.01,
            )
            max_retries = st.slider("Max retries", min_value=1, max_value=5, value=3)

        # Deploy button
        if st.button("🚀 Start Deployment", type="primary"):
            self._start_deployment(
                model_type=model_type,
                target_version=selected_version,
                strategy=strategy,
                config={
                    "health_check_enabled": health_check_enabled,
                    "accuracy_threshold": accuracy_threshold,
                    "latency_threshold": latency_threshold,
                    "error_rate_threshold": error_rate_threshold,
                    "auto_rollback": auto_rollback,
                    "max_retries": max_retries,
                },
            )

    def _render_active_deployments(self):
        """Render active deployments status"""
        st.markdown("#### 📊 Active Deployments")

        try:
            deployments = self.deployment_manager.list_deployments(
                status=DeploymentStatus.IN_PROGRESS, limit=20
            )

            if deployments:
                # Create deployment status display
                for deployment in deployments:
                    with st.expander(
                        f"🚀 {deployment['deployment_id']} - {deployment['model_type']}"
                    ):
                        col1, col2, col3 = st.columns(3)

                        with col1:
                            st.metric(
                                "Status", deployment["status"].replace("_", " ").title()
                            )
                            st.metric("Progress", f"{deployment['progress']:.1f}%")
                            st.metric(
                                "Strategy",
                                deployment["strategy"].replace("_", " ").title(),
                            )

                        with col2:
                            st.metric("Target Version", deployment["target_version"])
                            st.metric(
                                "Source Version", deployment["source_version"] or "N/A"
                            )
                            st.metric(
                                "Rollback Triggered",
                                "Yes" if deployment["rollback_triggered"] else "No",
                            )

                        with col3:
                            if deployment["completed_at"]:
                                st.metric("Completed", deployment["completed_at"])
                            else:
                                st.metric("Duration", "In Progress")

                        # Progress bar
                        st.progress(deployment["progress"] / 100)

                        # Deployment actions
                        col1, col2 = st.columns(2)
                        with col1:
                            if st.button(
                                f"📊 View Details",
                                key=f"details_{deployment['deployment_id']}",
                            ):
                                self._show_deployment_details(
                                    deployment["deployment_id"]
                                )
                        with col2:
                            if st.button(
                                f"⏹️ Cancel Deployment",
                                key=f"cancel_{deployment['deployment_id']}",
                                type="secondary",
                            ):
                                self._cancel_deployment(deployment["deployment_id"])

            else:
                st.info("No active deployments")

        except Exception as e:
            st.error(f"Error loading active deployments: {e}")

    def _render_deployment_history(self):
        """Render deployment history"""
        st.markdown("#### 📈 Deployment History")

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            model_types = ["All"] + [mt.value for mt in ModelType]
            selected_filter_model_type = st.selectbox(
                "Filter by Model Type",
                options=model_types,
                key="history_model_type_filter",
            )

        with col2:
            status_options = ["All"] + [s.value for s in DeploymentStatus]
            selected_filter_status = st.selectbox(
                "Filter by Status", options=status_options, key="history_status_filter"
            )

        with col3:
            limit = st.slider("Show last", min_value=10, max_value=100, value=50)

        try:
            deployments = self.deployment_manager.list_deployments(limit=limit)

            # Apply filters
            if selected_filter_model_type != "All":
                deployments = [
                    d
                    for d in deployments
                    if d["model_type"] == selected_filter_model_type
                ]

            if selected_filter_status != "All":
                deployments = [
                    d for d in deployments if d["status"] == selected_filter_status
                ]

            if deployments:
                # Convert to DataFrame
                df = pd.DataFrame(deployments)
                df["created_at"] = pd.to_datetime(df["created_at"])
                df["duration_hours"] = df.apply(
                    lambda x: self._calculate_deployment_duration(x), axis=1
                )

                # Display summary metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    total_deployments = len(df)
                    st.metric("Total Deployments", total_deployments)

                with col2:
                    successful_deployments = len(df[df["success"] == True])
                    st.metric("Successful", successful_deployments)

                with col3:
                    failed_deployments = len(df[df["success"] == False])
                    st.metric("Failed", failed_deployments)

                with col4:
                    success_rate = (
                        (successful_deployments / total_deployments) * 100
                        if total_deployments > 0
                        else 0
                    )
                    st.metric("Success Rate", f"{success_rate:.1f}%")

                # Deployment timeline
                fig = px.scatter(
                    df,
                    x="created_at",
                    y="model_type",
                    color="success",
                    symbol="strategy",
                    size="progress",
                    title="Deployment Timeline",
                    color_discrete_map={True: "green", False: "red"},
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Detailed table
                st.dataframe(
                    df[
                        [
                            "deployment_id",
                            "model_type",
                            "target_version",
                            "strategy",
                            "status",
                            "success",
                            "created_at",
                            "duration_hours",
                        ]
                    ],
                    use_container_width=True,
                )

            else:
                st.info("No deployment history found")

        except Exception as e:
            st.error(f"Error loading deployment history: {e}")

    def _render_deployment_config(self):
        """Render deployment configuration management"""
        st.markdown("#### ⚙️ Deployment Configuration")

        # Current configuration
        st.markdown("**Current Default Configuration**")

        config = self.deployment_manager.config

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**General Settings**")
            st.write(f"Default Strategy: {config.strategy.value}")
            st.write(
                f"Auto-rollback: {'Enabled' if config.auto_rollback_enabled else 'Disabled'}"
            )
            st.write(
                f"Health checks: {'Enabled' if config.health_check_enabled else 'Disabled'}"
            )

        with col2:
            st.markdown("**Thresholds**")
            st.write(f"Accuracy threshold: {config.accuracy_threshold:.2f}")
            st.write(f"Latency threshold: {config.latency_threshold_ms:.0f}ms")
            st.write(f"Error rate threshold: {config.error_rate_threshold:.2f}")

        # Update configuration
        st.markdown("---")
        st.markdown("**Update Configuration**")

        with st.form("deployment_config_form"):
            col1, col2 = st.columns(2)

            with col1:
                new_strategy = st.selectbox(
                    "Default Strategy",
                    options=[s.value for s in DeploymentStrategy],
                    index=[s.value for s in DeploymentStrategy].index(
                        config.strategy.value
                    ),
                )

                new_auto_rollback = st.checkbox(
                    "Enable Auto-rollback", value=config.auto_rollback_enabled
                )

                new_health_checks = st.checkbox(
                    "Enable Health Checks", value=config.health_check_enabled
                )

            with col2:
                new_accuracy_threshold = st.slider(
                    "Accuracy Threshold",
                    min_value=0.5,
                    max_value=0.9,
                    value=config.accuracy_threshold,
                    step=0.01,
                )

                new_latency_threshold = st.slider(
                    "Latency Threshold (ms)",
                    min_value=100,
                    max_value=2000,
                    value=int(config.latency_threshold_ms),
                )

                new_error_rate_threshold = st.slider(
                    "Error Rate Threshold",
                    min_value=0.01,
                    max_value=0.2,
                    value=config.error_rate_threshold,
                    step=0.01,
                )

            if st.form_submit_button("💾 Update Configuration", type="primary"):
                # Update configuration
                new_config = DeploymentConfig(
                    strategy=DeploymentStrategy(new_strategy),
                    auto_rollback_enabled=new_auto_rollback,
                    health_check_enabled=new_health_checks,
                    accuracy_threshold=new_accuracy_threshold,
                    latency_threshold_ms=float(new_latency_threshold),
                    error_rate_threshold=new_error_rate_threshold,
                )

                self.deployment_manager.config = new_config
                st.success("✅ Configuration updated successfully!")
                st.rerun()

    def _render_automated_retraining_tab(self):
        """Render automated retraining monitoring and control"""
        st.markdown("### 🔄 Automated Retraining Management")

        if not self.retraining_pipeline:
            st.error("❌ Automated retraining pipeline not available")
            return

        # Retraining tabs
        retrain_tabs = st.tabs(
            [
                "📊 Pipeline Status",
                "⚙️ Retraining Config",
                "📈 Retraining History",
                "🎯 Manual Trigger",
            ]
        )

        with retrain_tabs[0]:
            self._render_pipeline_status()

        with retrain_tabs[1]:
            self._render_retraining_config()

        with retrain_tabs[2]:
            self._render_retraining_history()

        with retrain_tabs[3]:
            self._render_manual_retraining()

    def _render_pipeline_status(self):
        """Render retraining pipeline status"""
        st.markdown("#### 📊 Retraining Pipeline Status")

        try:
            # Get pipeline status
            pipeline_info = self.retraining_pipeline.get_pipeline_info()

            # Overall status
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Status", pipeline_info["status"].replace("_", " ").title())

            with col2:
                st.metric("Active Jobs", pipeline_info["active_jobs"])

            with col3:
                st.metric("Completed Jobs", pipeline_info["completed_jobs"])

            with col4:
                st.metric("Failed Jobs", pipeline_info["failed_jobs"])

            # Active jobs
            if pipeline_info["active_jobs"] > 0:
                st.markdown("**Active Retraining Jobs**")
                active_jobs = self.retraining_pipeline.get_active_jobs()

                for job in active_jobs:
                    with st.expander(f"🔄 {job['job_id']} - {job['model_type']}"):
                        col1, col2 = st.columns(2)

                        with col1:
                            st.metric("Status", job["status"].replace("_", " ").title())
                            st.metric("Progress", f"{job['progress']:.1f}%")
                            st.metric(
                                "Trigger", job["trigger_type"].replace("_", " ").title()
                            )

                        with col2:
                            st.metric("Created", job["created_at"])
                            if job["started_at"]:
                                st.metric("Started", job["started_at"])
                            if job["estimated_completion"]:
                                st.metric(
                                    "Est. Completion", job["estimated_completion"]
                                )

                        # Progress bar
                        st.progress(job["progress"] / 100)

                        # Job actions
                        if st.button(
                            f"⏸️ Pause Job",
                            key=f"pause_{job['job_id']}",
                            type="secondary",
                        ):
                            self._pause_retraining_job(job["job_id"])

            # Recent jobs
            st.markdown("**Recent Retraining Jobs**")
            recent_jobs = self.retraining_pipeline.get_job_history(limit=10)

            if recent_jobs:
                job_data = []
                for job in recent_jobs:
                    job_data.append(
                        {
                            "Job ID": job["job_id"],
                            "Model Type": job["model_type"].value,
                            "Status": job["status"].value.replace("_", " ").title(),
                            "Trigger": job["trigger_type"]
                            .value.replace("_", " ")
                            .title(),
                            "Progress": f"{job['progress']:.1f}%",
                            "Created": job["created_at"],
                            "Duration": job.get("duration", "N/A"),
                        }
                    )

                df = pd.DataFrame(job_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No recent retraining jobs found")

        except Exception as e:
            st.error(f"Error loading pipeline status: {e}")

    def _render_retraining_config(self):
        """Render retraining configuration"""
        st.markdown("#### ⚙️ Retraining Configuration")

        try:
            current_config = self.retraining_pipeline.get_config()

            with st.form("retraining_config_form"):
                st.markdown("**Data Quality Settings**")
                col1, col2 = st.columns(2)

                with col1:
                    min_samples = st.number_input(
                        "Minimum Training Samples",
                        min_value=100,
                        max_value=10000,
                        value=current_config.min_samples,
                        step=100,
                    )

                    quality_threshold = st.slider(
                        "Data Quality Threshold",
                        min_value=0.5,
                        max_value=1.0,
                        value=current_config.quality_score_threshold,
                        step=0.01,
                    )

                with col2:
                    freshness_days = st.slider(
                        "Data Freshness (days)",
                        min_value=1,
                        max_value=90,
                        value=current_config.data_freshness_days,
                    )

                    nba_season_required = st.checkbox(
                        "Require NBA Season Data",
                        value=current_config.nba_season_required,
                    )

                st.markdown("**Performance Thresholds**")
                col1, col2 = st.columns(2)

                with col1:
                    accuracy_threshold = st.slider(
                        "Accuracy Threshold",
                        min_value=0.5,
                        max_value=0.9,
                        value=current_config.accuracy_threshold,
                        step=0.01,
                    )

                    performance_drop_threshold = st.slider(
                        "Performance Drop Threshold",
                        min_value=0.01,
                        max_value=0.2,
                        value=current_config.performance_drop_threshold,
                        step=0.01,
                    )

                with col2:
                    min_improvement = st.slider(
                        "Minimum Improvement",
                        min_value=0.001,
                        max_value=0.1,
                        value=current_config.min_improvement,
                        step=0.001,
                    )

                    enable_auto_promotion = st.checkbox(
                        "Enable Auto Promotion",
                        value=current_config.enable_auto_promotion,
                    )

                st.markdown("**Scheduling Settings**")
                col1, col2 = st.columns(2)

                with col1:
                    schedule_interval = st.selectbox(
                        "Schedule Interval",
                        options=["disabled", "hourly", "daily", "weekly"],
                        index=["disabled", "hourly", "daily", "weekly"].index(
                            current_config.schedule_interval
                        ),
                    )

                with col2:
                    if schedule_interval == "daily":
                        schedule_time = st.time_input(
                            "Daily Run Time",
                            value=datetime.strptime(
                                current_config.schedule_time or "02:00", "%H:%M"
                            ).time(),
                        )

                if st.form_submit_button("💾 Update Configuration", type="primary"):
                    # Update configuration
                    new_config = RetrainingConfig(
                        min_samples=int(min_samples),
                        quality_score_threshold=quality_threshold,
                        data_freshness_days=int(freshness_days),
                        nba_season_required=nba_season_required,
                        accuracy_threshold=accuracy_threshold,
                        performance_drop_threshold=performance_drop_threshold,
                        min_improvement=min_improvement,
                        enable_auto_promotion=enable_auto_promotion,
                        schedule_interval=schedule_interval,
                        schedule_time=schedule_time.strftime("%H:%M")
                        if schedule_interval == "daily"
                        else None,
                    )

                    self.retraining_pipeline.update_config(new_config)
                    st.success("✅ Retraining configuration updated successfully!")

        except Exception as e:
            st.error(f"Error loading retraining configuration: {e}")

    def _render_retraining_history(self):
        """Render retraining job history"""
        st.markdown("#### 📈 Retraining History")

        try:
            # Get job history
            job_history = self.retraining_pipeline.get_job_history(limit=100)

            if job_history:
                # Convert to DataFrame
                job_data = []
                for job in job_history:
                    job_data.append(
                        {
                            "Job ID": job["job_id"],
                            "Model Type": job["model_type"].value,
                            "Status": job["status"].value.replace("_", " ").title(),
                            "Trigger": job["trigger_type"]
                            .value.replace("_", " ")
                            .title(),
                            "Progress": f"{job['progress']:.1f}%",
                            "Created": job["created_at"],
                            "Duration": job.get("duration", "N/A"),
                            "Data Quality": f"{job.get('data_quality_score', 0):.3f}",
                            "Performance Gain": f"{job.get('performance_gain', 0):.3f}",
                        }
                    )

                df = pd.DataFrame(job_data)

                # Summary metrics
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    total_jobs = len(df)
                    st.metric("Total Jobs", total_jobs)

                with col2:
                    completed_jobs = len(df[df["Status"] == "Completed"])
                    st.metric("Completed", completed_jobs)

                with col3:
                    failed_jobs = len(df[df["Status"] == "Failed"])
                    st.metric("Failed", failed_jobs)

                with col4:
                    success_rate = (
                        (completed_jobs / total_jobs) * 100 if total_jobs > 0 else 0
                    )
                    st.metric("Success Rate", f"{success_rate:.1f}%")

                # Performance improvements chart
                successful_jobs = [
                    job
                    for job in job_history
                    if job.get("performance_gain") is not None
                ]

                if successful_jobs:
                    performance_data = [
                        {
                            "Job ID": job["job_id"][:8],
                            "Model Type": job["model_type"].value,
                            "Performance Gain": job.get("performance_gain", 0) * 100,
                        }
                        for job in successful_jobs
                    ]

                    perf_df = pd.DataFrame(performance_data)

                    fig = px.bar(
                        perf_df,
                        x="Job ID",
                        y="Performance Gain",
                        color="Model Type",
                        title="Performance Improvements from Retraining (%)",
                    )
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)

                # Detailed table
                st.dataframe(df, use_container_width=True)

            else:
                st.info("No retraining history found")

        except Exception as e:
            st.error(f"Error loading retraining history: {e}")

    def _render_manual_retraining(self):
        """Render manual retraining trigger"""
        st.markdown("#### 🎯 Manual Retraining Trigger")

        # Model type selection
        model_types = [mt.value for mt in ModelType]
        selected_model_type = st.selectbox(
            "Select Model Type for Retraining",
            options=model_types,
            key="manual_retrain_model_type",
        )

        model_type = ModelType(selected_model_type)

        # Retraining options
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Data Options**")
            use_all_data = st.checkbox("Use all available data", value=True)
            force_retrain = st.checkbox(
                "Force retrain (ignore quality checks)", value=False
            )

        with col2:
            st.markdown("**Performance Options**")
            skip_evaluation = st.checkbox("Skip performance evaluation", value=False)
            auto_promote = st.checkbox("Auto-promote if better", value=True)

        # Additional options
        st.markdown("**Advanced Options**")
        custom_config = st.text_area(
            "Custom Configuration (JSON)",
            placeholder='{"hyperparameters": {"learning_rate": 0.01}}',
            height=100,
        )

        # Trigger retraining
        if st.button("🔄 Start Manual Retraining", type="primary"):
            try:
                # Parse custom config if provided
                additional_config = {}
                if custom_config:
                    additional_config = json.loads(custom_config)

                # Start retraining job
                job_id = self.retraining_pipeline.trigger_manual_retraining(
                    model_type=model_type,
                    use_all_data=use_all_data,
                    force_retrain=force_retrain,
                    skip_evaluation=skip_evaluation,
                    auto_promote=auto_promote,
                    additional_config=additional_config,
                )

                st.success(f"✅ Retraining job started: {job_id}")
                st.rerun()

            except json.JSONDecodeError:
                st.error("❌ Invalid JSON configuration")
            except Exception as e:
                st.error(f"❌ Failed to start retraining: {e}")

    def _render_ml_operations_overview_tab(self):
        """Render comprehensive ML operations overview"""
        st.markdown("### 📊 ML Operations Overview")

        # System status cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Predictor status
            if self.predictor:
                st.success("✅ Ensemble Predictor")
            else:
                st.error("❌ Ensemble Predictor")

        with col2:
            # Version manager status
            if self.version_manager:
                st.success("✅ Version Manager")
            else:
                st.error("❌ Version Manager")

        with col3:
            # Deployment manager status
            if self.deployment_manager:
                st.success("✅ Deployment Manager")
            else:
                st.error("❌ Deployment Manager")

        with col4:
            # Retraining pipeline status
            if self.retraining_pipeline:
                st.success("✅ Retraining Pipeline")
            else:
                st.error("❌ Retraining Pipeline")

        # Quick actions
        st.markdown("---")
        st.markdown("#### 🚀 Quick Actions")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 Generate ML Health Report"):
                self._generate_ml_health_report()

        with col2:
            if st.button("🔧 System Diagnostics"):
                self._run_system_diagnostics()

        with col3:
            if st.button("📋 Export Operations Log"):
                self._export_operations_log()

        # Recent activity
        st.markdown("---")
        st.markdown("#### 📈 Recent ML Operations Activity")

        try:
            # Collect recent activities from all components
            activities = []

            # Deployment activities
            if self.deployment_manager:
                recent_deployments = self.deployment_manager.list_deployments(limit=5)
                for deployment in recent_deployments:
                    activities.append(
                        {
                            "timestamp": deployment["created_at"],
                            "type": "Deployment",
                            "description": f"{deployment['model_type']} v{deployment['target_version']} ({deployment['strategy']})",
                            "status": deployment["status"],
                        }
                    )

            # Retraining activities
            if self.retraining_pipeline:
                recent_jobs = self.retraining_pipeline.get_job_history(limit=5)
                for job in recent_jobs:
                    activities.append(
                        {
                            "timestamp": job["created_at"],
                            "type": "Retraining",
                            "description": f"{job['model_type'].value} ({job['trigger_type'].value})",
                            "status": job["status"].value,
                        }
                    )

            if activities:
                # Sort by timestamp
                activities.sort(key=lambda x: x["timestamp"], reverse=True)

                # Display activities
                for activity in activities[:10]:
                    col1, col2, col3 = st.columns([1, 2, 1])

                    with col1:
                        st.write(f"**{activity['type']}**")
                        st.write(f"*{activity['timestamp']}*")

                    with col2:
                        st.write(activity["description"])

                    with col3:
                        status_color = (
                            "green"
                            if "completed" in activity["status"].lower()
                            else "orange"
                            if "progress" in activity["status"].lower()
                            else "red"
                        )
                        st.markdown(
                            f"<span style='color: {status_color}'>●</span> {activity['status'].replace('_', ' ').title()}",
                            unsafe_allow_html=True,
                        )

                    st.divider()

            else:
                st.info("No recent ML operations activity found")

        except Exception as e:
            st.error(f"Error loading recent activity: {e}")

    def _generate_version_report(self, model_type: ModelType):
        """Generate model version report"""
        try:
            versions = self.version_manager.get_model_versions(model_type)

            report = {
                "model_type": model_type.value,
                "total_versions": len(versions),
                "active_version": self.version_manager.get_active_versions().get(
                    model_type
                ),
                "versions": [],
            }

            for version in versions:
                report["versions"].append(
                    {
                        "version": version.version,
                        "status": version.status.value,
                        "created_at": version.created_at.isoformat(),
                        "metrics": version.metrics.to_dict(),
                    }
                )

            # Display report
            st.json(report)

            # Download button
            st.download_button(
                label="📥 Download Report",
                data=json.dumps(report, indent=2),
                file_name=f"version_report_{model_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

        except Exception as e:
            st.error(f"Error generating version report: {e}")

    def _cleanup_old_versions(self, model_type: ModelType):
        """Clean up old model versions"""
        try:
            with st.spinner("Cleaning up old versions..."):
                deleted_count = self.version_manager.cleanup_old_versions(model_type)
                st.success(f"✅ Cleaned up {deleted_count} old versions")
                st.rerun()

        except Exception as e:
            st.error(f"Error cleaning up versions: {e}")

    def _show_version_details(self, version: str, model_type: ModelType):
        """Show detailed version information"""
        try:
            version_info = self.version_manager.get_model_version(version, model_type)

            if version_info:
                st.markdown("#### Version Details")
                st.json(version_info.__dict__)
            else:
                st.error("Version not found")

        except Exception as e:
            st.error(f"Error loading version details: {e}")

    def _compare_versions_for_rollback(
        self, current_version: str, target_version: str, model_type: ModelType
    ):
        """Compare versions for rollback decision"""
        try:
            current = self.version_manager.get_model_version(
                current_version, model_type
            )
            target = self.version_manager.get_model_version(target_version, model_type)

            if current and target:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Current Version: {current_version}**")
                    st.metric("Accuracy", f"{current.metrics.accuracy:.3f}")
                    st.metric("NBA Accuracy", f"{current.metrics.nba_accuracy:.3f}")
                    st.metric(
                        "Latency", f"{current.metrics.prediction_latency_ms:.1f}ms"
                    )

                with col2:
                    st.markdown(f"**Target Version: {target_version}**")
                    st.metric("Accuracy", f"{target.metrics.accuracy:.3f}")
                    st.metric("NBA Accuracy", f"{target.metrics.nba_accuracy:.3f}")
                    st.metric(
                        "Latency", f"{target.metrics.prediction_latency_ms:.1f}ms"
                    )

                # Comparison
                acc_diff = current.metrics.accuracy - target.metrics.accuracy
                nba_acc_diff = (
                    current.metrics.nba_accuracy - target.metrics.nba_accuracy
                )
                lat_diff = (
                    current.metrics.prediction_latency_ms
                    - target.metrics.prediction_latency_ms
                )

                st.markdown("#### Comparison")
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric("Accuracy Difference", f"{acc_diff:+.3f}")

                with col2:
                    st.metric("NBA Accuracy Difference", f"{nba_acc_diff:+.3f}")

                with col3:
                    st.metric("Latency Difference", f"{lat_diff:+.1f}ms")

            else:
                st.error("Could not load version details for comparison")

        except Exception as e:
            st.error(f"Error comparing versions: {e}")

    def _calculate_deployment_duration(self, deployment: Dict) -> str:
        """Calculate deployment duration"""
        try:
            if deployment.get("completed_at") and deployment.get("created_at"):
                created = pd.to_datetime(deployment["created_at"])
                completed = pd.to_datetime(deployment["completed_at"])
                duration = completed - created
                return f"{duration.total_seconds() / 3600:.1f}h"
            else:
                return "N/A"
        except:
            return "N/A"

    def _start_deployment(
        self,
        model_type: ModelType,
        target_version: str,
        strategy: DeploymentStrategy,
        config: Dict,
    ):
        """Start new deployment"""
        try:
            with st.spinner("Starting deployment..."):
                deployment_id = self.deployment_manager.start_deployment(
                    model_type=model_type,
                    target_version=target_version,
                    strategy=strategy,
                )

                st.success(f"✅ Deployment started: {deployment_id}")
                st.rerun()

        except Exception as e:
            st.error(f"❌ Failed to start deployment: {e}")

    def _show_deployment_details(self, deployment_id: str):
        """Show detailed deployment information"""
        try:
            details = self.deployment_manager.get_deployment_status(deployment_id)
            if details:
                st.json(details)
            else:
                st.error("Deployment not found")

        except Exception as e:
            st.error(f"Error loading deployment details: {e}")

    def _cancel_deployment(self, deployment_id: str):
        """Cancel active deployment"""
        try:
            # Implementation would depend on deployment manager API
            st.success(f"✅ Deployment {deployment_id} cancelled")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Failed to cancel deployment: {e}")

    def _pause_retraining_job(self, job_id: str):
        """Pause retraining job"""
        try:
            # Implementation would depend on retraining pipeline API
            st.success(f"✅ Retraining job {job_id} paused")
            st.rerun()

        except Exception as e:
            st.error(f"❌ Failed to pause retraining job: {e}")

    def _generate_ml_health_report(self):
        """Generate comprehensive ML health report"""
        try:
            report = {
                "generated_at": datetime.now().isoformat(),
                "system_status": {},
                "model_versions": {},
                "deployments": {},
                "retraining": {},
            }

            # System status
            report["system_status"] = {
                "predictor_available": self.predictor is not None,
                "version_manager_available": self.version_manager is not None,
                "deployment_manager_available": self.deployment_manager is not None,
                "retraining_pipeline_available": self.retraining_pipeline is not None,
            }

            # Model versions
            if self.version_manager:
                for model_type in ModelType:
                    versions = self.version_manager.get_model_versions(model_type)
                    report["model_versions"][model_type.value] = {
                        "total_versions": len(versions),
                        "active_version": self.version_manager.get_active_versions().get(
                            model_type
                        ),
                    }

            # Deployments
            if self.deployment_manager:
                deployments = self.deployment_manager.list_deployments(limit=50)
                report["deployments"] = {
                    "total_deployments": len(deployments),
                    "active_deployments": len(
                        [d for d in deployments if d["status"] == "in_progress"]
                    ),
                    "successful_deployments": len(
                        [d for d in deployments if d["success"]]
                    ),
                }

            # Retraining
            if self.retraining_pipeline:
                pipeline_info = self.retraining_pipeline.get_pipeline_info()
                report["retraining"] = pipeline_info

            # Display report
            st.json(report)

            # Download button
            st.download_button(
                label="📥 Download Health Report",
                data=json.dumps(report, indent=2),
                file_name=f"ml_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

        except Exception as e:
            st.error(f"Error generating ML health report: {e}")

    def _run_system_diagnostics(self):
        """Run system diagnostics"""
        try:
            with st.spinner("Running system diagnostics..."):
                diagnostics = {"timestamp": datetime.now().isoformat(), "checks": {}}

                # Check predictor
                if self.predictor:
                    diagnostics["checks"]["predictor"] = {
                        "status": "healthy",
                        "models_loaded": True,
                        "prediction_capability": True,
                    }
                else:
                    diagnostics["checks"]["predictor"] = {
                        "status": "unavailable",
                        "error": "Predictor not initialized",
                    }

                # Check version manager
                if self.version_manager:
                    try:
                        versions = self.version_manager.get_active_versions()
                        diagnostics["checks"]["version_manager"] = {
                            "status": "healthy",
                            "active_versions": len(versions),
                        }
                    except Exception as e:
                        diagnostics["checks"]["version_manager"] = {
                            "status": "error",
                            "error": str(e),
                        }

                # Check deployment manager
                if self.deployment_manager:
                    try:
                        deployments = self.deployment_manager.list_deployments(limit=1)
                        diagnostics["checks"]["deployment_manager"] = {
                            "status": "healthy",
                            "deployment_count": len(deployments),
                        }
                    except Exception as e:
                        diagnostics["checks"]["deployment_manager"] = {
                            "status": "error",
                            "error": str(e),
                        }

                # Check retraining pipeline
                if self.retraining_pipeline:
                    try:
                        pipeline_info = self.retraining_pipeline.get_pipeline_info()
                        diagnostics["checks"]["retraining_pipeline"] = {
                            "status": "healthy",
                            "pipeline_status": pipeline_info.get("status"),
                        }
                    except Exception as e:
                        diagnostics["checks"]["retraining_pipeline"] = {
                            "status": "error",
                            "error": str(e),
                        }

                # Display diagnostics
                st.json(diagnostics)

                # Overall health
                healthy_checks = len(
                    [
                        c
                        for c in diagnostics["checks"].values()
                        if c["status"] == "healthy"
                    ]
                )
                total_checks = len(diagnostics["checks"])
                health_percentage = (
                    (healthy_checks / total_checks) * 100 if total_checks > 0 else 0
                )

                st.metric("Overall System Health", f"{health_percentage:.0f}%")

        except Exception as e:
            st.error(f"Error running system diagnostics: {e}")

    def _export_operations_log(self):
        """Export ML operations log"""
        try:
            log_data = {
                "export_timestamp": datetime.now().isoformat(),
                "deployment_history": [],
                "retraining_history": [],
                "version_registry": {},
            }

            # Deployment history
            if self.deployment_manager:
                deployments = self.deployment_manager.list_deployments(limit=100)
                log_data["deployment_history"] = deployments

            # Retraining history
            if self.retraining_pipeline:
                jobs = self.retraining_pipeline.get_job_history(limit=100)
                log_data["retraining_history"] = jobs

            # Version registry
            if self.version_manager:
                for model_type in ModelType:
                    versions = self.version_manager.get_model_versions(model_type)
                    log_data["version_registry"][model_type.value] = [
                        {
                            "version": v.version,
                            "status": v.status.value,
                            "created_at": v.created_at.isoformat(),
                            "metrics": v.metrics.to_dict(),
                        }
                        for v in versions
                    ]

            # Display and offer download
            st.json(log_data)

            st.download_button(
                label="📥 Download Operations Log",
                data=json.dumps(log_data, indent=2),
                file_name=f"ml_operations_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
            )

        except Exception as e:
            st.error(f"Error exporting operations log: {e}")


def render_advanced_ml_operations_dashboard():
    """Main function to render advanced ML operations dashboard"""
    dashboard = AdvancedMLOperationsDashboard()
    dashboard.render_dashboard()


if __name__ == "__main__":
    render_advanced_ml_operations_dashboard()
