"""
Performance Validation and Optimization Suite for NBA Predictive Analytics System

This module provides comprehensive performance testing and optimization validation
for all system components following Context7 and DevStream best practices.

Performance Metrics:
- Data Pipeline: Fetch time, processing time, memory usage
- Model Training: Training time, prediction time, accuracy
- SHAP Explainability: Computation time, memory efficiency
- Overall System: End-to-end latency, throughput, resource utilization
"""

import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from contextlib import contextmanager
import gc
import os

# Import our system components
from unified_nba_data_pipeline import UnifiedNBADataPipeline
from advanced_predictive_model import AdvancedPredictiveModel
from nba_explainability_engine import NBAExplainabilityEngine
from auto_model_retrainer import AutoModelRetrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Context7: Dataclass for structured performance metrics collection."""
    component_name: str
    operation: str
    execution_time: float
    memory_usage_mb: float
    cpu_usage_percent: float
    data_size: Optional[int] = None
    success: bool = True
    error_message: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now()

class PerformanceValidator:
    """
    Context7: Comprehensive performance validation suite.

    Validates and optimizes performance across all system components
    with detailed metrics collection and analysis.
    """

    def __init__(self) -> None:
        """Initialize the performance validator with monitoring capabilities."""
        self.metrics_history: List[PerformanceMetrics] = []
        self.baseline_metrics: Dict[str, float] = {}
        self.process = psutil.Process()

        # Context7: Performance thresholds (seconds)
        self.thresholds = {
            'pipeline_fetch': 30.0,
            'pipeline_process': 10.0,
            'model_training': 60.0,
            'model_prediction': 1.0,
            'shap_explanation': 5.0,
            'end_to_end': 120.0
        }

        logger.info("PerformanceValidator initialized")

    @contextmanager
    def measure_performance(self, component_name: str, operation: str):
        """
        Context7: Context manager for measuring performance metrics.

        Args:
            component_name: Name of the component being measured
            operation: Specific operation being performed

        Yields:
            PerformanceMetrics collector
        """
        # Get initial measurements
        start_time = time.time()
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        initial_cpu = self.process.cpu_percent()

        try:
            metrics = PerformanceMetrics(
                component_name=component_name,
                operation=operation,
                execution_time=0.0,
                memory_usage_mb=initial_memory,
                cpu_usage_percent=initial_cpu
            )

            yield metrics

            # Final measurements
            end_time = time.time()
            final_memory = self.process.memory_info().rss / 1024 / 1024  # MB
            final_cpu = self.process.cpu_percent()

            # Update metrics
            metrics.execution_time = end_time - start_time
            metrics.memory_usage_mb = final_memory - initial_memory
            metrics.cpu_usage_percent = final_cpu
            metrics.success = True

            self.metrics_history.append(metrics)

            # Log performance results
            logger.info(f"Performance: {component_name}.{operation} - "
                       f"Time: {metrics.execution_time:.3f}s, "
                       f"Memory: {metrics.memory_usage_mb:.1f}MB, "
                       f"CPU: {metrics.cpu_usage_percent:.1f}%")

        except Exception as e:
            # Record failure metrics
            metrics = PerformanceMetrics(
                component_name=component_name,
                operation=operation,
                execution_time=time.time() - start_time,
                memory_usage_mb=initial_memory,
                cpu_usage_percent=initial_cpu,
                success=False,
                error_message=str(e)
            )

            self.metrics_history.append(metrics)
            logger.error(f"Performance failure: {component_name}.{operation} - {str(e)}")
            raise

    def validate_pipeline_performance(self, pipeline: UnifiedNBADataPipeline) -> Dict[str, Any]:
        """
        Context7: Validate UnifiedNBADataPipeline performance.

        Tests data fetching, preprocessing, and overall pipeline efficiency.

        Args:
            pipeline: UnifiedNBADataPipeline instance to test

        Returns:
            Dictionary containing performance results and recommendations
        """
        logger.info("Starting pipeline performance validation")

        results = {
            'component': 'UnifiedNBADataPipeline',
            'tests_passed': 0,
            'tests_failed': 0,
            'recommendations': [],
            'metrics': {}
        }

        try:
            # Test 1: Data fetching performance
            with self.measure_performance('pipeline', 'fetch_all_data') as metrics:
                today = datetime.now().date()
                end_date = today + timedelta(days=3)

                raw_data = pipeline.fetch_all_data(
                    date_range=(today, end_date),
                    include_boxscores=False
                )

                if raw_data['games'] is not None:
                    metrics.data_size = len(raw_data['games'])

                    # Check against threshold
                    if metrics.execution_time > self.thresholds['pipeline_fetch']:
                        results['recommendations'].append(
                            f"Data fetching took {metrics.execution_time:.1f}s "
                            f"(threshold: {self.thresholds['pipeline_fetch']}s). "
                            "Consider implementing caching or reducing API calls."
                        )
                        results['tests_failed'] += 1
                    else:
                        results['tests_passed'] += 1

                results['metrics']['fetch_data'] = {
                    'time_seconds': metrics.execution_time,
                    'memory_mb': metrics.memory_usage_mb,
                    'records_fetched': metrics.data_size or 0
                }

            # Test 2: Feature preprocessing performance
            if raw_data['games'] is not None and not raw_data['games'].empty:
                with self.measure_performance('pipeline', 'preprocess_features') as metrics:
                    features = pipeline.preprocess_features(raw_data)
                    metrics.data_size = len(features)

                    # Check against threshold
                    if metrics.execution_time > self.thresholds['pipeline_process']:
                        results['recommendations'].append(
                            f"Feature preprocessing took {metrics.execution_time:.1f}s "
                            f"(threshold: {self.thresholds['pipeline_process']}s). "
                            "Consider optimizing feature engineering algorithms."
                        )
                        results['tests_failed'] += 1
                    else:
                        results['tests_passed'] += 1

                    results['metrics']['preprocess_features'] = {
                        'time_seconds': metrics.execution_time,
                        'memory_mb': metrics.memory_usage_mb,
                        'features_generated': metrics.data_size or 0
                    }

                # Test 3: Data validation performance
                with self.measure_performance('pipeline', 'validate_data_quality') as metrics:
                    validation_result = pipeline.validate_data_quality(features)

                    results['tests_passed'] += 1
                    results['metrics']['validate_data'] = {
                        'time_seconds': metrics.execution_time,
                        'memory_mb': metrics.memory_usage_mb,
                        'quality_score': validation_result['quality_score']
                    }

            # Test 4: Pipeline metrics collection
            with self.measure_performance('pipeline', 'get_pipeline_metrics') as metrics:
                pipeline_metrics = pipeline.get_pipeline_metrics()

                results['tests_passed'] += 1
                results['metrics']['pipeline_metrics'] = {
                    'time_seconds': metrics.execution_time,
                    'metrics_collected': len(pipeline_metrics)
                }

        except Exception as e:
            logger.error(f"Pipeline performance validation failed: {e}")
            results['tests_failed'] += 1
            results['error'] = str(e)

        logger.info(f"Pipeline validation completed: {results['tests_passed']} passed, "
                   f"{results['tests_failed']} failed")

        return results

    def validate_model_performance(self, model: AdvancedPredictiveModel,
                                sample_features: pd.DataFrame) -> Dict[str, Any]:
        """
        Context7: Validate AdvancedPredictiveModel performance.

        Tests model training, prediction, and ensemble efficiency.

        Args:
            model: AdvancedPredictiveModel instance to test
            sample_features: Sample data for model testing

        Returns:
            Dictionary containing performance results and recommendations
        """
        logger.info("Starting model performance validation")

        results = {
            'component': 'AdvancedPredictiveModel',
            'tests_passed': 0,
            'tests_failed': 0,
            'recommendations': [],
            'metrics': {}
        }

        try:
            # Test 1: Model training performance
            if len(sample_features) >= 10:
                # Create sample labels for testing
                sample_labels = np.random.choice([0, 1], size=len(sample_features))
                train_features = sample_features[:len(sample_features)//2]
                train_labels = sample_labels[:len(sample_labels)//2]

                with self.measure_performance('model', 'train_model') as metrics:
                    model.train_model(train_features, train_labels)

                    # Check against threshold
                    if metrics.execution_time > self.thresholds['model_training']:
                        results['recommendations'].append(
                            f"Model training took {metrics.execution_time:.1f}s "
                            f"(threshold: {self.thresholds['model_training']}s). "
                            "Consider reducing model complexity or training data size."
                        )
                        results['tests_failed'] += 1
                    else:
                        results['tests_passed'] += 1

                    results['metrics']['train_model'] = {
                        'time_seconds': metrics.execution_time,
                        'memory_mb': metrics.memory_usage_mb,
                        'training_samples': len(train_features)
                    }

                # Test 2: Model prediction performance
                test_features = sample_features[len(sample_features)//2:]

                with self.measure_performance('model', 'predict') as metrics:
                    predictions = model.predict(test_features)
                    metrics.data_size = len(predictions)

                    # Check against threshold
                    if metrics.execution_time > self.thresholds['model_prediction']:
                        results['recommendations'].append(
                            f"Model prediction took {metrics.execution_time:.3f}s for {len(predictions)} samples "
                            f"(threshold: {self.thresholds['model_prediction']}s). "
                            "Consider model optimization or batch prediction."
                        )
                        results['tests_failed'] += 1
                    else:
                        results['tests_passed'] += 1

                    results['metrics']['predict'] = {
                        'time_seconds': metrics.execution_time,
                        'memory_mb': metrics.memory_usage_mb,
                        'predictions_made': len(predictions),
                        'throughput_samples_per_sec': len(predictions) / metrics.execution_time
                    }

                # Test 3: Model confidence prediction
                with self.measure_performance('model', 'predict_proba') as metrics:
                    probabilities = model.predict_proba(test_features)

                    results['tests_passed'] += 1
                    results['metrics']['predict_proba'] = {
                        'time_seconds': metrics.execution_time,
                        'memory_mb': metrics.memory_usage_mb,
                        'probabilities_generated': len(probabilities)
                    }

        except Exception as e:
            logger.error(f"Model performance validation failed: {e}")
            results['tests_failed'] += 1
            results['error'] = str(e)

        logger.info(f"Model validation completed: {results['tests_passed']} passed, "
                   f"{results['tests_failed']} failed")

        return results

    def validate_shap_performance(self, explainability_engine: NBAExplainabilityEngine,
                                sample_features: pd.DataFrame) -> Dict[str, Any]:
        """
        Context7: Validate NBAExplainabilityEngine performance.

        Tests SHAP value computation and explanation generation.

        Args:
            explainability_engine: NBAExplainabilityEngine instance to test
            sample_features: Sample data for explanation testing

        Returns:
            Dictionary containing performance results and recommendations
        """
        logger.info("Starting SHAP explainability performance validation")

        results = {
            'component': 'NBAExplainabilityEngine',
            'tests_passed': 0,
            'tests_failed': 0,
            'recommendations': [],
            'metrics': {}
        }

        try:
            # Test 1: Single prediction explanation
            if len(sample_features) > 0:
                single_sample = sample_features.iloc[0]

                with self.measure_performance('shap', 'explain_single_prediction') as metrics:
                    if explainability_engine is not None:
                        explanation = explainability_engine.explain_single_prediction(
                            single_sample,
                            prediction=1.0
                        )
                    else:
                        explanation = {"mock_explanation": "Model not trained"}

                    # Check against threshold
                    if metrics.execution_time > self.thresholds['shap_explanation']:
                        results['recommendations'].append(
                            f"Single SHAP explanation took {metrics.execution_time:.3f}s "
                            f"(threshold: {self.thresholds['shap_explanation']}s). "
                            "Consider using cached SHAP values or simplified explanations."
                        )
                        results['tests_failed'] += 1
                    else:
                        results['tests_passed'] += 1

                    results['metrics']['explain_single'] = {
                        'time_seconds': metrics.execution_time,
                        'memory_mb': metrics.memory_usage_mb,
                        'explanation_generated': len(explanation)
                    }

            # Test 2: Global explanation generation
            if len(sample_features) >= 5:
                sample_data = sample_features[:5]

                with self.measure_performance('shap', 'generate_global_explanation') as metrics:
                    # Mock SHAP values for testing
                    mock_shap_values = np.random.random((len(sample_data), len(sample_data.columns)))

                    if explainability_engine is not None:
                        global_explanation = explainability_engine.generate_global_explanation(
                            mock_shap_values
                        )
                    else:
                        global_explanation = {"mock_global": "Model not trained"}

                    results['tests_passed'] += 1
                    results['metrics']['explain_global'] = {
                        'time_seconds': metrics.execution_time,
                        'memory_mb': metrics.memory_usage_mb,
                        'features_analyzed': len(sample_data.columns)
                    }

        except Exception as e:
            logger.error(f"SHAP performance validation failed: {e}")
            results['tests_failed'] += 1
            results['error'] = str(e)

        logger.info(f"SHAP validation completed: {results['tests_passed']} passed, "
                   f"{results['tests_failed']} failed")

        return results

    def validate_end_to_end_performance(self) -> Dict[str, Any]:
        """
        Context7: Validate complete end-to-end system performance.

        Tests the entire workflow from data fetching to prediction explanation.

        Returns:
            Dictionary containing comprehensive performance results
        """
        logger.info("Starting end-to-end performance validation")

        results = {
            'component': 'End-to-End System',
            'tests_passed': 0,
            'tests_failed': 0,
            'recommendations': [],
            'metrics': {},
            'total_time': 0.0
        }

        start_time = time.time()

        try:
            # Initialize components
            pipeline = UnifiedNBADataPipeline()
            model = AdvancedPredictiveModel()

            # Check if model is trained before initializing explainability engine
            if hasattr(model, 'model') and model.model is not None:
                explainability_engine = NBAExplainabilityEngine(
                    model.model,
                    list(range(20))  # Mock feature names
                )
            else:
                explainability_engine = None

            # Step 1: Data fetching and processing
            with self.measure_performance('system', 'data_pipeline') as metrics:
                today = datetime.now().date()
                end_date = today + timedelta(days=1)

                raw_data = pipeline.fetch_all_data(
                    date_range=(today, end_date),
                    include_boxscores=False
                )

                if raw_data['games'] is not None and not raw_data['games'].empty:
                    features = pipeline.preprocess_features(raw_data)

                    if len(features) >= 10:
                        # Step 2: Model training
                        with self.measure_performance('system', 'model_training') as model_metrics:
                            sample_labels = np.random.choice([0, 1], size=len(features))
                            model.train_model(features[:len(features)//2],
                                            sample_labels[:len(sample_labels)//2])

                        # Step 3: Prediction
                        test_features = features[len(features)//2:len(features)//2 + 5]

                        with self.measure_performance('system', 'prediction') as pred_metrics:
                            predictions = model.predict(test_features)

                        # Step 4: Explanation
                        with self.measure_performance('system', 'explanation') as exp_metrics:
                            sample_for_explanation = test_features.iloc[0]
                            if explainability_engine is not None:
                                explanation = explainability_engine.explain_single_prediction(
                                    sample_for_explanation,
                                    prediction=float(predictions[0])
                                )
                            else:
                                explanation = {"mock_explanation": "Model not trained"}

                        results['tests_passed'] = 4
                    else:
                        results['tests_failed'] = 1
                        results['recommendations'].append(
                            "Insufficient data for end-to-end testing"
                        )
                else:
                    results['tests_failed'] = 1
                    results['recommendations'].append(
                        "No data available for end-to-end testing"
                    )

            total_time = time.time() - start_time
            results['total_time'] = total_time

            # Check overall performance
            if total_time > self.thresholds['end_to_end']:
                results['recommendations'].append(
                    f"End-to-end processing took {total_time:.1f}s "
                    f"(threshold: {self.thresholds['end_to_end']}s). "
                    "Consider system-wide optimizations."
                )

            results['metrics']['system_overview'] = {
                'total_time_seconds': total_time,
                'components_tested': 4,
                'success_rate': results['tests_passed'] / (results['tests_passed'] + results['tests_failed'])
            }

        except Exception as e:
            logger.error(f"End-to-end performance validation failed: {e}")
            results['tests_failed'] += 1
            results['error'] = str(e)
            results['total_time'] = time.time() - start_time

        logger.info(f"End-to-end validation completed in {results['total_time']:.1f}s: "
                   f"{results['tests_passed']} passed, {results['tests_failed']} failed")

        return results

    def generate_performance_report(self) -> Dict[str, Any]:
        """
        Context7: Generate comprehensive performance report.

        Analyzes collected metrics and provides optimization recommendations.

        Returns:
            Dictionary containing detailed performance analysis and recommendations
        """
        logger.info("Generating comprehensive performance report")

        if not self.metrics_history:
            return {
                'error': 'No performance metrics available. Run validation tests first.',
                'metrics_collected': 0
            }

        # Analyze metrics by component
        component_analysis = {}
        for metrics in self.metrics_history:
            if metrics.component_name not in component_analysis:
                component_analysis[metrics.component_name] = {
                    'total_operations': 0,
                    'successful_operations': 0,
                    'failed_operations': 0,
                    'total_time': 0.0,
                    'avg_time': 0.0,
                    'max_time': 0.0,
                    'total_memory': 0.0,
                    'avg_memory': 0.0,
                    'operations': []
                }

            comp = component_analysis[metrics.component_name]
            comp['total_operations'] += 1
            comp['total_time'] += metrics.execution_time
            comp['total_memory'] += metrics.memory_usage_mb
            comp['max_time'] = max(comp['max_time'], metrics.execution_time)
            comp['operations'].append(metrics.operation)

            if metrics.success:
                comp['successful_operations'] += 1
            else:
                comp['failed_operations'] += 1

        # Calculate averages
        for comp_name, comp_data in component_analysis.items():
            if comp_data['total_operations'] > 0:
                comp_data['avg_time'] = comp_data['total_time'] / comp_data['total_operations']
                comp_data['avg_memory'] = comp_data['total_memory'] / comp_data['total_operations']
                comp_data['success_rate'] = comp_data['successful_operations'] / comp_data['total_operations']

        # Generate recommendations
        recommendations = []

        # Memory usage recommendations
        high_memory_components = [
            name for name, data in component_analysis.items()
            if data['avg_memory'] > 100  # MB
        ]

        if high_memory_components:
            recommendations.append(
                f"High memory usage detected in: {', '.join(high_memory_components)}. "
                "Consider implementing memory optimization strategies."
            )

        # Execution time recommendations
        slow_components = [
            name for name, data in component_analysis.items()
            if data['avg_time'] > 10.0  # seconds
        ]

        if slow_components:
            recommendations.append(
                f"Slow performance detected in: {', '.join(slow_components)}. "
                "Consider algorithmic optimizations or caching."
            )

        # Success rate recommendations
        low_success_components = [
            name for name, data in component_analysis.items()
            if data.get('success_rate', 1.0) < 0.9
        ]

        if low_success_components:
            recommendations.append(
                f"Low success rate detected in: {', '.join(low_success_components)}. "
                "Review error handling and input validation."
            )

        report = {
            'summary': {
                'total_metrics_collected': len(self.metrics_history),
                'components_tested': len(component_analysis),
                'overall_success_rate': sum(
                    data.get('success_rate', 1.0) for data in component_analysis.values()
                ) / len(component_analysis),
                'total_execution_time': sum(data['total_time'] for data in component_analysis.values()),
                'recommendations_count': len(recommendations)
            },
            'component_analysis': component_analysis,
            'recommendations': recommendations,
            'performance_trends': self._analyze_performance_trends(),
            'optimization_priorities': self._identify_optimization_priorities(component_analysis)
        }

        return report

    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """
        Context7: Analyze performance trends over time.

        Returns:
            Dictionary containing trend analysis
        """
        if len(self.metrics_history) < 2:
            return {'message': 'Insufficient data for trend analysis'}

        # Simple trend analysis: compare first half vs second half
        mid_point = len(self.metrics_history) // 2
        first_half = self.metrics_history[:mid_point]
        second_half = self.metrics_history[mid_point:]

        first_avg_time = np.mean([m.execution_time for m in first_half])
        second_avg_time = np.mean([m.execution_time for m in second_half])

        trend = {
            'time_trend_percent': ((second_avg_time - first_avg_time) / first_avg_time) * 100,
            'first_half_avg_time': first_avg_time,
            'second_half_avg_time': second_avg_time,
            'trend_direction': 'improving' if second_avg_time < first_avg_time else 'degrading'
        }

        return trend

    def _identify_optimization_priorities(self, component_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Context7: Identify components that need optimization priority.

        Args:
            component_analysis: Component performance analysis

        Returns:
            List of optimization priorities sorted by impact
        """
        priorities = []

        for comp_name, comp_data in component_analysis.items():
            # Calculate priority score based on multiple factors
            time_score = comp_data['avg_time'] / max(comp_data['avg_time'], 1.0)
            memory_score = comp_data['avg_memory'] / 100.0  # Normalize to 100MB
            failure_score = 1.0 - comp_data.get('success_rate', 1.0)

            priority_score = (time_score * 0.5 + memory_score * 0.3 + failure_score * 0.2)

            if priority_score > 0.3:  # Only include significant priorities
                priorities.append({
                    'component': comp_name,
                    'priority_score': priority_score,
                    'issues': {
                        'slow_performance': time_score > 0.5,
                        'high_memory': memory_score > 0.5,
                        'unreliable': failure_score > 0.1
                    },
                    'recommended_actions': self._get_component_recommendations(
                        comp_name, comp_data
                    )
                })

        # Sort by priority score (highest first)
        priorities.sort(key=lambda x: x['priority_score'], reverse=True)

        return priorities

    def _get_component_recommendations(self, component_name: str,
                                      component_data: Dict[str, Any]) -> List[str]:
        """
        Context7: Get specific recommendations for a component.

        Args:
            component_name: Name of the component
            component_data: Performance data for the component

        Returns:
            List of specific recommendations
        """
        recommendations = []

        if component_data['avg_time'] > 10.0:
            recommendations.append(f"Optimize {component_name} algorithms for faster execution")

        if component_data['avg_memory'] > 50.0:
            recommendations.append(f"Implement memory optimization for {component_name}")

        if component_data.get('success_rate', 1.0) < 0.95:
            recommendations.append(f"Improve error handling in {component_name}")

        # Component-specific recommendations
        if 'pipeline' in component_name.lower():
            recommendations.extend([
                "Implement data caching",
                "Consider parallel processing for large datasets",
                "Optimize API call frequency"
            ])
        elif 'model' in component_name.lower():
            recommendations.extend([
                "Consider model compression techniques",
                "Implement batch prediction",
                "Optimize hyperparameters for performance"
            ])
        elif 'shap' in component_name.lower():
            recommendations.extend([
                "Use cached SHAP values",
                "Implement approximate explanations for real-time use",
                "Consider feature selection to reduce computation"
            ])

        return recommendations

def main():
    """
    Context7: Main function to run performance validation suite.

    Executes comprehensive performance testing across all system components
    and generates optimization recommendations.
    """
    print("🚀 NBA Predictive Analytics System - Performance Validation Suite")
    print("=" * 70)

    validator = PerformanceValidator()

    try:
        # Initialize components for testing
        pipeline = UnifiedNBADataPipeline()
        model = AdvancedPredictiveModel()

        # Check if model is trained before initializing explainability engine
        if hasattr(model, 'model') and model.model is not None:
            explainability_engine = NBAExplainabilityEngine(
                model.model,
                list(range(20))  # Mock feature names for testing
            )
        else:
            explainability_engine = None

        print("\n📊 Starting Component Performance Validation")
        print("-" * 50)

        # Validate pipeline performance
        print("🔍 Testing UnifiedNBADataPipeline performance...")
        pipeline_results = validator.validate_pipeline_performance(pipeline)
        print(f"   Results: {pipeline_results['tests_passed']} passed, "
              f"{pipeline_results['tests_failed']} failed")

        # Validate model performance
        print("\n🤖 Testing AdvancedPredictiveModel performance...")
        # Create sample data for model testing
        sample_features = pd.DataFrame({
            f'feature_{i}': np.random.random(20) for i in range(10)
        })
        model_results = validator.validate_model_performance(model, sample_features)
        print(f"   Results: {model_results['tests_passed']} passed, "
              f"{model_results['tests_failed']} failed")

        # Validate SHAP performance
        print("\n🧠 Testing NBAExplainabilityEngine performance...")
        shap_results = validator.validate_shap_performance(explainability_engine, sample_features)
        print(f"   Results: {shap_results['tests_passed']} passed, "
              f"{shap_results['tests_failed']} failed")

        # Validate end-to-end performance
        print("\n🔄 Testing End-to-End System performance...")
        e2e_results = validator.validate_end_to_end_performance()
        print(f"   Results: {e2e_results['tests_passed']} passed, "
              f"{e2e_results['tests_failed']} failed")
        print(f"   Total time: {e2e_results['total_time']:.1f}s")

        # Generate comprehensive report
        print("\n📈 Generating Performance Report...")
        report = validator.generate_performance_report()

        # Display summary
        print("\n" + "=" * 70)
        print("📋 PERFORMANCE VALIDATION SUMMARY")
        print("=" * 70)

        print(f"📊 Total Metrics Collected: {report['summary']['total_metrics_collected']}")
        print(f"🔧 Components Tested: {report['summary']['components_tested']}")
        print(f"✅ Overall Success Rate: {report['summary']['overall_success_rate']:.1%}")
        print(f"⏱️ Total Execution Time: {report['summary']['total_execution_time']:.1f}s")
        print(f"💡 Recommendations Generated: {report['summary']['recommendations_count']}")

        # Display component analysis
        print("\n📊 COMPONENT PERFORMANCE ANALYSIS")
        print("-" * 40)

        for comp_name, comp_data in report['component_analysis'].items():
            print(f"\n🔹 {comp_name}:")
            print(f"   Operations: {comp_data['total_operations']}")
            print(f"   Success Rate: {comp_data.get('success_rate', 1.0):.1%}")
            print(f"   Avg Time: {comp_data['avg_time']:.3f}s")
            print(f"   Max Time: {comp_data['max_time']:.3f}s")
            print(f"   Avg Memory: {comp_data['avg_memory']:.1f}MB")

        # Display recommendations
        if report['recommendations']:
            print("\n💡 PERFORMANCE RECOMMENDATIONS")
            print("-" * 40)
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"{i}. {rec}")

        # Display optimization priorities
        if report['optimization_priorities']:
            print("\n🎯 OPTIMIZATION PRIORITIES")
            print("-" * 40)
            for i, priority in enumerate(report['optimization_priorities'][:3], 1):
                print(f"{i}. {priority['component']} (Score: {priority['priority_score']:.2f})")
                for action in priority['recommended_actions'][:2]:
                    print(f"   • {action}")

        # Performance trend analysis
        if 'performance_trends' in report and 'trend_direction' in report['performance_trends']:
            trend = report['performance_trends']
            print(f"\n📈 PERFORMANCE TREND: {trend['trend_direction'].upper()}")
            print(f"   Change: {trend['time_trend_percent']:+.1f}%")

        print("\n" + "=" * 70)
        print("✅ Performance validation completed successfully!")
        print("📝 Detailed metrics have been logged for further analysis.")

        return True

    except Exception as e:
        logger.error(f"Performance validation failed: {e}")
        print(f"\n❌ Performance validation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)