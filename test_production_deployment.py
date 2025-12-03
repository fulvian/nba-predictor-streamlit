"""
Test suite for Production Deployment System
Phase 3 Day 12 - Task 3.5.4 Production Deployment Preparation
"""

import pytest
import time
import json
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, patch, MagicMock
import docker

from src.nba_predictor.deployment.production_deployment_system import (
    ProductionDeploymentSystem, DeploymentConfig, DeploymentEnvironment, DeploymentStatus,
    BuildArtifact, DeploymentStep, HealthCheckResult, DeploymentMetrics,
    create_production_deployment_config, deploy_nba_predictor_production
)


class TestDeploymentConfig:
    """Test deployment configuration"""

    def test_deployment_config_creation(self):
        """Test deployment configuration creation"""
        config = DeploymentConfig(
            environment=DeploymentEnvironment.PRODUCTION,
            app_name="test-app",
            version="1.0.0",
            port=8501,
            domain="test.com",
            ssl_enabled=True,
            database_url="test://db",
            context7_compliance=True
        )

        assert config.environment == DeploymentEnvironment.PRODUCTION
        assert config.app_name == "test-app"
        assert config.version == "1.0.0"
        assert config.port == 8501
        assert config.domain == "test.com"
        assert config.ssl_enabled == True
        assert config.context7_compliance == True

    def test_deployment_config_defaults(self):
        """Test deployment configuration defaults"""
        config = DeploymentConfig(
            environment=DeploymentEnvironment.DEVELOPMENT,
            app_name="test",
            version="1.0.0",
            port=8080,
            domain="localhost",
            ssl_enabled=False,
            database_url="sqlite://test.db"
        )

        assert config.redis_url is None
        assert config.monitoring_enabled == True
        assert config.backup_enabled == True
        assert config.auto_scaling == False
        assert config.context7_compliance == True


class TestProductionDeploymentSystem:
    """Test production deployment system"""

    @pytest.fixture
    def deployment_config(self):
        """Create test deployment configuration"""
        return DeploymentConfig(
            environment=DeploymentEnvironment.STAGING,
            app_name="nba-predictor-test",
            version="1.0.0-test",
            port=8501,
            domain="localhost",
            ssl_enabled=False,
            database_url="duckdb:///test.duckdb",
            context7_compliance=True
        )

    @pytest.fixture
    def workspace_dir(self, tmp_path):
        """Create temporary workspace directory"""
        workspace = tmp_path / "workspace"
        workspace.mkdir(exist_ok=True)
        return str(workspace)

    @pytest.fixture
    def deployment_system(self, deployment_config, workspace_dir):
        """Create deployment system instance"""
        with patch('src.nba_predictor.deployment.production_deployment_system.docker.from_env'):
            system = ProductionDeploymentSystem(deployment_config, workspace_dir)
            return system

    def test_deployment_system_initialization(self, deployment_system):
        """Test deployment system initialization"""
        assert deployment_system.config.app_name == "nba-predictor-test"
        assert deployment_system.config.environment == DeploymentEnvironment.STAGING
        assert deployment_system.deployment_id is not None
        assert deployment_system.current_status == DeploymentStatus.PENDING
        assert len(deployment_system.deployment_steps) == 0
        assert len(deployment_system.build_artifacts) == 0
        assert len(deployment_system.health_checks) == 0
        assert deployment_system.deployment_metrics is None

    def test_generate_deployment_id(self, deployment_system):
        """Test deployment ID generation"""
        deployment_id1 = deployment_system._generate_deployment_id()
        deployment_id2 = deployment_system._generate_deployment_id()

        assert deployment_id1 != deployment_id2
        assert deployment_id1.startswith("deploy_")
        assert deployment_id2.startswith("deploy_")
        assert len(deployment_id1.split("_")) == 3  # deploy_timestamp_hash

    def test_create_deployment_plan(self, deployment_system):
        """Test deployment plan creation"""
        plan = deployment_system.create_deployment_plan()

        # Verify plan structure
        required_fields = [
            "deployment_id", "environment", "app_name", "version", "created_at",
            "estimated_duration", "steps", "context7_patterns", "risk_assessment",
            "rollback_plan", "validation_checks"
        ]

        for field in required_fields:
            assert field in plan

        # Verify steps
        assert len(plan["steps"]) == 8
        assert plan["estimated_duration"] == 25
        assert plan["environment"] == "staging"
        assert plan["app_name"] == "nba-predictor-test"

        # Verify Context7 patterns
        expected_patterns = [
            "responsive_design_system",
            "accessibility_features",
            "adaptive_ui_layouts",
            "pwa_features",
            "real_time_updates",
            "intelligent_cache",
            "advanced_ml_operations"
        ]

        assert set(plan["context7_patterns"]) == set(expected_patterns)

        # Verify deployment steps created
        assert len(deployment_system.deployment_steps) == 8

    def test_assess_deployment_risks(self, deployment_system):
        """Test deployment risk assessment"""
        risks = deployment_system._assess_deployment_risks()

        assert "overall_risk" in risks
        assert "risk_factors" in risks
        assert "mitigation_strategies" in risks
        assert len(risks["risk_factors"]) > 0
        assert len(risks["mitigation_strategies"]) > 0

        # Check for Context7 compliance risk if enabled
        if deployment_system.config.context7_compliance:
            context7_risks = [f for f in risks["risk_factors"] if "Context7" in f["factor"]]
            assert len(context7_risks) > 0

    def test_create_rollback_plan(self, deployment_system):
        """Test rollback plan creation"""
        rollback_plan = deployment_system._create_rollback_plan()

        assert "rollback_triggers" in rollback_plan
        assert "rollback_steps" in rollback_plan
        assert "rollback_time_estimate" in rollback_plan
        assert "data_backup_procedure" in rollback_plan

        assert len(rollback_plan["rollback_triggers"]) > 0
        assert len(rollback_plan["rollback_steps"]) > 0
        assert rollback_plan["rollback_time_estimate"] > 0

    def test_get_validation_checks(self, deployment_system):
        """Test validation checks"""
        checks = deployment_system._get_validation_checks()

        assert len(checks) > 0

        # Check for required check types
        check_types = {check["check_type"] for check in checks}
        assert "health_check" in check_types
        assert "context7_check" in check_types

        # Check Context7 pattern checks
        context7_checks = [c for c in checks if c["check_type"] == "context7_check"]
        assert len(context7_checks) > 0

    def test_perform_pre_deployment_checks(self, deployment_system):
        """Test pre-deployment checks"""
        # Mock all check functions to return True
        deployment_system._check_docker = Mock(return_value=True)
        deployment_system._check_disk_space = Mock(return_value=True)
        deployment_system._check_memory = Mock(return_value=True)
        deployment_system._check_database_connection = Mock(return_value=True)
        deployment_system._check_port_availability = Mock(return_value=True)

        result = deployment_system._perform_pre_deployment_checks()
        assert result == True

    def test_perform_pre_deployment_checks_failure(self, deployment_system):
        """Test pre-deployment checks with failure"""
        # Mock one check to return False
        deployment_system._check_docker = Mock(return_value=True)
        deployment_system._check_disk_space = Mock(return_value=False)  # This fails
        deployment_system._check_memory = Mock(return_value=True)
        deployment_system._check_database_connection = Mock(return_value=True)
        deployment_system._check_port_availability = Mock(return_value=True)

        result = deployment_system._perform_pre_deployment_checks()
        assert result == False

    def test_backup_current_version(self, deployment_system, workspace_dir):
        """Test backup current version"""
        # Create some test files to backup
        deploy_dir = Path(workspace_dir) / "deploy"
        deploy_dir.mkdir(exist_ok=True)
        (deploy_dir / "test_file.txt").write_text("test content")

        data_dir = Path(workspace_dir) / "data"
        data_dir.mkdir(exist_ok=True)
        (data_dir / "test_db.duckdb").write_text("fake db content")

        result = deployment_system._backup_current_version()
        assert result == True

        # Check backup directory exists
        backup_dir = Path(workspace_dir) / "backups"
        assert backup_dir.exists()
        assert len(list(backup_dir.glob("backup_*"))) > 0

        # Check backup manifest
        latest_backup = max(backup_dir.glob("backup_*"), key=lambda x: x.stat().st_mtime)
        manifest_path = latest_backup / "manifest.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)
            assert "backup_name" in manifest
            assert "created_at" in manifest
            assert "files_backed_up" in manifest

    def test_create_dockerfile(self, deployment_system):
        """Test Dockerfile creation"""
        dockerfile_path = deployment_system._create_dockerfile()

        assert dockerfile_path.exists()

        with open(dockerfile_path) as f:
            content = f.read()

        # Check for required content
        assert "FROM python:3.11-slim" in content
        assert f"EXPOSE {deployment_system.config.port}" in content
        assert f"STREAMLIT_SERVER_PORT={deployment_system.config.port}" in content
        assert "streamlit run" in content
        assert "betting_workflow_dashboard.py" in content

    def test_build_application(self, deployment_system):
        """Test application build"""
        # Mock Docker client
        mock_image = Mock()
        mock_image.id = "test_image_id"
        mock_image.attrs = {'Size': 12345678}

        deployment_system.docker_client = Mock()
        deployment_system.docker_client.images.build.return_value = (mock_image, [])

        result = deployment_system._build_application()

        assert result == True
        assert len(deployment_system.build_artifacts) == 1
        assert deployment_system.build_artifacts[0].artifact_type == "docker_image"
        assert deployment_system.build_artifacts[0].checksum == "test_image_id"

    def test_build_application_failure(self, deployment_system):
        """Test application build failure"""
        # Mock Docker client to raise exception
        deployment_system.docker_client = Mock()
        deployment_system.docker_client.images.build.side_effect = Exception("Build failed")

        result = deployment_system._build_application()
        assert result == False

    def test_run_integration_tests(self, deployment_system):
        """Test integration tests execution"""
        with patch('subprocess.run') as mock_subprocess:
            # Mock successful test runs
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = "Tests passed"
            mock_subprocess.return_value.stderr = ""

            result = deployment_system._run_integration_tests()

            assert result == True
            assert mock_subprocess.call_count == 3  # E2E, UAT, and performance tests

    def test_run_integration_tests_failure(self, deployment_system):
        """Test integration tests with failure"""
        with patch('subprocess.run') as mock_subprocess:
            # Mock failed test run
            mock_subprocess.return_value.returncode = 1
            mock_subprocess.return_value.stderr = "Tests failed"

            result = deployment_system._run_integration_tests()
            assert result == False

    def test_perform_security_scan(self, deployment_system):
        """Test security scan"""
        with patch('subprocess.run') as mock_subprocess:
            # Mock successful security scan
            mock_subprocess.return_value.returncode = 0

            result = deployment_system._perform_security_scan()
            assert result == True

    def test_deploy_application(self, deployment_system):
        """Test application deployment"""
        # Mock Docker client
        mock_container = Mock()
        mock_container.status = 'running'
        mock_container.id = 'test_container_id'

        deployment_system.docker_client = Mock()
        deployment_system.docker_client.containers.get.side_effect = docker.errors.NotFound("No container")
        deployment_system.docker_client.containers.run.return_value = mock_container
        deployment_system.docker_client.containers.get = Mock(side_effect=docker.errors.NotFound("No container"))

        result = deployment_system._deploy_application()

        assert result == True
        deployment_system.docker_client.containers.run.assert_called_once()

    def test_deploy_application_failure(self, deployment_system):
        """Test application deployment failure"""
        # Mock Docker client to raise exception
        deployment_system.docker_client = Mock()
        deployment_system.docker_client.containers.get.side_effect = docker.errors.NotFound("No container")
        deployment_system.docker_client.containers.run.side_effect = Exception("Deployment failed")

        result = deployment_system._deploy_application()
        assert result == False

    def test_perform_health_checks(self, deployment_system):
        """Test health checks"""
        with patch('requests.get') as mock_requests:
            # Mock successful health checks
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.elapsed.total_seconds.return_value = 0.5
            mock_response.content = b"healthy"
            mock_response.headers = {}
            mock_requests.return_value = mock_response

            result = deployment_system._perform_health_checks()

            assert result == True
            assert len(deployment_system.health_checks) > 0

            # Check health check results
            for check in deployment_system.health_checks:
                assert check.status.value == "healthy"
                assert check.response_time > 0

    def test_perform_health_checks_failure(self, deployment_system):
        """Test health checks with failures"""
        with patch('requests.get') as mock_requests:
            # Mock failed health check
            mock_requests.side_effect = Exception("Connection failed")

            result = deployment_system._perform_health_checks()

            assert result == False
            assert len(deployment_system.health_checks) > 0

            # Check health check results
            for check in deployment_system.health_checks:
                assert check.status.value == "unhealthy"

    def test_validate_context7_compliance(self, deployment_system):
        """Test Context7 compliance validation"""
        with patch('requests.get') as mock_requests:
            # Mock successful HTTP responses
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '<html><head><meta name="viewport" content="width=device-width"></head><body></body></html>'
            mock_response.headers = {'cache-control': 'max-age=3600'}
            mock_response.elapsed.total_seconds.return_value = 0.1
            mock_requests.return_value = mock_response

            result = deployment_system._validate_context7_compliance()

            assert result == True
            assert len(deployment_system.context7_compliance_scores) == 7  # All 7 patterns
            assert len(deployment_system.context7_validations) == 7

    def test_validate_individual_context7_patterns(self, deployment_system):
        """Test individual Context7 pattern validation"""
        base_url = "http://localhost:8501"

        with patch('requests.get') as mock_requests:
            # Mock successful response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.text = '<html><head><meta name="viewport" content="width=device-width"></head><body></body></html>'
            mock_response.headers = {'cache-control': 'max-age=3600'}
            mock_requests.return_value = mock_response

            # Test each pattern validation
            patterns = [
                ("responsive_design_system", deployment_system._validate_responsive_design),
                ("accessibility_features", deployment_system._validate_accessibility),
                ("adaptive_ui_layouts", deployment_system._validate_adaptive_ui),
                ("pwa_features", deployment_system._validate_pwa_features),
                ("real_time_updates", deployment_system._validate_real_time_updates),
                ("intelligent_cache", deployment_system._validate_caching),
                ("advanced_ml_operations", deployment_system._validate_ml_operations)
            ]

            for pattern_name, validation_function in patterns:
                score, result = validation_function(base_url)
                assert 0 <= score <= 1, f"Pattern {pattern_name} should return score between 0 and 1"
                assert isinstance(result, dict), f"Pattern {pattern_name} should return dict result"

    def test_calculate_context7_compliance_score(self, deployment_system):
        """Test Context7 compliance score calculation"""
        # Set some test scores
        deployment_system.context7_compliance_scores = {
            "responsive_design_system": 0.9,
            "accessibility_features": 0.8,
            "adaptive_ui_layouts": 0.7,
            "pwa_features": 0.6,
            "real_time_updates": 0.85,
            "intelligent_cache": 0.75,
            "advanced_ml_operations": 0.95
        }

        score = deployment_system._calculate_context7_compliance_score()
        expected_score = sum(deployment_system.context7_compliance_scores.values()) / len(deployment_system.context7_compliance_scores)

        assert score == expected_score
        assert 0 <= score <= 1

    def test_execute_step_success(self, deployment_system):
        """Test step execution with success"""
        # Create test step
        step = DeploymentStep(
            step_id="test_step",
            name="Test Step",
            description="Test step description",
            command="test command",
            expected_duration=1
        )
        deployment_system.deployment_steps.append(step)

        # Mock step function to return True
        step_function = Mock(return_value=True)

        result = deployment_system._execute_step("test_step", step_function)

        assert result == True
        assert step.status == DeploymentStatus.COMPLETED
        assert step.start_time is not None
        assert step.end_time is not None

    def test_execute_step_with_retry(self, deployment_system):
        """Test step execution with retry logic"""
        # Create test step
        step = DeploymentStep(
            step_id="test_step_retry",
            name="Test Step Retry",
            description="Test step with retry",
            command="test command",
            expected_duration=1,
            max_retries=2
        )
        deployment_system.deployment_steps.append(step)

        # Mock step function to fail first time, succeed second time
        step_function = Mock(side_effect=[Exception("First failure"), True])

        with patch('time.sleep'):  # Mock sleep to speed up test
            result = deployment_system._execute_step("test_step_retry", step_function)

        assert result == True
        assert step.status == DeploymentStatus.COMPLETED
        assert step.retry_count == 1

    def test_execute_step_failure(self, deployment_system):
        """Test step execution with failure"""
        # Create test step
        step = DeploymentStep(
            step_id="test_step_fail",
            name="Test Step Fail",
            description="Test step that fails",
            command="test command",
            expected_duration=1,
            max_retries=1
        )
        deployment_system.deployment_steps.append(step)

        # Mock step function to always fail
        step_function = Mock(side_effect=Exception("Always fails"))

        with patch('time.sleep'):  # Mock sleep to speed up test
            result = deployment_system._execute_step("test_step_fail", step_function)

        assert result == False
        assert step.status == DeploymentStatus.FAILED
        assert step.retry_count == 1
        assert step.error_message is not None

    def test_generate_deployment_report(self, deployment_system):
        """Test deployment report generation"""
        # Set up deployment metrics
        deployment_system.deployment_metrics = DeploymentMetrics(
            deployment_id="test_deployment",
            environment=DeploymentEnvironment.STAGING,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            duration=300.0,
            status=DeploymentStatus.COMPLETED,
            build_time=60.0,
            test_time=120.0,
            deploy_time=30.0,
            rollback_count=0,
            context7_compliance_score=0.85
        )

        # Add some health checks
        health_check = HealthCheckResult(
            service_name="Test Service",
            status=HealthCheckStatus.HEALTHY,
            response_time=0.5,
            timestamp=datetime.now(timezone.utc),
            details={"test": "data"},
            endpoint="http://localhost:8501/health"
        )
        deployment_system.health_checks.append(health_check)

        # Set Context7 compliance scores
        deployment_system.context7_compliance_scores = {
            "responsive_design_system": 0.9,
            "accessibility_features": 0.8
        }

        report = deployment_system.generate_deployment_report()

        # Verify report structure
        required_sections = [
            "deployment_summary", "deployment_steps", "build_artifacts",
            "health_checks", "context7_compliance", "rollback_info",
            "recommendations", "context7_patterns_validated"
        ]

        for section in required_sections:
            assert section in report

        # Verify summary
        summary = report["deployment_summary"]
        assert summary["deployment_id"] == "test_deployment"
        assert summary["environment"] == "staging"
        assert summary["status"] == "completed"
        assert summary["duration_seconds"] == 300.0

        # Verify Context7 compliance section
        context7 = report["context7_compliance"]
        assert context7["overall_score"] == 0.85
        assert len(context7["pattern_scores"]) == 2
        assert "responsive_design_system" in context7["pattern_scores"]

    def test_generate_deployment_recommendations(self, deployment_system):
        """Test deployment recommendations generation"""
        # Set up deployment metrics with poor performance
        deployment_system.deployment_metrics = DeploymentMetrics(
            deployment_id="test_deployment",
            environment=DeploymentEnvironment.STAGING,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            duration=300.0,
            status=DeploymentStatus.COMPLETED,
            build_time=60.0,
            test_time=120.0,
            deploy_time=30.0,
            rollback_count=0,
            context7_compliance_score=0.5  # Low compliance score
        )

        # Add slow health checks
        slow_health_check = HealthCheckResult(
            service_name="Slow Service",
            status=HealthCheckStatus.HEALTHY,
            response_time=3.0,  # Slow response time
            timestamp=datetime.now(timezone.utc),
            details={"test": "data"},
            endpoint="http://localhost:8501/slow"
        )
        deployment_system.health_checks.append(slow_health_check)

        recommendations = deployment_system._generate_deployment_recommendations()

        assert len(recommendations) > 0
        assert any("Context7 compliance" in rec for rec in recommendations)
        assert any("optimizing application response times" in rec for rec in recommendations)

    def test_find_latest_backup(self, deployment_system, workspace_dir):
        """Test finding latest backup"""
        backup_dir = Path(workspace_dir) / "backups"
        backup_dir.mkdir(exist_ok=True)

        # Create some test backup directories
        old_backup = backup_dir / "backup_20230101_120000"
        new_backup = backup_dir / "backup_20230102_120000"

        old_backup.mkdir(exist_ok=True)
        time.sleep(0.1)  # Ensure timestamp difference
        new_backup.mkdir(exist_ok=True)

        latest = deployment_system._find_latest_backup()

        assert latest == new_backup.name

    def test_cleanup(self, deployment_system):
        """Test cleanup functionality"""
        # Mock cleanup operations
        with patch.object(deployment_system.logger, 'info') as mock_logger:
            deployment_system.cleanup()
            mock_logger.assert_called_with("Cleaning up deployment resources...")


class TestConvenienceFunctions:
    """Test convenience functions"""

    def test_create_production_deployment_config_defaults(self):
        """Test production deployment config creation with defaults"""
        config = create_production_deployment_config()

        assert config.environment == DeploymentEnvironment.PRODUCTION
        assert config.app_name == "nba-predictor"
        assert config.version == "1.0.0"
        assert config.port == 8501
        assert config.domain == "localhost"
        assert config.ssl_enabled == False
        assert config.database_url == "duckdb:///app/data/nba_data.duckdb"
        assert config.redis_url is None
        assert config.monitoring_enabled == True
        assert config.backup_enabled == True
        assert config.auto_scaling == False
        assert config.context7_compliance == True

    def test_create_production_deployment_config_custom(self):
        """Test production deployment config creation with custom values"""
        config = create_production_deployment_config(
            app_name="custom-app",
            version="2.0.0",
            environment=DeploymentEnvironment.STAGING,
            port=8080,
            domain="staging.example.com"
        )

        assert config.environment == DeploymentEnvironment.STAGING
        assert config.app_name == "custom-app"
        assert config.version == "2.0.0"
        assert config.port == 8080
        assert config.domain == "staging.example.com"

    @patch('src.nba_predictor.deployment.production_deployment_system.ProductionDeploymentSystem')
    def test_deploy_nba_predictor_production_success(self, mock_deployment_system):
        """Test NBA Predictor production deployment success"""
        # Mock successful deployment
        mock_metrics = DeploymentMetrics(
            deployment_id="test_deployment",
            environment=DeploymentEnvironment.PRODUCTION,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            duration=300.0,
            status=DeploymentStatus.COMPLETED,
            build_time=60.0,
            test_time=120.0,
            deploy_time=30.0,
            rollback_count=0,
            context7_compliance_score=0.9
        )

        mock_system_instance = Mock()
        mock_system_instance.create_deployment_plan.return_value = {"test": "plan"}
        mock_system_instance.execute_deployment.return_value = mock_metrics
        mock_system_instance.generate_deployment_report.return_value = {"test": "report"}
        mock_system_instance.deployment_id = "test_deployment"
        mock_deployment_system.return_value = mock_system_instance

        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.write = Mock()

            metrics = deploy_nba_predictor_production(
                app_name="test-app",
                version="1.0.0",
                workspace_dir="/tmp/test"
            )

            assert metrics == mock_metrics
            mock_system_instance.create_deployment_plan.assert_called_once()
            mock_system_instance.execute_deployment.assert_called_once()
            mock_system_instance.generate_deployment_report.assert_called_once()
            mock_system_instance.cleanup.assert_called_once()

    @patch('src.nba_predictor.deployment.production_deployment_system.ProductionDeploymentSystem')
    def test_deploy_nba_predictor_production_failure(self, mock_deployment_system):
        """Test NBA Predictor production deployment failure"""
        # Mock failed deployment
        mock_system_instance = Mock()
        mock_system_instance.create_deployment_plan.return_value = {"test": "plan"}
        mock_system_instance.execute_deployment.side_effect = Exception("Deployment failed")
        mock_system_instance.deployment_id = "test_deployment"
        mock_deployment_system.return_value = mock_system_instance

        with pytest.raises(Exception, match="Deployment failed"):
            deploy_nba_predictor_production(
                app_name="test-app",
                version="1.0.0",
                workspace_dir="/tmp/test"
            )

        mock_system_instance.cleanup.assert_called_once()


class TestEnvironmentAndStatusEnums:
    """Test environment and status enums"""

    def test_deployment_environment_enum(self):
        """Test DeploymentEnvironment enum values"""
        environments = [env.value for env in DeploymentEnvironment]
        expected = ["development", "staging", "production"]

        assert set(environments) == set(expected)

    def test_deployment_status_enum(self):
        """Test DeploymentStatus enum values"""
        statuses = [status.value for status in DeploymentStatus]
        expected = ["pending", "building", "testing", "deploying", "completed", "failed", "rolled_back"]

        assert set(statuses) == set(expected)

    def test_health_check_status_enum(self):
        """Test HealthCheckStatus enum values"""
        statuses = [status.value for status in HealthCheckStatus]
        expected = ["healthy", "unhealthy", "degraded", "unknown"]

        assert set(statuses) == set(expected)


if __name__ == "__main__":
    # Run specific tests
    pytest.main([__file__, "-v"])