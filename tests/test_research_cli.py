#!/usr/bin/env python3
"""
🧪 Research CLI Integration Tests
Test suite for the research prediction CLI interface.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import subprocess
import sys
import json


class TestResearchCLI(unittest.TestCase):
    """Test cases for research prediction CLI."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.data_path = Path(self.temp_dir) / "data"
        self.models_path = Path(self.temp_dir) / "models"

        # Create directories
        self.data_path.mkdir(parents=True)
        self.models_path.mkdir(parents=True)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)

    def run_cli_command(self, command_args):
        """Run CLI command and return result."""
        cmd = [
            sys.executable, "main_research_prediction.py",
            f"--data-path={self.data_path}",
            f"--models-path={self.models_path}"
        ] + command_args

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        return result

    def test_cli_help(self):
        """Test CLI help command."""
        result = self.run_cli_command(["--help"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("NBA Research Prediction System", result.stdout)
        self.assertIn("train", result.stdout)
        self.assertIn("predict", result.stdout)

    def test_train_command(self):
        """Test CLI train command."""
        result = self.run_cli_command(["train"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("Training Complete", result.stdout)
        self.assertIn("Model trained successfully", result.stdout)

        # Check if model file was created
        model_files = list(self.models_path.glob("*.pkl"))
        self.assertGreater(len(model_files), 0)

    def test_predict_command(self):
        """Test CLI predict command."""
        # First train a model
        self.run_cli_command(["train"])

        # Find the trained model
        model_files = list(self.models_path.glob("*.pkl"))
        self.assertGreater(len(model_files), 0)
        model_file = model_files[0].name

        # Test prediction
        result = self.run_cli_command([
            "--model-file", model_file,
            "predict",
            "--team1", "Boston Celtics",
            "--team2", "Los Angeles Lakers",
            "--line", "225.5"
        ])

        self.assertEqual(result.returncode, 0)
        self.assertIn("NBA Game Prediction", result.stdout)
        self.assertIn("Prediction:", result.stdout)
        self.assertIn("Recommendation:", result.stdout)

    def test_evaluate_command(self):
        """Test CLI evaluate command."""
        # First train a model
        self.run_cli_command(["train"])

        # Find the trained model
        model_files = list(self.models_path.glob("*.pkl"))
        self.assertGreater(len(model_files), 0)
        model_file = model_files[0].name

        # Test evaluation
        result = self.run_cli_command([
            "--model-file", model_file,
            "evaluate"
        ])

        self.assertEqual(result.returncode, 0)
        self.assertIn("Model Evaluation", result.stdout)
        self.assertIn("Model Type:", result.stdout)
        self.assertIn("Performance Metrics:", result.stdout)

    def test_info_command(self):
        """Test CLI info command."""
        # First train a model
        self.run_cli_command(["train"])

        # Find the trained model
        model_files = list(self.models_path.glob("*.pkl"))
        self.assertGreater(len(model_files), 0)
        model_file = model_files[0].name

        # Test info command
        result = self.run_cli_command([
            "--model-file", model_file,
            "info"
        ])

        self.assertEqual(result.returncode, 0)

        # Should be valid JSON
        try:
            info_data = json.loads(result.stdout)
            self.assertIn("model_type", info_data)
            self.assertIn("is_trained", info_data)
        except json.JSONDecodeError:
            self.fail("Info command should return valid JSON")

    def test_lightgbm_only_mode(self):
        """Test CLI with LightGBM only mode."""
        result = self.run_cli_command([
            "--lightgbm-only",
            "train"
        ])

        self.assertEqual(result.returncode, 0)
        self.assertIn("Training Complete", result.stdout)

    def test_predict_without_model(self):
        """Test prediction command without trained model."""
        result = self.run_cli_command([
            "predict",
            "--team1", "Boston Celtics",
            "--team2", "Los Angeles Lakers",
            "--line", "225.5"
        ])

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Model must be trained", result.stdout)

    def test_invalid_command(self):
        """Test CLI with invalid command."""
        result = self.run_cli_command(["invalid_command"])
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()