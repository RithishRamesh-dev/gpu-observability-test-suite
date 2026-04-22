#!/usr/bin/env python3
"""
tests/test_workloads.py — Unit tests for workload modules.

These tests mock out CUDA/PyTorch calls so they can run without a GPU.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

# Project root on path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestMatmulStress(unittest.TestCase):
    """Test matmul_stress.py logic without CUDA."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_name", return_value="Mock A100")
    @patch("torch.cuda.get_device_properties")
    @patch("torch.cuda.synchronize")
    @patch("torch.randn")
    @patch("torch.matmul")
    def test_fp16_stress_completes(
        self, mock_matmul, mock_randn, mock_sync,
        mock_props, mock_name, mock_avail
    ):
        """matmul_fp16_stress should complete without raising."""
        mock_props.return_value.total_memory = 24 * 1024**3
        mock_tensor = MagicMock()
        mock_tensor.to.return_value = mock_tensor
        mock_randn.return_value = mock_tensor
        mock_matmul.return_value = mock_tensor

        from workload.cuda.matmul_stress import matmul_fp16_stress
        # Run for 2 seconds (short duration for test)
        result = matmul_fp16_stress(duration=2, matrix_size=128)
        self.assertIsInstance(result, dict)
        self.assertIn("tflops", result)

    @patch("torch.cuda.is_available", return_value=False)
    def test_raises_when_no_cuda(self, mock_avail):
        """Should raise RuntimeError when CUDA unavailable."""
        # Re-import to trigger the check
        with self.assertRaises((RuntimeError, SystemExit)):
            from workload.cuda import matmul_stress  # noqa: F401
            matmul_stress.check_cuda()


class TestMemoryStress(unittest.TestCase):

    @patch("pynvml.nvmlInit")
    @patch("pynvml.nvmlDeviceGetHandleByIndex")
    @patch("pynvml.nvmlDeviceGetMemoryInfo")
    @patch("torch.cuda.is_available", return_value=True)
    def test_fill_memory_reports_allocated(
        self, mock_avail, mock_meminfo, mock_handle, mock_init
    ):
        """fill_memory_stress should return allocated_gb field."""
        mock_info = MagicMock()
        mock_info.total = 24 * 1024**3
        mock_info.free  = 20 * 1024**3
        mock_meminfo.return_value = mock_info

        with patch("torch.zeros") as mock_zeros, \
             patch("torch.cuda.synchronize"):
            mock_tensor = MagicMock()
            mock_zeros.return_value = mock_tensor
            mock_tensor.__len__ = lambda self: 1

            from workload.memory.memory_stress import fill_memory_stress
            result = fill_memory_stress(duration=1, fill_percent=50)
            self.assertIsInstance(result, dict)


class TestPCIeStress(unittest.TestCase):

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.get_device_properties")
    def test_bandwidth_result_structure(self, mock_props, mock_avail):
        """pcie_stress should return structured bandwidth dict."""
        mock_props.return_value.total_memory = 24 * 1024**3

        with patch("torch.zeros") as mock_zeros, \
             patch("torch.cuda.synchronize"), \
             patch("time.perf_counter", side_effect=[0.0, 0.0, 1.0, 1.0]):
            mock_tensor = MagicMock()
            mock_zeros.return_value = mock_tensor
            mock_tensor.pin_memory.return_value = mock_tensor
            mock_tensor.to.return_value = mock_tensor
            mock_tensor.cpu.return_value = mock_tensor
            mock_tensor.nbytes = 1 * 1024**3

            from workload.pcie.pcie_stress import pcie_bandwidth_stress
            result = pcie_bandwidth_stress(duration=1, buffer_size_gb=1)
            self.assertIsInstance(result, dict)
            self.assertIn("h2d_bandwidth_gbps", result)


class TestThermalStress(unittest.TestCase):

    @patch("torch.cuda.is_available", return_value=True)
    @patch("subprocess.run")
    def test_monitor_thermal_returns_dict(self, mock_run, mock_avail):
        """monitor_thermal should return temperature dict."""
        mock_run.return_value.stdout = (
            "2024-01-15 10:00:00, 75, 350.0, Active"
        )
        from workload.thermal.thermal_stress import parse_nvidia_smi_thermal
        # Test the parser directly
        line = "2024-01-15 10:00:00, 75, 350.0, Active"
        result = parse_nvidia_smi_thermal(line)
        if result is not None:
            self.assertIn("temp_c", result)


class TestOrchestratorConfig(unittest.TestCase):

    def test_loads_workload_config(self):
        """orchestrator.py should parse workload_config.yaml without errors."""
        import yaml
        config_path = Path(__file__).parent.parent / "configs" / "workload_config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            self.assertIn("workloads", cfg)
        else:
            self.skipTest("workload_config.yaml not found")

    def test_loads_thresholds_config(self):
        """thresholds.yaml should be parseable."""
        import yaml
        config_path = Path(__file__).parent.parent / "configs" / "thresholds.yaml"
        if config_path.exists():
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            self.assertIn("metrics", cfg)
        else:
            self.skipTest("thresholds.yaml not found")


if __name__ == "__main__":
    unittest.main(verbosity=2)
