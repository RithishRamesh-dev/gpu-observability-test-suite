"""
conftest.py — Shared pytest fixtures for the GPU observability test suite.
"""

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


@pytest.fixture
def synthetic_telemetry_path(tmp_dir):
    """Write a minimal synthetic telemetry JSON and return its path."""
    records = []
    for i in range(60):
        phase = "idle" if i < 30 else "load"
        util  = 2.0 if phase == "idle" else 85.0
        temp  = 35.0 if phase == "idle" else 72.0
        power = 25.0 if phase == "idle" else 340.0
        records.append({
            "timestamp": f"2024-01-15T10:{i//60:02d}:{i%60:02d}+00:00",
            "gpu_utilization":  util,
            "sm_occupancy":     util * 0.9,
            "tensor_core_util": util * 0.85 if phase == "load" else 0.5,
            "fp_activity":      util * 0.8,
            "memory_utilization": util * 0.75,
            "memory_used_mb":   18000.0 if phase == "load" else 512.0,
            "sm_clock_mhz":     1700.0 if phase == "load" else 210.0,
            "gpu_temp_c":       temp,
            "memory_temp_c":    temp - 7.0,
            "power_draw_w":     power,
            "power_limit_w":    400.0,
            "pcie_tx_mb":       8000.0 if phase == "load" else 50.0,
            "pcie_rx_mb":       7500.0 if phase == "load" else 40.0,
            "ecc_sbe_volatile": 0,
            "ecc_dbe_volatile": 0,
            "ecc_sbe_aggregate": 0,
            "ecc_dbe_aggregate": 0,
            "retired_pages_sbe": 0,
            "retired_pages_dbe": 0,
            "pcie_replay_counter": 0,
            "power_throttle":   False,
            "thermal_throttle": False,
        })

    data = {
        "records": records,
        "annotations": [
            {"event": "start", "workload": "idle_baseline", "timestamp": 1705312200.0},
            {"event": "end",   "workload": "idle_baseline", "timestamp": 1705312230.0},
            {"event": "start", "workload": "cuda_matmul",   "timestamp": 1705315800.0},
            {"event": "end",   "workload": "cuda_matmul",   "timestamp": 1705315830.0},
        ]
    }

    path = Path(tmp_dir) / "telemetry.json"
    path.write_text(json.dumps(data, indent=2))
    return str(path)


@pytest.fixture
def thresholds_path(tmp_dir):
    """Write a minimal thresholds config and return its path."""
    thresholds = {
        "metrics": [
            {
                "metric": "gpu_occupancy",
                "column": "gpu_utilization",
                "validation_type": "utilization",
                "thresholds": {"idle_max": 10.0, "load_min": 60.0},
                "workload": "cuda_matmul",
            },
            {
                "metric": "ecc_sbe_volatile",
                "column": "ecc_sbe_volatile",
                "validation_type": "ecc_counter",
                "thresholds": {"max_allowed": 0, "monotonic": True},
                "workload": "cuda_matmul",
            },
            {
                "metric": "gpu_temp",
                "column": "gpu_temp_c",
                "validation_type": "threshold_range",
                "thresholds": {"min": 30.0, "max": 87.0},
                "workload": "cuda_matmul",
            },
            {
                "metric": "power_usage",
                "column": "power_draw_w",
                "validation_type": "power",
                "thresholds": {
                    "min_tdp_fraction_under_load": 0.70,
                    "power_limit_column": "power_limit_w",
                },
                "workload": "cuda_matmul",
            },
            {
                "metric": "pcie_tx",
                "column": "pcie_tx_mb",
                "validation_type": "throughput",
                "thresholds": {"min_under_load": 5000.0},
                "workload": "cuda_matmul",
            },
            {
                "metric": "gpu_memory",
                "column": "memory_used_mb",
                "validation_type": "memory",
                "thresholds": {"min_percent_under_load": 70.0, "total_memory_mb": 24576.0},
                "workload": "cuda_matmul",
            },
            {
                "metric": "sm_clock",
                "column": "sm_clock_mhz",
                "validation_type": "clock",
                "thresholds": {"min_percent_of_max_under_load": 80.0, "max_clock_mhz": 1980.0},
                "workload": "cuda_matmul",
            },
            {
                "metric": "temp_power_dynamics",
                "column_x": "power_draw_w",
                "column_y": "gpu_temp_c",
                "validation_type": "time_series_correlation",
                "thresholds": {"min_correlation": 0.7},
                "workload": "cuda_matmul",
            },
        ]
    }
    path = Path(tmp_dir) / "thresholds.json"
    path.write_text(json.dumps(thresholds, indent=2))
    return str(path)
