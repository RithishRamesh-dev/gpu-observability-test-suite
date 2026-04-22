#!/usr/bin/env python3
"""
tests/test_validators.py — Unit tests for the GPU observability validation framework.

Run with:
    pytest tests/test_validators.py -v

No GPU hardware required — all tests use synthetic telemetry data.
"""

import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from validators.run_validators import (
    TelemetryData,
    validate_utilization,
    validate_ecc_counter,
    validate_rate_of_change,
    validate_threshold_range,
    validate_power,
    validate_throttle_flag,
    validate_throughput,
    validate_memory,
    validate_clock,
    validate_time_series_correlation,
    run_all_validators,
)


# ---------------------------------------------------------------------------
# Helpers to build synthetic telemetry JSON files
# ---------------------------------------------------------------------------

def _make_telemetry(records: list[dict], annotations: list[dict] | None = None) -> dict:
    return {
        "records": records,
        "annotations": annotations or [],
    }


def _write_telemetry(tmp_dir, data: dict) -> str:
    p = Path(tmp_dir) / "telemetry.json"
    p.write_text(json.dumps(data))
    return str(p)


def _idle_records(n=30, **overrides) -> list[dict]:
    """Generate n idle-state records (low GPU load)."""
    base = {
        "timestamp": None,
        "gpu_utilization": 2.0,
        "sm_occupancy": 1.5,
        "tensor_core_util": 0.5,
        "fp_activity": 1.0,
        "memory_utilization": 3.0,
        "memory_used_mb": 512.0,
        "sm_clock_mhz": 210.0,
        "gpu_temp_c": 35.0,
        "memory_temp_c": 30.0,
        "power_draw_w": 25.0,
        "power_limit_w": 400.0,
        "pcie_tx_mb": 100.0,
        "pcie_rx_mb": 80.0,
        "ecc_sbe_volatile": 0,
        "ecc_dbe_volatile": 0,
        "ecc_sbe_aggregate": 0,
        "ecc_dbe_aggregate": 0,
        "retired_pages_sbe": 0,
        "retired_pages_dbe": 0,
        "pcie_replay_counter": 0,
        "power_throttle": False,
        "thermal_throttle": False,
    }
    base.update(overrides)
    records = []
    for i in range(n):
        r = dict(base)
        r["timestamp"] = f"2024-01-15T10:{i//60:02d}:{i%60:02d}+00:00"
        records.append(r)
    return records


def _load_records(n=30, **overrides) -> list[dict]:
    """Generate n loaded-state records (high GPU load)."""
    base = {
        "timestamp": None,
        "gpu_utilization": 85.0,
        "sm_occupancy": 78.0,
        "tensor_core_util": 72.0,
        "fp_activity": 80.0,
        "memory_utilization": 75.0,
        "memory_used_mb": 18000.0,
        "sm_clock_mhz": 1700.0,
        "gpu_temp_c": 72.0,
        "memory_temp_c": 65.0,
        "power_draw_w": 340.0,
        "power_limit_w": 400.0,
        "pcie_tx_mb": 8000.0,
        "pcie_rx_mb": 7500.0,
        "ecc_sbe_volatile": 0,
        "ecc_dbe_volatile": 0,
        "ecc_sbe_aggregate": 0,
        "ecc_dbe_aggregate": 0,
        "retired_pages_sbe": 0,
        "retired_pages_dbe": 0,
        "pcie_replay_counter": 0,
        "power_throttle": False,
        "thermal_throttle": False,
    }
    base.update(overrides)
    records = []
    for i in range(n):
        r = dict(base)
        r["timestamp"] = f"2024-01-15T11:{i//60:02d}:{i%60:02d}+00:00"
        records.append(r)
    return records


def _make_annotations(workload: str = "cuda_matmul", idle_start_offset: int = 0) -> list[dict]:
    return [
        {"event": "start", "workload": "idle_baseline",
         "timestamp": 1705312200.0 + idle_start_offset},
        {"event": "end",   "workload": "idle_baseline",
         "timestamp": 1705312200.0 + idle_start_offset + 30},
        {"event": "start", "workload": workload,
         "timestamp": 1705315800.0},
        {"event": "end",   "workload": workload,
         "timestamp": 1705315800.0 + 30},
    ]


# ---------------------------------------------------------------------------
# Threshold config helpers
# ---------------------------------------------------------------------------

def _util_threshold(metric: str = "gpu_occupancy", idle_max=10.0, load_min=60.0) -> dict:
    return {
        "metric": metric,
        "column": "gpu_utilization",
        "validation_type": "utilization",
        "thresholds": {"idle_max": idle_max, "load_min": load_min},
        "workload": "cuda_matmul",
    }


def _ecc_threshold(metric: str = "ecc_sbe_volatile", max_allowed: int = 0) -> dict:
    return {
        "metric": metric,
        "column": "ecc_sbe_volatile",
        "validation_type": "ecc_counter",
        "thresholds": {"max_allowed": max_allowed, "monotonic": True},
        "workload": "cuda_matmul",
    }


def _range_threshold(metric: str = "gpu_temp", min_val=30.0, max_val=87.0) -> dict:
    return {
        "metric": metric,
        "column": "gpu_temp_c",
        "validation_type": "threshold_range",
        "thresholds": {"min": min_val, "max": max_val},
        "workload": "cuda_matmul",
    }


def _power_threshold() -> dict:
    return {
        "metric": "power_usage",
        "column": "power_draw_w",
        "validation_type": "power",
        "thresholds": {"min_tdp_fraction_under_load": 0.70, "power_limit_column": "power_limit_w"},
        "workload": "cuda_matmul",
    }


def _throughput_threshold(min_val: float = 5000.0) -> dict:
    return {
        "metric": "pcie_tx",
        "column": "pcie_tx_mb",
        "validation_type": "throughput",
        "thresholds": {"min_under_load": min_val},
        "workload": "cuda_matmul",
    }


def _memory_threshold(min_pct: float = 70.0, total_mb: float = 24576.0) -> dict:
    return {
        "metric": "gpu_memory",
        "column": "memory_used_mb",
        "validation_type": "memory",
        "thresholds": {"min_percent_under_load": min_pct, "total_memory_mb": total_mb},
        "workload": "cuda_matmul",
    }


def _clock_threshold(min_pct: float = 80.0, max_clock: float = 1980.0) -> dict:
    return {
        "metric": "sm_clock",
        "column": "sm_clock_mhz",
        "validation_type": "clock",
        "thresholds": {"min_percent_of_max_under_load": min_pct, "max_clock_mhz": max_clock},
        "workload": "cuda_matmul",
    }


def _throttle_threshold(col: str = "power_throttle") -> dict:
    return {
        "metric": col,
        "column": col,
        "validation_type": "throttle_flag",
        "thresholds": {"expected_false_at_idle": True},
        "workload": "cuda_matmul",
    }


def _roc_threshold(max_rate: float = 0.5) -> dict:
    return {
        "metric": "ecc_sbe_volatile_roc",
        "column": "ecc_sbe_volatile",
        "validation_type": "rate_of_change",
        "thresholds": {"max_rate_per_second": max_rate},
        "workload": "cuda_matmul",
    }


def _corr_threshold(min_r: float = 0.7) -> dict:
    return {
        "metric": "temp_power_corr",
        "column_x": "power_draw_w",
        "column_y": "gpu_temp_c",
        "validation_type": "time_series_correlation",
        "thresholds": {"min_correlation": min_r},
        "workload": "cuda_matmul",
    }


# ============================================================================
# Test classes
# ============================================================================

class TestTelemetryData(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        idle = _idle_records(30)
        load = _load_records(30)
        ann  = _make_annotations()
        data = _make_telemetry(idle + load, ann)
        self.path = _write_telemetry(self.tmp, data)
        self.tel  = TelemetryData(self.path)

    def test_loads_records(self):
        self.assertGreater(len(self.tel.records), 0)

    def test_annotations_parsed(self):
        anns = self.tel.annotations
        self.assertIsInstance(anns, list)

    def test_get_idle_values_returns_series(self):
        vals = self.tel.get_idle_values("gpu_utilization")
        self.assertIsNotNone(vals)

    def test_get_values_in_window(self):
        vals = self.tel.get_values_in_window("gpu_utilization", 1705315800.0, 1705315830.0)
        self.assertIsNotNone(vals)


# ---------------------------------------------------------------------------

class TestValidateUtilization(unittest.TestCase):

    def _telemetry(self, tmp, idle_util=2.0, load_util=85.0):
        idle = _idle_records(30, gpu_utilization=idle_util)
        load = _load_records(30, gpu_utilization=load_util)
        ann  = _make_annotations()
        data = _make_telemetry(idle + load, ann)
        path = _write_telemetry(tmp, data)
        return TelemetryData(path)

    def test_pass_on_healthy_values(self):
        tmp = tempfile.mkdtemp()
        tel = self._telemetry(tmp, idle_util=2.0, load_util=85.0)
        result = validate_utilization(tel, _util_threshold())
        self.assertEqual(result["status"], "PASS", result.get("message"))

    def test_fail_idle_too_high(self):
        tmp = tempfile.mkdtemp()
        tel = self._telemetry(tmp, idle_util=25.0, load_util=85.0)
        result = validate_utilization(tel, _util_threshold())
        self.assertEqual(result["status"], "FAIL")

    def test_fail_load_too_low(self):
        tmp = tempfile.mkdtemp()
        tel = self._telemetry(tmp, idle_util=2.0, load_util=30.0)
        result = validate_utilization(tel, _util_threshold())
        self.assertEqual(result["status"], "FAIL")


class TestValidateEccCounter(unittest.TestCase):

    def _telemetry(self, tmp, sbe_vals: list[int]):
        """Build records where ecc_sbe_volatile follows sbe_vals list."""
        records = []
        for i, v in enumerate(sbe_vals):
            r = _load_records(1, ecc_sbe_volatile=v)[0]
            r["timestamp"] = f"2024-01-15T11:{i//60:02d}:{i%60:02d}+00:00"
            records.append(r)
        ann = _make_annotations()
        return TelemetryData(_write_telemetry(tmp, _make_telemetry(
            _idle_records(10) + records, ann
        )))

    def test_pass_zero_errors(self):
        tmp = tempfile.mkdtemp()
        tel = self._telemetry(tmp, [0] * 20)
        result = validate_ecc_counter(tel, _ecc_threshold(max_allowed=0))
        self.assertEqual(result["status"], "PASS")

    def test_fail_nonzero_dbe(self):
        tmp = tempfile.mkdtemp()
        tel = self._telemetry(tmp, [0, 0, 1, 1, 2])
        # max_allowed=0 → any non-zero is FAIL
        result = validate_ecc_counter(tel, _ecc_threshold(max_allowed=0))
        self.assertEqual(result["status"], "FAIL")

    def test_fail_non_monotonic(self):
        tmp = tempfile.mkdtemp()
        # Counter decreasing — should fail monotonicity check
        tel = self._telemetry(tmp, [5, 4, 3, 2, 1])
        result = validate_ecc_counter(tel, _ecc_threshold(max_allowed=100))
        self.assertEqual(result["status"], "FAIL")

    def test_pass_monotonic_increasing(self):
        tmp = tempfile.mkdtemp()
        # Allowed count=10, monotonically increasing SBEs
        tel = self._telemetry(tmp, [0, 1, 2, 3, 4])
        result = validate_ecc_counter(tel, _ecc_threshold(max_allowed=10))
        self.assertEqual(result["status"], "PASS")


class TestValidateThresholdRange(unittest.TestCase):

    def _telemetry(self, tmp, temp_val=72.0):
        records = _idle_records(10, gpu_temp_c=35.0) + _load_records(20, gpu_temp_c=temp_val)
        ann = _make_annotations()
        return TelemetryData(_write_telemetry(tmp, _make_telemetry(records, ann)))

    def test_pass_normal_temp(self):
        tmp = tempfile.mkdtemp()
        tel = self._telemetry(tmp, temp_val=72.0)
        result = validate_threshold_range(tel, _range_threshold())
        self.assertEqual(result["status"], "PASS")

    def test_fail_overtemp(self):
        tmp = tempfile.mkdtemp()
        tel = self._telemetry(tmp, temp_val=92.0)
        result = validate_threshold_range(tel, _range_threshold(max_val=87.0))
        self.assertEqual(result["status"], "FAIL")

    def test_fail_undertemp(self):
        tmp = tempfile.mkdtemp()
        tel = self._telemetry(tmp, temp_val=20.0)
        result = validate_threshold_range(tel, _range_threshold(min_val=30.0, max_val=87.0))
        self.assertEqual(result["status"], "FAIL")


class TestValidatePower(unittest.TestCase):

    def _telemetry(self, tmp, power_w=340.0, limit_w=400.0):
        records = _idle_records(10) + _load_records(20, power_draw_w=power_w, power_limit_w=limit_w)
        ann = _make_annotations()
        return TelemetryData(_write_telemetry(tmp, _make_telemetry(records, ann)))

    def test_pass_high_power(self):
        tmp = tempfile.mkdtemp()
        tel = self._telemetry(tmp, power_w=340.0, limit_w=400.0)   # 85% TDP
        result = validate_power(tel, _power_threshold())
        self.assertEqual(result["status"], "PASS")

    def test_fail_low_power(self):
        tmp = tempfile.mkdtemp()
        tel = self._telemetry(tmp, power_w=100.0, limit_w=400.0)   # 25% TDP
        result = validate_power(tel, _power_threshold())
        self.assertEqual(result["status"], "FAIL")


class TestValidateThroughput(unittest.TestCase):

    def _telemetry(self, tmp, pcie_tx=8000.0):
        records = _idle_records(10, pcie_tx_mb=50.0) + _load_records(20, pcie_tx_mb=pcie_tx)
        ann = _make_annotations()
        return TelemetryData(_write_telemetry(tmp, _make_telemetry(records, ann)))

    def test_pass_high_throughput(self):
        tmp = tempfile.mkdtemp()
        tel = self._telemetry(tmp, pcie_tx=8000.0)
        result = validate_throughput(tel, _throughput_threshold(min_val=5000.0))
        self.assertEqual(result["status"], "PASS")

    def test_fail_low_throughput(self):
        tmp = tempfile.mkdtemp()
        tel = self._telemetry(tmp, pcie_tx=1000.0)
        result = validate_throughput(tel, _throughput_threshold(min_val=5000.0))
        self.assertEqual(result["status"], "FAIL")


class TestValidateMemory(unittest.TestCase):

    def _telemetry(self, tmp, mem_used_mb=18000.0):
        records = _idle_records(10, memory_used_mb=500.0) + _load_records(20, memory_used_mb=mem_used_mb)
        ann = _make_annotations()
        return TelemetryData(_write_telemetry(tmp, _make_telemetry(records, ann)))

    def test_pass_high_memory_use(self):
        tmp = tempfile.mkdtemp()
        tel = self._telemetry(tmp, mem_used_mb=18000.0)
        # 18000/24576 = 73.2% > 70%
        result = validate_memory(tel, _memory_threshold(min_pct=70.0, total_mb=24576.0))
        self.assertEqual(result["status"], "PASS")

    def test_fail_low_memory_use(self):
        tmp = tempfile.mkdtemp()
        tel = self._telemetry(tmp, mem_used_mb=1000.0)
        result = validate_memory(tel, _memory_threshold(min_pct=70.0, total_mb=24576.0))
        self.assertEqual(result["status"], "FAIL")


class TestValidateClock(unittest.TestCase):

    def _telemetry(self, tmp, clock_mhz=1700.0):
        records = _idle_records(10, sm_clock_mhz=210.0) + _load_records(20, sm_clock_mhz=clock_mhz)
        ann = _make_annotations()
        return TelemetryData(_write_telemetry(tmp, _make_telemetry(records, ann)))

    def test_pass_high_clock(self):
        tmp = tempfile.mkdtemp()
        tel = self._telemetry(tmp, clock_mhz=1700.0)
        result = validate_clock(tel, _clock_threshold(min_pct=80.0, max_clock=1980.0))
        # 1700/1980 = 85.9% > 80%
        self.assertEqual(result["status"], "PASS")

    def test_fail_low_clock(self):
        tmp = tempfile.mkdtemp()
        tel = self._telemetry(tmp, clock_mhz=600.0)
        result = validate_clock(tel, _clock_threshold(min_pct=80.0, max_clock=1980.0))
        self.assertEqual(result["status"], "FAIL")


class TestValidateThrottleFlag(unittest.TestCase):

    def _telemetry(self, tmp, idle_throttle=False, load_throttle=True):
        idle = _idle_records(10, power_throttle=idle_throttle)
        load = _load_records(20, power_throttle=load_throttle)
        ann  = _make_annotations()
        return TelemetryData(_write_telemetry(tmp, _make_telemetry(idle + load, ann)))

    def test_pass_no_throttle_at_idle(self):
        tmp = tempfile.mkdtemp()
        tel = self._telemetry(tmp, idle_throttle=False, load_throttle=False)
        result = validate_throttle_flag(tel, _throttle_threshold())
        self.assertEqual(result["status"], "PASS")

    def test_fail_throttle_at_idle(self):
        tmp = tempfile.mkdtemp()
        tel = self._telemetry(tmp, idle_throttle=True, load_throttle=True)
        result = validate_throttle_flag(tel, _throttle_threshold())
        self.assertEqual(result["status"], "FAIL")


class TestValidateRateOfChange(unittest.TestCase):

    def _telemetry(self, tmp, ecc_sequence: list[int]):
        records = _idle_records(5)
        for i, val in enumerate(ecc_sequence):
            r = _load_records(1, ecc_sbe_volatile=val)[0]
            r["timestamp"] = f"2024-01-15T11:{i//60:02d}:{i%60:02d}+00:00"
            records.append(r)
        ann = _make_annotations()
        return TelemetryData(_write_telemetry(tmp, _make_telemetry(records, ann)))

    def test_pass_stable_counter(self):
        tmp = tempfile.mkdtemp()
        tel = self._telemetry(tmp, [0] * 10)
        result = validate_rate_of_change(tel, _roc_threshold(max_rate=0.5))
        self.assertEqual(result["status"], "PASS")

    def test_fail_fast_rising_counter(self):
        tmp = tempfile.mkdtemp()
        # 10 new errors per second (each record is 1s apart)
        tel = self._telemetry(tmp, list(range(0, 100, 10)))
        result = validate_rate_of_change(tel, _roc_threshold(max_rate=0.5))
        self.assertEqual(result["status"], "FAIL")


class TestValidateTimeSeriesCorrelation(unittest.TestCase):

    def _telemetry(self, tmp, correlated=True):
        """Build correlated or anti-correlated power/temp series."""
        records = []
        # Idle baseline
        for r in _idle_records(10):
            records.append(r)
        # Load window: power and temp either correlated or anti-correlated
        for i in range(20):
            power = 200.0 + i * 7.0   # rising
            temp  = 50.0 + i * 1.5 if correlated else 80.0 - i * 1.5
            r = _load_records(1, power_draw_w=power, gpu_temp_c=temp)[0]
            r["timestamp"] = f"2024-01-15T11:{i//60:02d}:{i%60:02d}+00:00"
            records.append(r)
        ann = _make_annotations()
        return TelemetryData(_write_telemetry(tmp, _make_telemetry(records, ann)))

    def test_pass_correlated(self):
        tmp = tempfile.mkdtemp()
        tel = self._telemetry(tmp, correlated=True)
        result = validate_time_series_correlation(tel, _corr_threshold(min_r=0.7))
        self.assertEqual(result["status"], "PASS")

    def test_fail_anti_correlated(self):
        tmp = tempfile.mkdtemp()
        tel = self._telemetry(tmp, correlated=False)
        result = validate_time_series_correlation(tel, _corr_threshold(min_r=0.7))
        self.assertEqual(result["status"], "FAIL")


# ---------------------------------------------------------------------------
# Integration-level: run_all_validators with minimal threshold config
# ---------------------------------------------------------------------------

class TestRunAllValidators(unittest.TestCase):

    def _build_thresholds(self) -> dict:
        return {
            "metrics": [
                _util_threshold(),
                _ecc_threshold(),
                _range_threshold(),
                _power_threshold(),
                _throughput_threshold(),
                _memory_threshold(),
                _clock_threshold(),
                _throttle_threshold(),
                _corr_threshold(),
            ]
        }

    def test_all_pass(self):
        tmp   = tempfile.mkdtemp()
        idle  = _idle_records(30)
        load  = _load_records(30)
        ann   = _make_annotations()
        data  = _make_telemetry(idle + load, ann)
        tpath = _write_telemetry(tmp, data)
        tel   = TelemetryData(tpath)

        thr_path = Path(tmp) / "thresholds.json"
        thr_path.write_text(json.dumps(self._build_thresholds()))

        results = run_all_validators(tpath, str(thr_path))
        statuses = {r["metric"]: r["status"] for r in results}

        # At minimum, utilization and ECC should be PASS on good data
        self.assertIn(statuses.get("gpu_occupancy"), ("PASS", "WARN", "SKIP"))
        self.assertIn(statuses.get("ecc_sbe_volatile"), ("PASS",))

    def test_detects_ecc_failure(self):
        tmp   = tempfile.mkdtemp()
        idle  = _idle_records(10)
        # Load records with non-zero DBE (always FAIL regardless of max_allowed=0)
        load  = _load_records(20, ecc_sbe_volatile=3)
        ann   = _make_annotations()
        data  = _make_telemetry(idle + load, ann)
        tpath = _write_telemetry(tmp, data)
        tel   = TelemetryData(tpath)

        thresholds = {"metrics": [_ecc_threshold(max_allowed=0)]}
        thr_path = Path(tmp) / "thresholds.json"
        thr_path.write_text(json.dumps(thresholds))

        results = run_all_validators(tpath, str(thr_path))
        ecc_result = next((r for r in results if r["metric"] == "ecc_sbe_volatile"), None)
        self.assertIsNotNone(ecc_result)
        self.assertEqual(ecc_result["status"], "FAIL")


# ---------------------------------------------------------------------------
# Dashboard validator unit tests (mocked Prometheus)
# ---------------------------------------------------------------------------

class TestDashboardValidator(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        idle = _idle_records(30)
        load = _load_records(30)
        ann  = _make_annotations()
        data = _make_telemetry(idle + load, ann)
        self.path = _write_telemetry(self.tmp, data)

    def test_compare_series_pass(self):
        """Identical ground-truth and prometheus → PASS."""
        import pandas as pd
        from validators.dashboard_validator import compare_series

        vals = list(range(10, 20))
        gt   = pd.Series(vals, name="test")
        pm   = pd.Series(vals, name="test")
        result = compare_series(gt, pm, tolerance=0.05, label="test")
        self.assertEqual(result["status"], "PASS")

    def test_compare_series_fail(self):
        """25% delta → FAIL at 5% tolerance."""
        import pandas as pd
        from validators.dashboard_validator import compare_series

        gt   = pd.Series([100.0] * 10, name="test")
        pm   = pd.Series([125.0] * 10, name="test")   # 25% higher
        result = compare_series(gt, pm, tolerance=0.05, label="test")
        self.assertEqual(result["status"], "FAIL")

    def test_compare_series_empty_prometheus(self):
        """Empty Prometheus series → WARN (not FAIL)."""
        import pandas as pd
        from validators.dashboard_validator import compare_series

        gt   = pd.Series([80.0] * 10, name="test")
        pm   = pd.Series([], dtype=float, name="test")
        result = compare_series(gt, pm, tolerance=0.05, label="test")
        self.assertEqual(result["status"], "WARN")


# ---------------------------------------------------------------------------
# ECC fault validator unit tests
# ---------------------------------------------------------------------------

class TestEccFaultValidator(unittest.TestCase):

    def test_synthetic_injection_creates_nonzero(self):
        """inject_synthetic_ecc_into_telemetry should produce FAIL on validation."""
        from validators.ecc_fault_validator import inject_synthetic_ecc_into_telemetry

        tmp   = tempfile.mkdtemp()
        idle  = _idle_records(10)
        load  = _load_records(20)
        ann   = _make_annotations()
        data  = _make_telemetry(idle + load, ann)
        tpath = _write_telemetry(tmp, data)

        injected_path = inject_synthetic_ecc_into_telemetry(
            tpath, sbe_count=5, dbe_count=2,
            output_path=str(Path(tmp) / "injected.json")
        )

        with open(injected_path) as f:
            injected = json.load(f)

        # At least one record should have non-zero ecc_sbe_volatile
        records = injected.get("records", injected) if isinstance(injected, dict) else injected
        sbe_vals = [r.get("ecc_sbe_volatile", 0) for r in records]
        self.assertTrue(any(v > 0 for v in sbe_vals), "Expected non-zero SBE after injection")


if __name__ == "__main__":
    unittest.main(verbosity=2)
