#!/usr/bin/env python3
"""
validators/run_validators.py

Orchestrates all metric validators. Loads collected telemetry,
runs per-metric validation logic, and produces a PASS/FAIL report.

Usage:
    python3 validators/run_validators.py \
        --telemetry reports/telemetry.json \
        --config configs/thresholds.yaml \
        --output reports/validation_report.html
"""

import json
import math
import logging
import argparse
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("validators")


class Status(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"
    NO_DATA = "NO_DATA"


@dataclass
class ValidationResult:
    metric_name: str
    description: str
    status: Status
    observed_idle: Optional[float] = None
    observed_load: Optional[float] = None
    expected_idle: Optional[str] = None
    expected_load: Optional[str] = None
    details: str = ""
    sub_checks: List[Dict] = field(default_factory=list)


# ─── Data Loader ────────────────────────────────────────────────────────────

class TelemetryData:
    """Load and index telemetry records from JSON."""

    def __init__(self, telemetry_path: str):
        with open(telemetry_path) as f:
            self.raw = json.load(f)
        # Separate workload annotations from metric records
        self.annotations = [r for r in self.raw if r.get("type") == "workload_annotation"]
        self.metrics = [r for r in self.raw if r.get("type") != "workload_annotation"]
        logger.info(
            "Loaded %d metric records, %d annotations from %s",
            len(self.metrics), len(self.annotations), telemetry_path
        )

    def get_workload_windows(self) -> Dict[str, Tuple[str, str]]:
        """Return {workload_name: (start_ts, end_ts)} from annotations."""
        windows = {}
        start_times = {}
        for ann in self.annotations:
            wl = ann["workload"]
            ts = ann["snapshot_timestamp"]
            if ann["event"] == "start":
                start_times[wl] = ts
            elif ann["event"] == "stop" and wl in start_times:
                windows[wl] = (start_times[wl], ts)
        return windows

    def get_values_in_window(self, field: str, start_ts: str, end_ts: str) -> List[float]:
        """Get all numeric values for a field in a time window."""
        values = []
        for r in self.metrics:
            ts = r.get("snapshot_timestamp") or r.get("collected_at") or ""
            if start_ts <= ts <= end_ts:
                v = r.get(field)
                if v is not None:
                    try:
                        fv = float(v)
                        if not math.isnan(fv):
                            values.append(fv)
                    except (ValueError, TypeError):
                        pass
        return values

    def get_all_values(self, field: str) -> List[float]:
        """Get all values for a field across all records."""
        values = []
        for r in self.metrics:
            v = r.get(field)
            if v is not None:
                try:
                    fv = float(v)
                    if not math.isnan(fv):
                        values.append(fv)
                except (ValueError, TypeError):
                    pass
        return values

    def get_idle_values(self, field: str, idle_window_sec: int = 60) -> List[float]:
        """Get values from the first idle_window_sec seconds (before any workload)."""
        if not self.metrics:
            return []
        first_ts = self.metrics[0].get("snapshot_timestamp", "")
        if not first_ts:
            return []
        # Simple: take first N records
        idle_records = self.metrics[:idle_window_sec]
        values = []
        for r in idle_records:
            v = r.get(field)
            if v is not None:
                try:
                    fv = float(v)
                    if not math.isnan(fv):
                        values.append(fv)
                except (ValueError, TypeError):
                    pass
        return values


# ─── Validators ─────────────────────────────────────────────────────────────

def validate_utilization(metric_name: str, config: Dict, data: TelemetryData) -> ValidationResult:
    """Validate utilization metrics (occupancy, SM activity, tensor core, etc.)."""
    field = config.get("dcgm_field") or config.get("nvidia_smi_query", metric_name)
    # Map DCGM field constants to collector field names
    field_map = {
        "DCGM_FI_PROF_SM_OCCUPANCY": "sm_occupancy",
        "DCGM_FI_PROF_SM_ACTIVE": "sm_active",
        "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE": "pipe_tensor_active",
        "DCGM_FI_PROF_PIPE_FP32_ACTIVE": "pipe_fp32_active",
        "DCGM_FI_DEV_MEM_COPY_UTIL": "mem_copy_util",
        "utilization.gpu": "utilization.gpu",
        "utilization.memory": "utilization.memory",
    }
    field_name = field_map.get(field, metric_name)

    idle_vals = data.get_idle_values(field_name)
    all_vals = data.get_all_values(field_name)

    result = ValidationResult(
        metric_name=metric_name,
        description=config.get("description", metric_name),
        status=Status.NO_DATA,
    )

    if not all_vals:
        result.details = f"No data found for field '{field_name}'"
        return result

    sub_checks = []
    idle_cfg = config.get("idle", {})
    load_cfg = config.get("under_load", {})
    tolerance = config.get("tolerance_pct", 5)

    # Idle check
    if idle_vals:
        idle_p95 = sorted(idle_vals)[int(len(idle_vals) * 0.95)]
        result.observed_idle = idle_p95
        max_idle = idle_cfg.get("expected_lte", None)
        if max_idle is not None:
            if idle_p95 <= max_idle + tolerance:
                sub_checks.append({"check": f"idle p95 ≤ {max_idle}%", "status": "PASS", "value": idle_p95})
            else:
                sub_checks.append({"check": f"idle p95 ≤ {max_idle}%", "status": "FAIL", "value": idle_p95})

    # Load check
    load_vals = [v for v in all_vals if v > (idle_cfg.get("expected_lte", 10) / 2)]
    if load_vals:
        load_p50 = statistics.median(load_vals)
        result.observed_load = load_p50
        min_load = load_cfg.get("expected_gte", None)
        warn_load = load_cfg.get("warning_lt", None)

        if min_load is not None:
            result.expected_load = f"> {min_load}%"
            if load_p50 >= min_load - tolerance:
                sub_checks.append({"check": f"load p50 ≥ {min_load}%", "status": "PASS", "value": load_p50})
            elif warn_load and load_p50 >= warn_load:
                sub_checks.append({"check": f"load p50 ≥ {min_load}%", "status": "WARN", "value": load_p50})
            else:
                sub_checks.append({"check": f"load p50 ≥ {min_load}%", "status": "FAIL", "value": load_p50})
    else:
        sub_checks.append({"check": "load data availability", "status": "WARN", "value": None,
                            "detail": "No load-phase data detected (workload may not have run)"})

    result.sub_checks = sub_checks
    fails = [c for c in sub_checks if c["status"] == "FAIL"]
    warns = [c for c in sub_checks if c["status"] == "WARN"]
    result.status = Status.FAIL if fails else (Status.WARN if warns else Status.PASS)
    return result


def validate_ecc_counter(metric_name: str, config: Dict, data: TelemetryData) -> ValidationResult:
    """Validate ECC error counters — must be monotonically non-decreasing."""
    field_map = {
        "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL": "dbe_volatile",
        "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL": "sbe_volatile",
        "DCGM_FI_DEV_ECC_SBE_AGG_TOTAL": "sbe_aggregate",
        "DCGM_FI_DEV_ECC_DBE_AGG_TOTAL": "dbe_aggregate",
        "DCGM_FI_DEV_RETIRED_SBE": "retired_pages_sbe",
        "DCGM_FI_DEV_RETIRED_DBE": "retired_pages_dbe",
        "ecc.errors.corrected.aggregate.total": "ecc.errors.corrected.aggregate.total",
        "ecc.errors.uncorrected.volatile.total": "ecc.errors.uncorrected.volatile.total",
    }
    dcgm_field = config.get("dcgm_field", "")
    smi_field = config.get("nvidia_smi_query", "")
    field_name = field_map.get(dcgm_field) or field_map.get(smi_field) or metric_name

    all_vals = data.get_all_values(field_name)

    result = ValidationResult(
        metric_name=metric_name,
        description=config.get("description", metric_name),
        status=Status.NO_DATA,
    )

    if not all_vals:
        # Try alternative field name
        alt_field = smi_field.replace(".", "_").lower() if smi_field else None
        if alt_field:
            all_vals = data.get_all_values(alt_field)

    if not all_vals:
        result.details = f"No ECC data for '{field_name}'. ECC may not be enabled or reported."
        result.status = Status.SKIP
        return result

    sub_checks = []
    idle_cfg = config.get("idle", {})
    load_cfg = config.get("under_load", {})

    final_val = all_vals[-1]
    result.observed_load = final_val

    # Monotonicity check
    if config.get("under_load", {}).get("must_be_monotonic", False):
        is_monotonic = all(all_vals[i] <= all_vals[i + 1] for i in range(len(all_vals) - 1))
        if is_monotonic:
            sub_checks.append({"check": "monotonically non-decreasing", "status": "PASS", "value": final_val})
        else:
            # Non-monotonic ECC counters indicate driver reset or counter wrap
            sub_checks.append({"check": "monotonically non-decreasing", "status": "WARN",
                                "value": final_val, "detail": "Counter decreased — possible driver reset"})

    # Max allowed check
    max_allowed = idle_cfg.get("max_allowed")
    if max_allowed is not None and final_val > max_allowed:
        sub_checks.append({
            "check": f"≤ {max_allowed} errors",
            "status": "FAIL",
            "value": final_val,
            "detail": f"Non-zero ECC errors detected: {final_val}. Investigate hardware."
        })
    elif max_allowed is not None:
        sub_checks.append({"check": f"≤ {max_allowed} errors", "status": "PASS", "value": final_val})
    else:
        sub_checks.append({"check": "counter recorded", "status": "PASS", "value": final_val})

    result.sub_checks = sub_checks
    result.expected_idle = str(idle_cfg.get("expected", "≥ 0"))
    fails = [c for c in sub_checks if c["status"] == "FAIL"]
    warns = [c for c in sub_checks if c["status"] == "WARN"]
    result.status = Status.FAIL if fails else (Status.WARN if warns else Status.PASS)
    return result


def validate_rate_of_change(metric_name: str, config: Dict, data: TelemetryData) -> ValidationResult:
    """Validate rate-of-change metrics (ECC SBE volatile RoC, PCIe replay RoC)."""
    field_map = {
        "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL": "sbe_volatile",
        "DCGM_FI_DEV_PCIE_REPLAY_COUNTER": "pcie_replay",
    }
    field_name = field_map.get(config.get("dcgm_field", ""), metric_name)
    all_vals = data.get_all_values(field_name)

    result = ValidationResult(
        metric_name=metric_name,
        description=config.get("description", metric_name),
        status=Status.NO_DATA,
    )

    if len(all_vals) < 2:
        result.details = "Insufficient data for rate-of-change calculation"
        result.status = Status.SKIP
        return result

    # Calculate RoC over the last window
    window_sec = config.get("roc_window_sec", 60)
    window_vals = all_vals[-min(window_sec, len(all_vals)):]
    if len(window_vals) < 2:
        result.status = Status.SKIP
        return result

    delta = window_vals[-1] - window_vals[0]
    roc_per_min = delta / (len(window_vals) / 60.0)

    result.observed_load = roc_per_min
    max_roc_alert = config.get("max_roc_alert", config.get("max_roc_warn", float("inf")))
    expected_idle_roc = config.get("expected_roc_idle", 0.0)

    sub_checks = []
    if roc_per_min == 0:
        sub_checks.append({"check": f"RoC = {expected_idle_roc}/min", "status": "PASS", "value": roc_per_min})
    elif roc_per_min < max_roc_alert:
        sub_checks.append({"check": f"RoC < {max_roc_alert}/min", "status": "WARN", "value": roc_per_min,
                            "detail": "Non-zero rate — monitor"})
    else:
        sub_checks.append({"check": f"RoC < {max_roc_alert}/min", "status": "FAIL", "value": roc_per_min,
                            "detail": f"High error rate detected: {roc_per_min:.2f}/min"})

    result.sub_checks = sub_checks
    result.expected_load = f"RoC < {max_roc_alert}/min"
    fails = [c for c in sub_checks if c["status"] == "FAIL"]
    warns = [c for c in sub_checks if c["status"] == "WARN"]
    result.status = Status.FAIL if fails else (Status.WARN if warns else Status.PASS)
    return result


def validate_threshold_range(metric_name: str, config: Dict, data: TelemetryData) -> ValidationResult:
    """Validate temperature and similar range-bounded metrics."""
    field_map = {
        "DCGM_FI_DEV_GPU_TEMP": "gpu_temp",
        "DCGM_FI_DEV_MEMORY_TEMP": "mem_temp",
        "temperature.gpu": "temperature.gpu",
        "temperature.memory": "temperature.memory",
    }
    field = config.get("dcgm_field") or config.get("nvidia_smi_query", metric_name)
    field_name = field_map.get(field, metric_name.replace(".", "_"))

    idle_vals = data.get_idle_values(field_name)
    all_vals = data.get_all_values(field_name)

    result = ValidationResult(
        metric_name=metric_name,
        description=config.get("description", metric_name),
        status=Status.NO_DATA,
    )

    if not all_vals:
        result.details = f"No data for field '{field_name}'"
        return result

    idle_cfg = config.get("idle", {})
    load_cfg = config.get("under_load", {})
    sub_checks = []

    # Idle check
    if idle_vals:
        idle_mean = statistics.mean(idle_vals)
        result.observed_idle = idle_mean
        if "expected_lte" in idle_cfg and idle_mean > idle_cfg["expected_lte"]:
            sub_checks.append({"check": f"idle ≤ {idle_cfg['expected_lte']}°C", "status": "FAIL", "value": idle_mean})
        elif "expected_gte" in idle_cfg and idle_mean < idle_cfg["expected_gte"]:
            sub_checks.append({"check": f"idle ≥ {idle_cfg['expected_gte']}°C", "status": "FAIL", "value": idle_mean})
        else:
            sub_checks.append({"check": "idle temperature in range", "status": "PASS", "value": idle_mean})

    # Load check (use all values, filter outliers)
    if all_vals:
        load_max = max(all_vals)
        load_mean = statistics.mean(all_vals)
        result.observed_load = load_max

        # Hard upper limit check
        hard_max = load_cfg.get("expected_lte")
        if hard_max and load_max > hard_max:
            sub_checks.append({"check": f"peak ≤ {hard_max}°C", "status": "FAIL", "value": load_max,
                                "detail": "GPU approaching or exceeded thermal limit"})
        else:
            sub_checks.append({"check": f"peak ≤ {hard_max or 95}°C", "status": "PASS", "value": load_max})

        # Warning threshold
        warn_gte = load_cfg.get("warning_gte")
        if warn_gte and load_max >= warn_gte:
            sub_checks.append({"check": f"temp < {warn_gte}°C (warning)", "status": "WARN", "value": load_max})

        # Under-load minimum
        min_load = load_cfg.get("expected_gte")
        if min_load and load_mean < min_load:
            sub_checks.append({"check": f"load ≥ {min_load}°C", "status": "WARN", "value": load_mean,
                                "detail": "May not have run full thermal workload"})

    result.sub_checks = sub_checks
    fails = [c for c in sub_checks if c["status"] == "FAIL"]
    warns = [c for c in sub_checks if c["status"] == "WARN"]
    result.status = Status.FAIL if fails else (Status.WARN if warns else Status.PASS)
    return result


def validate_power(metric_name: str, config: Dict, data: TelemetryData) -> ValidationResult:
    """Validate power usage relative to TDP."""
    power_vals = data.get_all_values("power.draw") or data.get_all_values("power_usage")
    limit_vals = data.get_all_values("power.limit") or data.get_all_values("power_limit")

    result = ValidationResult(
        metric_name=metric_name,
        description=config.get("description", metric_name),
        status=Status.NO_DATA,
    )

    if not power_vals:
        result.details = "No power.draw data found"
        return result

    tdp = statistics.mean(limit_vals) if limit_vals else None
    load_cfg = config.get("under_load", {})
    idle_cfg = config.get("idle", {})
    sub_checks = []

    # Idle check
    idle_vals = power_vals[:60]
    if idle_vals and tdp:
        idle_pct = (statistics.mean(idle_vals) / tdp) * 100
        result.observed_idle = idle_pct
        max_idle_pct = idle_cfg.get("expected_lte_pct_tdp", 30)
        if idle_pct <= max_idle_pct:
            sub_checks.append({"check": f"idle ≤ {max_idle_pct}% TDP", "status": "PASS", "value": f"{idle_pct:.1f}%"})
        else:
            sub_checks.append({"check": f"idle ≤ {max_idle_pct}% TDP", "status": "WARN", "value": f"{idle_pct:.1f}%"})

    # Load check
    load_peak = max(power_vals)
    result.observed_load = load_peak
    if tdp:
        load_pct = (load_peak / tdp) * 100
        result.expected_load = f"≥ {load_cfg.get('expected_gte_pct_tdp', 70)}% TDP ({tdp:.0f}W)"
        min_load_pct = load_cfg.get("expected_gte_pct_tdp", 70)
        warn_pct = load_cfg.get("warning_lt_pct_tdp", 50)
        if load_pct >= min_load_pct:
            sub_checks.append({"check": f"peak ≥ {min_load_pct}% TDP", "status": "PASS", "value": f"{load_pct:.1f}%"})
        elif load_pct >= warn_pct:
            sub_checks.append({"check": f"peak ≥ {min_load_pct}% TDP", "status": "WARN", "value": f"{load_pct:.1f}%"})
        else:
            sub_checks.append({"check": f"peak ≥ {min_load_pct}% TDP", "status": "FAIL", "value": f"{load_pct:.1f}%"})
    else:
        sub_checks.append({"check": "power draw recorded", "status": "PASS", "value": f"{load_peak:.1f}W"})

    result.sub_checks = sub_checks
    fails = [c for c in sub_checks if c["status"] == "FAIL"]
    warns = [c for c in sub_checks if c["status"] == "WARN"]
    result.status = Status.FAIL if fails else (Status.WARN if warns else Status.PASS)
    return result


def validate_throttle_flag(metric_name: str, config: Dict, data: TelemetryData) -> ValidationResult:
    """Validate power/thermal throttling flag metrics."""
    # Map to nvidia-smi fields
    field_map = {
        "DCGM_FI_DEV_POWER_VIOLATION": "power_violation",
        "DCGM_FI_DEV_THERMAL_VIOLATION": "thermal_violation",
    }
    field_name = field_map.get(config.get("dcgm_field", ""), metric_name)
    vals = data.get_all_values(field_name)

    result = ValidationResult(
        metric_name=metric_name,
        description=config.get("description", metric_name),
        status=Status.NO_DATA,
    )

    load_cfg = config.get("under_load", {})
    may_be_nonzero = load_cfg.get("may_be_nonzero", False)

    if not vals:
        # Check nvidia-smi throttle reason fields
        throttle_fields = [
            "clocks_throttle_reasons.hw_power_brake_slowdown",
            "clocks_throttle_reasons.hw_thermal_slowdown",
        ]
        for tf in throttle_fields:
            tvals = data.get_all_values(tf)
            if tvals:
                vals = tvals
                break

    if not vals:
        result.status = Status.SKIP
        result.details = "No throttle data. Metric requires DCGM or nvidia-smi with full query support."
        return result

    nonzero = [v for v in vals if v and v != 0 and str(v) not in ("0", "N/A", "Not Active")]
    sub_checks = []

    if nonzero and may_be_nonzero:
        sub_checks.append({
            "check": "throttle flag observed (expected under max stress)",
            "status": "PASS",
            "value": f"{len(nonzero)} events",
            "detail": "Throttling confirms GPU reached thermal/power limits — metric is working"
        })
    elif not nonzero:
        sub_checks.append({
            "check": "throttle state recorded",
            "status": "WARN" if may_be_nonzero else "PASS",
            "value": 0,
            "detail": "No throttling observed — expected if workload didn't max out GPU"
        })
    else:
        sub_checks.append({
            "check": "throttle state recorded",
            "status": "PASS",
            "value": len(nonzero)
        })

    result.sub_checks = sub_checks
    fails = [c for c in sub_checks if c["status"] == "FAIL"]
    warns = [c for c in sub_checks if c["status"] == "WARN"]
    result.status = Status.FAIL if fails else (Status.WARN if warns else Status.PASS)
    return result


def validate_throughput(metric_name: str, config: Dict, data: TelemetryData) -> ValidationResult:
    """Validate throughput metrics (PCIe, NVLink)."""
    field_map = {
        "DCGM_FI_PROF_PCIE_TX_BYTES": "pcie_tx_bytes",
        "DCGM_FI_PROF_PCIE_RX_BYTES": "pcie_rx_bytes",
        "DCGM_FI_PROF_NVLINK_TX_BYTES": "nvlink_tx",
        "DCGM_FI_PROF_NVLINK_RX_BYTES": "nvlink_rx",
    }
    field_name = field_map.get(config.get("dcgm_field", ""), metric_name)
    vals = data.get_all_values(field_name)

    result = ValidationResult(
        metric_name=metric_name,
        description=config.get("description", metric_name),
        status=Status.NO_DATA,
    )

    if not vals:
        # Skip NVLink on non-NVLink systems
        if config.get("under_load", {}).get("skip_if_no_nvlink", False):
            result.status = Status.SKIP
            result.details = "NVLink not present or not active on this system"
        else:
            result.details = f"No data for '{field_name}'"
        return result

    load_cfg = config.get("under_load", {})
    unit = config.get("unit", "mb_per_sec")

    # Convert bytes/sec to MB/s if needed
    if vals[0] > 1e6:
        display_vals = [v / 1e6 for v in vals]  # bytes → MB/s
    else:
        display_vals = vals

    peak = max(display_vals)
    result.observed_load = peak

    sub_checks = []
    min_load = load_cfg.get("expected_gte", 0)
    if min_load and peak >= min_load:
        sub_checks.append({"check": f"peak ≥ {min_load} {unit}", "status": "PASS", "value": f"{peak:.1f}"})
    elif min_load:
        sub_checks.append({"check": f"peak ≥ {min_load} {unit}", "status": "FAIL", "value": f"{peak:.1f}",
                            "detail": f"Low throughput: {peak:.1f} {unit}. PCIe workload may not have run."})
    else:
        sub_checks.append({"check": "throughput recorded", "status": "PASS", "value": f"{peak:.1f}"})

    result.sub_checks = sub_checks
    fails = [c for c in sub_checks if c["status"] == "FAIL"]
    result.status = Status.FAIL if fails else Status.PASS
    return result


def validate_memory(metric_name: str, config: Dict, data: TelemetryData) -> ValidationResult:
    """Validate GPU memory utilization."""
    used_vals = data.get_all_values("memory.used") or data.get_all_values("fb_used")
    total_vals = data.get_all_values("memory.total")
    total = statistics.mean(total_vals) if total_vals else None

    result = ValidationResult(
        metric_name=metric_name,
        description=config.get("description", metric_name),
        status=Status.NO_DATA,
    )

    if not used_vals:
        result.details = "No memory.used data"
        return result

    load_cfg = config.get("under_load", {})
    peak = max(used_vals)
    result.observed_load = peak / 1024 if peak > 1024 else peak  # MiB → GiB

    sub_checks = []
    if total:
        peak_pct = (peak / total) * 100
        min_pct = load_cfg.get("expected_gte_pct_total", 70)
        tolerance = config.get("tolerance_pct", 5)
        result.expected_load = f"≥ {min_pct}% of {total / 1024:.1f} GiB"
        if peak_pct >= min_pct - tolerance:
            sub_checks.append({"check": f"peak ≥ {min_pct}% VRAM", "status": "PASS", "value": f"{peak_pct:.1f}%"})
        else:
            sub_checks.append({"check": f"peak ≥ {min_pct}% VRAM", "status": "WARN", "value": f"{peak_pct:.1f}%",
                                "detail": "Memory workload may not have fully exercised VRAM"})
    else:
        sub_checks.append({"check": "memory usage recorded", "status": "PASS", "value": f"{peak:.0f} MiB"})

    result.sub_checks = sub_checks
    warns = [c for c in sub_checks if c["status"] == "WARN"]
    result.status = Status.WARN if warns else Status.PASS
    return result


def validate_clock(metric_name: str, config: Dict, data: TelemetryData) -> ValidationResult:
    """Validate SM clock speeds."""
    field_name = "clocks.current.sm" if "sm_clock" in metric_name else "clocks.current.memory"
    vals = data.get_all_values(field_name) or data.get_all_values("sm_clock")

    result = ValidationResult(
        metric_name=metric_name,
        description=config.get("description", metric_name),
        status=Status.NO_DATA,
    )

    if not vals:
        result.details = "No clock data"
        return result

    max_clock = max(vals)
    load_cfg = config.get("under_load", {})
    min_pct = load_cfg.get("expected_gte_pct_max", 80)
    threshold = max_clock * (min_pct / 100)

    load_vals = [v for v in vals if v >= threshold * 0.5]
    peak = max(load_vals) if load_vals else max_clock
    result.observed_load = peak
    result.expected_load = f"≥ {min_pct}% of max ({max_clock:.0f} MHz)"

    sub_checks = []
    if peak >= threshold:
        sub_checks.append({"check": f"clock ≥ {min_pct}% of max", "status": "PASS", "value": f"{peak:.0f} MHz"})
    else:
        sub_checks.append({"check": f"clock ≥ {min_pct}% of max", "status": "WARN", "value": f"{peak:.0f} MHz",
                            "detail": "SM clock below expected — possible throttling or light workload"})

    result.sub_checks = sub_checks
    warns = [c for c in sub_checks if c["status"] == "WARN"]
    result.status = Status.WARN if warns else Status.PASS
    return result


def validate_time_series_correlation(metric_name: str, config: Dict, data: TelemetryData) -> ValidationResult:
    """Validate correlation between temperature and power over time."""
    temp_vals = data.get_all_values("temperature.gpu") or data.get_all_values("gpu_temp")
    power_vals = data.get_all_values("power.draw") or data.get_all_values("power_usage")

    result = ValidationResult(
        metric_name=metric_name,
        description=config.get("description", metric_name),
        status=Status.NO_DATA,
    )

    if not temp_vals or not power_vals:
        result.details = "Insufficient data for correlation analysis"
        return result

    # Align lengths
    n = min(len(temp_vals), len(power_vals))
    temp_vals = temp_vals[:n]
    power_vals = power_vals[:n]

    if n < 10:
        result.status = Status.SKIP
        result.details = "Too few samples for correlation"
        return result

    # Pearson correlation
    mean_t = statistics.mean(temp_vals)
    mean_p = statistics.mean(power_vals)
    std_t = statistics.stdev(temp_vals) if len(temp_vals) > 1 else 1
    std_p = statistics.stdev(power_vals) if len(power_vals) > 1 else 1

    if std_t == 0 or std_p == 0:
        result.status = Status.SKIP
        result.details = "Zero variance — constant values"
        return result

    cov = sum((t - mean_t) * (p - mean_p) for t, p in zip(temp_vals, power_vals)) / n
    corr = cov / (std_t * std_p)
    result.observed_load = corr

    min_corr = config.get("expected_correlation_gte", 0.6)
    sub_checks = []
    if corr >= min_corr:
        sub_checks.append({"check": f"temp-power correlation ≥ {min_corr}", "status": "PASS", "value": f"{corr:.3f}"})
    else:
        sub_checks.append({"check": f"temp-power correlation ≥ {min_corr}", "status": "WARN", "value": f"{corr:.3f}",
                            "detail": "Low correlation may indicate sensor or workload issue"})

    result.sub_checks = sub_checks
    result.expected_load = f"Pearson r ≥ {min_corr}"
    warns = [c for c in sub_checks if c["status"] == "WARN"]
    result.status = Status.WARN if warns else Status.PASS
    return result


# ─── Dispatcher ─────────────────────────────────────────────────────────────

VALIDATOR_MAP = {
    "utilization": validate_utilization,
    "ecc_counter": validate_ecc_counter,
    "rate_of_change": validate_rate_of_change,
    "threshold_range": validate_threshold_range,
    "power": validate_power,
    "throttle_flag": validate_throttle_flag,
    "throughput": validate_throughput,
    "memory": validate_memory,
    "clock": validate_clock,
    "time_series_correlation": validate_time_series_correlation,
}


def run_all_validators(telemetry_path: str, config_path: str) -> List[ValidationResult]:
    """Load telemetry + config, dispatch per-metric validators."""
    data = TelemetryData(telemetry_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    results = []
    metrics = config.get("metrics", {})

    for metric_name, metric_config in metrics.items():
        vtype = metric_config.get("validation_type", "utilization")
        validator_fn = VALIDATOR_MAP.get(vtype)

        if validator_fn is None:
            logger.warning("No validator for type '%s' (metric: %s)", vtype, metric_name)
            continue

        logger.info("Validating: %s (%s)", metric_name, vtype)
        try:
            result = validator_fn(metric_name, metric_config, data)
        except Exception as e:
            logger.error("Validator error for %s: %s", metric_name, e, exc_info=True)
            result = ValidationResult(
                metric_name=metric_name,
                description=metric_config.get("description", metric_name),
                status=Status.FAIL,
                details=f"Validator exception: {e}"
            )
        results.append(result)
        logger.info(
            "  %s → %s %s",
            metric_name,
            result.status.value,
            f"[{result.details[:80]}]" if result.details else ""
        )

    return results


# ─── Report Generator ────────────────────────────────────────────────────────

def generate_html_report(results: List[ValidationResult], output_path: str, telemetry_path: str):
    """Generate a styled HTML validation report."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    status_counts = {s: sum(1 for r in results if r.status == s) for s in Status}
    total = len(results)
    passed = status_counts.get(Status.PASS, 0)
    failed = status_counts.get(Status.FAIL, 0)
    warned = status_counts.get(Status.WARN, 0)
    skipped = status_counts.get(Status.SKIP, 0) + status_counts.get(Status.NO_DATA, 0)

    status_icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "SKIP": "⏭️", "NO_DATA": "❓"}
    status_color = {
        "PASS": "#22c55e", "FAIL": "#ef4444", "WARN": "#f59e0b",
        "SKIP": "#6b7280", "NO_DATA": "#9ca3af"
    }

    rows = ""
    for r in results:
        icon = status_icon.get(r.status.value, "?")
        color = status_color.get(r.status.value, "#6b7280")
        sub_html = ""
        for sc in r.sub_checks:
            sc_icon = status_icon.get(sc.get("status", "PASS"), "?")
            sc_detail = f" — {sc.get('detail', '')}" if sc.get("detail") else ""
            sc_val = f" ({sc.get('value', '')})" if sc.get("value") is not None else ""
            sub_html += f'<div class="sub-check">{sc_icon} {sc.get("check","")}{sc_val}{sc_detail}</div>'

        rows += f"""
        <tr>
            <td class="metric-name">{r.metric_name}</td>
            <td class="metric-desc">{r.description}</td>
            <td>{r.observed_idle if r.observed_idle is not None else "—"}</td>
            <td>{r.observed_load if r.observed_load is not None else "—"}</td>
            <td>{r.expected_load or r.expected_idle or "—"}</td>
            <td><span class="status-badge" style="background:{color}">{icon} {r.status.value}</span></td>
            <td class="details">{sub_html}{r.details}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GPU Observability Validation Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'IBM Plex Mono', 'Courier New', monospace; background: #0a0a0f; color: #e2e8f0; min-height: 100vh; }}
  .header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%); padding: 40px 60px; border-bottom: 2px solid #22d3ee; }}
  .header h1 {{ font-size: 2rem; color: #22d3ee; letter-spacing: 0.1em; margin-bottom: 8px; }}
  .header .subtitle {{ color: #94a3b8; font-size: 0.85rem; }}
  .summary {{ display: flex; gap: 20px; padding: 30px 60px; background: #0f172a; }}
  .summary-card {{ flex: 1; background: #1e293b; border-radius: 8px; padding: 20px; border-left: 4px solid; text-align: center; }}
  .summary-card .count {{ font-size: 2.5rem; font-weight: bold; }}
  .summary-card .label {{ font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 4px; }}
  .card-pass {{ border-color: #22c55e; }} .card-pass .count {{ color: #22c55e; }}
  .card-fail {{ border-color: #ef4444; }} .card-fail .count {{ color: #ef4444; }}
  .card-warn {{ border-color: #f59e0b; }} .card-warn .count {{ color: #f59e0b; }}
  .card-skip {{ border-color: #6b7280; }} .card-skip .count {{ color: #6b7280; }}
  .card-total {{ border-color: #22d3ee; }} .card-total .count {{ color: #22d3ee; }}
  .table-container {{ padding: 30px 60px; overflow-x: auto; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
  th {{ background: #1e293b; color: #22d3ee; padding: 12px 16px; text-align: left; border-bottom: 2px solid #334155; font-weight: 600; letter-spacing: 0.05em; white-space: nowrap; }}
  tr:nth-child(even) {{ background: #0f172a; }}
  tr:hover {{ background: #1e293b; }}
  td {{ padding: 10px 16px; border-bottom: 1px solid #1e293b; vertical-align: top; }}
  .metric-name {{ color: #22d3ee; font-weight: bold; white-space: nowrap; }}
  .metric-desc {{ color: #94a3b8; max-width: 200px; }}
  .status-badge {{ display: inline-block; padding: 3px 10px; border-radius: 4px; font-size: 0.78rem; font-weight: bold; color: white; white-space: nowrap; }}
  .sub-check {{ font-size: 0.75rem; color: #94a3b8; margin: 2px 0; padding-left: 8px; border-left: 2px solid #334155; }}
  .details {{ max-width: 300px; font-size: 0.78rem; color: #64748b; }}
  .footer {{ padding: 20px 60px; text-align: center; color: #475569; font-size: 0.78rem; border-top: 1px solid #1e293b; }}
  .section-header {{ padding: 16px 60px 8px; color: #64748b; font-size: 0.78rem; letter-spacing: 0.15em; text-transform: uppercase; }}
</style>
</head>
<body>
<div class="header">
  <h1>⚡ GPU Observability Validation Report</h1>
  <div class="subtitle">Generated: {now} | Telemetry: {telemetry_path}</div>
</div>
<div class="summary">
  <div class="summary-card card-total"><div class="count">{total}</div><div class="label">Total Checks</div></div>
  <div class="summary-card card-pass"><div class="count">{passed}</div><div class="label">Passed</div></div>
  <div class="summary-card card-fail"><div class="count">{failed}</div><div class="label">Failed</div></div>
  <div class="summary-card card-warn"><div class="count">{warned}</div><div class="label">Warnings</div></div>
  <div class="summary-card card-skip"><div class="count">{skipped}</div><div class="label">Skipped</div></div>
</div>
<div class="table-container">
<table>
  <thead>
    <tr>
      <th>Metric</th><th>Description</th><th>Idle Value</th><th>Load Value</th><th>Expected</th><th>Status</th><th>Details</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>
</div>
<div class="footer">GPU Observability Framework — Validation complete. {passed}/{total} checks passed.</div>
</body>
</html>"""

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(html)
    logger.info("HTML report written to %s", output_path)


def print_summary_table(results: List[ValidationResult]):
    """Print a CLI summary table."""
    status_icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "SKIP": "⏭️", "NO_DATA": "❓"}
    print("\n" + "═" * 90)
    print("  GPU OBSERVABILITY VALIDATION REPORT")
    print("═" * 90)
    print(f"  {'Metric':<35} {'Idle':<12} {'Load':<15} {'Status'}")
    print("─" * 90)
    for r in results:
        icon = status_icon.get(r.status.value, "?")
        idle = f"{r.observed_idle:.2f}" if r.observed_idle is not None else "—"
        load = f"{r.observed_load:.2f}" if r.observed_load is not None else "—"
        print(f"  {r.metric_name:<35} {idle:<12} {load:<15} {icon} {r.status.value}")
    print("─" * 90)
    passed = sum(1 for r in results if r.status == Status.PASS)
    failed = sum(1 for r in results if r.status == Status.FAIL)
    warned = sum(1 for r in results if r.status == Status.WARN)
    print(f"  PASSED: {passed}  FAILED: {failed}  WARNINGS: {warned}  TOTAL: {len(results)}")
    print("═" * 90 + "\n")


def main():
    parser = argparse.ArgumentParser(description="GPU Metric Validator")
    parser.add_argument("--telemetry", default="reports/telemetry.json", help="Path to telemetry JSON")
    parser.add_argument("--config", default="configs/thresholds.yaml", help="Path to thresholds config")
    parser.add_argument("--output", default="reports/validation_report.html", help="Output HTML report path")
    args = parser.parse_args()

    results = run_all_validators(args.telemetry, args.config)
    print_summary_table(results)
    generate_html_report(results, args.output, args.telemetry)
    logger.info("Validation complete. Report: %s", args.output)

    # Exit non-zero if any failures
    if any(r.status == Status.FAIL for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
