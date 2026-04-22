#!/usr/bin/env python3
"""
dashboard_validator.py — Compare Grafana/Prometheus dashboard values against
ground-truth telemetry collected by dcgm_collector.py.

Queries the Prometheus HTTP API for each metric within the workload time window
and computes delta vs the local JSON telemetry, flagging mismatches > tolerance.
"""

import json
import time
import argparse
import sys
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

import requests
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Prometheus metric names emitted by dcgm-exporter
# ---------------------------------------------------------------------------
PROM_METRIC_MAP = {
    "sm_active":              "DCGM_FI_PROF_SM_ACTIVE",
    "sm_occupancy":           "DCGM_FI_PROF_SM_OCCUPANCY",
    "tensor_active":          "DCGM_FI_PROF_PIPE_TENSOR_ACTIVE",
    "fp_active":              "DCGM_FI_PROF_PIPE_FP_ACTIVE",
    "dram_active":            "DCGM_FI_PROF_DRAM_ACTIVE",
    "gpu_util":               "DCGM_FI_DEV_GPU_UTIL",
    "mem_util":               "DCGM_FI_DEV_MEM_COPY_UTIL",
    "mem_used":               "DCGM_FI_DEV_FB_USED",
    "sm_clock":               "DCGM_FI_DEV_SM_CLOCK",
    "mem_clock":              "DCGM_FI_DEV_MEM_CLOCK",
    "gpu_temp":               "DCGM_FI_DEV_GPU_TEMP",
    "mem_temp":               "DCGM_FI_DEV_MEMORY_TEMP",
    "power_usage":            "DCGM_FI_DEV_POWER_USAGE",
    "power_limit":            "DCGM_FI_DEV_POWER_LIMIT",
    "pcie_tx":                "DCGM_FI_PROF_PCIE_TX_BYTES",
    "pcie_rx":                "DCGM_FI_PROF_PCIE_RX_BYTES",
    "ecc_sbe_volatile":       "DCGM_FI_DEV_ECC_SBE_VOL_TOTAL",
    "ecc_dbe_volatile":       "DCGM_FI_DEV_ECC_DBE_VOL_TOTAL",
    "ecc_sbe_aggregate":      "DCGM_FI_DEV_ECC_SBE_AGG_TOTAL",
    "ecc_dbe_aggregate":      "DCGM_FI_DEV_ECC_DBE_AGG_TOTAL",
    "retired_pages_sbe":      "DCGM_FI_DEV_RETIRED_SBE",
    "retired_pages_dbe":      "DCGM_FI_DEV_RETIRED_DBE",
    "pcie_replay":            "DCGM_FI_DEV_PCIE_REPLAY_COUNTER",
}

# Tolerance as a fraction of the ground-truth value (5% default)
DEFAULT_TOLERANCE = 0.05
# Absolute tolerance for near-zero values
ABS_TOLERANCE = 1.0


class PrometheusClient:
    """Minimal Prometheus HTTP API client."""

    def __init__(self, base_url: str = "http://localhost:9090"):
        self.base_url = base_url.rstrip("/")

    def query_range(
        self,
        metric: str,
        start: float,
        end: float,
        step: str = "5s",
        gpu: Optional[str] = None,
    ) -> pd.Series:
        """
        Execute a range query and return a pandas Series indexed by timestamp.
        If gpu is specified, filter by gpu label (e.g. "0").
        """
        if gpu is not None:
            promql = f'{metric}{{gpu="{gpu}"}}'
        else:
            promql = metric

        params = {
            "query": promql,
            "start": start,
            "end": end,
            "step": step,
        }
        url = f"{self.base_url}/api/v1/query_range"
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()

        data = resp.json()
        if data["status"] != "success":
            raise RuntimeError(f"Prometheus query failed: {data.get('error', 'unknown')}")

        results = data["data"]["result"]
        if not results:
            return pd.Series(dtype=float, name=metric)

        # If multiple series (multiple GPUs) take first matching or first overall
        series_data = results[0]["values"]  # list of [timestamp, value_str]
        ts = [float(v[0]) for v in series_data]
        vals = [float(v[1]) for v in series_data]
        return pd.Series(vals, index=ts, name=metric)

    def is_reachable(self) -> bool:
        try:
            resp = requests.get(f"{self.base_url}/-/healthy", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False


class TelemetryLoader:
    """Load ground-truth telemetry from JSON produced by dcgm_collector.py."""

    def __init__(self, telemetry_path: str):
        with open(telemetry_path) as f:
            self.raw = json.load(f)
        self.records = self.raw if isinstance(self.raw, list) else self.raw.get("records", [])
        self._df: Optional[pd.DataFrame] = None

    @property
    def df(self) -> pd.DataFrame:
        if self._df is None:
            self._df = pd.DataFrame(self.records)
            if "timestamp" in self._df.columns:
                self._df["timestamp"] = pd.to_datetime(self._df["timestamp"])
                self._df = self._df.set_index("timestamp").sort_index()
        return self._df

    def window(self, start_ts: float, end_ts: float) -> pd.DataFrame:
        start = pd.Timestamp(start_ts, unit="s", tz="UTC")
        end   = pd.Timestamp(end_ts,   unit="s", tz="UTC")
        mask  = (self.df.index >= start) & (self.df.index <= end)
        return self.df.loc[mask]

    def annotations(self) -> list[dict]:
        return self.raw.get("annotations", []) if isinstance(self.raw, dict) else []

    def workload_window(self, workload_name: str) -> tuple[Optional[float], Optional[float]]:
        """Return (start_unix, end_unix) for a named workload annotation."""
        for ann in self.annotations():
            if ann.get("workload") == workload_name and ann.get("event") == "start":
                start = ann["timestamp"]
                # Find matching end
                for ann2 in self.annotations():
                    if ann2.get("workload") == workload_name and ann2.get("event") == "end":
                        return start, ann2["timestamp"]
                return start, None
        return None, None


# ---------------------------------------------------------------------------
# Metric comparison helpers
# ---------------------------------------------------------------------------

def compare_series(
    ground_truth: pd.Series,
    prometheus: pd.Series,
    tolerance: float = DEFAULT_TOLERANCE,
    label: str = "",
) -> dict:
    """
    Resample both series to a common 5s grid and compare mean/p95.
    Returns a result dict with status PASS/WARN/FAIL.
    """
    if ground_truth.empty and prometheus.empty:
        return _result(label, "SKIP", "Both series empty", {})

    if ground_truth.empty:
        return _result(label, "SKIP", "No ground-truth data", {})

    gt_mean  = float(ground_truth.mean())
    gt_p95   = float(np.percentile(ground_truth, 95))

    if prometheus.empty:
        return _result(label, "WARN", "No Prometheus data (is dcgm-exporter running?)",
                       {"gt_mean": gt_mean, "gt_p95": gt_p95})

    pm_mean  = float(prometheus.mean())
    pm_p95   = float(np.percentile(prometheus, 95))

    # Compute relative delta (guard near-zero denominators)
    denom_mean = max(abs(gt_mean), ABS_TOLERANCE)
    denom_p95  = max(abs(gt_p95),  ABS_TOLERANCE)

    delta_mean = abs(gt_mean - pm_mean) / denom_mean
    delta_p95  = abs(gt_p95  - pm_p95)  / denom_p95

    stats = {
        "gt_mean":    round(gt_mean,  4),
        "pm_mean":    round(pm_mean,  4),
        "delta_mean": round(delta_mean, 4),
        "gt_p95":     round(gt_p95,   4),
        "pm_p95":     round(pm_p95,   4),
        "delta_p95":  round(delta_p95, 4),
    }

    if delta_mean > tolerance or delta_p95 > tolerance:
        status = "FAIL"
        msg = (f"Mean delta {delta_mean*100:.1f}% > tolerance {tolerance*100:.0f}% "
               f"or P95 delta {delta_p95*100:.1f}%")
    else:
        status = "PASS"
        msg = f"Mean delta {delta_mean*100:.1f}%, P95 delta {delta_p95*100:.1f}%"

    return _result(label, status, msg, stats)


def _result(label, status, message, stats):
    return {
        "metric":  label,
        "status":  status,
        "message": message,
        **stats,
    }


# ---------------------------------------------------------------------------
# Ground-truth column → Prometheus metric mapping
# ---------------------------------------------------------------------------
GROUND_TRUTH_COL_MAP = {
    # ground_truth_col          prom_metric_key    scale_factor
    "gpu_utilization":         ("sm_active",         1.0),
    "sm_occupancy":            ("sm_occupancy",      1.0),
    "tensor_core_util":        ("tensor_active",     1.0),
    "fp_activity":             ("fp_active",         1.0),
    "memory_utilization":      ("dram_active",       1.0),
    "memory_used_mb":          ("mem_used",          1.0),   # both in MiB
    "sm_clock_mhz":            ("sm_clock",          1.0),
    "gpu_temp_c":              ("gpu_temp",          1.0),
    "memory_temp_c":           ("mem_temp",          1.0),
    "power_draw_w":            ("power_usage",       1.0),
    "pcie_tx_mb":              ("pcie_tx",           1.0),
    "pcie_rx_mb":              ("pcie_rx",           1.0),
    "ecc_sbe_volatile":        ("ecc_sbe_volatile",  1.0),
    "ecc_dbe_volatile":        ("ecc_dbe_volatile",  1.0),
    "ecc_sbe_aggregate":       ("ecc_sbe_aggregate", 1.0),
    "ecc_dbe_aggregate":       ("ecc_dbe_aggregate", 1.0),
    "retired_pages_sbe":       ("retired_pages_sbe", 1.0),
    "retired_pages_dbe":       ("retired_pages_dbe", 1.0),
    "pcie_replay_counter":     ("pcie_replay",       1.0),
}


def run_dashboard_validation(
    telemetry_path: str,
    prometheus_url: str,
    gpu: str = "0",
    tolerance: float = DEFAULT_TOLERANCE,
    workload: Optional[str] = None,
    step: str = "5s",
) -> list[dict]:
    """
    Main entry point: load telemetry, query Prometheus, compare each metric.
    Returns list of result dicts.
    """
    loader = TelemetryLoader(telemetry_path)
    prom   = PrometheusClient(prometheus_url)

    if not prom.is_reachable():
        print(f"[WARN] Prometheus at {prometheus_url} is not reachable. "
              "Dashboard comparison will be skipped.")
        return []

    # Determine time window
    if workload:
        start_ts, end_ts = loader.workload_window(workload)
        if start_ts is None:
            print(f"[WARN] Workload '{workload}' not found in annotations.")
            start_ts = loader.df.index[0].timestamp()
            end_ts   = loader.df.index[-1].timestamp()
    else:
        idx = loader.df.index
        start_ts = idx[0].timestamp()
        end_ts   = idx[-1].timestamp()

    if end_ts is None:
        end_ts = time.time()

    print(f"\n{'='*60}")
    print(f"  Dashboard Validation")
    print(f"  Window : {datetime.fromtimestamp(start_ts)} → {datetime.fromtimestamp(end_ts)}")
    print(f"  GPU    : {gpu}")
    print(f"  Tol    : ±{tolerance*100:.0f}%")
    print(f"{'='*60}\n")

    gt_df   = loader.window(start_ts, end_ts)
    results = []

    for gt_col, (prom_key, scale) in GROUND_TRUTH_COL_MAP.items():
        if gt_col not in gt_df.columns:
            continue

        prom_metric = PROM_METRIC_MAP.get(prom_key)
        if prom_metric is None:
            continue

        # Ground-truth series (epoch float index)
        gt_series = gt_df[gt_col].dropna().astype(float) * scale

        # Prometheus series
        try:
            pm_series = prom.query_range(prom_metric, start_ts, end_ts, step=step, gpu=gpu)
        except Exception as exc:
            results.append(_result(gt_col, "ERROR", str(exc), {}))
            continue

        result = compare_series(gt_series, pm_series, tolerance=tolerance, label=gt_col)
        results.append(result)

        status_sym = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠", "SKIP": "—", "ERROR": "✗"}.get(
            result["status"], "?"
        )
        print(
            f"  [{status_sym}] {gt_col:<30s}  "
            f"GT={result.get('gt_mean','N/A'):<10}  "
            f"PM={result.get('pm_mean','N/A'):<10}  "
            f"Δ={result.get('delta_mean','N/A')}"
        )

    return results


def generate_report(results: list[dict], output_path: str = "dashboard_validation_report.json"):
    counts = {"PASS": 0, "FAIL": 0, "WARN": 0, "SKIP": 0, "ERROR": 0}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    report = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": counts,
        "overall": "PASS" if counts["FAIL"] == 0 and counts["ERROR"] == 0 else "FAIL",
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Summary: PASS={counts['PASS']}  FAIL={counts['FAIL']}  "
          f"WARN={counts['WARN']}  SKIP={counts['SKIP']}")
    print(f"  Overall: {report['overall']}")
    print(f"  Report : {output_path}")
    print(f"{'='*60}")

    return report["overall"] == "PASS"


def main():
    parser = argparse.ArgumentParser(
        description="Validate dashboard metric values against ground-truth telemetry"
    )
    parser.add_argument("--telemetry",  required=True, help="Path to telemetry JSON file")
    parser.add_argument("--prometheus", default="http://localhost:9090",
                        help="Prometheus base URL")
    parser.add_argument("--gpu",        default="0",   help="GPU index label in Prometheus")
    parser.add_argument("--tolerance",  type=float, default=0.05,
                        help="Relative tolerance fraction (default 0.05 = 5%%)")
    parser.add_argument("--workload",   default=None,
                        help="Restrict to a named workload window from annotations")
    parser.add_argument("--step",       default="5s",
                        help="Prometheus range query step (default 5s)")
    parser.add_argument("--output",     default="dashboard_validation_report.json",
                        help="Output JSON report path")
    args = parser.parse_args()

    results = run_dashboard_validation(
        telemetry_path=args.telemetry,
        prometheus_url=args.prometheus,
        gpu=args.gpu,
        tolerance=args.tolerance,
        workload=args.workload,
        step=args.step,
    )

    passed = generate_report(results, args.output)
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
