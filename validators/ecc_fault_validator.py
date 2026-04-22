#!/usr/bin/env python3
"""
validators/ecc_fault_validator.py

ECC Error Simulation and Validation.

NOTE ON SIMULATION:
Real ECC errors require actual hardware faults (memory cell failures, radiation events).
This script provides:
  1. Detection of existing ECC errors on the system
  2. Validation that ECC is enabled and monitoring is working
  3. Injection methods (where hardware/driver supports it)
  4. Synthetic counter injection into telemetry for validation testing

ECC Error Types:
  - Single-Bit Error (SBE/SECE): Corrected by hardware. Non-fatal.
    Volatile counter resets on driver reload; Aggregate counter persists.
  - Double-Bit Error (DBE/DECE): Uncorrectable. Fatal to the running process.
    Triggers page retirement.

Hardware-level injection (NVIDIA A100/H100):
  - Some Tesla/Ampere GPUs support error injection via nvidia-smi or DCGM diagnostic.
  - This requires special firmware/driver permissions.

Usage:
    python3 validators/ecc_fault_validator.py --check
    python3 validators/ecc_fault_validator.py --inject-synthetic --output reports/telemetry.json
    python3 validators/ecc_fault_validator.py --dcgm-diag  # Requires DCGM diagnostic suite
"""

import os
import sys
import json
import subprocess
import logging
import argparse
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("ecc_validator")


def check_ecc_enabled(gpu_index: int = 0) -> dict:
    """Query ECC mode and current error counts."""
    queries = [
        "ecc.mode.current",
        "ecc.errors.corrected.volatile.total",
        "ecc.errors.uncorrected.volatile.total",
        "ecc.errors.corrected.aggregate.total",
        "ecc.errors.uncorrected.aggregate.total",
        "retired_pages.single_bit_ecc.count",
        "retired_pages.double_bit.count",
        "retired_pages.blacklist",
    ]
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu_index),
         f"--query-gpu={','.join(queries)}",
         "--format=csv,noheader"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        logger.error("nvidia-smi ECC query failed: %s", result.stderr)
        return {}

    parts = [v.strip() for v in result.stdout.strip().split(",")]
    state = dict(zip(queries, parts))
    return state


def check_ecc_support(gpu_index: int = 0) -> bool:
    """Check if GPU supports ECC."""
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu_index), "--query-gpu=ecc.mode.current", "--format=csv,noheader"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return False
    return result.stdout.strip().lower() != "n/a"


def enable_ecc(gpu_index: int = 0) -> bool:
    """Enable ECC mode (requires reboot)."""
    result = subprocess.run(
        ["nvidia-smi", "-i", str(gpu_index), "--ecc-config=1"],
        capture_output=True, text=True
    )
    if result.returncode == 0:
        logger.info("ECC enabled. Reboot required.")
        return True
    logger.error("Failed to enable ECC: %s", result.stderr)
    return False


def try_dcgm_diag_injection(gpu_index: int = 0) -> bool:
    """
    Attempt DCGM diagnostic-based error injection.
    Only works on supported hardware with diagnostic access.
    Requires: dcgmi diag with inject capability.
    """
    logger.info("Attempting DCGM diagnostic injection (GPU %d)...", gpu_index)
    # DCGM error injection (only available on certain Tesla hardware)
    cmd = ["dcgmi", "diag", "-r", "3", "-i", str(gpu_index)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    logger.info("DCGM diag output: %s", result.stdout[:500])
    return result.returncode == 0


def try_nvml_injection() -> bool:
    """
    NVML-based error injection.
    nvmlDeviceInjectEccError is only available on specific hardware.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        # Try inject (will fail on most consumer/VM GPUs)
        try:
            pynvml.nvmlDeviceInjectEccError(handle, pynvml.NVML_SINGLE_BIT_ECC, pynvml.NVML_MEMORY_ERROR_TYPE_CORRECTED)
            logger.info("✅ NVML SBE injection succeeded")
            return True
        except pynvml.NVMLError as e:
            logger.info("NVML injection not supported on this hardware: %s", e)
            return False
    except ImportError:
        logger.warning("pynvml not installed")
        return False


def inject_synthetic_ecc_into_telemetry(telemetry_path: str, gpu_index: int = 0):
    """
    Inject synthetic ECC error counters into telemetry JSON for validation testing.
    This simulates what the telemetry would look like if ECC errors occurred.
    Used to test the validation logic without actual hardware errors.

    The synthetic records simulate:
    - 5 SBE volatile errors (corrected, non-fatal)
    - 0 DBE volatile errors (uncorrectable — this is the desired state)
    - Monotonically increasing aggregate SBE counter
    """
    logger.info("Injecting synthetic ECC counters into %s", telemetry_path)
    path = Path(telemetry_path)
    if path.exists():
        with open(path) as f:
            data = json.load(f)
    else:
        data = []

    # Find the approximate middle of the telemetry to insert ECC errors during load phase
    mid_idx = max(0, len(data) // 3)

    synthetic_records = []
    base_ts = datetime.now(timezone.utc).isoformat()

    for i in range(10):
        # SBE volatile increases by 1 every 2 records (simulates gradual SBE accumulation)
        sbe_vol = i // 2
        record = {
            "type": "synthetic_ecc",
            "gpu_index": gpu_index,
            "collection_source": "synthetic_injection",
            "snapshot_timestamp": base_ts,
            "sbe_volatile": sbe_vol,
            "dbe_volatile": 0,             # Must be 0 — any DBE is hardware failure
            "sbe_aggregate": 5 + sbe_vol,  # Aggregate persists
            "dbe_aggregate": 0,
            "retired_pages_sbe": 0,
            "retired_pages_dbe": 0,
            "ecc.errors.corrected.volatile.total": sbe_vol,
            "ecc.errors.uncorrected.volatile.total": 0,
        }
        synthetic_records.append(record)

    # Insert in the middle of the data to simulate errors during load
    data = data[:mid_idx] + synthetic_records + data[mid_idx:]

    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(
        "Injected %d synthetic ECC records (SBE volatile: 0→%d, DBE: 0)",
        len(synthetic_records),
        synthetic_records[-1]["sbe_volatile"]
    )


def validate_ecc_state(gpu_index: int = 0) -> dict:
    """
    Full ECC validation check:
    1. Confirm ECC is enabled
    2. Read current counters
    3. Report health status
    """
    results = {"gpu_index": gpu_index, "checks": [], "overall": "PASS"}

    # Check ECC support
    if not check_ecc_support(gpu_index):
        results["checks"].append({
            "check": "ECC hardware support",
            "status": "SKIP",
            "detail": "ECC not supported on this GPU (consumer GPU or ECC disabled)"
        })
        results["overall"] = "SKIP"
        return results

    state = check_ecc_enabled(gpu_index)
    logger.info("ECC state: %s", state)

    # ECC mode check
    ecc_mode = state.get("ecc.mode.current", "").lower()
    if ecc_mode == "enabled":
        results["checks"].append({"check": "ECC mode enabled", "status": "PASS", "value": ecc_mode})
    else:
        results["checks"].append({
            "check": "ECC mode enabled", "status": "FAIL", "value": ecc_mode,
            "detail": "ECC is disabled. Enable with: nvidia-smi --ecc-config=1 (requires reboot)"
        })
        results["overall"] = "FAIL"

    # DBE volatile check (must be 0)
    dbe_vol = state.get("ecc.errors.uncorrected.volatile.total", "0")
    try:
        dbe_count = int(dbe_vol)
    except ValueError:
        dbe_count = 0

    if dbe_count == 0:
        results["checks"].append({"check": "DBE volatile = 0", "status": "PASS", "value": 0})
    else:
        results["checks"].append({
            "check": "DBE volatile = 0", "status": "FAIL", "value": dbe_count,
            "detail": f"Non-zero double-bit ECC errors: {dbe_count}. Hardware may be faulty."
        })
        results["overall"] = "FAIL"

    # SBE check (non-zero is normal but should be monitored)
    sbe_vol = state.get("ecc.errors.corrected.volatile.total", "0")
    try:
        sbe_count = int(sbe_vol)
    except ValueError:
        sbe_count = 0

    if sbe_count == 0:
        results["checks"].append({"check": "SBE volatile count", "status": "PASS", "value": 0})
    elif sbe_count < 100:
        results["checks"].append({
            "check": "SBE volatile count", "status": "WARN", "value": sbe_count,
            "detail": "SBE errors present but within normal range. Monitor over time."
        })
    else:
        results["checks"].append({
            "check": "SBE volatile count", "status": "FAIL", "value": sbe_count,
            "detail": f"High SBE count ({sbe_count}) may indicate degrading memory."
        })
        results["overall"] = "FAIL" if results["overall"] != "FAIL" else "FAIL"

    # Retired pages
    retired_dbe = state.get("retired_pages.double_bit.count", "0")
    try:
        retired_dbe_count = int(retired_dbe)
    except ValueError:
        retired_dbe_count = 0

    if retired_dbe_count > 0:
        results["checks"].append({
            "check": "Retired pages (DBE)", "status": "WARN", "value": retired_dbe_count,
            "detail": "Pages retired due to DBE. GPU still functional but degraded."
        })
    else:
        results["checks"].append({"check": "Retired pages (DBE)", "status": "PASS", "value": 0})

    return results


def print_ecc_report(results: dict):
    status_icon = {"PASS": "✅", "FAIL": "❌", "WARN": "⚠️", "SKIP": "⏭️"}
    print("\n" + "═" * 60)
    print(f"  ECC Validation Report — GPU {results['gpu_index']}")
    print("═" * 60)
    for check in results.get("checks", []):
        icon = status_icon.get(check.get("status", "?"), "?")
        val = f" = {check.get('value', '')}" if check.get("value") is not None else ""
        detail = f"\n    → {check.get('detail', '')}" if check.get("detail") else ""
        print(f"  {icon} {check['check']}{val}{detail}")
    print("─" * 60)
    icon = status_icon.get(results.get("overall", "?"), "?")
    print(f"  Overall: {icon} {results.get('overall', '?')}")
    print("═" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="ECC Fault Validation")
    parser.add_argument("--check", action="store_true", help="Check current ECC state")
    parser.add_argument("--inject-synthetic", action="store_true",
                        help="Inject synthetic ECC counters into telemetry for validation testing")
    parser.add_argument("--dcgm-diag", action="store_true", help="Run DCGM diagnostic injection")
    parser.add_argument("--enable-ecc", action="store_true", help="Enable ECC mode (requires reboot)")
    parser.add_argument("--output", default="reports/telemetry.json", help="Telemetry JSON path")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    args = parser.parse_args()

    if args.enable_ecc:
        enable_ecc(args.gpu)
        return

    if args.check or not any([args.inject_synthetic, args.dcgm_diag]):
        results = validate_ecc_state(args.gpu)
        print_ecc_report(results)

    if args.inject_synthetic:
        inject_synthetic_ecc_into_telemetry(args.output, args.gpu)
        logger.info("Synthetic injection complete. Re-run validators to test ECC validation logic.")

    if args.dcgm_diag:
        try_dcgm_diag_injection(args.gpu)


if __name__ == "__main__":
    main()
