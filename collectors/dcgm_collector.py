#!/usr/bin/env python3
"""
collectors/dcgm_collector.py

High-frequency GPU telemetry collector using DCGM and nvidia-smi.
Polls at configurable intervals (default 1s) and writes to JSON/CSV.

Usage:
    python3 collectors/dcgm_collector.py --output reports/telemetry.json --interval 1
"""

import os
import sys
import json
import csv
import time
import signal
import logging
import argparse
import subprocess
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    PYNVML_AVAILABLE = False

try:
    import dcgm_fields
    import pydcgm
    import dcgm_structs
    DCGM_AVAILABLE = True
except ImportError:
    DCGM_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S"
)
logger = logging.getLogger("dcgm_collector")


# ─── DCGM Field IDs we care about ───────────────────────────────────────────
DCGM_FIELDS = {
    # ECC
    "dbe_volatile":       203,   # DCGM_FI_DEV_ECC_DBE_VOL_TOTAL
    "sbe_volatile":       202,   # DCGM_FI_DEV_ECC_SBE_VOL_TOTAL
    "sbe_aggregate":      204,   # DCGM_FI_DEV_ECC_SBE_AGG_TOTAL
    "dbe_aggregate":      205,   # DCGM_FI_DEV_ECC_DBE_AGG_TOTAL
    "retired_pages_sbe":  390,   # DCGM_FI_DEV_RETIRED_SBE
    "retired_pages_dbe":  391,   # DCGM_FI_DEV_RETIRED_DBE
    # Utilization
    "gpu_util":           203,   # DCGM_FI_DEV_GPU_UTIL (overloaded per dcgm ver)
    "mem_copy_util":      204,   # DCGM_FI_DEV_MEM_COPY_UTIL
    "sm_occupancy":       1003,  # DCGM_FI_PROF_SM_OCCUPANCY
    "sm_active":          1002,  # DCGM_FI_PROF_SM_ACTIVE
    "pipe_tensor_active": 1004,  # DCGM_FI_PROF_PIPE_TENSOR_ACTIVE
    "pipe_fp32_active":   1005,  # DCGM_FI_PROF_PIPE_FP32_ACTIVE
    "pipe_fp64_active":   1006,  # DCGM_FI_PROF_PIPE_FP64_ACTIVE
    "pipe_fp16_active":   1007,  # DCGM_FI_PROF_PIPE_FP16_ACTIVE
    # Memory
    "fb_used":            252,   # DCGM_FI_DEV_FB_USED (MiB)
    "fb_free":            253,   # DCGM_FI_DEV_FB_FREE
    # Clocks
    "sm_clock":           100,   # DCGM_FI_DEV_SM_CLOCK
    "mem_clock":          101,   # DCGM_FI_DEV_MEM_CLOCK
    # Temperature
    "gpu_temp":           150,   # DCGM_FI_DEV_GPU_TEMP
    "mem_temp":           151,   # DCGM_FI_DEV_MEMORY_TEMP
    # Power
    "power_usage":        155,   # DCGM_FI_DEV_POWER_USAGE
    "power_limit":        156,   # DCGM_FI_DEV_POWER_MGMT_LIMIT
    "power_violation":    240,   # DCGM_FI_DEV_POWER_VIOLATION
    "thermal_violation":  241,   # DCGM_FI_DEV_THERMAL_VIOLATION
    # PCIe
    "pcie_tx_bytes":      1009,  # DCGM_FI_PROF_PCIE_TX_BYTES
    "pcie_rx_bytes":      1010,  # DCGM_FI_PROF_PCIE_RX_BYTES
    "pcie_replay":        206,   # DCGM_FI_DEV_PCIE_REPLAY_COUNTER
    # NVLink
    "nvlink_tx":          1011,  # DCGM_FI_PROF_NVLINK_TX_BYTES
    "nvlink_rx":          1012,  # DCGM_FI_PROF_NVLINK_RX_BYTES
}

# nvidia-smi queries (parallel collection)
NVIDIASMI_QUERIES = [
    "timestamp",
    "gpu_uuid",
    "name",
    "index",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.free",
    "memory.total",
    "temperature.gpu",
    "temperature.memory",
    "power.draw",
    "power.limit",
    "clocks.current.sm",
    "clocks.current.memory",
    "clocks_throttle_reasons.hw_power_brake_slowdown",
    "clocks_throttle_reasons.hw_thermal_slowdown",
    "clocks_throttle_reasons.sw_thermal_slowdown",
    "ecc.errors.corrected.volatile.total",
    "ecc.errors.uncorrected.volatile.total",
    "ecc.errors.corrected.aggregate.total",
    "ecc.errors.uncorrected.aggregate.total",
    "retired_pages.single_bit_ecc.count",
    "retired_pages.double_bit.count",
    "pcie.link.gen.current",
    "pcie.link.width.current",
]


class NvidiaSmiCollector:
    """Collect GPU metrics via nvidia-smi."""

    def __init__(self, gpu_indices: List[int]):
        self.gpu_indices = gpu_indices
        self._validate()

    def _validate(self):
        result = subprocess.run(["nvidia-smi", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("nvidia-smi not found. Install NVIDIA drivers.")
        logger.info("nvidia-smi available.")

    def collect(self) -> List[Dict[str, Any]]:
        """Run nvidia-smi and return parsed metrics for each GPU."""
        query = ",".join(NVIDIASMI_QUERIES)
        cmd = [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
        ]
        if self.gpu_indices:
            gpu_list = ",".join(str(i) for i in self.gpu_indices)
            cmd.extend(["-i", gpu_list])

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logger.error("nvidia-smi failed: %s", result.stderr)
            return []

        records = []
        for line in result.stdout.strip().splitlines():
            values = [v.strip() for v in line.split(",")]
            if len(values) != len(NVIDIASMI_QUERIES):
                logger.warning("Unexpected nvidia-smi output: %s", line)
                continue
            record = dict(zip(NVIDIASMI_QUERIES, values))
            record["collection_source"] = "nvidia-smi"
            record["collected_at"] = datetime.now(timezone.utc).isoformat()
            # Cast numeric fields
            for field in ["utilization.gpu", "utilization.memory",
                          "memory.used", "memory.free", "memory.total",
                          "temperature.gpu", "power.draw", "power.limit",
                          "clocks.current.sm", "clocks.current.memory"]:
                try:
                    record[field] = float(record[field])
                except (ValueError, TypeError):
                    record[field] = None
            records.append(record)
        return records


class DcgmCollector:
    """Collect GPU metrics via dcgmi CLI (fallback when Python bindings unavailable)."""

    def __init__(self, gpu_indices: List[int]):
        self.gpu_indices = gpu_indices
        self._validate()

    def _validate(self):
        result = subprocess.run(["dcgmi", "discovery", "-l"], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError("dcgmi not found. Install DCGM: sudo apt-get install datacenter-gpu-manager")
        logger.info("dcgmi available.")

    def collect_field(self, gpu_id: int, field_id: int) -> Optional[float]:
        """Read a single DCGM field for a GPU."""
        cmd = ["dcgmi", "dmon", "-e", str(field_id), "-d", "1", "-c", "1", "-i", str(gpu_id)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            return None
        # Parse last data line
        for line in reversed(result.stdout.splitlines()):
            parts = line.split()
            if parts and parts[0].isdigit():
                try:
                    return float(parts[-1])
                except ValueError:
                    pass
        return None

    def collect(self) -> List[Dict[str, Any]]:
        """Collect all DCGM fields for all GPUs."""
        records = []
        for gpu_id in self.gpu_indices:
            record = {
                "gpu_index": gpu_id,
                "collection_source": "dcgm",
                "collected_at": datetime.now(timezone.utc).isoformat(),
            }
            for name, field_id in DCGM_FIELDS.items():
                record[name] = self.collect_field(gpu_id, field_id)
            records.append(record)
        return records


class TelemetryCollector:
    """
    Orchestrates high-frequency collection from nvidia-smi and DCGM.
    Writes time-synchronized records to JSON and CSV.
    """

    def __init__(
        self,
        output_path: str,
        interval_sec: float = 1.0,
        gpu_indices: Optional[List[int]] = None,
        use_dcgm: bool = True,
    ):
        self.output_path = Path(output_path)
        self.interval_sec = interval_sec
        self.gpu_indices = gpu_indices or self._detect_gpus()
        self.use_dcgm = use_dcgm
        self.records: List[Dict] = []
        self._running = False
        self._lock = threading.Lock()

        self.smi_collector = NvidiaSmiCollector(self.gpu_indices)
        if use_dcgm:
            try:
                self.dcgm_collector = DcgmCollector(self.gpu_indices)
            except RuntimeError as e:
                logger.warning("DCGM unavailable: %s. Falling back to nvidia-smi only.", e)
                self.dcgm_collector = None
                self.use_dcgm = False
        else:
            self.dcgm_collector = None

        logger.info(
            "TelemetryCollector initialized: GPUs=%s interval=%.1fs dcgm=%s",
            self.gpu_indices, interval_sec, self.use_dcgm
        )

    def _detect_gpus(self) -> List[int]:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            return [0]
        return [int(i.strip()) for i in result.stdout.splitlines() if i.strip().isdigit()]

    def _collect_once(self) -> List[Dict]:
        """Collect a single snapshot from all sources."""
        snapshot = []
        timestamp = datetime.now(timezone.utc).isoformat()

        # nvidia-smi records
        smi_records = self.smi_collector.collect()
        for r in smi_records:
            r["snapshot_timestamp"] = timestamp
            snapshot.append(r)

        # DCGM records (merge with smi by gpu_index)
        if self.use_dcgm and self.dcgm_collector:
            dcgm_records = self.dcgm_collector.collect()
            for dr in dcgm_records:
                dr["snapshot_timestamp"] = timestamp
                snapshot.append(dr)

        return snapshot

    def collect_loop(self):
        """Background collection loop."""
        logger.info("Starting collection loop (interval=%.1fs)", self.interval_sec)
        while self._running:
            start = time.monotonic()
            try:
                batch = self._collect_once()
                with self._lock:
                    self.records.extend(batch)
            except Exception as e:
                logger.error("Collection error: %s", e)
            elapsed = time.monotonic() - start
            sleep_time = max(0.0, self.interval_sec - elapsed)
            time.sleep(sleep_time)

    def start(self):
        """Start background collection thread."""
        self._running = True
        self._thread = threading.Thread(target=self.collect_loop, daemon=True)
        self._thread.start()
        logger.info("Collection started.")

    def stop(self):
        """Stop collection and flush to disk."""
        self._running = False
        if hasattr(self, "_thread"):
            self._thread.join(timeout=10)
        self.flush()
        logger.info("Collection stopped. %d records written to %s", len(self.records), self.output_path)

    def flush(self):
        """Write all collected records to JSON and CSV."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # JSON output
        with open(self.output_path, "w") as f:
            json.dump(self.records, f, indent=2, default=str)

        # CSV output
        csv_path = self.output_path.with_suffix(".csv")
        if self.records:
            fieldnames = sorted(set().union(*[r.keys() for r in self.records]))
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.records)
        logger.info("Flushed to %s and %s", self.output_path, csv_path)

    def annotate_workload(self, workload_name: str, event: str):
        """Add a workload start/stop annotation to the stream."""
        annotation = {
            "type": "workload_annotation",
            "workload": workload_name,
            "event": event,
            "snapshot_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            self.records.append(annotation)
        logger.info("Annotated: %s %s", workload_name, event)


def main():
    parser = argparse.ArgumentParser(description="GPU Telemetry Collector")
    parser.add_argument("--output", default="reports/telemetry.json", help="Output JSON path")
    parser.add_argument("--interval", type=float, default=1.0, help="Poll interval in seconds")
    parser.add_argument("--gpu", type=int, nargs="*", help="GPU indices to monitor (default: all)")
    parser.add_argument("--no-dcgm", action="store_true", help="Skip DCGM, use nvidia-smi only")
    parser.add_argument("--duration", type=int, default=0, help="Collect for N seconds (0=until Ctrl+C)")
    args = parser.parse_args()

    collector = TelemetryCollector(
        output_path=args.output,
        interval_sec=args.interval,
        gpu_indices=args.gpu,
        use_dcgm=not args.no_dcgm,
    )

    # Handle signals for clean shutdown
    def _signal_handler(sig, frame):
        logger.info("Received signal %s, stopping...", sig)
        collector.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    collector.start()

    if args.duration > 0:
        logger.info("Collecting for %d seconds...", args.duration)
        time.sleep(args.duration)
        collector.stop()
    else:
        logger.info("Collecting indefinitely. Press Ctrl+C to stop.")
        signal.pause()


if __name__ == "__main__":
    main()
