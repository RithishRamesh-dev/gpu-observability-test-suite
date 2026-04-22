#!/usr/bin/env python3
"""
scripts/orchestrator.py

Main orchestration script. Runs all workloads sequentially (or selectively),
with telemetry annotation for each workload phase.

Usage:
    python3 scripts/orchestrator.py --config configs/workload_config.yaml
    python3 scripts/orchestrator.py --workloads cuda_matmul,memory_stress,pcie_stress
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
import threading
import signal
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("orchestrator")


class Orchestrator:
    def __init__(self, config_path: str, telemetry_path: str = "reports/telemetry.json"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        self.telemetry_path = Path(telemetry_path)
        self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
        self.results = {}
        self._collector_process = None
        self.gpu = self.config.get("global", {}).get("gpu_index", 0)

    # ─── Telemetry Control ──────────────────────────────────────────

    def start_collector(self):
        """Start the telemetry collector as a subprocess."""
        cmd = [
            sys.executable,
            "collectors/dcgm_collector.py",
            "--output", str(self.telemetry_path),
            "--interval", str(self.config.get("global", {}).get("telemetry_interval_sec", 1)),
            "--gpu", str(self.gpu),
        ]
        logger.info("Starting telemetry collector: %s", " ".join(cmd))
        self._collector_process = subprocess.Popen(cmd)
        time.sleep(3)  # Let collector warm up

    def stop_collector(self):
        """Stop the telemetry collector."""
        if self._collector_process:
            self._collector_process.terminate()
            try:
                self._collector_process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self._collector_process.kill()
            logger.info("Telemetry collector stopped.")

    def annotate(self, workload: str, event: str):
        """Append a workload annotation to the telemetry file."""
        annotation = {
            "type": "workload_annotation",
            "workload": workload,
            "event": event,
            "snapshot_timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            if self.telemetry_path.exists():
                with open(self.telemetry_path) as f:
                    data = json.load(f)
            else:
                data = []
            data.append(annotation)
            with open(self.telemetry_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning("Failed to write annotation: %s", e)

    # ─── Workload Runners ───────────────────────────────────────────

    def run_workload(self, name: str, cmd: list, timeout: int = 7200) -> bool:
        """Run a workload subprocess with start/stop annotations."""
        logger.info("=" * 60)
        logger.info("Starting workload: %s", name)
        logger.info("Command: %s", " ".join(str(c) for c in cmd))
        self.annotate(name, "start")
        start = time.monotonic()
        try:
            result = subprocess.run(cmd, timeout=timeout)
            elapsed = time.monotonic() - start
            success = result.returncode == 0
            self.annotate(name, "stop")
            self.results[name] = {
                "success": success,
                "return_code": result.returncode,
                "duration_sec": elapsed,
            }
            if success:
                logger.info("✅ Workload %s completed in %.1fs", name, elapsed)
            else:
                logger.warning("⚠️ Workload %s exited with code %d", name, result.returncode)
            return success
        except subprocess.TimeoutExpired:
            logger.error("❌ Workload %s timed out after %ds", name, timeout)
            self.annotate(name, "stop")
            self.results[name] = {"success": False, "error": "timeout", "duration_sec": timeout}
            return False
        except FileNotFoundError as e:
            logger.error("❌ Workload %s: command not found: %s", name, e)
            self.annotate(name, "stop")
            self.results[name] = {"success": False, "error": str(e)}
            return False
        except KeyboardInterrupt:
            logger.info("Workload %s interrupted.", name)
            self.annotate(name, "stop")
            raise

    def run_cuda_matmul_fp16(self):
        cfg = self.config.get("workloads", {}).get("cuda_matmul", {})
        if not cfg.get("enabled", True):
            return
        dtype_cfg = cfg.get("dtype_fp16", {})
        if not dtype_cfg.get("enabled", True):
            return
        duration = dtype_cfg.get("duration_sec", 300)
        size = cfg.get("matrix_size", 8192)
        self.run_workload(
            "cuda_matmul_fp16",
            [sys.executable, "workload/cuda/matmul_stress.py",
             "--dtype", "fp16", "--duration", str(duration),
             "--matrix-size", str(size), "--gpu", str(self.gpu)],
            timeout=duration + 120,
        )

    def run_cuda_matmul_fp32(self):
        cfg = self.config.get("workloads", {}).get("cuda_matmul", {})
        if not cfg.get("enabled", True):
            return
        dtype_cfg = cfg.get("dtype_fp32", {})
        if not dtype_cfg.get("enabled", True):
            return
        duration = dtype_cfg.get("duration_sec", 300)
        size = cfg.get("matrix_size", 8192)
        self.run_workload(
            "cuda_matmul_fp32",
            [sys.executable, "workload/cuda/matmul_stress.py",
             "--dtype", "fp32", "--duration", str(duration),
             "--matrix-size", str(size), "--gpu", str(self.gpu)],
            timeout=duration + 120,
        )

    def run_memory_stress(self):
        cfg = self.config.get("workloads", {}).get("memory_stress", {})
        if not cfg.get("enabled", True):
            return
        fill = cfg.get("fill_percent", 90)
        duration = cfg.get("rapid_alloc", {}).get("duration_sec", 300)
        self.run_workload(
            "memory_stress",
            [sys.executable, "workload/memory/memory_stress.py",
             "--mode", "all", "--fill-percent", str(fill),
             "--duration", str(duration), "--gpu", str(self.gpu)],
            timeout=duration + 600,
        )

    def run_pcie_stress(self):
        cfg = self.config.get("workloads", {}).get("pcie_stress", {})
        if not cfg.get("enabled", True):
            return
        buf_gb = cfg.get("buffer_size_gb", 1)
        duration = cfg.get("duration_sec", 300)
        direction = cfg.get("direction", "both")
        streams = cfg.get("num_streams", 4)
        self.run_workload(
            "pcie_stress",
            [sys.executable, "workload/pcie/pcie_stress.py",
             "--buffer-size-gb", str(buf_gb),
             "--duration", str(duration),
             "--direction", direction,
             "--num-streams", str(streams),
             "--gpu", str(self.gpu)],
            timeout=duration + 120,
        )

    def run_thermal_stress(self):
        cfg = self.config.get("workloads", {}).get("thermal_stress", {})
        if not cfg.get("enabled", True):
            return
        duration = cfg.get("duration_sec", 900)
        self.run_workload(
            "thermal_stress",
            [sys.executable, "workload/thermal/thermal_stress.py",
             "--duration", str(duration),
             "--monitor",
             "--gpu", str(self.gpu)],
            timeout=duration + 300,
        )

    def run_llm_inference(self):
        cfg = self.config.get("workloads", {}).get("llm_inference", {})
        if not cfg.get("enabled", True):
            return
        model = cfg.get("model", "facebook/opt-125m")
        phases = cfg.get("phases", [])
        duration = next((p.get("duration_sec", 600) for p in phases if p.get("name") == "sustained_load"), 600)
        self.run_workload(
            "llm_inference",
            [sys.executable, "workload/llm/inference_workload.py",
             "--model", model,
             "--launch-server",
             "--gpu", str(self.gpu),
             "--sustained-duration", str(duration),
             "--phase", "all"],
            timeout=duration + 1800,
        )

    # ─── Main Run ───────────────────────────────────────────────────

    def run(self, workloads: list = None):
        """Run all (or specified) workloads with telemetry collection."""
        all_workloads = [
            "cuda_matmul_fp16",
            "cuda_matmul_fp32",
            "memory_stress",
            "pcie_stress",
            "thermal_stress",
            "llm_inference",
        ]
        to_run = workloads if workloads else all_workloads

        logger.info("Starting GPU Observability Test Suite")
        logger.info("Workloads to run: %s", to_run)
        logger.info("Telemetry output: %s", self.telemetry_path)

        self.start_collector()
        # Collect 30s baseline idle
        logger.info("Collecting 30s baseline idle metrics...")
        time.sleep(30)

        try:
            for wl in to_run:
                runner = {
                    "cuda_matmul_fp16": self.run_cuda_matmul_fp16,
                    "cuda_matmul_fp32": self.run_cuda_matmul_fp32,
                    "memory_stress": self.run_memory_stress,
                    "pcie_stress": self.run_pcie_stress,
                    "thermal_stress": self.run_thermal_stress,
                    "llm_inference": self.run_llm_inference,
                }.get(wl)

                if runner:
                    runner()
                    # Cool-down between workloads
                    logger.info("Cooling down 30s between workloads...")
                    time.sleep(30)
                else:
                    logger.warning("Unknown workload: %s", wl)

        except KeyboardInterrupt:
            logger.info("Orchestration interrupted by user.")
        finally:
            # Collect 30s post-workload
            logger.info("Collecting 30s post-workload metrics...")
            time.sleep(30)
            self.stop_collector()

        # Write results summary
        summary_path = Path("reports/orchestration_summary.json")
        with open(summary_path, "w") as f:
            json.dump({
                "run_at": datetime.now(timezone.utc).isoformat(),
                "workloads": self.results,
                "telemetry_path": str(self.telemetry_path),
            }, f, indent=2)
        logger.info("Orchestration complete. Summary: %s", summary_path)

        # Print table
        print("\n" + "=" * 60)
        print("  ORCHESTRATION SUMMARY")
        print("=" * 60)
        for wl, res in self.results.items():
            status = "✅ PASS" if res.get("success") else "❌ FAIL"
            dur = f"{res.get('duration_sec', 0):.0f}s"
            print(f"  {wl:<30} {status}  ({dur})")
        print("=" * 60)
        return self.results


def main():
    parser = argparse.ArgumentParser(description="GPU Workload Orchestrator")
    parser.add_argument("--config", default="configs/workload_config.yaml", help="Workload config YAML")
    parser.add_argument("--telemetry", default="reports/telemetry.json", help="Telemetry output path")
    parser.add_argument(
        "--workloads",
        default=None,
        help="Comma-separated workload names to run (default: all). "
             "Options: cuda_matmul_fp16,cuda_matmul_fp32,memory_stress,pcie_stress,thermal_stress,llm_inference"
    )
    args = parser.parse_args()

    workloads = args.workloads.split(",") if args.workloads else None
    orch = Orchestrator(args.config, args.telemetry)
    results = orch.run(workloads)
    failures = [k for k, v in results.items() if not v.get("success")]
    if failures:
        logger.warning("Failed workloads: %s", failures)
        sys.exit(1)


if __name__ == "__main__":
    main()
