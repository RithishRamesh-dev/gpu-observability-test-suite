#!/usr/bin/env python3
"""
workload/thermal/thermal_stress.py

Thermal and power stress workload.
Sustains maximum compute to push GPU to thermal and power throttle thresholds.
Monitors throttle state in real-time via nvidia-smi.

Usage:
    python3 workload/thermal/thermal_stress.py --duration 900 --monitor
"""

import sys
import time
import subprocess
import threading
import logging
import argparse
from datetime import datetime, timezone

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("thermal_stress")


def get_throttle_state(gpu_index: int = 0) -> dict:
    """Query current throttle reasons and power/temp from nvidia-smi."""
    queries = [
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "clocks.current.sm",
        "clocks_throttle_reasons.hw_power_brake_slowdown",
        "clocks_throttle_reasons.hw_thermal_slowdown",
        "clocks_throttle_reasons.sw_thermal_slowdown",
        "clocks_throttle_reasons.sw_power_cap",
        "clocks_throttle_reasons.gpu_idle",
    ]
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i", str(gpu_index),
            f"--query-gpu={','.join(queries)}",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        return {}
    parts = [v.strip() for v in result.stdout.strip().split(",")]
    if len(parts) < len(queries):
        return {}
    state = dict(zip(queries, parts))
    # Parse numeric
    for k in ["temperature.gpu", "power.draw", "power.limit", "clocks.current.sm"]:
        try:
            state[k] = float(state[k])
        except (ValueError, TypeError):
            state[k] = None
    state["timestamp"] = datetime.now(timezone.utc).isoformat()
    return state


def monitor_thermal(gpu_index: int, stop_event: threading.Event, interval_sec: float = 2.0):
    """Background thread: log thermal/power/throttle state every interval."""
    throttle_events = []
    while not stop_event.is_set():
        state = get_throttle_state(gpu_index)
        if state:
            power_pct = 0
            if state.get("power.draw") and state.get("power.limit"):
                power_pct = (state["power.draw"] / state["power.limit"]) * 100

            hw_power_throttle = state.get("clocks_throttle_reasons.hw_power_brake_slowdown", "0") != "0"
            hw_thermal_throttle = state.get("clocks_throttle_reasons.hw_thermal_slowdown", "0") != "0"
            sw_thermal_throttle = state.get("clocks_throttle_reasons.sw_thermal_slowdown", "0") != "0"
            sw_power_cap = state.get("clocks_throttle_reasons.sw_power_cap", "0") != "0"

            throttling = any([hw_power_throttle, hw_thermal_throttle, sw_thermal_throttle, sw_power_cap])
            if throttling:
                reasons = []
                if hw_power_throttle:
                    reasons.append("HW_POWER")
                if hw_thermal_throttle:
                    reasons.append("HW_THERMAL")
                if sw_thermal_throttle:
                    reasons.append("SW_THERMAL")
                if sw_power_cap:
                    reasons.append("SW_POWER_CAP")
                throttle_events.append({"timestamp": state["timestamp"], "reasons": reasons})

            logger.info(
                "Temp=%s°C Power=%.1fW (%.1f%%) SMClock=%sMHz %s",
                state.get("temperature.gpu", "?"),
                state.get("power.draw", 0) or 0,
                power_pct,
                state.get("clocks.current.sm", "?"),
                "⚠️ THROTTLING[" + ",".join(reasons if throttling else []) + "]" if throttling else "✓"
            )
        time.sleep(interval_sec)

    return throttle_events


def max_compute_stress(device: torch.device, matrix_size: int, duration_sec: int):
    """
    Sustained FP16 matmul at maximum throughput.
    Designed to push GPU to TDP and trigger thermal/power throttling.
    Uses multiple concurrent operations for maximum power draw.
    """
    logger.info("Starting maximum compute stress: size=%d duration=%ds", matrix_size, duration_sec)

    # Allocate multiple large tensors to increase power draw
    n_pairs = 4
    pairs = [
        (
            torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device),
            torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device),
        )
        for _ in range(n_pairs)
    ]
    streams = [torch.cuda.Stream(device=device) for _ in range(n_pairs)]

    # Warm up to avoid cold-start artifacts
    logger.info("Warming up GPU for 30s...")
    warmup_start = time.monotonic()
    while time.monotonic() - warmup_start < 30:
        for stream, (A, B) in zip(streams, pairs):
            with torch.cuda.stream(stream):
                C = torch.mm(A, B)
        torch.cuda.synchronize()

    logger.info("Warmup complete. Starting sustained stress for %ds...", duration_sec)
    start = time.monotonic()
    iters = 0
    while time.monotonic() - start < duration_sec:
        for stream, (A, B) in zip(streams, pairs):
            with torch.cuda.stream(stream):
                C = torch.mm(A, B)
        torch.cuda.synchronize()
        iters += 1

    elapsed = time.monotonic() - start
    logger.info("Max compute stress: %d iterations in %.1fs", iters, elapsed)


def power_ramp_test(device: torch.device, matrix_size: int = 4096, steps: int = 5):
    """
    Ramp power usage gradually: 20% → 40% → 60% → 80% → 100% of TDP.
    Observes temperature ramp-up and clock boost behavior.
    Uses sleep-based duty cycling to approximate power targets.
    """
    logger.info("Power ramp test: %d steps, matrix_size=%d", steps, matrix_size)

    A = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device)
    B = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device)

    duty_cycles = [0.2, 0.4, 0.6, 0.8, 1.0]
    step_duration = 60  # seconds per step

    for duty in duty_cycles:
        logger.info("Power ramp: duty_cycle=%.0f%% for %ds", duty * 100, step_duration)
        start = time.monotonic()
        while time.monotonic() - start < step_duration:
            compute_start = time.monotonic()
            C = torch.mm(A, B)
            torch.cuda.synchronize()
            compute_time = time.monotonic() - compute_start
            sleep_time = compute_time * (1 - duty) / duty
            if sleep_time > 0:
                time.sleep(sleep_time)

        state = get_throttle_state()
        logger.info("After duty=%.0f%%: temp=%s°C power=%sW",
                    duty * 100, state.get("temperature.gpu"), state.get("power.draw"))


def main():
    parser = argparse.ArgumentParser(description="Thermal/Power Stress Workload")
    parser.add_argument("--duration", type=int, default=900, help="Stress duration in seconds")
    parser.add_argument("--matrix-size", type=int, default=8192, help="Matrix dimension for compute stress")
    parser.add_argument("--monitor", action="store_true", default=True, help="Monitor thermal state in background")
    parser.add_argument("--monitor-interval", type=float, default=2.0, help="Monitoring interval (sec)")
    parser.add_argument("--ramp-test", action="store_true", help="Run power ramp test instead")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found.")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    props = torch.cuda.get_device_properties(device)
    logger.info("Thermal stress on GPU %d: %s | %.1f GB VRAM", args.gpu, props.name, props.total_memory / 1e9)

    initial_state = get_throttle_state(args.gpu)
    logger.info("Initial state: temp=%s°C power=%sW", initial_state.get("temperature.gpu"), initial_state.get("power.draw"))

    stop_event = threading.Event()
    throttle_log = []

    if args.monitor:
        monitor_thread = threading.Thread(
            target=lambda: throttle_log.extend(monitor_thermal(args.gpu, stop_event, args.monitor_interval) or []),
            daemon=True
        )
        monitor_thread.start()

    try:
        if args.ramp_test:
            power_ramp_test(device, args.matrix_size)
        else:
            max_compute_stress(device, args.matrix_size, args.duration)
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        stop_event.set()
        if args.monitor:
            monitor_thread.join(timeout=5)
        torch.cuda.empty_cache()

    # Summary
    final_state = get_throttle_state(args.gpu)
    logger.info(
        "Final state: temp=%s°C power=%sW | Throttle events: %d",
        final_state.get("temperature.gpu"),
        final_state.get("power.draw"),
        len(throttle_log)
    )
    if throttle_log:
        logger.info("First throttle event: %s", throttle_log[0])


if __name__ == "__main__":
    main()
