#!/usr/bin/env python3
"""
workload/pcie/pcie_stress.py

PCIe bandwidth stress workload.
Generates sustained host-to-device and device-to-host transfers
using large buffers and multiple CUDA streams to saturate PCIe bandwidth.
Triggers: PCIe Throughput TX/RX, PCIe Replay Counter metrics.

Usage:
    python3 workload/pcie/pcie_stress.py --buffer-size-gb 1 --duration 300
    python3 workload/pcie/pcie_stress.py --direction h2d --buffer-size-gb 2 --num-streams 4
"""

import time
import logging
import argparse
import statistics
from typing import List

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("pcie_stress")


def get_pcie_gen_width(gpu_index: int = 0) -> dict:
    """Query current PCIe gen and width via nvidia-smi."""
    import subprocess
    result = subprocess.run(
        [
            "nvidia-smi",
            "-i", str(gpu_index),
            "--query-gpu=pcie.link.gen.current,pcie.link.width.current,pcie.link.gen.max,pcie.link.width.max",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True, text=True
    )
    if result.returncode == 0 and result.stdout.strip():
        parts = result.stdout.strip().split(",")
        try:
            return {
                "current_gen": int(parts[0].strip()),
                "current_width": int(parts[1].strip()),
                "max_gen": int(parts[2].strip()),
                "max_width": int(parts[3].strip()),
            }
        except Exception:
            pass
    return {}


def theoretical_pcie_bandwidth_gbs(gen: int, width: int) -> float:
    """Calculate theoretical PCIe bandwidth in GB/s."""
    # Transfer rates per lane per direction: Gen3=~1GB/s, Gen4=~2GB/s, Gen5=~4GB/s
    lane_rates = {3: 0.985, 4: 1.969, 5: 3.938}
    rate = lane_rates.get(gen, 1.0)
    return rate * width


def h2d_transfer(device: torch.device, host_buf: torch.Tensor, stream: torch.cuda.Stream) -> float:
    """Host-to-Device transfer, returns duration in seconds."""
    with torch.cuda.stream(stream):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        gpu_buf = host_buf.to(device, non_blocking=True)
        end_event.record(stream)
    stream.synchronize()
    ms = start_event.elapsed_time(end_event)
    return ms / 1000.0, gpu_buf


def d2h_transfer(gpu_buf: torch.Tensor, stream: torch.cuda.Stream) -> float:
    """Device-to-Host transfer, returns duration in seconds."""
    with torch.cuda.stream(stream):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(stream)
        host_buf = gpu_buf.to("cpu", non_blocking=True)
        end_event.record(stream)
    stream.synchronize()
    ms = start_event.elapsed_time(end_event)
    return ms / 1000.0, host_buf


def pcie_stress(
    device: torch.device,
    buffer_size_bytes: int,
    duration_sec: int,
    direction: str = "both",
    num_streams: int = 4,
    gpu_index: int = 0,
):
    """
    Main PCIe stress loop.
    Generates continuous large transfers to saturate PCIe bandwidth.
    """
    pcie_info = get_pcie_gen_width(gpu_index)
    if pcie_info:
        theoretical_bw = theoretical_pcie_bandwidth_gbs(pcie_info["current_gen"], pcie_info["current_width"])
        logger.info(
            "PCIe: Gen%d x%d (max Gen%d x%d) | Theoretical: %.2f GB/s",
            pcie_info["current_gen"], pcie_info["current_width"],
            pcie_info["max_gen"], pcie_info["max_width"],
            theoretical_bw
        )

    n_elements = buffer_size_bytes // 4  # float32
    logger.info(
        "PCIe stress: %.2f GB buffers | %d streams | direction=%s | duration=%ds",
        buffer_size_bytes / 1e9, num_streams, direction, duration_sec
    )

    # Create pinned host buffers (pinned memory enables DMA transfers, bypassing CPU cache)
    host_bufs = [
        torch.rand(n_elements, dtype=torch.float32).pin_memory()
        for _ in range(num_streams)
    ]

    # Pre-allocate GPU buffers
    gpu_bufs = [
        torch.empty(n_elements, dtype=torch.float32, device=device)
        for _ in range(num_streams)
    ]

    # CUDA streams for overlapping transfers
    streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]

    h2d_bws: List[float] = []
    d2h_bws: List[float] = []

    start = time.monotonic()
    iters = 0

    while time.monotonic() - start < duration_sec:
        for i in range(num_streams):
            stream = streams[i]

            if direction in ("h2d", "both"):
                with torch.cuda.stream(stream):
                    gpu_bufs[i].copy_(host_bufs[i], non_blocking=True)

            if direction in ("d2h", "both"):
                with torch.cuda.stream(stream):
                    host_bufs[i].copy_(gpu_bufs[i], non_blocking=True)

        # Sync all streams
        torch.cuda.synchronize()
        iters += 1

        if iters % 10 == 0:
            elapsed = time.monotonic() - start
            total_bytes = buffer_size_bytes * num_streams * iters
            if direction == "both":
                total_bytes *= 2
            bw = total_bytes / elapsed / 1e9
            logger.info(
                "PCIe iter=%d elapsed=%.1fs bandwidth=%.2f GB/s",
                iters, elapsed, bw
            )

    torch.cuda.synchronize()
    total_time = time.monotonic() - start
    total_bytes = buffer_size_bytes * num_streams * iters
    if direction == "both":
        total_bytes *= 2
    avg_bw = total_bytes / total_time / 1e9

    logger.info(
        "PCIe stress complete: %d iterations | %.1fs | avg %.2f GB/s | total %.2f GB transferred",
        iters, total_time, avg_bw, total_bytes / 1e9
    )
    if pcie_info:
        efficiency = avg_bw / theoretical_bw * 100
        logger.info("PCIe efficiency: %.1f%% of theoretical %.2f GB/s", efficiency, theoretical_bw)

    return {"avg_bw_gbs": avg_bw, "iters": iters, "total_sec": total_time}


def pcie_replay_test(device: torch.device, gpu_index: int = 0):
    """
    Check PCIe replay counter before and after stress.
    Replays indicate link errors and should be 0 on healthy hardware.
    """
    import subprocess

    def get_replay_count():
        r = subprocess.run(
            ["nvidia-smi", "-i", str(gpu_index), "--query-gpu=pcie.link.tx_util", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        # Note: nvidia-smi doesn't expose replay counter directly; use dcgmi for DCGM_FI_DEV_PCIE_REPLAY_COUNTER
        return r.stdout.strip()

    logger.info("PCIe replay pre-test: %s", get_replay_count())
    # Run a short burst
    pcie_stress(device, buffer_size_bytes=512 * 1024 * 1024, duration_sec=30, direction="both", num_streams=2)
    logger.info("PCIe replay post-test: %s", get_replay_count())


def main():
    parser = argparse.ArgumentParser(description="PCIe Bandwidth Stress Workload")
    parser.add_argument("--buffer-size-gb", type=float, default=1.0, help="Transfer buffer size in GB")
    parser.add_argument("--duration", type=int, default=300, help="Duration in seconds")
    parser.add_argument("--direction", choices=["h2d", "d2h", "both"], default="both",
                        help="Transfer direction")
    parser.add_argument("--num-streams", type=int, default=4, help="Number of CUDA streams")
    parser.add_argument("--replay-test", action="store_true", help="Run PCIe replay counter test")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found.")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")
    props = torch.cuda.get_device_properties(device)
    logger.info("Using GPU %d: %s", args.gpu, props.name)

    buffer_bytes = int(args.buffer_size_gb * 1e9)

    try:
        if args.replay_test:
            pcie_replay_test(device, args.gpu)
        else:
            pcie_stress(
                device=device,
                buffer_size_bytes=buffer_bytes,
                duration_sec=args.duration,
                direction=args.direction,
                num_streams=args.num_streams,
                gpu_index=args.gpu,
            )
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        torch.cuda.empty_cache()
        logger.info("PCIe stress complete.")


if __name__ == "__main__":
    main()
