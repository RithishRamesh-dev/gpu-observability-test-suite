#!/usr/bin/env python3
"""
workload/cuda/matmul_stress.py

CUDA synthetic stress workload using PyTorch.
Exercises Tensor Cores (FP16/BF16) and FP32 pipelines via large matrix multiplications.
Also includes warp occupancy and memory bandwidth stress modes.

Usage:
    python3 workload/cuda/matmul_stress.py --duration 300 --dtype fp16
    python3 workload/cuda/matmul_stress.py --duration 300 --dtype fp32
    python3 workload/cuda/matmul_stress.py --mode warp_occupancy --duration 180
"""

import os
import sys
import time
import math
import logging
import argparse
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("matmul_stress")


def check_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU detected. Exiting.")
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        props = torch.cuda.get_device_properties(i)
        logger.info(
            "GPU %d: %s | %.1f GB | SM %d.%d | %d SMs | TDP unknown from PyTorch",
            i, props.name, props.total_memory / 1e9, props.major, props.minor, props.multi_processor_count
        )
    return torch.device("cuda:0")


def matmul_fp16_stress(device: torch.device, matrix_size: int, duration_sec: int):
    """
    FP16 matrix multiplication stress — hammers Tensor Cores.
    Uses torch.mm with half-precision tensors to maximize GEMM throughput.
    """
    logger.info(
        "Starting FP16 matmul stress: size=%dx%d duration=%ds",
        matrix_size, matrix_size, duration_sec
    )
    # Pin large FP16 tensors
    A = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device)
    B = torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device)
    C = torch.empty(matrix_size, matrix_size, dtype=torch.float16, device=device)

    # Warm up
    for _ in range(5):
        torch.mm(A, B, out=C)
    torch.cuda.synchronize()

    start = time.monotonic()
    iters = 0
    while time.monotonic() - start < duration_sec:
        # Alternate operands to prevent caching
        if iters % 2 == 0:
            torch.mm(A, B, out=C)
        else:
            torch.mm(B, A, out=C)
        iters += 1

    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    tflops = (2 * matrix_size**3 * iters) / elapsed / 1e12
    logger.info("FP16 matmul: %d iterations in %.1fs | %.2f TFLOPS", iters, elapsed, tflops)


def matmul_fp32_stress(device: torch.device, matrix_size: int, duration_sec: int):
    """
    FP32 matrix multiplication stress — exercises CUDA cores, not Tensor Cores.
    Uses float32 to maximize FP32 pipeline utilization.
    """
    logger.info(
        "Starting FP32 matmul stress: size=%dx%d duration=%ds",
        matrix_size, matrix_size, duration_sec
    )
    A = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)
    B = torch.randn(matrix_size, matrix_size, dtype=torch.float32, device=device)
    C = torch.empty(matrix_size, matrix_size, dtype=torch.float32, device=device)

    for _ in range(3):
        torch.mm(A, B, out=C)
    torch.cuda.synchronize()

    start = time.monotonic()
    iters = 0
    while time.monotonic() - start < duration_sec:
        torch.mm(A, B, out=C)
        iters += 1

    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    tflops = (2 * matrix_size**3 * iters) / elapsed / 1e12
    logger.info("FP32 matmul: %d iterations in %.1fs | %.2f TFLOPS", iters, elapsed, tflops)


def matmul_bf16_stress(device: torch.device, matrix_size: int, duration_sec: int):
    """BF16 matmul — also uses Tensor Cores on Ampere+."""
    logger.info("Starting BF16 matmul stress: size=%dx%d duration=%ds", matrix_size, matrix_size, duration_sec)
    A = torch.randn(matrix_size, matrix_size, dtype=torch.bfloat16, device=device)
    B = torch.randn(matrix_size, matrix_size, dtype=torch.bfloat16, device=device)
    start = time.monotonic()
    iters = 0
    while time.monotonic() - start < duration_sec:
        C = torch.mm(A, B)
        iters += 1
    torch.cuda.synchronize()
    logger.info("BF16 matmul: %d iterations in %.1fs", iters, time.monotonic() - start)


def warp_occupancy_stress(device: torch.device, duration_sec: int):
    """
    Launch many small kernels to stress warp scheduling and SM occupancy.
    Uses many small matrix ops to ensure many warps are live simultaneously.
    """
    logger.info("Starting warp occupancy stress: duration=%ds", duration_sec)

    # Many simultaneous small GEMMs fill the GPU with active warps
    batch = 512
    size = 256
    A = torch.randn(batch, size, size, dtype=torch.float16, device=device)
    B = torch.randn(batch, size, size, dtype=torch.float16, device=device)

    start = time.monotonic()
    iters = 0
    while time.monotonic() - start < duration_sec:
        C = torch.bmm(A, B)
        iters += 1
    torch.cuda.synchronize()
    logger.info("Warp occupancy stress: %d bmm iterations in %.1fs", iters, time.monotonic() - start)


def memory_bandwidth_stress(device: torch.device, duration_sec: int):
    """
    Memory bandwidth saturation — reads/writes large buffers continuously.
    Triggers high memory utilization and bandwidth metrics.
    """
    logger.info("Starting memory bandwidth stress: duration=%ds", duration_sec)
    props = torch.cuda.get_device_properties(device)
    # Allocate ~50% of GPU memory for the buffer
    mem_bytes = props.total_memory // 2
    elem_count = mem_bytes // 4  # float32
    buf = torch.zeros(elem_count, dtype=torch.float32, device=device)
    ones = torch.ones(elem_count, dtype=torch.float32, device=device)

    start = time.monotonic()
    iters = 0
    while time.monotonic() - start < duration_sec:
        buf.add_(ones)  # In-place add triggers both read and write
        iters += 1
    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    # Bandwidth: 2 * buffer_size * iters / elapsed (read + write)
    bw_gbs = (2 * mem_bytes * iters) / elapsed / 1e9
    logger.info(
        "Memory bandwidth: %d iterations | %.2f GB/s | %.1fs",
        iters, bw_gbs, elapsed
    )


def multi_stream_stress(device: torch.device, matrix_size: int, duration_sec: int, num_streams: int = 4):
    """
    Multi-stream FP16 matmul to maximize SM utilization across streams.
    Each CUDA stream runs independent GEMMs concurrently.
    """
    logger.info(
        "Starting multi-stream stress: %d streams, size=%d, duration=%ds",
        num_streams, matrix_size, duration_sec
    )
    streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
    pairs = [
        (
            torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device),
            torch.randn(matrix_size, matrix_size, dtype=torch.float16, device=device),
        )
        for _ in range(num_streams)
    ]

    start = time.monotonic()
    iters = 0
    while time.monotonic() - start < duration_sec:
        for i, (stream, (A, B)) in enumerate(zip(streams, pairs)):
            with torch.cuda.stream(stream):
                C = torch.mm(A, B)
        # Sync all streams
        torch.cuda.synchronize()
        iters += 1

    logger.info("Multi-stream: %d sync-iters in %.1fs", iters, time.monotonic() - start)


def log_gpu_stats(device: torch.device):
    """Print current GPU memory usage."""
    alloc = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    props = torch.cuda.get_device_properties(device)
    total = props.total_memory / 1e9
    logger.info(
        "GPU memory: allocated=%.2fGB reserved=%.2fGB total=%.2fGB",
        alloc, reserved, total
    )


def main():
    parser = argparse.ArgumentParser(description="CUDA Synthetic Stress Workload")
    parser.add_argument("--duration", type=int, default=300, help="Duration in seconds")
    parser.add_argument("--dtype", choices=["fp16", "fp32", "bf16", "all"], default="fp16",
                        help="Compute dtype. 'all' runs fp16, fp32, bf16 sequentially.")
    parser.add_argument("--mode", choices=["matmul", "warp_occupancy", "bandwidth", "multi_stream", "all"],
                        default="matmul", help="Stress mode")
    parser.add_argument("--matrix-size", type=int, default=8192, help="Matrix dimension N (NxN)")
    parser.add_argument("--num-streams", type=int, default=4, help="Number of CUDA streams (multi_stream mode)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    device = check_gpu()
    log_gpu_stats(device)

    try:
        if args.mode == "matmul" or args.mode == "all":
            if args.dtype in ("fp16", "all"):
                matmul_fp16_stress(device, args.matrix_size, args.duration)
                log_gpu_stats(device)
            if args.dtype in ("fp32", "all"):
                matmul_fp32_stress(device, args.matrix_size, args.duration)
                log_gpu_stats(device)
            if args.dtype in ("bf16", "all"):
                matmul_bf16_stress(device, args.matrix_size, args.duration)
                log_gpu_stats(device)

        if args.mode == "warp_occupancy" or args.mode == "all":
            warp_occupancy_stress(device, args.duration)

        if args.mode == "bandwidth" or args.mode == "all":
            memory_bandwidth_stress(device, args.duration)

        if args.mode == "multi_stream" or args.mode == "all":
            multi_stream_stress(device, args.matrix_size, args.duration, args.num_streams)

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        torch.cuda.empty_cache()
        logger.info("Workload complete. GPU memory freed.")


if __name__ == "__main__":
    main()
