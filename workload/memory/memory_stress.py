#!/usr/bin/env python3
"""
workload/memory/memory_stress.py

GPU memory stress workload.
Exercises:
  - Near-full VRAM allocation
  - Fragmentation patterns (allocate/free chunks of varying sizes)
  - Rapid alloc/free cycles
  - Memory bandwidth saturation (large sequential reads/writes)

Usage:
    python3 workload/memory/memory_stress.py --fill-percent 90 --duration 300
    python3 workload/memory/memory_stress.py --mode fragmentation --cycles 100
    python3 workload/memory/memory_stress.py --mode rapid_alloc --duration 120
"""

import gc
import time
import random
import logging
import argparse
from typing import List

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("memory_stress")


def get_gpu_memory_info(device: torch.device):
    total = torch.cuda.get_device_properties(device).total_memory
    allocated = torch.cuda.memory_allocated(device)
    reserved = torch.cuda.memory_reserved(device)
    free = total - allocated
    return {
        "total_gb": total / 1e9,
        "allocated_gb": allocated / 1e9,
        "reserved_gb": reserved / 1e9,
        "free_gb": free / 1e9,
        "used_pct": (allocated / total) * 100,
    }


def fill_memory_stress(device: torch.device, fill_percent: float, duration_sec: int):
    """
    Allocate fill_percent of GPU VRAM and hold it for duration_sec.
    Then perform sequential reads/writes to stress memory bandwidth.
    """
    info = get_gpu_memory_info(device)
    logger.info("GPU memory before fill: %s", info)

    target_bytes = int(info["total_gb"] * 1e9 * (fill_percent / 100.0))
    # Leave 500 MB headroom for PyTorch internals
    target_bytes = max(0, target_bytes - 500 * 1024 * 1024)

    # Allocate as float32 (4 bytes each)
    n_elements = target_bytes // 4
    logger.info(
        "Allocating %.2f GB (%.0f%% of %.2f GB total)...",
        target_bytes / 1e9, fill_percent, info["total_gb"]
    )

    try:
        big_tensor = torch.zeros(n_elements, dtype=torch.float32, device=device)
    except torch.cuda.OutOfMemoryError:
        logger.warning("OOM at %.0f%%, retrying at %.0f%%", fill_percent, fill_percent * 0.85)
        n_elements = int(n_elements * 0.85)
        big_tensor = torch.zeros(n_elements, dtype=torch.float32, device=device)

    info_after = get_gpu_memory_info(device)
    logger.info("GPU memory after fill: %s", info_after)

    # Now stress memory bandwidth: continuous reads + writes
    ones = torch.ones(n_elements, dtype=torch.float32, device=device)
    start = time.monotonic()
    iters = 0
    logger.info("Holding allocation and running memory bandwidth stress for %ds...", duration_sec)
    while time.monotonic() - start < duration_sec:
        big_tensor.add_(ones * 0.0001)   # Read + write
        if iters % 100 == 0:
            torch.cuda.synchronize()
        iters += 1

    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    bw_gbs = (2 * target_bytes * iters) / elapsed / 1e9
    logger.info("Memory bandwidth stress: %.2f GB/s over %.1fs", bw_gbs, elapsed)

    del big_tensor, ones
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Memory released. GPU memory: %s", get_gpu_memory_info(device))


def fragmentation_stress(device: torch.device, alloc_sizes_mb: List[int], cycles: int):
    """
    Simulate memory fragmentation by allocating chunks of varying sizes
    and freeing them in non-sequential order.
    This stresses the CUDA memory allocator and can reveal fragmentation bugs.
    """
    logger.info(
        "Starting fragmentation stress: %d cycles, sizes=%s MB",
        cycles, alloc_sizes_mb
    )
    info = get_gpu_memory_info(device)
    logger.info("GPU memory before: %s", info)

    for cycle in range(cycles):
        # Allocate chunks of varying sizes
        sizes = random.choices(alloc_sizes_mb, k=random.randint(4, 12))
        tensors = []
        for size_mb in sizes:
            n = (size_mb * 1024 * 1024) // 2  # float16 = 2 bytes
            try:
                t = torch.rand(n, dtype=torch.float16, device=device)
                tensors.append(t)
            except torch.cuda.OutOfMemoryError:
                logger.warning("OOM during fragmentation (cycle=%d, size=%dMB)", cycle, size_mb)
                break

        # Do some work on the tensors
        if tensors:
            result = sum(t.sum() for t in tensors)

        # Free in shuffled order to fragment address space
        random.shuffle(tensors)
        for t in tensors:
            del t
        tensors.clear()

        if cycle % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            info = get_gpu_memory_info(device)
            logger.info("Fragmentation cycle %d/%d | %s", cycle + 1, cycles, info)

    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Fragmentation stress complete. Final memory: %s", get_gpu_memory_info(device))


def rapid_alloc_free_stress(device: torch.device, alloc_size_mb: int, duration_sec: int):
    """
    Rapid allocation and deallocation cycles — stresses CUDA memory manager.
    Generates high memory utilization churn to stress ECC scrubbing and
    page retirement paths.
    """
    logger.info(
        "Starting rapid alloc/free stress: size=%dMB duration=%ds",
        alloc_size_mb, duration_sec
    )
    n = (alloc_size_mb * 1024 * 1024) // 4  # float32

    start = time.monotonic()
    iters = 0
    while time.monotonic() - start < duration_sec:
        t = torch.rand(n, dtype=torch.float32, device=device)
        _ = t.sum()    # Ensure the allocation is actually used
        del t
        if iters % 50 == 0:
            torch.cuda.empty_cache()
        iters += 1

    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    logger.info("Rapid alloc/free: %d cycles in %.1fs (%.1f cycles/sec)", iters, elapsed, iters / elapsed)


def memory_read_only_stress(device: torch.device, duration_sec: int):
    """
    Saturate memory read bandwidth with large sequential reads.
    Fill buffer once, then read repeatedly.
    """
    logger.info("Starting memory read-only stress: duration=%ds", duration_sec)
    props = torch.cuda.get_device_properties(device)
    # Use ~60% of total memory
    n = (props.total_memory * 6 // 10) // 4
    buf = torch.rand(n, dtype=torch.float32, device=device)

    start = time.monotonic()
    iters = 0
    while time.monotonic() - start < duration_sec:
        s = buf.sum()   # Forces full buffer read
        iters += 1

    torch.cuda.synchronize()
    elapsed = time.monotonic() - start
    read_bw_gbs = (n * 4 * iters) / elapsed / 1e9
    logger.info("Read-only bandwidth: %.2f GB/s over %d reads", read_bw_gbs, iters)

    del buf
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="GPU Memory Stress Workload")
    parser.add_argument("--mode", choices=["fill", "fragmentation", "rapid_alloc", "read_only", "all"],
                        default="fill", help="Stress mode")
    parser.add_argument("--fill-percent", type=float, default=90.0, help="Target VRAM fill percentage")
    parser.add_argument("--duration", type=int, default=300, help="Duration in seconds (for fill/rapid_alloc)")
    parser.add_argument("--cycles", type=int, default=100, help="Number of cycles (fragmentation mode)")
    parser.add_argument("--alloc-size-mb", type=int, default=512, help="Single alloc size MB (rapid_alloc)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found.")
    torch.cuda.set_device(args.gpu)
    device = torch.device(f"cuda:{args.gpu}")

    props = torch.cuda.get_device_properties(device)
    logger.info("Using GPU %d: %s (%.1f GB)", args.gpu, props.name, props.total_memory / 1e9)

    try:
        if args.mode == "fill" or args.mode == "all":
            fill_memory_stress(device, args.fill_percent, args.duration)

        if args.mode == "fragmentation" or args.mode == "all":
            fragmentation_stress(device, alloc_sizes_mb=[64, 256, 512, 1024], cycles=args.cycles)

        if args.mode == "rapid_alloc" or args.mode == "all":
            rapid_alloc_free_stress(device, args.alloc_size_mb, args.duration)

        if args.mode == "read_only" or args.mode == "all":
            memory_read_only_stress(device, args.duration)

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Done. Final GPU memory: %s", get_gpu_memory_info(device))


if __name__ == "__main__":
    main()
