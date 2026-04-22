#!/usr/bin/env python3
"""
workload/llm/inference_workload.py

LLM inference workload using vLLM.
Drives an inference server through multiple phases:
  - Concurrency ramp (batch_sweep)
  - Sustained load (10–30 min)
  - Burst load (high concurrency)
  - Long-context inference

Can also launch a vLLM server internally or connect to an existing one.

Usage:
    # Launch vLLM server and run all phases
    python3 workload/llm/inference_workload.py --model meta-llama/Llama-2-7b-hf --launch-server

    # Connect to existing server
    python3 workload/llm/inference_workload.py --server-url http://localhost:8000 --phase sustained_load
"""

import os
import sys
import time
import json
import random
import logging
import argparse
import asyncio
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Optional, Any

import aiohttp
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("llm_inference")


# ─── Prompt generators ──────────────────────────────────────────────────────

PROMPT_TEMPLATES = [
    "Explain the following concept in detail: {topic}. Include historical context, technical details, and practical applications.",
    "Write a comprehensive analysis of {topic}. Cover multiple perspectives and provide specific examples.",
    "Describe the mechanism by which {topic} works, step by step, with technical precision.",
    "Compare and contrast {topic} with three related concepts. Identify key similarities and differences.",
]

TOPICS = [
    "transformer attention mechanisms",
    "CUDA memory hierarchy and caching",
    "GPU kernel optimization strategies",
    "distributed training with model parallelism",
    "quantization techniques for LLM inference",
    "speculative decoding in language models",
    "flash attention algorithm",
    "gradient checkpointing in deep learning",
    "key-value cache management in LLM serving",
    "continuous batching in inference systems",
]


def generate_prompt(target_tokens: int = 512) -> str:
    """Generate a prompt that should produce approximately target_tokens output."""
    template = random.choice(PROMPT_TEMPLATES)
    topic = random.choice(TOPICS)
    prompt = template.format(topic=topic)
    # Pad with more context to increase input length
    if target_tokens > 256:
        context_sentences = [
            f"Provide at least {target_tokens} tokens in your response.",
            "Be thorough and comprehensive.",
            "Include all relevant technical details.",
        ]
        prompt += " " + " ".join(context_sentences)
    return prompt


def generate_long_context_prompt(target_input_tokens: int = 3000) -> str:
    """Generate a long-context prompt to stress the KV cache."""
    base = generate_prompt(512)
    padding = " ".join([
        f"Additional context point {i}: The field of artificial intelligence has seen remarkable growth "
        f"in recent years, with language models achieving unprecedented capabilities."
        for i in range(target_input_tokens // 30)
    ])
    return f"{padding}\n\nBased on all the above context:\n{base}"


# ─── vLLM Server Management ─────────────────────────────────────────────────

class VLLMServer:
    """Manages a vLLM OpenAI-compatible server process."""

    def __init__(
        self,
        model: str,
        port: int = 8000,
        dtype: str = "float16",
        max_model_len: int = 4096,
        gpu_memory_utilization: float = 0.90,
        gpu: int = 0,
    ):
        self.model = model
        self.port = port
        self.dtype = dtype
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.gpu = gpu
        self.base_url = f"http://localhost:{port}"
        self._process: Optional[subprocess.Popen] = None

    def start(self):
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", self.model,
            "--port", str(self.port),
            "--dtype", self.dtype,
            "--max-model-len", str(self.max_model_len),
            "--gpu-memory-utilization", str(self.gpu_memory_utilization),
            "--device", f"cuda",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.gpu)

        logger.info("Launching vLLM server: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._wait_ready()

    def _wait_ready(self, timeout_sec: int = 300):
        logger.info("Waiting for vLLM server to be ready (timeout=%ds)...", timeout_sec)
        start = time.monotonic()
        while time.monotonic() - start < timeout_sec:
            try:
                r = requests.get(f"{self.base_url}/health", timeout=3)
                if r.status_code == 200:
                    logger.info("vLLM server ready at %s", self.base_url)
                    return
            except Exception:
                pass
            time.sleep(3)
        raise TimeoutError(f"vLLM server not ready within {timeout_sec}s")

    def stop(self):
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
            logger.info("vLLM server stopped.")

    def is_alive(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False


# ─── Async Request Engine ────────────────────────────────────────────────────

async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> Dict:
    """Send one chat completion request, return timing and metadata."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    t0 = time.monotonic()
    try:
        async with session.post(
            f"{url}/v1/chat/completions",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=120),
        ) as resp:
            resp.raise_for_status()
            data = await resp.json()
            t1 = time.monotonic()
            latency = t1 - t0
            usage = data.get("usage", {})
            output_tokens = usage.get("completion_tokens", 0)
            input_tokens = usage.get("prompt_tokens", 0)
            return {
                "success": True,
                "latency_sec": latency,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "tokens_per_sec": output_tokens / latency if latency > 0 else 0,
            }
    except Exception as e:
        t1 = time.monotonic()
        return {"success": False, "error": str(e), "latency_sec": t1 - t0}


async def run_concurrent_requests(
    url: str,
    model: str,
    concurrency: int,
    n_requests: int,
    max_tokens: int = 256,
    input_tokens: int = 512,
) -> Dict:
    """Run n_requests with concurrency workers, return aggregated stats."""
    prompts = [generate_prompt(input_tokens) for _ in range(n_requests)]
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(session, prompt):
        async with semaphore:
            return await send_request(session, url, model, prompt, max_tokens=max_tokens)

    connector = aiohttp.TCPConnector(limit=concurrency + 10)
    async with aiohttp.ClientSession(connector=connector) as session:
        t0 = time.monotonic()
        tasks = [bounded_request(session, p) for p in prompts]
        results = await asyncio.gather(*tasks)
        total_time = time.monotonic() - t0

    successes = [r for r in results if r.get("success")]
    failures = [r for r in results if not r.get("success")]

    if successes:
        latencies = [r["latency_sec"] for r in successes]
        throughput_tokens = sum(r["output_tokens"] for r in successes)
        stats = {
            "concurrency": concurrency,
            "n_requests": n_requests,
            "n_success": len(successes),
            "n_failure": len(failures),
            "total_time_sec": total_time,
            "throughput_req_per_sec": len(successes) / total_time,
            "throughput_tokens_per_sec": throughput_tokens / total_time,
            "latency_p50": statistics.median(latencies),
            "latency_p95": sorted(latencies)[int(len(latencies) * 0.95)],
            "latency_p99": sorted(latencies)[int(len(latencies) * 0.99)],
            "latency_mean": statistics.mean(latencies),
        }
    else:
        stats = {"concurrency": concurrency, "n_requests": n_requests, "n_success": 0, "error": "All failed"}

    return stats


# ─── Workload Phases ─────────────────────────────────────────────────────────

async def phase_batch_sweep(url: str, model: str, config: Dict) -> List[Dict]:
    """Ramp concurrency from 1 to N and measure throughput at each level."""
    results = []
    concurrency_levels = config.get("concurrency", [1, 4, 8, 16, 32, 64])
    requests_each = config.get("requests_per_concurrency", 20)
    input_tokens = config.get("input_tokens", 512)
    output_tokens = config.get("output_tokens", 128)

    logger.info("=== Phase: Batch Sweep === concurrency levels: %s", concurrency_levels)
    for c in concurrency_levels:
        logger.info("Concurrency=%d: sending %d requests...", c, requests_each)
        stats = await run_concurrent_requests(
            url, model,
            concurrency=c,
            n_requests=requests_each,
            max_tokens=output_tokens,
            input_tokens=input_tokens,
        )
        logger.info(
            "c=%d | req/s=%.2f | tok/s=%.1f | p50=%.2fs | p95=%.2fs | success=%d/%d",
            c, stats.get("throughput_req_per_sec", 0),
            stats.get("throughput_tokens_per_sec", 0),
            stats.get("latency_p50", 0), stats.get("latency_p95", 0),
            stats.get("n_success", 0), stats.get("n_requests", 0)
        )
        results.append(stats)
    return results


async def phase_sustained_load(url: str, model: str, config: Dict) -> Dict:
    """Run sustained concurrency for a long duration."""
    concurrency = config.get("concurrency", 32)
    duration_sec = config.get("duration_sec", 600)
    input_tokens = config.get("input_tokens", 512)
    output_tokens = config.get("output_tokens", 256)

    logger.info(
        "=== Phase: Sustained Load === concurrency=%d duration=%ds",
        concurrency, duration_sec
    )

    all_results = []
    start = time.monotonic()

    while time.monotonic() - start < duration_sec:
        batch = await run_concurrent_requests(
            url, model,
            concurrency=concurrency,
            n_requests=concurrency * 2,
            max_tokens=output_tokens,
            input_tokens=input_tokens,
        )
        all_results.append(batch)
        elapsed = time.monotonic() - start
        logger.info(
            "[%.0fs/%.0fs] tok/s=%.1f req/s=%.2f",
            elapsed, duration_sec,
            batch.get("throughput_tokens_per_sec", 0),
            batch.get("throughput_req_per_sec", 0)
        )

    return {"phase": "sustained_load", "batches": len(all_results), "duration_sec": duration_sec}


async def phase_burst_load(url: str, model: str, config: Dict) -> Dict:
    """Short burst at very high concurrency."""
    concurrency = config.get("concurrency", 128)
    duration_sec = config.get("duration_sec", 120)
    input_tokens = config.get("input_tokens", 256)
    output_tokens = config.get("output_tokens", 64)

    logger.info("=== Phase: Burst Load === concurrency=%d duration=%ds", concurrency, duration_sec)
    stats = await run_concurrent_requests(
        url, model,
        concurrency=concurrency,
        n_requests=concurrency * 3,
        max_tokens=output_tokens,
        input_tokens=input_tokens,
    )
    logger.info("Burst: tok/s=%.1f | failures=%d", stats.get("throughput_tokens_per_sec", 0), stats.get("n_failure", 0))
    return stats


async def phase_long_context(url: str, model: str, config: Dict) -> Dict:
    """Long-context inference to stress KV cache and memory."""
    concurrency = config.get("concurrency", 4)
    duration_sec = config.get("duration_sec", 180)
    input_tokens = config.get("input_tokens", 3000)
    output_tokens = config.get("output_tokens", 512)

    logger.info("=== Phase: Long Context === concurrency=%d duration=%ds", concurrency, duration_sec)

    async def bounded_long(session, sem):
        async with sem:
            prompt = generate_long_context_prompt(input_tokens)
            return await send_request(session, url, model, prompt, max_tokens=output_tokens)

    sem = asyncio.Semaphore(concurrency)
    n_requests = concurrency * 5
    connector = aiohttp.TCPConnector(limit=concurrency + 5)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [bounded_long(session, sem) for _ in range(n_requests)]
        results = await asyncio.gather(*tasks)

    successes = [r for r in results if r.get("success")]
    logger.info("Long context: %d/%d succeeded", len(successes), n_requests)
    return {"phase": "long_context", "n_success": len(successes), "n_total": n_requests}


# ─── Main ────────────────────────────────────────────────────────────────────

async def async_main(args):
    server = None
    if args.launch_server:
        server = VLLMServer(
            model=args.model,
            port=args.port,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu=args.gpu,
        )
        server.start()

    url = args.server_url or f"http://localhost:{args.port}"
    model = args.model

    # Verify server is up
    try:
        requests.get(f"{url}/health", timeout=10)
    except Exception as e:
        logger.error("Cannot reach vLLM server at %s: %s", url, e)
        if server:
            server.stop()
        return

    # Warmup
    logger.info("Warming up server with %d requests...", 10)
    for _ in range(10):
        await send_request(
            aiohttp.ClientSession(),
            url, model, "Hello, world!", max_tokens=32
        )

    all_results = {}

    try:
        phases_to_run = args.phase.split(",") if args.phase != "all" else [
            "batch_sweep", "sustained_load", "burst_load", "long_context"
        ]

        for phase_name in phases_to_run:
            phase_config = {
                "batch_sweep": {
                    "concurrency": [1, 4, 8, 16, 32, 64],
                    "requests_per_concurrency": 20,
                    "input_tokens": 512, "output_tokens": 128,
                },
                "sustained_load": {
                    "concurrency": 32, "duration_sec": args.sustained_duration,
                    "input_tokens": 512, "output_tokens": 256,
                },
                "burst_load": {
                    "concurrency": 128, "duration_sec": 120,
                    "input_tokens": 256, "output_tokens": 64,
                },
                "long_context": {
                    "concurrency": 4, "duration_sec": 180,
                    "input_tokens": 3000, "output_tokens": 512,
                },
            }.get(phase_name, {})

            if phase_name == "batch_sweep":
                all_results[phase_name] = await phase_batch_sweep(url, model, phase_config)
            elif phase_name == "sustained_load":
                all_results[phase_name] = await phase_sustained_load(url, model, phase_config)
            elif phase_name == "burst_load":
                all_results[phase_name] = await phase_burst_load(url, model, phase_config)
            elif phase_name == "long_context":
                all_results[phase_name] = await phase_long_context(url, model, phase_config)

    finally:
        if server:
            server.stop()

    # Write results
    output_path = Path("reports/llm_benchmark_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results written to %s", output_path)


def main():
    parser = argparse.ArgumentParser(description="LLM Inference Workload (vLLM)")
    parser.add_argument("--model", default="facebook/opt-125m", help="HuggingFace model name")
    parser.add_argument("--server-url", default=None, help="Existing vLLM server URL")
    parser.add_argument("--launch-server", action="store_true", help="Launch vLLM server")
    parser.add_argument("--port", type=int, default=8000, help="vLLM server port")
    parser.add_argument("--dtype", default="float16", help="Model dtype")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max context length")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument(
        "--phase",
        default="all",
        help="Phases to run: all | batch_sweep | sustained_load | burst_load | long_context"
             " | comma-separated list"
    )
    parser.add_argument("--sustained-duration", type=int, default=600, help="Sustained load duration (sec)")
    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
