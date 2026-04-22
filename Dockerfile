# Dockerfile
# ──────────────────────────────────────────────────────────────────────────────
# GPU Observability Test Runner
#
# Base: NVIDIA CUDA 12.2 + cuDNN on Ubuntu 22.04
# Installs: Python deps, nvidia-smi bindings, optional vLLM
#
# Build (no vLLM):
#   docker build -t gpu-obs .
#
# Build (with vLLM — adds ~10 GB, takes ~15 min):
#   docker build --build-arg INSTALL_VLLM=true -t gpu-obs-vllm .
# ──────────────────────────────────────────────────────────────────────────────

FROM nvcr.io/nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ARG INSTALL_VLLM=false

# ── System packages ────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3-setuptools \
    git \
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Make python3 the default
RUN ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python  && \
    pip3 install --upgrade pip setuptools wheel --no-cache-dir

# ── Project files ──────────────────────────────────────────────────────────────
WORKDIR /workspace
COPY requirements.txt .
COPY setup.py .
COPY conftest.py .
COPY pytest.ini .
COPY collectors/   collectors/
COPY configs/      configs/
COPY dashboards/   dashboards/
COPY scripts/      scripts/
COPY tests/        tests/
COPY validators/   validators/
COPY workload/     workload/
# reports/ is intentionally NOT copied — it may be empty (git does not track
# empty directories) or contain stale output from a previous run.
# The directory is created here and bind-mounted at runtime so that reports
# produced inside the container land on the host:
#   -v $(pwd)/reports:/workspace/reports
RUN mkdir -p /workspace/reports

# ── Python dependencies ────────────────────────────────────────────────────────
# PyTorch with CUDA 12.1 wheels (compatible with CUDA 12.2 runtime)
RUN pip3 install --no-cache-dir \
    torch==2.2.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# All other deps including pytest
RUN pip3 install --no-cache-dir -r requirements.txt

# Explicit install so the ENTRYPOINT always finds pytest even if the
# requirements.txt layer is served from cache before pytest was added
RUN pip3 install --no-cache-dir "pytest>=7.4.0" "pytest-asyncio>=0.23.0"

# ── Optional: vLLM ────────────────────────────────────────────────────────────
# Skipped by default — large (~10 GB) and requires a model download.
# Enable with: docker build --build-arg INSTALL_VLLM=true -t gpu-obs-vllm .
RUN if [ "$INSTALL_VLLM" = "true" ]; then \
      pip3 install --no-cache-dir vllm; \
    fi

# ── Smoke-test: verify all project packages are importable ──────────────────
# Catches missing __init__.py or broken imports at build time rather than
# at test runtime.
RUN python3 -c "
import sys
sys.path.insert(0, '/workspace')
from collectors.dcgm_collector import TelemetryCollector
from validators.run_validators import run_all_validators
from workload.cuda.matmul_stress import matmul_fp16_stress
from workload.memory.memory_stress import fill_memory_stress
from workload.pcie.pcie_stress import pcie_stress
from workload.thermal.thermal_stress import max_compute_stress
print('All project imports OK')
"

# ── Environment ────────────────────────────────────────────────────────────────
ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1

# ── Entrypoint ─────────────────────────────────────────────────────────────────
# Default: run the full live test suite.
# Override with any pytest arguments after the image name.
ENTRYPOINT ["python3", "-m", "pytest"]
CMD ["tests/test_live_workloads.py", "tests/test_all_metrics.py", "-v", "--tb=short"]