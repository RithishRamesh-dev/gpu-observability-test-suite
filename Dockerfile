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
COPY collectors/   collectors/
COPY configs/      configs/
COPY dashboards/   dashboards/
COPY scripts/      scripts/
COPY tests/        tests/
COPY validators/   validators/
COPY workload/     workload/
COPY reports/      reports/

# ── Python dependencies ────────────────────────────────────────────────────────
# PyTorch with CUDA 12.1 wheels (compatible with CUDA 12.2 runtime)
RUN pip3 install --no-cache-dir \
    torch==2.2.0+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip3 install --no-cache-dir -r requirements.txt

# ── Optional: vLLM ────────────────────────────────────────────────────────────
# Skipped by default because it is large (~10 GB) and requires a model download.
# Enable with: docker build --build-arg INSTALL_VLLM=true ...
RUN if [ "$INSTALL_VLLM" = "true" ]; then \
      pip3 install --no-cache-dir vllm; \
    fi

# ── Environment ────────────────────────────────────────────────────────────────
ENV PYTHONPATH=/workspace
ENV PYTHONUNBUFFERED=1

# Reports directory is writable and mounted at runtime
RUN mkdir -p /workspace/reports

# ── Entrypoint ─────────────────────────────────────────────────────────────────
# Default: run the full live test suite.
# Override with any pytest command or the run_all.sh script.
ENTRYPOINT ["python3", "-m", "pytest"]
CMD ["tests/test_live_workloads.py", "tests/test_all_metrics.py", "-v", "--tb=short"]