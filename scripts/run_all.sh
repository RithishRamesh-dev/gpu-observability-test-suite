#!/usr/bin/env bash
# scripts/run_all.sh
# One-command GPU observability validation.
# Usage: sudo ./scripts/run_all.sh [--no-llm] [--no-prometheus]
#
# Steps:
#   1. Preflight checks (NVIDIA drivers, DCGM, Docker)
#   2. Start Prometheus + Grafana + DCGM Exporter (optional)
#   3. Collect 30s idle baseline
#   4. Run all GPU workloads via orchestrator
#   5. Validate all metrics
#   6. Open report

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

REPORT_DIR="reports"
TELEMETRY_FILE="$REPORT_DIR/telemetry.json"
REPORT_FILE="$REPORT_DIR/validation_report.html"
NO_LLM=false
NO_PROMETHEUS=false
WORKLOADS="cuda_matmul_fp16,cuda_matmul_fp32,memory_stress,pcie_stress,thermal_stress"

# Parse args
for arg in "$@"; do
  case $arg in
    --no-llm) NO_LLM=true;;
    --no-prometheus) NO_PROMETHEUS=true;;
    --workloads=*) WORKLOADS="${arg#*=}";;
    --help)
      echo "Usage: $0 [--no-llm] [--no-prometheus] [--workloads=w1,w2,...]"
      exit 0
      ;;
  esac
done

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log_info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
log_ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }
log_step()  { echo -e "\n${BOLD}${CYAN}══════ $* ══════${NC}\n"; }

# ─── Preflight ───────────────────────────────────────────────────────────────
log_step "Preflight Checks"

if ! command -v nvidia-smi &>/dev/null; then
  log_error "nvidia-smi not found. Install NVIDIA drivers."
  exit 1
fi
GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)
log_ok "GPU detected: $GPU_INFO"

if ! command -v python3 &>/dev/null; then
  log_error "python3 not found."
  exit 1
fi
log_ok "Python3: $(python3 --version)"

# Check PyTorch CUDA
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  log_warn "PyTorch CUDA not available. Installing requirements..."
  pip install -r requirements.txt --quiet
fi
log_ok "PyTorch with CUDA available"

if command -v dcgmi &>/dev/null; then
  log_ok "DCGM available: $(dcgmi --version 2>/dev/null | head -1 || echo 'ok')"
  USE_DCGM=true
else
  log_warn "DCGM not found — will use nvidia-smi only (add --no-dcgm flag to collector)"
  USE_DCGM=false
fi

mkdir -p "$REPORT_DIR"
log_ok "Report directory: $REPORT_DIR"

# ─── Start Prometheus Stack (optional) ───────────────────────────────────────
if [ "$NO_PROMETHEUS" = false ] && command -v docker &>/dev/null; then
  log_step "Starting Prometheus + Grafana Stack"
  if docker ps -q -f name=dcgm-exporter | grep -q .; then
    log_ok "DCGM Exporter already running"
  else
    log_info "Starting DCGM Exporter..."
    docker run -d --gpus all \
      --cap-add SYS_ADMIN \
      --rm \
      -p 9400:9400 \
      --name dcgm-exporter \
      nvcr.io/nvidia/k8s/dcgm-exporter:3.3.0-3.2.0-ubuntu22.04 \
      2>/dev/null || log_warn "Failed to start DCGM Exporter (non-fatal)"
  fi

  if [ -f "dashboards/docker-compose.yml" ]; then
    log_info "Starting Prometheus + Grafana..."
    docker compose -f dashboards/docker-compose.yml up -d 2>/dev/null || \
      log_warn "docker compose failed (non-fatal — continuing without Prometheus)"
    sleep 5
    log_ok "Prometheus: http://localhost:9090"
    log_ok "Grafana:    http://localhost:3000 (admin/admin)"
  fi
else
  log_info "Skipping Prometheus stack (--no-prometheus or docker unavailable)"
fi

# ─── Add LLM workload if enabled ─────────────────────────────────────────────
if [ "$NO_LLM" = false ]; then
  if python3 -c "import vllm" 2>/dev/null; then
    WORKLOADS="${WORKLOADS},llm_inference"
    log_ok "vLLM found — adding LLM inference workload"
  else
    log_warn "vLLM not installed — skipping LLM workload. To install: pip install vllm"
  fi
fi

log_step "Running GPU Workloads + Telemetry Collection"
log_info "Workloads: $WORKLOADS"
log_info "Telemetry: $TELEMETRY_FILE"

DCGM_FLAG=""
if [ "$USE_DCGM" = false ]; then
  DCGM_FLAG="--no-dcgm"
fi

# Run orchestrator (which starts collector internally)
python3 scripts/orchestrator.py \
  --config configs/workload_config.yaml \
  --telemetry "$TELEMETRY_FILE" \
  --workloads "$WORKLOADS"

# ─── Validate Metrics ─────────────────────────────────────────────────────────
log_step "Validating Metrics"
python3 validators/run_validators.py \
  --telemetry "$TELEMETRY_FILE" \
  --config configs/thresholds.yaml \
  --output "$REPORT_FILE"

# ─── Done ─────────────────────────────────────────────────────────────────────
log_step "Complete"
echo -e "${GREEN}${BOLD}"
echo "  ╔═══════════════════════════════════════════════════╗"
echo "  ║   GPU Observability Validation Complete!          ║"
echo "  ╚═══════════════════════════════════════════════════╝"
echo -e "${NC}"
echo -e "  📊 Validation Report: ${CYAN}$REPORT_FILE${NC}"
echo -e "  📁 Telemetry Data:    ${CYAN}$TELEMETRY_FILE${NC}"
if [ "$NO_PROMETHEUS" = false ] && command -v docker &>/dev/null; then
echo -e "  📈 Prometheus:        ${CYAN}http://localhost:9090${NC}"
echo -e "  📉 Grafana:           ${CYAN}http://localhost:3000${NC}"
fi
echo ""

# Try to open report
if command -v xdg-open &>/dev/null; then
  xdg-open "$REPORT_FILE" 2>/dev/null || true
elif command -v open &>/dev/null; then
  open "$REPORT_FILE" 2>/dev/null || true
fi
