#!/usr/bin/env bash
# install_deps.sh — provision a reliable kvboost dev environment.
#
# This version is designed to avoid "looks stuck" situations by:
# - preferring wheels over source builds
# - skipping expensive source builds when prerequisites are missing
# - adding timeouts around best-effort build steps
# - logging clearly what is happening
#
# Usage
# -----
#   ./install_deps.sh
#   ./install_deps.sh --cpu
#   ./install_deps.sh --skip-autoawq
#   ./install_deps.sh --skip-fa2
#   ./install_deps.sh --skip-flashinfer
#   ./install_deps.sh --skip-flash-attn
#   ./install_deps.sh --no-smoke-test
#
# Useful overrides
# -----------------
#   TORCH_CUDA_TAG=cu124
#   CUDA_HOME=/usr/local/cuda-12.4
#   MAX_JOBS=4
#   TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
#   FORCE_TORCH_REINSTALL=1
#   BUILDS_TIMEOUT_MIN=20
#
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

VENV_DIR="${REPO_ROOT}/env"
ENV_VARS_FILE="${REPO_ROOT}/env_vars.sh"
BUILD_LOG="${REPO_ROOT}/install_deps.log"

BUILD_AUTOAWQ=1
BUILD_FA2=1
BUILD_FLASHINFER=1
BUILD_FLASH_ATTN=1
CPU_ONLY=0
SMOKE_TEST=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-autoawq) BUILD_AUTOAWQ=0; shift ;;
        --skip-fa2) BUILD_FA2=0; shift ;;
        --skip-flashinfer) BUILD_FLASHINFER=0; shift ;;
        --skip-flash-attn) BUILD_FLASH_ATTN=0; shift ;;
        --cpu) CPU_ONLY=1; BUILD_AUTOAWQ=0; BUILD_FA2=0; BUILD_FLASHINFER=0; BUILD_FLASH_ATTN=0; shift ;;
        --no-smoke-test) SMOKE_TEST=0; shift ;;
        -h|--help)
            sed -n '1,120p' "$0"
            exit 0
            ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

log()  { printf "\n\033[1;36m[install_deps]\033[0m %s\n" "$*" | tee -a "$BUILD_LOG"; }
warn() { printf "\033[1;33m[install_deps] WARN:\033[0m %s\n" "$*" | tee -a "$BUILD_LOG" >&2; }
fail() { printf "\033[1;31m[install_deps] FATAL:\033[0m %s\n" "$*" | tee -a "$BUILD_LOG" >&2; exit 1; }

have_timeout() { command -v timeout >/dev/null 2>&1; }
RUN_TIMEOUT_MIN="${BUILDS_TIMEOUT_MIN:-20}"

run_best_effort() {
    local label="$1"; shift
    log "Starting: ${label}"
    if have_timeout; then
        if timeout "${RUN_TIMEOUT_MIN}m" "$@"; then
            log "Finished: ${label}"
            return 0
        fi
        warn "${label} failed or timed out after ${RUN_TIMEOUT_MIN}m; continuing"
        return 1
    fi
    if "$@"; then
        log "Finished: ${label}"
        return 0
    fi
    warn "${label} failed; continuing"
    return 1
}

command -v python3 >/dev/null || fail "python3 not found"
PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
log "python: $(python3 --version) (${PY_VER})"

GPU_NAME=""
COMPUTE_CAP=""
NVCC_VER=""
NVCC_MAJOR=""
NVCC_MINOR=""
CUDA_HOME_DETECTED=""

if (( CPU_ONLY == 0 )); then
    command -v nvidia-smi >/dev/null || fail "nvidia-smi not found — pass --cpu if you have no GPU"

    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/[[:space:]]\+$//')"
    COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
    log "GPU: ${GPU_NAME:-unknown} (compute cap ${COMPUTE_CAP:-unknown})"

    if command -v nvcc >/dev/null; then
        NVCC_VER="$(nvcc --version | awk -F'release ' '/release/ {print $2}' | awk -F',' '{print $1}' | head -1)"
        NVCC_MAJOR="${NVCC_VER%%.*}"
        NVCC_MINOR="${NVCC_VER#*.}"
        NVCC_MINOR="${NVCC_MINOR%%.*}"
        CUDA_HOME_DETECTED="$(dirname "$(dirname "$(command -v nvcc)")")"
        log "nvcc: ${NVCC_VER:-unknown}"
        log "CUDA_HOME detected as: ${CUDA_HOME_DETECTED}"
    else
        warn "nvcc not found in PATH; source builds will be skipped unless CUDA_HOME is set and nvcc becomes available"
    fi

    if [[ -n "${COMPUTE_CAP:-}" ]]; then
        TORCH_ARCH="${COMPUTE_CAP}"
    else
        TORCH_ARCH="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}"
    fi
    log "TORCH_CUDA_ARCH_LIST will be: ${TORCH_ARCH}"
fi

choose_torch_cuda_tag() {
    if [[ -n "${TORCH_CUDA_TAG:-}" && "${TORCH_CUDA_TAG}" != "auto" ]]; then
        echo "${TORCH_CUDA_TAG}"
        return 0
    fi

    if (( CPU_ONLY == 1 )); then
        echo "cpu"
        return 0
    fi

    if [[ -z "${NVCC_VER:-}" ]]; then
        echo "${TORCH_CUDA_TAG_DEFAULT:-cu124}"
        return 0
    fi

    case "${NVCC_MAJOR}.${NVCC_MINOR}" in
        11.*) echo "cu118" ;;
        12.0|12.1) echo "cu121" ;;
        12.2|12.3|12.4) echo "cu124" ;;
        12.5|12.6|12.7) echo "cu126" ;;
        12.8|12.9) echo "cu128" ;;
        13.*) echo "cu130" ;;
        *) echo "${TORCH_CUDA_TAG_DEFAULT:-cu124}" ;;
    esac
}

TORCH_CUDA_TAG="$(choose_torch_cuda_tag)"
log "torch wheel tag selected: ${TORCH_CUDA_TAG}"

install_torch() {
    if (( CPU_ONLY == 1 )); then
        log "Installing CPU-only torch"
        pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
        return 0
    fi

    if [[ -n "${FORCE_TORCH_REINSTALL:-}" ]]; then
        pip uninstall -y torch torchvision torchaudio >/dev/null 2>&1 || true
    fi

    local candidates=()
    case "${TORCH_CUDA_TAG}" in
        cu130) candidates=(cu130 cu128 cu126 cu124 cu121 cu118) ;;
        cu128) candidates=(cu128 cu126 cu124 cu121 cu118) ;;
        cu126) candidates=(cu126 cu124 cu121 cu118) ;;
        cu124) candidates=(cu124 cu121 cu118) ;;
        cu121) candidates=(cu121 cu124 cu118) ;;
        cu118) candidates=(cu118) ;;
        *)     candidates=("${TORCH_CUDA_TAG}" cu124 cu121 cu118) ;;
    esac

    local tag url
    for tag in "${candidates[@]}"; do
        url="https://download.pytorch.org/whl/${tag}"
        log "Installing torch from ${url}"
        if pip install --upgrade --index-url "${url}" torch torchvision; then
            log "torch install succeeded with ${tag}"
            TORCH_CUDA_TAG="${tag}"
            return 0
        fi
        warn "torch install failed for ${tag}"
    done

    fail "unable to install a CUDA-enabled torch wheel; set TORCH_CUDA_TAG explicitly or use --cpu"
}

log "Setting up virtualenv at ${VENV_DIR}"
if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
log "active python: $(which python)"
python -m pip install --upgrade pip wheel setuptools >/dev/null

log "Installing torch"
install_torch

TORCH_VER="$(python -c 'import torch; print(torch.__version__)')"
TORCH_CUDA="$(python -c 'import torch; print(torch.version.cuda or "none")')"
log "torch installed: ${TORCH_VER} (CUDA ${TORCH_CUDA})"

if (( CPU_ONLY == 0 )) && [[ -n "${NVCC_VER:-}" && "${TORCH_CUDA}" != "none" ]]; then
    TORCH_CUDA_MAJOR="${TORCH_CUDA%%.*}"
    if [[ "${TORCH_CUDA_MAJOR}" != "${NVCC_MAJOR}" ]]; then
        warn "torch CUDA ${TORCH_CUDA} does not match nvcc ${NVCC_VER} at the major version level; extension builds may fail"
    fi
fi

log "Installing kvboost (editable) with dev + streaming + server extras"
EXTRAS="dev,streaming,server"
if (( CPU_ONLY == 0 )); then
    EXTRAS="${EXTRAS},cuda"
fi
pip install -e ".[${EXTRAS}]"

prepare_cuda_build_env() {
    export CUDA_HOME="${CUDA_HOME:-${CUDA_HOME_DETECTED:-/usr/local/cuda}}"
    export PATH="${CUDA_HOME}/bin:${PATH}"

    local nvidia_includes=""
    nvidia_includes="$(find "${VENV_DIR}/lib/python${PY_VER}/site-packages/nvidia" \
        -name include -type d 2>/dev/null | grep -v cu13 | paste -sd: || true)"

    if [[ -n "${nvidia_includes}" ]]; then
        export CPATH="${nvidia_includes}:${CPATH:-}"
    fi

    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${TORCH_ARCH:-8.0;8.6;8.9;9.0}}"
    export MAX_JOBS="${MAX_JOBS:-4}"
    export PIP_PROGRESS_BAR="${PIP_PROGRESS_BAR:-off}"
    export PYTHONUNBUFFERED=1
}

if (( BUILD_AUTOAWQ == 1 )); then
    log "AutoAWQ kernels: best-effort install"
    if (( CPU_ONLY == 1 )); then
        warn "CPU-only mode; skipping AutoAWQ kernels"
    elif ! command -v nvcc >/dev/null; then
        warn "nvcc unavailable; skipping AutoAWQ kernels"
    else
        prepare_cuda_build_env
        log "CUDA_HOME=${CUDA_HOME}"
        log "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
        log "MAX_JOBS=${MAX_JOBS}"

        run_best_effort "autoawq-kernels install" \
            python -m pip install --force-reinstall --no-deps --no-build-isolation \
                "git+https://github.com/casper-hansen/AutoAWQ_kernels.git"

        if python -c "import awq_ext" 2>/dev/null; then
            log "awq_ext built successfully"
        else
            warn "awq_ext import failed; this is non-fatal"
        fi
    fi
fi

install_flash_attn() {
    if (( CPU_ONLY == 1 )); then
        warn "CPU-only mode; skipping FlashAttention-2"
        return 0
    fi

    if python - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("flash_attn") else 1)
PY
    then
        log "FlashAttention-2 already installed"
        return 0
    fi

    prepare_cuda_build_env
    log "FlashAttention build env: CUDA_HOME=${CUDA_HOME}, TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}, MAX_JOBS=${MAX_JOBS}"

    log "Installing FlashAttention-2 (wheel first, then source build if required)"
    if pip install --upgrade --no-cache-dir flash-attn --no-build-isolation; then
        if python -c "import flash_attn" 2>/dev/null; then
            log "flash_attn imported OK"
        else
            warn "flash-attn installed but is not importable yet; continuing"
        fi
    else
        warn "FlashAttention-2 install failed; continuing without it"
    fi
}

if (( BUILD_FA2 == 1 )); then
    install_flash_attn
fi

install_flashinfer() {
    if (( CPU_ONLY == 1 )); then
        warn "CPU-only mode; skipping FlashInfer"
        return 0
    fi

    if python -c "import flashinfer" 2>/dev/null; then
        log "FlashInfer already installed"
        return 0
    fi

    log "Installing FlashInfer"
    if pip install --upgrade flashinfer-python; then
        if python -c "import flashinfer" 2>/dev/null; then
            log "flashinfer imported OK"
        else
            warn "flashinfer installed but import failed; continuing"
        fi
    else
        warn "FlashInfer install failed; continuing without it"
    fi
}

if (( BUILD_FLASHINFER == 1 )); then
    install_flashinfer
fi

if (( BUILD_FLASH_ATTN == 1 )); then
    log "Building kvboost legacy bundled flash-attn extension (best effort)"
    if (( CPU_ONLY == 1 )); then
        warn "CPU-only mode; skipping bundled flash-attn extension"
    elif ! command -v nvcc >/dev/null; then
        warn "nvcc unavailable; skipping bundled flash-attn extension"
    else
        if [[ -z "${COMPUTE_CAP:-}" ]]; then
            warn "Could not determine compute capability; skipping bundled flash-attn extension"
        else
            MAJOR_CC="${COMPUTE_CAP%%.*}"
            if (( MAJOR_CC < 8 )); then
                warn "compute capability ${COMPUTE_CAP} < 8.0; skipping bundled flash-attn extension"
            else
                prepare_cuda_build_env
                export FORCE_CUDA=1
                run_best_effort "bundled flash-attn extension" \
                    pip install -e . --no-deps --force-reinstall
            fi
        fi
    fi
fi

log "Writing ${ENV_VARS_FILE}"
TORCH_LIB="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
NVIDIA_INCLUDES="$(find "${VENV_DIR}/lib/python${PY_VER}/site-packages/nvidia" \
    -name include -type d 2>/dev/null | grep -v cu13 | paste -sd: || true)"

cat > "${ENV_VARS_FILE}" <<EOF
# Auto-generated by install_deps.sh. Source before running kvboost:
#   source ./env_vars.sh

export VIRTUAL_ENV="${VENV_DIR}"
export PATH="\${VIRTUAL_ENV}/bin:\${PATH}"
export LD_LIBRARY_PATH="${TORCH_LIB}:\${LD_LIBRARY_PATH:-}"
export CPATH="${NVIDIA_INCLUDES}:\${CPATH:-}"
export CUDA_HOME="${CUDA_HOME:-${CUDA_HOME_DETECTED:-/usr/local/cuda}}"
export TORCH_CUDA_ARCH_LIST="${TORCH_ARCH:-${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}}"

alias kvboost-shell='python -c "import kvboost; print(kvboost.__version__)" && python'
EOF

if (( SMOKE_TEST == 1 )); then
    log "Smoke-testing the install"
    # shellcheck disable=SC1090
    source "${ENV_VARS_FILE}"

    python - <<'PYTEST'
import sys

ok = True

def check(msg, fn):
    global ok
    try:
        fn()
        print(f"  OK   {msg}")
    except Exception as exc:
        print(f"  FAIL {msg}: {type(exc).__name__}: {exc}")
        ok = False

check("import kvboost", lambda: __import__("kvboost"))
check("import kvboost.speculative", lambda: __import__("kvboost.speculative"))
check("import kvboost.streaming", lambda: __import__("kvboost.streaming"))
check("import torch", lambda: __import__("torch"))
check("KVBoost top-level export", lambda: getattr(__import__("kvboost"), "KVBoost"))
check("SpeculativeConfig importable", lambda: __import__("kvboost.speculative", fromlist=["SpeculativeConfig"]).SpeculativeConfig)

try:
    import torch
    print(f"  info torch={torch.__version__} cuda={torch.version.cuda} avail={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  info device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
except Exception:
    pass

for _mod, _note in (("flash_attn", "FlashAttention-2 / prefill attention"),
                    ("flashinfer", "FlashInfer / decode attention")):
    try:
        __import__(_mod)
        print(f"  OK   {_mod} ({_note})")
    except Exception:
        print(f"  WARN {_mod} not importable ({_note}); fallback may be used")

sys.exit(0 if ok else 1)
PYTEST
fi

log "Install complete."
log "Activate this env in a new shell:"
log "  source ${ENV_VARS_FILE}"
log ""
log "Run unit tests:"
log "  pytest tests/speculative/ -v"
log ""
log "Run a speculative-decode smoke test:"
log "  python -m kvboost.streaming.demo_speculative \\"
log "      --model       Qwen/Qwen2.5-7B-Instruct-AWQ \\"
log "      --draft-model Qwen/Qwen2.5-1.5B-Instruct-AWQ \\"
log "      --mode full_resident --gamma 5 --max-new-tokens 60 \\"
log "      --prompt 'Explain entropy in two sentences.'"
