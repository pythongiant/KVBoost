#!/usr/bin/env bash
# install_deps_version_agnostic.sh
#
# Goal
# ----
# Make the install flow resilient across CUDA toolkit versions without hardcoding
# exact CUDA->Torch build tables.
#
# What this script does
# ---------------------
# 1. Creates/reuses a virtualenv at ./env
# 2. Installs a CUDA-enabled PyTorch by trying the newest supported wheel index
#    first, then falling back until one works
# 3. Installs kvboost in editable mode
# 4. Installs FlashAttention-2 and FlashInfer as best-effort accelerators
# 5. Optionally installs AutoAWQ kernels and the legacy bundled flash-attn ext
# 6. Writes ./env_vars.sh for convenient activation
# 7. Runs a smoke test
#
# Design principles
# ------------------
# - No hardcoded "CUDA 12.4 -> cu124" style mappings
# - Prefer trying supported upstream wheel indexes instead of encoding policy
# - Use the local CUDA toolkit only for source builds and arch detection
# - Skip expensive source builds unless explicitly requested
# - Fail only when the base environment is broken; optional accelerators are
#   best-effort unless --strict is supplied

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

VENV_DIR="${REPO_ROOT}/env"
ENV_VARS_FILE="${REPO_ROOT}/env_vars.sh"
BUILD_LOG="${REPO_ROOT}/install_deps.log"

CPU_ONLY=0
SMOKE_TEST=1
STRICT=0

BUILD_AUTOAWQ=0
BUILD_AUTOAWQ_SOURCE=0
BUILD_FA2=1
BUILD_FLASHINFER=1
BUILD_FLASH_ATTN=0

TORCH_CUDA_INDEX="${TORCH_CUDA_INDEX:-auto}"
MAX_JOBS="${MAX_JOBS:-4}"
BUILDS_TIMEOUT_MIN="${BUILDS_TIMEOUT_MIN:-20}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu) CPU_ONLY=1; BUILD_FA2=0; BUILD_FLASHINFER=0; BUILD_FLASH_ATTN=0; shift ;;
        --skip-fa2) BUILD_FA2=0; shift ;;
        --skip-flashinfer) BUILD_FLASHINFER=0; shift ;;
        --skip-flash-attn) BUILD_FLASH_ATTN=0; shift ;;
        --skip-autoawq) BUILD_AUTOAWQ=0; BUILD_AUTOAWQ_SOURCE=0; shift ;;
        --autoawq) BUILD_AUTOAWQ=1; shift ;;
        --autoawq-source) BUILD_AUTOAWQ=1; BUILD_AUTOAWQ_SOURCE=1; shift ;;
        --strict) STRICT=1; shift ;;
        --no-smoke-test) SMOKE_TEST=0; shift ;;
        -h|--help)
            sed -n '1,220p' "$0"
            exit 0
            ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

log()  { printf "\n\033[1;36m[install_deps]\033[0m %s\n" "$*" | tee -a "$BUILD_LOG"; }
warn() { printf "\033[1;33m[install_deps] WARN:\033[0m %s\n" "$*" | tee -a "$BUILD_LOG" >&2; }
fail() { printf "\033[1;31m[install_deps] FATAL:\033[0m %s\n" "$*" | tee -a "$BUILD_LOG" >&2; exit 1; }

have_timeout() { command -v timeout >/dev/null 2>&1; }

run_best_effort() {
    local label="$1"
    shift

    log "Starting: ${label}"
    if have_timeout; then
        if timeout "${BUILDS_TIMEOUT_MIN}m" "$@"; then
            log "Finished: ${label}"
            return 0
        fi
        warn "${label} failed or timed out after ${BUILDS_TIMEOUT_MIN}m; continuing"
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
CUDA_HOME_DETECTED=""
ARCH_LIST_DEFAULT=""

if (( CPU_ONLY == 0 )); then
    command -v nvidia-smi >/dev/null || fail "nvidia-smi not found — pass --cpu if you have no GPU"

    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/[[:space:]]\+$//')"
    COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
    log "GPU: ${GPU_NAME:-unknown} (compute cap ${COMPUTE_CAP:-unknown})"

    if command -v nvcc >/dev/null; then
        NVCC_VER="$(nvcc --version | awk -F'release ' '/release/ {print $2}' | awk -F',' '{print $1}' | head -1)"
        CUDA_HOME_DETECTED="$(dirname "$(dirname "$(command -v nvcc)")")"
        log "nvcc: ${NVCC_VER:-unknown}"
        log "CUDA_HOME detected as: ${CUDA_HOME_DETECTED}"
    else
        warn "nvcc not found in PATH; source builds may be skipped"
    fi

    if [[ -n "${COMPUTE_CAP:-}" ]]; then
        ARCH_LIST_DEFAULT="${COMPUTE_CAP}"
    else
        ARCH_LIST_DEFAULT="8.0;8.6;8.9;9.0"
    fi
    log "TORCH_CUDA_ARCH_LIST default: ${ARCH_LIST_DEFAULT}"
fi

install_torch_cpu() {
    log "Installing CPU-only torch stack"
    pip install --upgrade --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio
}

install_torch_cuda() {
    local candidates=("${TORCH_CUDA_INDEX}")
    if [[ "${TORCH_CUDA_INDEX}" == "auto" ]]; then
        candidates=(cu130 cu128 cu126 cu124 cu121 cu118)
    fi

    local tag url
    for tag in "${candidates[@]}"; do
        url="https://download.pytorch.org/whl/${tag}"
        log "Trying torch wheel index: ${url}"
        if pip install --upgrade --index-url "${url}" torch torchvision torchaudio; then
            TORCH_CUDA_INDEX="${tag}"
            log "torch install succeeded via ${tag}"
            return 0
        fi
        warn "torch install failed via ${tag}"
    done

    return 1
}

if [[ ! -d "${VENV_DIR}" ]]; then
    log "Creating virtualenv at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
else
    log "Reusing virtualenv at ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
log "active python: $(which python)"
python -m pip install --upgrade pip wheel setuptools >/dev/null

log "Installing torch"
if (( CPU_ONLY == 1 )); then
    install_torch_cpu
else
    if ! install_torch_cuda; then
        if [[ "${STRICT}" == "1" ]]; then
            fail "unable to install a CUDA-enabled torch wheel"
        fi
        warn "unable to install CUDA-enabled torch wheel; falling back to CPU-only torch"
        install_torch_cpu
        CPU_ONLY=1
    fi
fi

TORCH_VER="$(python -c 'import torch; print(torch.__version__)')"
TORCH_CUDA="$(python -c 'import torch; print(torch.version.cuda or "none")')"
log "torch installed: ${TORCH_VER} (CUDA ${TORCH_CUDA})"

if (( CPU_ONLY == 0 )) && [[ "${TORCH_CUDA}" == "none" ]]; then
    if [[ "${STRICT}" == "1" ]]; then
        fail "CUDA mode requested but torch installed without CUDA support"
    fi
    warn "torch has no CUDA support; continuing in CPU mode"
    CPU_ONLY=1
fi

if (( CPU_ONLY == 0 )) && [[ -n "${NVCC_VER:-}" && "${TORCH_CUDA}" != "none" ]]; then
    TORCH_CUDA_MAJOR="${TORCH_CUDA%%.*}"
    NVCC_MAJOR="${NVCC_VER%%.*}"
    if [[ "${TORCH_CUDA_MAJOR}" != "${NVCC_MAJOR}" ]]; then
        warn "torch CUDA ${TORCH_CUDA} and nvcc ${NVCC_VER} differ at the major level; this can still work for wheel installs, but source builds may fail"
    fi
fi

log "Installing kvboost (editable)"
EXTRAS="dev,streaming,server"
if (( CPU_ONLY == 0 )); then
    EXTRAS="${EXTRAS},cuda"
fi
pip install -e ".[${EXTRAS}]"

prepare_cuda_build_env() {
    export CUDA_HOME="${CUDA_HOME:-${CUDA_HOME_DETECTED:-/usr/local/cuda}}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export MAX_JOBS="${MAX_JOBS:-4}"
    export PYTHONUNBUFFERED=1
    export PIP_PROGRESS_BAR="${PIP_PROGRESS_BAR:-off}"

    local nvidia_includes=""
    nvidia_includes="$(find "${VENV_DIR}/lib/python${PY_VER}/site-packages/nvidia" \
        -name include -type d 2>/dev/null | grep -v cu13 | paste -sd: || true)"
    if [[ -n "${nvidia_includes}" ]]; then
        export CPATH="${nvidia_includes}:${CPATH:-}"
    fi

    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${ARCH_LIST_DEFAULT}}"
}

install_autoawq() {
    if (( CPU_ONLY == 1 )); then
        warn "CPU-only mode; skipping AutoAWQ kernels"
        return 0
    fi

    if python -c "import awq_ext" >/dev/null 2>&1; then
        log "awq_ext already importable"
        return 0
    fi

    log "Attempting AutoAWQ kernels via wheel"
    if pip install --upgrade autoawq-kernels; then
        if python -c "import awq_ext" >/dev/null 2>&1; then
            log "awq_ext imported OK"
            return 0
        fi
        warn "autoawq-kernels installed but awq_ext still not importable"
    fi

    if (( BUILD_AUTOAWQ_SOURCE == 0 )); then
        warn "AutoAWQ source build disabled; skipping"
        return 0
    fi

    if ! command -v nvcc >/dev/null; then
        warn "nvcc unavailable; skipping AutoAWQ source build"
        return 0
    fi

    prepare_cuda_build_env
    log "CUDA_HOME=${CUDA_HOME}"
    log "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
    log "MAX_JOBS=${MAX_JOBS}"

    run_best_effort "AutoAWQ source build" \
        python -m pip install --force-reinstall --no-deps --no-build-isolation \
            "git+https://github.com/casper-hansen/AutoAWQ_kernels.git"

    if python -c "import awq_ext" >/dev/null 2>&1; then
        log "awq_ext built successfully"
    else
        warn "AutoAWQ source build did not produce an importable awq_ext"
    fi
}

install_flashattn() {
    if (( CPU_ONLY == 1 )); then
        warn "CPU-only mode; skipping FlashAttention-2"
        return 0
    fi

    if python -c "import flash_attn" >/dev/null 2>&1; then
        log "FlashAttention-2 already importable"
        return 0
    fi

    prepare_cuda_build_env
    log "Building/installing FlashAttention-2"
    log "CUDA_HOME=${CUDA_HOME}"
    log "TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
    log "MAX_JOBS=${MAX_JOBS}"

    run_best_effort "flash-attn install" python -m pip install --upgrade --no-cache-dir flash-attn --no-build-isolation || true

    if python -c "import flash_attn" >/dev/null 2>&1; then
        log "flash_attn imported OK"
        return 0
    fi

    if [[ "${STRICT}" == "1" ]]; then
        fail "FlashAttention-2 is required but not importable"
    fi

    warn "FlashAttention-2 not importable; continuing without it"
    return 0
}

install_flashinfer() {
    if (( CPU_ONLY == 1 )); then
        warn "CPU-only mode; skipping FlashInfer"
        return 0
    fi

    if python -c "import flashinfer" >/dev/null 2>&1; then
        log "FlashInfer already importable"
        return 0
    fi

    log "Installing FlashInfer"
    run_best_effort "flashinfer install" python -m pip install --upgrade flashinfer-python || true

    if python -c "import flashinfer" >/dev/null 2>&1; then
        log "flashinfer imported OK"
        return 0
    fi

    if [[ "${STRICT}" == "1" ]]; then
        fail "FlashInfer is required but not importable"
    fi

    warn "FlashInfer not importable; continuing without it"
    return 0
}

install_legacy_flash_attn_ext() {
    if (( BUILD_FLASH_ATTN == 0 )); then
        log "Legacy bundled flash-attn ext: skipped"
        return 0
    fi

    if (( CPU_ONLY == 1 )); then
        warn "CPU-only mode; skipping legacy bundled flash-attn ext"
        return 0
    fi

    if ! command -v nvcc >/dev/null; then
        warn "nvcc unavailable; skipping legacy bundled flash-attn ext"
        return 0
    fi

    prepare_cuda_build_env
    export FORCE_CUDA=1

    run_best_effort "legacy bundled flash-attn ext" python -m pip install -e . --no-deps --force-reinstall

    return 0
}

if (( BUILD_AUTOAWQ == 1 )); then
    install_autoawq
fi

if (( BUILD_FA2 == 1 )); then
    install_flashattn
fi

if (( BUILD_FLASHINFER == 1 )); then
    install_flashinfer
fi

install_legacy_flash_attn_ext

log "Writing ${ENV_VARS_FILE}"
TORCH_LIB="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
NVIDIA_INCLUDES="$(find "${VENV_DIR}/lib/python${PY_VER}/site-packages/nvidia" \
    -name include -type d 2>/dev/null | grep -v cu13 | paste -sd: || true)"

cat > "${ENV_VARS_FILE}" <<EOF
# Auto-generated by install_deps_version_agnostic.sh. Source before running kvboost:
#   source ./env_vars.sh

export VIRTUAL_ENV="${VENV_DIR}"
export PATH="\${VIRTUAL_ENV}/bin:\${PATH}"
export LD_LIBRARY_PATH="${TORCH_LIB}:\${LD_LIBRARY_PATH:-}"
export CPATH="${NVIDIA_INCLUDES}:\${CPATH:-}"
export CUDA_HOME="${CUDA_HOME:-${CUDA_HOME_DETECTED:-/usr/local/cuda}}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-${ARCH_LIST_DEFAULT}}"

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

for _mod, _note in (("flash_attn", "FlashAttention-2"),
                    ("flashinfer", "FlashInfer"),
                    ("awq_ext", "AutoAWQ kernels")):
    try:
        __import__(_mod)
        print(f"  OK   {_mod} ({_note})")
    except Exception:
        print(f"  WARN {_mod} not importable ({_note})")

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
log "  python -m kvboost.streaming.demo_speculative \\\"
log "      --model       Qwen/Qwen2.5-7B-Instruct-AWQ \\\"
log "      --draft-model Qwen/Qwen2.5-1.5B-Instruct-AWQ \\\"
log "      --mode full_resident --gamma 5 --max-new-tokens 60 \\\"
log "      --prompt 'Explain entropy in two sentences.'"
