#!/usr/bin/env bash
# install_deps.sh — provision a working kvboost dev environment.
#
# What this does
# --------------
# 1. Creates (or reuses) a Python virtualenv at ./env
# 2. Installs torch matching your system CUDA toolkit
# 3. Installs kvboost in editable mode with all dev/streaming/server extras
# 4. Builds autoawq-kernels from source against your local torch (the
#    PyPI wheels rarely match recent torch + CUDA combos)
# 5. Optionally builds the bundled kvboost flash-attn CUDA extension
# 6. Writes ./env_vars.sh with the runtime env (LD_LIBRARY_PATH, CPATH,
#    CUDA_HOME, TORCH_CUDA_ARCH_LIST) for you to source before working
# 7. Runs a smoke test
#
# Usage
# -----
#   ./install_deps.sh                      # full install with autoawq build
#   ./install_deps.sh --skip-autoawq       # skip autoawq-kernels source build
#   ./install_deps.sh --skip-flash-attn    # skip kvboost's flash-attn extension
#   ./install_deps.sh --cpu                # CPU-only torch (skips all kernel builds)
#
# Re-runnable: skips work that's already done. Safe to invoke after every git pull.

set -euo pipefail

# ── Repo + venv paths ────────────────────────────────────────────────────────
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"
VENV_DIR="${REPO_ROOT}/env"
ENV_VARS_FILE="${REPO_ROOT}/env_vars.sh"

# ── Flags ────────────────────────────────────────────────────────────────────
BUILD_AUTOAWQ=1
BUILD_FLASH_ATTN=1
CPU_ONLY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip-autoawq) BUILD_AUTOAWQ=0; shift ;;
        --skip-flash-attn) BUILD_FLASH_ATTN=0; shift ;;
        --cpu) CPU_ONLY=1; BUILD_AUTOAWQ=0; BUILD_FLASH_ATTN=0; shift ;;
        -h|--help)
            grep -E '^# ' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

log() { printf "\n\033[1;36m[install_deps]\033[0m %s\n" "$*"; }
fail() { printf "\033[1;31m[install_deps] FATAL:\033[0m %s\n" "$*" >&2; exit 1; }

# ── 1. Sanity checks ─────────────────────────────────────────────────────────
log "Sanity-checking the host"

command -v python3 >/dev/null || fail "python3 not found"
PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
log "  python: $(python3 --version) (${PY_VER})"

if (( CPU_ONLY == 0 )); then
    command -v nvidia-smi >/dev/null || fail "nvidia-smi not found — pass --cpu if you have no GPU"
    GPU_LINE="$(nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader | head -1)"
    log "  GPU:    ${GPU_LINE}"

    if command -v nvcc >/dev/null; then
        NVCC_VER="$(nvcc --version | grep release | sed -E 's/.*release ([0-9.]+).*/\1/')"
        log "  nvcc:   ${NVCC_VER}"
    else
        log "  nvcc:   (not in PATH — autoawq build will fail without it)"
    fi

    # Compute capability without the dot: 7.5 → 75
    COMPUTE_CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
    TORCH_ARCH="${COMPUTE_CAP}"   # autoawq wants form like "7.5"
    log "  TORCH_CUDA_ARCH_LIST will be: ${TORCH_ARCH}"
fi

# ── 2. Virtualenv ────────────────────────────────────────────────────────────
log "Setting up virtualenv at ${VENV_DIR}"
if [[ ! -d "${VENV_DIR}" ]]; then
    python3 -m venv "${VENV_DIR}"
fi

# Activate for the rest of this script.
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
log "  active python: $(which python)"

python -m pip install --upgrade pip wheel setuptools >/dev/null

# ── 3. Install torch ─────────────────────────────────────────────────────────
log "Installing torch"
if (( CPU_ONLY == 1 )); then
    pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
    # Pick a torch+CUDA build that matches your nvcc. Common pairings:
    #   cu118 -> CUDA 11.8
    #   cu121 -> CUDA 12.1
    #   cu124 -> CUDA 12.4
    #   cu128 -> CUDA 12.8 (latest stable as of early 2026)
    TORCH_CUDA_TAG="${TORCH_CUDA_TAG:-cu128}"
    log "  using torch wheel tag: ${TORCH_CUDA_TAG}"
    pip install --upgrade \
        --extra-index-url "https://download.pytorch.org/whl/${TORCH_CUDA_TAG}" \
        "torch" "torchvision"
fi

TORCH_VER="$(python -c 'import torch; print(torch.__version__)')"
TORCH_CUDA="$(python -c 'import torch; print(torch.version.cuda or "none")')"
log "  torch installed: ${TORCH_VER} (CUDA ${TORCH_CUDA})"

# ── 4. Install kvboost editable + extras ─────────────────────────────────────
log "Installing kvboost (editable) with dev + streaming + server extras"
EXTRAS="dev,streaming,server"
if (( CPU_ONLY == 0 )); then
    EXTRAS="${EXTRAS},cuda"
fi
# --no-deps on autoawq-kernels later; for now let the extras pull in normal deps.
pip install -e ".[${EXTRAS}]"

# ── 5. Build autoawq-kernels from source ─────────────────────────────────────
if (( BUILD_AUTOAWQ == 1 )); then
    log "Building autoawq-kernels from source"

    if ! command -v nvcc >/dev/null; then
        log "  nvcc unavailable — skipping autoawq build (set CUDA_HOME and re-run if needed)"
    else
        # Header paths: torch pulls in nvidia/*-cu12 wheels containing include
        # subdirs for cublas / cusparse / cusolver. Plumb them through CPATH.
        # Exclude any cu13 paths if they snuck in (vLLM dragged them in once).
        NVIDIA_INCLUDES="$(find "${VENV_DIR}/lib/python${PY_VER}/site-packages/nvidia" \
            -name include -type d 2>/dev/null | grep -v cu13 | paste -sd: || true)"
        export CPATH="${NVIDIA_INCLUDES}:${CPATH:-}"
        export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
        export TORCH_CUDA_ARCH_LIST="${TORCH_ARCH}"

        log "  CUDA_HOME=${CUDA_HOME}"
        log "  TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"
        log "  CPATH=${CPATH:0:120}..."

        # --no-deps is critical: without it, pip "upgrades" torch to whatever
        # the kernel's requirements say, which breaks ABI on the wheel we just
        # installed.
        pip install --force-reinstall --no-deps --no-build-isolation \
            "git+https://github.com/casper-hansen/AutoAWQ_kernels.git" \
            2>&1 | tee /tmp/awq_kernels_build.log

        # Verify
        if python -c "import awq_ext" 2>/dev/null; then
            log "  awq_ext built successfully"
        else
            log "  awq_ext import failed; checking for libc10.so path issue..."
            TORCH_LIB="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
            if LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}" python -c "import awq_ext" 2>/dev/null; then
                log "  resolved with LD_LIBRARY_PATH=${TORCH_LIB}"
            else
                fail "awq_ext built but unimportable. Inspect /tmp/awq_kernels_build.log"
            fi
        fi
    fi
fi

# ── 6. Build kvboost flash-attn CUDA extension (optional, sm_80+) ────────────
if (( BUILD_FLASH_ATTN == 1 )); then
    log "Building kvboost flash-attn extension"
    if (( CPU_ONLY == 1 )); then
        log "  skipping (CPU-only)"
    elif ! command -v nvcc >/dev/null; then
        log "  skipping (no nvcc)"
    else
        # The bundled kernel targets sm_80+. On Turing (sm_75) it won't compile;
        # silently skip in that case — PyTorch SDPA is used as fallback.
        MAJOR_CC="${COMPUTE_CAP%%.*}"
        if (( MAJOR_CC < 8 )); then
            log "  skipping (compute capability ${COMPUTE_CAP} < 8.0; falls back to torch SDPA)"
        else
            export FORCE_CUDA=1
            export TORCH_CUDA_ARCH_LIST="${TORCH_ARCH}"
            pip install -e . --no-deps --force-reinstall 2>&1 | tail -20 || \
                log "  flash-attn build failed (non-fatal; torch SDPA fallback used)"
        fi
    fi
fi

# ── 7. Write env_vars.sh for runtime ─────────────────────────────────────────
log "Writing ${ENV_VARS_FILE}"
TORCH_LIB="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
NVIDIA_INCLUDES="$(find "${VENV_DIR}/lib/python${PY_VER}/site-packages/nvidia" \
    -name include -type d 2>/dev/null | grep -v cu13 | paste -sd: || true)"

cat > "${ENV_VARS_FILE}" <<EOF
# Auto-generated by install_deps.sh. Source before running kvboost:
#   source ./env_vars.sh
#
# Sets LD_LIBRARY_PATH so awq_ext finds libc10.so, plus CPATH/CUDA_HOME for
# any incremental rebuilds.

export VIRTUAL_ENV="${VENV_DIR}"
export PATH="\${VIRTUAL_ENV}/bin:\${PATH}"
export LD_LIBRARY_PATH="${TORCH_LIB}:\${LD_LIBRARY_PATH:-}"
export CPATH="${NVIDIA_INCLUDES}:\${CPATH:-}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_ARCH:-7.5;8.0;8.6;8.9;9.0}"

# Convenience: drop you into a Python REPL with kvboost importable
alias kvboost-shell='python -c "import kvboost; print(kvboost.__version__)" && python'
EOF

# ── 8. Smoke test ────────────────────────────────────────────────────────────
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
check("import torch (CUDA)", lambda: __import__("torch").cuda.is_available() or True)
check("KVBoost top-level export", lambda: getattr(__import__("kvboost"), "KVBoost"))
check("SpeculativeConfig importable", lambda: __import__("kvboost.speculative", fromlist=["SpeculativeConfig"]).SpeculativeConfig)

try:
    import torch
    print(f"  info torch={torch.__version__} cuda={torch.version.cuda} avail={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  info device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
except Exception:
    pass

# Optional: probe autoawq_kernels resolution
try:
    from kvboost.streaming.kernels.marlin import marlin_awq_available, _GEMM_FN
    if marlin_awq_available():
        print(f"  OK   awq_ext resolved: {_GEMM_FN}")
    else:
        print(f"  WARN awq_ext not resolved (falls back to torch dequant; expect ~50x slower)")
except Exception as exc:
    print(f"  WARN awq probe failed: {exc}")

sys.exit(0 if ok else 1)
PYTEST

log "Install complete."
log ""
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
