#!/usr/bin/env bash
# install_deps.sh — device-agnostic KVBoost environment setup.
#
# Auto-detects the box and does the right thing on all three:
#   * CPU-only (no GPU / no nvcc)      -> CPU torch; runtime uses torch SDPA
#   * GPU, no nvcc                     -> CUDA torch + flash-attn (prebuilt wheel) + FlashInfer
#   * GPU + nvcc (CUDA 12.x / 13.x)    -> full path incl. bundled kernel + flash-attn
#
# flash-attn is REQUIRED on any GPU box (it's the prefill backend you want): a
# matching prebuilt wheel is installed when available (no nvcc needed), with a
# source build as fallback. The install fails loudly if it can't be installed,
# so you never silently end up on SDPA. The bundled kernel and FlashInfer stay
# best-effort (the repo falls back to SDPA for those). Use --skip-flash-attn to
# opt out. Every build is time-boxed and logged to install_deps.log.
#
# Usage
# -----
#   ./install_deps.sh                 # auto-detect
#   ./install_deps.sh --cpu           # force CPU-only
#   ./install_deps.sh --skip-flash-attn
#   ./install_deps.sh --no-smoke-test
#
# Overrides
# ---------
#   TORCH_CUDA_TAG=cu124            # pin a torch wheel tag (else inferred from nvcc)
#   TORCH_SPEC=torch==2.7.1         # pin a torch VERSION (if no flash-attn wheel matches latest)
#   CUDA_HOME=/usr/local/cuda-12.4
#   MAX_JOBS=8                      # parallel compile jobs (auto: min(nproc, 8))
#   BUILDS_TIMEOUT_MIN=30           # cap on each best-effort build
#   FLASH_ATTN_SPEC=flash-attn==2.7.4.post1   # pin a flash-attn version
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

VENV_DIR="${REPO_ROOT}/env"
ENV_VARS_FILE="${REPO_ROOT}/env_vars.sh"
BUILD_LOG="${REPO_ROOT}/install_deps.log"
: > "${BUILD_LOG}"

FORCE_CPU=0
SKIP_FLASH_ATTN=0
SMOKE_TEST=1
BUILDS_TIMEOUT_MIN="${BUILDS_TIMEOUT_MIN:-30}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu) FORCE_CPU=1; shift ;;
        --skip-flash-attn) SKIP_FLASH_ATTN=1; shift ;;
        --no-smoke-test) SMOKE_TEST=0; shift ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown flag: $1" >&2; exit 2 ;;
    esac
done

log()  { printf "\n\033[1;36m[install_deps]\033[0m %s\n" "$*" | tee -a "$BUILD_LOG"; }
warn() { printf "\033[1;33m[install_deps] WARN:\033[0m %s\n" "$*" | tee -a "$BUILD_LOG" >&2; }
fail() { printf "\033[1;31m[install_deps] FATAL:\033[0m %s\n" "$*" | tee -a "$BUILD_LOG" >&2; exit 1; }

# Run an optional build best-effort: time-boxed, streamed to the terminal AND
# the log, never fatal. A failure here just means the runtime falls back.
run_best_effort() {
    local label="$1"; shift
    log "Starting (best-effort): ${label}"
    local rc=0
    if command -v timeout >/dev/null 2>&1; then
        timeout "${BUILDS_TIMEOUT_MIN}m" "$@" > >(tee -a "$BUILD_LOG") 2>&1 || rc=$?
    else
        "$@" > >(tee -a "$BUILD_LOG") 2>&1 || rc=$?
    fi
    if (( rc == 0 )); then
        log "Finished: ${label}"
        return 0
    fi
    warn "${label} failed/timed out (rc=${rc}); see ${BUILD_LOG} — continuing (SDPA fallback)."
    return 1
}

detect_jobs() {
    local n
    n="$( { command -v nproc >/dev/null 2>&1 && nproc; } 2>/dev/null \
        || getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4 )"
    # Cap at 8: flash-attn's nvcc jobs are RAM-heavy and OOM on big core counts.
    (( n > 8 )) && n=8
    echo "$n"
}

# Query flash-attn's GitHub releases for a prebuilt wheel matching THIS env
# (torch minor, CUDA major, Python tag, C++11 ABI, machine). Prints the URL, or
# nothing if none matches. Prebuilt wheels need no nvcc and never fail to build.
find_flash_attn_wheel() {
    python - <<'PY' 2>/dev/null || true
import json, sys, platform, urllib.request
import torch
mm  = ".".join(torch.__version__.split("+")[0].split(".")[:2])      # e.g. 2.7
cu  = "cu" + (torch.version.cuda or "12").split(".")[0]             # e.g. cu12
try:
    abi = "TRUE" if torch._C._GLIBCXX_USE_CXX11_ABI else "FALSE"
except Exception:
    abi = "FALSE"
py  = f"cp{sys.version_info.major}{sys.version_info.minor}"
mach = platform.machine()                                          # x86_64 / aarch64
suffix = f"+{cu}torch{mm}cxx11abi{abi}-{py}-{py}-linux_{mach}.whl"
url = "https://api.github.com/repos/Dao-AILab/flash-attention/releases?per_page=100"
req = urllib.request.Request(url, headers={"User-Agent": "kvboost-installer"})
try:
    rels = json.load(urllib.request.urlopen(req, timeout=30))      # newest first
except Exception:
    sys.exit(0)
for rel in rels:
    for a in rel.get("assets", []):
        if a["name"].endswith(suffix):
            print(a["browser_download_url"]); sys.exit(0)
PY
}

# Install flash-attn for real: prebuilt wheel first, source build as fallback.
# Returns nonzero only if flash_attn still isn't importable afterwards.
install_flash_attn() {
    if python -c 'import flash_attn' 2>/dev/null; then
        log "flash_attn already importable ($(python -c 'import flash_attn; print(flash_attn.__version__)'))"
        return 0
    fi
    python -m pip install -q ninja packaging psutil wheel || true

    local whl
    whl="$(find_flash_attn_wheel)"
    if [[ -n "${whl}" ]]; then
        log "Matching prebuilt wheel: ${whl##*/}"
        if python -m pip install "${whl}" >>"$BUILD_LOG" 2>&1 && python -c 'import flash_attn' 2>/dev/null; then
            log "flash_attn installed from prebuilt wheel"
            return 0
        fi
        warn "prebuilt wheel install failed; see ${BUILD_LOG}"
    else
        warn "no prebuilt flash-attn wheel matches this torch/python/ABI; will try a source build"
    fi

    # Source build fallback (needs a matching nvcc). FORCE_BUILD skips the
    # wheel lookup; arch is already pinned to this GPU so it stays fast.
    if (( CAN_BUILD_EXT == 1 )); then
        run_best_effort "flash-attn source build" \
            env FLASH_ATTENTION_FORCE_BUILD=TRUE \
            python -m pip install -v "${FLASH_ATTN_SPEC:-flash-attn}" --no-build-isolation || true
        python -c 'import flash_attn' 2>/dev/null && return 0
    else
        warn "cannot source-build flash-attn (no matching nvcc on this box)"
    fi
    return 1
}

# --- detect device ----------------------------------------------------------
command -v python3 >/dev/null || fail "python3 not found"
PY_VER="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
log "python: $(python3 --version) (${PY_VER})"

MODE="cpu"
HAVE_NVCC=0
NVCC_MM=""
DRIVER_CUDA=""
GPU_NAME=""

if (( FORCE_CPU == 1 )); then
    log "Forced CPU-only mode (--cpu)."
elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    MODE="cuda"
    GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | sed 's/[[:space:]]*$//')"
    log "GPU detected: ${GPU_NAME:-unknown}"
    # The DRIVER's max CUDA version (nvidia-smi header) is what gates which torch
    # CUDA build can actually initialize. A torch built for a NEWER CUDA than the
    # driver supports fails with "CUDA driver initialization failed" →
    # torch.cuda.is_available()==False (the whole GPU is dead, not just flash-attn).
    # This is the binding constraint, NOT nvcc (the toolkit, used only to compile).
    DRIVER_CUDA="$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA Version: \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
    [[ -n "${DRIVER_CUDA}" ]] && log "driver supports CUDA ${DRIVER_CUDA} → torch build will target ≤ this"
    if command -v nvcc >/dev/null 2>&1; then
        HAVE_NVCC=1
        NVCC_MM="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | head -1)"
        log "nvcc: ${NVCC_MM} (CUDA toolkit present — source builds enabled)"
    else
        warn "nvcc not found: torch will still use its bundled CUDA runtime, but the bundled kernel and flash-attn (which need a compiler) will be skipped."
    fi
else
    warn "No NVIDIA GPU detected — installing CPU-only. Pass --cpu to silence this."
fi

# --- venv -------------------------------------------------------------------
log "Setting up virtualenv at ${VENV_DIR}"
[[ -d "${VENV_DIR}" ]] || python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
log "active python: $(command -v python)"
python -m pip install -U pip setuptools wheel >/dev/null

# --- torch ------------------------------------------------------------------
torch_cuda_candidates() {
    [[ -n "${TORCH_CUDA_TAG:-}" ]] && { echo "${TORCH_CUDA_TAG}"; return; }
    # Choose the torch CUDA tag from the DRIVER's CUDA (what can run), falling
    # back to nvcc only if the driver version is unknown. Tags are listed newest
    # -first but never ABOVE the driver's CUDA — installing a higher one yields
    # torch.cuda.is_available()==False on this driver.
    case "${DRIVER_CUDA:-${NVCC_MM:-}}" in
        13.*)              echo "cu130 cu128 cu126" ;;
        12.8|12.9)         echo "cu128 cu126 cu124" ;;
        12.6|12.7)         echo "cu126 cu124 cu121" ;;
        12.4|12.5)         echo "cu124 cu121 cu118" ;;
        12.1|12.2|12.3)    echo "cu121 cu124 cu118" ;;
        12.0)              echo "cu121 cu118" ;;
        11.*)              echo "cu118" ;;
        *)                 echo "cu128 cu126 cu124 cu121 cu118" ;;  # no nvcc: try newest first
    esac
}

if [[ "${MODE}" == "cpu" ]]; then
    log "Installing CPU-only torch"
    python -m pip install --upgrade --index-url https://download.pytorch.org/whl/cpu "${TORCH_SPEC:-torch}"
else
    # The torch CUDA build is constrained by the driver (the cu-tag candidates
    # above never exceed the driver's CUDA). We install the NEWEST torch on the
    # first compatible index — on an older driver that naturally lands on an
    # older torch (e.g. CUDA 12.4 driver → cu124 → torch ~2.6.x), which is
    # exactly the range flash-attn ships prebuilt wheels for. So matching the
    # driver fixes BOTH the dead-GPU (avail=False) and the no-flash-attn-wheel
    # problems at once. Override the torch version with TORCH_SPEC if needed.
    installed=0
    for tag in $(torch_cuda_candidates); do
        log "Trying CUDA torch wheel: ${tag} (${TORCH_SPEC:-torch})"
        if python -m pip install --upgrade --index-url "https://download.pytorch.org/whl/${tag}" "${TORCH_SPEC:-torch}" \
           && python -c 'import torch,sys; sys.exit(0 if torch.version.cuda else 1)'; then
            log "Installed torch $(python -c 'import torch; print(torch.__version__)') (CUDA $(python -c 'import torch; print(torch.version.cuda)')) via ${tag}"
            installed=1
            break
        fi
        warn "torch wheel ${tag} did not yield a CUDA build; trying next"
    done
    (( installed == 1 )) || fail "could not install a CUDA torch wheel; set TORCH_CUDA_TAG explicitly or use --cpu"
fi

# Can we compile native extensions? Needs nvcc AND a matching torch CUDA major.
CAN_BUILD_EXT=0
ARCH=""
TORCH_CUDA=""
[[ "${MODE}" == "cuda" ]] && TORCH_CUDA="$(python -c 'import torch; print(torch.version.cuda or "")')"
if [[ "${MODE}" == "cuda" && "${HAVE_NVCC}" == "1" ]]; then
    if [[ "${TORCH_CUDA%%.*}" == "${NVCC_MM%%.*}" ]]; then
        CAN_BUILD_EXT=1
        ARCH="$(python - <<'PY' 2>/dev/null || true
import torch
try:
    cc = torch.cuda.get_device_capability(0)
    print(f"{cc[0]}.{cc[1]}")
except Exception:
    pass
PY
)"
        [[ -z "${ARCH}" ]] && ARCH="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;8.9;9.0}"
        log "Native CUDA builds enabled (torch CUDA ${TORCH_CUDA} == nvcc ${NVCC_MM}); target arch ${ARCH}"
    else
        warn "torch CUDA ${TORCH_CUDA} != nvcc ${NVCC_MM} at the major level; skipping native extension builds (FlashInfer + SDPA still work)."
    fi
fi

# --- kvboost package --------------------------------------------------------
# Always install the package itself with the CUDA extension SKIPPED, so a
# kernel compile error can never abort the editable install. The kernel is
# then built as a separate best-effort step below.
EXTRAS="dev,streaming,server"
[[ "${MODE}" == "cuda" ]] && EXTRAS="${EXTRAS},cuda"
log "Installing kvboost (editable) with extras: ${EXTRAS}"
KVBOOST_SKIP_CUDA_EXT=1 python -m pip install -e ".[${EXTRAS}]" --no-build-isolation

# --- accelerators -----------------------------------------------------------
export MAX_JOBS="${MAX_JOBS:-$(detect_jobs)}"
if (( CAN_BUILD_EXT == 1 )); then
    export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
    export PATH="${CUDA_HOME}/bin:${PATH}"
    export TORCH_CUDA_ARCH_LIST="${ARCH}"
    log "Build env: CUDA_HOME=${CUDA_HOME} TORCH_CUDA_ARCH_LIST=${ARCH} MAX_JOBS=${MAX_JOBS}"

    # NOTE: the bundled kvboost._flash_attn_cuda kernel is intentionally NOT
    # built. It's dead on transformers >=5 (the runtime uses torch SDPA, which
    # already dispatches a FlashAttention-2-class kernel on Ampere+), and its
    # head_dim=128 path needs 64 KB of STATIC shared memory — uncompilable on
    # sm_86 (48 KB cap), so it only ever produced a 200-line ptxas error here.
    # flash-attn (installed below) is the real, working prefill backend.
    # (CUDA_HOME / TORCH_CUDA_ARCH_LIST above are still exported for the
    # flash-attn source-build fallback.)
fi

if [[ "${MODE}" == "cuda" ]]; then
    # FlashAttention-2 — REQUIRED (the prefill backend you want). Prebuilt wheel
    # first (no nvcc needed), source build as fallback. Fatal if it can't go in.
    if (( SKIP_FLASH_ATTN == 1 )); then
        warn "Skipping flash-attn at your request (--skip-flash-attn); prefill uses torch SDPA."
    elif install_flash_attn; then
        log "FlashAttention-2 ready ($(python -c 'import flash_attn; print(flash_attn.__version__)'))"
    else
        fail "flash-attn could not be installed (you asked for it explicitly).
  See ${BUILD_LOG} for the exact build error. Most common cause: the torch
  pulled from ${TORCH_CUDA_TAG:-the CUDA index} is newer than any published
  flash-attn wheel. Fixes:
    * pin torch to a release that has wheels:  TORCH_SPEC=torch==2.7.1 ./install_deps.sh
    * or pin a flash-attn version:             FLASH_ATTN_SPEC=flash-attn==2.7.4.post1 ./install_deps.sh"
    fi

    # FlashInfer (decode attention) — JIT, best-effort, works without nvcc.
    log "Installing FlashInfer (best-effort)"
    python -m pip install --upgrade flashinfer-python || warn "flashinfer-python install failed; decode uses SDPA."
fi

# --- env_vars.sh ------------------------------------------------------------
log "Writing ${ENV_VARS_FILE}"
TORCH_LIB="$(python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))')"
{
    echo "# Auto-generated by install_deps.sh — source before running kvboost:"
    echo "#   source ./env_vars.sh"
    echo "export VIRTUAL_ENV=\"${VENV_DIR}\""
    echo "export PATH=\"\${VIRTUAL_ENV}/bin:\${PATH}\""
    echo "export LD_LIBRARY_PATH=\"${TORCH_LIB}:\${LD_LIBRARY_PATH:-}\""
    if (( CAN_BUILD_EXT == 1 )); then
        echo "export CUDA_HOME=\"${CUDA_HOME}\""
        echo "export TORCH_CUDA_ARCH_LIST=\"${ARCH}\""
    fi
} > "${ENV_VARS_FILE}"

# --- smoke test -------------------------------------------------------------
if (( SMOKE_TEST == 1 )); then
    log "Smoke-testing the install"
    python - <<'PYTEST'
import importlib, sys
ok = True

def check(msg, fn):
    global ok
    try:
        fn(); print(f"  OK   {msg}")
    except Exception as exc:
        print(f"  FAIL {msg}: {type(exc).__name__}: {exc}"); ok = False

check("import kvboost", lambda: importlib.import_module("kvboost"))
check("import torch", lambda: importlib.import_module("torch"))
check("KVBoost export", lambda: getattr(importlib.import_module("kvboost"), "KVBoost"))

try:
    import torch
    print(f"  info torch={torch.__version__} cuda={torch.version.cuda} avail={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  info device={torch.cuda.get_device_name(0)} cap={torch.cuda.get_device_capability(0)}")
except Exception:
    pass

def have(mod):
    try: importlib.import_module(mod); return True
    except Exception: return False

fa2  = have("flash_attn")
fi   = have("flashinfer")
kern = have("kvboost._flash_attn_cuda")
print(f"  info prefill backend  : {'flash_attention_2' if fa2 else 'torch SDPA (flash-attn not installed)'}")
print(f"  info decode  backend  : {'flashinfer' if fi else 'torch SDPA (flashinfer not installed)'}")
print(f"  info bundled kernel   : {'kvboost._flash_attn_cuda' if kern else 'not built (SDPA patch path)'}")

sys.exit(0 if ok else 1)
PYTEST
fi

log "Install complete."
log "Activate in a new shell:  source ${ENV_VARS_FILE}"
log "Run unit tests:           pytest tests/speculative/ -v"
