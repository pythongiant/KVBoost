#!/usr/bin/env bash
set -euo pipefail

# Version-aware CUDA installer for KVBoost
# Key rule:
#   nvcc major MUST match torch.version.cuda major for CUDA extension builds.
# Handles CUDA 12.x / 13.x automatically.

log() { echo "[install_deps] $*"; }
fail() { echo "[install_deps] FATAL: $*" >&2; exit 1; }

python3 -m venv env
source env/bin/activate

python -m pip install -U pip setuptools wheel

command -v nvcc >/dev/null || fail "nvcc not found"

CUDA_VER=$(nvcc --version | sed -n 's/.*release \([0-9.]*\).*/\1/p' | head -1)
CUDA_MAJOR=${CUDA_VER%%.*}

log "Detected nvcc CUDA ${CUDA_VER}"

choose_index() {
    case "$1" in
        12) echo "https://download.pytorch.org/whl/cu128" ;;
        13) echo "https://download.pytorch.org/whl/cu130" ;;
        *) fail "Unsupported CUDA major: $1" ;;
    esac
}

TORCH_INDEX=$(choose_index "$CUDA_MAJOR")

log "Installing torch from $TORCH_INDEX"

pip install --upgrade \
  --index-url "$TORCH_INDEX" \
  torch torchvision torchaudio

TORCH_CUDA=$(python - <<'PY'
import torch
print(torch.version.cuda or "")
PY
)

TORCH_MAJOR=${TORCH_CUDA%%.*}

if [ "$TORCH_MAJOR" != "$CUDA_MAJOR" ]; then
    fail "Torch CUDA (${TORCH_CUDA}) does not match nvcc (${CUDA_VER})"
fi

log "Torch CUDA match verified: ${TORCH_CUDA}"

pip install -e ".[dev,streaming,server,cuda]"

ARCH=$(python - <<'PY'
import torch
cc=torch.cuda.get_device_capability(0)
print(f"{cc[0]}.{cc[1]}")
PY
)

export TORCH_CUDA_ARCH_LIST="$ARCH"
export MAX_JOBS="${MAX_JOBS:-4}"

log "GPU arch ${TORCH_CUDA_ARCH_LIST}"

pip install ninja packaging

log "Installing FlashAttention-2"
pip install flash-attn --no-build-isolation

python - <<'PY'
import flash_attn
print("flash_attn OK")
PY

log "Installing FlashInfer (best effort)"
pip install flashinfer-python || true

python - <<'PY'
mods=["torch","kvboost","flash_attn"]
for m in mods:
    __import__(m)
    print("OK",m)
PY

log "Install complete"
