#!/usr/bin/env bash
# Roofline profiling of the AWQ streaming + speculative decode path with Nsight Compute.
#
# What this does
# --------------
# Runs `ncu --set full` (includes the SpeedOfLight roofline section) against a
# short generation of demo_speculative. Skips the prefill kernels so the
# captured window is dominated by decode-phase work, where the streaming
# bandwidth ceiling is the interesting roof.
#
# Outputs
# -------
#   ${OUT_DIR}/${TAG}.ncu-rep                    -> open in `ncu-ui`
#   ${OUT_DIR}/${TAG}.csv                        -> per-kernel metrics flat CSV
#   ${OUT_DIR}/${TAG}.roofline.csv               -> filtered to matmul/attn kernels
#
# Requirements
# ------------
#   * Nsight Compute >= 2023.1 (`ncu --version`).
#   * On most Linux hosts you need either:
#       - `sudo` (recommended for one-shot runs), OR
#       - the NVIDIA "allow access to performance counters" sysctl set:
#         `sudo sh -c 'echo options nvidia NVreg_RestrictProfilingToAdminUsers=0 \
#           > /etc/modprobe.d/nvidia-profile.conf' && sudo reboot`
#     Without one of those ncu will fail with `ERR_NVGPUCTRPERM`.
#   * Profiling replays every captured kernel multiple times -- keep
#     MAX_NEW_TOKENS small (4-8 is plenty for a roofline plot).
#
# Usage
# -----
#   ./roofline_ncu.sh                              # defaults below
#   TAG=spec_g7 GAMMA=7 ./roofline_ncu.sh
#   MODE=full_resident TAG=resident ./roofline_ncu.sh
#   LAUNCH_SKIP=500 LAUNCH_COUNT=200 ./roofline_ncu.sh
#
# To compare specs, run this twice (e.g. GAMMA=0 vs GAMMA=5) and diff the
# resulting CSVs -- arithmetic intensity should march to the right as gamma
# grows because the same weight DMA is amortized over more token FLOPs.

set -euo pipefail

# ---------- knobs ------------------------------------------------------------
TAG="${TAG:-spec_default}"
OUT_DIR="${OUT_DIR:-$(cd "$(dirname "$0")" && pwd)/results/roofline}"

# Defaults tuned for a 12 GB GPU (RTX 3060 / similar). Override MODEL+keep_k
# from the env for bigger cards. KEEP_FIRST/LAST_K=1800 forces every Qwen2.5-7B
# layer (28 total) resident; in `partial_resident` mode the streaming hooks
# still fire (overhead!), in `full_resident` they're bypassed entirely.
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct-AWQ}"
MODE="${MODE:-partial_resident}"
KEEP_FIRST_K="${KEEP_FIRST_K:-1800}"
KEEP_LAST_K="${KEEP_LAST_K:-1800}"
N_STAGING_SLOTS="${N_STAGING_SLOTS:-4}"
GAMMA="${GAMMA:-5}"
SPEC_MODE="${SPEC_MODE:-greedy}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-6}"
PROMPT="${PROMPT:-Explain entropy in two sentences.}"

# Kernel-window controls. Prefill on a 32B model launches a few hundred
# kernels; skip past them so the captured window is pure decode.
LAUNCH_SKIP="${LAUNCH_SKIP:-400}"
LAUNCH_COUNT="${LAUNCH_COUNT:-150}"

# Restrict capture to kernels that actually move FLOPs around. Without this
# you'll spend hours replaying tokenizer/sampler/elementwise kernels that
# don't matter for the roofline.
KERNEL_REGEX="${KERNEL_REGEX:-regex:gemm|matmul|attention|wmma|mma|flash|attn|awq|dequant}"

# ncu section set. Valid choices vary by version:
#   ncu 2023.x: basic | default | detailed | full | source | roofline
#   ncu 2024.x: basic | default | detailed | full | source   (no `roofline`)
# `full` always works and includes the SpeedOfLight + roofline chart sections.
NCU_SET="${NCU_SET:-full}"

# ----------------------------------------------------------------------------

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu not on PATH. Install Nsight Compute or source the CUDA env." >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
REP="${OUT_DIR}/${TAG}.ncu-rep"
CSV="${OUT_DIR}/${TAG}.csv"
ROOFLINE_CSV="${OUT_DIR}/${TAG}.roofline.csv"

echo "[roofline] tag=${TAG} mode=${MODE} gamma=${GAMMA} tokens=${MAX_NEW_TOKENS}"
echo "[roofline] writing -> ${REP}"

# Profile run ----------------------------------------------------------------
# --target-processes all      : Hugging Face spawns workers; catch them too.
# --replay-mode kernel        : default; replays each kernel for all counters.
# --cache-control all         : flush caches between replays for stable counters.
# --clock-control base        : lock clocks to base for reproducible FLOPs/s.
# --import-source on          : embed Python source refs (helpful in ncu-ui).
# --launch-skip / -count       : -count was renamed --launch-count-per-target
#                                in newer ncu; the unambiguous spellings below
#                                work on both.
#
# NOTE: do NOT include a bare `--` separator before the python command.
# Older ncu treated `--` as end-of-options; ncu >= 2024.x parses it as an
# empty long-option name and errors with "option is ambiguous". Just put
# the application command directly after the ncu flags.
ncu \
  --set "${NCU_SET}" \
  --target-processes all \
  --replay-mode kernel \
  --cache-control all \
  --clock-control base \
  --import-source on \
  --kernel-name "${KERNEL_REGEX}" \
  --launch-skip "${LAUNCH_SKIP}" \
  --launch-count "${LAUNCH_COUNT}" \
  --export "${REP}" \
  --force-overwrite \
  python -m kvboost.streaming.demo_speculative \
    --model "${MODEL}" \
    --draft-model "${DRAFT_MODEL}" \
    --mode "${MODE}" \
    --keep-first-k "${KEEP_FIRST_K}" \
    --keep-last-k "${KEEP_LAST_K}" \
    --n-staging-slots "${N_STAGING_SLOTS}" \
    --gamma "${GAMMA}" \
    --spec-mode "${SPEC_MODE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --prompt "${PROMPT}"

# Export ----------------------------------------------------------------------
echo "[roofline] exporting CSV -> ${CSV}"
ncu --import "${REP}" --csv --page details > "${CSV}"

# Pull just the columns useful for a roofline plot. Metric names can shift
# between ncu versions, so we grep loosely and keep the kernel id column.
echo "[roofline] filtered roofline CSV -> ${ROOFLINE_CSV}"
ncu --import "${REP}" --csv --page details \
    --metrics \
      sm__sass_thread_inst_executed_op_fadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_ffma_pred_on.sum,\
sm__sass_thread_inst_executed_op_fmul_pred_on.sum,\
sm__sass_thread_inst_executed_op_hadd_pred_on.sum,\
sm__sass_thread_inst_executed_op_hfma_pred_on.sum,\
sm__sass_thread_inst_executed_op_hmul_pred_on.sum,\
dram__bytes.sum,\
lts__t_bytes.sum,\
l1tex__t_bytes.sum,\
gpu__time_duration.sum,\
sm__cycles_elapsed.avg.per_second \
    > "${ROOFLINE_CSV}"

echo "[roofline] done."
echo "  open report: ncu-ui ${REP}"
echo "  CSV:         ${CSV}"
echo "  roofline:    ${ROOFLINE_CSV}"
