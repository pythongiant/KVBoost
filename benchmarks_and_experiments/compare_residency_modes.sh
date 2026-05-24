#!/usr/bin/env bash
# Wall-clock A/B between StreamingConfig residency modes — without ncu.
#
# Runs demo_speculative once per mode on a fixed prompt and tabulates the
# decode_only_tok_per_s and avg_tok_per_s lines it prints. Use this to answer
# "does the partial_resident hook overhead actually cost me anything on this
# hardware when every layer is already resident?" without paying the
# kernel-replay cost of ncu.
#
# Usage
# -----
#   ./compare_residency_modes.sh                          # all four modes
#   MODES="partial_resident full_resident" ./compare_residency_modes.sh
#   MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ \
#     KEEP_FIRST_K=1800 KEEP_LAST_K=1800 \
#     ./compare_residency_modes.sh
#
# Outputs each run's full log to results/residency_compare/<mode>.log and a
# summary table to results/residency_compare/summary.csv.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/results/residency_compare}"

# Defaults tuned for a 12 GB GPU running Qwen2.5-7B-AWQ.
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
DRAFT_MODEL="${DRAFT_MODEL:-Qwen/Qwen2.5-1.5B-Instruct-AWQ}"
GAMMA="${GAMMA:-5}"
SPEC_MODE="${SPEC_MODE:-greedy}"
KEEP_FIRST_K="${KEEP_FIRST_K:-1800}"
KEEP_LAST_K="${KEEP_LAST_K:-1800}"
N_STAGING_SLOTS="${N_STAGING_SLOTS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
PROMPT="${PROMPT:-Explain the second law of thermodynamics in two short paragraphs.}"

MODES="${MODES:-partial_resident full_resident ffn_only_stream full_stream}"

mkdir -p "${OUT_DIR}"
SUMMARY="${OUT_DIR}/summary.csv"
echo "mode,load_s,prefill_s,tps_avg,tps_decode_only,avg_committed_per_round,peak_vram_mb" > "${SUMMARY}"

echo "[compare] model=${MODEL}"
echo "[compare] modes: ${MODES}"
echo

for mode in ${MODES}; do
  LOG="${OUT_DIR}/${mode}.log"
  echo "════════════════════════════════════════════════════════════════════"
  echo "  mode = ${mode}    log -> ${LOG}"
  echo "════════════════════════════════════════════════════════════════════"

  set +e
  python -m kvboost.streaming.demo_speculative \
    --model "${MODEL}" \
    --draft-model "${DRAFT_MODEL}" \
    --mode "${mode}" \
    --keep-first-k "${KEEP_FIRST_K}" \
    --keep-last-k "${KEEP_LAST_K}" \
    --n-staging-slots "${N_STAGING_SLOTS}" \
    --gamma "${GAMMA}" \
    --spec-mode "${SPEC_MODE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --prompt "${PROMPT}" \
    2>&1 | tee "${LOG}"
  rc=${PIPESTATUS[0]}
  set -e

  if [[ ${rc} -ne 0 ]]; then
    echo "[compare] mode=${mode} FAILED (rc=${rc}); appending blank row."
    echo "${mode},,,,,," >> "${SUMMARY}"
    continue
  fi

  # Pull the interesting numbers out of the log.
  load_s=$(grep -E "load_time:" "${LOG}" | grep -oE "[0-9]+\.?[0-9]*" | head -1 || echo "")
  prefill_s=$(grep -E "prefill_time:" "${LOG}" | grep -oE "[0-9]+\.?[0-9]*" | head -1 || echo "")
  tps_avg=$(grep -E "avg_tok_per_s:" "${LOG}" | grep -oE "[0-9]+\.?[0-9]*" | head -1 || echo "")
  tps_decode=$(grep -E "decode_only_tok_per_s:" "${LOG}" | grep -oE "[0-9]+\.?[0-9]*" | head -1 || echo "")
  avg_commit=$(grep -E "avg_committed/round:" "${LOG}" | grep -oE "[0-9]+\.?[0-9]*" | head -1 || echo "")
  peak_vram=$(grep -E "peak_vram_during_decode:" "${LOG}" | grep -oE "[0-9]+\.?[0-9]*" | head -1 || echo "")

  printf '%s,%s,%s,%s,%s,%s,%s\n' \
    "${mode}" "${load_s}" "${prefill_s}" "${tps_avg}" "${tps_decode}" "${avg_commit}" "${peak_vram}" \
    >> "${SUMMARY}"

  echo
done

echo
echo "════════════════════════════════════════════════════════════════════"
echo "  Summary"
echo "════════════════════════════════════════════════════════════════════"
column -t -s ',' "${SUMMARY}" 2>/dev/null || cat "${SUMMARY}"
echo
echo "[compare] CSV: ${SUMMARY}"
echo "[compare] per-mode logs in: ${OUT_DIR}/"
