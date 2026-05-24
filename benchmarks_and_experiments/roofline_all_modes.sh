#!/usr/bin/env bash
# Run roofline_ncu.sh against every streaming residency mode and surface a
# small summary so you can see how much the partial_resident scheduler
# overhead costs when all layers are *already* resident.
#
# Outputs land in:
#   results/roofline/${TAG}_${MODE}.ncu-rep   (open in ncu-ui)
#   results/roofline/${TAG}_${MODE}.csv       (per-kernel metrics)
#   results/roofline/${TAG}_${MODE}.roofline.csv
#
# Usage
# -----
#   ./roofline_all_modes.sh                       # all four modes, default knobs
#   MODES="partial_resident full_resident" ./roofline_all_modes.sh
#   TAG=qwen7b_12gb ./roofline_all_modes.sh
#
# Then compare the four .ncu-rep files in ncu-ui, or look at the printed
# total-kernel-time table at the end of this script.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TAG="${TAG:-qwen7b_12gb}"
MODES="${MODES:-partial_resident full_resident ffn_only_stream full_stream}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/results/roofline}"

mkdir -p "${OUT_DIR}"

echo "[all-modes] tag=${TAG}"
echo "[all-modes] modes: ${MODES}"
echo "[all-modes] writing to ${OUT_DIR}"
echo

# ── 1. Profile each mode ────────────────────────────────────────────────────
for mode in ${MODES}; do
  echo "════════════════════════════════════════════════════════════════════"
  echo "  Profiling mode = ${mode}"
  echo "════════════════════════════════════════════════════════════════════"

  MODE="${mode}" TAG="${TAG}_${mode}" OUT_DIR="${OUT_DIR}" \
    "${SCRIPT_DIR}/roofline_ncu.sh" || {
      echo "[all-modes] FAILED on mode=${mode}; continuing." >&2
      continue
    }
done

# ── 2. Summarize total kernel time per mode ────────────────────────────────
# A quick decode-time A/B: sum gpu__time_duration.sum across all captured
# kernels for each mode. Smaller = faster decode for the same workload.
echo
echo "════════════════════════════════════════════════════════════════════"
echo "  Summary: total captured kernel time per mode"
echo "════════════════════════════════════════════════════════════════════"
printf '  %-22s %18s %18s\n' "mode" "kernels" "total_ms"
printf '  %-22s %18s %18s\n' "----" "-------" "--------"

for mode in ${MODES}; do
  csv="${OUT_DIR}/${TAG}_${mode}.roofline.csv"
  if [[ ! -f "${csv}" ]]; then
    printf '  %-22s %18s %18s\n' "${mode}" "(no csv)" "—"
    continue
  fi

  # ncu CSV: column order varies by version; look up the duration column by
  # name in the header instead of hard-coding an index.
  python3 - "${csv}" "${mode}" <<'PY'
import csv, sys
path, mode = sys.argv[1], sys.argv[2]

dur_col = None
n_kernels = 0
total_ns = 0.0
with open(path, newline="") as f:
    rdr = csv.reader(f)
    header = next(rdr, None) or []
    # match "gpu__time_duration.sum" or the unit-suffixed variant
    for i, h in enumerate(header):
        if "gpu__time_duration" in h:
            dur_col = i
            break
    if dur_col is None:
        print(f"  {mode:<22} {'(no dur col)':>18} {'—':>18}")
        raise SystemExit(0)

    for row in rdr:
        if len(row) <= dur_col:
            continue
        cell = row[dur_col].replace(",", "").strip()
        if not cell:
            continue
        try:
            # ncu CSV often suffixes the unit (e.g. "12.34 us"); split first token
            val_str = cell.split()[0]
            unit = cell.split()[1] if len(cell.split()) > 1 else "ns"
            v = float(val_str)
            if   unit.lower().startswith("us"): v *= 1e3       # → ns
            elif unit.lower().startswith("ms"): v *= 1e6       # → ns
            elif unit.lower().startswith("s"):  v *= 1e9       # → ns
            total_ns += v
            n_kernels += 1
        except (ValueError, IndexError):
            continue

print(f"  {mode:<22} {n_kernels:>18d} {total_ns/1e6:>15.2f} ms")
PY
done

echo
echo "[all-modes] done. Open any of the .ncu-rep files in ncu-ui for full detail."
