#!/usr/bin/env bash
# Launch the kvboost server in a configuration that lets the OOM-planner
# integration test exercise the 413 / auto-truncate / calibration paths.
#
# Two profiles: TIGHT (--max-cache-bytes 2 GB, default 15% safety margin)
# makes a moderately-large prompt unfittable so 413 is reachable on any
# 12+ GB GPU. LOOSE relaxes the cache budget for normal-traffic profiling.
#
# Usage:
#   ./launch_oom_planner_server.sh [tight|loose] [port]
# Defaults: tight on port 9000.
#
# Stop with Ctrl-C; the server logs to stdout for the integration runner
# to grep against (or pipe to a file: `... 2>&1 | tee server.log`).

set -euo pipefail

PROFILE="${1:-tight}"
PORT="${2:-9000}"
MODEL="${KVBOOST_TEST_MODEL:-Qwen/Qwen2.5-3B-Instruct}"

case "$PROFILE" in
    tight)
        CACHE_BYTES=2e9
        MARGIN=0.15
        ;;
    loose)
        CACHE_BYTES=8e9
        MARGIN=0.10
        ;;
    truncate)
        # Same as tight but with auto-truncate on, so the 413 path
        # becomes a silent prefix-truncation success.
        CACHE_BYTES=2e9
        MARGIN=0.15
        EXTRA="--auto-truncate"
        ;;
    *)
        echo "unknown profile: $PROFILE (use: tight|loose|truncate)" >&2
        exit 2
        ;;
esac
EXTRA="${EXTRA:-}"

echo "Launching kvboost server"
echo "  model:       $MODEL"
echo "  port:        $PORT"
echo "  profile:     $PROFILE"
echo "  cache bytes: $CACHE_BYTES"
echo "  margin:      $MARGIN"
echo "  extra:       $EXTRA"
echo

exec python -m kvboost.server \
    --model "$MODEL" \
    --max-cache-bytes "$CACHE_BYTES" \
    --planner-safety-margin "$MARGIN" \
    --kv-cache-bits 8 \
    --release-cache-after-request \
    --max-batch-size 2 \
    --port "$PORT" \
    $EXTRA
