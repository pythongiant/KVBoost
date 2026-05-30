# Production-style integration tests

Real GPU, real model, real HTTP — no mocks. Use these to validate a
deploy under realistic traffic. CI uses `tests/test_server.py` (mocked
engine) for fast feedback.

## Files

| File | Purpose |
|---|---|
| `launch_oom_planner_server.sh` | Launches the server in one of three VRAM profiles |
| `workload.py` | Realistic prompt shapes: long-doc analysis, code review, multi-turn chat, short bursts, oversized |
| `load_oom_planner.py` | **Production load driver** — async, concurrent, reports p50/p95/p99 latency, throughput, planner calibration |
| `test_oom_planner_e2e.py` | Functional smoke (1 request per category, asserts pass/fail). For quick "is it wired" checks after a refactor |

## Production load — the main use case

Heavy-context prompts, configurable concurrency, real latency stats.

### Terminal 1: launch the server

```bash
# Tight VRAM (forces planner to pick small chunks for big prompts)
./tests/integration/launch_oom_planner_server.sh tight 9000

# Or loose (normal traffic profile)
./tests/integration/launch_oom_planner_server.sh loose 9000

# Or with auto-truncate (oversized prompts get silently trimmed)
./tests/integration/launch_oom_planner_server.sh truncate 9000
```

Default model: `Qwen/Qwen2.5-3B-Instruct`. Override via
`KVBOOST_TEST_MODEL=Qwen/Qwen2.5-7B-Instruct ./launch_oom_planner_server.sh ...`.

### Terminal 2: drive the load

```bash
# Realistic production mix (50% short, 20% long-doc, 15% multi-turn,
# 10% code review, 5% research). Concurrency 4, 30 requests.
python tests/integration/load_oom_planner.py \
    --base-url http://localhost:9000 \
    --model Qwen/Qwen2.5-3B-Instruct \
    --workload production \
    --concurrency 4 \
    --n-requests 30 \
    --verbose

# Heavy mix — biased toward long contexts (9-29K input each).
# Real stress test for the planner's chunk_size / kv_bits decisions.
python tests/integration/load_oom_planner.py \
    --workload heavy --n-requests 20 --concurrency 2 --verbose

# Burst short — all 80-token prompts. Measures planner *overhead*
# when the prompt is small (should be invisible).
python tests/integration/load_oom_planner.py \
    --workload burst --n-requests 100 --concurrency 8

# All-oversized — every request is 80K tokens, expect 100% planned 413s
# (or 100% truncation-succeeded if server has --auto-truncate).
python tests/integration/load_oom_planner.py \
    --workload oversized --n-requests 20 --concurrency 4
```

## Output you'll see

```
Workload: production, 30 requests, concurrency=4
Per-shape counts:
  code-review-120: 1
  code-review-60: 5
  long-doc-200: 4
  long-doc-300: 2
  long-doc-500: 1
  multi-turn-4: 5
  multi-turn-8: 1
  short-chat: 11

Planner snapshot (pre-load):
  free_vram_mb_now: 8420
  calibration:
    n_samples:        0
    suggested_margin: 15.00%

Planner snapshot (midpoint, t≈45s):
  free_vram_mb_now: 8200
  calibration:
    n_samples:        14
    residual_median:  -3.2%
    residual_p95:     +6.8%
    suggested_margin: 15.00%

Per-shape latency + throughput:
shape                  n_ok  413p  413u  5xx tout   p50ms   p95ms   p99ms   tok/s
-------------------------------------------------------------------------------
code-review-120           1     0     0    0    0    8420    8420    8420    24.3
code-review-60            5     0     0    0    0    5310    6890    6890    38.6
long-doc-200              4     0     0    0    0    6820    8200    8200    30.1
long-doc-300              2     0     0    0    0    9100    9450    9450    22.5
long-doc-500              1     0     0    0    0   13200   13200   13200    15.5
multi-turn-4              5     0     0    0    0    1140    1820    1820    44.3
multi-turn-8              1     0     0    0    0    1980    1980    1980    25.9
short-chat               11     0     0    0    0     280     410     460    91.2

totals: 30 requests, 30 ok / planned-413, 0 unexpected errors
Overall: 30 requests in 87.3s (0.34 req/s), 0 unexpected errors
```

## What to look for

- **`unexpected errors == 0`** — every request either succeeded or got
  a planned 413 (operator-correct rejection).
- **`residual_p95` vs `suggested_margin`** — if p95 > current margin,
  surprise OOMs will slip through; bump `--planner-safety-margin`. If
  p95 ≪ margin, you're wasting VRAM headroom.
- **per-shape `p95 / p50` ratio** — high ratio means the planner is
  occasionally picking a slow config for that shape (e.g. dropping to
  int4 when int8 would fit). Cross-check against the cohort table.
- **`413p` (planned-413)** — when running `--workload oversized` this
  should equal `n-requests`. If you see `413u` (unplanned), the
  planner thought a request would fit but the HTTP layer disagreed —
  serious bug worth investigating.

## Server log to grep alongside (Terminal 1)

```
Plan committed: chunk=512, kv_bits=8, prompt_tokens=8814, peak=2730/4400 MiB
Request done | mem-post: free=2100MiB ... | predicted=2730MiB actual=2580MiB err=-5.5%
Cleanup | pre: ... | post: ... | freed=950MiB
Request <id> rejected by planner: prompt of 374400 tokens cannot fit...
```

If you see any `OOM slipped past planner for <id>` lines, those are
planner mis-predictions. Cross-reference with the load driver's
`413u` / `5xx` counts and the `(fragmentation, not true OOM)` hint at
the end of the OOM log line to decide whether to raise the safety
margin or fix the memory model.

## Functional smoke (quick, after a refactor)

```bash
# Same Terminal 1 setup, then:
python tests/integration/test_oom_planner_e2e.py \
    --base-url http://localhost:9000 \
    --model Qwen/Qwen2.5-3B-Instruct
```

One request per scenario, asserts pass/fail, exits 0/1. ~30 seconds.

## Why not pytest

Pytest fixtures can't easily own a multi-GB model load + a live server
+ GPU state across many test functions. A bash launcher + a Python
driver is shorter, more inspectable, and matches what you'd actually
do to validate a deploy. If you want it under pytest later, wrap the
load driver in a single `test_load()` that spawns the server as a
subprocess; the workload + assertions stay the same.
