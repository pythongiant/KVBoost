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

The driver **streams tokens** (`stream: true`) so it measures real TTFT
and decode rate, and shows a **live ANSI dashboard** where each request
moves through WAITING → PREFILL → STREAM → DONE/ERR/413 with live token
counts. When stdout isn't a TTY (piped to a file / CI) it falls back to
one plain log line per completion — no flag needed.

```bash
# Realistic production mix (50% short, 20% long-doc, 15% multi-turn,
# 10% code review, 5% research). Concurrency 2, 30 requests.
python tests/integration/load_oom_planner.py \
    --base-url http://localhost:9000 \
    --model Qwen/Qwen2.5-3B-Instruct \
    --workload production \
    --concurrency 2 \
    --n-requests 30

# Heavy mix — biased toward long contexts (9-29K input each).
# Real stress test for the planner's chunk_size / kv_bits decisions.
python tests/integration/load_oom_planner.py --workload heavy --n-requests 20

# Burst short — all 80-token prompts. Measures TTFT + planner overhead
# on small prompts (should be near-instant). Higher concurrency OK here.
python tests/integration/load_oom_planner.py \
    --workload burst --n-requests 100 --concurrency 8

# All-oversized — every request is 80K tokens, expect 100% planned 413s
# (or 100% truncation-succeeded if server has --auto-truncate).
python tests/integration/load_oom_planner.py --workload oversized --n-requests 20
```

## Live dashboard (TTY)

While running, each request is a row updating in place:

```
kvboost load — production
  0 short-chat       DONE    in≈    80 out=  64 ttft=  210ms dec= 88.4t/s   2.6s ██████████
  1 long-doc-200     STREAM  in≈  9360 out= 412 ttft= 1840ms dec= 31.2t/s  14.9s ████······
  2 code-review-120  PREFILL in≈ 17628 out=   0 ttft=    ·ms dec=    ·t/s   8.1s
  3 long-doc-500     413     in≈ 23400 out=   0 ttft=    ·ms dec=    ·t/s   0.4s rejected
  4 short-chat       WAITING in≈    80 out=   0 ttft=    ·ms dec=    ·t/s   0.0s
done 1  active 2  rejected(413) 1  errors 0 | out_tokens 476  sys 27.3 tok/s  elapsed 17.4s
```

State legend: **WAITING** queued · **PREFILL** sent, awaiting first token ·
**STREAM** receiving tokens (bar = out/max_tokens) · **DONE** · **413**
planner rejection (counted as success) · **ERR** unexpected failure.

## Final summary (printed after the dashboard)

```
Per-shape TTFT + decode throughput:
shape              ok  413  err  ttftP50  ttftP95  decP50  sysTok/s
------------------------------------------------------------------
code-review-120     1    0    0     8120     8120    24.3      22.1
long-doc-200        4    0    0     1780     2310    31.0      29.5
long-doc-500        0    1    0        0        0     0.0       0.0
short-chat         11    0    0      205      290    89.1      84.7

Overall: 30 requests, 8740 output tokens in 92.4s (94.6 sys tok/s), 0 unexpected errors
```

## What to look for

- **`0 unexpected errors`** — every request either streamed to completion
  or got a planned 413 (operator-correct rejection). A 413 is **not** an
  error; it's the planner refusing an unfittable prompt up front.
- **TTFT p50/p95** — first-token latency. For long-doc shapes this is
  dominated by prefill; a p95 ≫ p50 means some requests queued behind
  others (raise concurrency cautiously, or it's just a slow GPU).
- **decode tok/s (decP50)** — steady-state generation rate, independent
  of prefill. If this is low (<10 on a 3B), the GPU is the bottleneck,
  not the planner.
- **sys tok/s** — system throughput (all output tokens / wall). The
  number that matters for capacity planning.
- **`residual_p95` vs `suggested_margin`** (planner snapshot) — if p95 >
  margin, surprise OOMs will slip through; bump `--planner-safety-margin`.
  If p95 ≪ margin, you're over-reserving VRAM.
- **err > 0 on long shapes** — if a long-doc / code-review shape shows
  `err` with a CUDA-OOM message, the planner mis-predicted (its
  chunk-based memory model was violated). Cross-check the server log for
  `OOM slipped past planner`.

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
