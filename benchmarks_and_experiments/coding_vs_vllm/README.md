# Coding benchmark: kvboost vs vLLM (real data)

Reports the two kvboost features plus full throughput, on a **real HuggingFace
coding dataset** (no synthetic prompts):

1. **Faster TTFT (KV reuse)** — a coding-agent reuse workload: a shared real
   repo-context prefix + varying real tasks, replayed **sequentially** so the
   prefix KV is reused across requests. TTFT should drop after the first
   request (kvboost chunk-reuse + CacheBlend; vLLM prefix caching).
2. **OOM recovery** — ramp real-code context length; record per backend whether
   each request **completes**, is **gracefully rejected** (4xx), or
   **hard-fails** (OOM 5xx / connection-drop crash / timeout). kvboost adapts
   (chunked prefill, per-request kv-bits, clean 413); vLLM OOMs past its budget.

Plus **throughput on both axes** for every completed request:
- **input/prefill tok/s** = `prompt_tokens / TTFT` — context-ingestion rate.
  This is where reuse pays off: reused chunks aren't re-prefilled, so effective
  input throughput climbs.
- **decode tok/s** = `(out_tokens-1) / (last_tok − first_tok)`.
- **system tok/s** = total output / wall.

Both backends are driven with **identical real prompts** over streaming, with
`stream_options.include_usage` so `prompt_tokens` (→ input throughput) is read
from each backend's own usage report. (kvboost now emits this on its stream.)

## Requirements

```bash
pip install datasets   # real dataset; there is NO synthetic fallback
```

Default dataset `openai_humaneval` (small, no auth, real Python). For
long-context coding agents, point `--dataset` at a repo-level set
(e.g. `repobench`); the adapter pulls code text from common field names.

## One GPU → run each backend separately, then compare

vLLM's `--gpu-memory-utilization` pre-allocates most of the VRAM, so **both
servers can't be resident at once**. So you bench them in **separate runs** —
start one server, benchmark it (save with `--out`), stop it, start the other,
benchmark it — then combine the two saved files into the side-by-side report
with `--compare` (no server needed for that step).

**Step 1 — kvboost.** Start its server (best setup — see `start_kvboost.sh`):
```bash
./start_kvboost.sh            # MODEL=... PORT=... MAX_CACHE_BYTES=... to override
```
That runs kvboost with `cacheblend_sparse` (faithful selective recompute),
int8 KV, and the OOM planner — the features the benchmark measures. Then in
another shell:
```bash
python bench_coding.py --backend kvboost --url http://localhost:9000 \
    --model Qwen/Qwen2.5-3B-Instruct \
    --dataset openai_humaneval --mode both --n 10 \
    --contexts 2000 8000 16000 32000 64000 96000 \
    --out kvboost.json
```
Stop the kvboost server when it finishes (frees the GPU).

**Step 2 — vLLM.** Start its server:
```bash
vllm serve Qwen/Qwen2.5-3B-Instruct --dtype float16 \
    --enable-prefix-caching --gpu-memory-utilization 0.85 \
    --max-model-len 32768 --port 8001
```
Then run the **same** workload flags (so prompts match) against it:
```bash
python bench_coding.py --backend vllm --url http://localhost:8001 \
    --model Qwen/Qwen2.5-3B-Instruct \
    --dataset openai_humaneval --mode both --n 10 \
    --contexts 2000 8000 16000 32000 64000 96000 \
    --out vllm.json
```

**Step 3 — compare** (no GPU/server needed):
```bash
python bench_coding.py --compare kvboost.json vllm.json
```

Each single run also prints its own (single-backend) report immediately, so
you get kvboost's numbers before vLLM is even started. Keep
`--dataset / --n / --n-files / --contexts / --corpus-size` identical across
the two runs — the prompts are deterministic from the dataset, so matching
those flags guarantees both backends saw the exact same inputs.

`--mode ttft` or `--mode oom` to run a single axis.

## Output (illustrative — yours depend on GPU/model)

```
FASTER TTFT + THROUGHPUT — coding-agent reuse (sequential, shared repo context)
backend     ok  ttft1st  ttftP50  ttftLast  inTok/s  decTok/s  sysTok/s
-----------------------------------------------------------------------
kvboost     10     2100      285       275    23916      59.5      48.1
vllm        10      640      300       300    22693      59.5      50.0

TTFT trace per request (ms) — watch reuse warm up after the 1st:
  kvboost     2100    520    360    300    290    285    280 ...
  vllm         640    300    290    640    300    290    640 ...

OOM RECOVERY — real-code context ramp
 ctx~tok            kvboost               vllm
    32000           ✓ ok 9s            ✓ ok 2s
    64000          ✓ ok 21s              ✗ OOM
    96000           ▲ reject              ✗ crash
  kvboost: largest COMPLETED ≈64000 tok; no hard OOM/crash observed
  vllm: largest COMPLETED ≈32000 tok; first hard-fail (OOM/crash) ≈64000 tok
```

## Reading it

- **Faster TTFT**: `ttftLast ≪ ttft1st` means reuse is working. Watch the
  per-request trace — kvboost should fall and stay low; vLLM falls only when
  the leading prefix matches exactly (it can spike back up when the targeted
  file in the task varies the suffix but the prefix is still cached, or when
  passages reorder in the RAG variant — see `bench_hf.py`).
- **Input throughput**: higher `inTok/s` = faster context ingestion; reuse
  inflates it because reused prefix tokens cost ~nothing to "ingest."
- **Decode throughput**: vLLM usually leads (continuous batching). Honest —
  the kvboost story is TTFT + input throughput on reused context, not decode.
- **OOM recovery**: the gap between *largest COMPLETED* and *first hard-fail*.
  `▲ reject` (clean 4xx) is **success** (server said no without dying);
  `✗ OOM/crash/t.o.` are the real failures.

## Sibling: RAG reuse (CacheBlend's specific edge) — `bench_hf.py`

`bench_coding.py` uses the realistic coding-agent **prefix** reuse pattern
(both backends can prefix-cache it). To isolate CacheBlend's advantage over
prefix caching — reuse of chunks that recur **out of prefix order** — use
`bench_hf.py`, which builds RAG prompts from a real dataset (`squad`) with
recurring, reordered passages:

```bash
python benchmarks_and_experiments/coding_vs_vllm/bench_hf.py \
    --kvboost-url http://localhost:9000 --vllm-url http://localhost:8001 \
    --dataset squad --n 10
```

There, prefix caching mostly misses (passages move position) while CacheBlend
reuses each chunk anywhere — the TTFT gap should be widest.

## Files

- `coding_workload.py` — real-dataset loader + reuse/OOM prompt builders.
- `bench_coding.py` — streaming driver, TTFT + OOM + input/decode/system throughput.
- `hf_workload.py` — real-dataset RAG prompts with recurring, reordered passages.
- `bench_hf.py` — sequential RAG reuse benchmark (CacheBlend vs prefix caching).
