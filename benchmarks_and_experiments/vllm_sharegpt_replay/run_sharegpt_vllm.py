#!/usr/bin/env python3
"""
vLLM Benchmark — ShareGPT Conversation Replay
==============================================
Replays real user-ChatGPT conversations from ShareGPT to measure vLLM's
prefix-cache (automatic KV reuse) on natural, unscripted multi-turn dialogue.

Key design:
  - Each conversation is replayed turn-by-turn, accumulating history exactly
    as a real user would. vLLM's prefix cache builds across turns naturally.
  - Cold: first turn of each conversation (cache miss, full prefill).
  - Warm: subsequent turns (cache hit on shared prefix).
  - Metrics per turn: TTFT, cache hit rate (tokens_cached / prompt_tokens),
    throughput.

The "money chart": TTFT-vs-turn-number. vLLM prefix cache stays flat;
a no-cache baseline grows linearly as history accumulates.

Usage:
  python run_sharegpt_vllm.py
  python run_sharegpt_vllm.py --n-conversations 500 --max-turns 8
  python run_sharegpt_vllm.py --model meta-llama/Llama-3.2-3B
  python run_sharegpt_vllm.py --verbose
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("vllm_sharegpt_replay")

RESULTS_DIR    = Path(__file__).resolve().parent / "results"
CHECKPOINT_DIR = Path(__file__).resolve().parent / ".checkpoints_vllm"
CHECKPOINT_INTERVAL = 10


# ── Data containers ───────────────────────────────────────────────

@dataclass
class TurnResult:
    conv_id: str
    turn_idx: int
    n_turns_total: int
    history_tokens: int
    prompt_tokens: int
    cached_tokens: int          # tokens served from prefix cache
    cache_hit_ratio: float      # cached_tokens / prompt_tokens
    ttft_ms: float
    total_ms: float


@dataclass
class ConvResult:
    conv_id: str
    n_turns: int
    turns: List[TurnResult] = field(default_factory=list)

    @property
    def final_history_tokens(self) -> int:
        return self.turns[-1].history_tokens if self.turns else 0

    @property
    def avg_cache_hit_ratio(self) -> float:
        ratios = [t.cache_hit_ratio for t in self.turns]
        return float(np.mean(ratios)) if ratios else 0.0


# ── Dataset helpers ───────────────────────────────────────────────

def load_sharegpt(
    n_conversations: int,
    min_turns: int,
    max_turns: int,
    max_tokens_per_turn: int,
    tokenizer,
    max_context_tokens: Optional[int] = None,
    seed: int = 42,
) -> List[dict]:
    from datasets import load_dataset

    log.info("Loading anon8231489123/ShareGPT_Vicuna_unfiltered ...")
    ds = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
        split="train",
    )
    log.info(f"  Raw conversations: {len(ds)}")

    rng = np.random.RandomState(seed)
    conversations = []

    for raw in ds:
        turns = raw.get("conversations", [])
        if not turns:
            continue

        human_turns = [t for t in turns if t.get("from") == "human"]
        if len(human_turns) < min_turns:
            continue

        capped = []
        human_count = 0
        for t in turns:
            capped.append(t)
            if t.get("from") == "human":
                human_count += 1
                if human_count >= max_turns:
                    break

        human_msgs = [t["value"] for t in capped if t.get("from") == "human"]
        too_long = any(
            len(tokenizer.encode(msg)) > max_tokens_per_turn
            for msg in human_msgs
        )
        if too_long:
            continue

        conversations.append({
            "id": raw.get("id", f"conv_{len(conversations)}"),
            "turns": capped,
        })

    log.info(
        f"  After filtering (min_turns={min_turns}, max_turns={max_turns}): "
        f"{len(conversations)} conversations"
    )

    if max_context_tokens is not None:
        before = len(conversations)
        conversations = [
            c for c in conversations
            if sum(len(tokenizer.encode(t["value"])) for t in c["turns"])
            <= max_context_tokens
        ]
        log.info(
            f"  max_context_tokens={max_context_tokens}: "
            f"{before} → {len(conversations)} conversations"
        )

    if len(conversations) > n_conversations:
        indices = rng.choice(len(conversations), n_conversations, replace=False)
        conversations = [conversations[i] for i in sorted(indices)]

    log.info(f"  Sampled: {len(conversations)} conversations")

    turn_counts = [
        sum(1 for t in c["turns"] if t.get("from") == "human")
        for c in conversations
    ]
    log.info(
        f"  Turn distribution: min={min(turn_counts)}, "
        f"max={max(turn_counts)}, mean={np.mean(turn_counts):.1f}, "
        f"median={np.median(turn_counts):.1f}"
    )

    return conversations


# ── vLLM runner ───────────────────────────────────────────────────

class VLLMRunner:
    """
    Uses AsyncLLMEngine + token streaming to measure true TTFT:
    wall-clock time from request submission to the first token chunk
    arriving in the async generator, before decode begins.
    """

    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 128,
        gpu_memory_utilization: float = 0.90,
        max_model_len: Optional[int] = None,
        tensor_parallel_size: int = 1,
        enable_prefix_caching: bool = True,
    ):
        from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
        from transformers import AutoTokenizer

        self.SamplingParams = SamplingParams
        self.max_new_tokens = max_new_tokens
        self.enable_prefix_caching = enable_prefix_caching

        log.info(
            f"Loading model: {model_name} "
            f"(prefix_caching={enable_prefix_caching}, "
            f"gpu_mem_util={gpu_memory_utilization}, async=True)"
        )

        engine_args = AsyncEngineArgs(
            model=model_name,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=enable_prefix_caching,
            trust_remote_code=True,
            max_model_len=max_model_len,
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        # Silence vLLM's per-request INFO lines
        logging.getLogger("vllm.engine.async_llm_engine").setLevel(logging.WARNING)
        logging.getLogger("vllm.core.scheduler").setLevel(logging.WARNING)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._request_counter = 0
        # Persistent event loop — reused for every run_turn() call so the
        # AsyncLLMEngine's internal tasks stay alive between requests.
        self._loop = asyncio.new_event_loop()
        log.info("  Async engine ready.")

    @property
    def tokenizer(self):
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=True))

    def run_turn(self, prompt: str) -> dict:
        """Sync wrapper — runs the async coroutine on the persistent event loop."""
        return self._loop.run_until_complete(self._run_turn_async(prompt))

    async def _run_turn_async(self, prompt: str) -> dict:
        """
        Stream tokens from AsyncLLMEngine.
        TTFT = wall time from generate() call to the first non-empty output chunk.
        total_ms = wall time to the final chunk (prefill + full decode).
        """
        from vllm import SamplingParams

        self._request_counter += 1
        request_id = f"req-{self._request_counter}"

        params = SamplingParams(max_tokens=self.max_new_tokens, temperature=0.0)

        t_submit = time.perf_counter()
        ttft_ms: Optional[float] = None
        output_text = ""
        final_output = None

        async for output in self.engine.generate(prompt, params, request_id=request_id):
            if ttft_ms is None and output.outputs and output.outputs[0].text:
                ttft_ms = (time.perf_counter() - t_submit) * 1000
            final_output = output

        total_ms = (time.perf_counter() - t_submit) * 1000

        # If the model returned no text at all, TTFT = total_ms (pathological)
        if ttft_ms is None:
            ttft_ms = total_ms

        if final_output is not None and final_output.outputs:
            output_text = final_output.outputs[0].text

        # Prompt token count
        if final_output is not None and final_output.prompt_token_ids:
            prompt_tokens = len(final_output.prompt_token_ids)
        else:
            prompt_tokens = self.count_tokens(prompt)

        # Cached tokens — available on RequestMetrics in vLLM v0.4+
        cached_tokens = 0
        if final_output is not None and final_output.metrics is not None:
            m = final_output.metrics
            for attr in ("num_cached_tokens", "num_prefix_cache_tokens", "cache_hit_tokens"):
                val = getattr(m, attr, None)
                if val is not None:
                    cached_tokens = int(val)
                    break

        cache_hit_ratio = cached_tokens / max(prompt_tokens, 1)

        return {
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "output_text": output_text,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "cache_hit_ratio": cache_hit_ratio,
        }


# ── Checkpoint helpers ────────────────────────────────────────────

def _checkpoint_key(model_name: str, n_conversations: int, max_turns: int) -> str:
    h = hashlib.md5(
        f"{model_name}_{n_conversations}_{max_turns}".encode()
    ).hexdigest()[:8]
    return h


def save_checkpoint(
    results: List[ConvResult],
    processed_ids: List[str],
    path: Path,
    meta: dict,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "meta": meta,
        "timestamp": time.time(),
        "processed_ids": processed_ids,
        "results": [
            {
                "conv_id": r.conv_id,
                "n_turns": r.n_turns,
                "turns": [
                    {
                        "conv_id": t.conv_id,
                        "turn_idx": t.turn_idx,
                        "n_turns_total": t.n_turns_total,
                        "history_tokens": t.history_tokens,
                        "prompt_tokens": t.prompt_tokens,
                        "cached_tokens": t.cached_tokens,
                        "cache_hit_ratio": t.cache_hit_ratio,
                        "ttft_ms": t.ttft_ms,
                        "total_ms": t.total_ms,
                    }
                    for t in r.turns
                ],
            }
            for r in results
        ],
    }
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(path)
    log.info(f"Checkpoint: {len(processed_ids)} conversations saved → {path.name}")


def load_checkpoint(path: Path) -> tuple[List[ConvResult], List[str]]:
    if not path.exists():
        return [], []
    try:
        with open(path) as f:
            data = json.load(f)
        results = []
        for cr in data.get("results", []):
            turns = [TurnResult(**t) for t in cr["turns"]]
            results.append(ConvResult(
                conv_id=cr["conv_id"],
                n_turns=cr["n_turns"],
                turns=turns,
            ))
        processed_ids = data.get("processed_ids", [])
        log.info(f"Checkpoint loaded: {len(results)} conversations resumed")
        return results, processed_ids
    except Exception as e:
        log.warning(f"Could not load checkpoint: {e} — starting fresh")
        return [], []


# ── Core replay loop ──────────────────────────────────────────────

def replay_conversations(
    runner: VLLMRunner,
    conversations: List[dict],
    no_checkpoint: bool = False,
    model_name: str = "",
    n_conversations: int = 0,
    max_turns: int = 0,
) -> List[ConvResult]:
    ck_key  = _checkpoint_key(model_name, n_conversations, max_turns)
    ck_path = CHECKPOINT_DIR / f"vllm_sharegpt_{ck_key}.json"

    if not no_checkpoint:
        all_results, processed_ids = load_checkpoint(ck_path)
    else:
        all_results, processed_ids = [], []

    processed_set = set(processed_ids)
    meta = {"model": model_name, "n_conversations": n_conversations, "max_turns": max_turns}

    total_start = time.perf_counter()

    for conv_idx, conv in enumerate(conversations):
        conv_id = conv["id"]
        if conv_id in processed_set:
            continue

        turns = conv["turns"]
        human_turns = [t for t in turns if t.get("from") == "human"]
        n_human = len(human_turns)

        conv_result = ConvResult(conv_id=conv_id, n_turns=n_human)

        history = ""
        human_turn_idx = 0

        for turn in turns:
            if turn.get("from") != "human":
                continue

            prompt = history + f"Human: {turn['value']}\nAssistant:"
            history_tokens = runner.count_tokens(prompt)

            result = runner.run_turn(prompt)

            conv_result.turns.append(TurnResult(
                conv_id=conv_id,
                turn_idx=human_turn_idx,
                n_turns_total=n_human,
                history_tokens=history_tokens,
                prompt_tokens=result["prompt_tokens"],
                cached_tokens=result["cached_tokens"],
                cache_hit_ratio=result["cache_hit_ratio"],
                ttft_ms=result["ttft_ms"],
                total_ms=result["total_ms"],
            ))

            history = prompt + result["output_text"] + "\n"
            human_turn_idx += 1

        all_results.append(conv_result)
        processed_ids.append(conv_id)
        processed_set.add(conv_id)

        if (conv_idx + 1) % 10 == 0 or conv_idx == 0:
            elapsed = time.perf_counter() - total_start
            done = len(all_results)
            total = len(conversations)
            avg_hit = np.mean([r.avg_cache_hit_ratio for r in all_results]) if all_results else 0
            log.info(
                f"  [{done}/{total}] conv={conv_id} turns={n_human} "
                f"avg_cache_hit={avg_hit:.2f}  elapsed={elapsed:.0f}s"
            )

        if len(all_results) % CHECKPOINT_INTERVAL == 0:
            try:
                save_checkpoint(all_results, processed_ids, ck_path, meta)
            except Exception as e:
                log.error(f"Checkpoint save failed: {e}")

    if all_results:
        try:
            save_checkpoint(all_results, processed_ids, ck_path, meta)
        except Exception as e:
            log.error(f"Final checkpoint save failed: {e}")

    total_elapsed = time.perf_counter() - total_start
    log.info(f"Replay completed: {len(all_results)} conversations in {total_elapsed:.1f}s")
    return all_results


# ── Metrics & reporting ───────────────────────────────────────────

def compute_metrics(results: List[ConvResult]) -> dict:
    all_turns = [t for r in results for t in r.turns]
    if not all_turns:
        return {}

    ttfts          = [t.ttft_ms for t in all_turns]
    cache_hits     = [t.cache_hit_ratio for t in all_turns]
    cached_tokens  = [t.cached_tokens for t in all_turns]
    prompt_tokens  = [t.prompt_tokens for t in all_turns]

    by_turn: Dict[int, dict] = defaultdict(lambda: {
        "ttfts": [], "cache_hits": [], "cached_tokens": [], "history_tokens": []
    })
    for t in all_turns:
        by_turn[t.turn_idx]["ttfts"].append(t.ttft_ms)
        by_turn[t.turn_idx]["cache_hits"].append(t.cache_hit_ratio)
        by_turn[t.turn_idx]["cached_tokens"].append(t.cached_tokens)
        by_turn[t.turn_idx]["history_tokens"].append(t.history_tokens)

    turn_metrics = {}
    for turn_idx in sorted(by_turn.keys()):
        d = by_turn[turn_idx]
        turn_metrics[turn_idx] = {
            "n": len(d["ttfts"]),
            "avg_ttft_ms":       float(np.mean(d["ttfts"])),
            "p50_ttft_ms":       float(np.percentile(d["ttfts"], 50)),
            "p90_ttft_ms":       float(np.percentile(d["ttfts"], 90)),
            "avg_cache_hit_ratio": float(np.mean(d["cache_hits"])),
            "avg_cached_tokens": float(np.mean(d["cached_tokens"])),
            "avg_history_tokens": float(np.mean(d["history_tokens"])),
        }

    total_time_s  = sum(t.total_ms for t in all_turns) / 1000
    n_requests    = len(all_turns)

    # Turn-0 (cold) vs turn 1+ (warm) TTFT comparison
    cold_ttfts = [t.ttft_ms for t in all_turns if t.turn_idx == 0]
    warm_ttfts = [t.ttft_ms for t in all_turns if t.turn_idx > 0]

    return {
        "n_conversations": len(results),
        "n_turns_total":   n_requests,
        "overall": {
            "ttft_p50":              float(np.percentile(ttfts, 50)),
            "ttft_p90":              float(np.percentile(ttfts, 90)),
            "ttft_p99":              float(np.percentile(ttfts, 99)),
            "cold_ttft_p50":         float(np.percentile(cold_ttfts, 50)) if cold_ttfts else None,
            "warm_ttft_p50":         float(np.percentile(warm_ttfts, 50)) if warm_ttfts else None,
            "warm_vs_cold_speedup":  (
                float(np.mean(cold_ttfts)) / max(float(np.mean(warm_ttfts)), 0.1)
                if cold_ttfts and warm_ttfts else None
            ),
            "avg_cache_hit_ratio":   float(np.mean(cache_hits)),
            "avg_cached_tokens":     float(np.mean(cached_tokens)),
            "avg_prompt_tokens":     float(np.mean(prompt_tokens)),
            "throughput_rps":        round(n_requests / max(total_time_s, 1e-6), 3),
        },
        "by_turn": turn_metrics,
    }


def print_summary(metrics: dict, args: argparse.Namespace):
    print(f"\n{'=' * 75}")
    print(f"  VLLM SHAREGPT REPLAY — Prefix Cache Benchmark")
    print(f"{'=' * 75}")
    print(f"  Conversations:  {metrics['n_conversations']}")
    print(f"  Total turns:    {metrics['n_turns_total']}")

    ov = metrics["overall"]
    print()
    print(f"  {'Metric':<30} {'Value':>15}")
    print(f"  {'-'*30} {'-'*15}")
    print(f"  {'TTFT p50 (ms)':<30} {ov['ttft_p50']:>14.1f}")
    print(f"  {'TTFT p90 (ms)':<30} {ov['ttft_p90']:>14.1f}")
    print(f"  {'TTFT p99 (ms)':<30} {ov['ttft_p99']:>14.1f}")
    if ov["cold_ttft_p50"] is not None:
        print(f"  {'Cold TTFT p50 (ms)':<30} {ov['cold_ttft_p50']:>14.1f}")
        print(f"  {'Warm TTFT p50 (ms)':<30} {ov['warm_ttft_p50']:>14.1f}")
    if ov["warm_vs_cold_speedup"] is not None:
        print(f"  {'Warm/Cold speedup':<30} {ov['warm_vs_cold_speedup']:>13.2f}x")
    print(f"  {'Avg cache hit ratio':<30} {ov['avg_cache_hit_ratio']:>14.1%}")
    print(f"  {'Avg cached tokens':<30} {ov['avg_cached_tokens']:>14.0f}")
    print(f"  {'Avg prompt tokens':<30} {ov['avg_prompt_tokens']:>14.0f}")
    print(f"  {'Throughput (req/s)':<30} {ov['throughput_rps']:>14.3f}")
    print()

    print(f"  TTFT vs Turn Number:")
    print(f"  {'Turn':>5} {'N':>5} {'Avg Ctx Tok':>12} {'TTFT p50':>10} "
          f"{'Cache Hit':>10} {'Cached Tok':>11}")
    print(f"  {'-'*5} {'-'*5} {'-'*12} {'-'*10} {'-'*10} {'-'*11}")
    for turn_idx, tm in sorted(metrics["by_turn"].items()):
        print(
            f"  {int(turn_idx)+1:>5} {tm['n']:>5} "
            f"{tm['avg_history_tokens']:>12.0f} "
            f"{tm['p50_ttft_ms']:>9.0f}ms "
            f"{tm['avg_cache_hit_ratio']:>9.0%}  "
            f"{tm['avg_cached_tokens']:>10.0f}"
        )
    print(f"\n{'=' * 75}\n")


# ── Plotting ──────────────────────────────────────────────────────

def plot_results(metrics: dict, model: str, output_dir: Path):
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        log.warning("matplotlib not installed — skipping plots")
        return

    by_turn     = metrics["by_turn"]
    turn_keys   = sorted(by_turn.keys(), key=int)
    turn_labels = [int(k) + 1 for k in turn_keys]

    ttfts       = [by_turn[k]["avg_ttft_ms"]         for k in turn_keys]
    p50_ttfts   = [by_turn[k]["p50_ttft_ms"]          for k in turn_keys]
    p90_ttfts   = [by_turn[k]["p90_ttft_ms"]          for k in turn_keys]
    hit_ratios  = [by_turn[k]["avg_cache_hit_ratio"]  for k in turn_keys]
    cached_tok  = [by_turn[k]["avg_cached_tokens"]    for k in turn_keys]
    hist_tok    = [by_turn[k]["avg_history_tokens"]   for k in turn_keys]
    ns          = [by_turn[k]["n"]                    for k in turn_keys]

    ov = metrics["overall"]
    n_conv  = metrics["n_conversations"]
    n_turns = metrics["n_turns_total"]

    fig = plt.figure(figsize=(20, 11))
    fig.suptitle(
        f"vLLM Prefix Cache — ShareGPT Replay  |  {model}\n"
        f"{n_conv} conversations · {n_turns} turns",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Row 0 ──────────────────────────────────────────────────────

    # 1. TTFT vs turn (mean + p50/p90 band)
    ax = fig.add_subplot(gs[0, :2])
    ax.plot(turn_labels, ttfts,     "o-", lw=2, color="#3498db", label="Avg TTFT")
    ax.plot(turn_labels, p50_ttfts, "s--", lw=1.2, color="#2ecc71", label="p50 TTFT")
    ax.plot(turn_labels, p90_ttfts, "^--", lw=1.2, color="#e74c3c", label="p90 TTFT")
    ax.fill_between(turn_labels, p50_ttfts, p90_ttfts, alpha=0.10, color="#3498db")
    ax.set_xlabel("Turn Number", fontsize=10)
    ax.set_ylabel("TTFT (ms)", fontsize=10)
    ax.set_title("TTFT vs Turn Number  (vLLM prefix cache, ShareGPT)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(turn_labels)

    # 2. Summary stats table
    ax_s = fig.add_subplot(gs[0, 2])
    ax_s.axis("off")
    rows = [
        ("TTFT p50 (ms)",        f"{ov['ttft_p50']:.1f}"),
        ("TTFT p90 (ms)",        f"{ov['ttft_p90']:.1f}"),
        ("TTFT p99 (ms)",        f"{ov['ttft_p99']:.1f}"),
        ("Cold TTFT p50 (ms)",   f"{ov['cold_ttft_p50']:.1f}" if ov['cold_ttft_p50'] else "—"),
        ("Warm TTFT p50 (ms)",   f"{ov['warm_ttft_p50']:.1f}" if ov['warm_ttft_p50'] else "—"),
        ("Warm/Cold speedup",    f"{ov['warm_vs_cold_speedup']:.2f}×" if ov['warm_vs_cold_speedup'] else "—"),
        ("Avg cache hit ratio",  f"{ov['avg_cache_hit_ratio']:.1%}"),
        ("Avg cached tokens",    f"{ov['avg_cached_tokens']:.0f}"),
        ("Avg prompt tokens",    f"{ov['avg_prompt_tokens']:.0f}"),
        ("Throughput (rps)",     f"{ov['throughput_rps']}"),
    ]
    table = ax_s.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif col == 1:
            cell.set_facecolor("#eaf4fb")
        else:
            cell.set_facecolor("#fdfefe")
    ax_s.set_title("Overall Summary", fontsize=11, pad=8)

    # ── Row 1 ──────────────────────────────────────────────────────

    # 3. Cache hit ratio per turn
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(turn_labels, [r * 100 for r in hit_ratios],
            "D-", lw=2, color="#9b59b6", markersize=7)
    ax.fill_between(turn_labels, [r * 100 for r in hit_ratios],
                    alpha=0.15, color="#9b59b6")
    ax.set_xlabel("Turn Number", fontsize=10)
    ax.set_ylabel("Prefix Cache Hit (%)", fontsize=10)
    ax.set_title("Prefix Cache Hit Rate by Turn", fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(turn_labels)

    # 4. Cached tokens vs history tokens per turn
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(turn_labels, hist_tok,   "o-",  lw=2, color="#e67e22", label="Avg history tokens")
    ax.plot(turn_labels, cached_tok, "s--", lw=2, color="#27ae60", label="Avg cached tokens")
    ax.fill_between(turn_labels, cached_tok, hist_tok,
                    alpha=0.10, color="#e74c3c", label="Recomputed tokens")
    ax.set_xlabel("Turn Number", fontsize=10)
    ax.set_ylabel("Tokens", fontsize=10)
    ax.set_title("History vs Cached Token Growth", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(turn_labels)

    # 5. Sample count per turn
    ax = fig.add_subplot(gs[1, 2])
    ax.bar(turn_labels, ns, color="#95a5a6", alpha=0.75, width=0.6)
    for i, (x, n) in enumerate(zip(turn_labels, ns)):
        ax.text(x, n + max(ns) * 0.01, str(n), ha="center", va="bottom", fontsize=8)
    ax.set_xlabel("Turn Number", fontsize=10)
    ax.set_ylabel("Conversations (n)", fontsize=10)
    ax.set_title("Conversations Reaching Each Turn", fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticks(turn_labels)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "vllm_sharegpt_ttft_vs_turn.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    log.info(f"Plot saved: {out_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="vLLM ShareGPT Replay Benchmark — Prefix Cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--n-conversations", type=int, default=500)
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Alias for --n-conversations (overrides if set)")
    parser.add_argument("--min-turns", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens-per-turn", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-context-tokens", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--no-prefix-caching", action="store_true",
                        help="Disable vLLM prefix caching (ablation baseline)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.n_samples is not None:
        args.n_conversations = args.n_samples

    if args.no_prefix_caching:
        log.warning(
            "⚠  Prefix caching is DISABLED (--no-prefix-caching). "
            "This is an ablation run only — results are not comparable to KVBoost."
        )

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    print(f"\n{'=' * 75}")
    print(f"  VLLM SHAREGPT REPLAY BENCHMARK — Prefix Cache")
    print(f"{'=' * 75}")
    print(f"  Model:         {args.model}")
    print(f"  Conversations: {args.n_conversations}")
    print(f"  Turns:         {args.min_turns}–{args.max_turns}")
    print(f"  Prefix cache:  {'enabled' if not args.no_prefix_caching else 'DISABLED'}")
    print(f"{'=' * 75}\n")

    runner = VLLMRunner(
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=not args.no_prefix_caching,
    )

    conversations = load_sharegpt(
        n_conversations=args.n_conversations,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        max_tokens_per_turn=args.max_tokens_per_turn,
        tokenizer=runner.tokenizer,
        max_context_tokens=args.max_context_tokens,
    )

    if not conversations:
        log.error("No conversations loaded after filtering.")
        sys.exit(1)

    results = replay_conversations(
        runner=runner,
        conversations=conversations,
        no_checkpoint=args.no_checkpoint,
        model_name=args.model,
        n_conversations=args.n_conversations,
        max_turns=args.max_turns,
    )

    if not results:
        log.error("No results produced.")
        sys.exit(1)

    metrics = compute_metrics(results)
    print_summary(metrics, args)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(args.output) if args.output
        else RESULTS_DIR / "vllm_sharegpt_replay.json"
    )
    payload = {
        "benchmark": "vllm_sharegpt_replay",
        "dataset": "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "model": args.model,
        "config": {
            "n_conversations": args.n_conversations,
            "min_turns": args.min_turns,
            "max_turns": args.max_turns,
            "max_new_tokens": args.max_new_tokens,
            "prefix_caching": not args.no_prefix_caching,
            "gpu_memory_utilization": args.gpu_memory_utilization,
        },
        "metrics": metrics,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    log.info(f"Results saved: {out_path}")

    if not args.no_plot:
        plot_results(metrics, args.model, RESULTS_DIR)

    # Clean checkpoint on success
    ck_key  = _checkpoint_key(args.model, args.n_conversations, args.max_turns)
    ck_path = CHECKPOINT_DIR / f"vllm_sharegpt_{ck_key}.json"
    if ck_path.exists():
        ck_path.unlink()
        log.info("Checkpoint cleaned up")

    runner._loop.close()


if __name__ == "__main__":
    main()
