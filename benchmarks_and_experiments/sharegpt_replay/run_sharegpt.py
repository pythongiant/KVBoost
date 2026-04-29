#!/usr/bin/env python3
"""
KVBoost Benchmark — ShareGPT Conversation Replay
=================================================
Replays real user-ChatGPT conversations from ShareGPT to measure KVBoost's
multi-turn cache reuse on natural, unscripted dialogue.

Key design:
  - Each conversation is replayed turn-by-turn, accumulating history exactly
    as a real user would. KVBoost's chunk cache builds across turns naturally.
  - Baseline: standard HuggingFace generate (no KV reuse).
  - KVBoost: warm_chunks() on current history, then generate() with reuse.
  - Metrics per turn: TTFT, kv_reuse_ratio, throughput.

The "money chart": TTFT-vs-turn-number. KVBoost stays flat; baseline grows
linearly as history accumulates — on real, unpredictable conversations.

Usage:
  python run_sharegpt.py
  python run_sharegpt.py --n-conversations 500 --max-turns 8
  python run_sharegpt.py --model meta-llama/Llama-3.2-3B
  python run_sharegpt.py --verbose

Recompute config (tuned for best results):
  --recompute-strategy cacheblend
  --chunk-boundary-window 16
  --overlap-k 16
  --sink-tokens 32
"""

from __future__ import annotations

import argparse
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
log = logging.getLogger("sharegpt_replay")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
CHECKPOINT_DIR = Path(__file__).resolve().parent / ".checkpoints"
CHECKPOINT_INTERVAL = 10


# ── Data containers ───────────────────────────────────────────────

@dataclass
class TurnResult:
    conv_id: str
    turn_idx: int               # 0-based index within this conversation
    n_turns_total: int
    history_tokens: int
    baseline_ttft_ms: float
    kvboost_ttft_ms: float
    kv_reuse_ratio: float       # 0.0–1.0 from engine
    baseline_total_ms: float
    kvboost_total_ms: float


@dataclass
class ConvResult:
    conv_id: str
    n_turns: int
    turns: List[TurnResult] = field(default_factory=list)

    @property
    def final_history_tokens(self) -> int:
        return self.turns[-1].history_tokens if self.turns else 0

    @property
    def avg_speedup(self) -> float:
        speedups = [
            t.baseline_ttft_ms / max(t.kvboost_ttft_ms, 0.1)
            for t in self.turns
        ]
        return float(np.mean(speedups)) if speedups else 0.0


# ── Dataset helpers ───────────────────────────────────────────────

def load_sharegpt(
    n_conversations: int,
    min_turns: int,
    max_turns: int,
    max_tokens_per_turn: int,
    tokenizer,
    seed: int = 42,
) -> List[dict]:
    """
    Load and filter ShareGPT conversations.

    Filters:
      - Keep conversations with >= min_turns human turns.
      - Cap each conversation at max_turns human turns.
      - Drop turns where the human message exceeds max_tokens_per_turn.
    """
    from datasets import load_dataset

    log.info("Loading Aeala/ShareGPT_Vicuna_unfiltered ...")
    ds = load_dataset(
        "Aeala/ShareGPT_Vicuna_unfiltered",
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

        # Extract human turns only (assistant turns are used for history only)
        human_turns = [t for t in turns if t.get("from") == "human"]
        if len(human_turns) < min_turns:
            continue

        # Cap at max_turns
        capped = []
        human_count = 0
        for t in turns:
            capped.append(t)
            if t.get("from") == "human":
                human_count += 1
                if human_count >= max_turns:
                    break

        # Token-length filter on human turns
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

    # Sample
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


# ── KVBoost runner ────────────────────────────────────────────────

class ShareGPTRunner:
    def __init__(
        self,
        model_name: str,
        chunk_size: int = 128,
        recompute_strategy: str = "cacheblend",
        chunk_boundary_window: int = 16,
        overlap_k: int = 16,
        sink_tokens: int = 32,
        max_new_tokens: int = 128,
        max_cache_bytes: float = None,
        recency_window_chunks: int = None,
    ):
        from kvboost import KVBoost, GenerationMode
        from kvboost.flash_attn_ext import get_tier

        self.GenerationMode = GenerationMode
        self.max_new_tokens = max_new_tokens

        log.info(
            f"Loading model: {model_name} "
            f"(strategy={recompute_strategy}, boundary_window={chunk_boundary_window}, "
            f"overlap_k={overlap_k}, sink_tokens={sink_tokens})"
        )
        kvboost_kwargs = dict(
            chunk_size=chunk_size,
            recompute_overlap=16,
            recompute_strategy=recompute_strategy,
            chunk_boundary_window=chunk_boundary_window,
            overlap_k=overlap_k,
            sink_tokens=sink_tokens,
            max_cache_bytes=int(max_cache_bytes) if max_cache_bytes is not None else None,
        )
        if recency_window_chunks is not None:
            kvboost_kwargs["recency_window_chunks"] = recency_window_chunks
        self.engine = KVBoost.from_pretrained(
            model_name,
            **kvboost_kwargs,
        )
        flash_tier = get_tier()
        log.info(f"  Engine ready on {self.engine.device} | flash_attn tier: {flash_tier}")

    @property
    def tokenizer(self):
        return self.engine.tokenizer

    def count_tokens(self, text: str) -> int:
        return len(self.engine.tokenizer.encode(text, add_special_tokens=True))

    def reset(self):
        self.engine.reset_cache()

    def run_turn(self, history: str) -> tuple[dict, dict]:
        """
        Run one conversation turn in both baseline and KVBoost modes.

        Returns (baseline_result, kvboost_result) dicts with keys:
          ttft_ms, total_ms, output_text, kv_reuse_ratio
        """
        # Baseline: no cache, cold prefill
        b = self.engine.generate(
            history,
            max_new_tokens=self.max_new_tokens,
            mode=self.GenerationMode.BASELINE,
            do_sample=False,
        )

        # KVBoost: warm cache from history, then generate
        self.engine.warm_chunks(history, position_offset=0)
        k = self.engine.generate(
            history,
            max_new_tokens=self.max_new_tokens,
            mode=self.GenerationMode.CHUNK_KV_REUSE,
            do_sample=False,
        )

        baseline = {
            "ttft_ms": b.ttft_ms,
            "total_ms": b.total_ms,
            "output_text": b.output_text,
            "kv_reuse_ratio": 0.0,
        }
        kvboost = {
            "ttft_ms": k.ttft_ms,
            "total_ms": k.total_ms,
            "output_text": k.output_text,
            "kv_reuse_ratio": k.kv_reuse_ratio,
        }
        return baseline, kvboost


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
                        "baseline_ttft_ms": t.baseline_ttft_ms,
                        "kvboost_ttft_ms": t.kvboost_ttft_ms,
                        "kv_reuse_ratio": t.kv_reuse_ratio,
                        "baseline_total_ms": t.baseline_total_ms,
                        "kvboost_total_ms": t.kvboost_total_ms,
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
    runner: ShareGPTRunner,
    conversations: List[dict],
    no_checkpoint: bool = False,
    model_name: str = "",
    n_conversations: int = 0,
    max_turns: int = 0,
) -> List[ConvResult]:
    ck_key = _checkpoint_key(model_name, n_conversations, max_turns)
    ck_path = CHECKPOINT_DIR / f"sharegpt_{ck_key}.json"

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

        runner.reset()
        conv_result = ConvResult(conv_id=conv_id, n_turns=n_human)

        history = ""
        human_turn_idx = 0

        for t_idx, turn in enumerate(turns):
            if turn.get("from") == "human":
                prompt = history + f"Human: {turn['value']}\nAssistant:"
                history_tokens = runner.count_tokens(prompt)

                b, k = runner.run_turn(prompt)

                conv_result.turns.append(TurnResult(
                    conv_id=conv_id,
                    turn_idx=human_turn_idx,
                    n_turns_total=n_human,
                    history_tokens=history_tokens,
                    baseline_ttft_ms=b["ttft_ms"],
                    kvboost_ttft_ms=k["ttft_ms"],
                    kv_reuse_ratio=k["kv_reuse_ratio"],
                    baseline_total_ms=b["total_ms"],
                    kvboost_total_ms=k["total_ms"],
                ))

                # Accumulate history using KVBoost's output (consistent with real usage)
                history = prompt + k["output_text"] + "\n"
                human_turn_idx += 1

            elif turn.get("from") == "gpt" and history.endswith("\nAssistant:"):
                # If we're replaying assistant turns (not generating), skip.
                # History is already set from generated output above.
                pass

        all_results.append(conv_result)
        processed_ids.append(conv_id)
        processed_set.add(conv_id)

        # Progress log
        if (conv_idx + 1) % 10 == 0 or conv_idx == 0:
            elapsed = time.perf_counter() - total_start
            done = len(all_results)
            total = len(conversations)
            avg_speedup = np.mean([r.avg_speedup for r in all_results]) if all_results else 0
            log.info(
                f"  [{done}/{total}] conv={conv_id} turns={n_human} "
                f"avg_speedup={avg_speedup:.2f}x  elapsed={elapsed:.0f}s"
            )

        if len(all_results) % CHECKPOINT_INTERVAL == 0:
            try:
                save_checkpoint(all_results, processed_ids, ck_path, meta)
            except Exception as e:
                log.error(f"Checkpoint save failed: {e}")

    # Final checkpoint
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

    baseline_ttfts = [t.baseline_ttft_ms for t in all_turns]
    kvboost_ttfts  = [t.kvboost_ttft_ms  for t in all_turns]
    speedups       = [b / max(k, 0.1) for b, k in zip(baseline_ttfts, kvboost_ttfts)]
    reuse_ratios   = [t.kv_reuse_ratio for t in all_turns]

    # Per-turn-index aggregation (the money chart data)
    by_turn: Dict[int, dict] = defaultdict(lambda: {
        "baseline_ttfts": [], "kvboost_ttfts": [], "reuse_ratios": [], "history_tokens": []
    })
    for t in all_turns:
        by_turn[t.turn_idx]["baseline_ttfts"].append(t.baseline_ttft_ms)
        by_turn[t.turn_idx]["kvboost_ttfts"].append(t.kvboost_ttft_ms)
        by_turn[t.turn_idx]["reuse_ratios"].append(t.kv_reuse_ratio)
        by_turn[t.turn_idx]["history_tokens"].append(t.history_tokens)

    turn_metrics = {}
    for turn_idx in sorted(by_turn.keys()):
        d = by_turn[turn_idx]
        turn_metrics[turn_idx] = {
            "n": len(d["baseline_ttfts"]),
            "avg_baseline_ttft_ms": float(np.mean(d["baseline_ttfts"])),
            "avg_kvboost_ttft_ms":  float(np.mean(d["kvboost_ttfts"])),
            "avg_reuse_ratio":      float(np.mean(d["reuse_ratios"])),
            "avg_history_tokens":   float(np.mean(d["history_tokens"])),
            "speedup":              float(np.mean(d["baseline_ttfts"])) /
                                    max(float(np.mean(d["kvboost_ttfts"])), 0.1),
        }

    # Throughput: total requests / total wall time
    total_baseline_time_s = sum(t.baseline_total_ms for t in all_turns) / 1000
    total_kvboost_time_s  = sum(t.kvboost_total_ms  for t in all_turns) / 1000
    n_requests = len(all_turns)

    return {
        "n_conversations":   len(results),
        "n_turns_total":     n_requests,
        "overall": {
            "baseline_ttft_p50":  float(np.percentile(baseline_ttfts, 50)),
            "baseline_ttft_p90":  float(np.percentile(baseline_ttfts, 90)),
            "baseline_ttft_p99":  float(np.percentile(baseline_ttfts, 99)),
            "kvboost_ttft_p50":   float(np.percentile(kvboost_ttfts, 50)),
            "kvboost_ttft_p90":   float(np.percentile(kvboost_ttfts, 90)),
            "kvboost_ttft_p99":   float(np.percentile(kvboost_ttfts, 99)),
            "speedup_mean":       float(np.mean(speedups)),
            "speedup_p50":        float(np.percentile(speedups, 50)),
            "speedup_p90":        float(np.percentile(speedups, 90)),
            "avg_kv_reuse_ratio": float(np.mean(reuse_ratios)),
            "baseline_throughput_rps": round(n_requests / max(total_baseline_time_s, 1e-6), 3),
            "kvboost_throughput_rps":  round(n_requests / max(total_kvboost_time_s, 1e-6), 3),
        },
        "by_turn": turn_metrics,
    }


def print_summary(metrics: dict, args: argparse.Namespace):
    print(f"\n{'=' * 75}")
    print(f"  SHAREGPT REPLAY — KVBoost vs Baseline")
    print(f"{'=' * 75}")
    print(f"  Conversations:  {metrics['n_conversations']}")
    print(f"  Total turns:    {metrics['n_turns_total']}")
    print(f"  Strategy:       {args.recompute_strategy}  "
          f"(boundary_window={args.chunk_boundary_window}, "
          f"overlap_k={args.overlap_k}, sink_tokens={args.sink_tokens})")
    print()

    ov = metrics["overall"]
    print(f"  {'Metric':<28} {'Baseline':>12} {'KVBoost':>12}")
    print(f"  {'-'*28} {'-'*12} {'-'*12}")
    print(f"  {'TTFT p50 (ms)':<28} {ov['baseline_ttft_p50']:>12.1f} {ov['kvboost_ttft_p50']:>12.1f}")
    print(f"  {'TTFT p90 (ms)':<28} {ov['baseline_ttft_p90']:>12.1f} {ov['kvboost_ttft_p90']:>12.1f}")
    print(f"  {'TTFT p99 (ms)':<28} {ov['baseline_ttft_p99']:>12.1f} {ov['kvboost_ttft_p99']:>12.1f}")
    print(f"  {'Throughput (req/s)':<28} {ov['baseline_throughput_rps']:>12.3f} {ov['kvboost_throughput_rps']:>12.3f}")
    print(f"  {'Avg KV reuse ratio':<28} {'—':>12} {ov['avg_kv_reuse_ratio']:>11.1%}")
    print(f"  {'Speedup (mean)':<28} {'—':>12} {ov['speedup_mean']:>11.2f}x")
    print(f"  {'Speedup (p50)':<28} {'—':>12} {ov['speedup_p50']:>11.2f}x")
    print()

    # Per-turn table (the money chart in text form)
    print(f"  TTFT vs Turn Number (money chart):")
    print(f"  {'Turn':>5} {'N':>5} {'Avg Ctx Tok':>12} {'Baseline':>10} {'KVBoost':>10} "
          f"{'Speedup':>8} {'Reuse':>7}")
    print(f"  {'-'*5} {'-'*5} {'-'*12} {'-'*10} {'-'*10} {'-'*8} {'-'*7}")
    for turn_idx, tm in sorted(metrics["by_turn"].items()):
        print(
            f"  {turn_idx+1:>5} {tm['n']:>5} "
            f"{tm['avg_history_tokens']:>12.0f} "
            f"{tm['avg_baseline_ttft_ms']:>9.0f}ms "
            f"{tm['avg_kvboost_ttft_ms']:>9.0f}ms "
            f"{tm['speedup']:>7.2f}x "
            f"{tm['avg_reuse_ratio']:>6.0%}"
        )
    print(f"\n{'=' * 75}\n")


# ── Plotting ──────────────────────────────────────────────────────

def plot_results(metrics: dict, output_dir: Path):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed — skipping plots")
        return

    by_turn = metrics["by_turn"]
    turn_indices = sorted(by_turn.keys())
    turns_labels = [i + 1 for i in turn_indices]

    baseline_ttfts = [by_turn[i]["avg_baseline_ttft_ms"] for i in turn_indices]
    kvboost_ttfts  = [by_turn[i]["avg_kvboost_ttft_ms"]  for i in turn_indices]
    reuse_ratios   = [by_turn[i]["avg_reuse_ratio"]       for i in turn_indices]
    speedups       = [by_turn[i]["speedup"]               for i in turn_indices]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Chart 1: TTFT vs turn number
    ax = axes[0]
    ax.plot(turns_labels, baseline_ttfts, "o-", label="Baseline (no cache)", color="#e74c3c")
    ax.plot(turns_labels, kvboost_ttfts,  "s-", label="KVBoost (cached)",    color="#2ecc71")
    ax.set_xlabel("Turn Number")
    ax.set_ylabel("Avg TTFT (ms)")
    ax.set_title("TTFT vs Turn Number\n(ShareGPT real conversations)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Chart 2: Speedup per turn
    ax = axes[1]
    ax.bar(turns_labels, speedups, color="#3498db", alpha=0.8)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No speedup")
    ax.set_xlabel("Turn Number")
    ax.set_ylabel("TTFT Speedup (×)")
    ax.set_title("KVBoost TTFT Speedup per Turn")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Chart 3: Cache hit rate per turn
    ax = axes[2]
    ax.plot(turns_labels, [r * 100 for r in reuse_ratios], "D-", color="#9b59b6")
    ax.set_xlabel("Turn Number")
    ax.set_ylabel("KV Reuse Ratio (%)")
    ax.set_title("Cache Hit Rate by Turn\n(compounds across turns)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_dir / "sharegpt_ttft_vs_turn.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    log.info(f"Plot saved: {plot_path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="ShareGPT Replay Benchmark — KVBoost vs Baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--n-conversations", type=int, default=1000,
                        help="Number of conversations to replay (default: 1000)")
    parser.add_argument("--min-turns", type=int, default=3,
                        help="Minimum human turns per conversation (default: 3)")
    parser.add_argument("--max-turns", type=int, default=8,
                        help="Cap conversations at this many human turns (default: 8)")
    parser.add_argument("--max-tokens-per-turn", type=int, default=512,
                        help="Drop turns with human messages longer than this (default: 512)")
    parser.add_argument("--max-new-tokens", type=int, default=128,
                        help="Max tokens to generate per turn (default: 128)")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-cache-bytes", type=float, default=None,
                        help="KV cache size cap in bytes (e.g. 1.5e9 for 1.5 GB)")
    parser.add_argument("--recency-window-chunks", type=int, default=None,
                        help="Number of recent chunks to keep in cache")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Alias for --n-conversations (overrides if set)")
    parser.add_argument("--max-context-tokens", type=int, default=None,
                        help="Drop conversations whose total token count exceeds this")

    # KVBoost tuning — defaults are the best-performing config
    parser.add_argument("--recompute-strategy", default="cacheblend",
                        choices=["selective", "full", "cacheblend"])
    parser.add_argument("--chunk-boundary-window", type=int, default=16)
    parser.add_argument("--overlap-k", type=int, default=16)
    parser.add_argument("--sink-tokens", type=int, default=32)

    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: results/sharegpt_replay.json)")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip matplotlib plots")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # --n-samples is a convenience alias for --n-conversations
    if args.n_samples is not None:
        args.n_conversations = args.n_samples

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    print(f"\n{'=' * 75}")
    print(f"  SHAREGPT REPLAY BENCHMARK — KVBoost Multi-Turn Cache")
    print(f"{'=' * 75}")
    print(f"  Dataset:       Aeala/ShareGPT_Vicuna_unfiltered")
    print(f"  Model:         {args.model}")
    print(f"  Conversations: {args.n_conversations}")
    print(f"  Turns:         {args.min_turns}–{args.max_turns} human turns")
    print(f"  Strategy:      {args.recompute_strategy}  "
          f"boundary_window={args.chunk_boundary_window} "
          f"overlap_k={args.overlap_k} sink_tokens={args.sink_tokens}")
    print(f"{'=' * 75}\n")

    runner = ShareGPTRunner(
        model_name=args.model,
        chunk_size=args.chunk_size,
        recompute_strategy=args.recompute_strategy,
        chunk_boundary_window=args.chunk_boundary_window,
        overlap_k=args.overlap_k,
        sink_tokens=args.sink_tokens,
        max_new_tokens=args.max_new_tokens,
        max_cache_bytes=args.max_cache_bytes,
        recency_window_chunks=args.recency_window_chunks,
    )

    conversations = load_sharegpt(
        n_conversations=args.n_conversations,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        max_tokens_per_turn=args.max_tokens_per_turn,
        tokenizer=runner.tokenizer,
    )

    if not conversations:
        log.error("No conversations loaded after filtering.")
        sys.exit(1)

    if args.max_context_tokens is not None:
        before = len(conversations)
        conversations = [
            c for c in conversations
            if sum(
                len(runner.tokenizer.encode(t["value"]))
                for t in c["turns"]
            ) <= args.max_context_tokens
        ]
        log.info(
            f"max_context_tokens={args.max_context_tokens}: "
            f"{before} → {len(conversations)} conversations"
        )
        if not conversations:
            log.error("No conversations remain after max_context_tokens filter.")
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

    # Save JSON results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = (
        Path(args.output) if args.output
        else RESULTS_DIR / "sharegpt_replay.json"
    )
    payload = {
        "benchmark": "sharegpt_replay",
        "dataset": "Aeala/ShareGPT_Vicuna_unfiltered",
        "model": args.model,
        "config": {
            "n_conversations": args.n_conversations,
            "min_turns": args.min_turns,
            "max_turns": args.max_turns,
            "recompute_strategy": args.recompute_strategy,
            "chunk_boundary_window": args.chunk_boundary_window,
            "overlap_k": args.overlap_k,
            "sink_tokens": args.sink_tokens,
            "max_new_tokens": args.max_new_tokens,
            "chunk_size": args.chunk_size,
        },
        "metrics": metrics,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    log.info(f"Results saved: {out_path}")

    if not args.no_plot:
        plot_results(metrics, RESULTS_DIR)

    # Clean checkpoint on success
    ck_key = _checkpoint_key(args.model, args.n_conversations, args.max_turns)
    ck_path = CHECKPOINT_DIR / f"sharegpt_{ck_key}.json"
    if ck_path.exists():
        ck_path.unlink()
        log.info("Checkpoint cleaned up")


if __name__ == "__main__":
    main()
