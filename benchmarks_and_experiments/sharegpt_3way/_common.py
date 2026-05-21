"""
Shared scaffolding for the 3-way ShareGPT benchmark.

Every backend runner (KVBoost / vLLM / llama.cpp) writes the SAME
TurnResult JSON schema so that `compare.py` can read them side-by-side.

Per-turn metrics captured
-------------------------
- ttft_ms                  : time-to-first-token (engine-reported when available)
- total_ms                 : full request wall time
- decode_ms                : total_ms - ttft_ms
- output_tokens            : decoded token count
- itl_ms                   : mean inter-token latency = decode_ms / max(output_tokens-1, 1)
- decode_tps               : output_tokens / (decode_ms / 1000)
- prompt_tokens            : tokenized full prompt
- cached_tokens            : tokens served from cache (KV reuse / prefix hit)
- cache_hit_ratio          : cached_tokens / prompt_tokens
- spec_accepted            : draft tokens accepted this turn (None if no spec)
- spec_proposed            : draft tokens proposed this turn (None if no spec)
- spec_rounds              : speculative rounds this turn (None if no spec)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

log = logging.getLogger("sharegpt_3way")


# ── Data containers ─────────────────────────────────────────────────────

@dataclass
class TurnResult:
    conv_id: str
    turn_idx: int
    n_turns_total: int
    history_tokens: int
    prompt_tokens: int
    cached_tokens: int
    cache_hit_ratio: float
    ttft_ms: float
    total_ms: float
    decode_ms: float
    output_tokens: int
    itl_ms: float
    decode_tps: float
    spec_accepted: Optional[int] = None
    spec_proposed: Optional[int] = None
    spec_rounds: Optional[int] = None


@dataclass
class ConvResult:
    conv_id: str
    n_turns: int
    turns: List[TurnResult] = field(default_factory=list)


# ── ShareGPT loading ────────────────────────────────────────────────────

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
        n = 0
        for t in turns:
            capped.append(t)
            if t.get("from") == "human":
                n += 1
                if n >= max_turns:
                    break

        msgs = [t["value"] for t in capped if t.get("from") == "human"]
        if any(len(tokenizer.encode(m)) > max_tokens_per_turn for m in msgs):
            continue

        conversations.append({"id": raw.get("id", f"conv_{len(conversations)}"), "turns": capped})

    log.info(f"  After turn-filter: {len(conversations)} conversations")

    if max_context_tokens is not None:
        before = len(conversations)
        conversations = [
            c for c in conversations
            if sum(len(tokenizer.encode(t["value"])) for t in c["turns"]) <= max_context_tokens
        ]
        log.info(f"  max_context_tokens={max_context_tokens}: {before} → {len(conversations)}")

    if len(conversations) > n_conversations:
        idx = rng.choice(len(conversations), n_conversations, replace=False)
        conversations = [conversations[i] for i in sorted(idx)]

    log.info(f"  Sampled: {len(conversations)} conversations")
    turn_counts = [sum(1 for t in c["turns"] if t.get("from") == "human") for c in conversations]
    log.info(
        f"  Turn distribution: min={min(turn_counts)} max={max(turn_counts)} "
        f"mean={np.mean(turn_counts):.1f} median={np.median(turn_counts):.1f}"
    )
    return conversations


# ── Checkpointing ───────────────────────────────────────────────────────

def checkpoint_key(backend: str, model: str, n_conversations: int, max_turns: int) -> str:
    return hashlib.md5(f"{backend}_{model}_{n_conversations}_{max_turns}".encode()).hexdigest()[:8]


def save_checkpoint(results: List[ConvResult], processed_ids: List[str], path: Path, meta: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "meta": meta,
        "timestamp": time.time(),
        "processed_ids": processed_ids,
        "results": [
            {
                "conv_id": r.conv_id,
                "n_turns": r.n_turns,
                "turns": [asdict(t) for t in r.turns],
            }
            for r in results
        ],
    }
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, default=str)
    tmp.replace(path)
    log.info(f"Checkpoint: {len(processed_ids)} conversations → {path.name}")


def load_checkpoint(path: Path) -> tuple[List[ConvResult], List[str]]:
    if not path.exists():
        return [], []
    try:
        with open(path) as f:
            data = json.load(f)
        results = [
            ConvResult(
                conv_id=cr["conv_id"],
                n_turns=cr["n_turns"],
                turns=[TurnResult(**t) for t in cr["turns"]],
            )
            for cr in data.get("results", [])
        ]
        return results, data.get("processed_ids", [])
    except Exception as e:
        log.warning(f"Checkpoint load failed: {e} — starting fresh")
        return [], []


# ── Replay loop (generic over backend.run_turn) ─────────────────────────

def replay_conversations(
    *,
    run_turn: Callable[[str], dict],
    count_tokens: Callable[[str], int],
    reset_between_convs: Optional[Callable[[], None]],
    conversations: List[dict],
    ck_path: Path,
    meta: dict,
    no_checkpoint: bool = False,
    checkpoint_interval: int = 10,
) -> List[ConvResult]:
    if no_checkpoint:
        all_results, processed_ids = [], []
    else:
        all_results, processed_ids = load_checkpoint(ck_path)
    processed_set = set(processed_ids)

    total_start = time.perf_counter()
    for conv_idx, conv in enumerate(conversations):
        conv_id = conv["id"]
        if conv_id in processed_set:
            continue

        if reset_between_convs is not None:
            reset_between_convs()

        turns = conv["turns"]
        n_human = sum(1 for t in turns if t.get("from") == "human")
        conv_result = ConvResult(conv_id=conv_id, n_turns=n_human)

        history = ""
        human_turn_idx = 0
        for turn in turns:
            if turn.get("from") != "human":
                continue
            prompt = history + f"Human: {turn['value']}\nAssistant:"
            history_tokens = count_tokens(prompt)

            r = run_turn(prompt)

            decode_ms = max(r["total_ms"] - r["ttft_ms"], 0.0)
            out_tok = int(r.get("output_tokens", 0))
            itl_ms = decode_ms / max(out_tok - 1, 1)
            decode_tps = (out_tok / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0
            prompt_tokens = int(r.get("prompt_tokens", history_tokens))
            cached = int(r.get("cached_tokens", 0))

            conv_result.turns.append(TurnResult(
                conv_id=conv_id,
                turn_idx=human_turn_idx,
                n_turns_total=n_human,
                history_tokens=history_tokens,
                prompt_tokens=prompt_tokens,
                cached_tokens=cached,
                cache_hit_ratio=cached / max(prompt_tokens, 1),
                ttft_ms=float(r["ttft_ms"]),
                total_ms=float(r["total_ms"]),
                decode_ms=decode_ms,
                output_tokens=out_tok,
                itl_ms=itl_ms,
                decode_tps=decode_tps,
                spec_accepted=r.get("spec_accepted"),
                spec_proposed=r.get("spec_proposed"),
                spec_rounds=r.get("spec_rounds"),
            ))

            history = prompt + r.get("output_text", "") + "\n"
            human_turn_idx += 1

        all_results.append(conv_result)
        processed_ids.append(conv_id)
        processed_set.add(conv_id)

        if (conv_idx + 1) % 10 == 0 or conv_idx == 0:
            elapsed = time.perf_counter() - total_start
            ttfts = [t.ttft_ms for r in all_results for t in r.turns]
            ttft_p50 = float(np.percentile(ttfts, 50)) if ttfts else 0.0
            log.info(
                f"  [{len(all_results)}/{len(conversations)}] conv={conv_id} "
                f"turns={n_human} ttft_p50={ttft_p50:.0f}ms elapsed={elapsed:.0f}s"
            )

        if len(all_results) % checkpoint_interval == 0:
            try:
                save_checkpoint(all_results, processed_ids, ck_path, meta)
            except Exception as e:
                log.error(f"Checkpoint save failed: {e}")

    if all_results:
        try:
            save_checkpoint(all_results, processed_ids, ck_path, meta)
        except Exception as e:
            log.error(f"Final checkpoint save failed: {e}")

    log.info(f"Replay completed: {len(all_results)} conversations in "
             f"{time.perf_counter() - total_start:.1f}s")
    return all_results


# ── Metrics ─────────────────────────────────────────────────────────────

def compute_metrics(results: List[ConvResult], total_wall_s: float) -> dict:
    all_turns = [t for r in results for t in r.turns]
    if not all_turns:
        return {}

    ttfts        = [t.ttft_ms for t in all_turns]
    itls         = [t.itl_ms for t in all_turns if t.output_tokens > 1]
    decode_tps   = [t.decode_tps for t in all_turns if t.decode_tps > 0]
    out_tokens   = [t.output_tokens for t in all_turns]
    hit_ratios   = [t.cache_hit_ratio for t in all_turns]
    cached_tok   = [t.cached_tokens for t in all_turns]
    prompt_tok   = [t.prompt_tokens for t in all_turns]

    total_out_tok = sum(out_tokens)
    total_req_time_s = sum(t.total_ms for t in all_turns) / 1000.0

    by_turn: Dict[int, dict] = defaultdict(lambda: {
        "ttfts": [], "itls": [], "decode_tps": [], "hit": [], "hist": [], "out": []
    })
    for t in all_turns:
        by_turn[t.turn_idx]["ttfts"].append(t.ttft_ms)
        if t.output_tokens > 1:
            by_turn[t.turn_idx]["itls"].append(t.itl_ms)
        if t.decode_tps > 0:
            by_turn[t.turn_idx]["decode_tps"].append(t.decode_tps)
        by_turn[t.turn_idx]["hit"].append(t.cache_hit_ratio)
        by_turn[t.turn_idx]["hist"].append(t.history_tokens)
        by_turn[t.turn_idx]["out"].append(t.output_tokens)

    turn_metrics = {}
    for k in sorted(by_turn.keys()):
        d = by_turn[k]
        turn_metrics[k] = {
            "n": len(d["ttfts"]),
            "ttft_p50": float(np.percentile(d["ttfts"], 50)),
            "ttft_p90": float(np.percentile(d["ttfts"], 90)),
            "itl_p50":  float(np.percentile(d["itls"], 50)) if d["itls"] else None,
            "decode_tps_mean": float(np.mean(d["decode_tps"])) if d["decode_tps"] else None,
            "cache_hit_ratio_mean": float(np.mean(d["hit"])),
            "avg_history_tokens": float(np.mean(d["hist"])),
            "avg_output_tokens":  float(np.mean(d["out"])),
        }

    # Speculative aggregates (only where reported)
    spec_acc = [t.spec_accepted for t in all_turns if t.spec_accepted is not None]
    spec_prop = [t.spec_proposed for t in all_turns if t.spec_proposed is not None]
    spec_acceptance_rate = (
        float(sum(spec_acc) / max(sum(spec_prop), 1)) if spec_acc and spec_prop else None
    )

    return {
        "n_conversations": len(results),
        "n_turns_total":   len(all_turns),
        "overall": {
            "ttft_p50_ms":      float(np.percentile(ttfts, 50)),
            "ttft_p90_ms":      float(np.percentile(ttfts, 90)),
            "ttft_p99_ms":      float(np.percentile(ttfts, 99)),
            "ttft_mean_ms":     float(np.mean(ttfts)),
            "itl_p50_ms":       float(np.percentile(itls, 50)) if itls else None,
            "itl_p90_ms":       float(np.percentile(itls, 90)) if itls else None,
            "decode_tps_mean":  float(np.mean(decode_tps)) if decode_tps else None,
            "decode_tps_p50":   float(np.percentile(decode_tps, 50)) if decode_tps else None,
            "avg_cache_hit_ratio": float(np.mean(hit_ratios)),
            "avg_cached_tokens": float(np.mean(cached_tok)),
            "avg_prompt_tokens": float(np.mean(prompt_tok)),
            "avg_output_tokens": float(np.mean(out_tokens)),
            "request_throughput_rps":   round(len(all_turns) / max(total_wall_s, 1e-6), 4),
            "output_token_throughput":  round(total_out_tok / max(total_wall_s, 1e-6), 2),
            "per_request_tps_mean":     round(total_out_tok / max(total_req_time_s, 1e-6), 2),
            "spec_acceptance_rate":     spec_acceptance_rate,
        },
        "by_turn": turn_metrics,
    }


# ── Pretty-printing ─────────────────────────────────────────────────────

def print_summary(backend: str, metrics: dict):
    ov = metrics["overall"]
    print(f"\n{'=' * 72}")
    print(f"  {backend.upper()} — ShareGPT replay summary")
    print(f"{'=' * 72}")
    print(f"  Conversations: {metrics['n_conversations']}   Turns: {metrics['n_turns_total']}")
    print()
    print(f"  {'Metric':<32}{'Value':>20}")
    print(f"  {'-' * 32} {'-' * 19}")
    rows = [
        ("TTFT p50 (ms)",            f"{ov['ttft_p50_ms']:.1f}"),
        ("TTFT p90 (ms)",            f"{ov['ttft_p90_ms']:.1f}"),
        ("TTFT p99 (ms)",            f"{ov['ttft_p99_ms']:.1f}"),
        ("ITL p50 (ms/tok)",         f"{ov['itl_p50_ms']:.2f}" if ov['itl_p50_ms'] else "—"),
        ("ITL p90 (ms/tok)",         f"{ov['itl_p90_ms']:.2f}" if ov['itl_p90_ms'] else "—"),
        ("Decode tok/s (mean)",      f"{ov['decode_tps_mean']:.2f}" if ov['decode_tps_mean'] else "—"),
        ("Cache hit ratio (mean)",   f"{ov['avg_cache_hit_ratio']:.1%}"),
        ("Avg cached tokens",        f"{ov['avg_cached_tokens']:.0f}"),
        ("Avg prompt tokens",        f"{ov['avg_prompt_tokens']:.0f}"),
        ("Avg output tokens",        f"{ov['avg_output_tokens']:.0f}"),
        ("Request throughput (rps)", f"{ov['request_throughput_rps']:.3f}"),
        ("Output tok/s (wall)",      f"{ov['output_token_throughput']:.2f}"),
        ("Spec acceptance rate",     f"{ov['spec_acceptance_rate']:.1%}" if ov['spec_acceptance_rate'] is not None else "—"),
    ]
    for k, v in rows:
        print(f"  {k:<32}{v:>20}")
    print(f"{'=' * 72}\n")


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of conversations to replay (default: 500)")
    parser.add_argument("--min-turns", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens-per-turn", type=int, default=512)
    parser.add_argument("--max-context-tokens", type=int, default=6000)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gamma", type=int, default=5,
                        help="Speculative draft tokens per round (default: 5)")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: results/<backend>.json)")
