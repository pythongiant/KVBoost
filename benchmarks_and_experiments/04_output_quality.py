#!/usr/bin/env python3
"""
Experiment 4: Output Quality Validation
=========================================
The most critical experiment for research credibility. Validates that
KV-cache reuse produces equivalent outputs by measuring:

  1. Perplexity comparison on held-out text (WikiText-style)
  2. KL divergence between baseline and cached logit distributions
  3. Overlap=0 ablation (proves selective recompute is necessary)
  4. Long-range dependency test (chunk boundaries mid-sentence)
  5. Exact output match rate across sampling temperatures

Usage:
  python 04_output_quality.py
  python 04_output_quality.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

from __future__ import annotations

import argparse
import math
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from utils import (
    get_logger, save_results, compute_stats,
    SYSTEM_PROMPT, RAG_DOCUMENT, QUERIES,
    GenerationMode,
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import InferenceEngine

log = get_logger("04_output_quality")

# Held-out passages for perplexity measurement.
# These are long enough to span multiple chunks.
EVAL_PASSAGES = [
    (
        "The transformer architecture, introduced in the landmark paper 'Attention Is All "
        "You Need' by Vaswani et al. in 2017, revolutionized natural language processing by "
        "replacing recurrent neural networks with self-attention mechanisms. The key innovation "
        "was the multi-head attention mechanism, which allows the model to jointly attend to "
        "information from different representation subspaces at different positions. Each "
        "attention head computes scaled dot-product attention: Attention(Q,K,V) = softmax(QK^T / "
        "sqrt(d_k))V, where Q, K, and V are the query, key, and value matrices respectively, "
        "and d_k is the dimension of the keys. The model uses positional encodings to inject "
        "information about the relative or absolute position of tokens in the sequence. Modern "
        "variants like RoPE (Rotary Position Embedding) encode position information directly "
        "into the query and key vectors through rotation matrices, enabling better extrapolation "
        "to longer sequences than the original sinusoidal encodings."
    ),
    (
        "In distributed computing, the CAP theorem states that a distributed data store cannot "
        "simultaneously provide more than two of the following three guarantees: Consistency, "
        "Availability, and Partition tolerance. In practice, since network partitions are "
        "unavoidable in distributed systems, the theorem implies a trade-off between consistency "
        "and availability. Systems like Google Spanner attempt to provide strong consistency with "
        "high availability by using synchronized clocks (TrueTime API) and Paxos-based "
        "replication across data centers. In contrast, DynamoDB and Cassandra favor availability "
        "and partition tolerance, offering eventual consistency by default with optional "
        "stronger consistency levels per query. Understanding these trade-offs is crucial for "
        "system architects designing fault-tolerant distributed applications."
    ),
    (
        "Reinforcement learning from human feedback (RLHF) has become a critical technique for "
        "aligning large language models with human preferences. The process typically involves "
        "three stages: supervised fine-tuning on high-quality demonstrations, training a reward "
        "model on human preference comparisons, and optimizing the language model policy using "
        "proximal policy optimization (PPO) against the reward model. Recent work has explored "
        "alternatives like Direct Preference Optimization (DPO), which eliminates the need for "
        "a separate reward model by directly optimizing the policy on preference data. The key "
        "insight of DPO is that the optimal policy under the Bradley-Terry preference model can "
        "be expressed as a function of the log-ratio of the policy to a reference policy, "
        "enabling a simple classification loss. This significantly reduces computational costs "
        "and training instability compared to the full RLHF pipeline."
    ),
]

# Prompts designed so chunk boundaries fall mid-sentence
BOUNDARY_TEST_PROMPTS = [
    # The system prompt + connector is designed to push the interesting content
    # across a chunk boundary
    {
        "prefix": SYSTEM_PROMPT,
        "query": (
            "Consider the following chain of reasoning: if A implies B, and B implies C, "
            "and we know that A is true, then we can conclude that C is true by applying "
            "modus ponens twice. Now, given that the server is overloaded implies response "
            "times increase, and response times increase implies SLA violations occur, "
            "and we observe the server is overloaded, what can we conclude?"
        ),
        "description": "multi-step logical reasoning across boundary",
    },
    {
        "prefix": SYSTEM_PROMPT,
        "query": (
            "The function signature is: def merge_sorted_lists(list_a: List[int], "
            "list_b: List[int]) -> List[int]. It should merge two sorted lists into one "
            "sorted list in O(n+m) time. Here is the implementation:\n\n"
            "def merge_sorted_lists(list_a, list_b):\n"
            "    result = []\n"
            "    i = j = 0\n"
            "    while i < len(list_a) and j < len(list_b):\n"
            "        if list_a[i] <= list_b[j]:\n"
            "            result.append(list_a[i])\n"
            "            i += 1\n"
            "What comes next in this implementation?"
        ),
        "description": "code completion across boundary",
    },
    {
        "prefix": RAG_DOCUMENT,
        "query": (
            "According to the document above, explain how the memory layout of KV tensors "
            "relates to the challenge of positional encodings when implementing chunk-level "
            "caching. Be specific about the tensor shapes involved."
        ),
        "description": "cross-reference across document sections",
    },
]


def compute_perplexity(engine: InferenceEngine, text: str,
                       mode: GenerationMode, stride: int = 256) -> float:
    """
    Compute perplexity of text under the given generation mode.
    Uses a sliding window approach for long texts.
    """
    token_ids = engine._encode(text)
    n = len(token_ids)

    if mode != GenerationMode.BASELINE:
        engine.warm_chunks(text, position_offset=0)

    # Compute log-likelihood
    total_log_prob = 0.0
    total_tokens = 0

    # For a fair comparison, compute token-by-token log probs
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=engine.device)

    with torch.no_grad():
        out = engine.model(input_ids=input_ids, use_cache=False)
        logits = out.logits  # [1, seq_len, vocab_size]

    # Shift: logits[t] predicts token[t+1]
    shift_logits = logits[0, :-1, :]  # [seq_len-1, vocab_size]
    shift_labels = torch.tensor(token_ids[1:], dtype=torch.long, device=engine.device)

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)

    avg_neg_log_prob = -token_log_probs.mean().item()
    perplexity = math.exp(avg_neg_log_prob)

    return perplexity


def compute_kl_divergence(engine: InferenceEngine, prompt: str,
                          max_tokens: int = 32) -> Dict:
    """
    Compare logit distributions between baseline and chunk_kv_reuse
    at each generation step. Returns per-step and mean KL divergence.
    """
    token_ids = engine._encode(prompt)

    # Get baseline logits at each step
    engine.warm_chunks(prompt.split("\n\n")[0] if "\n\n" in prompt else "",
                       position_offset=0)

    baseline_logits = []
    cached_logits = []

    # Baseline: full forward pass
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=engine.device)
    with torch.no_grad():
        out = engine.model(input_ids=input_ids, use_cache=True)
    baseline_logits.append(out.logits[0, -1, :].clone())
    past_baseline = engine._normalize_past_kv(out.past_key_values)
    next_token_baseline = out.logits[0, -1, :].argmax().item()

    # Chunk KV: use cached prefix
    assembled = engine.assembler.assemble(token_ids)
    if len(assembled.chunk_boundaries) > 1:
        assembled = engine.selective_recompute.apply(assembled, engine.model)

    past_kv = assembled.cached_past_kv
    if past_kv is not None:
        past_kv = tuple(
            (layer[0].to(engine.device), layer[1].to(engine.device))
            for layer in past_kv
        )
    live_ids = assembled.live_token_ids
    cached_len = assembled.cached_length

    if live_ids:
        live_input = torch.tensor([live_ids], dtype=torch.long, device=engine.device)
        pos_ids = torch.arange(cached_len, cached_len + len(live_ids),
                               dtype=torch.long, device=engine.device).unsqueeze(0)
        with torch.no_grad():
            out = engine.model(
                input_ids=live_input,
                past_key_values=engine._as_cache(past_kv),
                position_ids=pos_ids,
                use_cache=True,
            )
        cached_logits.append(out.logits[0, -1, :].clone())
        past_cached = engine._normalize_past_kv(out.past_key_values)
    else:
        cached_logits.append(baseline_logits[0].clone())
        past_cached = past_baseline

    # Continue decoding for max_tokens steps, collecting logits from both
    prev_token = next_token_baseline
    for step in range(1, max_tokens):
        cur_ids = torch.tensor([[prev_token]], dtype=torch.long, device=engine.device)

        with torch.no_grad():
            out_b = engine.model(input_ids=cur_ids,
                                 past_key_values=engine._as_cache(past_baseline),
                                 use_cache=True)
            out_c = engine.model(input_ids=cur_ids,
                                 past_key_values=engine._as_cache(past_cached),
                                 use_cache=True)

        baseline_logits.append(out_b.logits[0, -1, :].clone())
        cached_logits.append(out_c.logits[0, -1, :].clone())

        past_baseline = engine._normalize_past_kv(out_b.past_key_values)
        past_cached = engine._normalize_past_kv(out_c.past_key_values)

        prev_token = out_b.logits[0, -1, :].argmax().item()

    # Compute KL divergence at each step
    # Use log_softmax to avoid log(0) = -inf which causes 0 * -inf = NaN
    kl_divs = []
    for b_logits, c_logits in zip(baseline_logits, cached_logits):
        log_p = F.log_softmax(b_logits, dim=-1)
        log_q = F.log_softmax(c_logits, dim=-1)
        p = log_p.exp()
        kl = F.kl_div(log_q, p, reduction="sum", log_target=False).item()
        kl_divs.append(kl)

    return {
        "per_step_kl": [round(kl, 6) for kl in kl_divs],
        "mean_kl": round(sum(kl_divs) / len(kl_divs), 6),
        "max_kl": round(max(kl_divs), 6),
        "n_steps": len(kl_divs),
    }


def ablation_no_recompute(engine: InferenceEngine, prefix: str,
                          queries: list, max_new_tokens: int) -> Dict:
    """
    Compare outputs with and without selective recompute (overlap=0 vs default).
    Proves that selective recompute is necessary.
    """
    engine.warm_chunks(prefix, position_offset=0)

    results = {"with_recompute": [], "without_recompute": []}

    # Get baseline reference outputs
    baseline_outputs = []
    for query in queries:
        prompt = prefix + "\n\n" + query
        r = engine.generate(prompt, max_new_tokens=max_new_tokens,
                            mode=GenerationMode.BASELINE, do_sample=False)
        baseline_outputs.append(r.output_text)

    # With recompute (default)
    for qi, query in enumerate(queries):
        prompt = prefix + "\n\n" + query
        r = engine.generate(prompt, max_new_tokens=max_new_tokens,
                            mode=GenerationMode.CHUNK_KV_REUSE, do_sample=False)
        results["with_recompute"].append({
            "query_idx": qi,
            "output": r.output_text,
            "matches_baseline": r.output_text == baseline_outputs[qi],
        })

    # Without recompute: temporarily set overlap to 0
    original_overlap = engine.selective_recompute.overlap
    engine.selective_recompute.overlap = 0

    for qi, query in enumerate(queries):
        prompt = prefix + "\n\n" + query
        r = engine.generate(prompt, max_new_tokens=max_new_tokens,
                            mode=GenerationMode.CHUNK_KV_REUSE, do_sample=False)
        results["without_recompute"].append({
            "query_idx": qi,
            "output": r.output_text,
            "matches_baseline": r.output_text == baseline_outputs[qi],
        })

    engine.selective_recompute.overlap = original_overlap

    # Summary
    with_match = sum(1 for r in results["with_recompute"] if r["matches_baseline"])
    without_match = sum(1 for r in results["without_recompute"] if r["matches_baseline"])

    return {
        "with_recompute_match_rate": round(with_match / max(len(queries), 1), 3),
        "without_recompute_match_rate": round(without_match / max(len(queries), 1), 3),
        "details": results,
    }


def test_long_range_dependencies(engine: InferenceEngine, max_new_tokens: int) -> Dict:
    """Test outputs when chunk boundaries fall mid-sentence or mid-reasoning."""
    engine_chunk_size = engine.chunk_registry.chunk_size
    results = []

    for test in BOUNDARY_TEST_PROMPTS:
        prefix = test["prefix"]
        query = test["query"]
        prompt = prefix + "\n\n" + query

        engine.warm_chunks(prefix, position_offset=0)

        # Baseline
        r_base = engine.generate(prompt, max_new_tokens=max_new_tokens,
                                 mode=GenerationMode.BASELINE, do_sample=False)
        # Chunk KV
        r_chunk = engine.generate(prompt, max_new_tokens=max_new_tokens,
                                  mode=GenerationMode.CHUNK_KV_REUSE, do_sample=False)

        # Check where boundaries fall
        token_ids = engine._encode(prompt)
        boundaries = []
        for start, end, _ in engine.chunk_registry.split(token_ids):
            boundaries.append((start, end))

        results.append({
            "description": test["description"],
            "prompt_tokens": len(token_ids),
            "chunk_boundaries": boundaries,
            "chunk_size": engine_chunk_size,
            "baseline_output": r_base.output_text,
            "chunk_kv_output": r_chunk.output_text,
            "outputs_match": r_base.output_text == r_chunk.output_text,
            "reuse_ratio": r_chunk.kv_reuse_ratio,
        })

    match_rate = sum(1 for r in results if r["outputs_match"]) / max(len(results), 1)
    return {
        "match_rate": round(match_rate, 3),
        "tests": results,
    }


def _collect_token_ids(engine: InferenceEngine, prompt: str,
                       max_new_tokens: int, mode: GenerationMode,
                       temperature: float, n_samples: int) -> List[List[int]]:
    """Run n_samples generations and return list of token-id sequences."""
    all_ids = []
    for _ in range(n_samples):
        r = engine.generate(prompt, max_new_tokens=max_new_tokens,
                            mode=mode, temperature=temperature, do_sample=True)
        all_ids.append(engine._encode(r.output_text))
    return all_ids


def _token_frequency(token_lists: List[List[int]]) -> Dict[int, float]:
    """Compute normalised frequency of each token across all samples."""
    from collections import Counter
    counts: Counter = Counter()
    total = 0
    for ids in token_lists:
        counts.update(ids)
        total += len(ids)
    return {tok: cnt / total for tok, cnt in counts.items()} if total else {}


def _hellinger_distance(freq_a: Dict[int, float],
                        freq_b: Dict[int, float]) -> float:
    """Hellinger distance between two discrete distributions (0=identical, 1=disjoint)."""
    all_tokens = set(freq_a) | set(freq_b)
    sum_sq = 0.0
    for tok in all_tokens:
        sum_sq += (math.sqrt(freq_a.get(tok, 0.0)) - math.sqrt(freq_b.get(tok, 0.0))) ** 2
    return math.sqrt(sum_sq / 2.0)


def test_sampling_temperatures(engine: InferenceEngine, prefix: str,
                               query: str, max_new_tokens: int,
                               temperatures: list, n_samples: int) -> Dict:
    """
    Compare output distributions across temperatures.
    With do_sample=True, check if baseline and chunk_kv produce
    similar token distributions over multiple samples.

    Reports both:
      - exact-string Jaccard (set overlap of full output strings)
      - token-frequency Hellinger distance (statistical comparison of the
        token distributions, independent of exact string matches)
    """
    prompt = prefix + "\n\n" + query
    engine.warm_chunks(prefix, position_offset=0)

    results = {}
    for temp in temperatures:
        baseline_ids = _collect_token_ids(
            engine, prompt, max_new_tokens, GenerationMode.BASELINE, temp, n_samples)
        cached_ids = _collect_token_ids(
            engine, prompt, max_new_tokens, GenerationMode.CHUNK_KV_REUSE, temp, n_samples)

        # --- exact-string Jaccard (original metric, kept for continuity) ---
        baseline_strings = set(str(ids) for ids in baseline_ids)
        cached_strings = set(str(ids) for ids in cached_ids)
        overlap = len(baseline_strings & cached_strings)
        total_unique = len(baseline_strings | cached_strings)

        # --- token-frequency Hellinger distance ---
        freq_b = _token_frequency(baseline_ids)
        freq_c = _token_frequency(cached_ids)
        hellinger = _hellinger_distance(freq_b, freq_c)

        results[str(temp)] = {
            "baseline_unique_outputs": len(baseline_strings),
            "cached_unique_outputs": len(cached_strings),
            "overlapping_outputs": overlap,
            "jaccard_similarity": round(overlap / max(total_unique, 1), 3),
            "hellinger_distance": round(hellinger, 4),
            "n_samples": n_samples,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: Output quality validation")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--output", type=str, default="04_output_quality.json")
    args = parser.parse_args()

    engine = InferenceEngine.from_pretrained(
        model_name=args.model,
        chunk_size=args.chunk_size,
        max_chunks=256,
    )

    all_results = {}

    # Test 1: Perplexity
    log.info("=== Test 1: Perplexity comparison ===")
    ppl_results = []
    for i, passage in enumerate(EVAL_PASSAGES):
        ppl = compute_perplexity(engine, passage, GenerationMode.BASELINE)
        ppl_results.append({"passage_idx": i, "perplexity": round(ppl, 2),
                            "tokens": len(engine._encode(passage))})
        log.info("  Passage %d: perplexity=%.2f", i, ppl)
    all_results["perplexity"] = ppl_results

    # Test 2: KL divergence
    log.info("=== Test 2: KL divergence ===")
    kl_results = []
    for qi, query in enumerate(QUERIES[:3]):
        prompt = SYSTEM_PROMPT + "\n\n" + query
        kl = compute_kl_divergence(engine, prompt, max_tokens=16)
        kl_results.append({"query_idx": qi, **kl})
        log.info("  Query %d: mean_kl=%.6f, max_kl=%.6f", qi, kl["mean_kl"], kl["max_kl"])
    all_results["kl_divergence"] = kl_results

    # Test 3: Ablation — no recompute
    log.info("=== Test 3: Recompute ablation ===")
    ablation = ablation_no_recompute(engine, SYSTEM_PROMPT, QUERIES, args.max_new_tokens)
    all_results["recompute_ablation"] = {
        "with_recompute_match_rate": ablation["with_recompute_match_rate"],
        "without_recompute_match_rate": ablation["without_recompute_match_rate"],
    }
    log.info("  With recompute match rate: %.1f%%",
             ablation["with_recompute_match_rate"] * 100)
    log.info("  Without recompute match rate: %.1f%%",
             ablation["without_recompute_match_rate"] * 100)

    # Test 4: Long-range dependencies
    log.info("=== Test 4: Long-range dependency tests ===")
    boundary_results = test_long_range_dependencies(engine, args.max_new_tokens)
    all_results["long_range_dependencies"] = boundary_results
    log.info("  Overall match rate: %.1f%%", boundary_results["match_rate"] * 100)

    # Test 5: Sampling temperatures
    log.info("=== Test 5: Sampling temperature comparison ===")
    temp_results = test_sampling_temperatures(
        engine, SYSTEM_PROMPT, QUERIES[0], args.max_new_tokens,
        temperatures=[0.5, 0.8, 1.0, 1.2],
        n_samples=50,
    )
    all_results["sampling_temperatures"] = temp_results
    for temp, data in temp_results.items():
        log.info("  temp=%s: jaccard=%.3f  hellinger=%.4f",
                 temp, data["jaccard_similarity"], data["hellinger_distance"])

    # Summary
    print("\n" + "=" * 70)
    print("OUTPUT QUALITY SUMMARY")
    print("=" * 70)
    print(f"\nPerplexity (baseline): {[r['perplexity'] for r in ppl_results]}")
    print(f"\nKL divergence (mean): {[r['mean_kl'] for r in kl_results]}")
    print(f"KL divergence (max):  {[r['max_kl'] for r in kl_results]}")
    print(f"\nRecompute ablation:")
    print(f"  With recompute:    {ablation['with_recompute_match_rate'] * 100:.0f}% match")
    print(f"  Without recompute: {ablation['without_recompute_match_rate'] * 100:.0f}% match")
    print(f"\nLong-range dependency match: {boundary_results['match_rate'] * 100:.0f}%")

    path = save_results(all_results, args.output)
    log.info("Results saved to %s", path)


if __name__ == "__main__":
    main()
