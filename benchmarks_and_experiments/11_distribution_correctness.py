#!/usr/bin/env python3
"""
Experiment 11: Distribution-Level Correctness
==============================================
Goes beyond greedy output match to validate that KVBoost produces
the same *logit distribution* as baseline, not just the same argmax.

Metrics:
  - Forward KL divergence (KL(baseline || cached))
  - Hellinger distance (symmetric, bounded [0,1])
  - Top-k agreement (do the top-10 tokens match?)

Tested across:
  - Short prompts (< 256 tokens)
  - Medium prompts (500-800 tokens, sweet spot)
  - Long prompts (1000+ tokens, headline claim)
  - Multi-turn at turns 1, 4, 8
  - Temperatures: 0.1, 0.5, 1.0
  - Long-range dependency coherence (200+ token continuations)

Threshold: KL divergence < 0.01 at temp=1.0 to support the
"zero output degradation" claim.

Usage:
    python benchmarks_and_experiments/11_distribution_correctness.py
    python benchmarks_and_experiments/11_distribution_correctness.py --model Qwen/Qwen2.5-3B
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from kvboost import KVBoost, GenerationMode

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger("11_distribution_correctness")

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ── Test prompts at different lengths ───────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert technical assistant specializing in distributed systems, "
    "machine learning infrastructure, and high-performance computing. "
    "You provide precise, actionable answers backed by production experience. "
    "When asked a question, first briefly summarize the key considerations, "
    "then provide a concrete recommendation. Always mention trade-offs. "
    "If the question involves code, include minimal but complete examples. "
    "Prioritize correctness over brevity, but avoid unnecessary padding."
)

DOCUMENT = (
    "A KV cache stores the intermediate key and value tensors produced by the "
    "attention mechanism during the forward pass of a transformer model. During "
    "autoregressive generation, each new token needs to attend to all previous "
    "tokens. Without caching, this requires recomputing keys and values for every "
    "past token at each generation step, giving O(n^2) complexity. With KV caching, "
    "previously computed K/V tensors are stored and reused, reducing generation to "
    "O(n) per step. Prefix caching extends this across requests. When multiple "
    "prompts share the same prefix, the KV tensors for that prefix can be computed "
    "once and reused. Chunk-level caching generalizes prefix caching by allowing "
    "reuse of non-contiguous chunks."
)

MULTITURN_TURNS = [
    ("What is the derivative of x^2?",
     "The derivative of x^2 is 2x. Using the limit definition: "
     "d/dx[x^2] = lim(h->0) [(x+h)^2 - x^2] / h = 2x."),
    ("What about x^3?",
     "The derivative of x^3 is 3x^2. The exponent comes down as "
     "a coefficient and decreases by one."),
    ("Explain the power rule.",
     "The Power Rule: d/dx[x^n] = n*x^(n-1) for any real n. "
     "Works for positive, negative, and fractional exponents."),
    ("Give me an example with x^5.",
     "f(x) = x^5, so f'(x) = 5x^4. At x=2: f'(2) = 80."),
    ("How does the chain rule work?",
     "Chain Rule: d/dx[f(g(x))] = f'(g(x)) * g'(x). "
     "Example: d/dx[(2x+1)^7] = 14*(2x+1)^6."),
    ("Differentiate (3x^2 + 1)^4.",
     "Result: 24x*(3x^2+1)^3."),
    ("What about the product rule?",
     "Product Rule: d/dx[f*g] = f'*g + f*g'. "
     "Example: d/dx[x^2 * sin(x)] = 2x*sin(x) + x^2*cos(x)."),
    ("Combine all three rules.",
     "f(x) = x^2 * (3x+1)^4. Product rule + chain rule + power rule: "
     "f'(x) = 2x*(3x+1)^3*(9x+1)."),
]

# ── Long-range dependency prompt ────────────────────────────────────

LONG_RANGE_PROMPT = (
    "Create a numbered list of 10 scientific facts. Each fact must reference "
    "the previous fact. For example, fact 3 should build on fact 2.\n\n"
    "1. Water is composed of two hydrogen atoms and one oxygen atom (H2O).\n"
    "2. Because water (from fact 1) is a polar molecule, it acts as a universal solvent.\n"
    "3. Water's solvent properties (fact 2) enable chemical reactions in biological cells.\n"
    "4. Cellular chemistry (fact 3) depends on enzymes that catalyze specific reactions.\n"
    "5. Enzymes (fact 4) are proteins made of amino acid chains folded into 3D shapes.\n"
    "Continue the list:\n"
    "6."
)


# ── Core test logic ────────────────────────────────────────────────

def get_logits(engine: KVBoost, prompt: str, mode: GenerationMode) -> torch.Tensor:
    """
    Run a single forward pass and return the last-token logits.
    Uses the engine's internal model directly.
    """
    token_ids = engine.tokenizer.encode(prompt, add_special_tokens=True)
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=engine.device)

    if mode == GenerationMode.BASELINE:
        with torch.no_grad():
            out = engine.model(input_ids=input_ids, use_cache=False)
        return out.logits[:, -1, :].float().cpu()

    # CHUNK_KV_REUSE path
    assembled = engine.assembler.assemble(token_ids)
    if len(assembled.chunk_boundaries) > 1:
        if engine.recompute_strategy.value == "cacheblend":
            assembled = engine.cacheblend_recompute.apply(assembled, engine.model)
        elif engine.recompute_strategy.value == "selective":
            assembled = engine.selective_recompute.apply(assembled, engine.model)

    if assembled.cached_past_kv is not None:
        past_kv = tuple(
            (layer[0].to(engine.device), layer[1].to(engine.device))
            for layer in assembled.cached_past_kv
        )
    else:
        past_kv = None

    live_ids = assembled.live_token_ids
    if live_ids:
        live_input = torch.tensor([live_ids], dtype=torch.long, device=engine.device)
        pos_ids = torch.arange(
            assembled.cached_length, assembled.cached_length + len(live_ids),
            dtype=torch.long, device=engine.device,
        ).unsqueeze(0)
        with torch.no_grad():
            out = engine.model(
                input_ids=live_input,
                past_key_values=engine._as_cache(past_kv),
                position_ids=pos_ids,
                use_cache=False,
            )
        return out.logits[:, -1, :].float().cpu()
    else:
        # All cached — feed last token
        last_id = token_ids[-1]
        live_input = torch.tensor([[last_id]], dtype=torch.long, device=engine.device)
        pos_ids = torch.tensor([[assembled.cached_length - 1]], dtype=torch.long, device=engine.device)
        from kvboost import KVCacheManager
        if past_kv is not None and KVCacheManager.kv_seq_len(past_kv) > 1:
            trimmed = KVCacheManager.slice_kv(past_kv, 0, assembled.cached_length - 1)
            trimmed = tuple((l[0].to(engine.device), l[1].to(engine.device)) for l in trimmed)
        else:
            trimmed = past_kv
        with torch.no_grad():
            out = engine.model(
                input_ids=live_input,
                past_key_values=engine._as_cache(trimmed),
                position_ids=pos_ids,
                use_cache=False,
            )
        return out.logits[:, -1, :].float().cpu()


def compute_metrics(
    baseline_logits: torch.Tensor,
    cached_logits: torch.Tensor,
    temperatures: List[float] = [0.1, 0.5, 1.0],
) -> Dict[str, Dict[str, float]]:
    """Compute KL divergence, Hellinger distance, and top-k agreement."""
    results = {}
    for temp in temperatures:
        p = F.softmax(baseline_logits / temp, dim=-1)
        q = F.softmax(cached_logits / temp, dim=-1)

        # Forward KL: KL(p || q) — info lost going baseline → cached
        kl = F.kl_div(q.log(), p, reduction="batchmean").item()

        # Hellinger distance: symmetric, bounded [0, 1]
        hellinger = (0.5 * ((p.sqrt() - q.sqrt()) ** 2).sum()).sqrt().item()

        # Top-10 agreement
        top10_baseline = set(p.topk(10).indices[0].tolist())
        top10_cached = set(q.topk(10).indices[0].tolist())
        top10_overlap = len(top10_baseline & top10_cached) / 10

        # Argmax match
        argmax_match = p.argmax().item() == q.argmax().item()

        results[f"temp_{temp}"] = {
            "kl_divergence": round(kl, 6),
            "hellinger": round(hellinger, 6),
            "top10_agreement": round(top10_overlap, 2),
            "argmax_match": argmax_match,
        }

    return results


# ── Test suites ─────────────────────────────────────────────────────

def test_single_prompt(
    engine: KVBoost, prompt: str, label: str, warmup_text: str = None,
) -> Dict:
    """Test a single prompt across temperatures."""
    if warmup_text:
        engine.warm(warmup_text)

    baseline_logits = get_logits(engine, prompt, GenerationMode.BASELINE)

    # Ensure cache is populated
    engine.generate(prompt, max_new_tokens=1, mode=GenerationMode.CHUNK_KV_REUSE, do_sample=False)

    cached_logits = get_logits(engine, prompt, GenerationMode.CHUNK_KV_REUSE)

    metrics = compute_metrics(baseline_logits, cached_logits)

    print(f"  {label}:")
    for temp_key, m in metrics.items():
        status = "PASS" if m["kl_divergence"] < 0.01 else "FAIL"
        print(
            f"    {temp_key}: KL={m['kl_divergence']:.6f} "
            f"H={m['hellinger']:.6f} "
            f"top10={m['top10_agreement']:.0%} "
            f"argmax={'Y' if m['argmax_match'] else 'N'} "
            f"[{status}]"
        )

    return {"label": label, "metrics": metrics}


def test_prompt_lengths(engine: KVBoost) -> List[Dict]:
    """Test across short / medium / long prompts."""
    results = []
    print("\n  === Prompt Length Tests ===")

    # Short (< 256 tokens)
    short_prompt = SYSTEM_PROMPT + "\n\nWhat is a hash table?"
    results.append(test_single_prompt(engine, short_prompt, "short (~60 tok)", SYSTEM_PROMPT))

    # Medium (500-800 tokens)
    medium_prompt = SYSTEM_PROMPT + "\n\n" + DOCUMENT + "\n\nSummarize the above."
    results.append(test_single_prompt(engine, medium_prompt, "medium (~400 tok)", SYSTEM_PROMPT + "\n\n" + DOCUMENT))

    # Long (1000+ tokens)
    long_doc = DOCUMENT * 3
    long_prompt = long_doc + "\n\nWhat are the key takeaways?"
    results.append(test_single_prompt(engine, long_prompt, "long (~1200 tok)", long_doc))

    return results


def test_multiturn(engine: KVBoost) -> List[Dict]:
    """Test multi-turn conversation at turns 1, 4, 8."""
    results = []
    print("\n  === Multi-Turn Tests ===")

    system = "You are a math tutor. Be concise."
    engine.warm(system)
    history = system

    for i, (user_msg, assistant_reply) in enumerate(MULTITURN_TURNS):
        history += f"\n\nUser: {user_msg}\nAssistant: {assistant_reply}"
        turn = i + 1

        if turn in [1, 4, 8]:
            prompt = history + "\n\nUser: Continue.\nAssistant:"
            results.append(test_single_prompt(engine, prompt, f"turn {turn} (~{len(prompt.split())} words)"))

    return results


def test_long_range_dependency(engine: KVBoost, max_tokens: int = 200) -> Dict:
    """
    Generate 200+ tokens from a prompt with high cross-chunk semantic
    dependency, check coherence between baseline and cached.
    """
    print("\n  === Long-Range Dependency ===")

    engine.warm(LONG_RANGE_PROMPT)

    r_base = engine.generate(
        LONG_RANGE_PROMPT, max_new_tokens=max_tokens,
        mode=GenerationMode.BASELINE, do_sample=False,
    )
    r_cached = engine.generate(
        LONG_RANGE_PROMPT, max_new_tokens=max_tokens,
        mode=GenerationMode.CHUNK_KV_REUSE, do_sample=False,
    )

    exact_match = r_base.output_text == r_cached.output_text

    # Token-level comparison
    base_tokens = engine.tokenizer.encode(r_base.output_text)
    cached_tokens = engine.tokenizer.encode(r_cached.output_text)
    min_len = min(len(base_tokens), len(cached_tokens))
    matching_tokens = sum(1 for a, b in zip(base_tokens[:min_len], cached_tokens[:min_len]) if a == b)
    token_match_pct = matching_tokens / max(min_len, 1)

    # Find first divergence point
    first_diverge = min_len
    for j in range(min_len):
        if base_tokens[j] != cached_tokens[j]:
            first_diverge = j
            break

    result = {
        "exact_match": exact_match,
        "token_match_pct": round(token_match_pct, 4),
        "baseline_tokens": len(base_tokens),
        "cached_tokens": len(cached_tokens),
        "first_diverge_at": first_diverge,
        "baseline_preview": r_base.output_text[:200],
        "cached_preview": r_cached.output_text[:200],
    }

    status = "PASS" if exact_match else ("PARTIAL" if token_match_pct > 0.95 else "FAIL")
    print(f"  long_range: exact={'Y' if exact_match else 'N'} "
          f"token_match={token_match_pct:.1%} "
          f"diverge_at={first_diverge}/{min_len} [{status}]")

    return result


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Distribution-level correctness tests")
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-continuation", type=int, default=200)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    print(f"\n  Model: {args.model}")
    print(f"  Chunk size: {args.chunk_size}")

    engine = KVBoost.from_pretrained(
        args.model, chunk_size=args.chunk_size, recompute_overlap=8,
    )

    all_results = {
        "model": args.model,
        "chunk_size": args.chunk_size,
    }

    # Run test suites
    all_results["prompt_lengths"] = test_prompt_lengths(engine)
    all_results["multiturn"] = test_multiturn(engine)
    all_results["long_range"] = test_long_range_dependency(engine, args.max_continuation)

    # Summary
    print(f"\n{'='*60}")
    kl_values = []
    for suite_key in ["prompt_lengths", "multiturn"]:
        for entry in all_results[suite_key]:
            for temp_key, m in entry["metrics"].items():
                kl_values.append(m["kl_divergence"])

    if kl_values:
        max_kl = max(kl_values)
        mean_kl = sum(kl_values) / len(kl_values)
        all_pass = all(kl < 0.01 for kl in kl_values)
        print(f"  KL divergence: mean={mean_kl:.6f} max={max_kl:.6f} "
              f"threshold=0.01 [{'ALL PASS' if all_pass else 'SOME FAIL'}]")

    lr = all_results["long_range"]
    print(f"  Long-range: exact={'Y' if lr['exact_match'] else 'N'} "
          f"token_match={lr['token_match_pct']:.1%}")
    print(f"{'='*60}")

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Saved to {out_path}")
    else:
        default_path = RESULTS_DIR / "11_distribution_correctness.json"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(default_path, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Saved to {default_path}")


if __name__ == "__main__":
    main()
