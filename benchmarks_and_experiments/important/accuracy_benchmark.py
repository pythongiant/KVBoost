#!/usr/bin/env python3
"""
Accuracy Benchmark: KVBoost vs vLLM (prefix-caching) vs Baseline

Tests exact match accuracy on LongBench long-context QA tasks.
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

DATASET_REPO = "JetBrains-Research/lca-bug-localization"
DATASET_CONFIG = "py"

_SAMPLE_CACHE: Dict[tuple, List[Dict]] = {}


@dataclass
class AccuracyResult:
    """Accuracy test result for one sample"""
    sample_id: str
    backend: str
    model: str
    task_name: str
    context_length: int
    answer_length: int
    gold_answer: str
    predicted: str
    is_exact_match: bool
    is_f1_match: bool
    f1_score: float


def compute_f1(gold: str, pred: str) -> float:
    """Compute F1 score between gold and predicted answer"""
    gold_tokens = set(gold.lower().split())
    pred_tokens = set(pred.lower().split())

    if not gold_tokens or not pred_tokens:
        return 1.0 if gold == pred else 0.0

    common = gold_tokens & pred_tokens
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)

    return f1


LETTERS = ["A", "B", "C", "D"]


def _load_longbench_samples(
    n_samples: int,
    max_context_tokens: int,
    model_name: str = "Qwen/Qwen2.5-3B",
) -> List[Dict]:
    """
    Load bug-localization samples from JetBrains-Research/lca-bug-localization.

    Two-pass loading:
      Pass 1 — collect all_files from entire dataset + filter by token length.
      Pass 2 — build 4-choice MC questions using global distractor pool so every
               sample has 3 valid distractors even when only 1 file was changed.

    Paired queries (q1/q2) share the same diff but use different distractors and
    question phrasing. q2's correct answer is guaranteed in a different slot than
    q1's to eliminate warm-cache positional bias.
    """
    cache_key = (n_samples, max_context_tokens, model_name)
    if cache_key in _SAMPLE_CACHE:
        return _SAMPLE_CACHE[cache_key]

    from datasets import load_dataset
    from transformers import AutoTokenizer

    log.info("Loading %s ...", DATASET_REPO)
    try:
        ds = load_dataset(DATASET_REPO, DATASET_CONFIG, split="train")
    except Exception as e:
        log.error("Could not load %s: %s", DATASET_REPO, e)
        return []

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Pass 1: filter by token length, accumulate global file pool
    raw_samples = []
    all_files: set = set()
    skipped_long = 0
    skipped_nodiff = 0

    for row in ds:
        diff = row.get("diff", "")
        if not diff or len(diff.strip()) < 10:
            skipped_nodiff += 1
            continue

        ctx_tokens = len(tok.encode(diff))
        if ctx_tokens > max_context_tokens:
            skipped_long += 1
            continue

        changed_files_raw = row.get("changed_files", "")
        if isinstance(changed_files_raw, str):
            changed_files = [f.strip() for f in changed_files_raw.replace(",", "\n").split("\n") if f.strip()]
        elif isinstance(changed_files_raw, list):
            changed_files = changed_files_raw
        else:
            changed_files = []

        if not changed_files:
            skipped_nodiff += 1
            continue

        all_files.update(changed_files)
        issue_title = (row.get("issue_title", "") or "").strip()
        issue_body = (row.get("issue_body", "") or "").strip()
        question = issue_title
        if issue_body:
            body_preview = issue_body[:1000] + ("..." if len(issue_body) > 1000 else "")
            question = f"{issue_title}\n{body_preview}"

        raw_samples.append({
            "diff": diff,
            "correct_file": changed_files[0],
            "all_changed_files": changed_files,
            "question": question,
            "ctx_tokens": ctx_tokens,
            "repo": f"{row.get('repo_owner', '')}/{row.get('repo_name', '')}",
            "changed_files_count": row.get("changed_files_count", len(changed_files)),
        })

    log.info("After filtering: %d eligible, %d too long, %d no diff/files",
             len(raw_samples), skipped_long, skipped_nodiff)

    if not raw_samples:
        log.error("No samples fit within %d tokens!", max_context_tokens)
        return []

    all_files_list = sorted(all_files)

    # Stratified sampling by token-length bucket
    rng = np.random.RandomState(42)

    def _bucket(t: int) -> str:
        if t < 512: return "0-512"
        elif t < 1024: return "512-1K"
        elif t < 2048: return "1K-2K"
        elif t < 4096: return "2K-4K"
        else: return "4K+"

    bucket_pools: Dict[str, list] = defaultdict(list)
    for s in raw_samples:
        bucket_pools[_bucket(s["ctx_tokens"])].append(s)
    for bk in bucket_pools:
        rng.shuffle(bucket_pools[bk])

    n_buckets = len(bucket_pools)
    per_bucket = n_samples // n_buckets if n_buckets else n_samples
    selected = []
    overflow = []
    for bk in ["0-512", "512-1K", "1K-2K", "2K-4K", "4K+"]:
        pool = bucket_pools.get(bk, [])
        take = min(per_bucket, len(pool))
        selected.extend(pool[:take])
        overflow.extend(pool[take:])
    remaining = n_samples - len(selected)
    if remaining > 0 and overflow:
        rng.shuffle(overflow)
        selected.extend(overflow[:remaining])

    log.info("Stratified sampling: %d samples across %d buckets", len(selected), n_buckets)

    # Pass 2: build paired MC samples with global distractors
    samples = []
    for idx, raw in enumerate(selected):
        correct_file = raw["correct_file"]
        distractor_pool = [f for f in all_files_list if f not in raw["all_changed_files"]]
        rng.shuffle(distractor_pool)
        while len(distractor_pool) < 6:
            distractor_pool.append(f"src/fake_module_{len(distractor_pool)}.py")

        base_info = {
            "task": "bug_localization",
            "context": raw["diff"],
            "ctx_tokens": raw["ctx_tokens"],
            "repo": raw["repo"],
        }

        # q1 (cold)
        d1 = distractor_pool[:3]
        choices_1 = d1 + [correct_file]
        rng.shuffle(choices_1)
        q1_pos = choices_1.index(correct_file)

        samples.append({
            **base_info,
            "id": f"bug_{idx}_q1",
            "input": f"Bug report: {raw['question']}\n\nWhich file was modified to fix this bug?",
            "choices": choices_1,
            "answers": [LETTERS[q1_pos], correct_file],
            "correct_idx": q1_pos,
            "approx_tokens": raw["ctx_tokens"],
            "pair_group": idx,
        })

        # q2 (warm) — force different answer slot than q1
        d2 = distractor_pool[3:6]
        choices_2 = d2 + [correct_file]
        rng.shuffle(choices_2)
        for _ in range(20):
            if choices_2.index(correct_file) != q1_pos:
                break
            rng.shuffle(choices_2)
        else:
            pos = choices_2.index(correct_file)
            target = (q1_pos + 1) % 4
            choices_2[pos], choices_2[target] = choices_2[target], choices_2[pos]
        q2_pos = choices_2.index(correct_file)

        samples.append({
            **base_info,
            "id": f"bug_{idx}_q2",
            "input": f"Issue: {raw['question']}\n\nIdentify the file that contains the fix for this issue.",
            "choices": choices_2,
            "answers": [LETTERS[q2_pos], correct_file],
            "correct_idx": q2_pos,
            "approx_tokens": raw["ctx_tokens"],
            "pair_group": idx,
        })

    log.info("Loaded %d paired samples from %s", len(samples), DATASET_REPO)
    _SAMPLE_CACHE[cache_key] = samples
    return samples


def _format_prompt_prefix(context: str) -> str:
    return (
        "Read the following code diff and answer the question.\n\n"
        f"--- BEGIN DIFF ---\n{context}\n--- END DIFF ---\n\n"
    )


def _format_prompt_suffix(question: str, choices: List[str]) -> str:
    parts = [f"{question}\n"]
    for i, choice in enumerate(choices):
        parts.append(f"{LETTERS[i]}. {choice}")
    parts.append("\nAnswer with just the letter (A, B, C, or D):")
    return "\n".join(parts)


def _format_prompt(context: str, question: str, choices: Optional[List[str]] = None) -> str:
    if choices:
        return _format_prompt_prefix(context) + _format_prompt_suffix(question, choices)
    return (
        "Read the following code diff and answer the question.\n\n"
        f"--- BEGIN DIFF ---\n{context}\n--- END DIFF ---\n\n"
        f"{question}\n\nAnswer:"
    )


import re as _re


def _extract_answer(raw_output: str) -> str:
    """Extract A/B/C/D letter from model output."""
    text = raw_output.strip()
    m = _re.match(r"^\(?([A-Da-d])\)?[\.\)\s:]?", text)
    if m:
        return m.group(1).upper()
    for ch in text:
        if ch.upper() in LETTERS:
            return ch.upper()
    if "\n" in text:
        text = text[: text.index("\n")].strip()
    return text


def _score(gold_answers: List[str], predicted: str) -> Tuple[bool, bool, float]:
    """Return (exact_match, f1_match, best_f1) across all gold answers."""
    pred_norm = predicted.lower().strip()
    best_f1 = 0.0
    exact = False
    for gold in gold_answers:
        gold_norm = gold.lower().strip()
        if pred_norm == gold_norm:
            exact = True
        f1 = compute_f1(gold_norm, pred_norm)
        if f1 > best_f1:
            best_f1 = f1
    f1_match = best_f1 >= 0.5
    return exact, f1_match, best_f1


# ---------------------------------------------------------------------------
# Backend inference helpers
# ---------------------------------------------------------------------------

def _run_kvboost(
    samples: List[Dict],
    model: str,
    max_new_tokens: int = 64,
    recompute_strategy: str = "selective",
    chunk_boundary_window: int = 0,
    overlap_k: int = 0,
    sink_tokens: int = 0,
) -> List[str]:
    from kvboost import KVBoost, GenerationMode
    import torch

    engine = KVBoost.from_pretrained(
        model_name=model,
        max_cache_bytes=4_000_000_000,
        chunk_size=128,
        recompute_overlap=16,
        recompute_strategy=recompute_strategy,
        chunk_boundary_window=chunk_boundary_window,
        overlap_k=overlap_k,
        sink_tokens=sink_tokens,
    )
    outputs = []
    for sample in samples:
        prefix = _format_prompt_prefix(sample["context"])
        suffix = _format_prompt_suffix(sample["input"], sample["choices"])
        prompt = prefix + suffix
        cacheable_prefix_len = len(engine.tokenizer.encode(prefix, add_special_tokens=True))
        result = engine.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            mode=GenerationMode.CHUNK_KV_REUSE,
            do_sample=False,
            cacheable_prefix_len=cacheable_prefix_len,
        )
        outputs.append(result.output_text)
    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def _run_vllm_prefixcache(samples: List[Dict], model: str, max_new_tokens: int = 64) -> List[str]:
    from vllm import LLM, SamplingParams

    prompts = [_format_prompt(s["context"], s["input"], s.get("choices")) for s in samples]
    llm = LLM(model=model, enable_prefix_caching=True)
    params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, params)
    del llm
    return [o.outputs[0].text for o in outputs]


def _run_baseline(samples: List[Dict], model: str, max_new_tokens: int = 64) -> List[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model, torch_dtype=torch.float16 if device == "cuda" else torch.float32
    ).to(device)
    hf_model.eval()

    outputs = []
    with torch.no_grad():
        for sample in samples:
            prompt = _format_prompt(sample["context"], sample["input"], sample.get("choices"))
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            prompt_len = inputs["input_ids"].shape[1]
            gen_ids = hf_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_ids = gen_ids[0][prompt_len:]
            outputs.append(tokenizer.decode(new_ids, skip_special_tokens=True))

    del hf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs




def benchmark_accuracy(
    backend: str,
    model: str,
    n_samples: int = 50,
    dataset_path: Optional[Path] = None,
    max_context_tokens: int = 8192,
    kvboost_recompute_strategy: str = "selective",
    kvboost_chunk_boundary_window: int = 0,
    kvboost_overlap_k: int = 0,
    kvboost_sink_tokens: int = 0,
) -> List[AccuracyResult]:
    """
    Run accuracy benchmark for a specific backend.

    Args:
        backend: 'kvboost', 'vllm_prefixcache', or 'baseline'
        model: Model name (e.g., 'Qwen/Qwen2.5-3B')
        n_samples: Number of test samples
        dataset_path: Unused; kept for API compatibility
        max_context_tokens: Max context length to test
        kvboost_recompute_strategy: KVBoost recompute strategy ('selective', 'cacheblend', 'none')
        kvboost_chunk_boundary_window: Adaptive boundary splitting window
        kvboost_overlap_k: Overlapping chunk encoding tokens
        kvboost_sink_tokens: Attention sink prefix tokens

    Returns:
        List of AccuracyResult objects
    """
    log.info("Starting accuracy benchmark for %s with %s", backend, model)

    samples = _load_longbench_samples(n_samples, max_context_tokens, model_name=model)
    if not samples:
        log.warning("No samples loaded — check dataset availability")
        return []

    log.info("Running inference on %d samples with %s...", len(samples), backend)
    if backend == "kvboost":
        raw_outputs = _run_kvboost(
            samples, model,
            recompute_strategy=kvboost_recompute_strategy,
            chunk_boundary_window=kvboost_chunk_boundary_window,
            overlap_k=kvboost_overlap_k,
            sink_tokens=kvboost_sink_tokens,
        )
    elif backend == "vllm_prefixcache":
        raw_outputs = _run_vllm_prefixcache(samples, model)
    elif backend == "baseline":
        raw_outputs = _run_baseline(samples, model)
    else:
        raise ValueError(f"Unknown backend: {backend!r}")

    results = []
    for i, (sample, raw) in enumerate(zip(samples, raw_outputs)):
        predicted = _extract_answer(raw)
        gold_answers = sample["answers"]
        is_exact, is_f1, f1_score = _score(gold_answers, predicted)

        results.append(AccuracyResult(
            sample_id=sample.get("id", f"{sample['task']}_{i:04d}"),
            backend=backend,
            model=model,
            task_name=sample["task"],
            context_length=sample["approx_tokens"],
            answer_length=len(predicted.split()),
            gold_answer=gold_answers[0] if gold_answers else "",
            predicted=predicted,
            is_exact_match=is_exact,
            is_f1_match=is_f1,
            f1_score=f1_score,
        ))

    log.info(
        "Processed %d accuracy samples for %s (exact=%.1f%%, F1=%.3f)",
        len(results),
        backend,
        100 * sum(r.is_exact_match for r in results) / max(len(results), 1),
        np.mean([r.f1_score for r in results]) if results else 0.0,
    )
    return results


def aggregate_accuracy_results(
    results: Dict[str, List[AccuracyResult]]
) -> Dict[str, Dict]:
    """Aggregate accuracy results by backend"""
    aggregated = {}

    for backend, backend_results in results.items():
        if not backend_results:
            continue

        exact_matches = sum(1 for r in backend_results if r.is_exact_match)
        f1_matches = sum(1 for r in backend_results if r.is_f1_match)
        avg_f1 = np.mean([r.f1_score for r in backend_results])

        aggregated[backend] = {
            "n_samples": len(backend_results),
            "exact_match_accuracy": exact_matches / len(backend_results),
            "f1_match_accuracy": f1_matches / len(backend_results),
            "avg_f1_score": float(avg_f1),
            "samples": [asdict(r) for r in backend_results],
        }

    return aggregated


def print_accuracy_table(aggregated: Dict[str, Dict]):
    """Print accuracy comparison table"""
    print("\n" + "="*90)
    print("  ACCURACY BENCHMARK RESULTS")
    print("="*90)
    print(
        f"  {'Backend':<20} {'N Samples':>10} {'Exact Match':>15} "
        f"{'F1 Match':>12} {'Avg F1':>12}"
    )
    print(f"  {'-'*20} {'-'*10} {'-'*15} {'-'*12} {'-'*12}")

    baseline_exact = None
    for backend in ['kvboost', 'vllm_prefixcache', 'baseline']:
        if backend not in aggregated:
            continue

        agg = aggregated[backend]
        exact = agg["exact_match_accuracy"]
        f1 = agg["f1_match_accuracy"]
        avg_f1 = agg["avg_f1_score"]

        if backend == 'baseline':
            baseline_exact = exact

        delta = (exact - baseline_exact) * 100 if baseline_exact is not None and backend != 'baseline' else 0
        marker = f" {delta:+.1f}%" if backend != 'baseline' else " <<<BASELINE"

        print(
            f"  {backend:<20} {agg['n_samples']:>10} {exact:>14.1%} "
            f"{f1:>11.1%} {avg_f1:>11.3f}{marker}"
        )

    print("="*90 + "\n")


def save_accuracy_results(
    aggregated: Dict[str, Dict],
    model: str,
    output_path: Optional[Path] = None
) -> Path:
    """Save accuracy results to JSON"""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"accuracy_benchmark_{model.replace('/', '_')}_{timestamp}.json"

    output_path.parent.mkdir(exist_ok=True, parents=True)

    output_data = {
        "timestamp": datetime.now().isoformat(),
        "benchmark": "accuracy",
        "model": model,
        "results": aggregated,
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2, default=str)

    log.info("Accuracy results saved to %s", output_path)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Accuracy Benchmark")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--output", type=Path, default=None)

    args = parser.parse_args()

    results = {}
    for backend in ['kvboost', 'vllm_prefixcache', 'baseline']:
        results[backend] = benchmark_accuracy(backend, args.model, args.n_samples)

    aggregated = aggregate_accuracy_results(results)
    print_accuracy_table(aggregated)
    save_accuracy_results(aggregated, args.model, args.output)
