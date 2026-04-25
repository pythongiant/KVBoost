#!/usr/bin/env python3
"""
Accuracy Benchmark: KVBoost vs vLLM (prefix-caching) vs Baseline

Tests exact match accuracy on LongBench long-context QA tasks.
"""

import json
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# LongBench tasks best suited for QA accuracy evaluation
LONGBENCH_TASKS = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa"]


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


def _load_longbench_samples(
    n_samples: int,
    max_context_tokens: int,
    tasks: List[str] = LONGBENCH_TASKS,
) -> List[Dict]:
    """
    Load samples from LongBench via HuggingFace datasets.

    Each returned dict has keys: task, context, input (question), answers (list[str]).
    Samples are filtered to fit within max_context_tokens and spread evenly across tasks.
    """
    from datasets import load_dataset

    samples = []
    per_task = max(1, n_samples // len(tasks))

    for task in tasks:
        try:
            ds = load_dataset("THUDM/LongBench", task, split="test")
        except Exception as e:
            log.warning("Could not load LongBench task %s: %s", task, e)
            continue

        count = 0
        for item in ds:
            if count >= per_task:
                break
            context = item.get("context", "")
            question = item.get("input", "")
            answers = item.get("answers", [])
            if not context or not question or not answers:
                continue
            # Rough token estimate: ~0.75 tokens per word
            approx_tokens = int(len((context + question).split()) * 0.75)
            if approx_tokens > max_context_tokens:
                continue
            samples.append({
                "task": task,
                "context": context,
                "input": question,
                "answers": answers,
                "approx_tokens": approx_tokens,
            })
            count += 1

    # Trim/pad to exactly n_samples
    return samples[:n_samples]


def _format_prompt(context: str, question: str) -> str:
    return (
        f"Read the following passage carefully and answer the question based only on "
        f"the information provided.\n\n"
        f"Passage:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def _extract_answer(raw_output: str) -> str:
    """Strip prompt echo and trailing whitespace from generated text."""
    # vLLM and transformers return only the completion; KVBoost also returns completion only
    answer = raw_output.strip()
    # Truncate at the first newline to avoid multi-sentence drift
    if "\n" in answer:
        answer = answer[: answer.index("\n")].strip()
    return answer


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

def _run_kvboost(prompts: List[str], model: str, max_new_tokens: int = 64) -> List[str]:
    from kvboost import KVBoost, GenerationMode
    import torch

    engine = KVBoost.from_pretrained(
        model_name=model,
        max_cache_bytes=4_000_000_000,
        chunk_size=128,
        recompute_overlap=16,
    )
    outputs = []
    for prompt in prompts:
        result = engine.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            mode=GenerationMode.CHUNK_KV_REUSE,
            do_sample=False,
        )
        outputs.append(result.output_text)
    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return outputs


def _run_vllm_prefixcache(prompts: List[str], model: str, max_new_tokens: int = 64) -> List[str]:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, enable_prefix_caching=True)
    params = SamplingParams(temperature=0, max_tokens=max_new_tokens)
    outputs = llm.generate(prompts, params)
    del llm
    return [o.outputs[0].text for o in outputs]


def _run_baseline(prompts: List[str], model: str, max_new_tokens: int = 64) -> List[str]:
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
        for prompt in prompts:
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


_BACKEND_RUNNERS = {
    "kvboost": _run_kvboost,
    "vllm_prefixcache": _run_vllm_prefixcache,
    "baseline": _run_baseline,
}


def benchmark_accuracy(
    backend: str,
    model: str,
    n_samples: int = 50,
    dataset_path: Optional[Path] = None,
    max_context_tokens: int = 8192,
) -> List[AccuracyResult]:
    """
    Run accuracy benchmark for a specific backend.

    Args:
        backend: 'kvboost', 'vllm_prefixcache', or 'baseline'
        model: Model name (e.g., 'Qwen/Qwen2.5-3B')
        n_samples: Number of test samples
        dataset_path: Unused; kept for API compatibility
        max_context_tokens: Max context length to test

    Returns:
        List of AccuracyResult objects
    """
    log.info("Starting accuracy benchmark for %s with %s", backend, model)

    samples = _load_longbench_samples(n_samples, max_context_tokens)
    if not samples:
        log.warning("No LongBench samples loaded — check dataset availability")
        return []

    prompts = [_format_prompt(s["context"], s["input"]) for s in samples]

    runner = _BACKEND_RUNNERS.get(backend)
    if runner is None:
        raise ValueError(f"Unknown backend: {backend!r}")

    log.info("Running inference on %d samples with %s...", len(prompts), backend)
    raw_outputs = runner(prompts, model)

    results = []
    for i, (sample, raw) in enumerate(zip(samples, raw_outputs)):
        predicted = _extract_answer(raw)
        gold_answers = sample["answers"]
        is_exact, is_f1, f1_score = _score(gold_answers, predicted)

        results.append(AccuracyResult(
            sample_id=f"{sample['task']}_{i:04d}",
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
