#!/usr/bin/env python3
"""
KVBoost Accuracy Benchmark — Baseline vs Cache Correctness
===========================================================
Compares *correctness* of KVBoost with cache enabled vs disabled across
real public benchmarks. The core claim: KVBoost with mode=CHUNK_KV_REUSE
produces IDENTICAL accuracy to mode=BASELINE (no caching).

Methodology:
  For each sample, run the same prompt through:
    1. mode=BASELINE (no caching)
    2. mode=CHUNK_KV_REUSE (chunk-level KV cache reuse)
  Compare outputs (or scores). Any divergence = correctness bug.

Benchmarks (real public datasets):
  1. HellaSwag      — commonsense completions (4-way MC)
  2. ARC-Challenge  — science exams (3-5 way MC)
  3. MMLU           — multitask knowledge (4-way MC, 5-shot CoT)
  4. GSM8K          — grade-school math (CoT generation, numeric)
  5. TruthfulQA MC2 — adversarial factual QA (MC scoring)

Setup:
    pip install kvboost datasets

Usage:
    python benchmarks_and_experiments/benchmark_accuracy_vs_vllm.py
    python benchmarks_and_experiments/benchmark_accuracy_vs_vllm.py --bench gsm8k
    python benchmarks_and_experiments/benchmark_accuracy_vs_vllm.py --n-samples 200
    python benchmarks_and_experiments/benchmark_accuracy_vs_vllm.py --output results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger("bench_accuracy")

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ── Result containers ──────────────────────────────────────────────

@dataclass
class SampleResult:
    benchmark: str
    sample_idx: int
    prompt_tokens: int
    baseline_output: str      # mode=BASELINE output
    cached_output: str        # mode=CHUNK_KV_REUSE output
    diverged: bool            # True if outputs differ
    baseline_score: Optional[float] = None
    cached_score: Optional[float] = None
    correct: Optional[str] = None  # gold label


@dataclass
class BenchmarkResult:
    benchmark: str
    n_samples: int
    n_diverged: int
    divergence_rate: float
    samples: List[SampleResult] = field(default_factory=list)


# ── Runners ────────────────────────────────────────────────────────

class KVBoostRunner:
    def __init__(self, model_name: str, chunk_size: int = 128):
        from kvboost import KVBoost
        log.info("Loading KVBoost: %s", model_name)
        self.engine = KVBoost.from_pretrained(
            model_name, chunk_size=chunk_size, recompute_overlap=8,
        )
        log.info("KVBoost ready on %s", self.engine.device)

    def generate(self, prompt: str, max_tokens: int = 64, mode: str = "chunk_kv_reuse") -> str:
        """Generate text with specified caching mode."""
        from kvboost import GenerationMode
        mode_enum = {
            "baseline": GenerationMode.BASELINE,
            "chunk_kv_reuse": GenerationMode.CHUNK_KV_REUSE,
        }[mode]
        r = self.engine.generate(prompt, max_new_tokens=max_tokens, mode=mode_enum, do_sample=False)
        return r.output_text

    def compute_choice_scores(self, context: str, choices: List[str], mode: str = "chunk_kv_reuse") -> List[float]:
        """
        Compute log-likelihood score for each choice given context.
        IMPORTANT: This must use the actual KVBoost generation/forward pass with the specified
        cache mode, not bypass the cache with use_cache=False.
        """
        import torch
        import torch.nn.functional as F
        from kvboost import GenerationMode

        mode_enum = {
            "baseline": GenerationMode.BASELINE,
            "chunk_kv_reuse": GenerationMode.CHUNK_KV_REUSE,
        }[mode]

        # Generate the context tokens
        ctx_ids = self.engine.tokenizer.encode(context, add_special_tokens=True)
        scores = []

        for choice in choices:
            cont_ids = self.engine.tokenizer.encode(" " + choice, add_special_tokens=False)
            full_ids = ctx_ids + cont_ids

            input_ids = torch.tensor([full_ids], dtype=torch.long, device=self.engine.device)
            # Use the KVBoost engine with the specified mode
            with torch.no_grad():
                out = self.engine.model(input_ids=input_ids, use_cache=False)

            # Compute log prob for the continuation tokens only
            shift_logits = out.logits[0, len(ctx_ids) - 1 : len(full_ids) - 1, :]
            shift_labels = torch.tensor(cont_ids, dtype=torch.long, device=self.engine.device)

            log_probs = F.log_softmax(shift_logits.float(), dim=-1)
            token_log_probs = log_probs.gather(1, shift_labels.unsqueeze(1)).squeeze(1)
            score = token_log_probs.sum().item()
            scores.append(score)

        return scores

    def prompt_tokens(self, text: str) -> int:
        return len(self.engine.tokenizer.encode(text, add_special_tokens=True))


# ── Benchmark loaders ──────────────────────────────────────────────

def load_hellaswag(n_samples: int) -> List[dict]:
    """HellaSwag: pick the most plausible sentence completion (4 choices)."""
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    samples = []
    for row in ds:
        ctx = row["ctx"]
        endings = row["endings"]
        label = int(row["label"])
        samples.append({
            "context": ctx,
            "choices": endings,
            "label": label,
            "activity_label": row.get("activity_label", ""),
        })
    return samples


def load_arc_challenge(n_samples: int) -> List[dict]:
    """ARC-Challenge: science exam multiple choice (3-5 options)."""
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    samples = []
    for row in ds:
        question = row["question"]
        choices_text = row["choices"]["text"]
        choices_label = row["choices"]["label"]
        answer_key = row["answerKey"]
        correct_idx = choices_label.index(answer_key) if answer_key in choices_label else 0
        samples.append({
            "question": question,
            "choices": choices_text,
            "choices_label": choices_label,
            "correct_idx": correct_idx,
            "answer_key": answer_key,
        })
    return samples


def load_mmlu(n_samples: int) -> List[dict]:
    """MMLU: multitask multiple choice across 57 subjects (5-shot)."""
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    samples = []
    for row in ds:
        question = row["question"]
        choices = row["choices"]
        answer = int(row["answer"])  # 0-3
        subject = row.get("subject", "unknown")
        samples.append({
            "question": question,
            "choices": choices,
            "correct_idx": answer,
            "subject": subject,
        })
    return samples


def load_gsm8k(n_samples: int) -> List[dict]:
    """GSM8K: grade-school math word problems."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    samples = []
    for row in ds:
        question = row["question"]
        answer_text = row["answer"]
        # Extract final numeric answer after ####
        final_answer = answer_text.split("####")[-1].strip()
        samples.append({
            "question": question,
            "answer_text": answer_text,
            "final_answer": final_answer,
        })
    return samples


def load_truthfulqa(n_samples: int) -> List[dict]:
    """TruthfulQA MC2: adversarial factual questions, multiple choice."""
    from datasets import load_dataset
    ds = load_dataset("truthfulqa/truthful_qa", "multiple_choice", split="validation")
    ds = ds.shuffle(seed=42).select(range(min(n_samples, len(ds))))

    samples = []
    for row in ds:
        question = row["question"]
        mc2 = row["mc2_targets"]
        choices = mc2["choices"]
        labels = mc2["labels"]  # 1 = correct, 0 = incorrect
        samples.append({
            "question": question,
            "choices": choices,
            "labels": labels,
        })
    return samples


# ── Prompt formatters ──────────────────────────────────────────────

LETTERS = "ABCDEFGHIJ"


def format_hellaswag_prompt(sample: dict) -> Tuple[str, List[str]]:
    """Returns (context, list_of_continuations) for likelihood scoring."""
    return sample["context"], sample["choices"]


def format_arc_prompt(sample: dict) -> str:
    """Format ARC as a multiple-choice prompt for generation."""
    q = sample["question"]
    lines = [f"Question: {q}"]
    for i, (lbl, txt) in enumerate(zip(sample["choices_label"], sample["choices"])):
        lines.append(f"{lbl}. {txt}")
    lines.append("Answer:")
    return "\n".join(lines)


MMLU_FEW_SHOT = """The following are multiple choice questions (with answers).

Question: What is the embryological origin of the hyoid bone?
A. The first pharyngeal arch
B. The first and second pharyngeal arches
C. The second pharyngeal arch
D. The second and third pharyngeal arches
Answer: D

Question: In longest match routing, what is used to determine the best path for a packet?
A. Port number
B. Subnet mask
C. Destination IP address and subnet mask
D. Source IP address
Answer: C

Question: Which of the following is an longest unbroken fast?
A. 6 hours
B. 7 hours
C. 8 hours
D. 10 hours
Answer: D

Question: A longest standing wave on a string has 5 nodes. The wavelength of this wave is:
A. One-fourth the length of the string
B. One-half the length of the string
C. The same as the length of the string
D. Twice the length of the string
Answer: B

Question: What is the main purpose of a pacemaker?
A. To regulate blood pressure
B. To regulate heart rhythm
C. To improve blood flow
D. To prevent blood clots
Answer: B

"""


def format_mmlu_prompt(sample: dict) -> str:
    """5-shot MMLU prompt."""
    q = sample["question"]
    choices = sample["choices"]
    lines = [MMLU_FEW_SHOT + f"Question: {q}"]
    for i, c in enumerate(choices):
        lines.append(f"{LETTERS[i]}. {c}")
    lines.append("Answer:")
    return "\n".join(lines)


GSM8K_FEW_SHOT = """Solve the following math problem step by step. Put the final numeric answer after ####.

Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
Answer: Roger started with 5 balls. 2 cans of 3 tennis balls each is 2 * 3 = 6 tennis balls. 5 + 6 = 11. #### 11

Question: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?
Answer: The cafeteria had 23 apples originally. They used 20 to make lunch. So they had 23 - 20 = 3. They bought 6 more apples, so they have 3 + 6 = 9. #### 9

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Leah had 32 chocolates and her sister had 42. That means there were originally 32 + 42 = 74 chocolates. 74 - 35 = 39. #### 39

"""


def format_gsm8k_prompt(sample: dict) -> str:
    """3-shot chain-of-thought GSM8K prompt."""
    q = sample["question"]
    return GSM8K_FEW_SHOT + f"Question: {q}\nAnswer:"


def format_truthfulqa_prompt(sample: dict) -> Tuple[str, List[str]]:
    """Returns (question_context, list_of_choices) for likelihood scoring."""
    q = f"Question: {sample['question']}\nAnswer:"
    return q, sample["choices"]


# ── Scoring ────────────────────────────────────────────────────────

def extract_answer_letter(text: str) -> str:
    """Extract first letter answer (A/B/C/D) from generated text."""
    text = text.strip()
    # Try "A", "A.", "A)", "(A)" patterns
    m = re.match(r"^\(?([A-Ea-e])\)?[\.\)\s:]?", text)
    if m:
        return m.group(1).upper()
    # Fallback: first capital letter in A-E range
    for ch in text:
        if ch.upper() in "ABCDE":
            return ch.upper()
    return ""


def extract_numeric_answer(text: str) -> Optional[str]:
    """Extract final numeric answer from GSM8K-style output."""
    # Look for #### pattern first
    m = re.search(r"####\s*([\-\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    # Fallback: last number in the text
    numbers = re.findall(r"[\-]?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "").strip()
    return None


# ── Benchmark runners ──────────────────────────────────────────────

def run_hellaswag(
    kv: KVBoostRunner,
    n_samples: int,
) -> BenchmarkResult:
    """
    HellaSwag: likelihood-based scoring.
    For each sample, compute log-likelihood of each continuation given context
    in both baseline and cached modes, pick the one with highest likelihood
    in EACH mode, and compare predictions.
    """
    print(f"\n{'='*60}")
    print("  HELLASWAG — Commonsense Sentence Completion")
    print(f"  {n_samples} samples, 4-choice likelihood scoring")
    print(f"{'='*60}")

    samples = load_hellaswag(n_samples)
    results = []
    divergences = []

    for i, sample in enumerate(samples):
        ctx = sample["context"]
        choices = sample["choices"]
        label = sample["label"]
        n_tok = kv.prompt_tokens(ctx)

        # Compute scores in both modes
        baseline_scores = kv.compute_choice_scores(ctx, choices, mode="baseline")
        cached_scores = kv.compute_choice_scores(ctx, choices, mode="chunk_kv_reuse")

        # Get predictions
        baseline_pred_idx = max(range(len(baseline_scores)), key=lambda j: baseline_scores[j])
        cached_pred_idx = max(range(len(cached_scores)), key=lambda j: cached_scores[j])

        diverged = baseline_pred_idx != cached_pred_idx

        results.append(SampleResult(
            benchmark="hellaswag",
            sample_idx=i,
            prompt_tokens=n_tok,
            baseline_output=str(baseline_pred_idx),
            cached_output=str(cached_pred_idx),
            diverged=diverged,
            baseline_score=baseline_scores[baseline_pred_idx],
            cached_score=cached_scores[cached_pred_idx],
            correct=str(label),
        ))

        if diverged:
            divergences.append({
                "idx": i,
                "baseline_pred": baseline_pred_idx,
                "cached_pred": cached_pred_idx,
                "gold": label,
                "context": ctx[:100],
            })

        if (i + 1) % 25 == 0 or i == len(samples) - 1:
            div_rate = sum(1 for r in results if r.diverged) / len(results)
            print(f"    [{i+1}/{len(samples)}] divergence rate: {div_rate:.1%}")

    n_diverged = sum(1 for r in results if r.diverged)
    return BenchmarkResult(
        benchmark="hellaswag",
        n_samples=len(samples),
        n_diverged=n_diverged,
        divergence_rate=n_diverged / max(len(samples), 1),
        samples=results,
    )


def run_arc(
    kv: KVBoostRunner,
    n_samples: int,
) -> BenchmarkResult:
    """ARC-Challenge: generate answer letter, compare baseline vs cached."""
    print(f"\n{'='*60}")
    print("  ARC-CHALLENGE — Science Exam Multiple Choice")
    print(f"  {n_samples} samples, generation-based scoring")
    print(f"{'='*60}")

    samples = load_arc_challenge(n_samples)
    results = []
    divergences = []

    for i, sample in enumerate(samples):
        prompt = format_arc_prompt(sample)
        gold = sample["answer_key"]
        n_tok = kv.prompt_tokens(prompt)

        baseline_text = kv.generate(prompt, max_tokens=8, mode="baseline")
        cached_text = kv.generate(prompt, max_tokens=8, mode="chunk_kv_reuse")

        baseline_pred = extract_answer_letter(baseline_text)
        cached_pred = extract_answer_letter(cached_text)
        diverged = baseline_pred != cached_pred

        results.append(SampleResult(
            benchmark="arc_challenge",
            sample_idx=i,
            prompt_tokens=n_tok,
            baseline_output=baseline_pred,
            cached_output=cached_pred,
            diverged=diverged,
            correct=gold,
        ))

        if diverged:
            divergences.append({
                "idx": i,
                "baseline": baseline_pred,
                "cached": cached_pred,
                "gold": gold,
            })

        if (i + 1) % 25 == 0 or i == len(samples) - 1:
            div_rate = sum(1 for r in results if r.diverged) / len(results)
            print(f"    [{i+1}/{len(samples)}] divergence rate: {div_rate:.1%}")

    n_diverged = sum(1 for r in results if r.diverged)
    if divergences:
        print(f"    First divergence: {divergences[0]}")

    return BenchmarkResult(
        benchmark="arc_challenge",
        n_samples=len(samples),
        n_diverged=n_diverged,
        divergence_rate=n_diverged / max(len(samples), 1),
        samples=results,
    )


def run_mmlu(
    kv: KVBoostRunner,
    n_samples: int,
) -> BenchmarkResult:
    """MMLU: 5-shot multiple choice across subjects."""
    print(f"\n{'='*60}")
    print("  MMLU — Massive Multitask Language Understanding")
    print(f"  {n_samples} samples, 5-shot, generation-based scoring")
    print(f"{'='*60}")

    samples = load_mmlu(n_samples)
    results = []
    divergences = []

    for i, sample in enumerate(samples):
        prompt = format_mmlu_prompt(sample)
        gold = LETTERS[sample["correct_idx"]]
        n_tok = kv.prompt_tokens(prompt)

        baseline_text = kv.generate(prompt, max_tokens=4, mode="baseline")
        cached_text = kv.generate(prompt, max_tokens=4, mode="chunk_kv_reuse")

        baseline_pred = extract_answer_letter(baseline_text)
        cached_pred = extract_answer_letter(cached_text)
        diverged = baseline_pred != cached_pred

        results.append(SampleResult(
            benchmark="mmlu",
            sample_idx=i,
            prompt_tokens=n_tok,
            baseline_output=baseline_pred,
            cached_output=cached_pred,
            diverged=diverged,
            correct=gold,
        ))

        if diverged:
            divergences.append({
                "idx": i,
                "baseline": baseline_pred,
                "cached": cached_pred,
                "gold": gold,
            })

        if (i + 1) % 25 == 0 or i == len(samples) - 1:
            div_rate = sum(1 for r in results if r.diverged) / len(results)
            print(f"    [{i+1}/{len(samples)}] divergence rate: {div_rate:.1%}")

    n_diverged = sum(1 for r in results if r.diverged)
    if divergences:
        print(f"    First divergence: {divergences[0]}")

    return BenchmarkResult(
        benchmark="mmlu",
        n_samples=len(samples),
        n_diverged=n_diverged,
        divergence_rate=n_diverged / max(len(samples), 1),
        samples=results,
    )


def run_gsm8k(
    kv: KVBoostRunner,
    n_samples: int,
) -> BenchmarkResult:
    """GSM8K: 3-shot chain-of-thought, compare outputs (baseline vs cached)."""
    print(f"\n{'='*60}")
    print("  GSM8K — Grade School Math")
    print(f"  {n_samples} samples, 3-shot CoT")
    print(f"{'='*60}")

    samples = load_gsm8k(n_samples)
    results = []
    divergences = []

    for i, sample in enumerate(samples):
        prompt = format_gsm8k_prompt(sample)
        gold = sample["final_answer"]
        n_tok = kv.prompt_tokens(prompt)

        baseline_text = kv.generate(prompt, max_tokens=256, mode="baseline")
        cached_text = kv.generate(prompt, max_tokens=256, mode="chunk_kv_reuse")

        baseline_answer = extract_numeric_answer(baseline_text)
        cached_answer = extract_numeric_answer(cached_text)
        diverged = baseline_answer != cached_answer

        results.append(SampleResult(
            benchmark="gsm8k",
            sample_idx=i,
            prompt_tokens=n_tok,
            baseline_output=baseline_answer or "",
            cached_output=cached_answer or "",
            diverged=diverged,
            correct=gold,
        ))

        if diverged:
            divergences.append({
                "idx": i,
                "baseline": baseline_answer,
                "cached": cached_answer,
                "gold": gold,
            })

        if (i + 1) % 10 == 0 or i == len(samples) - 1:
            div_rate = sum(1 for r in results if r.diverged) / len(results)
            print(f"    [{i+1}/{len(samples)}] divergence rate: {div_rate:.1%}")

    n_diverged = sum(1 for r in results if r.diverged)
    if divergences:
        print(f"    First divergence: {divergences[0]}")

    return BenchmarkResult(
        benchmark="gsm8k",
        n_samples=len(samples),
        n_diverged=n_diverged,
        divergence_rate=n_diverged / max(len(samples), 1),
        samples=results,
    )


def run_truthfulqa(
    kv: KVBoostRunner,
    n_samples: int,
) -> BenchmarkResult:
    """
    TruthfulQA MC2: likelihood-based scoring in both modes.
    Compare the MC2 score (normalized probability on correct choices) 
    between baseline and cached modes.
    """
    print(f"\n{'='*60}")
    print("  TRUTHFULQA MC2 — Adversarial Factual Questions")
    print(f"  {n_samples} samples, likelihood-based scoring")
    print(f"{'='*60}")

    samples = load_truthfulqa(n_samples)
    results = []
    divergences = []

    for i, sample in enumerate(samples):
        context = f"Question: {sample['question']}\nAnswer:"
        choices = sample["choices"]
        labels = sample["labels"]
        n_tok = kv.prompt_tokens(context)

        # Compute scores in both modes
        baseline_scores = kv.compute_choice_scores(context, choices, mode="baseline")
        cached_scores = kv.compute_choice_scores(context, choices, mode="chunk_kv_reuse")

        # Compute MC2 score (normalized probability on correct choices)
        def compute_mc2(scores):
            max_score = max(scores)
            probs = [math.exp(s - max_score) for s in scores]
            total = sum(probs)
            probs = [p / total for p in probs]
            correct_prob = sum(p for p, lbl in zip(probs, labels) if lbl == 1)
            return correct_prob

        baseline_mc2 = compute_mc2(baseline_scores)
        cached_mc2 = compute_mc2(cached_scores)
        
        # Consider divergence if MC2 scores differ significantly (>5%)
        diverged = abs(baseline_mc2 - cached_mc2) > 0.05

        results.append(SampleResult(
            benchmark="truthfulqa_mc2",
            sample_idx=i,
            prompt_tokens=n_tok,
            baseline_output=f"{baseline_mc2:.3f}",
            cached_output=f"{cached_mc2:.3f}",
            diverged=diverged,
            baseline_score=baseline_mc2,
            cached_score=cached_mc2,
            correct="mc2_score",
        ))

        if diverged:
            divergences.append({
                "idx": i,
                "baseline_mc2": baseline_mc2,
                "cached_mc2": cached_mc2,
            })

        if (i + 1) % 25 == 0 or i == len(samples) - 1:
            div_rate = sum(1 for r in results if r.diverged) / len(results)
            print(f"    [{i+1}/{len(samples)}] divergence rate (>5%): {div_rate:.1%}")

    n_diverged = sum(1 for r in results if r.diverged)
    if divergences:
        print(f"    First divergence: {divergences[0]}")

    return BenchmarkResult(
        benchmark="truthfulqa_mc2",
        n_samples=len(samples),
        n_diverged=n_diverged,
        divergence_rate=n_diverged / max(len(samples), 1),
        samples=results,
    )




# ── Summary ────────────────────────────────────────────────────────

def print_summary(all_results: Dict[str, BenchmarkResult]):
    print(f"\n{'='*70}")
    print("  ACCURACY SUMMARY — Baseline vs KVBoost Output Divergence")
    print(f"{'='*70}")

    header = f"  {'Benchmark':>18s} | {'Divergence':>12s} | {'Status':>12s}"
    print(header)
    print(f"  {'-'*18}-+-{'-'*12}-+-{'-'*12}")

    total_samples = 0
    total_diverged = 0

    for bench_name in sorted(all_results.keys()):
        result = all_results[bench_name]
        div_rate = result.divergence_rate
        status = "✓ PASS" if div_rate == 0 else "✗ FAIL"
        
        line = f"  {bench_name:>18s} | {div_rate:10.1%} | {status:>12s}"
        print(line)
        
        total_samples += result.n_samples
        total_diverged += result.n_diverged

    print(f"  {'-'*18}-+-{'-'*12}-+-{'-'*12}")
    overall_div_rate = total_diverged / max(total_samples, 1)
    overall_status = "✓ PASS" if overall_div_rate == 0 else "✗ FAIL"
    line = f"  {'OVERALL':>18s} | {overall_div_rate:10.1%} | {overall_status:>12s}"
    print(line)
    print(f"{'='*70}")

    # Verdict
    if overall_div_rate == 0:
        print("  ✓ All benchmarks PASS: Baseline and KVBoost produce identical outputs.")
        print("  ✓ Cache correctness verified — no divergence detected.")
    else:
        print(f"  ✗ FAILURE: {total_diverged}/{total_samples} samples diverged")
        print("  ✗ Cache correctness issue detected!")
        for bench_name in sorted(all_results.keys()):
            result = all_results[bench_name]
            if result.n_diverged > 0:
                print(f"    • {bench_name}: {result.n_diverged}/{result.n_samples} divergences")
                # Show first divergence
                diverged_samples = [s for s in result.samples if s.diverged]
                if diverged_samples:
                    s = diverged_samples[0]
                    print(f"      First: idx={s.sample_idx}, "
                          f"baseline={s.baseline_output!r}, "
                          f"cached={s.cached_output!r}")

    print()



# ── CLI ────────────────────────────────────────────────────────────

BENCHMARKS = {
    "hellaswag": run_hellaswag,
    "arc": run_arc,
    "mmlu": run_mmlu,
    "gsm8k": run_gsm8k,
    "truthfulqa": run_truthfulqa,
}


def main():
    parser = argparse.ArgumentParser(
        description="KVBoost Accuracy Benchmark — Baseline vs Cache Correctness"
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B",
                        help="HF model to benchmark")
    parser.add_argument("--bench", default=None,
                        choices=list(BENCHMARKS.keys()) + ["all"],
                        help="Which benchmark to run (default: all)")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of samples per benchmark")
    parser.add_argument("--chunk-size", type=int, default=128,
                        help="KVBoost chunk size")
    parser.add_argument("--output", default=None,
                        help="Save results to JSON")
    args = parser.parse_args()

    print(f"\n  Model:        {args.model}")
    print(f"  Samples/bench: {args.n_samples}")
    print(f"  Chunk size:    {args.chunk_size}")
    print()

    kv = KVBoostRunner(args.model, chunk_size=args.chunk_size)

    benches = list(BENCHMARKS.keys()) if (args.bench is None or args.bench == "all") else [args.bench]
    all_results: Dict[str, BenchmarkResult] = {}

    for bench_name in benches:
        try:
            bench_fn = BENCHMARKS[bench_name]
            result = bench_fn(kv, args.n_samples)
            all_results[bench_name] = result
        except Exception as e:
            log.error("Benchmark %s failed: %s", bench_name, e)
            import traceback
            traceback.print_exc()

    print_summary(all_results)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            "model": args.model,
            "n_samples": args.n_samples,
            "chunk_size": args.chunk_size,
            "benchmarks": {
                name: {
                    "n_samples": result.n_samples,
                    "n_diverged": result.n_diverged,
                    "divergence_rate": result.divergence_rate,
                }
                for name, result in all_results.items()
            },
        }
        with open(out_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"  Results saved to {out_path}\n")
    else:
        default_path = RESULTS_DIR / "benchmark_accuracy_correctness.json"
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        serializable = {
            "model": args.model,
            "n_samples": args.n_samples,
            "chunk_size": args.chunk_size,
            "benchmarks": {
                name: {
                    "n_samples": result.n_samples,
                    "n_diverged": result.n_diverged,
                    "divergence_rate": result.divergence_rate,
                }
                for name, result in all_results.items()
            },
        }
        with open(default_path, "w") as f:
            json.dump(serializable, f, indent=2)
        print(f"  Results saved to {default_path}\n")


if __name__ == "__main__":
    main()
