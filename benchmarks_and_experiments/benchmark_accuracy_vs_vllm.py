#!/usr/bin/env python3
"""
KVBoost Accuracy Benchmark — Baseline vs Cache Correctness
===========================================================
Tests whether KVBoost with cache enabled produces equivalent accuracy to baseline.

Core hypothesis: mode=CHUNK_KV_REUSE achieves statistically identical accuracy
to mode=BASELINE on public benchmarks.

Methodology:
  1. For each sample, generate predictions in BOTH modes (randomized run order)
  2. Score both against ground truth
  3. Track: accuracy_baseline, accuracy_cached, divergence_rate, cache_hit_rate
  4. Statistical test: McNemar's test for paired correctness

Benchmarks (all established public benchmarks):
  - HellaSwag: commonsense completions
  - ARC-Challenge: science QA
  - MMLU: multitask knowledge (5-shot; shared prefix stays cached across samples)
  - GSM8K: math reasoning (3-shot CoT; shared prefix stays cached across samples)
  - TruthfulQA MC2: factual QA (multi-label scoring)

Fixes applied vs original:
  - Cache is NOT reset between samples in cached mode — shared few-shot prefixes
    (MMLU, GSM8K) stay resident in cache across samples, which is the real use case.
  - Run order is truly randomized per sample (half baseline-first, half cached-first)
    to eliminate order bias.
  - TruthfulQA scores against ALL correct answers, not just the first one.
  - output_similarity uses Jaccard (token overlap), correctly named.
  - full_text_divergence_rate and extracted_divergence_rate are tracked and reported
    as separate, clearly named fields throughout.
  - Default n_samples raised to 500 for adequate McNemar statistical power.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger("bench_accuracy")

RESULTS_DIR = Path(__file__).resolve().parent / "results"


# ── Result containers ──────────────────────────────────────────────

@dataclass
class SampleResult:
    benchmark: str
    sample_idx: int
    prompt_tokens: int

    baseline_output: str
    cached_output: str
    gold_answer: str

    baseline_correct: bool
    cached_correct: bool

    # Extracted-answer divergence (predicted letters/numbers differ)
    extracted_diverged: bool
    # Full-text divergence (any token in raw output differs)
    full_text_equal: bool = True

    # Jaccard similarity between raw output token sets
    output_similarity: float = 1.0

    # Timing
    baseline_time_ms: float = 0.0
    cached_time_ms: float = 0.0

    # Cache metrics (disaggregated)
    exact_cache_hits: int = 0
    approximate_hits: int = 0
    cache_misses: int = 0
    kv_reuse_ratio: float = 0.0

    # True = baseline ran before cached for this sample
    baseline_run_first: bool = True


@dataclass
class BenchmarkResult:
    benchmark: str
    n_samples: int

    baseline_accuracy: float
    cached_accuracy: float

    # Full-text divergence rate (more sensitive than extracted)
    full_text_divergence_rate: float
    # Extracted-answer divergence rate
    extracted_divergence_rate: float

    mcnemar_statistic: Optional[float] = None
    mcnemar_pvalue: Optional[float] = None

    avg_kv_reuse_ratio: float = 0.0
    avg_output_similarity: float = 0.0
    avg_speedup: float = 0.0

    # Disaggregated cache hits
    avg_exact_hits: float = 0.0
    avg_approx_hits: float = 0.0

    # Order bias: |full_text_div_rate when baseline_first - when cached_first|
    order_bias_delta: Optional[float] = None

    samples: List[SampleResult] = field(default_factory=list)

    both_correct: int = 0
    both_wrong: int = 0
    baseline_only_correct: int = 0
    cached_only_correct: int = 0


# ── KVBoost runner ─────────────────────────────────────────────────

class KVBoostRunner:
    """
    Wrapper for KVBoost that can switch between BASELINE and CHUNK_KV_REUSE.

    Key design decision: the cache is NOT cleared between individual samples.
    This lets shared few-shot prefixes (MMLU, GSM8K) remain resident across
    samples — which is exactly the real-world use case being validated.

    The cache IS cleared between benchmark suites (different benchmarks should
    not bleed into each other) via clear_cache().
    """

    def __init__(self, model_name: str, chunk_size: int = 128):
        from kvboost import KVBoost, GenerationMode

        self._mode_enums = {
            "baseline": GenerationMode.BASELINE,
            "cached": GenerationMode.CHUNK_KV_REUSE,
        }
        self.current_mode = "baseline"

        log.info(f"Loading KVBoost: {model_name}")
        self.engine = KVBoost.from_pretrained(
            model_name,
            chunk_size=chunk_size,
            recompute_overlap=8,
        )
        log.info(f"KVBoost ready on {self.engine.device}")

    def set_mode(self, mode: str):
        if mode not in self._mode_enums:
            raise ValueError(f"Unknown mode: {mode!r}")
        self.current_mode = mode

    def clear_cache(self):
        """Clear KV cache — call between benchmarks, NOT between samples."""
        if hasattr(self.engine, '_cache_manager'):
            self.engine._cache_manager.hits = 0
            self.engine._cache_manager.misses = 0
            self.engine._cache_manager.approximate_hits = 0
            self.engine._cache_manager.kv_cache.clear()
        if hasattr(self.engine, 'kv_cache'):
            self.engine.kv_cache.clear()

    def _snapshot_stats(self) -> dict:
        if hasattr(self.engine, 'cache_stats') and callable(self.engine.cache_stats):
            return dict(self.engine.cache_stats())
        return {}

    def generate(
        self,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.0,
    ) -> Tuple[str, dict]:
        """
        Generate from prompt in current mode.
        Returns (output_text, stats_dict).
        """
        pre = self._snapshot_stats()

        start = time.perf_counter()
        result = self.engine.generate(
            prompt,
            max_new_tokens=max_tokens,
            mode=self._mode_enums[self.current_mode],
            do_sample=(temperature > 0),
            temperature=temperature if temperature > 0 else 1.0,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        post = self._snapshot_stats()

        def _delta(key: str) -> int:
            a = pre.get(key, 0) or 0
            b = post.get(key, 0) or 0
            return max(0, int(b) - int(a))

        return result.output_text, {
            'time_ms': elapsed_ms,
            'exact_cache_hits': _delta('cache_hits'),
            'approximate_hits': _delta('approximate_hits'),
            'cache_misses': _delta('cache_misses'),
            'kv_reuse_ratio': getattr(result, 'kv_reuse_ratio', 0.0),
        }

    def prompt_tokens(self, text: str) -> int:
        return len(self.engine.tokenizer.encode(text, add_special_tokens=True))


# ── Similarity metric ──────────────────────────────────────────────

def jaccard_similarity(text1: str, text2: str) -> float:
    """Token-level Jaccard similarity between two strings."""
    t1 = set(text1.lower().split())
    t2 = set(text2.lower().split())
    if not t1 and not t2:
        return 1.0
    if not t1 or not t2:
        return 0.0
    return len(t1 & t2) / len(t1 | t2)


# ── Dataset loaders ────────────────────────────────────────────────

def load_hellaswag(n: int) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset("Rowan/hellaswag", split="validation")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    return [
        {"context": r["ctx"], "choices": r["endings"], "label": int(r["label"])}
        for r in ds
    ]


def load_arc_challenge(n: int) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    out = []
    for r in ds:
        labels = r["choices"]["label"]
        texts = r["choices"]["text"]
        key = r["answerKey"]
        out.append({
            "question": r["question"],
            "choices": texts,
            "choices_label": labels,
            "answer_key": key,
            "correct_idx": labels.index(key) if key in labels else 0,
        })
    return out


def load_mmlu(n: int) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    return [
        {
            "question": r["question"],
            "choices": r["choices"],
            "correct_idx": int(r["answer"]),
            "subject": r.get("subject", "unknown"),
        }
        for r in ds
    ]


def load_gsm8k(n: int) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    out = []
    for r in ds:
        final = r["answer"].split("####")[-1].strip()
        out.append({
            "question": r["question"],
            "answer_text": r["answer"],
            "final_answer": final,
        })
    return out


def load_truthfulqa(n: int) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    ds = ds.shuffle(seed=42).select(range(min(n, len(ds))))
    out = []
    for r in ds:
        mc2 = r["mc2_targets"]
        out.append({
            "question": r["question"],
            "choices": mc2["choices"],
            "labels": mc2["labels"],   # 1 = correct, 0 = incorrect; multiple 1s possible
        })
    return out


# ── Prompt formatters ──────────────────────────────────────────────

LETTERS = "ABCDEFGHIJ"


def format_mc_prompt(question: str, choices: List[str]) -> str:
    lines = [f"Question: {question}"]
    for i, c in enumerate(choices):
        lines.append(f"{LETTERS[i]}. {c}")
    lines.append("Answer:")
    return "\n".join(lines)


MMLU_FEW_SHOT_PREFIX = """The following are multiple choice questions (with answers).

Question: What is the embryological origin of the hyoid bone?
A. The first pharyngeal arch
B. The first and second pharyngeal arches
C. The second pharyngeal arch
D. The second and third pharyngeal arches
Answer: D

Question: Which of the following best describes the subnet mask 255.255.255.240?
A. /26
B. /27
C. /28
D. /29
Answer: C

Question: A standing wave on a string has 4 nodes. The wavelength is:
A. One-fourth the length of the string
B. One-half the length of the string
C. The same as the length of the string
D. Twice the length of the string
Answer: B

Question: What is the main purpose of a cardiac pacemaker?
A. To regulate blood pressure
B. To regulate heart rhythm
C. To improve blood flow
D. To prevent blood clots
Answer: B

"""

GSM8K_FEW_SHOT_PREFIX = """Solve step-by-step. Put the final numeric answer after ####.

Question: Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?
Answer: Roger started with 5 balls. 2 cans x 3 balls = 6 balls. 5 + 6 = 11. #### 11

Question: The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?
Answer: Started with 23, used 20: 23 - 20 = 3. Bought 6 more: 3 + 6 = 9. #### 9

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Total: 32 + 42 = 74. After eating 35: 74 - 35 = 39. #### 39

"""


# ── Answer extraction ──────────────────────────────────────────────

def extract_letter_answer(text: str, valid_letters: str = "ABCDE") -> Optional[str]:
    text = text.strip()
    m = re.match(r"^\(?([A-Ja-j])\)?[\.\)\s:]?", text)
    if m and m.group(1).upper() in valid_letters:
        return m.group(1).upper()
    for ch in text:
        if ch.upper() in valid_letters:
            return ch.upper()
    return None


def extract_numeric_answer(text: str) -> Optional[str]:
    m = re.search(r"####\s*([\-\d,\.]+)", text)
    if m:
        return m.group(1).replace(",", "").strip()
    numbers = re.findall(r"[\-]?\d[\d,]*\.?\d*", text)
    return numbers[-1].replace(",", "").strip() if numbers else None


def normalize_answer(text: Optional[str]) -> str:
    return (text or "").strip().lower()


# ── Divergence logging ─────────────────────────────────────────────

def log_divergence(
    benchmark: str,
    sample_idx: int,
    prompt: str,
    baseline_raw: str,
    cached_raw: str,
    baseline_pred: Optional[str],
    cached_pred: Optional[str],
    gold: str,
):
    """Log raw outputs for a divergent sample."""
    log.warning(
        f"\n{'='*70}\n"
        f"DIVERGENCE DETECTED in {benchmark} (sample {sample_idx})\n"
        f"{'='*70}\n"
        f"Gold answer: {gold!r}\n"
        f"\n--- Baseline Raw Output ---\n{baseline_raw!r}\n"
        f"\n--- Cached Raw Output ---\n{cached_raw!r}\n"
        f"\n--- Extracted Answers ---\n"
        f"Baseline pred: {baseline_pred!r}\n"
        f"Cached pred:   {cached_pred!r}\n"
        f"{'='*70}"
    )


# ── Core per-sample runner ─────────────────────────────────────────

def run_both_modes(
    runner: KVBoostRunner,
    prompt: str,
    max_tokens: int,
    baseline_first: bool,
) -> Tuple[Tuple[str, dict], Tuple[str, dict]]:
    """
    Run prompt in both modes. Order is controlled by baseline_first.
    Returns ((baseline_output, baseline_stats), (cached_output, cached_stats)).
    """
    if baseline_first:
        runner.set_mode("baseline")
        base_out, base_stats = runner.generate(prompt, max_tokens=max_tokens)
        runner.set_mode("cached")
        cached_out, cached_stats = runner.generate(prompt, max_tokens=max_tokens)
    else:
        runner.set_mode("cached")
        cached_out, cached_stats = runner.generate(prompt, max_tokens=max_tokens)
        runner.set_mode("baseline")
        base_out, base_stats = runner.generate(prompt, max_tokens=max_tokens)

    return (base_out, base_stats), (cached_out, cached_stats)


# ── Benchmark runners ──────────────────────────────────────────────

def run_hellaswag(runner: KVBoostRunner, n_samples: int) -> BenchmarkResult:
    """
    HellaSwag: 4-way commonsense completion.
    No shared prefix — each prompt is independent.
    """
    print(f"\n{'='*70}")
    print(f"  HELLASWAG — Commonsense Completion ({n_samples} samples)")
    print(f"{'='*70}")

    runner.clear_cache()
    samples = load_hellaswag(n_samples)
    rng = random.Random(42)
    results = []

    for i, sample in enumerate(samples):
        prompt = format_mc_prompt(sample["context"], sample["choices"])
        gold = LETTERS[sample["label"]]
        baseline_first = rng.random() < 0.5

        (base_out, base_stats), (cached_out, cached_stats) = run_both_modes(
            runner, prompt, max_tokens=4, baseline_first=baseline_first
        )

        base_pred = extract_letter_answer(base_out, valid_letters="ABCD")
        cached_pred = extract_letter_answer(cached_out, valid_letters="ABCD")

        if base_pred != cached_pred:
            log_divergence(
                "hellaswag", i, prompt,
                base_out, cached_out,
                base_pred, cached_pred, gold
            )

        results.append(SampleResult(
            benchmark="hellaswag",
            sample_idx=i,
            prompt_tokens=runner.prompt_tokens(prompt),
            baseline_output=base_pred or "",
            cached_output=cached_pred or "",
            gold_answer=gold,
            baseline_correct=(base_pred == gold),
            cached_correct=(cached_pred == gold),
            extracted_diverged=(base_pred != cached_pred),
            full_text_equal=(base_out.strip() == cached_out.strip()),
            output_similarity=jaccard_similarity(base_out, cached_out),
            baseline_time_ms=base_stats['time_ms'],
            cached_time_ms=cached_stats['time_ms'],
            exact_cache_hits=cached_stats['exact_cache_hits'],
            approximate_hits=cached_stats['approximate_hits'],
            cache_misses=cached_stats['cache_misses'],
            kv_reuse_ratio=cached_stats['kv_reuse_ratio'],
            baseline_run_first=baseline_first,
        ))

        if (i + 1) % 100 == 0:
            acc_b = sum(r.baseline_correct for r in results) / len(results)
            acc_c = sum(r.cached_correct for r in results) / len(results)
            print(f"  [{i+1}/{n_samples}] baseline={acc_b:.1%} cached={acc_c:.1%}")

    return _aggregate("hellaswag", results)


def run_arc(runner: KVBoostRunner, n_samples: int) -> BenchmarkResult:
    """
    ARC-Challenge: science QA.
    No shared prefix — each prompt is independent.
    """
    print(f"\n{'='*70}")
    print(f"  ARC-CHALLENGE — Science QA ({n_samples} samples)")
    print(f"{'='*70}")

    runner.clear_cache()
    samples = load_arc_challenge(n_samples)
    rng = random.Random(42)
    results = []

    for i, sample in enumerate(samples):
        prompt = format_mc_prompt(sample["question"], sample["choices"])
        gold = sample["answer_key"]
        valid = "".join(sample["choices_label"])
        baseline_first = rng.random() < 0.5

        (base_out, base_stats), (cached_out, cached_stats) = run_both_modes(
            runner, prompt, max_tokens=4, baseline_first=baseline_first
        )

        base_pred = extract_letter_answer(base_out, valid_letters=valid)
        cached_pred = extract_letter_answer(cached_out, valid_letters=valid)

        if base_pred != cached_pred:
            log_divergence(
                "arc_challenge", i, prompt,
                base_out, cached_out,
                base_pred, cached_pred, gold
            )

        results.append(SampleResult(
            benchmark="arc_challenge",
            sample_idx=i,
            prompt_tokens=runner.prompt_tokens(prompt),
            baseline_output=base_pred or "",
            cached_output=cached_pred or "",
            gold_answer=gold,
            baseline_correct=(base_pred == gold),
            cached_correct=(cached_pred == gold),
            extracted_diverged=(base_pred != cached_pred),
            full_text_equal=(base_out.strip() == cached_out.strip()),
            output_similarity=jaccard_similarity(base_out, cached_out),
            baseline_time_ms=base_stats['time_ms'],
            cached_time_ms=cached_stats['time_ms'],
            exact_cache_hits=cached_stats['exact_cache_hits'],
            approximate_hits=cached_stats['approximate_hits'],
            cache_misses=cached_stats['cache_misses'],
            kv_reuse_ratio=cached_stats['kv_reuse_ratio'],
            baseline_run_first=baseline_first,
        ))

        if (i + 1) % 100 == 0:
            acc_b = sum(r.baseline_correct for r in results) / len(results)
            acc_c = sum(r.cached_correct for r in results) / len(results)
            print(f"  [{i+1}/{n_samples}] baseline={acc_b:.1%} cached={acc_c:.1%}")

    return _aggregate("arc_challenge", results)


def run_mmlu(runner: KVBoostRunner, n_samples: int) -> BenchmarkResult:
    """
    MMLU: 5-shot multitask QA.

    The 5-shot prefix is identical for every sample. The cache is NOT reset
    between samples — so after the first sample the prefix KV entries are
    already resident and will be reused. This is the correct test of whether
    caching helps (and doesn't hurt) in a real few-shot serving scenario.
    """
    print(f"\n{'='*70}")
    print(f"  MMLU — Multitask QA, 5-shot ({n_samples} samples)")
    print(f"  Shared prefix stays cached across samples (real serving scenario)")
    print(f"{'='*70}")

    runner.clear_cache()
    samples = load_mmlu(n_samples)
    rng = random.Random(42)
    results = []

    for i, sample in enumerate(samples):
        prompt = MMLU_FEW_SHOT_PREFIX + format_mc_prompt(sample["question"], sample["choices"])
        gold = LETTERS[sample["correct_idx"]]
        baseline_first = rng.random() < 0.5

        (base_out, base_stats), (cached_out, cached_stats) = run_both_modes(
            runner, prompt, max_tokens=4, baseline_first=baseline_first
        )

        base_pred = extract_letter_answer(base_out, valid_letters="ABCD")
        cached_pred = extract_letter_answer(cached_out, valid_letters="ABCD")

        if base_pred != cached_pred:
            log_divergence(
                "mmlu", i, prompt,
                base_out, cached_out,
                base_pred, cached_pred, gold
            )

        results.append(SampleResult(
            benchmark="mmlu",
            sample_idx=i,
            prompt_tokens=runner.prompt_tokens(prompt),
            baseline_output=base_pred or "",
            cached_output=cached_pred or "",
            gold_answer=gold,
            baseline_correct=(base_pred == gold),
            cached_correct=(cached_pred == gold),
            extracted_diverged=(base_pred != cached_pred),
            full_text_equal=(base_out.strip() == cached_out.strip()),
            output_similarity=jaccard_similarity(base_out, cached_out),
            baseline_time_ms=base_stats['time_ms'],
            cached_time_ms=cached_stats['time_ms'],
            exact_cache_hits=cached_stats['exact_cache_hits'],
            approximate_hits=cached_stats['approximate_hits'],
            cache_misses=cached_stats['cache_misses'],
            kv_reuse_ratio=cached_stats['kv_reuse_ratio'],
            baseline_run_first=baseline_first,
        ))

        if (i + 1) % 100 == 0:
            acc_b = sum(r.baseline_correct for r in results) / len(results)
            acc_c = sum(r.cached_correct for r in results) / len(results)
            avg_reuse = np.mean([r.kv_reuse_ratio for r in results])
            print(f"  [{i+1}/{n_samples}] baseline={acc_b:.1%} cached={acc_c:.1%} "
                  f"kv_reuse={avg_reuse:.1%}")

    return _aggregate("mmlu", results)


def run_gsm8k(runner: KVBoostRunner, n_samples: int) -> BenchmarkResult:
    """
    GSM8K: 3-shot math reasoning (CoT).

    The 3-shot prefix is identical for every sample. The cache is NOT reset
    between samples — so after the first sample the prefix KV entries are
    already resident and will be reused. This correctly tests caching in a
    real few-shot serving scenario.
    """
    print(f"\n{'='*70}")
    print(f"  GSM8K — Math Reasoning, 3-shot CoT ({n_samples} samples)")
    print(f"  Shared prefix stays cached across samples (real serving scenario)")
    print(f"{'='*70}")

    runner.clear_cache()
    samples = load_gsm8k(n_samples)
    rng = random.Random(42)
    results = []

    for i, sample in enumerate(samples):
        prompt = GSM8K_FEW_SHOT_PREFIX + f"Question: {sample['question']}\nAnswer:"
        gold = sample["final_answer"]
        baseline_first = rng.random() < 0.5

        (base_out, base_stats), (cached_out, cached_stats) = run_both_modes(
            runner, prompt, max_tokens=256, baseline_first=baseline_first
        )

        base_ans = extract_numeric_answer(base_out)
        cached_ans = extract_numeric_answer(cached_out)

        if normalize_answer(base_ans) != normalize_answer(cached_ans):
            log_divergence(
                "gsm8k", i, prompt,
                base_out, cached_out,
                base_ans, cached_ans, gold
            )

        results.append(SampleResult(
            benchmark="gsm8k",
            sample_idx=i,
            prompt_tokens=runner.prompt_tokens(prompt),
            baseline_output=base_ans or "",
            cached_output=cached_ans or "",
            gold_answer=gold,
            baseline_correct=(normalize_answer(base_ans) == normalize_answer(gold)),
            cached_correct=(normalize_answer(cached_ans) == normalize_answer(gold)),
            extracted_diverged=(normalize_answer(base_ans) != normalize_answer(cached_ans)),
            full_text_equal=(base_out.strip() == cached_out.strip()),
            output_similarity=jaccard_similarity(base_out, cached_out),
            baseline_time_ms=base_stats['time_ms'],
            cached_time_ms=cached_stats['time_ms'],
            exact_cache_hits=cached_stats['exact_cache_hits'],
            approximate_hits=cached_stats['approximate_hits'],
            cache_misses=cached_stats['cache_misses'],
            kv_reuse_ratio=cached_stats['kv_reuse_ratio'],
            baseline_run_first=baseline_first,
        ))

        if (i + 1) % 100 == 0:
            acc_b = sum(r.baseline_correct for r in results) / len(results)
            acc_c = sum(r.cached_correct for r in results) / len(results)
            avg_reuse = np.mean([r.kv_reuse_ratio for r in results])
            print(f"  [{i+1}/{n_samples}] baseline={acc_b:.1%} cached={acc_c:.1%} "
                  f"kv_reuse={avg_reuse:.1%}")

    return _aggregate("gsm8k", results)


def run_truthfulqa(runner: KVBoostRunner, n_samples: int) -> BenchmarkResult:
    """
    TruthfulQA MC2: factual QA with multiple correct answers.

    Scoring: a prediction is correct if it matches ANY answer marked label=1.
    Previously the code only checked against the FIRST correct answer, which
    underestimates accuracy for both modes.
    """
    print(f"\n{'='*70}")
    print(f"  TRUTHFULQA MC2 — Factual QA ({n_samples} samples)")
    print(f"  Scoring: correct if prediction matches ANY label=1 answer")
    print(f"{'='*70}")

    runner.clear_cache()
    raw_samples = load_truthfulqa(n_samples)
    rng = random.Random(42)
    results = []

    for i, sample in enumerate(raw_samples):
        correct_indices = {j for j, lbl in enumerate(sample["labels"]) if lbl == 1}
        if not correct_indices:
            continue

        prompt = format_mc_prompt(sample["question"], sample["choices"])
        valid = LETTERS[:len(sample["choices"])]
        correct_letters = {LETTERS[j] for j in correct_indices}
        gold_display = "/".join(sorted(correct_letters))

        baseline_first = rng.random() < 0.5

        (base_out, base_stats), (cached_out, cached_stats) = run_both_modes(
            runner, prompt, max_tokens=4, baseline_first=baseline_first
        )

        base_pred = extract_letter_answer(base_out, valid_letters=valid)
        cached_pred = extract_letter_answer(cached_out, valid_letters=valid)

        # Correct = predicted letter is in the set of correct letters
        base_correct = base_pred in correct_letters if base_pred else False
        cached_correct = cached_pred in correct_letters if cached_pred else False

        if base_pred != cached_pred:
            log_divergence(
                "truthfulqa_mc2", i, prompt,
                base_out, cached_out,
                base_pred, cached_pred, gold_display
            )

        results.append(SampleResult(
            benchmark="truthfulqa_mc2",
            sample_idx=i,
            prompt_tokens=runner.prompt_tokens(prompt),
            baseline_output=base_pred or "",
            cached_output=cached_pred or "",
            gold_answer=gold_display,
            baseline_correct=base_correct,
            cached_correct=cached_correct,
            extracted_diverged=(base_pred != cached_pred),
            full_text_equal=(base_out.strip() == cached_out.strip()),
            output_similarity=jaccard_similarity(base_out, cached_out),
            baseline_time_ms=base_stats['time_ms'],
            cached_time_ms=cached_stats['time_ms'],
            exact_cache_hits=cached_stats['exact_cache_hits'],
            approximate_hits=cached_stats['approximate_hits'],
            cache_misses=cached_stats['cache_misses'],
            kv_reuse_ratio=cached_stats['kv_reuse_ratio'],
            baseline_run_first=baseline_first,
        ))

        if len(results) % 100 == 0:
            acc_b = sum(r.baseline_correct for r in results) / len(results)
            acc_c = sum(r.cached_correct for r in results) / len(results)
            print(f"  [{len(results)}/{n_samples}] baseline={acc_b:.1%} cached={acc_c:.1%}")

        if len(results) >= n_samples:
            break

    return _aggregate("truthfulqa_mc2", results)


# ── Statistics ─────────────────────────────────────────────────────

def _aggregate(bench_name: str, samples: List[SampleResult]) -> BenchmarkResult:
    n = len(samples)
    if n == 0:
        return BenchmarkResult(
            benchmark=bench_name, n_samples=0,
            baseline_accuracy=0.0, cached_accuracy=0.0,
            full_text_divergence_rate=0.0, extracted_divergence_rate=0.0,
        )

    baseline_acc = sum(s.baseline_correct for s in samples) / n
    cached_acc = sum(s.cached_correct for s in samples) / n

    full_text_div_rate = sum(not s.full_text_equal for s in samples) / n
    extracted_div_rate = sum(s.extracted_diverged for s in samples) / n

    both_correct = sum(s.baseline_correct and s.cached_correct for s in samples)
    both_wrong = sum(not s.baseline_correct and not s.cached_correct for s in samples)
    baseline_only = sum(s.baseline_correct and not s.cached_correct for s in samples)
    cached_only = sum(not s.baseline_correct and s.cached_correct for s in samples)

    # McNemar's test (continuity-corrected)
    mcnemar_stat = mcnemar_p = None
    discordant = baseline_only + cached_only
    if discordant > 0:
        mcnemar_stat = (abs(baseline_only - cached_only) - 1) ** 2 / discordant
        mcnemar_p = float(1 - stats.chi2.cdf(mcnemar_stat, df=1))

    avg_kv_reuse = float(np.mean([s.kv_reuse_ratio for s in samples]))
    avg_sim = float(np.mean([s.output_similarity for s in samples]))
    avg_exact = float(np.mean([s.exact_cache_hits for s in samples]))
    avg_approx = float(np.mean([s.approximate_hits for s in samples]))

    # Speedup
    timed = [s for s in samples if s.baseline_time_ms > 0 and s.cached_time_ms > 0]
    avg_speedup = float(
        np.mean([s.baseline_time_ms / s.cached_time_ms for s in timed])
    ) if timed else 0.0

    # Order bias: compare full-text divergence rate split by run order
    bf = [s for s in samples if s.baseline_run_first]
    cf = [s for s in samples if not s.baseline_run_first]
    order_bias = None
    if bf and cf:
        div_bf = sum(not s.full_text_equal for s in bf) / len(bf)
        div_cf = sum(not s.full_text_equal for s in cf) / len(cf)
        order_bias = abs(div_bf - div_cf)
        if order_bias > 0.05:
            log.warning(
                f"  ⚠ Order bias in {bench_name}: "
                f"div_when_baseline_first={div_bf:.1%}, "
                f"div_when_cached_first={div_cf:.1%} (delta={order_bias:.1%})"
            )

    return BenchmarkResult(
        benchmark=bench_name,
        n_samples=n,
        baseline_accuracy=baseline_acc,
        cached_accuracy=cached_acc,
        full_text_divergence_rate=full_text_div_rate,
        extracted_divergence_rate=extracted_div_rate,
        mcnemar_statistic=mcnemar_stat,
        mcnemar_pvalue=mcnemar_p,
        avg_kv_reuse_ratio=avg_kv_reuse,
        avg_output_similarity=avg_sim,
        avg_speedup=avg_speedup,
        avg_exact_hits=avg_exact,
        avg_approx_hits=avg_approx,
        order_bias_delta=order_bias,
        samples=samples,
        both_correct=both_correct,
        both_wrong=both_wrong,
        baseline_only_correct=baseline_only,
        cached_only_correct=cached_only,
    )


# ── Summary report ─────────────────────────────────────────────────

def print_summary(all_results: Dict[str, BenchmarkResult], verbose: bool = False):
    print(f"\n{'='*80}")
    print("  ACCURACY BENCHMARK SUMMARY")
    print(f"{'='*80}\n")

    # Main accuracy table
    print(f"  {'Benchmark':<16} {'Baseline':>9} {'Cached':>9} {'Delta':>8} "
          f"{'ExtDiv':>8} {'FullDiv':>8} {'p-value':>9} {'Result':>8}")
    print(f"  {'-'*16} {'-'*9} {'-'*9} {'-'*8} {'-'*8} {'-'*8} {'-'*9} {'-'*8}")

    overall_pass = True
    for name in sorted(all_results):
        r = all_results[name]
        delta = r.cached_accuracy - r.baseline_accuracy
        if r.mcnemar_pvalue is not None:
            p_str = f"{r.mcnemar_pvalue:.3f}"
            verdict = "PASS" if r.mcnemar_pvalue > 0.05 else "FAIL"
            if r.mcnemar_pvalue <= 0.05:
                overall_pass = False
        else:
            p_str = "N/A"
            verdict = "PASS" if abs(delta) < 0.01 else "WARN"

        print(f"  {name:<16} {r.baseline_accuracy:>8.1%} {r.cached_accuracy:>8.1%} "
              f"{delta:>+7.1%} {r.extracted_divergence_rate:>7.1%} "
              f"{r.full_text_divergence_rate:>7.1%} {p_str:>9} {verdict:>8}")

    # Aggregate row
    total = sum(r.n_samples for r in all_results.values())
    tot_b = sum(sum(s.baseline_correct for s in r.samples) for r in all_results.values())
    tot_c = sum(sum(s.cached_correct for s in r.samples) for r in all_results.values())
    ov_b = tot_b / total if total else 0
    ov_c = tot_c / total if total else 0
    ov_delta = ov_c - ov_b
    ov_ft = sum(
        sum(not s.full_text_equal for s in r.samples) for r in all_results.values()
    ) / total if total else 0

    print(f"\n  {'OVERALL':<16} {ov_b:>8.1%} {ov_c:>8.1%} {ov_delta:>+7.1%} "
          f"{'':>8} {ov_ft:>7.1%}")

    # Cache & similarity table
    print(f"\n  {'Benchmark':<16} {'KV Reuse':>9} {'Speedup':>9} "
          f"{'ExactHits':>10} {'ApproxHits':>11} {'Similarity':>11} {'OrderBias':>10}")
    print(f"  {'-'*16} {'-'*9} {'-'*9} {'-'*10} {'-'*11} {'-'*11} {'-'*10}")

    for name in sorted(all_results):
        r = all_results[name]
        ob = f"{r.order_bias_delta:.1%}" if r.order_bias_delta is not None else "N/A"
        ob_flag = " !" if r.order_bias_delta and r.order_bias_delta > 0.05 else ""
        print(f"  {name:<16} {r.avg_kv_reuse_ratio:>8.1%} {r.avg_speedup:>8.2f}x "
              f"{r.avg_exact_hits:>10.2f} {r.avg_approx_hits:>11.2f} "
              f"{r.avg_output_similarity:>11.3f} {ob+ob_flag:>10}")

    # Final verdict
    print(f"\n{'='*80}")
    if overall_pass and ov_ft < 0.01:
        print("  RESULT: Cache correctness VERIFIED")
        print("  No statistically significant accuracy degradation (McNemar p > 0.05)")
        print("  Full-text divergence rate < 1%")
        print(f"  Overall accuracy delta: {ov_delta:+.2%}")
    elif overall_pass:
        print("  WARNING: Extracted answers match, but subtle full-text divergences exist")
        print(f"  Full-text divergence rate: {ov_ft:.1%}")
        print("  Could indicate floating-point differences in intermediate tokens.")
        print("  Run with --verbose to inspect examples.")
    else:
        failing = [n for n, r in all_results.items()
                   if r.mcnemar_pvalue is not None and r.mcnemar_pvalue <= 0.05]
        print("  RESULT: Accuracy degradation detected")
        print(f"  Failing benchmarks: {', '.join(failing)}")
    print(f"{'='*80}\n")

    if verbose:
        for name, r in all_results.items():
            diverged = [s for s in r.samples if not s.full_text_equal]
            if diverged:
                print(f"\n  Full-text divergences in {name} ({len(diverged)}/{r.n_samples}):")
                for s in diverged[:5]:
                    print(f"    Sample {s.sample_idx} (similarity={s.output_similarity:.3f}):")
                    print(f"      Baseline: {s.baseline_output[:100]!r}")
                    print(f"      Cached:   {s.cached_output[:100]!r}")
                if len(diverged) > 5:
                    print(f"    ... and {len(diverged)-5} more")


# ── Main ───────────────────────────────────────────────────────────

BENCHMARKS = {
    "hellaswag": run_hellaswag,
    "arc": run_arc,
    "mmlu": run_mmlu,
    "gsm8k": run_gsm8k,
    "truthfulqa": run_truthfulqa,
}


def main():
    parser = argparse.ArgumentParser(
        description="KVBoost Accuracy Benchmark — Baseline vs CHUNK_KV_REUSE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B",
                        help="HuggingFace model name")
    parser.add_argument("--bench", default=None,
                        choices=list(BENCHMARKS.keys()) + ["all"],
                        help="Benchmark to run (default: all)")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Samples per benchmark (default: 500 for McNemar power)")
    parser.add_argument("--chunk-size", type=int, default=128,
                        help="KVBoost chunk size")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    parser.add_argument("--verbose", action="store_true",
                        help="Show divergence examples in summary")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    print(f"\n{'='*80}")
    print("  KVBoost Accuracy Benchmark")
    print(f"{'='*80}")
    print(f"  Model:           {args.model}")
    print(f"  Samples/bench:   {args.n_samples}")
    print(f"  Chunk size:      {args.chunk_size}")
    print(f"  Cache reset:     Between benchmarks only (NOT per sample)")
    print(f"  Run order:       Randomized per sample (seed=42)")
    print(f"  TruthfulQA:      Scores against all label=1 answers")
    print(f"{'='*80}\n")

    benches = (
        list(BENCHMARKS.keys())
        if args.bench is None or args.bench == "all"
        else [args.bench]
    )

    # Single shared runner — avoids reloading the model per benchmark
    runner = KVBoostRunner(args.model, chunk_size=args.chunk_size)

    all_results: Dict[str, BenchmarkResult] = {}
    total_start = time.perf_counter()

    for bench_name in benches:
        try:
            result = BENCHMARKS[bench_name](runner, args.n_samples)
            all_results[bench_name] = result
        except Exception as e:
            log.error(f"Benchmark {bench_name} failed: {e}", exc_info=True)

    total_time = time.perf_counter() - total_start
    log.info(f"All benchmarks completed in {total_time:.1f}s")

    print_summary(all_results, verbose=args.verbose)

    # Save results
    out_path = (
        Path(args.output) if args.output
        else RESULTS_DIR / "benchmark_accuracy.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        "model": args.model,
        "n_samples_per_bench": args.n_samples,
        "chunk_size": args.chunk_size,
        "cache_reset_policy": "between_benchmarks_only",
        "run_order": "randomized_per_sample",
        "total_time_seconds": total_time,
        "benchmarks": {},
    }

    for name, r in all_results.items():
        serializable["benchmarks"][name] = {
            "n_samples": r.n_samples,
            "baseline_accuracy": r.baseline_accuracy,
            "cached_accuracy": r.cached_accuracy,
            "accuracy_delta": r.cached_accuracy - r.baseline_accuracy,
            "full_text_divergence_rate": r.full_text_divergence_rate,
            "extracted_divergence_rate": r.extracted_divergence_rate,
            "avg_output_similarity": r.avg_output_similarity,
            "mcnemar_statistic": r.mcnemar_statistic,
            "mcnemar_pvalue": r.mcnemar_pvalue,
            "cache_metrics": {
                "avg_kv_reuse_ratio": r.avg_kv_reuse_ratio,
                "avg_exact_hits": r.avg_exact_hits,
                "avg_approximate_hits": r.avg_approx_hits,
                "avg_speedup": r.avg_speedup,
            },
            "order_bias_delta": r.order_bias_delta,
            "confusion_matrix": {
                "both_correct": r.both_correct,
                "both_wrong": r.both_wrong,
                "baseline_only_correct": r.baseline_only_correct,
                "cached_only_correct": r.cached_only_correct,
            },
            "samples": [
                {
                    "idx": s.sample_idx,
                    "prompt_tokens": s.prompt_tokens,
                    "baseline_output": s.baseline_output,
                    "cached_output": s.cached_output,
                    "gold_answer": s.gold_answer,
                    "baseline_correct": s.baseline_correct,
                    "cached_correct": s.cached_correct,
                    "extracted_diverged": s.extracted_diverged,
                    "full_text_equal": s.full_text_equal,
                    "output_similarity": s.output_similarity,
                    "baseline_time_ms": s.baseline_time_ms,
                    "cached_time_ms": s.cached_time_ms,
                    "exact_cache_hits": s.exact_cache_hits,
                    "approximate_hits": s.approximate_hits,
                    "cache_misses": s.cache_misses,
                    "kv_reuse_ratio": s.kv_reuse_ratio,
                    "baseline_run_first": s.baseline_run_first,
                }
                for s in r.samples
            ],
        }

    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)

    print(f"  Results saved to: {out_path}\n")


if __name__ == "__main__":
    main()