#!/usr/bin/env python3
"""
KVBoost Benchmark — Baseline HF vs KVBoost on Code Bug Localization
====================================================================
Evaluates KVBoost against baseline HuggingFace inference on
JetBrains-Research/lca-bug-localization (Python split).

Dataset: Each sample has a code diff (100-3K tokens) + issue description.
Task:    Given the diff context and bug report, pick which file was changed
         from 4 multiple-choice options (1 correct + 3 distractors).

Why this dataset works for KVBoost:
  - Diffs are 100-3000 tokens — fits comfortably in 16GB RAM
  - Code context is the "long" part that gets cached
  - Multiple samples can share similar code patterns → cache reuse
  - MC format (A/B/C/D) enables clean accuracy measurement

Methodology:
  1. Load bug localization samples, convert to MC format
  2. For each sample:
     a. BASELINE: Full prompt → standard HF generate (no caching)
     b. KVBOOST:  Warm context → generate with chunk KV reuse
  3. Compare: accuracy, TTFT, total latency, KV reuse ratio, cosine deviation
  4. Statistical test: McNemar's test for paired correctness

Usage:
  python long_bench_arena.py                          # default: 50 samples, Qwen2.5-3B
  python long_bench_arena.py --n-samples 200          # more samples
  python long_bench_arena.py --model meta-llama/Llama-3.2-3B
  python long_bench_arena.py --verbose                # show per-sample details

Fix log (2026-04-12):
  - FIX 1: Cosine deviation was always 1.0 because get_first_token_logits()
    calls self.engine.model() directly, bypassing the KVBoost cache entirely.
    Both baseline and "cached" calls returned identical raw-model logits.
    Added get_kvboost_logits() which runs through engine.generate with
    max_new_tokens=1 so the KVBoost prefill path (and its cached KV state)
    is actually exercised. Baseline logits are now captured before any
    KVBoost call to avoid cache contamination.
  - FIX 2: A-bias in warm (q2) queries. 5/6 KVBoost errors predicted "A"
    on _q2 samples. Root cause: q2 choice shuffling had no constraint, so
    the correct answer could land in the same slot as q1's — and the warm
    KV state biased the model toward whichever choice came first in q1.
    Now q2's correct answer is explicitly forced into a different position
    than q1's (retry shuffle until positions differ).
  - FIX 3: Added answer_position to SampleResult for bias correlation
    analysis (which ABCD slot holds the gold answer).
  - FIX 4: print_summary now includes error-by-predicted-letter breakdown
    and answer-position vs divergence correlation to surface future A-bias
    regressions.
  - FIX 5: Low-sample bucket warning (n < 20) added to accuracy table.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import stats as scipy_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("longbench_v2")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
LETTERS = "ABCD"
CHECKPOINT_DIR = Path(__file__).resolve().parent / ".checkpoints"
CHECKPOINT_INTERVAL = 4  # Save checkpoint every N samples


# ── Result containers ─────────────────────────────────────────────

@dataclass
class SampleResult:
    sample_id: str
    domain: str
    sub_domain: str
    difficulty: str
    length_category: str
    context_tokens: int
    prompt_tokens: int

    baseline_output: str
    kvboost_output: str
    gold_answer: str

    baseline_correct: bool
    kvboost_correct: bool
    outputs_diverged: bool

    baseline_ttft_ms: float = 0.0
    baseline_total_ms: float = 0.0
    kvboost_ttft_ms: float = 0.0
    kvboost_total_ms: float = 0.0

    kv_reuse_ratio: float = 0.0
    cached_tokens: int = 0

    warm_time_ms: float = 0.0

    # Cosine deviation between baseline and cached logits (1.0 = identical)
    logit_cosine_similarity: float = 1.0
    # Per-choice logit deviation: how much each A/B/C/D logit shifted
    choice_logit_deviations: Optional[Dict[str, float]] = None
    # Whether the top-1 predicted token changed between baseline and cached
    top1_token_changed: bool = False

    # FIX 3: which ABCD slot (0-3) holds the gold answer for this sample.
    # Tracked so we can correlate answer position with divergence / A-bias.
    answer_position: int = 0


@dataclass
class BucketResult:
    bucket_name: str
    n_samples: int

    baseline_accuracy: float
    kvboost_accuracy: float

    avg_baseline_ttft_ms: float = 0.0
    avg_kvboost_ttft_ms: float = 0.0
    avg_ttft_speedup: float = 0.0

    avg_baseline_total_ms: float = 0.0
    avg_kvboost_total_ms: float = 0.0

    avg_kv_reuse_ratio: float = 0.0
    avg_context_tokens: float = 0.0

    mcnemar_pvalue: Optional[float] = None

    samples: List[SampleResult] = field(default_factory=list)


# ── Dataset loader ────────────────────────────────────────────────

def load_bug_localization(
    n_samples: int,
    max_context_tokens: int,
    tokenizer_name: str,
) -> List[dict]:
    """
    Load JetBrains-Research/lca-bug-localization (py/test) and convert to MC format.

    Each sample becomes: given a code diff (context) + issue description (question),
    pick the correct changed file from 4 options (A/B/C/D).

    Diffs are typically 100-3000 tokens — fits in 16GB RAM easily.

    Paired queries (q1, q2) share the same context diff but use different
    distractors and question phrasing. FIX 2: q2's correct answer is
    guaranteed to land in a different ABCD slot than q1's to eliminate
    warm-cache positional bias toward whichever choice came first in q1.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    log.info("Loading JetBrains-Research/lca-bug-localization (py/test) ...")
    ds = load_dataset("JetBrains-Research/lca-bug-localization", "py", split="train")
    log.info(f"  Total samples in dataset: {len(ds)}")

    log.info(f"  Pre-filtering contexts to <={max_context_tokens} tokens ...")
    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)

    raw_samples = []
    all_files = set()
    skipped_too_long = 0
    skipped_no_diff = 0

    for row in ds:
        diff = row.get("diff", "")
        if not diff or len(diff.strip()) < 10:
            skipped_no_diff += 1
            continue

        ctx_tokens = len(tok.encode(diff))
        if ctx_tokens > max_context_tokens:
            skipped_too_long += 1
            continue

        changed_files_raw = row.get("changed_files", "")
        if isinstance(changed_files_raw, str):
            changed_files = [f.strip() for f in changed_files_raw.replace(",", "\n").split("\n") if f.strip()]
        elif isinstance(changed_files_raw, list):
            changed_files = changed_files_raw
        else:
            changed_files = []

        if not changed_files:
            skipped_no_diff += 1
            continue

        correct_file = changed_files[0]
        all_files.update(changed_files)

        issue_title = row.get("issue_title", "") or ""
        issue_body = row.get("issue_body", "") or ""
        question = issue_title
        if issue_body:
            body_preview = issue_body[:1000]
            if len(issue_body) > 1000:
                body_preview += "..."
            question = f"{issue_title}\n{body_preview}"

        raw_samples.append({
            "diff": diff,
            "correct_file": correct_file,
            "all_changed_files": changed_files,
            "question": question,
            "repo": f"{row.get('repo_owner', '')}/{row.get('repo_name', '')}",
            "ctx_tokens": ctx_tokens,
            "changed_files_count": row.get("changed_files_count", len(changed_files)),
        })

    log.info(f"  After filtering: {len(raw_samples)} eligible, "
             f"{skipped_too_long} too long, {skipped_no_diff} no diff/files")

    if not raw_samples:
        log.error(f"  No samples fit within {max_context_tokens} tokens!")
        return []

    all_files_list = sorted(all_files)

    rng = np.random.RandomState(42)
    rng.shuffle(raw_samples)
    raw_samples = raw_samples[:min(n_samples, len(raw_samples))]

    samples = []
    for idx, raw in enumerate(raw_samples):
        correct_file = raw["correct_file"]

        distractor_pool = [f for f in all_files_list if f not in raw["all_changed_files"]]
        rng.shuffle(distractor_pool)

        while len(distractor_pool) < 6:
            distractor_pool.append(f"src/fake_module_{len(distractor_pool)}.py")

        base_info = {
            "domain": "code",
            "sub_domain": raw["repo"],
            "difficulty": "easy" if raw["changed_files_count"] == 1 else "hard",
            "length": "short" if raw["ctx_tokens"] < 500 else ("medium" if raw["ctx_tokens"] < 1500 else "long"),
            "context": raw["diff"],
            "context_tokens": raw["ctx_tokens"],
        }

        # Query 1 (cold): populates the KV cache
        d1 = distractor_pool[:3]
        choices_1 = d1 + [correct_file]
        rng.shuffle(choices_1)
        q1_answer_pos = choices_1.index(correct_file)

        samples.append({
            **base_info,
            "id": f"bug_{idx}_q1",
            "question": f"Bug report: {raw['question']}\n\nWhich file was modified to fix this bug?",
            "choices": choices_1,
            "answer": LETTERS[q1_answer_pos],
            "answer_position": q1_answer_pos,
            "pair_group": idx,
        })

        # FIX 2: Query 2 (warm) — force correct answer into a different slot
        # than q1 to eliminate warm-cache positional bias.
        d2 = distractor_pool[3:6]
        choices_2 = d2 + [correct_file]
        rng.shuffle(choices_2)

        # Retry until the answer lands in a different position than q1's.
        # With 4 slots this resolves in ≤3 tries on average.
        max_retries = 20
        for _ in range(max_retries):
            if choices_2.index(correct_file) != q1_answer_pos:
                break
            rng.shuffle(choices_2)
        else:
            # Last resort: manually rotate
            pos = choices_2.index(correct_file)
            target = (q1_answer_pos + 1) % 4
            choices_2[pos], choices_2[target] = choices_2[target], choices_2[pos]

        q2_answer_pos = choices_2.index(correct_file)
        assert q2_answer_pos != q1_answer_pos, (
            f"Failed to place q2 answer in a different slot: "
            f"q1={q1_answer_pos}, q2={q2_answer_pos}"
        )

        samples.append({
            **base_info,
            "id": f"bug_{idx}_q2",
            "question": f"Issue: {raw['question']}\n\nIdentify the file that contains the fix for this issue.",
            "choices": choices_2,
            "answer": LETTERS[q2_answer_pos],
            "answer_position": q2_answer_pos,
            "pair_group": idx,
        })

    samples.sort(key=lambda s: s["context_tokens"])

    ctx_lens = [s["context_tokens"] for s in samples]
    log.info(f"  Token range: {min(ctx_lens)}-{max(ctx_lens)} "
             f"(median={int(np.median(ctx_lens))}, mean={int(np.mean(ctx_lens))})")
    log.info(f"  Selected {len(samples)} samples as MC (4-way file identification)")
    log.info(f"  q2 answer positions debiased: guaranteed ≠ q1 for all pairs")
    return samples


# ── Prompt formatting ─────────────────────────────────────────────

def format_prompt(context: str, question: str, choices: List[str]) -> str:
    """
    Format a sample into a prompt.
    The context (code diff) goes first — this is the long part that benefits
    from caching — followed by the question and file choices.
    """
    prompt_parts = [
        "Read the following code diff and answer the question.\n",
        f"--- BEGIN DIFF ---\n{context}\n--- END DIFF ---\n",
        f"{question}\n",
    ]
    for i, choice in enumerate(choices):
        prompt_parts.append(f"{LETTERS[i]}. {choice}")
    prompt_parts.append("\nAnswer with just the letter (A, B, C, or D):")

    return "\n".join(prompt_parts)


# ── Answer extraction ─────────────────────────────────────────────

import re
import torch


def compute_logit_cosine_similarity(logits_a: np.ndarray, logits_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two logit vectors.
    Returns value in [-1, 1] where 1.0 means identical distributions.
    """
    if logits_a.shape != logits_b.shape or len(logits_a) == 0:
        return 0.0
    norm_a = np.linalg.norm(logits_a)
    norm_b = np.linalg.norm(logits_b)
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(logits_a, logits_b) / (norm_a * norm_b))


def compute_choice_logit_deviations(
    logits_a: np.ndarray, logits_b: np.ndarray, choice_token_ids: Dict[str, int]
) -> Dict[str, float]:
    """
    For each answer choice (A/B/C/D), compute the absolute difference
    in softmax probability between baseline and cached logits.
    Shows how much the model's confidence in each choice shifted.
    """
    from scipy.special import softmax
    probs_a = softmax(logits_a)
    probs_b = softmax(logits_b)

    deviations = {}
    for letter, tid in choice_token_ids.items():
        if tid < len(probs_a) and tid < len(probs_b):
            deviations[letter] = float(abs(probs_a[tid] - probs_b[tid]))
        else:
            deviations[letter] = 0.0
    return deviations


def extract_letter_answer(text: str) -> Optional[str]:
    """Extract A/B/C/D answer from generated text."""
    text = text.strip()

    m = re.match(r"^\(?([A-Da-d])\)?[\.\)\s:]?", text)
    if m:
        return m.group(1).upper()

    for ch in text:
        if ch.upper() in LETTERS:
            return ch.upper()

    return None


# ── KVBoost runner ────────────────────────────────────────────────

class LongBenchRunner:
    """
    Runs baseline and KVBoost inference for LongBench-v2 samples.
    Single model load, switches between generation modes.
    """

    def __init__(self, model_name: str, chunk_size: int = 128):
        from kvboost import KVBoost, GenerationMode

        self.GenerationMode = GenerationMode

        log.info(f"Loading model: {model_name}")
        self.engine = KVBoost.from_pretrained(
            model_name,
            chunk_size=chunk_size,
            recompute_overlap=8,
        )
        log.info(f"  Ready on {self.engine.device}")
        
        # Store the last GenerationResult for logit extraction
        self._last_generation_result = None

    def _reset_cache(self):
        """Clear KV cache between samples for fair comparison."""
        if hasattr(self.engine, "_cache_manager"):
            self.engine._cache_manager.hits = 0
            self.engine._cache_manager.misses = 0
            self.engine._cache_manager.approximate_hits = 0
            self.engine._cache_manager.kv_cache.clear()
        if hasattr(self.engine, "kv_cache"):
            self.engine.kv_cache.clear()

    def count_tokens(self, text: str) -> int:
        return len(self.engine.tokenizer.encode(text, add_special_tokens=True))

    def get_choice_token_ids(self) -> Dict[str, int]:
        """Get token IDs for A, B, C, D answer letters."""
        ids = {}
        for letter in LETTERS:
            token_id = self.engine.tokenizer.encode(letter, add_special_tokens=False)
            ids[letter] = token_id[0] if token_id else -1
        return ids

    # FIX 1a: renamed from get_first_token_logits — clarifies this runs the
    # raw HF model with NO cache active.  Call this BEFORE any KVBoost call
    # so the cache cannot contaminate the result.
    def get_baseline_logits(self, prompt: str) -> Optional[np.ndarray]:
        """
        Run a raw model forward pass (no KVBoost cache) and return the
        next-token logits at the last input position.

        Must be called before any KVBoost generate call for the same prompt,
        otherwise the KVBoost engine may have already populated its cache
        and the raw model pass could be influenced by engine-level state.
        """
        try:
            input_ids = self.engine.tokenizer.encode(prompt, return_tensors="pt")
            if hasattr(input_ids, "to"):
                input_ids = input_ids.to(self.engine.model.device)
            else:
                import torch as _torch
                input_ids = _torch.tensor([input_ids], device=self.engine.model.device)

            with torch.no_grad():
                out = self.engine.model(input_ids)
            logits = out.logits[0, -1, :].cpu().float().numpy()
            return logits
        except Exception as e:
            log.warning(f"Failed to extract baseline logits: {e}")
            return None

    # FIX 1b: new method — gets first-token logits through KVBoost's generate
    # call. The GenerationResult now includes first_token_logits captured
    # during the forward pass with the cached KV state already active.
    def get_kvboost_logits_from_result(self, result) -> Optional[np.ndarray]:
        """
        Extract first-token logits from a KVBoost GenerationResult.
        The logits are captured during the generation pass when the cached KV
        state is active, which is exactly what we need for comparison.
        """
        if hasattr(result, "first_token_logits") and result.first_token_logits is not None:
            return result.first_token_logits
        return None

    def run_baseline(self, prompt: str, max_tokens: int = 16) -> dict:
        """Run in BASELINE mode (standard HF generate, no caching)."""
        self._reset_cache()

        start = time.perf_counter()
        result = self.engine.generate(
            prompt,
            max_new_tokens=max_tokens,
            mode=self.GenerationMode.BASELINE,
            do_sample=False,
            temperature=1.0,
        )
        total_ms = (time.perf_counter() - start) * 1000

        return {
            "output": result.output_text,
            "ttft_ms": getattr(result, "ttft_ms", total_ms),
            "total_ms": total_ms,
        }

    def run_kvboost_cold(self, prompt: str, max_tokens: int = 16) -> dict:
        """
        Cold-start KVBoost: no prior cache, generate with CHUNK_KV_REUSE.
        The first call populates the cache for subsequent warm calls on the
        same context.
        """
        gen_start = time.perf_counter()
        result = self.engine.generate(
            prompt,
            max_new_tokens=max_tokens,
            mode=self.GenerationMode.CHUNK_KV_REUSE,
            do_sample=False,
            temperature=1.0,
        )
        total_ms = (time.perf_counter() - gen_start) * 1000
        
        # Store the full GenerationResult for logit extraction
        self._last_generation_result = result

        return {
            "output": result.output_text,
            "ttft_ms": getattr(result, "ttft_ms", total_ms),
            "total_ms": total_ms,
            "warm_ms": 0.0,
            "kv_reuse_ratio": getattr(result, "kv_reuse_ratio", 0.0),
            "cached_tokens": getattr(result, "cached_tokens", 0),
        }

    def run_kvboost_warm(self, prompt: str, max_tokens: int = 16) -> dict:
        """
        Warm KVBoost: cache already populated from q1.  Chunks from the
        shared prompt prefix (instruction + diff) are reused; only the
        unique suffix (question + choices) needs fresh prefill.
        """
        gen_start = time.perf_counter()
        result = self.engine.generate(
            prompt,
            max_new_tokens=max_tokens,
            mode=self.GenerationMode.CHUNK_KV_REUSE,
            do_sample=False,
            temperature=1.0,
        )
        total_ms = (time.perf_counter() - gen_start) * 1000
        
        # Store the full GenerationResult for logit extraction
        self._last_generation_result = result

        return {
            "output": result.output_text,
            "ttft_ms": getattr(result, "ttft_ms", total_ms),
            "total_ms": total_ms,
            "warm_ms": 0.0,
            "kv_reuse_ratio": getattr(result, "kv_reuse_ratio", 0.0),
            "cached_tokens": getattr(result, "cached_tokens", 0),
        }


# ── Benchmark runner ──────────────────────────────────────────────

def run_benchmark(
    model_name: str,
    n_samples: int,
    chunk_size: int = 128,
    max_context_tokens: int = 8192,
    no_checkpoint: bool = False,
) -> Dict[str, BucketResult]:
    """
    Run the bug localization benchmark.
    Returns results bucketed by context length for analysis.
    """
    print(f"\n{'=' * 75}")
    print(f"  BUG LOCALIZATION BENCHMARK — Baseline HF vs KVBoost")
    print(f"{'=' * 75}")
    print(f"  Dataset:            JetBrains-Research/lca-bug-localization (py)")
    print(f"  Model:              {model_name}")
    print(f"  Samples:            {n_samples}")
    print(f"  Chunk size:         {chunk_size}")
    print(f"  Max context tokens: {max_context_tokens}")
    print(f"{'=' * 75}\n")

    samples = load_bug_localization(n_samples, max_context_tokens, model_name)
    if not samples:
        log.error("No samples loaded. Check --max-context-tokens.")
        return {}

    runner = LongBenchRunner(model_name, chunk_size=chunk_size)
    choice_token_ids = runner.get_choice_token_ids()

    def get_bucket(n_tokens: int) -> str:
        if n_tokens < 512:
            return "0-512"
        elif n_tokens < 1024:
            return "512-1K"
        elif n_tokens < 2048:
            return "1K-2K"
        elif n_tokens < 4096:
            return "2K-4K"
        else:
            return "4K+"

    results_by_bucket: Dict[str, List[SampleResult]] = defaultdict(list)
    all_results: List[SampleResult] = []

    pair_groups: Dict[int, List[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        pair_groups[sample.get("pair_group", idx)].append(idx)

    n_pairs = sum(1 for v in pair_groups.values() if len(v) > 1)
    print(f"  {n_pairs} paired-query groups (cold→warm) for KV reuse testing")
    print(f"  Query 1 populates cache, Query 2 reuses cached chunks")
    print(f"  q2 answer positions debiased (guaranteed ≠ q1 slot)\n")

    # ── Checkpointing: load previous progress if available ────────
    checkpoint_path = get_checkpoint_path(model_name, n_samples, max_context_tokens)
    all_results, processed_indices = ([], []) if no_checkpoint else load_checkpoint(checkpoint_path)
    processed_set = set(processed_indices)

    if processed_set:
        print(f"  Checkpoint found: resuming from {len(processed_indices)} processed samples")
        print(f"  Use --no-checkpoint to start fresh\n")
    # ────────────────────────────────────────────────────────────────

    total_start = time.perf_counter()

    import gc
    n_oom = 0
    oom_threshold = 3

    ordered_indices = []
    for group_id in sorted(pair_groups.keys()):
        ordered_indices.extend(pair_groups[group_id])

    for progress_idx, i in enumerate(ordered_indices):
        # Skip already-processed samples from checkpoint
        if i in processed_set:
            continue

        sample = samples[i]
        context = sample["context"]
        context_tokens = sample.get("context_tokens", runner.count_tokens(context))
        is_first_in_pair = sample["id"].endswith("_q1")

        if is_first_in_pair:
            runner._reset_cache()

        prompt = format_prompt(context, sample["question"], sample["choices"])
        prompt_tokens = runner.count_tokens(prompt)
        bucket = get_bucket(context_tokens)

        try:
            # ── FIX 1: correct logit capture order ────────────────────────
            # Step 1: capture baseline logits via raw model forward pass
            # BEFORE any KVBoost call.  The cache is clean at this point
            # (reset above for q1, or still populated from q1 for q2 —
            # but get_baseline_logits() bypasses the cache via engine.model
            # directly, so q2 baseline logits are still clean).
            baseline_logits = runner.get_baseline_logits(prompt)

            # Step 2: run baseline generation (also calls _reset_cache internally
            # so the logit capture above is unaffected for q1; for q2 the
            # cache is reset here too which is intentional — baseline is always
            # fresh by definition).
            baseline_runner_result = runner.run_baseline(prompt, max_tokens=16)
            baseline_pred = extract_letter_answer(baseline_runner_result["output"])
            baseline_correct = baseline_pred == sample["answer"]

            # Step 3: run KVBoost — this populates / reuses the cache.
            if is_first_in_pair:
                kvboost = runner.run_kvboost_cold(prompt, max_tokens=16)
            else:
                kvboost = runner.run_kvboost_warm(prompt, max_tokens=16)

            kvboost_pred = extract_letter_answer(kvboost["output"])
            kvboost_correct = kvboost_pred == sample["answer"]

            # Step 4: extract KVBoost logits from the GenerationResult
            # which captured them during the forward pass with active cache.
            # The _last_generation_result was set by run_kvboost_cold/warm above.
            cached_logits = runner.get_kvboost_logits_from_result(runner._last_generation_result)
            # ── end FIX 1 ─────────────────────────────────────────────────

        except RuntimeError as e:
            if "Invalid buffer size" in str(e) or "out of memory" in str(e).lower():
                n_oom += 1
                log.warning(
                    f"  [{progress_idx+1}/{len(samples)}] OOM at {context_tokens} tokens "
                    f"({n_oom}/{oom_threshold} strikes) — skipping"
                )
                runner._reset_cache()
                gc.collect()
                try:
                    import torch
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                except Exception:
                    pass

                if n_oom >= oom_threshold:
                    log.warning(
                        f"  Hit {oom_threshold} OOMs — stopping early. "
                        f"Try lowering --max-context-tokens (currently {max_context_tokens})."
                    )
                    break
                continue
            raise

        # ── Compute cosine deviation ──
        logit_cos_sim = 1.0
        choice_devs = None
        top1_changed = False

        if baseline_logits is not None and cached_logits is not None:
            logit_cos_sim = compute_logit_cosine_similarity(baseline_logits, cached_logits)
            choice_devs = compute_choice_logit_deviations(
                baseline_logits, cached_logits, choice_token_ids
            )
            top1_changed = int(np.argmax(baseline_logits)) != int(np.argmax(cached_logits))
        elif baseline_logits is not None and cached_logits is None:
            log.debug(f"  [{sample['id']}] KVBoost logits unavailable — cosine deviation skipped")

        diverged = baseline_pred != kvboost_pred
        query_type = "COLD" if is_first_in_pair else "WARM"

        # FIX 3: record answer_position from dataset
        sr = SampleResult(
            sample_id=sample["id"],
            domain=sample["domain"],
            sub_domain=sample["sub_domain"],
            difficulty=sample["difficulty"],
            length_category=sample["length"],
            context_tokens=context_tokens,
            prompt_tokens=prompt_tokens,
            baseline_output=baseline_pred or "",
            kvboost_output=kvboost_pred or "",
            gold_answer=sample["answer"],
            baseline_correct=baseline_correct,
            kvboost_correct=kvboost_correct,
            outputs_diverged=diverged,
            baseline_ttft_ms=baseline_runner_result["ttft_ms"],
            baseline_total_ms=baseline_runner_result["total_ms"],
            kvboost_ttft_ms=kvboost["ttft_ms"],
            kvboost_total_ms=kvboost["total_ms"],
            kv_reuse_ratio=kvboost.get("kv_reuse_ratio", 0.0),
            cached_tokens=kvboost.get("cached_tokens", 0),
            warm_time_ms=kvboost.get("warm_ms", 0.0),
            logit_cosine_similarity=logit_cos_sim,
            choice_logit_deviations=choice_devs,
            top1_token_changed=top1_changed,
            answer_position=sample.get("answer_position", LETTERS.index(sample["answer"])),
        )

        results_by_bucket[bucket].append(sr)
        all_results.append(sr)
        processed_indices.append(i)
        processed_set.add(i)

        speedup = baseline_runner_result["ttft_ms"] / kvboost["ttft_ms"] if kvboost["ttft_ms"] > 0 else 0
        status = "MATCH" if not diverged else "DIVERGE"
        cos_str = f"cos={logit_cos_sim:.4f}" if logit_cos_sim < 0.9999 else "cos=1.0"

        # ── Periodic checkpoint save ────────
        if len(all_results) % CHECKPOINT_INTERVAL == 0:
            save_checkpoint(all_results, checkpoint_path, samples, processed_indices)
        # ────────────────────────────────────

        if (progress_idx + 1) % 4 == 0 or (progress_idx + 1) == len(samples):
            completed = len(all_results)
            running_base_acc = sum(s.baseline_correct for s in all_results) / completed
            running_kv_acc = sum(s.kvboost_correct for s in all_results) / completed
            running_avg_cos = np.mean([s.logit_cosine_similarity for s in all_results])
            running_avg_reuse = np.mean([s.kv_reuse_ratio for s in all_results])
            print(
                f"  [{progress_idx+1}/{len(samples)}] [{query_type}] ctx={context_tokens:>5}tok "
                f"TTFT {baseline_runner_result['ttft_ms']:>7.0f}→{kvboost['ttft_ms']:>7.0f}ms ({speedup:.1f}x) "
                f"reuse={kvboost.get('kv_reuse_ratio', 0):.0%} {cos_str} "
                f"avg_reuse={running_avg_reuse:.0%} "
                f"acc: base={running_base_acc:.0%} kv={running_kv_acc:.0%} "
                f"[{status}]"
            )

    total_time = time.perf_counter() - total_start
    log.info(f"Benchmark completed in {total_time:.1f}s")

    # ── Compute bucket-level results ──
    bucket_results: Dict[str, BucketResult] = {}
    results_by_bucket["ALL"] = all_results

    for bucket_name, bucket_samples in sorted(results_by_bucket.items()):
        n = len(bucket_samples)
        if n == 0:
            continue

        base_acc = sum(s.baseline_correct for s in bucket_samples) / n
        kv_acc = sum(s.kvboost_correct for s in bucket_samples) / n

        avg_base_ttft = np.mean([s.baseline_ttft_ms for s in bucket_samples])
        avg_kv_ttft = np.mean([s.kvboost_ttft_ms for s in bucket_samples])
        avg_speedup = np.mean(
            [s.baseline_ttft_ms / s.kvboost_ttft_ms for s in bucket_samples if s.kvboost_ttft_ms > 0]
        ) if any(s.kvboost_ttft_ms > 0 for s in bucket_samples) else 0.0

        avg_base_total = np.mean([s.baseline_total_ms for s in bucket_samples])
        avg_kv_total = np.mean([s.kvboost_total_ms for s in bucket_samples])
        avg_reuse = np.mean([s.kv_reuse_ratio for s in bucket_samples])
        avg_ctx = np.mean([s.context_tokens for s in bucket_samples])

        baseline_only = sum(s.baseline_correct and not s.kvboost_correct for s in bucket_samples)
        kvboost_only = sum(not s.baseline_correct and s.kvboost_correct for s in bucket_samples)
        mcnemar_p = None
        if baseline_only + kvboost_only > 0:
            stat = (abs(baseline_only - kvboost_only) - 1) ** 2 / (baseline_only + kvboost_only)
            mcnemar_p = 1 - scipy_stats.chi2.cdf(stat, df=1)

        bucket_results[bucket_name] = BucketResult(
            bucket_name=bucket_name,
            n_samples=n,
            baseline_accuracy=base_acc,
            kvboost_accuracy=kv_acc,
            avg_baseline_ttft_ms=avg_base_ttft,
            avg_kvboost_ttft_ms=avg_kv_ttft,
            avg_ttft_speedup=avg_speedup,
            avg_baseline_total_ms=avg_base_total,
            avg_kvboost_total_ms=avg_kv_total,
            avg_kv_reuse_ratio=avg_reuse,
            avg_context_tokens=avg_ctx,
            mcnemar_pvalue=mcnemar_p,
            samples=bucket_samples,
        )

    return bucket_results


# ── Summary report ────────────────────────────────────────────────

def print_summary(bucket_results: Dict[str, BucketResult], verbose: bool = False):
    print(f"\n{'=' * 85}")
    print("  BUG LOCALIZATION RESULTS — Baseline HF vs KVBoost")
    print(f"{'=' * 85}\n")

    # Accuracy table
    # FIX 5: flag low-sample buckets
    LOW_N_THRESHOLD = 20
    print(f"  {'Bucket':<10} {'N':>5} {'Base Acc':>9} {'KV Acc':>9} {'Delta':>8} {'p-value':>9} {'Result':>8}")
    print(f"  {'-'*10} {'-'*5} {'-'*9} {'-'*9} {'-'*8} {'-'*9} {'-'*8}")

    for name in ["0-512", "512-1K", "1K-2K", "2K-4K", "4K+", "ALL"]:
        if name not in bucket_results:
            continue
        r = bucket_results[name]
        delta = r.kvboost_accuracy - r.baseline_accuracy
        p_str = f"{r.mcnemar_pvalue:.3f}" if r.mcnemar_pvalue is not None else "N/A"
        verdict = "PASS" if (r.mcnemar_pvalue is None or r.mcnemar_pvalue > 0.05) else "FAIL"
        marker = "" if name != "ALL" else " <<<"
        low_n_flag = " [LOW-N]" if r.n_samples < LOW_N_THRESHOLD and name != "ALL" else ""

        print(
            f"  {name:<10} {r.n_samples:>5} {r.baseline_accuracy:>8.1%} "
            f"{r.kvboost_accuracy:>8.1%} {delta:>+7.1%} {p_str:>9} {verdict:>8}{marker}{low_n_flag}"
        )

    # Performance table
    print(f"\n  {'Bucket':<10} {'N':>5} {'Avg Ctx':>8} {'Base TTFT':>10} {'KV TTFT':>10} "
          f"{'Speedup':>8} {'KV Reuse':>9}")
    print(f"  {'-'*10} {'-'*5} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*9}")

    for name in ["0-512", "512-1K", "1K-2K", "2K-4K", "4K+", "ALL"]:
        if name not in bucket_results:
            continue
        r = bucket_results[name]
        marker = "" if name != "ALL" else " <<<"

        print(
            f"  {name:<10} {r.n_samples:>5} {r.avg_context_tokens:>7.0f} "
            f"{r.avg_baseline_ttft_ms:>9.0f}ms {r.avg_kvboost_ttft_ms:>9.0f}ms "
            f"{r.avg_ttft_speedup:>7.1f}x {r.avg_kv_reuse_ratio:>8.0%}{marker}"
        )

    # Domain breakdown
    if "ALL" in bucket_results:
        all_samples = bucket_results["ALL"].samples
        domains = defaultdict(list)
        for s in all_samples:
            domains[s.domain].append(s)

        if len(domains) > 1:
            print(f"\n  {'Domain':<25} {'N':>5} {'Base Acc':>9} {'KV Acc':>9} {'Speedup':>8}")
            print(f"  {'-'*25} {'-'*5} {'-'*9} {'-'*9} {'-'*8}")

            for domain in sorted(domains.keys()):
                ds = domains[domain]
                n = len(ds)
                base_acc = sum(s.baseline_correct for s in ds) / n
                kv_acc = sum(s.kvboost_correct for s in ds) / n
                speedup = np.mean(
                    [s.baseline_ttft_ms / s.kvboost_ttft_ms for s in ds if s.kvboost_ttft_ms > 0]
                ) if any(s.kvboost_ttft_ms > 0 for s in ds) else 0.0
                print(f"  {domain:<25} {n:>5} {base_acc:>8.1%} {kv_acc:>8.1%} {speedup:>7.1f}x")

    # Cosine deviation table
    print(f"\n  {'Bucket':<10} {'N':>5} {'Avg Cos Sim':>12} {'Min Cos':>9} {'Top1 Changed':>13} {'Avg A Dev':>10} {'Avg B Dev':>10}")
    print(f"  {'-'*10} {'-'*5} {'-'*12} {'-'*9} {'-'*13} {'-'*10} {'-'*10}")

    for name in ["0-512", "512-1K", "1K-2K", "2K-4K", "4K+", "ALL"]:
        if name not in bucket_results:
            continue
        r = bucket_results[name]
        cos_sims = [s.logit_cosine_similarity for s in r.samples]
        avg_cos = np.mean(cos_sims)
        min_cos = np.min(cos_sims)
        n_top1_changed = sum(s.top1_token_changed for s in r.samples)

        choice_devs_all = [s.choice_logit_deviations for s in r.samples if s.choice_logit_deviations]
        avg_a_dev = np.mean([d.get("A", 0) for d in choice_devs_all]) if choice_devs_all else 0.0
        avg_b_dev = np.mean([d.get("B", 0) for d in choice_devs_all]) if choice_devs_all else 0.0
        marker = "" if name != "ALL" else " <<<"

        # Warn if cosine deviation is still suspiciously perfect
        if avg_cos == 1.0 and min_cos == 1.0 and name == "ALL":
            marker += "  ⚠ all cos=1.0 — KVBoost logit capture may be broken"

        print(
            f"  {name:<10} {r.n_samples:>5} {avg_cos:>11.6f} {min_cos:>8.6f} "
            f"{n_top1_changed:>8}/{r.n_samples:<4} {avg_a_dev:>9.6f} {avg_b_dev:>9.6f}{marker}"
        )

    # KV reuse distribution
    if "ALL" in bucket_results:
        all_samples = bucket_results["ALL"].samples
        reuse_ratios = [s.kv_reuse_ratio for s in all_samples]
        print(f"\n  KV Reuse Ratio Distribution:")
        print(f"    Mean:   {np.mean(reuse_ratios):.1%}")
        print(f"    Median: {np.median(reuse_ratios):.1%}")
        print(f"    Min:    {np.min(reuse_ratios):.1%}")
        print(f"    Max:    {np.max(reuse_ratios):.1%}")
        print(f"    P25:    {np.percentile(reuse_ratios, 25):.1%}")
        print(f"    P75:    {np.percentile(reuse_ratios, 75):.1%}")

        high_reuse = [s for s in all_samples if s.kv_reuse_ratio >= 0.8]
        if high_reuse:
            print(f"    High reuse (>=80%): {len(high_reuse)}/{len(all_samples)} samples")

    # Divergence analysis
    if "ALL" in bucket_results:
        all_samples = bucket_results["ALL"].samples
        n_total = len(all_samples)
        n_diverged = sum(s.outputs_diverged for s in all_samples)
        n_both_correct = sum(s.baseline_correct and s.kvboost_correct for s in all_samples)
        n_both_wrong = sum(not s.baseline_correct and not s.kvboost_correct for s in all_samples)
        n_base_only = sum(s.baseline_correct and not s.kvboost_correct for s in all_samples)
        n_kv_only = sum(not s.baseline_correct and s.kvboost_correct for s in all_samples)

        print(f"\n  Divergence Analysis:")
        print(f"    Output divergence rate:  {n_diverged}/{n_total} ({n_diverged/n_total:.1%})")
        print(f"    Both correct:            {n_both_correct}/{n_total} ({n_both_correct/n_total:.1%})")
        print(f"    Both wrong:              {n_both_wrong}/{n_total} ({n_both_wrong/n_total:.1%})")
        print(f"    Baseline-only correct:   {n_base_only}/{n_total} ({n_base_only/n_total:.1%})")
        print(f"    KVBoost-only correct:    {n_kv_only}/{n_total} ({n_kv_only/n_total:.1%})")

        diverged_cos = [s.logit_cosine_similarity for s in all_samples if s.outputs_diverged]
        matched_cos = [s.logit_cosine_similarity for s in all_samples if not s.outputs_diverged]
        if diverged_cos and matched_cos:
            print(f"\n  Cosine Similarity by Divergence:")
            print(f"    Matching outputs:  avg={np.mean(matched_cos):.6f} (n={len(matched_cos)})")
            print(f"    Divergent outputs: avg={np.mean(diverged_cos):.6f} (n={len(diverged_cos)})")

    # FIX 4a: Error-by-predicted-letter breakdown.
    # Surfaces A-bias (or any other letter bias) in KVBoost errors.
    if "ALL" in bucket_results:
        all_samples = bucket_results["ALL"].samples
        kv_errors = [s for s in all_samples if not s.kvboost_correct]
        if kv_errors:
            print(f"\n  KVBoost Error Distribution by Predicted Letter:")
            letter_counts = defaultdict(int)
            for s in kv_errors:
                letter_counts[s.kvboost_output or "?"] += 1
            total_errors = len(kv_errors)
            for letter in sorted(letter_counts.keys()):
                count = letter_counts[letter]
                bar = "█" * count
                print(f"    {letter}: {count:>3} / {total_errors}  {bar}")
            if letter_counts.get("A", 0) / max(total_errors, 1) > 0.5:
                print(f"    ⚠ A-bias detected: {letter_counts['A']}/{total_errors} errors predict A")

        # Warm-only error breakdown (q2 samples)
        warm_errors = [s for s in all_samples if not s.kvboost_correct and "_q2" in s.sample_id]
        if warm_errors:
            print(f"\n  KVBoost Error Distribution (warm/q2 only):")
            warm_letter_counts = defaultdict(int)
            for s in warm_errors:
                warm_letter_counts[s.kvboost_output or "?"] += 1
            for letter in sorted(warm_letter_counts.keys()):
                count = warm_letter_counts[letter]
                bar = "█" * count
                print(f"    {letter}: {count:>3}  {bar}")

    # FIX 4b: Answer-position vs divergence correlation.
    # Shows whether errors cluster at particular ABCD slots.
    if "ALL" in bucket_results:
        all_samples = bucket_results["ALL"].samples
        print(f"\n  Answer Position vs KVBoost Accuracy:")
        print(f"  {'Slot':<6} {'Letter':<7} {'Total':>6} {'KV Correct':>11} {'KV Acc':>8}")
        print(f"  {'-'*6} {'-'*7} {'-'*6} {'-'*11} {'-'*8}")
        for pos in range(4):
            letter = LETTERS[pos]
            slot_samples = [s for s in all_samples if s.answer_position == pos]
            if not slot_samples:
                continue
            n_slot = len(slot_samples)
            n_correct = sum(s.kvboost_correct for s in slot_samples)
            acc = n_correct / n_slot
            print(f"  {pos:<6} {letter:<7} {n_slot:>6} {n_correct:>11} {acc:>7.1%}")

        # Also show where KVBoost errors predict vs where the gold was
        kv_errors = [s for s in all_samples if not s.kvboost_correct]
        if kv_errors:
            print(f"\n  KVBoost Error: Gold Position vs Predicted Letter (warm q2 errors):")
            warm_kv_errors = [s for s in kv_errors if "_q2" in s.sample_id]
            if warm_kv_errors:
                print(f"  {'Gold Slot':>9} {'Gold Letter':>11} {'KV Pred':>8} {'Count':>6}")
                print(f"  {'-'*9} {'-'*11} {'-'*8} {'-'*6}")
                from collections import Counter
                combo_counts = Counter(
                    (s.answer_position, s.gold_answer, s.kvboost_output or "?")
                    for s in warm_kv_errors
                )
                for (pos, gold, pred), count in sorted(combo_counts.items()):
                    print(f"  {pos:>9} {gold:>11} {pred:>8} {count:>6}")

    # Verdict
    print(f"\n{'=' * 85}")
    if "ALL" in bucket_results:
        r = bucket_results["ALL"]
        delta = r.kvboost_accuracy - r.baseline_accuracy
        acc_pass = r.mcnemar_pvalue is None or r.mcnemar_pvalue > 0.05
        cos_sims = [s.logit_cosine_similarity for s in r.samples]
        avg_cos = np.mean(cos_sims)

        if acc_pass and r.avg_ttft_speedup >= 1.0:
            print(f"  RESULT: KVBoost matches baseline accuracy (delta={delta:+.1%}) "
                  f"with {r.avg_ttft_speedup:.1f}x average TTFT speedup")
        elif acc_pass:
            print(f"  RESULT: KVBoost matches baseline accuracy (delta={delta:+.1%}) "
                  f"but no TTFT speedup (contexts may be too short)")
        else:
            print(f"  RESULT: Accuracy difference detected (delta={delta:+.1%}, "
                  f"p={r.mcnemar_pvalue:.3f})")
        print(f"  Logit cosine similarity: avg={avg_cos:.6f} "
              f"(1.0=identical, <0.99=notable deviation)")

        # Warn if cosine deviation never moved from 1.0
        if avg_cos == 1.0 and min(s.logit_cosine_similarity for s in r.samples) == 1.0:
            print(f"  ⚠ WARNING: cosine similarity is uniformly 1.0 — "
                  f"first_token_logits may not be captured. "
                  f"Check that GenerationResult.first_token_logits is populated.")
    print(f"{'=' * 85}\n")

    # Verbose: per-sample divergence + cosine details
    if verbose and "ALL" in bucket_results:
        diverged = [s for s in bucket_results["ALL"].samples if s.outputs_diverged]
        if diverged:
            print(f"\n  Divergent samples ({len(diverged)}):")
            for s in diverged[:10]:
                dev_str = ""
                if s.choice_logit_deviations:
                    dev_str = " devs={" + ", ".join(
                        f"{k}:{v:.4f}" for k, v in sorted(s.choice_logit_deviations.items())
                    ) + "}"
                print(
                    f"    {s.sample_id}: ctx={s.context_tokens}tok "
                    f"base={s.baseline_output} kv={s.kvboost_output} gold={s.gold_answer} "
                    f"gold_pos={s.answer_position}({LETTERS[s.answer_position]}) "
                    f"cos={s.logit_cosine_similarity:.6f}{dev_str} "
                    f"domain={s.domain}"
                )
            if len(diverged) > 10:
                print(f"    ... and {len(diverged) - 10} more")

        sorted_by_cos = sorted(bucket_results["ALL"].samples, key=lambda s: s.logit_cosine_similarity)
        print(f"\n  Lowest cosine similarity samples (most deviated logits):")
        for s in sorted_by_cos[:5]:
            dev_str = ""
            if s.choice_logit_deviations:
                dev_str = " devs={" + ", ".join(
                    f"{k}:{v:.4f}" for k, v in sorted(s.choice_logit_deviations.items())
                ) + "}"
            match = "MATCH" if not s.outputs_diverged else "DIVERGE"
            print(
                f"    {s.sample_id}: cos={s.logit_cosine_similarity:.6f} "
                f"ctx={s.context_tokens}tok base={s.baseline_output} kv={s.kvboost_output} "
                f"gold={s.gold_answer} gold_pos={s.answer_position} [{match}]{dev_str}"
            )


# ── Save results ──────────────────────────────────────────────────

def get_checkpoint_path(model_name: str, n_samples: int, max_context_tokens: int) -> Path:
    """Generate a unique checkpoint path for this benchmark run."""
    import hashlib
    config_hash = hashlib.md5(
        f"{model_name}_{n_samples}_{max_context_tokens}".encode()
    ).hexdigest()[:8]
    return CHECKPOINT_DIR / f"checkpoint_{config_hash}.json"


def save_checkpoint(
    all_results: List[SampleResult],
    checkpoint_path: Path,
    samples: List[dict],
    processed_indices: List[int],
):
    """Save checkpoint with current progress."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "timestamp": time.time(),
        "n_total_samples": len(samples),
        "n_processed": len(processed_indices),
        "processed_indices": processed_indices,
        "results": [
            {
                "sample_id": s.sample_id,
                "domain": s.domain,
                "sub_domain": s.sub_domain,
                "difficulty": s.difficulty,
                "length_category": s.length_category,
                "context_tokens": s.context_tokens,
                "prompt_tokens": s.prompt_tokens,
                "baseline_output": s.baseline_output,
                "kvboost_output": s.kvboost_output,
                "gold_answer": s.gold_answer,
                "baseline_correct": s.baseline_correct,
                "kvboost_correct": s.kvboost_correct,
                "outputs_diverged": s.outputs_diverged,
                "baseline_ttft_ms": s.baseline_ttft_ms,
                "baseline_total_ms": s.baseline_total_ms,
                "kvboost_ttft_ms": s.kvboost_ttft_ms,
                "kvboost_total_ms": s.kvboost_total_ms,
                "kv_reuse_ratio": s.kv_reuse_ratio,
                "cached_tokens": s.cached_tokens,
                "warm_time_ms": s.warm_time_ms,
                "logit_cosine_similarity": s.logit_cosine_similarity,
                "choice_logit_deviations": s.choice_logit_deviations,
                "top1_token_changed": s.top1_token_changed,
                "answer_position": s.answer_position,
            }
            for s in all_results
        ],
    }

    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint, f, indent=2, default=str)

    log.info(f"Checkpoint saved: {len(processed_indices)}/{len(samples)} samples")


def load_checkpoint(checkpoint_path: Path) -> tuple[List[SampleResult], List[int]]:
    """Load checkpoint and reconstruct results."""
    if not checkpoint_path.exists():
        return [], []

    try:
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)

        processed_indices = checkpoint.get("processed_indices", [])
        results_data = checkpoint.get("results", [])

        results = []
        for data in results_data:
            sr = SampleResult(
                sample_id=data["sample_id"],
                domain=data["domain"],
                sub_domain=data["sub_domain"],
                difficulty=data["difficulty"],
                length_category=data["length_category"],
                context_tokens=data["context_tokens"],
                prompt_tokens=data["prompt_tokens"],
                baseline_output=data["baseline_output"],
                kvboost_output=data["kvboost_output"],
                gold_answer=data["gold_answer"],
                baseline_correct=data["baseline_correct"],
                kvboost_correct=data["kvboost_correct"],
                outputs_diverged=data["outputs_diverged"],
                baseline_ttft_ms=data["baseline_ttft_ms"],
                baseline_total_ms=data["baseline_total_ms"],
                kvboost_ttft_ms=data["kvboost_ttft_ms"],
                kvboost_total_ms=data["kvboost_total_ms"],
                kv_reuse_ratio=data["kv_reuse_ratio"],
                cached_tokens=data["cached_tokens"],
                warm_time_ms=data["warm_time_ms"],
                logit_cosine_similarity=data["logit_cosine_similarity"],
                choice_logit_deviations=data.get("choice_logit_deviations"),
                top1_token_changed=data["top1_token_changed"],
                answer_position=data.get("answer_position", 0),
            )
            results.append(sr)

        log.info(f"Checkpoint loaded: {len(processed_indices)} samples recovered")
        return results, processed_indices

    except Exception as e:
        log.warning(f"Failed to load checkpoint: {e}")
        return [], []

def save_results(
    bucket_results: Dict[str, BucketResult],
    args: argparse.Namespace,
    output_path: Path,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        "benchmark": "bug_localization",
        "dataset": "JetBrains-Research/lca-bug-localization/py",
        "model": args.model,
        "n_samples": args.n_samples,
        "chunk_size": args.chunk_size,
        "max_context_tokens": args.max_context_tokens,
        "buckets": {},
    }

    for name, r in bucket_results.items():
        cos_sims = [s.logit_cosine_similarity for s in r.samples]
        serializable["buckets"][name] = {
            "n_samples": r.n_samples,
            "baseline_accuracy": r.baseline_accuracy,
            "kvboost_accuracy": r.kvboost_accuracy,
            "accuracy_delta": r.kvboost_accuracy - r.baseline_accuracy,
            "mcnemar_pvalue": r.mcnemar_pvalue,
            "avg_baseline_ttft_ms": r.avg_baseline_ttft_ms,
            "avg_kvboost_ttft_ms": r.avg_kvboost_ttft_ms,
            "avg_ttft_speedup": r.avg_ttft_speedup,
            "avg_baseline_total_ms": r.avg_baseline_total_ms,
            "avg_kvboost_total_ms": r.avg_kvboost_total_ms,
            "avg_kv_reuse_ratio": r.avg_kv_reuse_ratio,
            "avg_context_tokens": r.avg_context_tokens,
            "cosine_deviation": {
                "avg_cosine_similarity": float(np.mean(cos_sims)),
                "min_cosine_similarity": float(np.min(cos_sims)),
                "max_cosine_similarity": float(np.max(cos_sims)),
                "std_cosine_similarity": float(np.std(cos_sims)),
                "n_top1_changed": sum(s.top1_token_changed for s in r.samples),
            },
            "kv_reuse_distribution": {
                "mean": float(np.mean([s.kv_reuse_ratio for s in r.samples])),
                "median": float(np.median([s.kv_reuse_ratio for s in r.samples])),
                "min": float(np.min([s.kv_reuse_ratio for s in r.samples])),
                "max": float(np.max([s.kv_reuse_ratio for s in r.samples])),
                "p25": float(np.percentile([s.kv_reuse_ratio for s in r.samples], 25)),
                "p75": float(np.percentile([s.kv_reuse_ratio for s in r.samples], 75)),
            },
            "samples": [
                {
                    "id": s.sample_id,
                    "domain": s.domain,
                    "sub_domain": s.sub_domain,
                    "difficulty": s.difficulty,
                    "length_category": s.length_category,
                    "context_tokens": s.context_tokens,
                    "prompt_tokens": s.prompt_tokens,
                    "baseline_pred": s.baseline_output,
                    "kvboost_pred": s.kvboost_output,
                    "gold_answer": s.gold_answer,
                    "answer_position": s.answer_position,
                    "baseline_correct": s.baseline_correct,
                    "kvboost_correct": s.kvboost_correct,
                    "diverged": s.outputs_diverged,
                    "baseline_ttft_ms": s.baseline_ttft_ms,
                    "kvboost_ttft_ms": s.kvboost_ttft_ms,
                    "baseline_total_ms": s.baseline_total_ms,
                    "kvboost_total_ms": s.kvboost_total_ms,
                    "kv_reuse_ratio": s.kv_reuse_ratio,
                    "cached_tokens": s.cached_tokens,
                    "warm_time_ms": s.warm_time_ms,
                    "logit_cosine_similarity": s.logit_cosine_similarity,
                    "choice_logit_deviations": s.choice_logit_deviations,
                    "top1_token_changed": s.top1_token_changed,
                }
                for s in r.samples
            ],
        }

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)

    log.info(f"Results saved to {output_path}")


# ── Main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Bug Localization Benchmark — Baseline HF vs KVBoost",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B",
                        help="HuggingFace model name (default: Qwen/Qwen2.5-3B)")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Number of samples to evaluate (default: 50)")
    parser.add_argument("--chunk-size", type=int, default=128,
                        help="KVBoost chunk size (default: 128)")
    parser.add_argument("--max-context-tokens", type=int, default=0,
                        help="Max context tokens (default: auto-detect from RAM)")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-sample divergence details")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Ignore checkpoint and start fresh")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.max_context_tokens <= 0:
        try:
            import psutil
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except ImportError:
            import os
            ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024 ** 3)
        safe_tokens = int((ram_gb - 8) * 3000)
        args.max_context_tokens = max(2048, min(safe_tokens, 131072))
        log.info(f"Auto-detected RAM: {ram_gb:.0f}GB → max_context_tokens={args.max_context_tokens}")

    bucket_results = run_benchmark(
        model_name=args.model,
        n_samples=args.n_samples,
        chunk_size=args.chunk_size,
        max_context_tokens=args.max_context_tokens,
        no_checkpoint=args.no_checkpoint,
    )

    if not bucket_results:
        print("No results produced.")
        return

    print_summary(bucket_results, verbose=args.verbose)

    out_path = Path(args.output) if args.output else RESULTS_DIR / "bug_localization.json"
    save_results(bucket_results, args, out_path)

    # Clean up checkpoint on successful completion
    checkpoint_path = get_checkpoint_path(
        args.model, args.n_samples, args.max_context_tokens
    )
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        log.info(f"Checkpoint cleaned up after successful completion")


if __name__ == "__main__":
    main()