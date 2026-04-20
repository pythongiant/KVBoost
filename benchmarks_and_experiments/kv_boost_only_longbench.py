#!/usr/bin/env python3
"""
KVBoost Benchmark — KVBoost-only on Code Bug Localization
==========================================================
Evaluates KVBoost prefix-caching performance on
JetBrains-Research/lca-bug-localization (Python split).

Dataset: Each sample has a code diff (100-3K tokens) + issue description.
Task:    Given the diff context and bug report, pick which file was changed
         from 4 multiple-choice options (1 correct + 3 distractors).

Methodology:
  1. Load bug localization samples, convert to MC format
  2. Paired queries per diff:
     a. Q1 (cold): populates the KVBoost cache with diff prefix KVs
     b. Q2 (warm): reuses cached prefix, different distractors + phrasing
  3. Measure: accuracy, TTFT (cold vs warm), KV reuse ratio
  4. McNemar's test for cold vs warm accuracy difference

Usage:
  python long_bench_kvboost.py                        # 50 samples, Qwen2.5-3B
  python long_bench_kvboost.py --n-samples 200
  python long_bench_kvboost.py --model meta-llama/Llama-3.2-3B
  python long_bench_kvboost.py --verbose
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
from scipy.stats import binomtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("longbench_kvboost")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
LETTERS = "ABCD"
CHECKPOINT_DIR = Path(__file__).resolve().parent / ".checkpoints"
CHECKPOINT_INTERVAL = 4
MCNEMAR_EXACT_THRESHOLD = 25


# ── Statistical helpers ───────────────────────────────────────────

def mcnemar_pvalue(cold_only: int, warm_only: int) -> Optional[float]:
    """McNemar's test p-value for paired cold/warm correctness."""
    n = cold_only + warm_only
    if n == 0:
        return None
    if n < MCNEMAR_EXACT_THRESHOLD:
        return float(binomtest(min(cold_only, warm_only), n=n, p=0.5).pvalue)
    stat = (abs(cold_only - warm_only) - 1) ** 2 / n
    return float(1 - scipy_stats.chi2.cdf(stat, df=1))


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

    predicted: str
    gold_answer: str
    correct: bool

    ttft_ms: float = 0.0
    total_ms: float = 0.0
    kv_reuse_ratio: float = 0.0
    cached_tokens: int = 0

    is_warm: bool = False       # True = q2 (cache warm), False = q1 (cold)
    answer_position: int = 0    # Which ABCD slot holds the gold answer


@dataclass
class BucketResult:
    bucket_name: str
    n_samples: int

    cold_accuracy: float
    warm_accuracy: float
    overall_accuracy: float

    avg_cold_ttft_ms: float = 0.0
    avg_warm_ttft_ms: float = 0.0
    avg_ttft_speedup: float = 0.0

    avg_cold_total_ms: float = 0.0
    avg_warm_total_ms: float = 0.0

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
    Load JetBrains-Research/lca-bug-localization (py) and convert to MC format.

    Each diff produces two queries:
      q1 (cold) — populates the KVBoost prefix cache with diff KV blocks
      q2 (warm) — reuses cached prefix, different distractors and phrasing

    q2's correct answer is forced into a different ABCD slot than q1's to
    eliminate warm-cache positional bias toward whichever slot came first in q1.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    log.info("Loading JetBrains-Research/lca-bug-localization (py/train) ...")
    ds = load_dataset("JetBrains-Research/lca-bug-localization", "py", split="train")
    log.info(f"  Total samples in dataset: {len(ds)}")

    tok = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    log.info(f"  Pre-filtering contexts to <={max_context_tokens} tokens ...")

    raw_samples = []
    all_files: set = set()
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
        body_preview = (issue_body[:1000] + "...") if len(issue_body) > 1000 else issue_body
        question = f"{issue_title}\n{body_preview}" if body_preview else issue_title

        raw_samples.append({
            "diff": diff,
            "correct_file": correct_file,
            "all_changed_files": changed_files,
            "question": question,
            "repo": f"{row.get('repo_owner', '')}/{row.get('repo_name', '')}",
            "ctx_tokens": ctx_tokens,
            "changed_files_count": row.get("changed_files_count", len(changed_files)),
        })

    log.info(
        f"  After filtering: {len(raw_samples)} eligible, "
        f"{skipped_too_long} too long, {skipped_no_diff} no diff/files"
    )

    if not raw_samples:
        log.error(f"  No samples fit within {max_context_tokens} tokens!")
        return []

    all_files_list = sorted(all_files)
    rng = np.random.RandomState(42)

    def _bucket_key(t: int) -> str:
        if t < 512:    return "0-512"
        elif t < 1024: return "512-1K"
        elif t < 2048: return "1K-2K"
        elif t < 4096: return "2K-4K"
        else:          return "4K+"

    bucket_pools: Dict[str, list] = defaultdict(list)
    for s in raw_samples:
        bucket_pools[_bucket_key(s["ctx_tokens"])].append(s)
    for bk in bucket_pools:
        rng.shuffle(bucket_pools[bk])

    n_buckets = len(bucket_pools)
    per_bucket = n_samples // n_buckets if n_buckets > 0 else n_samples

    selected, overflow = [], []
    for bk in ["0-512", "512-1K", "1K-2K", "2K-4K", "4K+"]:
        pool = bucket_pools.get(bk, [])
        take = min(per_bucket, len(pool))
        selected.extend(pool[:take])
        overflow.extend(pool[take:])
        if take < per_bucket:
            log.info(f"  Bucket {bk}: only {take}/{per_bucket} available")

    remaining = n_samples - len(selected)
    if remaining > 0 and overflow:
        rng.shuffle(overflow)
        selected.extend(overflow[:remaining])

    log.info(
        f"  Stratified sampling: {len(selected)} diffs → {len(selected)*2} queries "
        f"across {n_buckets} token-length buckets"
    )

    samples = []
    for idx, raw in enumerate(selected):
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

        # Q1 (cold) — populates the KVBoost prefix cache
        d1 = distractor_pool[:3]
        choices_1 = d1 + [correct_file]
        rng.shuffle(choices_1)
        q1_pos = choices_1.index(correct_file)

        samples.append({
            **base_info,
            "id": f"bug_{idx}_q1",
            "question": f"Bug report: {raw['question']}\n\nWhich file was modified to fix this bug?",
            "choices": choices_1,
            "answer": LETTERS[q1_pos],
            "answer_position": q1_pos,
            "pair_group": idx,
            "is_warm": False,
        })

        # Q2 (warm) — guarantee different answer slot from q1 to avoid positional bias
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
        assert q2_pos != q1_pos, f"Failed to deplace q2 answer: q1={q1_pos}, q2={q2_pos}"

        samples.append({
            **base_info,
            "id": f"bug_{idx}_q2",
            "question": f"Issue: {raw['question']}\n\nIdentify the file that contains the fix for this issue.",
            "choices": choices_2,
            "answer": LETTERS[q2_pos],
            "answer_position": q2_pos,
            "pair_group": idx,
            "is_warm": True,
        })

    samples.sort(key=lambda s: s["context_tokens"])

    ctx_lens = [s["context_tokens"] for s in samples]
    log.info(
        f"  Token range: {min(ctx_lens)}–{max(ctx_lens)} "
        f"(median={int(np.median(ctx_lens))}, mean={int(np.mean(ctx_lens))})"
    )
    log.info(f"  q2 answer positions debiased (guaranteed ≠ q1 slot)")
    return samples


# ── Prompt formatting ─────────────────────────────────────────────

def format_prompt_prefix(context: str) -> str:
    """Cacheable prefix: identical across q1/q2, so KVBoost can reuse its KVs."""
    return (
        "Read the following code diff and answer the question.\n\n"
        f"--- BEGIN DIFF ---\n{context}\n--- END DIFF ---\n\n"
    )


def format_prompt_suffix(question: str, choices: List[str]) -> str:
    """Non-cacheable suffix: always fresh-prefilled to avoid q1→q2 choice bleed."""
    parts = [f"{question}\n"]
    for i, choice in enumerate(choices):
        parts.append(f"{LETTERS[i]}. {choice}")
    parts.append("\nAnswer with just the letter (A, B, C, or D):")
    return "\n".join(parts)


def format_prompt(context: str, question: str, choices: List[str]) -> str:
    return format_prompt_prefix(context) + format_prompt_suffix(question, choices)


# ── Answer extraction ─────────────────────────────────────────────

import re


def extract_letter_answer(text: str) -> Optional[str]:
    text = text.strip()
    m = re.match(r"^\(?([A-Da-d])\)?[\.\)\s:]?", text)
    if m:
        return m.group(1).upper()
    for ch in text:
        if ch.upper() in LETTERS:
            return ch.upper()
    return None


# ── KVBoost runner ────────────────────────────────────────────────

class KVBoostRunner:
    """
    Runs KVBoost inference for cold (q1) and warm (q2) queries.

    Q1 resets the cache and runs CHUNK_KV_REUSE, storing only the
    cacheable diff prefix (not the per-query suffix) so q2 can reuse
    the diff KVs without picking up q1's choice-position state.

    Q2 runs CHUNK_KV_REUSE with the warm cache — the diff prefix is
    served from cache, only the suffix is freshly prefilled.
    """

    def __init__(
        self,
        model_name: str,
        max_cache_bytes: int,
        chunk_size: int = 128,
        recompute_strategy: str = "selective",
        chunk_boundary_window: int = 0,
        overlap_k: int = 0,
        sink_tokens: int = 0,
        recency_window_chunks: int = 8,
    ):
        from kvboost import KVBoost, GenerationMode

        self.GenerationMode = GenerationMode

        log.info(
            f"Loading KVBoost model: {model_name} "
            f"(chunk_size={chunk_size}, strategy={recompute_strategy}, "
            f"boundary_window={chunk_boundary_window}, overlap_k={overlap_k}, "
            f"sink_tokens={sink_tokens}, "
            f"max_cache_bytes={max_cache_bytes/1e9:.2f}GB, "
            f"recency_window={recency_window_chunks} chunks)"
        )
        self.engine = KVBoost.from_pretrained(
            model_name,
            max_cache_bytes=max_cache_bytes,
            recency_window_chunks=recency_window_chunks,
            chunk_size=chunk_size,
            recompute_overlap=16,
            recompute_strategy=recompute_strategy,
            chunk_boundary_window=chunk_boundary_window,
            overlap_k=overlap_k,
            sink_tokens=sink_tokens,
        )
        log.info(f"  Ready on {self.engine.device}")

    def reset_cache(self):
        self.engine.reset_cache()

    def count_tokens(self, text: str) -> int:
        return len(self.engine.tokenizer.encode(text, add_special_tokens=True))

    def cacheable_prefix_len(self, prefix: str) -> int:
        """Token count of the diff-only prefix — caps what gets cached."""
        return len(self.engine.tokenizer.encode(prefix, add_special_tokens=True))

    def _run(
        self,
        prompt: str,
        max_tokens: int = 16,
        cacheable_prefix_len: Optional[int] = None,
    ) -> dict:
        start = time.perf_counter()
        result = self.engine.generate(
            prompt,
            max_new_tokens=max_tokens,
            mode=self.GenerationMode.CHUNK_KV_REUSE,
            do_sample=False,
            temperature=1.0,
            cacheable_prefix_len=cacheable_prefix_len,
        )
        total_ms = (time.perf_counter() - start) * 1000
        return {
            "output": result.output_text,
            "ttft_ms": getattr(result, "ttft_ms", total_ms),
            "total_ms": total_ms,
            "kv_reuse_ratio": getattr(result, "kv_reuse_ratio", 0.0),
            "cached_tokens": getattr(result, "cached_tokens", 0),
        }

    def run_cold(self, prompt: str, max_tokens: int = 16, cacheable_prefix_len: Optional[int] = None) -> dict:
        """Cold run: cache is empty, diff prefix is stored for q2 to reuse."""
        return self._run(prompt, max_tokens=max_tokens, cacheable_prefix_len=cacheable_prefix_len)

    def run_warm(self, prompt: str, max_tokens: int = 16, cacheable_prefix_len: Optional[int] = None) -> dict:
        """Warm run: diff prefix already in cache, only suffix is prefilled."""
        return self._run(prompt, max_tokens=max_tokens, cacheable_prefix_len=cacheable_prefix_len)


# ── Checkpoint helpers ────────────────────────────────────────────

def get_checkpoint_path(
    model_name: str,
    n_samples: int,
    max_context_tokens: int,
    recompute_strategy: str = "selective",
) -> Path:
    import hashlib
    h = hashlib.md5(
        f"{model_name}_{n_samples}_{max_context_tokens}_{recompute_strategy}".encode()
    ).hexdigest()[:8]
    return CHECKPOINT_DIR / f"kvboost_checkpoint_{h}.json"


def save_checkpoint(
    all_results: List[SampleResult],
    checkpoint_path: Path,
    samples: List[dict],
    processed_indices: List[int],
    model_name: str = "",
    max_context_tokens: int = 0,
):
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "timestamp": time.time(),
        "model_name": model_name,
        "max_context_tokens": max_context_tokens,
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
                "predicted": s.predicted,
                "gold_answer": s.gold_answer,
                "correct": s.correct,
                "ttft_ms": s.ttft_ms,
                "total_ms": s.total_ms,
                "kv_reuse_ratio": s.kv_reuse_ratio,
                "cached_tokens": s.cached_tokens,
                "is_warm": s.is_warm,
                "answer_position": s.answer_position,
            }
            for s in all_results
        ],
    }
    temp_path = checkpoint_path.with_suffix(".tmp")
    try:
        with open(temp_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)
        temp_path.replace(checkpoint_path)
        log.info(f"Checkpoint saved: {len(processed_indices)}/{len(samples)} samples")
    except Exception as e:
        log.error(f"Failed to save checkpoint: {e}")
        if temp_path.exists():
            temp_path.unlink()
        raise


def load_checkpoint_by_sample_id(
    checkpoint_path: Path,
    current_samples: List[dict],
) -> tuple[List[SampleResult], List[int]]:
    if not checkpoint_path.exists():
        return [], []
    try:
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        id_to_result = {
            r["sample_id"]: r
            for r in checkpoint.get("results", [])
            if r.get("sample_id")
        }
        matched_results, matched_indices = [], []
        for idx, sample in enumerate(current_samples):
            sid = sample.get("id")
            if sid in id_to_result:
                d = id_to_result[sid]
                try:
                    matched_results.append(SampleResult(
                        sample_id=d["sample_id"],
                        domain=d["domain"],
                        sub_domain=d["sub_domain"],
                        difficulty=d["difficulty"],
                        length_category=d["length_category"],
                        context_tokens=d["context_tokens"],
                        prompt_tokens=d["prompt_tokens"],
                        predicted=d["predicted"],
                        gold_answer=d["gold_answer"],
                        correct=d["correct"],
                        ttft_ms=d["ttft_ms"],
                        total_ms=d["total_ms"],
                        kv_reuse_ratio=d["kv_reuse_ratio"],
                        cached_tokens=d["cached_tokens"],
                        is_warm=d["is_warm"],
                        answer_position=d.get("answer_position", 0),
                    ))
                    matched_indices.append(idx)
                except KeyError as ke:
                    log.warning(f"Skipping {sid}: missing key {ke}")
        if matched_results:
            log.info(
                f"Checkpoint loaded: {len(matched_results)}/{len(current_samples)} "
                f"samples matched by ID"
            )
        return matched_results, matched_indices
    except json.JSONDecodeError as e:
        log.warning(f"Checkpoint corrupted: {e}. Starting fresh.")
        return [], []
    except Exception as e:
        log.warning(f"Failed to load checkpoint: {e}. Starting fresh.")
        return [], []


# ── Bucket helper ─────────────────────────────────────────────────

def get_bucket(n_tokens: int) -> str:
    if n_tokens < 512:    return "0-512"
    elif n_tokens < 1024: return "512-1K"
    elif n_tokens < 2048: return "1K-2K"
    elif n_tokens < 4096: return "2K-4K"
    else:                 return "4K+"


# ── Benchmark runner ──────────────────────────────────────────────

def run_benchmark(
    model_name: str,
    n_samples: int,
    max_cache_bytes: int,
    chunk_size: int = 128,
    max_context_tokens: int = 8192,
    recompute_strategy: str = "selective",
    chunk_boundary_window: int = 0,
    overlap_k: int = 0,
    sink_tokens: int = 0,
    recency_window_chunks: int = 8,
    no_checkpoint: bool = False,
) -> Dict[str, BucketResult]:
    print(f"\n{'=' * 75}")
    print(f"  BUG LOCALIZATION BENCHMARK — KVBoost")
    print(f"{'=' * 75}")
    print(f"  Dataset:            JetBrains-Research/lca-bug-localization (py)")
    print(f"  Model:              {model_name}")
    print(f"  Samples (pairs):    {n_samples} diffs → {n_samples*2} queries")
    print(f"  Max context tokens: {max_context_tokens}")
    print(f"  Chunk size:         {chunk_size}")
    print(f"  Recompute strategy: {recompute_strategy}")
    print(f"  Boundary window:    {chunk_boundary_window}")
    print(f"  Overlap K:          {overlap_k}")
    print(f"  Sink tokens:        {sink_tokens}")
    print(f"  Max cache bytes:    {max_cache_bytes/1e9:.2f} GB (hard budget)")
    print(f"  Recency window:     {recency_window_chunks} chunks (pinned)")
    print(f"{'=' * 75}\n")

    samples = load_bug_localization(n_samples, max_context_tokens, model_name)
    if not samples:
        log.error("No samples loaded.")
        return {}

    runner = KVBoostRunner(
        model_name,
        max_cache_bytes=max_cache_bytes,
        chunk_size=chunk_size,
        recompute_strategy=recompute_strategy,
        chunk_boundary_window=chunk_boundary_window,
        overlap_k=overlap_k,
        sink_tokens=sink_tokens,
        recency_window_chunks=recency_window_chunks,
    )

    pair_groups: Dict[int, List[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        pair_groups[sample["pair_group"]].append(idx)

    n_pairs = sum(1 for v in pair_groups.values() if len(v) > 1)
    print(f"  {n_pairs} cold→warm pairs for KV reuse testing")
    print(f"  Q1 populates cache, Q2 reuses cached diff prefix")
    print(f"  q2 answer slots debiased (guaranteed ≠ q1 slot)\n")

    # Dataset distribution
    bucket_counts: Dict[str, list] = defaultdict(list)
    for s in samples:
        bucket_counts[get_bucket(s["context_tokens"])].append(s)
    print(f"  Dataset Bucket Distribution:")
    print(f"  {'Bucket':<10} {'Queries':>8} {'Min':>6} {'Max':>6} {'Median':>7} {'Mean':>7}")
    print(f"  {'-'*10} {'-'*8} {'-'*6} {'-'*6} {'-'*7} {'-'*7}")
    for bk in ["0-512", "512-1K", "1K-2K", "2K-4K", "4K+"]:
        bd = bucket_counts.get(bk, [])
        if not bd:
            continue
        toks = [s["context_tokens"] for s in bd]
        print(
            f"  {bk:<10} {len(bd):>8} {min(toks):>6} {max(toks):>6} "
            f"{int(np.median(toks)):>7} {int(np.mean(toks)):>7}"
        )
    print()

    # Checkpoint
    checkpoint_path = get_checkpoint_path(
        model_name, n_samples, max_context_tokens, recompute_strategy
    )
    if not no_checkpoint:
        all_results, processed_indices = load_checkpoint_by_sample_id(checkpoint_path, samples)
    else:
        all_results, processed_indices = [], []
    processed_set = set(processed_indices)

    if processed_set:
        print(f"  Resuming from checkpoint: {len(processed_indices)} samples done")
        print(f"  (use --no-checkpoint to start fresh)\n")

    results_by_bucket: Dict[str, List[SampleResult]] = defaultdict(list)
    for sr in all_results:
        results_by_bucket[get_bucket(sr.context_tokens)].append(sr)

    ordered_indices = []
    for gid in sorted(pair_groups.keys()):
        ordered_indices.extend(pair_groups[gid])

    import gc
    total_start = time.perf_counter()

    from tqdm import tqdm
    for i in tqdm(ordered_indices, desc="Benchmarking", unit="query"):
        if i in processed_set:
            continue
        if i < 0 or i >= len(samples):
            log.warning(f"Invalid index {i} — skipping")
            continue

        sample = samples[i]
        context = sample["context"]
        context_tokens = sample.get("context_tokens", runner.count_tokens(context))
        is_warm = sample["is_warm"]

        # Reset cache at the start of each pair so q1 always starts cold
        if not is_warm:
            runner.reset_cache()

        prefix_text = format_prompt_prefix(context)
        suffix_text = format_prompt_suffix(sample["question"], sample["choices"])
        prompt = prefix_text + suffix_text
        prefix_len = runner.cacheable_prefix_len(prefix_text)
        prompt_tokens = runner.count_tokens(prompt)
        bucket = get_bucket(context_tokens)
        query_type = "WARM" if is_warm else "COLD"

        try:
            if is_warm:
                result = runner.run_warm(prompt, max_tokens=16, cacheable_prefix_len=prefix_len)
            else:
                result = runner.run_cold(prompt, max_tokens=16, cacheable_prefix_len=prefix_len)
        except RuntimeError as e:
            if "Invalid buffer size" in str(e) or "out of memory" in str(e).lower():
                import traceback
                log.warning(
                    f"OOM at {context_tokens} tokens [{query_type}] — skipping pair\n"
                    f"  exception: {type(e).__name__}: {e}\n"
                    f"  traceback:\n{traceback.format_exc()}"
                )
                runner.reset_cache()
                gc.collect()
                continue
            raise

        pred = extract_letter_answer(result["output"])
        correct = pred == sample["answer"]

        sr = SampleResult(
            sample_id=sample["id"],
            domain=sample["domain"],
            sub_domain=sample["sub_domain"],
            difficulty=sample["difficulty"],
            length_category=sample["length"],
            context_tokens=context_tokens,
            prompt_tokens=prompt_tokens,
            predicted=pred or "",
            gold_answer=sample["answer"],
            correct=correct,
            ttft_ms=result["ttft_ms"],
            total_ms=result["total_ms"],
            kv_reuse_ratio=result["kv_reuse_ratio"],
            cached_tokens=result["cached_tokens"],
            is_warm=is_warm,
            answer_position=sample.get("answer_position", LETTERS.index(sample["answer"])),
        )

        results_by_bucket[bucket].append(sr)
        all_results.append(sr)
        processed_indices.append(i)
        processed_set.add(i)

        if len(all_results) % CHECKPOINT_INTERVAL == 0:
            try:
                save_checkpoint(
                    all_results, checkpoint_path, samples, processed_indices,
                    model_name=model_name, max_context_tokens=max_context_tokens,
                )
            except Exception as e:
                log.error(f"Checkpoint save failed: {e}")

        if len(all_results) % 4 == 0:
            n_done = len(all_results)
            cold_srs = [s for s in all_results if not s.is_warm]
            warm_srs = [s for s in all_results if s.is_warm]
            cold_acc = sum(s.correct for s in cold_srs) / len(cold_srs) if cold_srs else 0
            warm_acc = sum(s.correct for s in warm_srs) / len(warm_srs) if warm_srs else 0
            avg_reuse = np.mean([s.kv_reuse_ratio for s in all_results])
            print(
                f"  [{n_done}/{len(samples)}] [{query_type}] ctx={context_tokens:>5}tok "
                f"TTFT={result['ttft_ms']:>7.0f}ms reuse={result['kv_reuse_ratio']:.0%} "
                f"avg_reuse={avg_reuse:.0%} "
                f"acc: cold={cold_acc:.0%} warm={warm_acc:.0%}"
            )

    total_time = time.perf_counter() - total_start
    log.info(f"Benchmark completed in {total_time:.1f}s")

    if all_results and len(all_results) % CHECKPOINT_INTERVAL != 0:
        try:
            save_checkpoint(
                all_results, checkpoint_path, samples, processed_indices,
                model_name=model_name, max_context_tokens=max_context_tokens,
            )
        except Exception as e:
            log.error(f"Final checkpoint save failed: {e}")

    # ── Compute bucket results ──────────────────────────────────────
    results_by_bucket["ALL"] = all_results
    bucket_results: Dict[str, BucketResult] = {}

    for bucket_name, bsrs in sorted(results_by_bucket.items()):
        if not bsrs:
            continue
        cold_srs = [s for s in bsrs if not s.is_warm]
        warm_srs = [s for s in bsrs if s.is_warm]

        cold_acc  = sum(s.correct for s in cold_srs) / len(cold_srs) if cold_srs else 0.0
        warm_acc  = sum(s.correct for s in warm_srs) / len(warm_srs) if warm_srs else 0.0
        overall   = sum(s.correct for s in bsrs) / len(bsrs)

        avg_cold_ttft  = np.mean([s.ttft_ms for s in cold_srs]) if cold_srs else 0.0
        avg_warm_ttft  = np.mean([s.ttft_ms for s in warm_srs]) if warm_srs else 0.0
        speedup = avg_cold_ttft / avg_warm_ttft if avg_warm_ttft > 0 else 0.0

        avg_cold_total = np.mean([s.total_ms for s in cold_srs]) if cold_srs else 0.0
        avg_warm_total = np.mean([s.total_ms for s in warm_srs]) if warm_srs else 0.0

        avg_reuse = np.mean([s.kv_reuse_ratio for s in bsrs])
        avg_ctx   = np.mean([s.context_tokens for s in bsrs])

        cold_only = max(0, round(len(cold_srs) * cold_acc) - round(len(warm_srs) * warm_acc))
        warm_only = max(0, round(len(warm_srs) * warm_acc) - round(len(cold_srs) * cold_acc))
        mcnemar_p = mcnemar_pvalue(cold_only, warm_only)

        bucket_results[bucket_name] = BucketResult(
            bucket_name=bucket_name,
            n_samples=len(bsrs),
            cold_accuracy=cold_acc,
            warm_accuracy=warm_acc,
            overall_accuracy=overall,
            avg_cold_ttft_ms=avg_cold_ttft,
            avg_warm_ttft_ms=avg_warm_ttft,
            avg_ttft_speedup=speedup,
            avg_cold_total_ms=avg_cold_total,
            avg_warm_total_ms=avg_warm_total,
            avg_kv_reuse_ratio=avg_reuse,
            avg_context_tokens=avg_ctx,
            mcnemar_pvalue=mcnemar_p,
            samples=bsrs,
        )

    return bucket_results


# ── Summary report ────────────────────────────────────────────────

def print_summary(bucket_results: Dict[str, BucketResult], verbose: bool = False):
    LOW_N = 20

    print(f"\n{'=' * 85}")
    print(f"  BUG LOCALIZATION RESULTS — KVBoost")
    print(f"{'=' * 85}\n")

    # Accuracy table
    print(
        f"  {'Bucket':<10} {'N':>5} {'Cold Acc':>9} {'Warm Acc':>9} "
        f"{'Delta':>8} {'p-value':>9} {'Result':>8}"
    )
    print(f"  {'-'*10} {'-'*5} {'-'*9} {'-'*9} {'-'*8} {'-'*9} {'-'*8}")

    for name in ["0-512", "512-1K", "1K-2K", "2K-4K", "4K+", "ALL"]:
        if name not in bucket_results:
            continue
        r = bucket_results[name]
        delta = r.warm_accuracy - r.cold_accuracy
        p_str = f"{r.mcnemar_pvalue:.3f}" if r.mcnemar_pvalue is not None else "N/A"
        verdict = "PASS" if (r.mcnemar_pvalue is None or r.mcnemar_pvalue > 0.05) else "FAIL"
        marker = " <<<" if name == "ALL" else ""
        low_n_flag = " [LOW-N]" if r.n_samples < LOW_N and name != "ALL" else ""
        print(
            f"  {name:<10} {r.n_samples:>5} {r.cold_accuracy:>8.1%} "
            f"{r.warm_accuracy:>8.1%} {delta:>+7.1%} {p_str:>9} {verdict:>8}{marker}{low_n_flag}"
        )

    # TTFT + reuse table
    print(
        f"\n  {'Bucket':<10} {'N':>5} {'Avg Ctx':>8} {'Cold TTFT':>10} "
        f"{'Warm TTFT':>10} {'Speedup':>8} {'KV Reuse':>9}"
    )
    print(f"  {'-'*10} {'-'*5} {'-'*8} {'-'*10} {'-'*10} {'-'*8} {'-'*9}")

    for name in ["0-512", "512-1K", "1K-2K", "2K-4K", "4K+", "ALL"]:
        if name not in bucket_results:
            continue
        r = bucket_results[name]
        marker = " <<<" if name == "ALL" else ""
        print(
            f"  {name:<10} {r.n_samples:>5} {r.avg_context_tokens:>7.0f} "
            f"{r.avg_cold_ttft_ms:>9.0f}ms {r.avg_warm_ttft_ms:>9.0f}ms "
            f"{r.avg_ttft_speedup:>7.1f}x {r.avg_kv_reuse_ratio:>8.0%}{marker}"
        )

    # KV reuse distribution
    if "ALL" in bucket_results:
        all_srs = bucket_results["ALL"].samples
        reuse_ratios = [s.kv_reuse_ratio for s in all_srs]
        print(f"\n  KV Reuse Ratio Distribution:")
        stats_fn = [
            ("Mean",   np.mean),
            ("Median", np.median),
            ("P25",    lambda x: np.percentile(x, 25)),
            ("P75",    lambda x: np.percentile(x, 75)),
            ("Min",    np.min),
            ("Max",    np.max),
        ]
        for label, fn in stats_fn:
            print(f"    {label:<8} {fn(reuse_ratios):.1%}")
        high = [s for s in all_srs if s.kv_reuse_ratio >= 0.8]
        if high:
            print(f"    High reuse (>=80%): {len(high)}/{len(all_srs)} samples")

    # Query-type breakdown
    if "ALL" in bucket_results:
        all_srs = bucket_results["ALL"].samples
        cold_srs = [s for s in all_srs if not s.is_warm]
        warm_srs = [s for s in all_srs if s.is_warm]
        print(f"\n  Query-Type Breakdown:")
        print(f"  {'Type':<10} {'N':>5} {'Accuracy':>9} {'Avg TTFT':>10} {'Avg Reuse':>10}")
        print(f"  {'-'*10} {'-'*5} {'-'*9} {'-'*10} {'-'*10}")
        for label, srs in [("Cold (q1)", cold_srs), ("Warm (q2)", warm_srs)]:
            if not srs:
                continue
            acc = sum(s.correct for s in srs) / len(srs)
            avg_ttft = np.mean([s.ttft_ms for s in srs])
            avg_reuse = np.mean([s.kv_reuse_ratio for s in srs])
            print(f"  {label:<10} {len(srs):>5} {acc:>8.1%} {avg_ttft:>9.0f}ms {avg_reuse:>9.0%}")

    # Conditional accuracy: KVBoost in its intended operating regime
    if "ALL" in bucket_results:
        all_srs = bucket_results["ALL"].samples
        print(f"\n  Conditional Accuracy:")
        print(
            f"  {'Condition':<30} {'N':>5} {'Cold Acc':>9} {'Warm Acc':>9} "
            f"{'Delta':>8} {'KV Reuse':>9}"
        )
        print(f"  {'-'*30} {'-'*5} {'-'*9} {'-'*9} {'-'*8} {'-'*9}")

        conditions = [
            ("reuse > 0%",      [s for s in all_srs if s.kv_reuse_ratio > 0]),
            ("reuse >= 50%",    [s for s in all_srs if s.kv_reuse_ratio >= 0.5]),
            ("reuse >= 80%",    [s for s in all_srs if s.kv_reuse_ratio >= 0.8]),
            ("ctx >= 512 tok",  [s for s in all_srs if s.context_tokens >= 512]),
            ("ctx >= 1K tok",   [s for s in all_srs if s.context_tokens >= 1024]),
        ]
        for label, subset in conditions:
            if not subset:
                continue
            cold_sub = [s for s in subset if not s.is_warm]
            warm_sub = [s for s in subset if s.is_warm]
            c_acc = sum(s.correct for s in cold_sub) / len(cold_sub) if cold_sub else 0
            w_acc = sum(s.correct for s in warm_sub) / len(warm_sub) if warm_sub else 0
            avg_reuse = np.mean([s.kv_reuse_ratio for s in subset])
            print(
                f"  {label:<30} {len(subset):>5} {c_acc:>8.1%} "
                f"{w_acc:>8.1%} {w_acc-c_acc:>+7.1%} {avg_reuse:>8.0%}"
            )

    # Answer-position accuracy (slot bias check)
    if "ALL" in bucket_results:
        all_srs = bucket_results["ALL"].samples
        print(f"\n  Accuracy by Answer Slot (bias check):")
        print(f"  {'Slot':<6} {'Letter':<7} {'Total':>6} {'Correct':>8} {'Acc':>8}")
        print(f"  {'-'*6} {'-'*7} {'-'*6} {'-'*8} {'-'*8}")
        for pos in range(4):
            slot_srs = [s for s in all_srs if s.answer_position == pos]
            if not slot_srs:
                continue
            n_c = sum(s.correct for s in slot_srs)
            print(
                f"  {pos:<6} {LETTERS[pos]:<7} {len(slot_srs):>6} "
                f"{n_c:>8} {n_c/len(slot_srs):>7.1%}"
            )

    # Error breakdown by predicted letter
    if "ALL" in bucket_results:
        all_srs = bucket_results["ALL"].samples
        errors = [s for s in all_srs if not s.correct]
        if errors:
            from collections import Counter
            print(f"\n  Error Distribution by Predicted Letter:")
            lc = Counter(s.predicted or "?" for s in errors)
            for letter in sorted(lc):
                count = lc[letter]
                print(f"    {letter}: {count:>3} / {len(errors)}  {'█' * count}")
            if lc.get("A", 0) / max(len(errors), 1) > 0.5:
                print(f"    ⚠ A-bias: {lc['A']}/{len(errors)} errors predict A")

            warm_errors = [s for s in errors if s.is_warm]
            if warm_errors:
                warm_lc = Counter(s.predicted or "?" for s in warm_errors)
                print(f"\n  Warm (q2) Error Distribution:")
                for letter in sorted(warm_lc):
                    count = warm_lc[letter]
                    print(f"    {letter}: {count:>3}  {'█' * count}")

    # Difficulty breakdown
    if "ALL" in bucket_results:
        all_srs = bucket_results["ALL"].samples
        by_diff: Dict[str, list] = defaultdict(list)
        for s in all_srs:
            by_diff[s.difficulty].append(s)
        if len(by_diff) > 1:
            print(f"\n  Accuracy by Difficulty:")
            print(f"  {'Difficulty':<12} {'N':>5} {'Accuracy':>9} {'KV Reuse':>9}")
            print(f"  {'-'*12} {'-'*5} {'-'*9} {'-'*9}")
            for diff in sorted(by_diff):
                ds = by_diff[diff]
                acc = sum(s.correct for s in ds) / len(ds)
                avg_reuse = np.mean([s.kv_reuse_ratio for s in ds])
                print(f"  {diff:<12} {len(ds):>5} {acc:>8.1%} {avg_reuse:>8.0%}")

    # Verdict
    print(f"\n{'=' * 85}")
    if "ALL" in bucket_results:
        r = bucket_results["ALL"]
        delta = r.warm_accuracy - r.cold_accuracy
        acc_ok = r.mcnemar_pvalue is None or r.mcnemar_pvalue > 0.05
        speed_label = f"{r.avg_ttft_speedup:.1f}x" if r.avg_ttft_speedup > 0 else "N/A"
        if acc_ok:
            print(
                f"  RESULT: Warm accuracy matches cold (delta={delta:+.1%}, "
                f"p={r.mcnemar_pvalue or 'N/A'}) with {speed_label} TTFT speedup "
                f"at {r.avg_kv_reuse_ratio:.0%} avg KV reuse"
            )
        else:
            print(
                f"  RESULT: Accuracy difference detected cold vs warm "
                f"(delta={delta:+.1%}, p={r.mcnemar_pvalue:.3f})"
            )
    print(f"{'=' * 85}\n")

    if verbose and "ALL" in bucket_results:
        all_srs = bucket_results["ALL"].samples
        warm_wrong = [s for s in all_srs if s.is_warm and not s.correct]
        if warm_wrong:
            print(f"  Warm queries answered incorrectly ({len(warm_wrong)}):")
            for s in warm_wrong[:15]:
                print(
                    f"    {s.sample_id}: ctx={s.context_tokens}tok "
                    f"pred={s.predicted} gold={s.gold_answer}({LETTERS[s.answer_position]}) "
                    f"ttft={s.ttft_ms:.0f}ms reuse={s.kv_reuse_ratio:.0%}"
                )
            if len(warm_wrong) > 15:
                print(f"    ... and {len(warm_wrong) - 15} more")


# ── Save results ──────────────────────────────────────────────────

def save_results(
    bucket_results: Dict[str, BucketResult],
    args: argparse.Namespace,
    output_path: Path,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "benchmark": "bug_localization_kvboost",
        "dataset": "JetBrains-Research/lca-bug-localization/py",
        "model": args.model,
        "n_samples": args.n_samples,
        "chunk_size": args.chunk_size,
        "max_context_tokens": args.max_context_tokens,
        "recompute_strategy": args.recompute_strategy,
        "buckets": {},
    }
    for name, r in bucket_results.items():
        cold_srs = [s for s in r.samples if not s.is_warm]
        warm_srs = [s for s in r.samples if s.is_warm]
        serializable["buckets"][name] = {
            "n_samples": r.n_samples,
            "cold_accuracy": r.cold_accuracy,
            "warm_accuracy": r.warm_accuracy,
            "overall_accuracy": r.overall_accuracy,
            "accuracy_delta": r.warm_accuracy - r.cold_accuracy,
            "mcnemar_pvalue": r.mcnemar_pvalue,
            "avg_cold_ttft_ms": r.avg_cold_ttft_ms,
            "avg_warm_ttft_ms": r.avg_warm_ttft_ms,
            "avg_ttft_speedup": r.avg_ttft_speedup,
            "avg_cold_total_ms": r.avg_cold_total_ms,
            "avg_warm_total_ms": r.avg_warm_total_ms,
            "avg_kv_reuse_ratio": r.avg_kv_reuse_ratio,
            "avg_context_tokens": r.avg_context_tokens,
            "kv_reuse_distribution": {
                "mean":   float(np.mean([s.kv_reuse_ratio for s in r.samples])),
                "median": float(np.median([s.kv_reuse_ratio for s in r.samples])),
                "p25":    float(np.percentile([s.kv_reuse_ratio for s in r.samples], 25)),
                "p75":    float(np.percentile([s.kv_reuse_ratio for s in r.samples], 75)),
                "min":    float(np.min([s.kv_reuse_ratio for s in r.samples])),
                "max":    float(np.max([s.kv_reuse_ratio for s in r.samples])),
            },
            "ttft_distribution": {
                "cold": {
                    "mean":   float(np.mean([s.ttft_ms for s in cold_srs])) if cold_srs else 0,
                    "median": float(np.median([s.ttft_ms for s in cold_srs])) if cold_srs else 0,
                    "p25":    float(np.percentile([s.ttft_ms for s in cold_srs], 25)) if cold_srs else 0,
                    "p75":    float(np.percentile([s.ttft_ms for s in cold_srs], 75)) if cold_srs else 0,
                },
                "warm": {
                    "mean":   float(np.mean([s.ttft_ms for s in warm_srs])) if warm_srs else 0,
                    "median": float(np.median([s.ttft_ms for s in warm_srs])) if warm_srs else 0,
                    "p25":    float(np.percentile([s.ttft_ms for s in warm_srs], 25)) if warm_srs else 0,
                    "p75":    float(np.percentile([s.ttft_ms for s in warm_srs], 75)) if warm_srs else 0,
                },
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
                    "predicted": s.predicted,
                    "gold_answer": s.gold_answer,
                    "answer_position": s.answer_position,
                    "correct": s.correct,
                    "ttft_ms": s.ttft_ms,
                    "total_ms": s.total_ms,
                    "kv_reuse_ratio": s.kv_reuse_ratio,
                    "cached_tokens": s.cached_tokens,
                    "is_warm": s.is_warm,
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
        description="Bug Localization Benchmark — KVBoost",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B",
                        help="HuggingFace model name (default: Qwen/Qwen2.5-3B)")
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Number of diff pairs to evaluate (default: 50 → 100 queries)")
    parser.add_argument("--chunk-size", type=int, default=128,
                        help="KVBoost chunk size (default: 128)")
    parser.add_argument("--max-context-tokens", type=int, default=0,
                        help="Max context tokens (default: auto-detect from RAM)")
    parser.add_argument("--recompute-strategy", default="selective",
                        choices=["selective", "cacheblend", "none"],
                        help="KVBoost recompute strategy (default: selective)")
    parser.add_argument("--chunk-boundary-window", type=int, default=0,
                        help="Adaptive boundary window (0=disabled, 16=recommended)")
    parser.add_argument("--overlap-k", type=int, default=0,
                        help="Overlap tokens from previous chunk (0=disabled, 16=recommended)")
    parser.add_argument("--sink-tokens", type=int, default=0,
                        help="Attention sink tokens (0=disabled, 32=recommended)")
    parser.add_argument("--max-cache-bytes", type=float, required=True,
                        help="REQUIRED. Hard KV cache memory budget in bytes "
                             "(accepts float for e.g. 2e9 = 2GB). When exceeded, "
                             "older chunks outside the recency window are evicted "
                             "lowest-importance first. Pruned = evicted: dropped "
                             "chunks are recomputed on demand.")
    parser.add_argument("--recency-window-chunks", type=int, default=8,
                        help="Number of most-recent chunks pinned against eviction "
                             "(default: 8). These form the hard sliding window.")
    parser.add_argument("--output", default=None,
                        help="Output JSON path")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Ignore checkpoint and start fresh")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-sample details for warm errors")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging")

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
        max_cache_bytes=int(args.max_cache_bytes),
        chunk_size=args.chunk_size,
        max_context_tokens=args.max_context_tokens,
        recompute_strategy=args.recompute_strategy,
        chunk_boundary_window=args.chunk_boundary_window,
        overlap_k=args.overlap_k,
        sink_tokens=args.sink_tokens,
        recency_window_chunks=args.recency_window_chunks,
        no_checkpoint=args.no_checkpoint,
    )

    if not bucket_results:
        print("No results produced.")
        return

    print_summary(bucket_results, verbose=args.verbose)

    strategy_suffix = f"_{args.recompute_strategy}" if args.recompute_strategy != "selective" else ""
    out_path = (
        Path(args.output) if args.output
        else RESULTS_DIR / f"bug_localization_kvboost{strategy_suffix}.json"
    )
    save_results(bucket_results, args, out_path)

    checkpoint_path = get_checkpoint_path(
        args.model, args.n_samples, args.max_context_tokens, args.recompute_strategy
    )
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        log.info("Checkpoint cleaned up after successful run")


if __name__ == "__main__":
    main()
