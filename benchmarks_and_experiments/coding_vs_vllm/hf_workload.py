"""HuggingFace-dataset RAG workload for the reuse comparison.

The point of comparing kvboost **CacheBlend** against vLLM **prefix
caching** is cross-request KV reuse. Prefix caching only reuses an
*exact shared leading prefix*; CacheBlend reuses any cached *chunk*
wherever it lands in the prompt (recomputing only the seams). So a
workload that exposes the difference is RAG: the same retrieved
passages recur across queries **in different orders**.

We build that from a real HF dataset (default ``squad``): take a small
pool of context passages, and for each sample concatenate several of
them (shuffled, so a given passage appears at varied positions across
samples) under a shared system prompt, then ask a question. Replayed
sequentially, later samples re-encounter passages they've seen before:
  * vLLM prefix-caching hits only when the leading passages match exactly.
  * kvboost CacheBlend hits on every recurring passage, anywhere.

Needs ``pip install datasets`` — real data only, no synthetic fallback.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RagSample:
    name: str
    system: str
    user: str
    max_tokens: int
    target_tokens: int

    def to_body(self, model: str, stream: bool = True) -> dict:
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.user},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.0,
            "stream": stream,
            # Ask the server to report usage on the final stream chunk so we
            # can read prefix-cache hits (vLLM populates cached_tokens).
            "stream_options": {"include_usage": True},
        }


_SYS = ("You are a careful research assistant. Answer the question using ONLY "
        "the provided passages. Cite the passage number you used.")


def _build(passage_pool: List[str], questions: List[str], *,
           n: int, passages_per: int, seed: int, max_tokens: int) -> List[RagSample]:
    rng = random.Random(seed)
    samples: List[RagSample] = []
    for i in range(n):
        # Pick passages_per passages from the shared pool and SHUFFLE — this
        # is what makes the same passage appear at different positions across
        # samples (CacheBlend reuse; prefix-cache miss).
        chosen = rng.sample(passage_pool, min(passages_per, len(passage_pool)))
        rng.shuffle(chosen)
        ctx = "\n\n".join(f"[Passage {j+1}]\n{p}" for j, p in enumerate(chosen))
        q = questions[i % len(questions)]
        user = f"{ctx}\n\nQuestion: {q}\nAnswer:"
        samples.append(RagSample(
            name=f"rag-{i}",
            system=_SYS,
            user=user,
            max_tokens=max_tokens,
            target_tokens=len(user) // 4,
        ))
    return samples


def load_rag_samples(
    *, dataset: str = "squad", n: int = 10, passages_per: int = 4,
    pool_size: int = 8, seed: int = 0, max_tokens: int = 128,
) -> List[RagSample]:
    """Return ``n`` RAG samples with recurring, reordered REAL passages.

    ``pool_size`` controls how many distinct passages circulate — smaller
    pool ⇒ more recurrence ⇒ more reuse opportunity. ``passages_per`` is
    how many passages each prompt concatenates. Real data only — raises
    SystemExit with an install hint if ``datasets`` is missing.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "ERROR: this benchmark uses a real HuggingFace dataset.\n"
            "Run: pip install datasets"
        )
    try:
        ds = load_dataset(dataset, split="validation")
    except Exception as e:
        raise SystemExit(
            f"ERROR: could not load dataset '{dataset}': {e}\n"
            "Pick another with --dataset (e.g. squad) or check network/HF auth."
        )
    seen, pool, questions = set(), [], []
    for row in ds:
        c = row.get("context") or row.get("passage") or ""
        qn = row.get("question") or ""
        if c and c not in seen and len(pool) < pool_size:
            seen.add(c)
            pool.append(c.strip())
        if qn:
            questions.append(qn.strip())
        if len(pool) >= pool_size and len(questions) >= n:
            break
    if not pool or not questions:
        raise SystemExit(
            f"ERROR: dataset '{dataset}' yielded no usable context/question. "
            "Try --dataset squad."
        )
    return _build(pool, questions, n=n, passages_per=passages_per,
                  seed=seed, max_tokens=max_tokens)
