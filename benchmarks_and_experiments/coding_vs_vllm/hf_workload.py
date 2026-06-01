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

Needs ``pip install datasets``. A ``--synthetic`` fallback generates
passages locally if the dataset can't be fetched, so the script still
runs offline.
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
    synthetic: bool = False,
) -> List[RagSample]:
    """Return ``n`` RAG samples with recurring, reordered passages.

    ``pool_size`` controls how many distinct passages circulate — smaller
    pool ⇒ more recurrence ⇒ more reuse opportunity. ``passages_per`` is
    how many passages each prompt concatenates.
    """
    pool: List[str]
    questions: List[str]

    if not synthetic:
        try:
            from datasets import load_dataset
            ds = load_dataset(dataset, split="validation")
            # De-dup contexts to form the passage pool; collect questions.
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
                raise RuntimeError("dataset yielded no usable context/question")
            return _build(pool, questions, n=n, passages_per=passages_per,
                          seed=seed, max_tokens=max_tokens)
        except Exception as e:  # ImportError, network, schema mismatch …
            print(f"[hf_workload] dataset '{dataset}' unavailable ({e}); "
                  "using --synthetic fallback.")

    # Synthetic fallback: self-contained passages so the script runs offline.
    _missions = ["Apollo", "Voyager", "Hubble", "Kepler", "Webb",
                 "Cassini", "Juno", "Galileo"]
    _body = (
        " program produced extensive findings. It involved multiple "
        "instruments, a long mission timeline, and a large international "
        "team. Key results included measurements of radiation, imaging of "
        "distant bodies, and refinements to orbital models. The data "
        "informed subsequent missions and theory. "
    )
    pool = [
        f"Passage topic {k}: The {_missions[k % len(_missions)]}" + _body * 6
        for k in range(pool_size)
    ]
    questions = [
        "Which program is described and what was a key result?",
        "What instruments were involved?",
        "How did the data inform later work?",
        "Summarize the mission timeline.",
    ]
    return _build(pool, questions, n=n, passages_per=passages_per,
                  seed=seed, max_tokens=max_tokens)
