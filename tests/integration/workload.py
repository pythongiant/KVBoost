"""Realistic prompt workloads for OOM-planner load testing.

Each workload returns a list of ``(system_prompt, user_content, max_tokens)``
tuples representing one "session" of traffic. Prompts are deliberately
shaped to resemble what production LLM serving actually carries —
long-document analysis, code review, multi-turn chat — not toy
``"Hello world"`` requests.

Token counts are approximate (rough word-to-token ratio of 1.3); exact
counts come from the server's tokenizer. The goal is shape, not
precision: 15K input / 1K output exercises a very different planner
path than 200 input / 50 output.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple


# ── Building blocks ──────────────────────────────────────────────────────────


_FINANCE_SENTENCE = (
    "The 2024 quarterly report shows revenue of $12.4B with operating "
    "margins at 18.2%. Engineering headcount grew 22% year-over-year "
    "while customer acquisition cost dropped 14%. The product team "
    "shipped 47 features across three platforms during the period. "
)

_CODE_SNIPPET = """\
def process_batch(items: list[Item], *, max_workers: int = 4) -> list[Result]:
    \"\"\"Process a batch of items in parallel.

    Returns results in the same order as input. Raises BatchError if more
    than half the items fail. Individual failures are wrapped in a Result
    with .ok=False so callers can inspect them per-item.
    \"\"\"
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_process_one, item): i for i, item in enumerate(items)}
        results: list[Result] = [None] * len(items)
        n_fail = 0
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                results[idx] = Result(ok=False, error=str(exc), item=items[idx])
                n_fail += 1
        if n_fail > len(items) // 2:
            raise BatchError(f"{n_fail}/{len(items)} items failed")
        return results
"""

_RESEARCH_PARAGRAPH = (
    "Recent advances in long-context modeling have focused on two distinct "
    "directions: architectural changes (sparse attention, state-space models, "
    "linear attention) and inference-time optimizations (KV-cache compression, "
    "chunked prefill, speculative decoding). Each approach trades different "
    "axes — sparse attention reduces compute at the cost of recall, KV "
    "compression reduces memory at the cost of quality, speculative decoding "
    "trades drafter cost for verifier parallelism. The interaction between "
    "these techniques is under-studied; most papers evaluate one in isolation. "
)


# ── Workload generators ──────────────────────────────────────────────────────


@dataclass
class WorkloadItem:
    """One request shape — feeds straight into the chat-completions endpoint."""
    name: str          # short label for telemetry
    system: str
    user: str
    max_tokens: int
    expected_prompt_tokens: int   # approximate, for sanity checks

    def to_body(self, model: str) -> dict:
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.user},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.7,
        }


def long_document_analysis(scale: int = 200) -> WorkloadItem:
    """Long-document summarization — ~10-15K input, 2K output.

    ``scale`` controls the document length (sentence multiplier).
    scale=200 ≈ 12K tokens; scale=500 ≈ 30K tokens.
    """
    doc = _FINANCE_SENTENCE * scale
    return WorkloadItem(
        name=f"long-doc-{scale}",
        system="You are a careful financial analyst.",
        user=(
            "Read the report below and produce: (a) five specific revenue "
            "trends with numerical citations, (b) three risks ordered by "
            "magnitude, (c) a forecast for the next three quarters with "
            "confidence intervals.\n\n"
            f"{doc}"
        ),
        max_tokens=2048,
        expected_prompt_tokens=int(len(doc.split()) * 1.3),
    )


def code_review(scale: int = 80) -> WorkloadItem:
    """Multi-file code review — ~8-12K input, 2K output."""
    codebase = "\n\n".join([_CODE_SNIPPET] * scale)
    return WorkloadItem(
        name=f"code-review-{scale}",
        system="You are a senior software engineer reviewing a pull request.",
        user=(
            "Review the following code. For each function: (1) summarize its "
            "purpose in one sentence, (2) note any correctness bugs, "
            "(3) suggest one performance improvement.\n\n"
            f"{codebase}"
        ),
        max_tokens=2048,
        expected_prompt_tokens=int(len(codebase.split()) * 1.3),
    )


def research_summary(scale: int = 100) -> WorkloadItem:
    """Research-paper summarization — ~6-10K input, 1K output."""
    paper = _RESEARCH_PARAGRAPH * scale
    return WorkloadItem(
        name=f"research-{scale}",
        system="You are a senior ML researcher.",
        user=(
            "Summarize the key technical contributions of this paper in "
            "exactly 5 bullet points. Then identify the most important "
            "open question the paper leaves unanswered.\n\n"
            f"{paper}"
        ),
        max_tokens=1024,
        expected_prompt_tokens=int(len(paper.split()) * 1.3),
    )


def short_chat() -> WorkloadItem:
    """Short conversational reply — 200-500 input, 100-300 output."""
    return WorkloadItem(
        name="short-chat",
        system="You are a helpful assistant.",
        user="Explain the difference between speculative decoding and KV-cache "
             "quantization in three sentences.",
        max_tokens=256,
        expected_prompt_tokens=80,
    )


def multi_turn(history_turns: int = 6) -> WorkloadItem:
    """Synthesized multi-turn chat — ~3-6K input growing with history."""
    history = []
    for i in range(history_turns):
        history.append(
            f"User asked about topic #{i}: How does {['caching', 'attention', 'quantization', 'streaming'][i % 4]} work?"
        )
        history.append(
            f"Assistant gave a detailed explanation about topic #{i} covering "
            f"the underlying mechanism, typical implementation, and three "
            f"common pitfalls. {_RESEARCH_PARAGRAPH}"
        )
    full = "\n\n".join(history)
    return WorkloadItem(
        name=f"multi-turn-{history_turns}",
        system="Continue the conversation naturally, referring back to prior turns.",
        user=full + "\n\nGiven the conversation above, what's the single "
                    "most important takeaway?",
        max_tokens=512,
        expected_prompt_tokens=int(len(full.split()) * 1.3),
    )


def oversized() -> WorkloadItem:
    """Deliberately too-big prompt — for 413/auto-truncate paths.

    ~80K tokens; should not fit on any 12-24 GB GPU with sensible config.
    """
    doc = _FINANCE_SENTENCE * 8000
    return WorkloadItem(
        name="oversized",
        system="You are a financial analyst.",
        user="Summarize this document.\n\n" + doc,
        max_tokens=512,
        expected_prompt_tokens=int(len(doc.split()) * 1.3),
    )


# ── Workload mixes ───────────────────────────────────────────────────────────


def production_mix(seed: int = 0, n: int = 30) -> List[WorkloadItem]:
    """Realistic traffic mix for OOM planner stress.

    Weights chosen to look like a typical RAG/agent server:
      50% short conversational (most production traffic)
      20% long-doc analysis
      15% multi-turn chat
      10% code review
       5% research-paper summarization

    Set ``n`` to control session length. ``seed`` makes the order
    deterministic for reproducibility.
    """
    rng = random.Random(seed)
    pool = (
        [short_chat()] * 50
        + [long_document_analysis(scale=200)] * 10
        + [long_document_analysis(scale=300)] * 5
        + [long_document_analysis(scale=500)] * 5
        + [multi_turn(history_turns=4)] * 10
        + [multi_turn(history_turns=8)] * 5
        + [code_review(scale=60)] * 8
        + [code_review(scale=120)] * 2
        + [research_summary(scale=80)] * 3
        + [research_summary(scale=200)] * 2
    )
    rng.shuffle(pool)
    return pool[:n]


def heavy_mix(seed: int = 0, n: int = 20) -> List[WorkloadItem]:
    """Heavier mix — biased toward long contexts. Stresses planner more."""
    rng = random.Random(seed)
    pool = (
        [long_document_analysis(scale=200)] * 6
        + [long_document_analysis(scale=400)] * 4
        + [long_document_analysis(scale=600)] * 2
        + [code_review(scale=100)] * 4
        + [code_review(scale=200)] * 2
        + [research_summary(scale=200)] * 3
        + [multi_turn(history_turns=10)] * 3
    )
    rng.shuffle(pool)
    return pool[:n]


def burst_short(n: int = 50) -> List[WorkloadItem]:
    """All short — measures planner overhead per request when prompts are
    small. Should hit the fastest config every time."""
    return [short_chat() for _ in range(n)]
