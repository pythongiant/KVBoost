#!/usr/bin/env python3
"""
Universal KV-Cache Reuse Examples
=================================
Showcases real-world KV cache reuse on any HuggingFace causal LM.

Configuration is loaded from .env in this directory (copy .env.example to .env).
You can also override any setting via CLI flags or environment variables.

Usage:
    # Use defaults from .env
    python examples/run.py

    # Override model on the fly
    python examples/run.py --model Qwen/Qwen2-0.5B

    # Run a specific example
    python examples/run.py --example chatbot

    # List available examples
    python examples/run.py --list
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# ── Load .env from this script's directory ──────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
_ENV_FILE = _SCRIPT_DIR / ".env"


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader — no external dependency needed."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        # Don't overwrite vars already set in the shell
        if key not in os.environ:
            os.environ[key] = value


_load_dotenv(_ENV_FILE)

# ── Make the parent package importable ──────────────────────────────
sys.path.insert(0, str(_SCRIPT_DIR.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s — %(message)s")
log = logging.getLogger("examples")


# ── Configuration from env ──────────────────────────────────────────
def _cfg():
    """Read typed config from environment (set by .env or shell)."""
    return {
        "model_name": os.getenv("MODEL_NAME", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        "hf_token": os.getenv("HF_TOKEN", "") or None,
        "chunk_size": int(os.getenv("CHUNK_SIZE", "64")),
        "max_chunks": int(os.getenv("MAX_CHUNKS", "128")),
        "recompute_overlap": int(os.getenv("RECOMPUTE_OVERLAP", "8")),
        "disk_cache_dir": os.getenv("DISK_CACHE_DIR", "") or None,
        "max_new_tokens": int(os.getenv("MAX_NEW_TOKENS", "64")),
        "temperature": float(os.getenv("TEMPERATURE", "1.0")),
        "do_sample": os.getenv("DO_SAMPLE", "false").lower() == "true",
        "device": os.getenv("DEVICE", "") or None,
    }


# ── Engine loader (cached across examples within one run) ───────────
_engine = None


def get_engine(cfg: dict):
    global _engine
    if _engine is not None:
        return _engine

    from engine import InferenceEngine

    # Set HF token for gated models
    if cfg["hf_token"]:
        os.environ["HF_TOKEN"] = cfg["hf_token"]

    log.info("Loading model: %s", cfg["model_name"])
    _engine = InferenceEngine.from_pretrained(
        model_name=cfg["model_name"],
        chunk_size=cfg["chunk_size"],
        max_chunks=cfg["max_chunks"],
        recompute_overlap=cfg["recompute_overlap"],
        disk_cache_dir=cfg["disk_cache_dir"],
        device=cfg["device"],
    )
    log.info("Ready on device: %s", _engine.device)
    return _engine


# ── Helper ──────────────────────────────────────────────────────────
def _compare(engine, prompt: str, cfg: dict, label: str = ""):
    """Run baseline vs chunk KV reuse side-by-side and print results."""
    from engine import GenerationMode

    if label:
        print(f"\n  --- {label} ---")

    for mode in [GenerationMode.BASELINE, GenerationMode.CHUNK_KV_REUSE]:
        r = engine.generate(
            prompt,
            max_new_tokens=cfg["max_new_tokens"],
            mode=mode,
            temperature=cfg["temperature"],
            do_sample=cfg["do_sample"],
        )
        tag = "BASELINE" if mode == GenerationMode.BASELINE else "KV_REUSE"
        print(
            f"  [{tag:>8}]  TTFT {r.ttft_ms:7.1f}ms | "
            f"reuse {r.kv_reuse_ratio * 100:4.0f}% | "
            f"{r.tokens_per_sec:5.1f} tok/s | "
            f'"{r.output_text[:80]}"'
        )


# =====================================================================
#  Example 1: System Prompt Reuse (chatbot pattern)
# =====================================================================
def example_chatbot(cfg: dict):
    """
    Simulates a chatbot with a fixed system prompt serving multiple users.
    The system prompt KV is computed once and reused across every query.
    """
    engine = get_engine(cfg)

    system_prompt = (
        "You are an expert coding assistant specializing in software engineering, "
        "algorithms, data structures, and distributed system design. You provide "
        "concise, accurate, and well-structured answers. When relevant, include a "
        "brief code example in Python unless the user specifies another language. "
        "Keep answers under 3 sentences unless the topic requires more depth. "
        "Always consider edge cases and performance implications in your answers. "
        "When discussing algorithms, mention time and space complexity using Big-O "
        "notation. When discussing system design, consider scalability, fault "
        "tolerance, and consistency trade-offs. If a question is ambiguous, state "
        "your assumptions before answering. Prefer practical, production-ready "
        "advice over theoretical descriptions. When discussing trade-offs, present "
        "both sides fairly before recommending an approach. Cite well-known papers "
        "or resources when they are directly relevant. Do not hallucinate library "
        "names or API signatures — if unsure, say so. Format code blocks with "
        "proper indentation and include brief inline comments for non-obvious logic. "
        "If the user asks about a deprecated pattern, explain the modern alternative."
    )

    queries = [
        "How do I reverse a linked list?",
        "What is the time complexity of quicksort?",
        "Explain the difference between a mutex and a semaphore.",
        "How does consistent hashing work?",
    ]

    print("\n" + "=" * 70)
    print("  EXAMPLE 1: System Prompt Reuse (chatbot with fixed instructions)")
    print("=" * 70)

    n = engine.warm_chunks(system_prompt)
    print(f"  Warmed {n} chunks from system prompt\n")

    for i, q in enumerate(queries, 1):
        prompt = system_prompt + f"\n\nUser: {q}\nAssistant:"
        _compare(engine, prompt, cfg, label=f"Query {i}: {q}")


# =====================================================================
#  Example 2: RAG Document Reuse
# =====================================================================
def example_rag(cfg: dict):
    """
    Simulates a RAG pipeline where the same retrieved document is queried
    multiple times with different questions.
    """
    engine = get_engine(cfg)

    document = (
        "Retrieval-Augmented Generation (RAG) combines a retriever module with a "
        "generative language model to produce answers grounded in external knowledge. "
        "The retriever searches a document corpus using dense embeddings (e.g., from "
        "a bi-encoder like DPR or Contriever) or sparse methods (BM25, SPLADE) to "
        "find the most relevant passages for a given query. These retrieved passages "
        "are concatenated with the user query and fed to the language model as context, "
        "providing factual grounding that the model's parameters alone may not contain. "
        "RAG significantly reduces hallucinations by anchoring generation in retrieved "
        "evidence rather than relying solely on memorized knowledge from pretraining. "
        "Key architectural decisions include the chunk size for splitting documents in "
        "the vector store (typically 256-512 tokens), the number of retrieved passages "
        "(top-k, usually 3-10), the embedding model dimensionality, and whether to "
        "apply a cross-encoder re-ranker after initial retrieval to improve precision. "
        "Production RAG systems often employ hybrid search that combines BM25 lexical "
        "matching with dense semantic retrieval using reciprocal rank fusion (RRF) for "
        "better recall across both exact keyword matches and semantic similarity. "
        "Advanced techniques include query expansion, hypothetical document embeddings "
        "(HyDE), iterative retrieval where the model generates follow-up queries, and "
        "hierarchical indexing with document summaries for coarse-to-fine retrieval. "
        "Evaluation frameworks measure both retrieval quality (recall@k, MRR, NDCG) "
        "and end-to-end answer quality using exact match (EM), token-level F1, and "
        "human preference ratings. Common failure modes include retriever-generator "
        "mismatch (retrieving relevant passages that the LM ignores), lost-in-the-middle "
        "effects where models attend primarily to the first and last passages, and "
        "conflicting information across retrieved documents. Chunking strategy has an "
        "outsized impact on system quality: chunks that are too small lose context while "
        "chunks that are too large dilute the relevant signal with noise. Overlap between "
        "adjacent chunks (typically 10-20%) helps preserve context at boundaries. "
        "Metadata filtering (by date, source, category) applied before or during retrieval "
        "can dramatically improve precision for domain-specific applications."
    )

    questions = [
        "What does the retriever module do in RAG?",
        "How does RAG reduce hallucinations?",
        "What are the key design choices in a RAG system?",
    ]

    print("\n" + "=" * 70)
    print("  EXAMPLE 2: RAG — Same Document, Multiple Questions")
    print("=" * 70)

    n = engine.warm_chunks(document)
    print(f"  Warmed {n} chunks from document\n")

    for i, q in enumerate(questions, 1):
        prompt = f"Context:\n{document}\n\nQuestion: {q}\nAnswer:"
        _compare(engine, prompt, cfg, label=f"Q{i}: {q}")


# =====================================================================
#  Example 3: Few-Shot Prompting
# =====================================================================
def example_fewshot(cfg: dict):
    """
    Fixed few-shot examples are cached once.
    Only the new query requires fresh computation each time.
    """
    engine = get_engine(cfg)

    few_shot_prefix = (
        "Classify the sentiment of each product review as POSITIVE or NEGATIVE. "
        "Consider the overall tone, specific praise or complaints, and whether "
        "the reviewer would recommend the product.\n\n"
        'Review: "The battery life is incredible, easily lasts all day with heavy '
        "use including GPS navigation, streaming music, and constant notifications. "
        "I used to carry a power bank everywhere but haven't needed it once since "
        'switching to this phone. The fast charging is a nice bonus too."\n'
        "Sentiment: POSITIVE\n\n"
        'Review: "Screen cracked after just one week of normal use — no drops, no '
        "pressure, it just developed a hairline fracture from the top corner. Build "
        "quality feels cheap despite the premium price tag. The customer support "
        "process took three weeks and they initially tried to blame me for the damage "
        'before finally agreeing to a replacement."\n'
        "Sentiment: NEGATIVE\n\n"
        'Review: "Fast shipping, product arrived exactly as described in the listing. '
        "The setup process was straightforward with clear instructions. Sound quality "
        "exceeded my expectations for this price range — rich bass without muddiness "
        "and clear vocals. Very happy with this purchase and would buy from this "
        'seller again without hesitation."\n'
        "Sentiment: POSITIVE\n\n"
        'Review: "Arrived broken with a visible dent in the casing. Customer support '
        "was unhelpful and rude — the first agent hung up on me, and the second said "
        "returns weren't possible for 'cosmetic damage' even though the unit doesn't "
        "power on. Had to dispute the charge through my credit card company. Avoid "
        'this seller at all costs."\n'
        "Sentiment: NEGATIVE\n\n"
        'Review: "Bought this as a gift for my daughter and she absolutely loves it. '
        "The color options are beautiful, the interface is intuitive enough for a "
        "twelve-year-old to figure out without help, and the parental controls are "
        "comprehensive without being intrusive. Battery lasts about two days with "
        'moderate use. Solid value for the money."\n'
        "Sentiment: POSITIVE\n\n"
        'Review: "Firmware update bricked the device on day three. Factory reset '
        "didn't help. The online troubleshooting guide is outdated and references "
        "menu options that no longer exist. Ended up spending four hours with tech "
        "support only to be told I need to ship it back at my own expense for "
        'diagnosis. Incredibly frustrating experience from start to finish."\n'
        "Sentiment: NEGATIVE\n\n"
    )

    new_reviews = [
        "Excellent camera quality, photos look professional even in low light.",
        "The software crashes constantly and the interface is confusing.",
        "Good value for the price, does everything I need it to.",
    ]

    print("\n" + "=" * 70)
    print("  EXAMPLE 3: Few-Shot Classification (cached examples)")
    print("=" * 70)

    n = engine.warm_chunks(few_shot_prefix)
    print(f"  Warmed {n} chunks from few-shot examples\n")

    for i, review in enumerate(new_reviews, 1):
        prompt = few_shot_prefix + f'Review: "{review}"\nSentiment:'
        _compare(engine, prompt, cfg, label=f"Review {i}")


# =====================================================================
#  Example 4: Multi-Turn Conversation
# =====================================================================
def example_multiturn(cfg: dict):
    """
    Simulates a growing conversation where prior turns accumulate.
    Cache reuse increases with each turn as more history is cached.
    """
    engine = get_engine(cfg)
    from engine import GenerationMode

    system = (
        "You are an expert mathematics tutor specializing in calculus, linear algebra, "
        "and differential equations. Explain concepts step by step using clear notation. "
        "When introducing a new rule or theorem, first state it formally, then give an "
        "intuitive explanation of why it works, and finally demonstrate with a concrete "
        "example. Use standard mathematical notation. If the student makes an error, "
        "gently correct it and explain the misconception. Build on previous turns in "
        "the conversation to create a coherent lesson flow. Keep explanations concise "
        "but thorough — aim for understanding, not just the answer."
    )

    # Pre-baked realistic responses so history grows predictably.
    # Using model output would give ~60 chars per turn — too short to ever
    # cross the ~600-token threshold where KV reuse pays off.
    conversation = [
        ("What is the derivative of x^2?",
         "The derivative of x^2 is 2x. Formally, using the limit definition: "
         "d/dx[x^2] = lim(h->0) [(x+h)^2 - x^2] / h = lim(h->0) [2xh + h^2] / h "
         "= 2x. Intuitively, this tells us the slope of the parabola y=x^2 at any "
         "point x — at x=3 the slope is 6, meaning the curve is rising steeply."),

        ("What about x^3?",
         "The derivative of x^3 is 3x^2. Following the same pattern: "
         "d/dx[x^3] = lim(h->0) [(x+h)^3 - x^3] / h. Expanding (x+h)^3 = "
         "x^3 + 3x^2h + 3xh^2 + h^3, subtracting x^3, dividing by h, and taking "
         "the limit gives 3x^2. Notice the pattern: the exponent comes down as a "
         "coefficient and decreases by one."),

        ("Now explain the power rule in general.",
         "The Power Rule: For any real number n, d/dx[x^n] = n*x^(n-1). "
         "This works for all real exponents — positive, negative, and fractional. "
         "For example, d/dx[x^(-1)] = -x^(-2) = -1/x^2, and "
         "d/dx[sqrt(x)] = d/dx[x^(1/2)] = (1/2)*x^(-1/2) = 1/(2*sqrt(x)). "
         "The proof for positive integers uses the binomial theorem on (x+h)^n; "
         "the general case extends via logarithmic differentiation."),

        ("Give me an example with x^5.",
         "Using the power rule on f(x) = x^5: bring down the exponent as a "
         "coefficient and subtract one from it. f'(x) = 5*x^(5-1) = 5x^4. "
         "We can verify at a point: at x=2, f(2)=32 and f'(2)=5*16=80, meaning "
         "the function is increasing rapidly. The second derivative f''(x) = 20x^3 "
         "tells us the rate of change of the slope itself is also increasing."),

        ("How does the chain rule extend this to composite functions?",
         "The Chain Rule handles compositions f(g(x)): d/dx[f(g(x))] = f'(g(x)) * g'(x). "
         "Think of it as 'derivative of the outer function evaluated at the inner, "
         "times the derivative of the inner function.' Combined with the power rule, "
         "this lets us differentiate expressions like (2x+1)^7: the outer is u^7 "
         "(derivative 7u^6) and the inner is 2x+1 (derivative 2), giving "
         "7*(2x+1)^6 * 2 = 14*(2x+1)^6."),

        ("Walk me through differentiating (3x^2 + 1)^4 step by step.",
         "Step 1: Identify outer and inner. Outer: u^4, Inner: u = 3x^2 + 1. "
         "Step 2: Differentiate outer with respect to u: d/du[u^4] = 4u^3. "
         "Step 3: Differentiate inner with respect to x: d/dx[3x^2 + 1] = 6x. "
         "Step 4: Multiply by chain rule: 4*(3x^2 + 1)^3 * 6x = 24x*(3x^2 + 1)^3. "
         "To verify, at x=1: inner = 4, outer = 256, derivative = 24*1*64 = 1536."),

        ("What about the product rule? When do I need that instead?",
         "The Product Rule handles f(x)*g(x): d/dx[f*g] = f'*g + f*g'. "
         "Use it when two separate functions are multiplied, not composed. "
         "For example, d/dx[x^2 * sin(x)] = 2x*sin(x) + x^2*cos(x). "
         "Compare this with the chain rule: sin(x^2) is a composition (chain rule), "
         "but x^2 * sin(x) is a product (product rule). Sometimes you need both: "
         "d/dx[x^3 * (2x+1)^5] requires the product rule on x^3 and (2x+1)^5, "
         "and the chain rule to differentiate (2x+1)^5 within that."),

        ("Can you combine all three rules in one example?",
         "Let's differentiate f(x) = x^2 * (3x + 1)^4. We need the product rule "
         "for the multiplication, the chain rule for (3x+1)^4, and the power rule "
         "throughout. Product rule: f'(x) = [d/dx x^2] * (3x+1)^4 + x^2 * [d/dx (3x+1)^4]. "
         "First term: 2x * (3x+1)^4. Second term: x^2 * 4*(3x+1)^3 * 3 = 12x^2*(3x+1)^3. "
         "Combined: f'(x) = 2x*(3x+1)^4 + 12x^2*(3x+1)^3. Factor out 2x*(3x+1)^3: "
         "f'(x) = 2x*(3x+1)^3 * [(3x+1) + 6x] = 2x*(3x+1)^3*(9x+1)."),
    ]

    print("\n" + "=" * 70)
    print("  EXAMPLE 4: Multi-Turn Conversation (growing history)")
    print("=" * 70)

    engine.warm_chunks(system)
    history = system

    for i, (user_msg, assistant_reply) in enumerate(conversation, 1):
        history += f"\n\nUser: {user_msg}\nAssistant: {assistant_reply}"

        # Only measure on the generation call (append query for next turn)
        prompt = history + f"\n\nUser: Continue.\nAssistant:"

        r_base = engine.generate(
            prompt, max_new_tokens=cfg["max_new_tokens"],
            mode=GenerationMode.BASELINE, do_sample=False,
        )
        r_kv = engine.generate(
            prompt, max_new_tokens=cfg["max_new_tokens"],
            mode=GenerationMode.CHUNK_KV_REUSE, do_sample=False,
        )

        speedup = r_base.ttft_ms / r_kv.ttft_ms if r_kv.ttft_ms > 0 else 0
        print(
            f"\n  Turn {i}: \"{user_msg}\"\n"
            f"    History tokens: {r_kv.prompt_tokens} | "
            f"Reuse: {r_kv.kv_reuse_ratio * 100:.0f}% | "
            f"TTFT baseline: {r_base.ttft_ms:.1f}ms → "
            f"KV reuse: {r_kv.ttft_ms:.1f}ms "
            f"({speedup:.1f}x)"
        )


# =====================================================================
#  Example 5: Code Context Reuse
# =====================================================================
def example_code_context(cfg: dict):
    """
    Simulates providing a shared code file as context, then asking
    multiple questions about it — common in code assistants / copilots.
    """
    engine = get_engine(cfg)

    code_context = '''
"""
Thread-safe LRU cache with TTL expiration.
Supports both sync and async usage patterns.
"""

import threading
import time
from collections import OrderedDict
from typing import Any, Hashable, Optional, Tuple


class CacheEntry:
    """Single cache entry with value, metadata, and expiration tracking."""

    __slots__ = ("key", "value", "created_at", "last_accessed", "ttl", "hit_count")

    def __init__(self, key: Hashable, value: Any, ttl: Optional[float] = None):
        self.key = key
        self.value = value
        self.created_at = time.monotonic()
        self.last_accessed = self.created_at
        self.ttl = ttl
        self.hit_count = 0

    @property
    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return (time.monotonic() - self.created_at) > self.ttl

    def touch(self) -> None:
        self.last_accessed = time.monotonic()
        self.hit_count += 1


class LRUCache:
    """
    Thread-safe LRU cache with optional per-entry TTL.

    Features:
        - O(1) get/put via OrderedDict
        - Thread safety via reentrant lock
        - Lazy TTL expiration (checked on access)
        - Bulk eviction when capacity exceeded
        - Statistics tracking (hits, misses, evictions)
    """

    def __init__(self, capacity: int = 128, default_ttl: Optional[float] = None):
        if capacity < 1:
            raise ValueError(f"Capacity must be >= 1, got {capacity}")
        self._capacity = capacity
        self._default_ttl = default_ttl
        self._store: OrderedDict[Hashable, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: Hashable, default: Any = None) -> Any:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return default
            if entry.is_expired:
                del self._store[key]
                self._misses += 1
                return default
            self._store.move_to_end(key)
            entry.touch()
            self._hits += 1
            return entry.value

    def put(self, key: Hashable, value: Any, ttl: Optional[float] = None) -> None:
        effective_ttl = ttl if ttl is not None else self._default_ttl
        with self._lock:
            if key in self._store:
                del self._store[key]
            elif len(self._store) >= self._capacity:
                self._evict_one()
            self._store[key] = CacheEntry(key, value, effective_ttl)
            self._store.move_to_end(key)

    def delete(self, key: Hashable) -> bool:
        with self._lock:
            if key in self._store:
                del self._store[key]
                return True
            return False

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def _evict_one(self) -> None:
        if self._store:
            self._store.popitem(last=False)
            self._evictions += 1

    def purge_expired(self) -> int:
        removed = 0
        with self._lock:
            expired_keys = [k for k, v in self._store.items() if v.is_expired]
            for k in expired_keys:
                del self._store[k]
                removed += 1
        return removed

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._store),
            "capacity": self._capacity,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / total if total > 0 else 0.0,
            "evictions": self._evictions,
        }

    def __len__(self) -> int:
        return len(self._store)

    def __contains__(self, key: Hashable) -> bool:
        with self._lock:
            entry = self._store.get(key)
            if entry is None or entry.is_expired:
                return False
            return True
'''

    questions = [
        "What is the time complexity of the insert method?",
        "How would you add a delete method to this BST?",
        "Is this BST self-balancing? Why or why not?",
    ]

    print("\n" + "=" * 70)
    print("  EXAMPLE 5: Code Context Reuse (code assistant pattern)")
    print("=" * 70)

    n = engine.warm_chunks(code_context)
    print(f"  Warmed {n} chunks from code context\n")

    for i, q in enumerate(questions, 1):
        prompt = f"Given this code:\n{code_context}\n\nQuestion: {q}\nAnswer:"
        _compare(engine, prompt, cfg, label=f"Q{i}: {q}")


# =====================================================================
#  Registry & CLI
# =====================================================================
EXAMPLES = {
    "chatbot": ("System prompt reuse (chatbot)", example_chatbot),
    "rag": ("RAG document reuse", example_rag),
    "fewshot": ("Few-shot classification", example_fewshot),
    "multiturn": ("Multi-turn conversation", example_multiturn),
    "code": ("Code context reuse", example_code_context),
}


def main():
    parser = argparse.ArgumentParser(
        description="KV-Cache Reuse Examples — works with any HuggingFace causal LM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", default=None,
        help="HuggingFace model name (overrides MODEL_NAME in .env)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Chunk size in tokens (overrides CHUNK_SIZE in .env)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=None,
        help="Max tokens to generate (overrides MAX_NEW_TOKENS in .env)",
    )
    parser.add_argument(
        "--device", default=None,
        help="Device: cuda, mps, cpu (overrides DEVICE in .env)",
    )
    parser.add_argument(
        "--example", default=None, choices=list(EXAMPLES.keys()),
        help="Run a specific example (default: run all)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available examples and exit",
    )
    args = parser.parse_args()

    if args.list:
        print("Available examples:\n")
        for name, (desc, _) in EXAMPLES.items():
            print(f"  {name:12s}  {desc}")
        print(f"\nRun one:  python examples/run.py --example {list(EXAMPLES.keys())[0]}")
        print("Run all:  python examples/run.py")
        return

    # CLI flags override env vars
    if args.model:
        os.environ["MODEL_NAME"] = args.model
    if args.chunk_size is not None:
        os.environ["CHUNK_SIZE"] = str(args.chunk_size)
    if args.max_new_tokens is not None:
        os.environ["MAX_NEW_TOKENS"] = str(args.max_new_tokens)
    if args.device:
        os.environ["DEVICE"] = args.device

    cfg = _cfg()

    print(f"\n  Model:       {cfg['model_name']}")
    print(f"  Chunk size:  {cfg['chunk_size']}")
    print(f"  Max tokens:  {cfg['max_new_tokens']}")
    print(f"  Device:      {cfg['device'] or 'auto'}")
    print(f"  Sampling:    {'yes' if cfg['do_sample'] else 'greedy'}")

    if args.example:
        _, fn = EXAMPLES[args.example]
        fn(cfg)
    else:
        for name, (desc, fn) in EXAMPLES.items():
            fn(cfg)

    # Final stats
    if _engine is not None:
        print("\n" + "=" * 70)
        print("  Final cache stats:", _engine.cache_stats())
        print("=" * 70)


if __name__ == "__main__":
    main()
