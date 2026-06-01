"""Realistic coding-agent prompts at controlled context sizes.

Coding assistants (Cursor, Copilot, Aider, Claude Code) stuff large
repo context into the prompt — many files, then "implement/refactor/fix
function X". That's a long-prefill, modest-decode shape, and it's where
KV-cache reuse and OOM behavior actually matter. We synthesize that shape
so the benchmark is self-contained (no dataset download), with a target
*approximate* token count per prompt (the server reports the exact count
via ``usage.prompt_tokens``).

Optionally, if ``datasets`` + HumanEval are available, ``humaneval_items``
yields real HumanEval prompts — but the default synthetic path needs no
network and lets us dial context length precisely for the OOM ramp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

# A chunk of realistic-looking Python (~480 tokens). Repeated/varied to
# build repo context of a target size.
_CODE_BLOCK = '''\
class {cls}Service:
    """Coordinates {name} operations across the storage and cache tiers.

    Thread-safety: all public methods are safe to call concurrently; the
    internal _lock guards the write path. Reads are lock-free against an
    immutable snapshot swapped in under the lock.
    """

    def __init__(self, store: Store, cache: Cache, *, max_workers: int = 8):
        self._store = store
        self._cache = cache
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._lock = threading.RLock()
        self._snapshot: dict[str, Record] = {{}}
        self._dirty: set[str] = set()
        self._metrics = Metrics(namespace="{name}")

    def get(self, key: str) -> Optional[Record]:
        hit = self._cache.get(key)
        if hit is not None:
            self._metrics.incr("cache_hit")
            return hit
        rec = self._store.load(key)
        if rec is not None:
            self._cache.put(key, rec, ttl=self._ttl_for(rec))
        self._metrics.incr("cache_miss")
        return rec

    def put(self, key: str, rec: Record) -> None:
        with self._lock:
            self._snapshot[key] = rec
            self._dirty.add(key)
        self._cache.invalidate(key)
        self._metrics.incr("write")

    def flush(self) -> int:
        with self._lock:
            dirty = list(self._dirty)
            self._dirty.clear()
        futures = [self._pool.submit(self._store.save, k, self._snapshot[k]) for k in dirty]
        return sum(1 for f in as_completed(futures) if f.result() is not None)

    def _ttl_for(self, rec: Record) -> int:
        return 300 if rec.hot else 30
'''


@dataclass
class CodingPrompt:
    name: str
    system: str
    user: str
    max_tokens: int
    target_tokens: int  # approximate prompt size we built to

    def to_body(self, model: str, stream: bool = True) -> dict:
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.user},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.2,
            "stream": stream,
        }


def _repo_context(target_tokens: int) -> str:
    """Build ~target_tokens of varied Python 'repo context'.

    Rough heuristic: ~4 chars/token. Each block ≈ 480 tokens; vary class
    names so it isn't trivially compressible / prefix-cacheable.
    """
    approx_block_tokens = 480
    n_blocks = max(1, target_tokens // approx_block_tokens)
    names = ["Order", "Payment", "Inventory", "Shipment", "Catalog",
             "User", "Session", "Pricing", "Ledger", "Audit", "Notify",
             "Search", "Index", "Queue", "Cache", "Auth"]
    blocks = []
    for i in range(n_blocks):
        nm = names[i % len(names)] + (str(i // len(names)) if i >= len(names) else "")
        blocks.append(
            "# ── module: {0}.py ──\n".format(nm.lower())
            + _CODE_BLOCK.format(cls=nm, name=nm.lower())
        )
    return "\n\n".join(blocks)


_HEADER = (
    "import threading\n"
    "from concurrent.futures import ThreadPoolExecutor, as_completed\n"
    "from typing import Optional\n\n"
)

_TASK = (
    "\n\n# ── TASK ──\n"
    "Given the codebase above, implement a new `ReconciliationService` that "
    "reconciles records across two `{0}Service` instances: it must (1) detect "
    "keys present in one but not the other, (2) for keys in both, pick the "
    "record with the newer `updated_at`, (3) be thread-safe and reuse the "
    "existing ThreadPoolExecutor pattern, and (4) expose a `reconcile() -> "
    "ReconcileReport` method. Return only the implementation with type hints."
)


def coding_prompt(target_tokens: int, *, max_tokens: int = 512) -> CodingPrompt:
    """One coding-agent prompt of approximately ``target_tokens`` input."""
    ctx = _HEADER + _repo_context(target_tokens)
    user = ctx + _TASK.format("Order")
    return CodingPrompt(
        name=f"code-{target_tokens // 1000}k" if target_tokens >= 1000
             else f"code-{target_tokens}",
        system="You are a senior Python engineer. Write correct, idiomatic, "
               "production-quality code with type hints.",
        user=user,
        max_tokens=max_tokens,
        target_tokens=target_tokens,
    )


def throughput_mix(seed: int = 0, n: int = 24) -> List[CodingPrompt]:
    """A realistic coding-agent traffic mix: mostly small/medium context
    (single-file edits), some large (multi-file repo context)."""
    import random
    rng = random.Random(seed)
    pool = (
        [coding_prompt(800, max_tokens=384)] * 8     # small: one-file edit
        + [coding_prompt(2000, max_tokens=512)] * 6   # medium
        + [coding_prompt(6000, max_tokens=512)] * 4   # multi-file
        + [coding_prompt(12000, max_tokens=512)] * 3  # large repo context
        + [coding_prompt(20000, max_tokens=512)] * 1  # very large
    )
    rng.shuffle(pool)
    return pool[:n]


def oom_ramp(contexts: List[int], *, max_tokens: int = 256) -> List[CodingPrompt]:
    """One prompt per target context length, ascending — for the OOM ramp."""
    return [coding_prompt(c, max_tokens=max_tokens) for c in contexts]
