"""CUDA-graph decode path — eliminates per-token launch overhead.

On a bandwidth-bound GPU (e.g. RTX 3060), eager autoregressive decode spends
the majority of each token's wall time on kernel-launch + Python overhead, not
on math or memory. vLLM closes that gap with CUDA graphs over a *static* KV
cache. This module does the same for kvboost:

  1. After kvboost's (reuse-based) prefill produces the prompt KV, copy it into
     a HuggingFace ``StaticCache`` (pre-allocated, fixed-address buffers) via the
     cache's own ``update`` API — so reuse/TTFT is preserved, decode just runs
     on a graph-capturable, static-shape cache.
  2. Capture the single-token decode forward into CUDA graphs via
     ``torch.compile(mode="reduce-overhead")``, replayed per token; sampling is
     eager and outside the graph.

Why torch.compile rather than a hand-rolled ``torch.cuda.CUDAGraph``: an HF
model forward contains host-side ops (causal-mask construction, cache
bookkeeping, occasional ``.item()``/``.any()`` syncs) that are ILLEGAL inside a
manual capture region — raw capture fails with "operation failed due to a
previous error during capture". ``torch.compile`` traces the forward, graph-
breaks around those host ops, and applies cudagraph trees to the capturable
compute. It is still real CUDA graphs; it just does the surgery raw capture
can't. The model is compiled with decode-shaped (1,1) inputs ONLY (prefill uses
the uncompiled model), so it compiles once and never recompiles per prefill
length.

Safety (graph capture is validated on the GPU box, not in CI):
  * Capability-gated (CUDA + StaticCache); ``applicable()`` is False otherwise.
  * Shape-gated to batch==1, non-speculative.
  * One-time multi-step GREEDY self-check on first use: the compiled-graph
    tokens are compared against an eager reference (original model, fresh
    DynamicCache); on mismatch the path is PERMANENTLY disabled and the caller
    falls back to the eager loop — before any token is emitted.
  * Per-call exception fallback (e.g. compile/capture errors → eager).
  * The static-cache *forward* (KV copy + cache_position + masking) is
    CPU-testable and covered by tests/test_cuda_graph_decode.py via the eager
    static-cache mode; only the compiled cudagraph step is GPU-only.

Stacks with weight quant (Marlin int4): int4 weights cut the bandwidth floor,
graphs cut the launch overhead — together they target both decode costs.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, List, Optional

import torch

log = logging.getLogger("kvboost.cuda_graph_decode")


def _iter_kv(past_kv):
    """Yield (key, value) per layer for DynamicCache (5.x ``layers`` or older
    ``key_cache``) or a tuple-of-tuples legacy cache."""
    if past_kv is None:
        return []
    if hasattr(past_kv, "layers"):                 # transformers 5.x DynamicCache
        return [(l.keys, l.values) for l in past_kv.layers]
    if hasattr(past_kv, "key_cache"):              # older DynamicCache
        return list(zip(past_kv.key_cache, past_kv.value_cache))
    return [(k, v) for (k, v) in past_kv]          # tuple-of-tuples


@dataclass
class _Captured:
    """A StaticCache + static input buffers for one bucketed cache length."""
    cache: Any
    input_ids: torch.Tensor       # (1, 1) static
    pos_ids: torch.Tensor         # (1, 1) static
    cache_pos: torch.Tensor       # (1,)   static


class CUDAGraphDecoder:
    def __init__(self, model, *, device, dtype, eos_token_id,
                 bucket: int = 256, force_eager: bool = False):
        self.model = model
        self.device = torch.device(device)
        self.dtype = dtype
        self.eos = eos_token_id
        self.bucket = max(1, int(bucket))
        self.force_eager = force_eager      # tests: eager static-cache, no compile
        self._config = getattr(model, "config", None)
        self._caches: dict[int, _Captured] = {}
        self._decode_fn = None              # torch.compile(model) wrapper, lazy
        self._disabled = False
        self._self_checked = False
        self._ok = self._probe()
        # Compile only on a real CUDA device (cudagraphs need it); force_eager
        # runs the static-cache forward eagerly (CPU-testable).
        self._use_compiled = self._ok and not self.force_eager and \
            torch.cuda.is_available() and self.device.type == "cuda"

    # ------------------------------------------------------------------
    def _probe(self) -> bool:
        if self._config is None:
            return False
        try:
            from transformers import StaticCache  # noqa: F401
        except Exception:
            return False
        if self.force_eager:
            return True
        return torch.cuda.is_available() and self.device.type == "cuda"

    def applicable(self, batch_size: int = 1) -> bool:
        return self._ok and not self._disabled and batch_size == 1

    # ------------------------------------------------------------------
    def _dims(self):
        c = self._config
        n_heads = int(getattr(c, "num_attention_heads"))
        n_kv = int(getattr(c, "num_key_value_heads", n_heads))
        head_dim = int(getattr(c, "head_dim", 0) or (c.hidden_size // n_heads))
        return n_kv, head_dim

    def _max_pos(self) -> int:
        return int(getattr(self._config, "max_position_embeddings", 1 << 30))

    def _round_up(self, n: int) -> int:
        b = self.bucket
        return min(((n + b - 1) // b) * b, self._max_pos())

    def _compiled(self):
        """Lazily build the reduce-overhead compiled model (decode-only use)."""
        if not self._use_compiled:
            return None
        if self._decode_fn is None:
            try:
                self._decode_fn = torch.compile(
                    self.model, mode="reduce-overhead", fullgraph=False,
                )
            except Exception as e:
                log.warning("torch.compile for CUDA-graph decode failed (%s); "
                            "eager static-cache.", e)
                self._use_compiled = False
                self._decode_fn = None
        return self._decode_fn

    def _build(self, bucket_len: int) -> _Captured:
        from transformers import StaticCache

        n_kv, head_dim = self._dims()
        cache = StaticCache(config=self._config, max_cache_len=bucket_len)
        cache.early_initialization(1, n_kv, head_dim, self.dtype, self.device)

        inp = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        pos = torch.zeros((1, 1), dtype=torch.long, device=self.device)
        cpos = torch.zeros((1,), dtype=torch.long, device=self.device)

        # Pin static addresses so reduce-overhead's cudagraphs don't re-record
        # when our buffers / cache tensors are reused across steps. Best-effort.
        if self._use_compiled:
            try:
                import torch._dynamo as _dyn
                for b in (inp, pos, cpos):
                    _dyn.mark_static_address(b)
                for lyr in getattr(cache, "layers", []):
                    if hasattr(lyr, "keys"):
                        _dyn.mark_static_address(lyr.keys)
                    if hasattr(lyr, "values"):
                        _dyn.mark_static_address(lyr.values)
            except Exception:
                pass
        return _Captured(cache=cache, input_ids=inp, pos_ids=pos, cache_pos=cpos)

    def _get(self, bucket_len: int) -> _Captured:
        cap = self._caches.get(bucket_len)
        if cap is None:
            cap = self._build(bucket_len)
            self._caches[bucket_len] = cap
        return cap

    def _populate(self, cache, past_kv, L: int) -> None:
        """Copy the prefill KV (length L) into the static cache via its update
        API, which also sets the length counter to L."""
        cache.reset()
        cpos = torch.arange(L, device=self.device)
        for i, (k, v) in enumerate(_iter_kv(past_kv)):
            k = k[:, :, :L, :].to(self.device, self.dtype)
            v = v[:, :, :L, :].to(self.device, self.dtype)
            cache.update(k, v, i, cache_kwargs={"cache_position": cpos})

    def _forward(self, cap: _Captured):
        fn = self._compiled() if self._use_compiled else None
        target = fn if fn is not None else self.model
        return target(
            input_ids=cap.input_ids,
            position_ids=cap.pos_ids,
            cache_position=cap.cache_pos,
            past_key_values=cap.cache,
            use_cache=True,
        )

    def _step_logits(self, cap: _Captured, tok: int, cur: int) -> torch.Tensor:
        cap.input_ids[0, 0] = tok
        cap.pos_ids[0, 0] = cur
        cap.cache_pos[0] = cur
        out = self._forward(cap)
        return out.logits[:, -1, :]

    def _self_check(self, cap: _Captured, past_kv, L: int, seed: int,
                    as_cache, k: int = 4) -> bool:
        """Compare the first ``k`` GREEDY tokens from the compiled-graph path
        against an eager reference (original model, fresh DynamicCache). Catches
        capture/compile bugs (incl. a frozen mask) before any token is emitted.
        Mutates cap.cache (caller re-populates before the real loop)."""
        ref: List[int] = []
        with torch.no_grad():
            dyn = as_cache(past_kv)
            tok, cur = seed, L
            for _ in range(k):
                o = self.model(
                    input_ids=torch.tensor([[tok]], device=self.device),
                    position_ids=torch.tensor([[cur]], device=self.device),
                    past_key_values=dyn, use_cache=True,
                )
                tok = int(o.logits[:, -1, :].argmax(-1).item())
                ref.append(tok)
                cur += 1
                if tok == self.eos:
                    break
        self._populate(cap.cache, past_kv, L)
        got: List[int] = []
        tok, cur = seed, L
        for _ in range(len(ref)):
            lg = self._step_logits(cap, tok, cur)
            tok = int(lg.argmax(-1).item())
            got.append(tok)
            cur += 1
            if tok == self.eos:
                break
        ok = got == ref
        if ok:
            log.info("CUDA-graph decode self-check passed (%d greedy tokens "
                     "match eager).", len(ref))
        else:
            log.error("CUDA-graph decode self-check FAILED (graph %s != eager "
                      "%s) — DISABLING, eager fallback.", got, ref)
        return ok

    # ------------------------------------------------------------------
    def decode(self, *, past_kv, start_pos: int, seed_token: int,
               max_new_tokens: int, sample_fn: Callable[[torch.Tensor], int],
               as_cache: Callable[[Any], Any],
               on_token: Optional[Callable[[int], None]] = None
               ) -> Optional[List[int]]:
        """Generate up to ``max_new_tokens`` continuation tokens, feeding
        ``seed_token`` at ``start_pos`` first. Returns the new tokens (the seed
        is already held by the caller), or ``None`` to signal the caller should
        fall back to the eager loop (self-check failed / unsupported / error)."""
        if max_new_tokens <= 0 or seed_token == self.eos:
            return []
        try:
            L = start_pos
            bucket_len = self._round_up(L + max_new_tokens)
            if L + 1 > bucket_len:
                return None  # prompt longer than we can bucket — eager handles it
            cap = self._get(bucket_len)

            # One-time multi-step self-check (compiled path only), BEFORE any
            # token is emitted, so a failure cleanly falls back to eager.
            if self._use_compiled and not self._self_checked:
                self._self_checked = True
                if not self._self_check(cap, past_kv, L, seed_token, as_cache):
                    self._disabled = True
                    return None

            self._populate(cap.cache, past_kv, L)
            out: List[int] = []
            cur = L
            tok = seed_token
            for _ in range(max_new_tokens):
                logits = self._step_logits(cap, tok, cur)
                nxt = sample_fn(logits)
                out.append(nxt)
                if on_token is not None:
                    on_token(nxt)
                if nxt == self.eos:
                    break
                tok = nxt
                cur += 1
            return out
        except Exception as e:
            # Never corrupt a response: bail to the eager loop. If tokens were
            # already streamed we must NOT (caller would double-run), so only
            # signal fallback when nothing was emitted.
            if not locals().get("out"):
                log.warning("CUDA-graph decode errored (%s); eager fallback.", e)
                return None
            log.error("CUDA-graph decode errored mid-stream (%s); truncating.", e)
            return out
