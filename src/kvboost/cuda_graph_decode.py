"""CUDA-graph decode path — eliminates per-token launch overhead.

On a bandwidth-bound GPU (e.g. RTX 3060), eager autoregressive decode spends
the majority of each token's wall time on kernel-launch + Python overhead, not
on math or memory. vLLM closes that gap with CUDA graphs over a *static* KV
cache. This module does the same for kvboost:

  1. After kvboost's (reuse-based) prefill produces the prompt KV, copy it into
     a HuggingFace ``StaticCache`` (pre-allocated, fixed-address buffers) via
     direct tensor writes — so reuse/TTFT is preserved, decode just runs on a
     graph-capturable, static-shape cache.
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


def _iter_kv(past_kv) -> List[tuple]:
    """Return ``[(key_tensor, value_tensor), ...]`` for any supported cache
    format. Never returns the StaticCache — only called on the *input* KV
    (DynamicCache or tuple-of-tuples) that we copy *into* the StaticCache."""
    if past_kv is None:
        return []
    # transformers 5.x: both DynamicCache and StaticCache have `.layers`.
    # DynamicLayer / StaticLayer both expose `.keys` / `.values`.
    if hasattr(past_kv, "layers"):
        return [(layer.keys, layer.values) for layer in past_kv.layers]
    # Older DynamicCache (4.x): flat key_cache / value_cache lists.
    if hasattr(past_kv, "key_cache"):
        return list(zip(past_kv.key_cache, past_kv.value_cache))
    # Legacy tuple-of-tuples: ((k0, v0), (k1, v1), ...)
    return list(past_kv)


@dataclass
class _Captured:
    """A StaticCache + static input buffers for one bucketed cache length."""
    cache: Any
    input_ids: torch.Tensor       # (1, 1) static
    pos_ids: torch.Tensor         # (1, 1) static
    cache_pos: torch.Tensor       # (1,)   static


class CUDAGraphDecoder:
    def __init__(self, model, *, device, dtype, eos_token_id,
                 max_cache_len: int = 8192, force_eager: bool = False):
        self.model = model
        self.device = torch.device(device)
        self.dtype = dtype
        self.eos = eos_token_id
        self.force_eager = force_eager      # tests: eager static-cache, no compile
        self._config = getattr(model, "config", None)
        # ONE fixed cache size for all requests. A per-request size would make
        # torch.compile recompile for every distinct prompt length (multi-turn
        # prompts grow each turn) and quickly hit dynamo's recompile_limit ->
        # fall back to eager. A single size compiles once -> the cudagraph
        # sticks. The oversized-but-fixed cache reads ~the whole buffer per step
        # (mask handles validity), which is ~5% of weight bandwidth — cheap next
        # to eliminating the launch overhead. Prompts longer than this fall back
        # to the eager loop.
        self._cap = min(int(max_cache_len),
                        int(getattr(self._config, "max_position_embeddings", 1 << 30))
                        if self._config is not None else int(max_cache_len))
        self._cache: Optional[_Captured] = None   # built once, for self._cap
        self._decode_fn = None              # torch.compile(model) wrapper, lazy
        self._disabled = False
        self._self_checked = False
        self._ok = self._probe()
        # Compile only on a real CUDA device (cudagraphs need it); force_eager
        # runs the static-cache forward eagerly (CPU-testable).
        self._use_compiled = self._ok and not self.force_eager and \
            torch.cuda.is_available() and self.device.type == "cuda"

    # ------------------------------------------------------------------
    # Module classes whose forward calls a custom CUDA kernel (int4 AWQ GEMM,
    # streaming weight rebind) that torch.compile cannot trace. Their presence
    # graph-breaks the WHOLE forward: reduce-overhead captures an EMPTY CUDA
    # graph (flooding "The CUDA Graph is empty") and gives no speedup. CUDA-graph
    # decode only helps a plain fp16 model (native Linear → cuBLAS → traceable).
    _UNCAPTURABLE_MODULES = ("StreamingQLinear",)

    def _uncapturable_module(self) -> Optional[str]:
        try:
            for m in self.model.modules():
                cls = type(m).__name__
                if cls in self._UNCAPTURABLE_MODULES:
                    return cls
        except Exception:
            pass
        return None

    def _probe(self) -> bool:
        if self._config is None:
            return False
        try:
            from transformers import StaticCache  # noqa: F401
        except Exception:
            return False
        if self.force_eager:
            return True
        if not torch.cuda.is_available() or self.device.type != "cuda":
            return False
        # Refuse models torch.compile can't capture (AWQ streaming / int4): they
        # graph-break to an empty CUDA graph with no speedup, just noise.
        kind = self._uncapturable_module()
        if kind is not None:
            log.warning(
                "CUDA-graph decode disabled: model uses %s (custom int4/"
                "streaming kernel) that torch.compile cannot capture — it would "
                "graph-break to an empty CUDA graph with no speedup. CUDA-graph "
                "decode targets the plain fp16 path; for int4 decode speed use a "
                "real Marlin kernel instead.", kind,
            )
            return False
        return True

    def applicable(self, batch_size: int = 1) -> bool:
        return self._ok and not self._disabled and batch_size == 1

    # ------------------------------------------------------------------
    def _dims(self):
        c = self._config
        n_heads = int(getattr(c, "num_attention_heads", 0) or 1)
        n_kv = int(getattr(c, "num_key_value_heads", n_heads))
        head_dim = int(getattr(c, "head_dim", 0) or (
            c.hidden_size // n_heads if hasattr(c, "hidden_size") else 64
        ))
        return n_kv, head_dim

    def _compiled(self):
        """Lazily build the reduce-overhead compiled model (decode-only use)."""
        if not self._use_compiled:
            return None
        if self._decode_fn is None:
            try:
                # Headroom over the default recompile_limit (8) — with a single
                # fixed cache size we expect ~1 compile, but the self-check +
                # first real step can trip a couple; don't let dynamo bail.
                try:
                    import torch._dynamo as _dyn
                    if getattr(_dyn.config, "recompile_limit", 8) < 32:
                        _dyn.config.recompile_limit = 32
                    if getattr(_dyn.config, "cache_size_limit", 8) < 32:
                        _dyn.config.cache_size_limit = 32
                except Exception:
                    pass
                self._decode_fn = torch.compile(
                    self.model, mode="reduce-overhead", fullgraph=False,
                )
            except Exception as e:
                log.warning("torch.compile for CUDA-graph decode failed (%s); "
                            "eager static-cache.", e)
                self._use_compiled = False
                self._decode_fn = None
        return self._decode_fn

    def _build(self) -> _Captured:
        from transformers import StaticCache

        n_kv, head_dim = self._dims()
        cache = StaticCache(config=self._config, max_cache_len=self._cap)
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
                    if hasattr(lyr, "keys") and lyr.is_initialized:
                        _dyn.mark_static_address(lyr.keys)
                    if hasattr(lyr, "values") and lyr.is_initialized:
                        _dyn.mark_static_address(lyr.values)
            except Exception:
                pass
        return _Captured(cache=cache, input_ids=inp, pos_ids=pos, cache_pos=cpos)

    def _get_cache(self) -> _Captured:
        if self._cache is None:
            self._cache = self._build()
        return self._cache

    def _populate(self, cache, past_kv, L: int) -> None:
        """Copy the prefill KV (first L positions) into the static cache.

        Resets the cache first (zeroes tensors + cumulative_length counters),
        then writes each layer's KV directly into the static buffers.
        StaticLayer.update auto-increments cumulative_length by the number of
        positions written — so after this call cumulative_length == L for every
        layer, which is what the decode loop's position tracking assumes.
        """
        cache.reset()
        for i, (k, v) in enumerate(_iter_kv(past_kv)):
            k = k[:, :, :L, :].to(self.device, self.dtype)
            v = v[:, :, :L, :].to(self.device, self.dtype)
            # StaticLayer.update ignores any kwargs; it tracks position via its
            # internal cumulative_length tensor. After reset() it starts at 0
            # and advances by the number of positions we write (= L here).
            cache.update(k, v, i)

    @torch.no_grad()
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

    @torch.no_grad()
    def _step_logits(self, cap: _Captured, tok: int, cur: int) -> torch.Tensor:
        """Run one decode step: feed token ``tok`` at position ``cur``,
        return logits ``(1, vocab)`` for the next position.

        Sets the three static input buffers in-place (avoids a new tensor
        allocation per step — required for CUDA-graph stable addresses).
        """
        cap.input_ids[0, 0] = tok
        cap.pos_ids[0, 0] = cur
        cap.cache_pos[0] = cur
        out = self._forward(cap)
        return out.logits[:, -1, :]

    @torch.no_grad()
    def _self_check(self, cap: _Captured, past_kv, L: int, seed: int,
                    k: int = 4) -> bool:
        """Compare the first ``k`` GREEDY tokens from the compiled-graph path
        against an eager reference (original model, fresh DynamicCache). Catches
        capture/compile bugs (incl. a frozen mask) before any token is emitted.

        IMPORTANT: builds a *copy* of past_kv for the reference decode so the
        original is not mutated in-place (DynamicCache is extended by the model
        on every step; sharing it would corrupt the caller's view of the prefill
        KV and shift the layer sequence lengths seen by subsequent _populate
        calls).
        """
        from transformers import DynamicCache

        # ── Reference: eager model + fresh DynamicCache copy ─────────────
        ref: List[int] = []
        ref_kv = DynamicCache()
        for i, (lk, lv) in enumerate(_iter_kv(past_kv)):
            ref_kv.update(
                lk[:, :, :L, :].clone().to(self.device),
                lv[:, :, :L, :].clone().to(self.device),
                i,
            )

        tok, cur = seed, L
        for _ in range(k):
            o = self.model(
                input_ids=torch.tensor([[tok]], device=self.device),
                position_ids=torch.tensor([[cur]], device=self.device),
                past_key_values=ref_kv,
                use_cache=True,
            )
            tok = int(o.logits[:, -1, :].argmax(-1).item())
            ref.append(tok)
            cur += 1
            if tok == self.eos:
                break

        # ── Compiled / eager-static path ─────────────────────────────────
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

    @torch.no_grad()
    def _measure_speedup(self, cc: _Captured, past_kv, L: int, seed: int,
                         steps: int = 12) -> None:
        """Time the compiled step vs an eager step on the SAME static cache and
        log the ratio. Makes the 'compiles but graph-breaks → no real speedup'
        outcome LOUD: torch.compile won't error or fail the self-check in that
        case, so without this probe a useless graph looks like a working one.
        Best-effort; never affects decode."""
        if not (self._use_compiled and torch.cuda.is_available()):
            return
        try:
            import time

            compiled = self._compiled()
            if compiled is None:
                return

            kw = dict(input_ids=cc.input_ids, position_ids=cc.pos_ids,
                      cache_position=cc.cache_pos, past_key_values=cc.cache,
                      use_cache=True)

            def _run(fn) -> float:
                # Reset cache to a clean state for every timing run so that
                # cumulative_length starts at L (not L + previous-step-count).
                # Without this, each fn() writes at an ever-increasing position
                # and the warmup / timed runs operate on different cache states.
                self._populate(cc.cache, past_kv, L)
                cc.input_ids[0, 0] = seed
                cc.pos_ids[0, 0] = L
                cc.cache_pos[0] = L
                for _ in range(3):           # warm
                    fn(**kw)
                    # Reset between warmup steps so each writes at position L
                    self._populate(cc.cache, past_kv, L)
                    cc.cache_pos[0] = L
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(steps):
                    fn(**kw)
                    self._populate(cc.cache, past_kv, L)
                    cc.cache_pos[0] = L
                torch.cuda.synchronize()
                return (time.perf_counter() - t0) / steps * 1000.0

            ms_compiled = _run(compiled)
            ms_eager = _run(self.model)
            ratio = ms_eager / max(ms_compiled, 1e-6)
            if ratio >= 1.15:
                log.info("CUDA-graph decode speedup: %.1f→%.1f ms/step "
                         "(%.2f× vs eager).", ms_eager, ms_compiled, ratio)
            else:
                log.warning(
                    "CUDA-graph decode gives NO meaningful speedup "
                    "(compiled %.1f ms vs eager %.1f ms/step, %.2f×). "
                    "torch.compile is graph-breaking over transformers' "
                    "StaticCache bookkeeping/mask, so the launch overhead "
                    "survives. The reliable decode lever here is weight quant "
                    "(Marlin int4): run with MODEL=...-AWQ.",
                    ms_compiled, ms_eager, ratio)
        except Exception as e:
            log.debug("CUDA-graph speedup probe skipped (%s).", e)

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
            # Single fixed cache size; over-cap prompts use the eager loop.
            if L + max_new_tokens > self._cap:
                return None
            cc = self._get_cache()

            # One-time multi-step self-check (compiled path only), BEFORE any
            # token is emitted, so a failure cleanly falls back to eager.
            if self._use_compiled and not self._self_checked:
                self._self_checked = True
                if not self._self_check(cc, past_kv, L, seed_token):
                    self._disabled = True
                    return None
                # Self-check only proves CORRECTNESS. Also measure SPEED so a
                # graph-breaking compile (correct but no faster) is surfaced.
                self._measure_speedup(cc, past_kv, L, seed_token)

            self._populate(cc.cache, past_kv, L)
            out: List[int] = []
            cur = L
            tok = seed_token
            for _ in range(max_new_tokens):
                logits = self._step_logits(cc, tok, cur)
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
