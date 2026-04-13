"""
InferenceEngine (exported as KVBoost)
=====================================
Ties together:
  model / tokenizer
  KVCacheManager
  ChunkRegistry
  PromptAssembler
  SelectiveRecompute

Exposes three generation modes for benchmarking:
  BASELINE        — standard HF generate, no caching
  PREFIX_CACHE    — exact prefix caching only (control)
  CHUNK_KV_REUSE  — full chunk-level KV reuse + selective recompute

Usage
-----
    from kvboost import KVBoost

    engine = KVBoost.from_pretrained("Qwen/Qwen2.5-3B")
    engine.warm("You are a helpful assistant.")
    result = engine.generate("You are a helpful assistant.\n\nHello!")
    print(result.output_text)
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DynamicCache

from .models import AssembledPrompt, CachedChunk, PastKVType, WarmResult, content_hash_from_tokens, chained_hash
from .cache_manager import KVCacheManager
from .chunk_registry import ChunkRegistry, ChunkStrategy
from .prompt_assembler import AssemblyMode, PromptAssembler
from .selective_recompute import SelectiveRecompute
from .cacheblend import CacheBlendRecompute
from .compat import check_model_compatibility, SUPPORTED_ARCHITECTURES

log = logging.getLogger(__name__)


class GenerationMode(str, enum.Enum):
    BASELINE = "baseline"
    PREFIX_CACHE = "prefix_cache"
    CHUNK_KV_REUSE = "chunk_kv_reuse"


class RecomputeStrategy(str, enum.Enum):
    SELECTIVE = "selective"    # fix last R tokens at each seam (original)
    CACHEBLEND = "cacheblend"  # fix top-k% most deviated tokens (smarter)
    NONE = "none"              # no recompute — fastest, slight quality risk


@dataclass
class GenerationResult:
    mode: str
    prompt: str
    output_text: str
    generated_tokens: int
    ttft_ms: float          # time-to-first-token
    total_ms: float         # end-to-end
    tokens_per_sec: float
    kv_reuse_ratio: float   # fraction of prompt tokens served from cache
    prompt_tokens: int
    cached_tokens: int
    first_token_logits: Optional["np.ndarray"] = None  # logits for first generated token


class InferenceEngine:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        chunk_size: int = 128,
        max_chunks: int = 128,
        recompute_overlap: int = 16,
        recompute_strategy: RecomputeStrategy = RecomputeStrategy.SELECTIVE,
        recompute_ratio: float = 0.15,
        kv_cache_bits: int = 16,
        disk_cache_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        if device is None:
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"

        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.recompute_strategy = RecomputeStrategy(recompute_strategy)

        # Sub-systems (CPU storage for cache tensors, move to device on use)
        self.cache_manager = KVCacheManager(
            max_chunks=max_chunks,
            disk_dir=disk_cache_dir,
            device="cpu",
            kv_cache_bits=kv_cache_bits,
        )
        self.chunk_registry = ChunkRegistry(
            chunk_size=chunk_size,
            strategy=ChunkStrategy.FIXED,
        )
        self.assembler = PromptAssembler(
            cache_manager=self.cache_manager,
            chunk_registry=self.chunk_registry,
            mode=AssemblyMode.CHUNK_REUSE,
        )
        self.selective_recompute = SelectiveRecompute(
            recompute_overlap=recompute_overlap,
            skip_if_no_seams=True,
            device="cpu",
        )
        self.cacheblend_recompute = CacheBlendRecompute(
            recompute_ratio=recompute_ratio,
            device="cpu",
        )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        strict: bool = True,
        **kwargs,
    ) -> "InferenceEngine":
        """
        Load a HuggingFace model and create a KVBoost engine.

        Args:
            model_name: Any HF decoder-only causal LM (must use RoPE).
            strict: If True (default), raise on unsupported architectures
                    and warn on untested ones. Set False to skip checks.
            **kwargs: Passed to InferenceEngine.__init__().
        """
        log.info("Loading model %s ...", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.eval()

        check_model_compatibility(model, strict=strict)

        return cls(model=model, tokenizer=tokenizer, **kwargs)

    # ------------------------------------------------------------------
    # Public generate API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 64,
        mode: GenerationMode = GenerationMode.CHUNK_KV_REUSE,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> GenerationResult:
        token_ids = self._encode(prompt)

        if mode == GenerationMode.BASELINE:
            return self._generate_baseline(prompt, token_ids, max_new_tokens, temperature, do_sample)
        elif mode == GenerationMode.PREFIX_CACHE:
            return self._generate_prefix_cache(prompt, token_ids, max_new_tokens, temperature, do_sample)
        elif mode == GenerationMode.CHUNK_KV_REUSE:
            return self._generate_chunk_reuse(prompt, token_ids, max_new_tokens, temperature, do_sample)
        raise ValueError(f"Unknown mode {mode}")

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> List[GenerationResult]:
        """
        Generate responses for multiple prompts sharing a common prefix.
        Loads shared prefix KV once, runs batched prefill and decode.

        Args:
            prompts: List of prompts (should share a common prefix for best results).
            max_new_tokens: Max tokens to generate per prompt.
            temperature: Sampling temperature.
            do_sample: Greedy (False) or sampling (True).

        Returns:
            List of GenerationResult, one per prompt.
        """
        from .batch import (
            find_common_chunk_prefix, broadcast_kv, pad_and_mask, batched_decode,
        )

        if len(prompts) == 1:
            return [self.generate(prompts[0], max_new_tokens, temperature=temperature, do_sample=do_sample)]

        t0 = time.perf_counter()
        batch_size = len(prompts)

        # Tokenize all prompts
        all_token_ids = [self._encode(p) for p in prompts]

        # Find shared chunk-aligned prefix
        common_len = find_common_chunk_prefix(all_token_ids, self.chunk_registry.chunk_size)

        # Load shared prefix KV from cache
        shared_kv = None
        if common_len > 0:
            assembled = self.assembler.assemble(all_token_ids[0][:common_len + 1])
            shared_kv = assembled.cached_past_kv
            common_len = assembled.cached_length

        # Collect suffix token IDs (non-shared tail of each prompt)
        suffix_ids_list = [ids[common_len:] for ids in all_token_ids]

        # Pad suffixes and build attention masks
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        padded_suffixes, attn_masks = pad_and_mask(suffix_ids_list, pad_id)
        max_suffix_len = max(len(s) for s in suffix_ids_list)

        # Build batched input tensors
        suffix_input = torch.tensor(padded_suffixes, dtype=torch.long, device=self.device)
        pos_ids = torch.arange(
            common_len, common_len + max_suffix_len,
            dtype=torch.long, device=self.device,
        ).unsqueeze(0).expand(batch_size, -1)

        # Broadcast shared KV across batch (zero-copy expand)
        batched_past = None
        if shared_kv is not None:
            shared_kv_device = tuple(
                (k.to(self.device), v.to(self.device)) for k, v in shared_kv
            )
            batched_past = broadcast_kv(shared_kv_device, batch_size)

        # Batched prefill
        with torch.no_grad():
            out = self.model(
                input_ids=suffix_input,
                past_key_values=self._as_cache(batched_past),
                position_ids=pos_ids,
                use_cache=True,
            )

        first_token_time = time.perf_counter()
        past_kv = self._normalize_past_kv(out.past_key_values)

        # Sample first token per sequence (using each sequence's last real token logits)
        first_tokens = []
        for b in range(batch_size):
            real_len = len(suffix_ids_list[b])
            logits_b = out.logits[b, real_len - 1, :].unsqueeze(0)
            tok = self._sample(logits_b, temperature, do_sample)
            first_tokens.append(tok)

        first_tokens_t = torch.tensor(first_tokens, dtype=torch.long, device=self.device)

        # Batched decode
        generated_ids, _ = batched_decode(
            model=self.model,
            past_kv=past_kv,
            first_tokens=first_tokens_t,
            start_pos=common_len + max_suffix_len,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            temperature=temperature,
            do_sample=do_sample,
            device=self.device,
        )

        t1 = time.perf_counter()

        # Store prompt chunks for future reuse
        for ids in all_token_ids:
            self._store_prompt_chunks(ids)

        # Build results
        results = []
        ttft = (first_token_time - t0) * 1000
        total_ms = (t1 - t0) * 1000
        hit_ratio = common_len / max(max(len(ids) for ids in all_token_ids), 1)

        for b in range(batch_size):
            output_text = self.tokenizer.decode(generated_ids[b], skip_special_tokens=True)
            results.append(GenerationResult(
                mode="chunk_kv_reuse_batch",
                prompt=prompts[b],
                output_text=output_text,
                generated_tokens=len(generated_ids[b]),
                ttft_ms=ttft,
                total_ms=total_ms,
                tokens_per_sec=len(generated_ids[b]) / max(t1 - t0, 1e-6),
                kv_reuse_ratio=hit_ratio,
                prompt_tokens=len(all_token_ids[b]),
                cached_tokens=common_len,
            ))

        return results

    def generate_many(
        self,
        prompts: List[str],
        max_new_tokens: int = 64,
        temperature: float = 1.0,
        do_sample: bool = False,
    ) -> List[GenerationResult]:
        """
        Like generate_batch(), but auto-groups prompts by shared prefix.
        Prompts without shared prefixes are processed individually.

        Args:
            prompts: List of prompts (may or may not share prefixes).
            max_new_tokens: Max tokens to generate per prompt.

        Returns:
            List of GenerationResult in the same order as input prompts.
        """
        from .batch import group_by_prefix

        all_token_ids = [self._encode(p) for p in prompts]
        groups = group_by_prefix(
            prompts, all_token_ids, self.chunk_registry.chunk_size,
        )

        results: List[Optional[GenerationResult]] = [None] * len(prompts)

        for group_indices in groups.values():
            group_prompts = [prompts[i] for i in group_indices]
            if len(group_prompts) == 1:
                group_results = [self.generate(
                    group_prompts[0], max_new_tokens,
                    temperature=temperature, do_sample=do_sample,
                )]
            else:
                group_results = self.generate_batch(
                    group_prompts, max_new_tokens,
                    temperature=temperature, do_sample=do_sample,
                )
            for idx, result in zip(group_indices, group_results):
                results[idx] = result

        return results

    # ------------------------------------------------------------------
    # Cache population helper
    # ------------------------------------------------------------------

    def warm(self, text: str, position_offset: int = 0) -> WarmResult:
        """
        Encode `text` and cache all its fixed-size chunks.

        Returns a WarmResult with diagnostics including alignment warnings.
        The result is truthy (usable as int) via chunks_stored.

        Call this for your system prompt / few-shot examples / documents
        BEFORE calling generate() so the cache is already populated.
        """
        token_ids = self._encode(text)
        chunks_added = 0
        pos = position_offset
        parent_hash = None

        for start, end, slice_ids in self.chunk_registry.split(token_ids, text):
            p_hash = chained_hash(slice_ids, parent_hash)
            c_hash = content_hash_from_tokens(slice_ids)

            if self.cache_manager.get(p_hash) is not None:
                parent_hash = p_hash
                pos += len(slice_ids)
                continue

            kv = self._encode_to_kv(slice_ids, position_offset=pos)
            chunk = CachedChunk(
                chunk_id=p_hash,
                text=self.tokenizer.decode(slice_ids),
                token_ids=slice_ids,
                past_key_values=kv,
                position_start=pos,
                position_end=pos + len(slice_ids),
                prefix_hash=p_hash,
                content_hash=c_hash,
            )
            self.cache_manager.store(chunk)
            parent_hash = p_hash
            pos += len(slice_ids)
            chunks_added += 1

        # Build diagnostic
        chunk_size = self.chunk_registry.chunk_size
        n_tokens = len(token_ids)
        partial_tail = n_tokens % chunk_size
        aligned = partial_tail == 0 or partial_tail < self.chunk_registry.min_chunk_tokens

        warning = None
        if not aligned:
            warning = (
                f"Prompt length {n_tokens} tokens is not a multiple of "
                f"chunk_size {chunk_size}. The last {partial_tail} tokens "
                f"will not be cached and must be recomputed on every "
                f"generate() call."
            )
            log.warning("warm(): %s", warning)

        return WarmResult(
            chunks_stored=chunks_added,
            token_count=n_tokens,
            chunk_size=chunk_size,
            chunk_boundary_aligned=aligned,
            partial_tail_tokens=partial_tail,
            alignment_warning=warning,
        )

    # Keep old name as alias
    warm_chunks = warm

    # ------------------------------------------------------------------
    # Generation implementations
    # ------------------------------------------------------------------

    def _generate_baseline(
        self,
        prompt: str,
        token_ids: List[int],
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
    ) -> GenerationResult:
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        t0 = time.perf_counter()
        first_token_time = None
        generated = []
        first_token_logits = None

        with torch.no_grad():
            past = None
            cur_ids = input_ids
            for step in range(max_new_tokens):
                out = self.model(input_ids=cur_ids, past_key_values=self._as_cache(past), use_cache=True)
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                # Capture first-token logits for comparison with cached versions
                if step == 0 and first_token_logits is None:
                    import numpy as np
                    first_token_logits = out.logits[0, -1, :].cpu().float().numpy()
                # Normalize: newer transformers returns DynamicCache, not plain tuple
                past = self._normalize_past_kv(out.past_key_values)
                next_token = self._sample(out.logits[:, -1, :], temperature, do_sample)
                generated.append(next_token)
                if next_token == self.tokenizer.eos_token_id:
                    break
                cur_ids = torch.tensor([[next_token]], dtype=torch.long, device=self.device)

        t1 = time.perf_counter()
        output_text = self.tokenizer.decode(generated, skip_special_tokens=True)
        ttft = (first_token_time - t0) * 1000 if first_token_time else 0
        total_ms = (t1 - t0) * 1000
        tps = len(generated) / max((t1 - t0), 1e-6)

        return GenerationResult(
            mode="baseline",
            prompt=prompt,
            output_text=output_text,
            generated_tokens=len(generated),
            ttft_ms=ttft,
            total_ms=total_ms,
            tokens_per_sec=tps,
            kv_reuse_ratio=0.0,
            prompt_tokens=len(token_ids),
            cached_tokens=0,
            first_token_logits=first_token_logits,
        )

    def _generate_prefix_cache(
        self,
        prompt: str,
        token_ids: List[int],
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
    ) -> GenerationResult:
        """Standard prefix caching: reuse contiguous leading chunks."""
        merged_kv, covered = self.cache_manager.build_prefix_kv(
            token_ids, self.chunk_registry.chunk_size
        )
        live_ids = token_ids[covered:]
        return self._decode_with_kv(
            prompt, token_ids, merged_kv, covered, live_ids,
            max_new_tokens, temperature, do_sample, mode_name="prefix_cache"
        )

    def _generate_chunk_reuse(
        self,
        prompt: str,
        token_ids: List[int],
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
    ) -> GenerationResult:
        """Full chunk-level KV reuse + recompute (strategy-dependent)."""
        assembled = self.assembler.assemble(token_ids)

        # Apply recompute strategy when multiple chunks are stitched
        if len(assembled.chunk_boundaries) > 1:
            if assembled.has_approximate:
                # Approximate matches (content-only key) have wrong position
                # encodings and/or wrong preceding context — always use
                # CacheBlend to fix the full KV, not just boundaries
                log.debug("Approximate match detected — forcing CacheBlend recompute")
                assembled = self.cacheblend_recompute.apply(assembled, self.model)
            elif self.recompute_strategy == RecomputeStrategy.SELECTIVE:
                assembled = self.selective_recompute.apply(assembled, self.model)
            elif self.recompute_strategy == RecomputeStrategy.CACHEBLEND:
                assembled = self.cacheblend_recompute.apply(assembled, self.model)
            # NONE: skip recompute entirely

        return self._decode_with_kv(
            prompt, token_ids,
            assembled.cached_past_kv,
            assembled.cached_length,
            assembled.live_token_ids,
            max_new_tokens, temperature, do_sample,
            mode_name="chunk_kv_reuse",
            hit_ratio=assembled.cache_hit_ratio,
        )

    # ------------------------------------------------------------------
    # Shared decode loop
    # ------------------------------------------------------------------

    def _decode_with_kv(
        self,
        prompt: str,
        full_token_ids: List[int],
        past_kv: Optional[PastKVType],
        cached_len: int,
        live_ids: List[int],
        max_new_tokens: int,
        temperature: float,
        do_sample: bool,
        mode_name: str,
        hit_ratio: Optional[float] = None,
    ) -> GenerationResult:
        t0 = time.perf_counter()
        first_token_time = None
        generated = []
        first_token_logits = None

        # Move past_kv to model device
        if past_kv is not None:
            past_kv = tuple(
                (layer[0].to(self.device), layer[1].to(self.device)) for layer in past_kv
            )

        # ----- encode live tokens (prompt tail) -------------------------
        if live_ids:
            input_ids = torch.tensor([live_ids], dtype=torch.long, device=self.device)
            pos_ids = torch.arange(
                cached_len, cached_len + len(live_ids),
                dtype=torch.long, device=self.device,
            ).unsqueeze(0)

            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    past_key_values=self._as_cache(past_kv),
                    position_ids=pos_ids,
                    use_cache=True,
                )
            first_token_time = time.perf_counter()
            # Capture first-token logits for comparison with baseline
            import numpy as np
            first_token_logits = out.logits[0, -1, :].cpu().float().numpy()
            past_kv = self._normalize_past_kv(out.past_key_values)
            next_token = self._sample(out.logits[:, -1, :], temperature, do_sample)
            generated.append(next_token)
            if next_token == self.tokenizer.eos_token_id:
                pass  # let loop handle
        else:
            # All tokens were cached — run a dummy forward to get first logits
            # by feeding the last cached token again at its position
            last_id = full_token_ids[-1] if full_token_ids else 0
            input_ids = torch.tensor([[last_id]], dtype=torch.long, device=self.device)
            pos_ids = torch.tensor([[cached_len - 1]], dtype=torch.long, device=self.device)
            # Trim past_kv to exclude last position so re-encoding is valid
            trimmed_kv: Optional[PastKVType] = None
            if past_kv is not None and KVCacheManager.kv_seq_len(past_kv) > 1:
                trimmed_kv = KVCacheManager.slice_kv(past_kv, 0, cached_len - 1)
                trimmed_kv = tuple(
                    (layer[0].to(self.device), layer[1].to(self.device)) for layer in trimmed_kv
                )
            with torch.no_grad():
                out = self.model(
                    input_ids=input_ids,
                    past_key_values=self._as_cache(trimmed_kv),
                    position_ids=pos_ids,
                    use_cache=True,
                )
            first_token_time = time.perf_counter()
            # Capture first-token logits for comparison with baseline
            import numpy as np
            first_token_logits = out.logits[0, -1, :].cpu().float().numpy()
            past_kv = self._normalize_past_kv(out.past_key_values)
            next_token = self._sample(out.logits[:, -1, :], temperature, do_sample)
            generated.append(next_token)

        # ----- autoregressive decode ------------------------------------
        cur_pos = cached_len + len(live_ids)
        while len(generated) < max_new_tokens:
            if generated[-1] == self.tokenizer.eos_token_id:
                break
            cur_ids = torch.tensor([[generated[-1]]], dtype=torch.long, device=self.device)
            pos_ids = torch.tensor([[cur_pos]], dtype=torch.long, device=self.device)
            with torch.no_grad():
                out = self.model(
                    input_ids=cur_ids,
                    past_key_values=self._as_cache(past_kv),
                    position_ids=pos_ids,
                    use_cache=True,
                )
            past_kv = self._normalize_past_kv(out.past_key_values)
            next_token = self._sample(out.logits[:, -1, :], temperature, do_sample)
            generated.append(next_token)
            cur_pos += 1

        t1 = time.perf_counter()

        # ----- store newly computed chunks into cache -------------------
        self._store_prompt_chunks(full_token_ids)

        output_text = self.tokenizer.decode(generated, skip_special_tokens=True)
        ttft = (first_token_time - t0) * 1000 if first_token_time else 0
        total_ms = (t1 - t0) * 1000
        tps = len(generated) / max(t1 - t0, 1e-6)
        actual_hit = hit_ratio if hit_ratio is not None else (cached_len / max(len(full_token_ids), 1))

        return GenerationResult(
            mode=mode_name,
            prompt=prompt,
            output_text=output_text,
            generated_tokens=len(generated),
            ttft_ms=ttft,
            total_ms=total_ms,
            tokens_per_sec=tps,
            kv_reuse_ratio=actual_hit,
            prompt_tokens=len(full_token_ids),
            cached_tokens=cached_len,
            first_token_logits=first_token_logits,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=True)

    @staticmethod
    def _as_cache(past_kv):
        """Convert tuple-of-tuples KV to DynamicCache for newer transformers."""
        if past_kv is None or hasattr(past_kv, "get_seq_length"):
            return past_kv
        cache = DynamicCache()
        for layer_k, layer_v in past_kv:
            cache.update(layer_k, layer_v, len(cache))
        return cache

    @staticmethod
    def _normalize_past_kv(past_key_values) -> PastKVType:
        """
        Normalize past_key_values → tuple[ (key_Tensor, val_Tensor), ... ]
        one entry per layer, each tensor shape [batch, heads, seq, head_dim].

        Handles:
          • transformers < 4.38   : plain tuple of (k, v) tuples
          • transformers 4.38–4.44: DynamicCache with .to_legacy_cache()
          • transformers ≥ 4.45   : DynamicCache with .key_cache / .value_cache
        """
        if past_key_values is None:
            return None

        if hasattr(past_key_values, "get_seq_length"):
            return past_key_values

        if hasattr(past_key_values, "to_legacy_cache"):
            legacy = past_key_values.to_legacy_cache()
            return tuple((layer[0], layer[1]) for layer in legacy)

        return tuple((layer[0], layer[1]) for layer in past_key_values)

    def _encode_to_kv(
        self, token_ids: List[int], position_offset: int = 0
    ) -> PastKVType:
        """Run a forward pass and return only the KV cache (on CPU)."""
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=self.device)
        pos_ids = torch.arange(
            position_offset, position_offset + len(token_ids),
            dtype=torch.long, device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                position_ids=pos_ids,
                use_cache=True,
            )
        kv = out.past_key_values
        # Extract (k, v) tuples for CPU storage
        if hasattr(kv, "layers"):
            return tuple((l.keys.cpu(), l.values.cpu()) for l in kv.layers)
        if hasattr(kv, "key_cache") and hasattr(kv, "value_cache"):
            return tuple((k.cpu(), v.cpu()) for k, v in zip(kv.key_cache, kv.value_cache))
        return tuple((layer[0].cpu(), layer[1].cpu()) for layer in kv)

    def _store_prompt_chunks(self, token_ids: List[int]) -> None:
        """Cache all un-cached fixed-size chunks from this prompt."""
        pos = 0
        parent_hash = None
        for start, end, slice_ids in self.chunk_registry.split(token_ids):
            p_hash = chained_hash(slice_ids, parent_hash)
            c_hash = content_hash_from_tokens(slice_ids)
            if self.cache_manager.get(p_hash) is None:
                kv = self._encode_to_kv(slice_ids, position_offset=pos)
                chunk = CachedChunk(
                    chunk_id=p_hash,
                    text=self.tokenizer.decode(slice_ids),
                    token_ids=slice_ids,
                    past_key_values=kv,
                    position_start=pos,
                    position_end=pos + len(slice_ids),
                    prefix_hash=p_hash,
                    content_hash=c_hash,
                )
                self.cache_manager.store(chunk)
            parent_hash = p_hash
            pos += len(slice_ids)

    @staticmethod
    def _sample(logits: "torch.Tensor", temperature: float, do_sample: bool) -> int:
        if temperature != 1.0:
            logits = logits / temperature
        if do_sample:
            probs = torch.softmax(logits, dim=-1)
            return torch.multinomial(probs, 1).item()
        return logits.argmax(dim=-1).item()

    def cache_stats(self) -> Dict:
        return self.cache_manager.stats()

    def verify_correctness(self, max_new_tokens: int = 32) -> bool:
        """
        Quick self-test: runs greedy decode on a synthetic prompt with
        both BASELINE and CHUNK_KV_REUSE, verifies identical output.

        Returns True if outputs match, False otherwise.
        Use this to validate untested model architectures before trusting
        cached outputs in production.
        """
        test_prefix = (
            "The following is a factual statement about mathematics. "
            "Two plus two equals four. Three times three equals nine. "
            "The square root of sixteen is four. Pi is approximately "
            "three point one four one five nine. Euler's number e is "
            "approximately two point seven one eight."
        )
        test_query = "\n\nQuestion: What is two plus two?\nAnswer:"
        prompt = test_prefix + test_query

        # Warm the prefix
        self.warm(test_prefix)

        # Run both modes with greedy decoding
        r_base = self.generate(
            prompt, max_new_tokens=max_new_tokens,
            mode=GenerationMode.BASELINE, do_sample=False,
        )
        r_cached = self.generate(
            prompt, max_new_tokens=max_new_tokens,
            mode=GenerationMode.CHUNK_KV_REUSE, do_sample=False,
        )

        match = r_base.output_text == r_cached.output_text
        arch = type(self.model).__name__

        if match:
            log.info(
                "verify_correctness PASSED for %s — "
                "baseline and cached outputs are identical", arch,
            )
        else:
            log.warning(
                "verify_correctness FAILED for %s — "
                "outputs differ!\n  baseline: %r\n  cached:   %r",
                arch, r_base.output_text[:100], r_cached.output_text[:100],
            )

        return match
