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

from .models import AssembledPrompt, CachedChunk, PastKVType, chunk_id_from_tokens
from .cache_manager import KVCacheManager
from .chunk_registry import ChunkRegistry, ChunkStrategy
from .prompt_assembler import AssemblyMode, PromptAssembler
from .selective_recompute import SelectiveRecompute

log = logging.getLogger(__name__)


class GenerationMode(str, enum.Enum):
    BASELINE = "baseline"
    PREFIX_CACHE = "prefix_cache"
    CHUNK_KV_REUSE = "chunk_kv_reuse"


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


class InferenceEngine:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        chunk_size: int = 128,
        max_chunks: int = 128,
        recompute_overlap: int = 16,
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

        # Sub-systems (CPU storage for cache tensors, move to device on use)
        self.cache_manager = KVCacheManager(
            max_chunks=max_chunks,
            disk_dir=disk_cache_dir,
            device="cpu",
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

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        **kwargs,
    ) -> "InferenceEngine":
        log.info("Loading model %s ...", model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        model.eval()
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

    # ------------------------------------------------------------------
    # Cache population helper
    # ------------------------------------------------------------------

    def warm(self, text: str, position_offset: int = 0) -> int:
        """
        Encode `text` and cache all its fixed-size chunks.
        Returns number of new chunks stored.

        Call this for your system prompt / few-shot examples / documents
        BEFORE calling generate() so the cache is already populated.
        """
        token_ids = self._encode(text)
        chunks_added = 0
        pos = position_offset

        for start, end, slice_ids in self.chunk_registry.split(token_ids, text):
            cid = chunk_id_from_tokens(slice_ids)
            if self.cache_manager.get(cid) is not None:
                pos += len(slice_ids)
                continue

            kv = self._encode_to_kv(slice_ids, position_offset=pos)
            chunk = CachedChunk(
                chunk_id=cid,
                text=self.tokenizer.decode(slice_ids),
                token_ids=slice_ids,
                past_key_values=kv,
                position_start=pos,
                position_end=pos + len(slice_ids),
            )
            self.cache_manager.store(chunk)
            pos += len(slice_ids)
            chunks_added += 1

        return chunks_added

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

        with torch.no_grad():
            past = None
            cur_ids = input_ids
            for step in range(max_new_tokens):
                out = self.model(input_ids=cur_ids, past_key_values=self._as_cache(past), use_cache=True)
                if first_token_time is None:
                    first_token_time = time.perf_counter()
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
        """Full chunk-level KV reuse + selective boundary recompute."""
        assembled = self.assembler.assemble(token_ids)

        # Selective recompute at seams (only if multiple chunks stitched)
        if len(assembled.chunk_boundaries) > 1:
            assembled = self.selective_recompute.apply(assembled, self.model)

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
        for start, end, slice_ids in self.chunk_registry.split(token_ids):
            cid = chunk_id_from_tokens(slice_ids)
            if self.cache_manager.get(cid) is None:
                kv = self._encode_to_kv(slice_ids, position_offset=pos)
                chunk = CachedChunk(
                    chunk_id=cid,
                    text=self.tokenizer.decode(slice_ids),
                    token_ids=slice_ids,
                    past_key_values=kv,
                    position_start=pos,
                    position_end=pos + len(slice_ids),
                )
                self.cache_manager.store(chunk)
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
