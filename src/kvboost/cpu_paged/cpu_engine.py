"""
CPUPagedEngine
==============
Subclass of InferenceEngine that replaces the decode loop with a paged
attention implementation backed by BlockAllocator.

Key differences from InferenceEngine
-------------------------------------
1. During _decode_with_kv():
   - Cached KV tensors (assembled.cached_past_kv) are loaded into the block
     pool via ChunkBlockMapper rather than passed as past_key_values.
   - The prefill forward and each decode step use paged_attention_fwd() to
     read K/V from the block pool.
   - New K/V computed during prefill/decode are appended back into the pool
     via append_kv_to_blocks().

2. The HuggingFace model's attention layers are monkey-patched to redirect
   their K/V reads/writes through the block pool during paged_attn phases.

3. Everything else (warm(), chunk hashing, recompute strategies, quantization,
   eviction) is fully inherited from InferenceEngine.

Usage
-----
    from kvboost.cpu_paged import CPUPagedEngine

    engine = CPUPagedEngine.from_pretrained(
        "Qwen/Qwen2.5-3B",
        max_cache_bytes=4_000_000_000,
        block_size=16,
        num_blocks=8192,
    )
    engine.warm("System prompt ...")
    result = engine.generate("System prompt ...\n\nUser question", max_new_tokens=256)
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..engine import InferenceEngine, GenerationMode, GenerationResult, RecomputeStrategy
from ..models import AssembledPrompt, CachedChunk, PastKVType, WarmResult
from ..cache_manager import KVCacheManager
from ..compat import last_logit_only, default_device
from ..flash_attn_ext import install_paged_attention, uninstall_paged_attention
from .block_allocator import BlockAllocator
from .paged_attn_cpu import paged_attention_fwd, append_kv_to_blocks
from .chunk_to_blocks import ChunkBlockMapper

log = logging.getLogger(__name__)


def _infer_model_dims(model: AutoModelForCausalLM) -> Tuple[int, int, int]:
    """
    Infer (num_layers, num_kv_heads, head_dim) from the model config.
    Works for Llama, Qwen2, Gemma, Mistral, Phi families.
    """
    cfg = model.config
    num_layers = cfg.num_hidden_layers

    # num_key_value_heads is used by GQA models; fall back to num_attention_heads
    num_kv_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    return num_layers, num_kv_heads, head_dim


class CPUPagedEngine(InferenceEngine):
    """
    InferenceEngine with a CPU paged attention decode loop.

    Additional constructor parameters (passed as kwargs):
        block_size  : tokens per physical block (default 16)
        num_blocks  : total blocks in the pre-allocated pool (default 4096)
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        *,
        max_cache_bytes: int,
        block_size: int = 16,
        num_blocks: int = 4096,
        **kwargs,
    ) -> None:
        # Force CPU device — paged attention is CPU-specific
        kwargs.setdefault("device", "cpu")
        super().__init__(model, tokenizer, max_cache_bytes=max_cache_bytes, **kwargs)

        num_layers, num_kv_heads, head_dim = _infer_model_dims(model)

        self.allocator = BlockAllocator(
            num_layers=num_layers,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            num_blocks=num_blocks,
            block_size=block_size,
            dtype=torch.float16,
        )
        self.chunk_mapper = ChunkBlockMapper(self.allocator)
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._block_size = block_size

        # Register eviction callback so blocks are freed when chunks leave cache
        self.cache_manager.register_eviction_callback(self.chunk_mapper.on_evict)

        log.info(
            "CPUPagedEngine: block_size=%d, num_blocks=%d, "
            "layers=%d, kv_heads=%d, head_dim=%d",
            block_size, num_blocks, num_layers, num_kv_heads, head_dim,
        )

    # ── from_pretrained override ──────────────────────────────────────────────

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        max_cache_bytes: int,
        block_size: int = 16,
        num_blocks: int = 4096,
        **kwargs,
    ) -> "CPUPagedEngine":
        kwargs["device"] = "cpu"
        device = "cpu"
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16,
            device_map=device,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        return cls(
            model,
            tokenizer,
            max_cache_bytes=max_cache_bytes,
            block_size=block_size,
            num_blocks=num_blocks,
            **kwargs,
        )

    # ── Paged decode override ─────────────────────────────────────────────────

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
        cacheable_prefix_len: Optional[int] = None,
    ) -> GenerationResult:
        """
        Override of InferenceEngine._decode_with_kv that uses the block pool
        for KV management instead of HuggingFace past_key_values.

        Strategy:
        1. If we have cached KV (past_kv), load it into the block pool.
        2. Run a standard HF prefill on live tokens — capture the new K/V
           and append them to the block pool (we don't use past_key_values
           from HF for the decode loop).
        3. Autoregressive decode using paged_attention_fwd() to read K/V and
           append_kv_to_blocks() to write each step's new K/V.

        Note: for the prefill step we still call model.forward() via HF because
        that's where the projections and RoPE live.  We intercept the returned
        past_key_values and copy them into the block pool rather than carrying
        them in memory as a growing tensor.
        """
        t0 = time.perf_counter()
        first_token_time = None
        generated: List[int] = []
        first_token_logits = None

        # ── Step 1: Load cached KV into block pool ────────────────────────────
        block_table: List[int] = []
        seq_len_in_pool: int = 0

        if past_kv is not None and cached_len > 0:
            # Reconstruct the list of CachedChunks from the assembled KV.
            # We re-derive them from the cache_manager using the prefix hashes
            # embedded in the full_token_ids split.
            cached_chunks = self._find_chunks_for_kv(full_token_ids, cached_len)
            if cached_chunks:
                for chunk in cached_chunks:
                    chunk_blocks = self.chunk_mapper.load_chunk(chunk)
                    block_table.extend(chunk_blocks)
                seq_len_in_pool = cached_len
            else:
                # Fallback: write past_kv directly into pool (no chunk mapping)
                block_table, seq_len_in_pool = self._load_past_kv_into_pool(
                    past_kv, cached_len
                )

        # ── Step 2: Prefill live tokens via HF forward ────────────────────────
        if live_ids:
            input_ids = torch.tensor([live_ids], dtype=torch.long, device=self.device)
            pos_ids = torch.arange(
                cached_len, cached_len + len(live_ids),
                dtype=torch.long, device=self.device,
            ).unsqueeze(0)

            # Run HF forward — we need the returned past_key_values to extract
            # the newly computed K/V for the live tokens.
            # Pass the cached portion as standard past_key_values so the model
            # sees the full context during the prefill projection.
            past_kv_dev = None
            if past_kv is not None:
                past_kv_dev = tuple(
                    (k.to(self.device), v.to(self.device))
                    for k, v in past_kv
                )

            with torch.no_grad(), last_logit_only(self.model):
                out = self.model(
                    input_ids=input_ids,
                    past_key_values=self._as_cache(past_kv_dev),
                    position_ids=pos_ids,
                    use_cache=True,
                )

            first_token_time = time.perf_counter()
            import numpy as np
            first_token_logits = out.logits[0, -1, :].cpu().float().numpy()

            # Extract and store only the *new* K/V (live tokens portion)
            new_kv = self._normalize_past_kv(out.past_key_values)
            block_table, seq_len_in_pool = self._append_new_kv_to_pool(
                new_kv, block_table, seq_len_in_pool,
                skip_prefix_len=cached_len,  # only append live token KVs
            )

            next_token = self._sample(out.logits[:, -1, :], temperature, do_sample)
            generated.append(next_token)

        else:
            # Fully cached — decode starting from last cached token
            last_id = full_token_ids[-1] if full_token_ids else 0
            input_ids = torch.tensor([[last_id]], dtype=torch.long, device=self.device)
            pos_ids = torch.tensor([[cached_len - 1]], dtype=torch.long, device=self.device)

            trimmed_kv = None
            if past_kv is not None and KVCacheManager.kv_seq_len(past_kv) > 1:
                trimmed_kv = KVCacheManager.slice_kv(past_kv, 0, cached_len - 1)
                trimmed_kv = tuple(
                    (k.to(self.device), v.to(self.device))
                    for k, v in trimmed_kv
                )

            with torch.no_grad(), last_logit_only(self.model):
                out = self.model(
                    input_ids=input_ids,
                    past_key_values=self._as_cache(trimmed_kv),
                    position_ids=pos_ids,
                    use_cache=True,
                )

            first_token_time = time.perf_counter()
            import numpy as np
            first_token_logits = out.logits[0, -1, :].cpu().float().numpy()

            new_kv = self._normalize_past_kv(out.past_key_values)
            block_table, seq_len_in_pool = self._append_new_kv_to_pool(
                new_kv, block_table, seq_len_in_pool,
                skip_prefix_len=cached_len - 1,
            )
            next_token = self._sample(out.logits[:, -1, :], temperature, do_sample)
            generated.append(next_token)

        # ── Step 3: Autoregressive decode via paged attention ─────────────────
        cur_pos = cached_len + len(live_ids)

        while len(generated) < max_new_tokens:
            if generated[-1] == self.tokenizer.eos_token_id:
                break

            cur_id = generated[-1]
            input_ids = torch.tensor([[cur_id]], dtype=torch.long, device=self.device)
            pos_tensor = torch.tensor([[cur_pos]], dtype=torch.long, device=self.device)

            # Run HF forward for one token — get K/V projections + logits.
            # We pass no past_key_values so the model computes Q/K/V from
            # scratch for this single token, then we handle the KV context
            # ourselves via the block pool.
            # NOTE: to avoid O(N) attention cost we hook into the model's
            # attention output using the paged KV gathered from the pool.
            # For simplicity here we fall back to full past_key_values for the
            # decode logits but keep the pool as the authoritative KV store.
            # A full paged decode (intercepting attn) is done via _paged_decode_step.
            next_token, block_table, seq_len_in_pool = self._paged_decode_step(
                token_id=cur_id,
                position=cur_pos,
                block_table=block_table,
                seq_len_in_pool=seq_len_in_pool,
                temperature=temperature,
                do_sample=do_sample,
            )
            generated.append(next_token)
            cur_pos += 1

        t1 = time.perf_counter()

        # Free the sequence's blocks (not the shared chunk blocks — those are
        # freed when chunks are evicted from KVCacheManager)
        self._free_sequence_blocks(block_table, cached_len)

        # Store newly computed prompt chunks for future cache hits
        self._store_prompt_chunks(full_token_ids, cacheable_prefix_len=cacheable_prefix_len)

        output_ids = generated[:]
        output_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        ttft_ms = ((first_token_time or t0) - t0) * 1000
        total_ms = (t1 - t0) * 1000
        tps = len(generated) / max((t1 - t0), 1e-6)

        return GenerationResult(
            mode=mode_name,
            prompt=prompt,
            output_text=output_text,
            generated_tokens=len(generated),
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            tokens_per_sec=tps,
            kv_reuse_ratio=hit_ratio or 0.0,
            prompt_tokens=len(full_token_ids),
            cached_tokens=cached_len,
            first_token_logits=first_token_logits,
        )

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _paged_decode_step(
        self,
        token_id: int,
        position: int,
        block_table: List[int],
        seq_len_in_pool: int,
        temperature: float,
        do_sample: bool,
    ) -> Tuple[int, List[int], int]:
        """
        One decode step using true paged attention.

        Installs the paged attention interceptor onto every attention module
        so that each layer's SDPA call:
          1. Writes the newly projected K/V token into the block pool.
          2. Reads the full K/V history from the pool (no contiguous tensor).
          3. Computes attention via paged_attention_fwd and returns the result.

        model.forward() is called with NO past_key_values — the block pool
        is the sole source of KV context, accessed mid-layer by the interceptor.
        """
        input_ids = torch.tensor([[token_id]], dtype=torch.long, device=self.device)
        pos_ids   = torch.tensor([[position]],  dtype=torch.long, device=self.device)

        # Shared mutable state threaded through all layer interceptors.
        # All layers share the same block_table pointer; each layer writes its
        # new K/V to the same slot offset (seq_len_in_pool) then increments it.
        # To avoid double-advancing seq_len across layers we track which layer
        # last advanced it — only layer 0's write drives the pointer forward;
        # subsequent layers write to the same slot without re-advancing.
        state: Dict = {
            "allocator":   self.allocator,
            "block_table": list(block_table),
            "seq_len":     seq_len_in_pool,
            # Track the slot we want every layer to write into.
            # _make_paged_sdpa advances state["seq_len"] on every call, but
            # we only want it advanced once (after the last layer).  We fix
            # this by resetting seq_len to seq_len_in_pool before each layer
            # and capturing the post-write value only from the final layer.
            "_write_slot": seq_len_in_pool,
            "_layer_count": 0,
            "_num_layers":  self._num_layers,
        }

        install_paged_attention(self.model, state)
        try:
            with torch.no_grad(), last_logit_only(self.model):
                out = self.model(
                    input_ids=input_ids,
                    position_ids=pos_ids,
                    use_cache=False,   # KV managed by pool, not HF cache
                )
        finally:
            uninstall_paged_attention(self.model)

        # After all layers have run, state["block_table"] and state["seq_len"]
        # reflect the pool after writing this token.  But _make_paged_sdpa
        # increments seq_len once per layer; we only want it incremented once
        # total (one new token slot).  Use _write_slot + 1 as the canonical
        # new seq_len.
        new_block_table = state["block_table"]
        new_seq_len     = state["_write_slot"] + 1

        next_token = self._sample(out.logits[:, -1, :], temperature, do_sample)
        return next_token, new_block_table, new_seq_len

    def _load_past_kv_into_pool(
        self, past_kv: PastKVType, seq_len: int
    ) -> Tuple[List[int], int]:
        """Write a PastKVType tensor directly into the block pool (fallback path)."""
        bs = self._block_size
        num_blocks_needed = (seq_len + bs - 1) // bs
        block_table = self.allocator.allocate(num_blocks_needed)

        for layer_idx, (K, V) in enumerate(past_kv):
            # K, V: [1, num_kv_heads, seq_len, head_dim]
            K = K.squeeze(0).to(self.allocator.dtype)  # [H, seq_len, D]
            V = V.squeeze(0).to(self.allocator.dtype)

            tokens_written = 0
            for blk_pos, block_id in enumerate(block_table):
                tokens_this_block = min(bs, seq_len - tokens_written)
                k_slice = K[:, tokens_written:tokens_written + tokens_this_block, :]
                v_slice = V[:, tokens_written:tokens_written + tokens_this_block, :]
                self.allocator.write_kv_chunk(
                    layer=layer_idx,
                    block_id=block_id,
                    slot_start=0,
                    k_chunk=k_slice,
                    v_chunk=v_slice,
                )
                tokens_written += tokens_this_block

        return block_table, seq_len

    def _append_new_kv_to_pool(
        self,
        new_kv: PastKVType,
        block_table: List[int],
        seq_len_in_pool: int,
        skip_prefix_len: int = 0,
    ) -> Tuple[List[int], int]:
        """
        Append only the newly computed tokens (skip_prefix_len onward) from
        new_kv into the block pool.
        """
        # All layers produce the same number of new tokens
        first_K = new_kv[0][0]  # [1, H, full_seq, D]
        full_seq = first_K.size(2)
        num_new = full_seq - skip_prefix_len
        if num_new <= 0:
            return block_table, seq_len_in_pool

        for layer_idx, (K, V) in enumerate(new_kv):
            # K, V: [1, num_kv_heads, full_seq, head_dim]
            k_new = K.squeeze(0)[:, skip_prefix_len:, :].to(self.allocator.dtype)
            v_new = V.squeeze(0)[:, skip_prefix_len:, :].to(self.allocator.dtype)

            block_table, seq_len_in_pool = append_kv_to_blocks(
                allocator=self.allocator,
                layer=layer_idx,
                block_table=block_table,
                slot_offset=seq_len_in_pool if layer_idx == 0 else seq_len_in_pool - num_new,
                k=k_new,
                v=v_new,
            )
            # Only advance seq_len_in_pool on first layer (same tokens for all layers)
            if layer_idx == 0:
                # seq_len_in_pool was advanced by append_kv_to_blocks
                pass

        return block_table, seq_len_in_pool

    def _find_chunks_for_kv(
        self, full_token_ids: List[int], cached_len: int
    ) -> List[CachedChunk]:
        """
        Re-look up the CachedChunks that correspond to the cached portion of
        full_token_ids so we can load them into the block pool.
        """
        chunk_size = self.chunk_registry.chunk_size
        chunks: List[CachedChunk] = []
        splits = self.chunk_registry.split(full_token_ids)

        for start, end, token_slice in splits:
            if end > cached_len:
                break
            match = self.cache_manager.lookup(token_slice, parent_hash=None)
            if match is not None:
                chunks.append(match.chunk)

        return chunks

    def _free_sequence_blocks(
        self, block_table: List[int], cached_len: int
    ) -> None:
        """
        Free the blocks that belong exclusively to this sequence (i.e., the
        blocks written during prefill/decode, not the shared chunk blocks).
        The shared chunk blocks are freed via the eviction callback.
        """
        bs = self._block_size
        num_cached_blocks = (cached_len + bs - 1) // bs if cached_len > 0 else 0
        # Blocks beyond the cached prefix are sequence-private
        private_blocks = block_table[num_cached_blocks:]
        if private_blocks:
            self.allocator.free(private_blocks)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def paged_stats(self) -> dict:
        return {
            "block_utilization": self.allocator.utilization(),
            "free_blocks": self.allocator.free_blocks,
            "used_blocks": self.allocator.used_blocks,
            "num_blocks": self.allocator.num_blocks,
            "loaded_chunks": self.chunk_mapper.loaded_chunks(),
            "blocks_used_by_chunks": self.chunk_mapper.blocks_used_by_chunks(),
        }
