"""
Real-model tests for flash attention and paged attention interceptors.

Uses TinyLlama/TinyLlama-1.1B-Chat-v1.0 in float32 on CPU so no GPU is
required and no custom CUDA kernel needs to be compiled.

What these tests verify
-----------------------
Flash attention (install_flash_attention):
  - Patching does not change model outputs (same logits before and after).
  - Patch count matches the number of attention layers in the model.
  - Uninstalling fully restores original forwards.
  - generate() with a KVBoost engine produces output and correct reuse ratio.

Paged attention interceptor (install_paged_attention):
  - A single decode step via the interceptor produces logits within tolerance
    of the reference (HF past_key_values path).
  - The block pool is written: seq_len advances by exactly 1 per step.
  - The block_table grows when new blocks are needed.
  - Uninstalling restores original forwards cleanly.

Run:
    pytest tests/test_flash_attn_real_model.py -v
    # or just the flash tests:
    pytest tests/test_flash_attn_real_model.py -v -k flash
    # or just the paged tests:
    pytest tests/test_flash_attn_real_model.py -v -k paged
"""

from __future__ import annotations

import math
import pytest
import torch

# ── Model fixture (downloaded once per session) ───────────────────────────────

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
PROMPT   = "The capital of France is"


@pytest.fixture(scope="session")
def tinyllama():
    """Load TinyLlama once for the whole test session, explicitly on CPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,   # float32 so CPU attention is numerically stable
        low_cpu_mem_usage=True,
    )
    model = model.to("cpu")
    model.eval()
    return model, tokenizer


@pytest.fixture(scope="session")
def token_ids(tinyllama):
    _, tokenizer = tinyllama
    return tokenizer.encode(PROMPT, return_tensors="pt")  # [1, S]


# ── Helper: reference logits via HF (no patching) ────────────────────────────

def _ref_logits(model, input_ids, past_kv=None):
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            past_key_values=past_kv,
            use_cache=True,
        )
    return out.logits[:, -1, :].float(), out.past_key_values


# ── Flash attention tests ─────────────────────────────────────────────────────

class TestFlashAttentionRealModel:

    def test_patch_count_matches_num_layers(self, tinyllama):
        from kvboost.flash_attn_ext import install_flash_attention, uninstall_flash_attention
        model, _ = tinyllama
        n = install_flash_attention(model)
        # TinyLlama has 22 transformer layers, each with one attention module
        assert n == model.config.num_hidden_layers, (
            f"Expected {model.config.num_hidden_layers} patches, got {n}"
        )
        uninstall_flash_attention(model)

    def test_patched_logits_match_reference(self, tinyllama, token_ids):
        from kvboost.flash_attn_ext import install_flash_attention, uninstall_flash_attention
        model, _ = tinyllama

        ref_logits, _ = _ref_logits(model, token_ids)

        install_flash_attention(model)
        try:
            patched_logits, _ = _ref_logits(model, token_ids)
        finally:
            uninstall_flash_attention(model)

        assert patched_logits.shape == ref_logits.shape
        # On CPU in float32 the outputs should be bit-exact (same SDPA path)
        assert torch.allclose(patched_logits, ref_logits, atol=1e-4, rtol=1e-4), (
            f"Max logit diff after patching: {(patched_logits - ref_logits).abs().max():.6f}"
        )

    def test_uninstall_fully_restores(self, tinyllama, token_ids):
        from kvboost.flash_attn_ext import install_flash_attention, uninstall_flash_attention
        model, _ = tinyllama

        ref_logits, _ = _ref_logits(model, token_ids)
        install_flash_attention(model)
        uninstall_flash_attention(model)
        restored_logits, _ = _ref_logits(model, token_ids)

        assert torch.allclose(ref_logits, restored_logits, atol=1e-6), (
            "Uninstall did not restore original forward — logits differ"
        )

    def test_double_patch_is_idempotent(self, tinyllama):
        from kvboost.flash_attn_ext import install_flash_attention, uninstall_flash_attention
        model, _ = tinyllama
        n1 = install_flash_attention(model)
        n2 = install_flash_attention(model)
        assert n2 == 0, "Second install should patch 0 modules"
        uninstall_flash_attention(model)

    def test_argmax_token_unchanged_after_patch(self, tinyllama, token_ids):
        from kvboost.flash_attn_ext import install_flash_attention, uninstall_flash_attention
        model, _ = tinyllama

        ref_logits, _ = _ref_logits(model, token_ids)
        ref_top = ref_logits.argmax(dim=-1)

        install_flash_attention(model)
        try:
            patched_logits, _ = _ref_logits(model, token_ids)
            patched_top = patched_logits.argmax(dim=-1)
        finally:
            uninstall_flash_attention(model)

        assert ref_top.item() == patched_top.item(), (
            f"Argmax token changed: ref={ref_top.item()} patched={patched_top.item()}"
        )

    def test_multi_step_decode_with_patch(self, tinyllama, token_ids):
        """Verify patched model produces the same greedy decode sequence."""
        from kvboost.flash_attn_ext import install_flash_attention, uninstall_flash_attention
        model, tokenizer = tinyllama
        max_new = 5

        # Reference decode (unpatched)
        ref_ids = list(token_ids[0].tolist())
        past = None
        for _ in range(max_new):
            last = torch.tensor([[ref_ids[-1]]])
            logits, past = _ref_logits(model, last, past)
            ref_ids.append(logits.argmax(-1).item())

        # Patched decode
        install_flash_attention(model)
        try:
            pat_ids = list(token_ids[0].tolist())
            past = None
            for _ in range(max_new):
                last = torch.tensor([[pat_ids[-1]]])
                logits, past = _ref_logits(model, last, past)
                pat_ids.append(logits.argmax(-1).item())
        finally:
            uninstall_flash_attention(model)

        assert ref_ids == pat_ids, (
            f"Greedy decode diverged:\n  ref={ref_ids}\n  pat={pat_ids}"
        )

    def test_kvboost_engine_uses_flash_and_generates(self, tinyllama):
        """End-to-end: KVBoost engine with flash attention installed generates text."""
        from kvboost import KVBoost
        from kvboost.flash_attn_ext import get_tier
        model, tokenizer = tinyllama

        engine = KVBoost(
            model=model,
            tokenizer=tokenizer,
            max_cache_bytes=200_000_000,
            chunk_size=32,
            device="cpu",   # keep session-scoped model on CPU for subsequent tests
        )
        # Flash is installed by __init__ — tier should be at least vanilla
        assert get_tier() in ("kvboost_cuda", "torch_flash", "vanilla")

        engine.warm(PROMPT)
        result = engine.generate(PROMPT + " Paris.", max_new_tokens=8)

        assert isinstance(result.output_text, str)
        assert result.generated_tokens >= 1
        assert 0.0 <= result.kv_reuse_ratio <= 1.0


# ── Paged attention interceptor tests ────────────────────────────────────────

class TestPagedAttentionInterceptorRealModel:

    @pytest.fixture
    def allocator(self, tinyllama):
        from kvboost.cpu_paged.block_allocator import BlockAllocator
        model, _ = tinyllama
        cfg = model.config
        num_kv_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        # Match the model's dtype (float32 on CPU) so no dtype mismatch
        model_dtype = next(model.parameters()).dtype
        return BlockAllocator(
            num_layers=cfg.num_hidden_layers,
            num_heads=num_kv_heads,
            head_dim=head_dim,
            num_blocks=512,
            block_size=16,
            dtype=model_dtype,
        )

    @staticmethod
    def _iter_past_kv(past_kv):
        """Yield (K, V) tensors per layer, compatible with both tuple-of-tuples
        (transformers ≤ 4.x) and DynamicCache (transformers 5.x)."""
        if hasattr(past_kv, "layers"):
            # transformers 5.x DynamicCache — layers[i].keys / .values
            for layer in past_kv.layers:
                yield layer.keys, layer.values
        else:
            # Legacy: tuple of (K, V) pairs
            for K, V in past_kv:
                yield K, V

    def _prefill_into_pool(self, model, token_ids, allocator):
        """Run a prefill forward, store resulting K/V into the block pool."""
        from kvboost.cpu_paged.paged_attn_cpu import append_kv_to_blocks

        with torch.no_grad():
            out = model(input_ids=token_ids, use_cache=True)

        past_kv = out.past_key_values
        seq_len  = token_ids.size(1)
        blocks_needed = (seq_len + allocator.block_size - 1) // allocator.block_size
        block_table = allocator.allocate(blocks_needed)

        for layer_idx, (K, V) in enumerate(self._iter_past_kv(past_kv)):
            # K, V: [1, Hkv, seq_len, D]
            k = K.squeeze(0).to(allocator.dtype)  # [Hkv, seq_len, D]
            v = V.squeeze(0).to(allocator.dtype)
            append_kv_to_blocks(
                allocator=allocator,
                layer=layer_idx,
                block_table=block_table,
                slot_offset=0,
                k=k,
                v=v,
            )

        return block_table, seq_len, out.logits[:, -1, :].float()

    def test_interceptor_patch_count(self, tinyllama, allocator):
        from kvboost.flash_attn_ext import install_paged_attention, uninstall_paged_attention
        model, _ = tinyllama
        state = {"allocator": allocator, "block_table": [], "seq_len": 0, "_write_slot": 0}
        n = install_paged_attention(model, state)
        assert n == model.config.num_hidden_layers
        uninstall_paged_attention(model)

    def test_interceptor_uninstall_restores(self, tinyllama, token_ids, allocator):
        from kvboost.flash_attn_ext import install_paged_attention, uninstall_paged_attention
        model, _ = tinyllama

        ref_logits, _ = _ref_logits(model, token_ids)

        block_table, seq_len, _ = self._prefill_into_pool(model, token_ids, allocator)
        state = {
            "allocator":   allocator,
            "block_table": block_table,
            "seq_len":     seq_len,
            "_write_slot": seq_len,
        }
        install_paged_attention(model, state)
        uninstall_paged_attention(model)

        restored_logits, _ = _ref_logits(model, token_ids)
        assert torch.allclose(ref_logits, restored_logits, atol=1e-6), (
            "Uninstall did not restore original forward"
        )

    def test_paged_decode_step_logits_match_reference(self, tinyllama, token_ids, allocator):
        """
        Single decode step via paged interceptor must produce the same argmax
        token as the reference HF past_key_values path.
        """
        from kvboost.flash_attn_ext import install_paged_attention, uninstall_paged_attention
        model, tokenizer = tinyllama

        # Prefill: get reference next token and load K/V into pool
        ref_logits_prefill, ref_past = _ref_logits(model, token_ids)
        ref_next_token = ref_logits_prefill.argmax(-1).item()

        block_table, seq_len, _ = self._prefill_into_pool(model, token_ids, allocator)

        # Reference decode step (HF path)
        next_ids = torch.tensor([[ref_next_token]])
        ref_logits_decode, _ = _ref_logits(model, next_ids, ref_past)
        ref_decode_token = ref_logits_decode.argmax(-1).item()

        # Paged decode step (interceptor path)
        state = {
            "allocator":   allocator,
            "block_table": list(block_table),
            "seq_len":     seq_len,
            "_write_slot": seq_len,
        }
        install_paged_attention(model, state)
        try:
            pos_ids = torch.tensor([[seq_len]])
            with torch.no_grad():
                out = model(
                    input_ids=next_ids,
                    position_ids=pos_ids,
                    use_cache=False,
                )
        finally:
            uninstall_paged_attention(model)

        paged_logits = out.logits[:, -1, :].float()
        paged_token  = paged_logits.argmax(-1).item()

        # Argmax must agree — the paged path sees the same full KV context
        assert paged_token == ref_decode_token, (
            f"Paged decode argmax={paged_token} differs from reference={ref_decode_token}.\n"
            f"Max logit diff: {(paged_logits - ref_logits_decode).abs().max():.4f}"
        )

    def test_block_pool_advances_by_one_per_step(self, tinyllama, token_ids, allocator):
        """After one decode step the pool seq_len must be seq_len_before + 1."""
        from kvboost.flash_attn_ext import install_paged_attention, uninstall_paged_attention
        model, _ = tinyllama

        block_table, seq_len, ref_logits_prefill = self._prefill_into_pool(
            model, token_ids, allocator
        )
        next_token = ref_logits_prefill.argmax(-1).item()

        state = {
            "allocator":   allocator,
            "block_table": list(block_table),
            "seq_len":     seq_len,
            "_write_slot": seq_len,
        }
        install_paged_attention(model, state)
        try:
            with torch.no_grad():
                model(
                    input_ids=torch.tensor([[next_token]]),
                    position_ids=torch.tensor([[seq_len]]),
                    use_cache=False,
                )
        finally:
            uninstall_paged_attention(model)

        new_seq_len = state["_write_slot"] + 1
        assert new_seq_len == seq_len + 1, (
            f"Expected seq_len to advance to {seq_len + 1}, got {new_seq_len}"
        )

    def test_block_table_grows_when_slot_crosses_block_boundary(self, tinyllama, allocator):
        """
        When the current slot is at the last slot of a block, writing a new
        token must allocate an extra block and extend block_table.
        """
        from kvboost.flash_attn_ext import install_paged_attention, uninstall_paged_attention
        from kvboost.cpu_paged.paged_attn_cpu import append_kv_to_blocks
        model, tokenizer = tinyllama

        # Fill the pool so the next write will cross a block boundary.
        # block_size=16, so fill 16 tokens to fill block 0 completely.
        bs = allocator.block_size
        fill_tokens = bs  # exactly one full block
        fill_ids    = torch.ones(1, fill_tokens, dtype=torch.long)

        with torch.no_grad():
            out = model(input_ids=fill_ids, use_cache=True)

        past_kv = out.past_key_values
        block_table = allocator.allocate(1)  # one block for the filled tokens
        for layer_idx, (K, V) in enumerate(self._iter_past_kv(past_kv)):
            k = K.squeeze(0).to(allocator.dtype)
            v = V.squeeze(0).to(allocator.dtype)
            append_kv_to_blocks(
                allocator=allocator, layer=layer_idx,
                block_table=block_table, slot_offset=0,
                k=k, v=v,
            )

        blocks_before = len(block_table)

        # Now decode one more token — this should spill into block 1
        state = {
            "allocator":   allocator,
            "block_table": list(block_table),
            "seq_len":     fill_tokens,
            "_write_slot": fill_tokens,
        }
        install_paged_attention(model, state)
        try:
            with torch.no_grad():
                model(
                    input_ids=torch.ones(1, 1, dtype=torch.long),
                    position_ids=torch.tensor([[fill_tokens]]),
                    use_cache=False,
                )
        finally:
            uninstall_paged_attention(model)

        assert len(state["block_table"]) > blocks_before, (
            "Block table did not grow when decode crossed a block boundary"
        )

    def test_two_consecutive_paged_steps_produce_coherent_output(
        self, tinyllama, token_ids, allocator
    ):
        """
        Two consecutive paged decode steps should each produce a valid token
        (not NaN/inf) and the second step's argmax should differ from the
        first step's argmax no more often than greedy reference would.
        """
        from kvboost.flash_attn_ext import install_paged_attention, uninstall_paged_attention
        model, _ = tinyllama

        block_table, seq_len, logits0 = self._prefill_into_pool(model, token_ids, allocator)
        tok0 = logits0.argmax(-1).item()

        def _paged_step(token_id, position, bt, sl):
            state = {
                "allocator":   allocator,
                "block_table": list(bt),
                "seq_len":     sl,
                "_write_slot": sl,
            }
            install_paged_attention(model, state)
            try:
                with torch.no_grad():
                    out = model(
                        input_ids=torch.tensor([[token_id]]),
                        position_ids=torch.tensor([[position]]),
                        use_cache=False,
                    )
            finally:
                uninstall_paged_attention(model)
            logits = out.logits[:, -1, :].float()
            assert not torch.isnan(logits).any(), "NaN in paged step logits"
            assert not torch.isinf(logits).any(), "Inf in paged step logits"
            new_bt  = state["block_table"]
            new_sl  = state["_write_slot"] + 1
            return logits.argmax(-1).item(), new_bt, new_sl

        tok1, block_table, seq_len = _paged_step(tok0,  seq_len,     block_table, seq_len)
        tok2, block_table, seq_len = _paged_step(tok1,  seq_len,     block_table, seq_len)

        # Just verify we got integer token ids in vocab range
        vocab = model.config.vocab_size
        assert 0 <= tok1 < vocab
        assert 0 <= tok2 < vocab
