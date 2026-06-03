"""CPU tests for the CUDA-graph decode path.

Graph *capture* is CUDA-only, but the hard correctness logic — copying the
prefill KV into a StaticCache, driving cache_position, and the masked forward —
runs on CPU in ``force_eager`` mode. These tests assert that the static-cache
decode produces token-for-token identical greedy output to the ordinary eager
DynamicCache loop the engine uses. If this holds, only the (separately
self-checked) graph capture/replay remains GPU-validated.
"""
import pytest
import torch

transformers = pytest.importorskip("transformers")
from transformers import Qwen2Config, Qwen2ForCausalLM  # noqa: E402

from kvboost.cuda_graph_decode import CUDAGraphDecoder  # noqa: E402


def _tiny_model():
    cfg = Qwen2Config(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=256, tie_word_embeddings=False,
    )
    torch.manual_seed(0)
    return Qwen2ForCausalLM(cfg).eval()


def _greedy(lg):
    return int(lg.argmax(-1).item())


def _prefill(model, prompt_ids):
    with torch.no_grad():
        out = model(input_ids=prompt_ids, use_cache=True)
    seed = _greedy(out.logits[:, -1, :])
    return out.past_key_values, seed


def _eager_reference(model, past_kv, seed, L, n):
    """The engine's eager DynamicCache decode loop (greedy)."""
    toks, tok, cur = [], seed, L
    with torch.no_grad():
        for _ in range(n):
            o = model(
                input_ids=torch.tensor([[tok]]),
                position_ids=torch.tensor([[cur]]),
                past_key_values=past_kv, use_cache=True,
            )
            tok = _greedy(o.logits[:, -1, :])
            toks.append(tok)
            cur += 1
    return toks


@pytest.mark.parametrize("L,n", [(10, 12), (1, 8), (33, 5)])
def test_static_cache_decode_matches_eager(L, n):
    model = _tiny_model()
    torch.manual_seed(1)
    prompt = torch.randint(0, 128, (1, L))

    # Snapshot the prefill KV BEFORE the reference loop mutates its cache.
    past, seed = _prefill(model, prompt)
    kv_snap = tuple((l.keys.clone(), l.values.clone()) for l in past.layers)

    ref = _eager_reference(model, past, seed, L, n)

    dec = CUDAGraphDecoder(
        model, device="cpu", dtype=torch.float32, eos_token_id=-1,
        max_cache_len=64, force_eager=True,
    )
    assert dec.applicable()
    got = dec.decode(
        past_kv=kv_snap, start_pos=L, seed_token=seed, max_new_tokens=n,
        sample_fn=_greedy, as_cache=lambda x: x,
    )
    assert got == ref, f"static-cache decode diverged from eager: {got} != {ref}"


def test_seed_eos_returns_empty():
    model = _tiny_model()
    dec = CUDAGraphDecoder(
        model, device="cpu", dtype=torch.float32, eos_token_id=7,
        force_eager=True,
    )
    out = dec.decode(past_kv=(), start_pos=5, seed_token=7, max_new_tokens=10,
                     sample_fn=_greedy, as_cache=lambda x: x)
    assert out == []


def test_stops_on_eos():
    # eos = the first token the eager path would produce → decode stops at 1.
    model = _tiny_model()
    torch.manual_seed(1)
    prompt = torch.randint(0, 128, (1, 8))
    past, seed = _prefill(model, prompt)
    kv_snap = tuple((l.keys.clone(), l.values.clone()) for l in past.layers)
    first = _eager_reference(model, past, seed, 8, 1)[0]

    dec = CUDAGraphDecoder(
        model, device="cpu", dtype=torch.float32, eos_token_id=first,
        max_cache_len=64, force_eager=True,
    )
    out = dec.decode(past_kv=kv_snap, start_pos=8, seed_token=seed,
                     max_new_tokens=20, sample_fn=_greedy, as_cache=lambda x: x)
    assert out == [first]
