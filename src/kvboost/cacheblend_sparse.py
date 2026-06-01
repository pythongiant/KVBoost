"""Sparse (faithful) CacheBlend selective recompute.

The existing :class:`CacheBlendRecompute` does a **full** forward over all
cached tokens to measure deviation, then patches ~15% of positions — so a
warm request pays a full prefill's compute and the KV reuse buys nothing on
TTFT (it's actually slower than cold). That's the opposite of what the
CacheBlend paper (Yao et al., EuroSys '25, arXiv:2405.16444) does.

The paper's win comes from *selective recompute*: only the high-KV-deviation
(HKVD) tokens are forwarded through the layers, each attending to the blended
KV (cached values for non-HKVD positions, freshly recomputed for HKVD).
Per-layer cost is ~r×full, so the whole procedure is a fraction of a full
prefill → the reported 2.2-3.3× TTFT reduction.

This module implements that faithfully by reimplementing the decoder forward
layer-by-layer over a *subset* of token positions. It is architecture-coupled
(Llama / Qwen2-family: per-layer ``self_attn`` with q/k/v/o_proj, RMSNorm
input/post layernorms, an MLP submodule, and a model-level ``rotary_emb``).
A capability check (:func:`supports_sparse_recompute`) gates it; callers fall
back to the full-forward CacheBlend on unsupported models.

Correctness anchor
------------------
With the selected set = *all* positions, sparse recompute is mathematically a
clean full prefill (every token attends to all-fresh KV). The test suite
pins this: ``sparse_recompute(S=all) == model(all_ids).past_key_values``
bit-for-bit (fp tolerance). That validates the RoPE / GQA / norm / residual /
MLP reimplementation end-to-end. With S = top-r%, it's the paper's
approximation (no bit-exact ground truth; quality is evaluated separately).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .models import PastKVType

log = logging.getLogger(__name__)


# ── Architecture capability check ─────────────────────────────────────────────


def _inner_model(model):
    """Return the module exposing ``layers`` / ``embed_tokens`` / ``rotary_emb``.

    HF causal LMs wrap the decoder stack as ``model.model``; streaming shells
    wrap it again. Walk a couple of levels to find it.
    """
    for cand in (getattr(model, "model", None), model,
                 getattr(getattr(model, "model", None), "model", None)):
        if cand is not None and hasattr(cand, "layers") and hasattr(cand, "embed_tokens"):
            return cand
    return None


def supports_sparse_recompute(model) -> bool:
    """True iff the model exposes the Llama/Qwen2-style internals we need."""
    inner = _inner_model(model)
    if inner is None or not hasattr(inner, "rotary_emb"):
        return False
    try:
        layer0 = inner.layers[0]
    except (AttributeError, IndexError, TypeError):
        return False
    sa = getattr(layer0, "self_attn", None)
    needed = ("q_proj", "k_proj", "v_proj", "o_proj")
    if sa is None or not all(hasattr(sa, a) for a in needed):
        return False
    if not (hasattr(layer0, "input_layernorm")
            and hasattr(layer0, "post_attention_layernorm")
            and hasattr(layer0, "mlp")):
        return False
    return True


# ── RoPE / GQA helpers (Llama/Qwen2 "rotate_half" form) ───────────────────────


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    # q,k: [B, H, S, D]; cos,sin: [B, S, D] → unsqueeze head dim.
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_out = (q * cos) + (_rotate_half(q) * sin)
    k_out = (k * cos) + (_rotate_half(k) * sin)
    return q_out, k_out


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, n_kv, s, d = x.shape
    return (
        x[:, :, None, :, :]
        .expand(b, n_kv, n_rep, s, d)
        .reshape(b, n_kv * n_rep, s, d)
    )


# ── Core sparse recompute ─────────────────────────────────────────────────────


class SparseCacheBlend:
    """Faithful selective recompute via a manual layer-by-layer forward.

    Construct once per engine; call :meth:`recompute` per request with the
    assembled (stale) cached KV and the set of positions to refresh.
    """

    def __init__(self, recompute_ratio: float = 0.15, min_deviation: float = 0.01):
        self.recompute_ratio = recompute_ratio
        self.min_deviation = min_deviation

    # -- public API ---------------------------------------------------

    @torch.no_grad()
    def recompute(
        self,
        model,
        full_token_ids: Sequence[int],
        cached_kv: PastKVType,
        *,
        selected_positions: Optional[Sequence[int]] = None,
    ) -> PastKVType:
        """Return blended KV: cached for non-selected positions, freshly
        recomputed for ``selected_positions`` (each attending causally to the
        blended KV).

        ``selected_positions=None`` → auto-select HKVD tokens via the cheap
        early-layer deviation probe. Passing an explicit list (e.g. ``range(N)``
        for the bit-exact test, or a boundary-derived set) overrides selection.
        """
        inner = _inner_model(model)
        if inner is None:
            raise RuntimeError("sparse recompute: model internals not reachable")

        device = next(model.parameters()).device
        n = len(full_token_ids)
        ids = torch.tensor([list(full_token_ids)], dtype=torch.long, device=device)

        cfg = model.config
        n_heads = cfg.num_attention_heads
        n_kv = getattr(cfg, "num_key_value_heads", n_heads)
        layers = inner.layers
        num_layers = len(layers)

        # Move cached KV to device as a working [n_layers] list of [1,n_kv,N,d].
        blended: List[List[torch.Tensor]] = []
        for lk, lv in cached_kv:
            blended.append([lk.to(device), lv.to(device)])

        # Select which positions to refresh.
        if selected_positions is None:
            sel = self._select_hkvd(model, inner, ids, blended, n_heads, n_kv)
        else:
            sel = sorted(set(int(p) for p in selected_positions))
        if not sel:
            return cached_kv  # nothing to do

        sel_t = torch.tensor(sel, dtype=torch.long, device=device)
        pos_sel = sel_t.clone()  # original positions == position_ids for RoPE

        # Hidden states for the selected tokens only.
        h = inner.embed_tokens(ids[:, sel_t])  # [1, |S|, H]
        cos, sin = inner.rotary_emb(h, pos_sel.unsqueeze(0))  # [1,|S|,D] each

        # Causal mask: selected query i (at original pos sel[i]) attends to key
        # position j iff j <= sel[i]. Shape [1,1,|S|,N], additive.
        key_pos = torch.arange(n, device=device).view(1, 1, 1, n)
        allow = key_pos <= sel_t.view(1, 1, -1, 1)
        neg_inf = torch.finfo(h.dtype).min
        attn_mask = torch.where(
            allow, torch.zeros((), dtype=h.dtype, device=device),
            torch.full((), neg_inf, dtype=h.dtype, device=device),
        )

        n_rep = n_heads // n_kv
        for li in range(num_layers):
            layer = layers[li]
            sa = layer.self_attn
            scaling = getattr(sa, "scaling", None) or (1.0 / (h.shape[-1] / n_heads) ** 0.5)
            head_dim = sa.q_proj.out_features // n_heads

            normed = layer.input_layernorm(h)
            q = sa.q_proj(normed).view(1, -1, n_heads, head_dim).transpose(1, 2)
            k = sa.k_proj(normed).view(1, -1, n_kv, head_dim).transpose(1, 2)
            v = sa.v_proj(normed).view(1, -1, n_kv, head_dim).transpose(1, 2)
            q, k = _apply_rope(q, k, cos, sin)

            # Blend: overwrite the selected positions' K/V in the full cache.
            K_full = blended[li][0]
            V_full = blended[li][1]
            K_full = K_full.index_copy(2, sel_t, k)
            V_full = V_full.index_copy(2, sel_t, v)
            blended[li][0] = K_full
            blended[li][1] = V_full

            # Selected queries attend to the full blended KV (GQA-repeated).
            attn = F.scaled_dot_product_attention(
                q, _repeat_kv(K_full, n_rep), _repeat_kv(V_full, n_rep),
                attn_mask=attn_mask, scale=scaling,
            )
            attn = attn.transpose(1, 2).reshape(1, -1, n_heads * head_dim)
            attn = sa.o_proj(attn)
            h = h + attn
            h = h + layer.mlp(layer.post_attention_layernorm(h))

        # Return blended KV in the input format (tuple of (K,V)).
        return tuple((blended[li][0], blended[li][1]) for li in range(num_layers))

    # -- HKVD selection (cheap, no full forward) ----------------------

    @torch.no_grad()
    def _select_hkvd(
        self, model, inner, ids, blended, n_heads, n_kv,
    ) -> List[int]:
        """Pick the top-r% tokens by KV deviation using a TWO-LAYER probe.

        Layer-0 K/V are pure input projections (no cross-token attention), so
        they never deviate from the cache. Cross-chunk staleness first appears
        after one attention layer — so we recompute layers 0-1 for *all* tokens
        (cheap relative to the full stack: 2/num_layers), measure layer-1 K
        deviation vs the cached layer-1 K, and select the top-r%. The paper
        shows an early-layer HKVD set transfers to deeper layers.
        """
        device = ids.device
        n = ids.shape[1]
        cfg = model.config
        layers = inner.layers
        head_dim = layers[0].self_attn.q_proj.out_features // n_heads
        n_rep = n_heads // n_kv

        h = inner.embed_tokens(ids)  # all tokens [1,N,H]
        pos = torch.arange(n, device=device).unsqueeze(0)
        cos, sin = inner.rotary_emb(h, pos)
        full_mask = torch.triu(
            torch.full((n, n), torch.finfo(h.dtype).min, device=device, dtype=h.dtype),
            diagonal=1,
        ).view(1, 1, n, n)

        probe_layers = min(2, len(layers))
        layer1_fresh_k = None
        for li in range(probe_layers):
            layer = layers[li]
            sa = layer.self_attn
            scaling = getattr(sa, "scaling", None) or (1.0 / head_dim ** 0.5)
            normed = layer.input_layernorm(h)
            q = sa.q_proj(normed).view(1, n, n_heads, head_dim).transpose(1, 2)
            k = sa.k_proj(normed).view(1, n, n_kv, head_dim).transpose(1, 2)
            v = sa.v_proj(normed).view(1, n, n_kv, head_dim).transpose(1, 2)
            q, k = _apply_rope(q, k, cos, sin)
            if li == probe_layers - 1:
                layer1_fresh_k = k  # [1,n_kv,N,d]
            attn = F.scaled_dot_product_attention(
                q, _repeat_kv(k, n_rep), _repeat_kv(v, n_rep),
                attn_mask=full_mask, scale=scaling,
            )
            attn = attn.transpose(1, 2).reshape(1, n, n_heads * head_dim)
            h = h + sa.o_proj(attn)
            h = h + layer.mlp(layer.post_attention_layernorm(h))

        # Deviation = 1 - cosine(fresh layer-1 K, cached layer-1 K), per token,
        # averaged over heads.
        probe_idx = probe_layers - 1
        cached_k = blended[probe_idx][0]  # [1,n_kv,N,d]
        fk = layer1_fresh_k.permute(0, 2, 1, 3).reshape(1, n, -1)
        ck = cached_k.permute(0, 2, 1, 3).reshape(1, n, -1)
        dev = (1.0 - F.cosine_similarity(fk, ck, dim=-1)).squeeze(0)  # [N]

        num_to = max(1, int(n * self.recompute_ratio))
        above = dev > self.min_deviation
        if not above.any():
            return []
        masked = dev.clone()
        masked[~above] = -1.0
        k_pick = min(num_to, int(above.sum().item()))
        idx = masked.topk(k_pick).indices
        return sorted(int(i) for i in idx.tolist())
