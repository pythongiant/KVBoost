/*
 * pybind11 bindings for the KVBoost flash attention CUDA extension.
 *
 * Exposed as:  kvboost._flash_attn_cuda.flash_attn_fwd(Q, K, V, scale, causal)
 *              -> (O, L)
 *
 * All validation (shapes, dtypes, contiguity, device) happens here so the
 * CUDA kernel can assume well-formed inputs.
 */

#include <torch/extension.h>
#include "flash_attn.h"

// ── Input validation ──────────────────────────────────────────────────────────

static void check_inputs(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V)
{
    TORCH_CHECK(Q.dim() == 4, "Q must be 4-D [B, H, S, D], got ", Q.dim());
    TORCH_CHECK(K.dim() == 4, "K must be 4-D [B, H, S, D], got ", K.dim());
    TORCH_CHECK(V.dim() == 4, "V must be 4-D [B, H, S, D], got ", V.dim());

    TORCH_CHECK(Q.size(0) == K.size(0) && Q.size(0) == V.size(0),
                "Batch dimension mismatch: Q=", Q.size(0),
                " K=", K.size(0), " V=", V.size(0));
    TORCH_CHECK(Q.size(1) == K.size(1) && Q.size(1) == V.size(1),
                "Head dimension mismatch");
    TORCH_CHECK(K.size(2) == V.size(2),
                "K/V sequence length mismatch: K=", K.size(2), " V=", V.size(2));
    TORCH_CHECK(Q.size(3) == K.size(3) && Q.size(3) == V.size(3),
                "Head-dim mismatch");

    TORCH_CHECK(Q.scalar_type() == torch::kFloat16 ||
                Q.scalar_type() == torch::kBFloat16,
                "flash_attn only supports float16 or bfloat16, got ",
                Q.scalar_type());
    TORCH_CHECK(Q.scalar_type() == K.scalar_type() &&
                Q.scalar_type() == V.scalar_type(),
                "Q, K, V must have the same dtype");

    TORCH_CHECK(Q.is_cuda(), "Q must be on CUDA");
    TORCH_CHECK(K.is_cuda(), "K must be on CUDA");
    TORCH_CHECK(V.is_cuda(), "V must be on CUDA");

    TORCH_CHECK(Q.is_contiguous(), "Q must be contiguous");
    TORCH_CHECK(K.is_contiguous(), "K must be contiguous");
    TORCH_CHECK(V.is_contiguous(), "V must be contiguous");

    const int D = Q.size(3);
    TORCH_CHECK(D == 64 || D == 96 || D == 128,
                "flash_attn supports head_dim ∈ {64, 96, 128}, got ", D);
}

// ── Public entry point ────────────────────────────────────────────────────────

std::tuple<torch::Tensor, torch::Tensor> flash_attn_fwd(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    float scale,
    bool causal)
{
    check_inputs(Q, K, V);

    const int B = Q.size(0);
    const int H = Q.size(1);
    const int S = Q.size(2);

    // Allocate output and logsumexp tensors on the same device / stream
    auto O = torch::empty_like(Q);
    auto L = torch::empty({B, H, S}, Q.options().dtype(torch::kFloat32));

    flash_attn_fwd_dispatch(Q, K, V, O, L, scale, causal);

    // Synchronise so callers can safely read O/L immediately
    C10_CUDA_CHECK(cudaGetLastError());

    return {O, L};
}

// ── Python module ─────────────────────────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "KVBoost Flash Attention CUDA extension";
    m.def(
        "flash_attn_fwd",
        &flash_attn_fwd,
        "FlashAttention-2 forward pass.\n\n"
        "Args:\n"
        "    Q, K, V : float16/bfloat16 tensors of shape [B, H, S, D]\n"
        "    scale   : attention scale (typically 1/sqrt(D))\n"
        "    causal  : apply causal (lower-triangular) mask\n\n"
        "Returns:\n"
        "    (O, L) where O is [B,H,S,D] output and L is [B,H,S] logsumexp",
        py::arg("Q"),
        py::arg("K"),
        py::arg("V"),
        py::arg("scale"),
        py::arg("causal") = true);
}
