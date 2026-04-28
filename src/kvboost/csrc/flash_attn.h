#pragma once
#include <torch/extension.h>

// Forward declaration used by both flash_attn.cpp (pybind) and flash_attn_kernel.cu
void flash_attn_fwd_dispatch(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    torch::Tensor& O,
    torch::Tensor& L,
    float scale,
    bool causal);
