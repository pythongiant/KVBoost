/*
 * FlashAttention-2 forward kernel for KVBoost.
 *
 * Implements tiled, online-softmax attention (Dao et al., 2022 / 2023) in
 * CUDA.  The kernel replaces the O(N²) HBM traffic of naive scaled dot-product
 * attention with O(N) reads/writes by keeping Q/K/V tiles in shared memory and
 * accumulating the output in registers.
 *
 * Supported:
 *   - float16 and bfloat16
 *   - Head dimensions: 64, 96, 128  (compile-time template param)
 *   - Causal masking
 *   - Arbitrary sequence lengths (S need not be divisible by block size)
 *
 * Layout (all tensors row-major):
 *   Q, K, V : [B, H, S, D]   (batch, heads, seq, head_dim)
 *   O       : [B, H, S, D]   output
 *   L       : [B, H, S]      logsumexp (needed for backward / fused ops)
 *
 * Each CUDA thread block handles one (batch_idx, head_idx) pair and tiles
 * over the sequence dimension.
 *
 * Tile sizes:
 *   Br (query tile rows)  = 64
 *   Bc (key/value cols)   = 64
 *   These fit comfortably in 48 KB shared memory for D=128, fp16.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <float.h>
#include <math.h>

#include "flash_attn.h"

// ── Tile sizes ────────────────────────────────────────────────────────────────
constexpr int BR = 64;   // query tile rows
constexpr int BC = 64;   // key/value tile cols

// ── Warp helpers ──────────────────────────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_max(float val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, mask);
    return val;
}

// ── fp16 / bf16 load helpers ──────────────────────────────────────────────────
template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t x);

template <>
__device__ __forceinline__ float to_float<__half>(__half x) { return __half2float(x); }

template <>
__device__ __forceinline__ float to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float x);

template <>
__device__ __forceinline__ __half from_float<__half>(float x) { return __float2half(x); }

template <>
__device__ __forceinline__ __nv_bfloat16 from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}

// ── Main kernel ───────────────────────────────────────────────────────────────
/*
 * Template params:
 *   scalar_t  : __half or __nv_bfloat16
 *   HEAD_DIM  : compile-time head dimension (64, 96, or 128)
 *
 * Grid  : (ceil(S/BR), B*H)   — each block handles one (batch,head) row tile
 * Block : (BR, 1, 1)          — one thread per query row in the tile
 */
template <typename scalar_t, int HEAD_DIM>
__global__ void flash_attn_fwd_kernel(
    const scalar_t* __restrict__ Q,      // [B, H, S, D]
    const scalar_t* __restrict__ K,      // [B, H, S, D]
    const scalar_t* __restrict__ V,      // [B, H, S, D]
    scalar_t*       __restrict__ O,      // [B, H, S, D]
    float*          __restrict__ L,      // [B, H, S]  logsumexp
    const int B,
    const int H,
    const int S,
    const float scale,
    const bool causal)
{
    // ── Identify which (batch, head, query-tile) this block handles ───────────
    const int bh_idx   = blockIdx.y;          // flat (batch * H + head) index
    const int b_idx    = bh_idx / H;
    const int h_idx    = bh_idx % H;
    const int q_tile   = blockIdx.x;          // which BR-sized query tile
    const int q_start  = q_tile * BR;         // first query row in this tile
    const int tid      = threadIdx.x;         // thread == query row within tile

    const int q_row = q_start + tid;          // absolute query position

    // Base pointer for this (batch, head) in each tensor
    const long bh_offset = ((long)b_idx * H + h_idx) * S * HEAD_DIM;
    const scalar_t* Q_bh = Q + bh_offset;
    const scalar_t* K_bh = K + bh_offset;
    const scalar_t* V_bh = V + bh_offset;
    scalar_t*       O_bh = O + bh_offset;
    float* L_bh = L + ((long)b_idx * H + h_idx) * S;

    // ── Shared memory for K and V tiles ──────────────────────────────────────
    // K tile: BC x HEAD_DIM, V tile: BC x HEAD_DIM
    __shared__ float K_tile[BC][HEAD_DIM];
    __shared__ float V_tile[BC][HEAD_DIM];

    // ── Thread-local registers: Q row, output accumulator ────────────────────
    float q_row_reg[HEAD_DIM];       // cached Q row (stays in registers)
    float acc[HEAD_DIM];             // output accumulator
    float m = -FLT_MAX;              // running row-max for online softmax
    float l = 0.0f;                  // running row-sum of exp

    // Load Q row into registers (or zero if out-of-bounds)
    if (q_row < S) {
        const scalar_t* q_ptr = Q_bh + q_row * HEAD_DIM;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; d++)
            q_row_reg[d] = to_float(q_ptr[d]);
    }
#pragma unroll
    for (int d = 0; d < HEAD_DIM; d++) acc[d] = 0.0f;

    // ── Tile loop over K/V ───────────────────────────────────────────────────
    const int num_kv_tiles = (S + BC - 1) / BC;

    for (int kv_tile = 0; kv_tile < num_kv_tiles; kv_tile++) {
        const int kv_start = kv_tile * BC;

        // Causal: skip future tiles entirely
        if (causal && kv_start > q_row)
            break;

        // Collaborative load of K tile (BC x HEAD_DIM)
        // Each thread loads one row of K (stride = 1 thread per K row using tid mod BC)
        for (int row = tid; row < BC; row += BR) {
            const int kv_row = kv_start + row;
            if (kv_row < S) {
                const scalar_t* k_ptr = K_bh + kv_row * HEAD_DIM;
#pragma unroll
                for (int d = 0; d < HEAD_DIM; d++)
                    K_tile[row][d] = to_float(k_ptr[d]);
            } else {
#pragma unroll
                for (int d = 0; d < HEAD_DIM; d++)
                    K_tile[row][d] = 0.0f;
            }
        }

        // Collaborative load of V tile
        for (int row = tid; row < BC; row += BR) {
            const int kv_row = kv_start + row;
            if (kv_row < S) {
                const scalar_t* v_ptr = V_bh + kv_row * HEAD_DIM;
#pragma unroll
                for (int d = 0; d < HEAD_DIM; d++)
                    V_tile[row][d] = to_float(v_ptr[d]);
            } else {
#pragma unroll
                for (int d = 0; d < HEAD_DIM; d++)
                    V_tile[row][d] = 0.0f;
            }
        }
        __syncthreads();

        // ── Compute QK^T for this thread's Q row ─────────────────────────────
        if (q_row < S) {
            // Find tile row-max for online softmax update
            float m_new = m;

            for (int j = 0; j < BC; j++) {
                const int kv_row = kv_start + j;
                if (kv_row >= S) break;

                // Causal: mask future keys
                if (causal && kv_row > q_row) break;

                float qk = 0.0f;
#pragma unroll
                for (int d = 0; d < HEAD_DIM; d++)
                    qk += q_row_reg[d] * K_tile[j][d];
                qk *= scale;

                m_new = fmaxf(m_new, qk);
            }

            // Rescale accumulator for new max
            float alpha = expf(m - m_new);
            float l_new = l * alpha;
#pragma unroll
            for (int d = 0; d < HEAD_DIM; d++) acc[d] *= alpha;

            // Accumulate softmax(QK^T) * V
            for (int j = 0; j < BC; j++) {
                const int kv_row = kv_start + j;
                if (kv_row >= S) break;
                if (causal && kv_row > q_row) break;

                float qk = 0.0f;
#pragma unroll
                for (int d = 0; d < HEAD_DIM; d++)
                    qk += q_row_reg[d] * K_tile[j][d];
                qk *= scale;

                float p = expf(qk - m_new);
                l_new += p;
#pragma unroll
                for (int d = 0; d < HEAD_DIM; d++)
                    acc[d] += p * V_tile[j][d];
            }

            m = m_new;
            l = l_new;
        }
        __syncthreads();
    }

    // ── Write output ─────────────────────────────────────────────────────────
    if (q_row < S) {
        float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
        scalar_t* o_ptr = O_bh + q_row * HEAD_DIM;
#pragma unroll
        for (int d = 0; d < HEAD_DIM; d++)
            o_ptr[d] = from_float<scalar_t>(acc[d] * inv_l);

        // Logsumexp: log(l) + m  (numerically stable)
        L_bh[q_row] = logf(l) + m;
    }
}

// ── Dispatcher ────────────────────────────────────────────────────────────────

void flash_attn_fwd_dispatch(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    torch::Tensor& O,
    torch::Tensor& L,
    float scale,
    bool causal)
{
    const int B = Q.size(0);
    const int H = Q.size(1);
    const int S = Q.size(2);
    const int D = Q.size(3);

    // Grid: (ceil(S/BR), B*H)
    dim3 grid((S + BR - 1) / BR, B * H);
    dim3 block(BR);

    auto dispatch_dtype = [&]<int HEAD_DIM>() {
        if (Q.scalar_type() == torch::kFloat16) {
            flash_attn_fwd_kernel<__half, HEAD_DIM><<<grid, block>>>(
                reinterpret_cast<const __half*>(Q.data_ptr()),
                reinterpret_cast<const __half*>(K.data_ptr()),
                reinterpret_cast<const __half*>(V.data_ptr()),
                reinterpret_cast<__half*>(O.data_ptr()),
                L.data_ptr<float>(),
                B, H, S, scale, causal);
        } else {
            flash_attn_fwd_kernel<__nv_bfloat16, HEAD_DIM><<<grid, block>>>(
                reinterpret_cast<const __nv_bfloat16*>(Q.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(K.data_ptr()),
                reinterpret_cast<const __nv_bfloat16*>(V.data_ptr()),
                reinterpret_cast<__nv_bfloat16*>(O.data_ptr()),
                L.data_ptr<float>(),
                B, H, S, scale, causal);
        }
    };

    if (D == 64)       dispatch_dtype.template operator()<64>();
    else if (D == 96)  dispatch_dtype.template operator()<96>();
    else if (D == 128) dispatch_dtype.template operator()<128>();
    else
        TORCH_CHECK(false, "flash_attn: unsupported head_dim=", D,
                    ". Supported: 64, 96, 128.");
}
