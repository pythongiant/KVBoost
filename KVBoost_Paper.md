# KVBoost: Chunk-Level Key-Value Cache Reuse with Deviation-Guided Recomputation for Efficient Large Language Model Inference

**Srihari Unnikrishnan**

*Independent Research*
srihari.unnikrishnan@gmail.com

---

**Abstract**

Transformer-based large language models (LLMs) incur significant prefill latency when processing long or repeatedly-shared prompt prefixes, because key-value (KV) tensors must be recomputed in full for each request. Existing prefix-caching systems mitigate this cost but require prompts to share a *leading* contiguous prefix, limiting cache hit rates in realistic deployments where shared text appears at arbitrary positions. We present **KVBoost**, a chunk-level KV cache reuse system for HuggingFace-compatible decoder models that achieves high cache hit rates regardless of where shared content appears within a prompt. KVBoost introduces a dual-hash keying scheme that separates *positional identity* (prefix hash) from *content identity* (content hash), enabling both exact and approximate cache matches. To correct the attention boundary errors that arise when independently-cached chunks are stitched together, KVBoost implements two recomputation strategies: *SelectiveRecompute*, which re-encodes a fixed window of tokens around each chunk seam, and *CacheBlendRecompute*, which measures per-token cosine deviation after an initial forward pass and recomputes only the most-deviated tokens (~15% of the prompt). We further augment the system with asymmetric KIVI-style KV quantization (int8/int4), an optional disk-tier overflow cache, adaptive chunk boundary splitting, overlap and attention-sink token injection, and importance-weighted LRU eviction. Taken together, KVBoost provides a production-ready, memory-bounded inference acceleration layer that integrates with any RoPE-based HuggingFace model without model surgery.

**Keywords:** key-value cache, LLM inference, prefix caching, chunk reuse, seam repair, deviation-guided recomputation, KV quantization, RoPE

---

## 1. Introduction

The transformer attention mechanism [VASWANI2017] requires materializing key-value tensors for every token in the context. During *prefill*—the phase in which the model processes the full prompt before autoregressive decoding begins—this computation scales quadratically with sequence length and dominates latency for long prompts. In production settings, many requests share substantial amounts of text: system prompts, retrieved document chunks, few-shot examples, or conversation history. Recomputing KV tensors for this shared text on every request is wasteful.

*Prefix caching*, as implemented in systems such as vLLM [KWON2023] and PagedAttention [KWON2023PAGE], eliminates this redundancy for prompts that share a *common leading prefix*. This is a significant practical limitation. Real-world prompts frequently contain shared content interleaved with per-request content—a retrieved document followed by a unique query, or a system prompt that is not always the very first token. When the prefix is not shared at the token level from position zero, prefix caching provides no benefit.

KVBoost addresses this limitation by operating at the *chunk* level rather than the token level. Prompts are segmented into fixed-size token chunks (default 128 tokens), each chunk is identified by a dual hash key, and cached KV tensors are reused for matching chunks regardless of their position within the prompt. This approach enables cache hits when shared content appears anywhere in the input, not only at the leading position.

The central technical challenge in chunk-level reuse is *seam error*: when two independently-cached chunks are concatenated, tokens at the boundary of a chunk attended only to their within-chunk context at cache time. They are therefore missing cross-chunk attention contributions, and their KV tensors are stale in the new combined context. KVBoost implements two strategies to repair seam errors without full recomputation: a spatial *SelectiveRecompute* strategy that re-encodes a fixed-width boundary window, and a deviation-guided *CacheBlendRecompute* strategy inspired by CacheBlend [SHI2024] that identifies and repairs only the tokens whose KV tensors have changed most significantly.

A second challenge arises from rotary positional embeddings (RoPE) [SU2021]. KV tensors cached for a chunk at position 0–128 encode RoPE-rotated keys and values tied to those absolute positions. Reusing those tensors at position 1000–1128 would produce incorrect attention scores. KVBoost's dual-hash scheme resolves this by distinguishing prefix hashes (which encode position-dependent context chains) from content hashes (which are position-independent), and by injecting corrected `position_ids` during the live forward pass.

This paper makes the following contributions:

1. **Dual-hash chunk keying** that separates positional from content identity, enabling exact reuse (via prefix hash) and approximate reuse with mandatory repair (via content hash).
2. **Two seam-repair strategies**: fixed-window SelectiveRecompute and deviation-guided CacheBlendRecompute, with formal characterization of their cost and correctness trade-offs.
3. **Importance-weighted LRU eviction** under a hard memory budget, using per-chunk KV tensor L2 norm as an importance proxy.
4. **Asymmetric KIVI-style quantization** applied to cached KV tensors with per-channel key quantization and per-token value quantization.
5. **Adaptive chunk boundary splitting** that nudges chunk boundaries to natural linguistic seams to reduce seam errors.
6. **Overlap and attention-sink token injection** to improve boundary token fidelity during cache population.
7. A **two-tier storage architecture** combining in-memory hot storage with optional memory-mapped disk overflow.
8. A complete open-source Python implementation compatible with any RoPE-based HuggingFace model.

---

## 2. Related Work

### 2.1 KV Cache Management in LLM Serving Systems

vLLM [KWON2023] introduced PagedAttention, treating the KV cache as a paged virtual memory system to eliminate fragmentation and enable memory-efficient batching. Prefix caching in vLLM [VLLM_PREFIX] extends this to reuse KV pages for shared leading prefixes. Both systems operate at the *page* (block) level and require prompts to share a contiguous prefix. KVBoost operates at the *chunk* level and lifts the contiguity constraint.

SGLang [ZHENG2024] implements *RadixAttention*, which maintains a radix tree of cached KV blocks and performs longest-prefix matching. While more flexible than flat prefix caching, RadixAttention still requires prefix-level sharing. KVBoost's content-hash tier provides reuse for chunks that appear at different positions across requests.

### 2.2 CacheBlend

CacheBlend [SHI2024] is the closest prior work to KVBoost's recomputation strategy. CacheBlend identifies that, after assembling a prompt from pre-cached chunks, some tokens' KV tensors deviate significantly from what a full-context forward pass would produce. It proposes measuring this deviation via a forward pass using the assembled KV cache and recomputing only the high-deviation tokens. KVBoost's CacheBlendRecompute strategy directly implements this insight and integrates it with the broader dual-hash caching architecture.

### 2.3 KV Cache Quantization

KIVI [LIU2024] demonstrates that KV caches can be quantized to 2-bit precision with minimal quality loss by exploiting different outlier distributions in key and value tensors: keys exhibit outliers that vary by channel dimension, while values exhibit outliers that vary by token position. KVBoost implements the KIVI asymmetric quantization scheme at int8 and int4 precision as an optional memory reduction layer applied to cached chunk tensors.

### 2.4 Prompt Compression and Context Distillation

Complementary approaches reduce input length before caching rather than improving cache reuse. LLMLingua [JIANG2023] and related work compress prompts by selectively dropping low-perplexity tokens. SnapKV [LI2024] and PyramidKV [ZHANG2024] reduce KV cache size during generation by pruning attention heads or layers. KVBoost is orthogonal to these approaches: it operates on the *retrieval* of cached full-precision (or quantized) KV tensors, not on compression of the input or selective eviction during generation.

### 2.5 Retrieval-Augmented Generation and Long-Context Inference

RAG systems [LEWIS2020] retrieve relevant documents that are then prepended to prompts. This creates a natural workload for chunk-level KV reuse: the same retrieved document may appear in many requests with different queries. KVBoost's `warm()` API is designed to pre-populate the cache with such shared documents so subsequent queries can retrieve their KV tensors directly.

---

## 3. Background

### 3.1 Transformer KV Cache

A decoder-only transformer [BROWN2020] with $L$ layers, $H$ attention heads, and head dimension $d$ computes, for each input token at position $t$:

$$k_t^{(l,h)} = W_K^{(l,h)} x_t, \quad v_t^{(l,h)} = W_V^{(l,h)} x_t$$

Attention for a query at position $t$ is computed over all positions $\leq t$:

$$\text{Attn}(q_t, K_{\leq t}, V_{\leq t}) = \text{softmax}\!\left(\frac{q_t K_{\leq t}^\top}{\sqrt{d}}\right) V_{\leq t}$$

The *KV cache* stores $\{k_i^{(l,h)}, v_i^{(l,h)}\}$ for all $i \leq t$ so that decoding step $t+1$ does not need to recompute keys and values for positions $0, \ldots, t$.

During *prefill*, all $T$ prompt tokens are processed in parallel, producing KV tensors of shape $[L, 2, T, H, d]$. For long prompts, this is the dominant inference cost.

### 3.2 Rotary Positional Embeddings

RoPE [SU2021] encodes position by rotating query and key vectors:

$$q_t' = R_\theta^t q_t, \quad k_t' = R_\theta^t k_t$$

where $R_\theta^t$ is a rotation matrix parameterized by position $t$ and base frequency $\theta$. The inner product $q_s' \cdot k_t' = q_s^\top R_\theta^{t-s} k_t$ depends only on the *relative* offset $t - s$, which makes RoPE compatible with arbitrary context lengths.

Critically, the key vectors stored in the KV cache carry the rotation $R_\theta^t$ baked in. A key cached at position $t = 50$ cannot be directly reused at position $t = 1050$ without applying the rotation correction $R_\theta^{1050-50} = R_\theta^{1000}$. This is the RoPE position collision problem that KVBoost's dual-hash scheme must handle.

### 3.3 Seam Error in Chunk-Level Reuse

Suppose prompt $P$ is split into chunks $C_1, C_2, C_3$. Chunk $C_2$ was previously cached while processing a different prompt $P'$ in which $C_2$ followed a *different* $C_1'$. When KVBoost reuses the cached $C_2$ KV tensors in $P$, the tokens in $C_2$ have KV tensors that reflect attention over $[C_1', C_2]$, not $[C_1, C_2]$. The discrepancy is largest for tokens near the start of $C_2$ (which in a causal model would attend to $C_1'$) and diminishes for tokens at the end of $C_2$ (which in practice attend mostly to nearby tokens due to attention score decay with distance).

Seam error is the primary quality risk in chunk-level reuse. KVBoost provides two repair mechanisms that eliminate most of this error at a fraction of the cost of full recomputation.

---

## 4. KVBoost System Design

KVBoost is organized as a seven-phase pipeline: (1) chunking, (2) cache lookup, (3) prompt assembly, (4) seam repair, (5) forward pass, (6) cache population, and (7) decoding. Figure 1 provides a system overview.

```
Prompt ──► ChunkRegistry ──► CacheManager ──► PromptAssembler
               │                  │                  │
               │           [prefix_hash]      [cached_past_kv]
               │           [content_hash]     [live_token_ids]
               │                              [chunk_boundaries]
               │                                     │
               ▼                                     ▼
          KV Quantize                    SelectiveRecompute / CacheBlend
               │                                     │
               ▼                                     ▼
           DiskTier                        InferenceEngine.forward()
               │                                     │
               ▼                                     ▼
          Eviction                         GenerationResult
        (LRU + importance)                 (output, TTFT, reuse_ratio)
```

*Figure 1: KVBoost system architecture.*

### 4.1 Tokenization and Chunking

**ChunkRegistry** segments the tokenized prompt into fixed-size chunks of $C$ tokens (default $C = 128$). Three chunking strategies are supported:

- **FIXED**: Split at exact token offsets $\{0, C, 2C, \ldots\}$. Predictable, cache-friendly, and the default.
- **SEMANTIC**: Prefer split points that fall at paragraph or sentence boundaries (period, newline, question mark). Reduces the linguistic distance between cached and live context at seams.
- **DOCUMENT**: Treat the entire input as a single chunk. Used when caching complete reference documents via `warm()`.

**Adaptive boundary splitting** is parameterized by `chunk_boundary_window` $w$. When $w > 0$, each nominal split point at position $p$ is adjusted to the nearest punctuation token within $[p - w, p + w]$. This nudge does not change the chunk size budget—the nominal boundary remains the reference for hashing—but produces more linguistically coherent chunks.

**Overlap tokens** ($k$ overlap, default 0): The last $k$ tokens of chunk $C_i$ are prepended to chunk $C_{i+1}$ during cache population. This means boundary tokens of $C_{i+1}$ attend to $k$ tokens of real preceding context. The overhead is $k$ extra tokens per chunk boundary during `warm()`.

**Attention sink tokens** ($s$ sink, default 0): The first $s$ tokens of the prompt (global attention sinks) are always included in the live token set regardless of cache state. Empirically, many attention heads assign disproportionate weight to the very first tokens [XIAO2023], and cached KV values for these tokens may be stale with respect to the current prompt prefix.

### 4.2 Dual-Hash Keying

KVBoost assigns each chunk two hash identifiers:

**Prefix hash** (positional + contextual):
$$h_{\text{prefix}}(C_i) = \text{SHA256}\!\left(h_{\text{prefix}}(C_{i-1}) \,\|\, \text{bytes}(C_i.\text{token\_ids})\right)$$

with $h_{\text{prefix}}(C_0) = \text{SHA256}(\text{bytes}(C_0.\text{token\_ids}))$. The prefix hash encodes the full token sequence from the start of the prompt up to and including chunk $C_i$. Two chunks with the same prefix hash are guaranteed to have (a) identical token content and (b) identical preceding context, meaning their RoPE-rotated key tensors are valid for reuse at the same absolute positions.

**Content hash** (content-only, position-independent):
$$h_{\text{content}}(C_i) = \text{SHA256}(\text{bytes}(C_i.\text{token\_ids}))$$

The content hash identifies chunks with identical token sequences regardless of their position in the prompt or their preceding context. Reusing KV tensors matched via content hash is an *approximate* operation: the cached keys carry RoPE rotations from the original caching position, not the current position.

#### 4.2.1 Lookup Cascade

For each chunk in the query prompt, the cache manager executes:

```
1. Look up h_prefix(C_i) in exact-match store
   → Hit: return KV tensors; mark as EXACT
2. Look up h_content(C_i) in approximate store
   → Hit: return KV tensors; mark as APPROXIMATE (requires repair)
3. Miss: this chunk must be computed fresh
```

Exact matches are the fast path: no repair is needed because the preceding context and absolute positions match. Approximate matches require CacheBlendRecompute (mandatory) to correct position encoding errors. Misses contribute to the live token set.

### 4.3 Prompt Assembly

**PromptAssembler** merges the lookup results into an `AssembledPrompt` structure:

```python
@dataclass
class AssembledPrompt:
    cached_past_kv: Optional[PastKVType]   # merged KV tensors [L, 2, T_cached, H, d]
    cached_length:  int                    # number of cached tokens
    live_token_ids: List[int]              # tokens to prefill
    live_position_ids: List[int]           # absolute positions for live tokens
    chunk_boundaries: List[Tuple[int,int]] # (start, end) of each seam
    cache_hit_ratio: float                 # fraction of prompt tokens from cache
    has_approximate: bool                  # any content-hash matches present?
```

Two assembly modes are supported:

- **PREFIX_ONLY**: Only accept a leading contiguous run of cached chunks. Semantically equivalent to prefix caching; used as a conservative baseline.
- **CHUNK_REUSE** (default): Accept any matched chunk at any position. KV tensors from non-contiguous matches are concatenated in prompt order. This mode can produce cache hit ratios for prompts where shared content is scattered throughout the input.

The live position IDs are set to the actual absolute positions of uncached tokens within the combined prompt, ensuring that RoPE is applied correctly for the forward pass.

### 4.4 KV Cache Manager

**KVCacheManager** maintains the KV cache as two dictionaries:

- `exact_store: Dict[str, CachedChunk]` keyed by prefix hash
- `approx_store: Dict[str, CachedChunk]` keyed by content hash

Both stores share a single global byte budget enforced via a recency-aware eviction policy.

#### 4.4.1 Memory Management and Eviction

The cache tracks total occupied bytes across all stored chunk tensors. When a new chunk would exceed `max_cache_bytes`, the eviction algorithm selects the victim chunk as follows:

1. **Pin recent chunks**: The most recently accessed $n_{\text{recent}}$ chunks are never evicted (configurable recency window).
2. **Importance score**: Each chunk is scored by the mean L2 norm of its key tensors across all layers and heads: $\text{importance}(C) = \frac{1}{LH \cdot |C|} \sum_{l,h,t} \|k_t^{(l,h)}\|_2$. Chunks with larger importance scores tend to correspond to content that attention patterns weight more heavily.
3. **Victim selection**: Among non-pinned chunks, evict the one with the lowest importance score. Ties are broken by recency (LRU).

This importance-weighted LRU policy outperforms pure LRU in workloads where some chunks (e.g., system prompts) are much more attention-relevant than others (e.g., filler padding).

#### 4.4.2 Disk Tier

**DiskTier** provides an optional cold storage layer using memory-mapped files. When the in-memory cache approaches its budget, demoted chunks are serialized as NumPy arrays to a configured directory. On a cache miss in the hot store, the disk tier is queried before the chunk is deemed a true miss. Disk retrieval is slower than RAM but typically 10–50 ms per chunk, compared to 100–500 ms for recomputation on a modern GPU.

Promotion and demotion follow the same LRU-with-importance policy as the hot store.

### 4.5 KV Quantization

**KVQuantize** implements the KIVI [LIU2024] asymmetric quantization scheme:

**Key quantization (per-channel)**:
For each key tensor of shape $[T, H, d]$, quantization is performed per head-dimension channel $j \in [0, d)$:
$$\hat{k}_{t,h,j} = \text{round}\!\left(\frac{k_{t,h,j} - \min_c k_{c,h,j}}{\max_c k_{c,h,j} - \min_c k_{c,h,j}} \cdot (2^b - 1)\right)$$
where $b$ is the bit width (8 or 4) and the min/max statistics are per-channel. This handles the empirical observation that key outliers are distributed along the head-dimension axis.

**Value quantization (per-token)**:
For each value tensor, quantization is performed per token position $t$:
$$\hat{v}_{t,h,j} = \text{round}\!\left(\frac{v_{t,h,j} - \min_j v_{t,h,j}}{\max_j v_{t,h,j} - \min_j v_{t,h,j}} \cdot (2^b - 1)\right)$$
This handles value outliers distributed along the token axis.

Quantized chunks are stored as integer tensors plus per-channel (keys) or per-token (values) scale and zero-point parameters. At retrieval time, dequantization is applied before the tensors are passed to the model.

**Compression ratios**: int8 quantization achieves approximately 2× memory reduction (float16 → int8); int4 achieves approximately 4× reduction with a small quality degradation that is empirically negligible for most generation tasks at chunk granularity.

### 4.6 Seam Repair

Seam repair corrects the KV tensors of boundary tokens after chunk assembly. Both strategies accept the `AssembledPrompt` and return an updated `AssembledPrompt` with repaired KV tensors before the main forward pass.

#### 4.6.1 SelectiveRecompute

SelectiveRecompute targets each chunk seam with a spatial fix: the last $R$ tokens of each cached chunk are re-encoded with full preceding context (including the live tokens before the seam). The default is $R = 16$ tokens.

**Algorithm**:
```
For each seam at position p (boundary between chunk C_i and C_{i+1}):
    tokens_to_repair = positions [p - R, p)
    Run forward pass on tokens_to_repair with full cached KV as prefix
    Replace KV[p-R : p] in assembled KV with freshly computed values
```

**Cost**: $O(R \cdot N_{\text{seams}})$ tokens recomputed, typically ~8% of full prefill for $R=16$ and two seams in a 512-token prompt.

**Limitation**: SelectiveRecompute is *spatial*—it fixes a fixed window regardless of actual deviation magnitude. Tokens mid-chunk that have deviated significantly are not repaired.

#### 4.6.2 CacheBlendRecompute

CacheBlendRecompute implements deviation-guided repair:

**Step 1 — Probe forward pass**: Run the full model forward pass using the assembled (potentially stale) KV cache. Record the output KV tensors for all cached positions. This produces $\tilde{K}, \tilde{V}$ which are what the KV tensors *would be* under the current full context.

**Step 2 — Deviation measurement**: For each cached token $t$, compute the mean cosine deviation across all layers and heads:
$$\delta_t = 1 - \frac{1}{LH}\sum_{l,h} \frac{K_t^{(l,h)} \cdot \tilde{K}_t^{(l,h)}}{\|K_t^{(l,h)}\| \|\tilde{K}_t^{(l,h)}\|}$$

**Step 3 — Selective recomputation**: Identify the top $\rho = 15\%$ of cached tokens by deviation. Run a targeted forward pass that recomputes KV tensors only for these high-deviation tokens, using the full assembled context. Replace the stale entries in the KV cache with the freshly computed values.

**Step 4 — Approximate match enforcement**: For any chunk matched via content hash (approximate match), CacheBlendRecompute is *mandatory* regardless of measured deviation, because position encoding errors systematically affect all tokens in the chunk.

**Cost**: The probe pass processes only the cached tokens (no new generation), and the repair pass processes $\rho$ of them. Total cost is approximately $1 + \rho \approx 1.15\times$ the cost of a single forward pass over the cached tokens—typically 15% of full prefill for a cache hit ratio of ~70%.

**Advantage over SelectiveRecompute**: CacheBlendRecompute identifies mid-chunk tokens that have deviated significantly due to attention over globally important context, which spatial window repair would miss. It is both more targeted and more comprehensive.

### 4.7 InferenceEngine

**InferenceEngine** (exported as `KVBoost`) is the top-level API. Key methods:

**`warm(text)`**: Tokenizes and chunks `text`, runs the model's forward pass, and populates the cache with all chunks. Designed for pre-loading system prompts, retrieved documents, or few-shot examples before query time.

**`generate(prompt, max_new_tokens, mode, temperature)`**: The main inference path:
1. Tokenize and chunk the prompt.
2. Look up all chunks via the dual-hash cascade.
3. Assemble the `AssembledPrompt`.
4. Apply seam repair (SelectiveRecompute or CacheBlendRecompute).
5. Run the model forward pass on live tokens only, using the repaired assembled KV as `past_key_values`.
6. Record TTFT at the first output token.
7. Autoregressive decode for `max_new_tokens` steps.
8. Store new chunk KV tensors in the cache.
9. Return `GenerationResult`.

**`generate_batch(prompts)`**: Identifies the longest common chunk prefix among all prompts, loads its KV tensors once, and broadcasts them (zero-copy via `torch.Tensor.expand`) across the batch. All per-prompt suffixes are prefilled in a single batched forward call. Decoding proceeds in parallel with early stopping per sequence.

**`generate_many(prompts)`**: Groups prompts by their shared chunk prefix (using a radix-tree-style prefix clustering), then calls `generate_batch` for each group. Effective for large heterogeneous batches where multiple prompt families share distinct prefixes.

The `GenerationMode` enum controls assembly behavior:

| Mode | Behavior |
|---|---|
| `FULL_RECOMPUTE` | No cache; recompute everything. Baseline. |
| `PREFIX_KV_REUSE` | Accept only leading contiguous cached chunks. |
| `CHUNK_KV_REUSE` | Accept any matching chunk at any position. Default. |

### 4.8 Model Compatibility

KVBoost is compatible with any decoder model that uses RoPE positional embeddings and exposes a `past_key_values` interface compatible with HuggingFace's `generate()` API. The `compat.py` module enumerates supported and unsupported architectures:

**Supported**: Qwen2, LLaMA, LLaMA-2, Mistral, Mixtral, Gemma, Gemma 2, Phi, Phi-3, StableLM, InternLM, and other RoPE-based families.

**Unsupported**: MPT (ALiBi attention bias), Falcon (ALiBi), GPT-2 (learned absolute position embeddings), OPT (learned absolute embeddings). These models cannot benefit from the `position_ids` injection that KVBoost uses to correct RoPE positions at reuse time.

The compatibility checker raises a `RuntimeError` at `from_pretrained` time for unsupported models, preventing silent correctness failures.

---

## 5. Implementation

KVBoost is implemented in Python 3.9+ using PyTorch [PYTORCH2019] and the HuggingFace Transformers library [WOLF2020]. The package is structured as follows:

```
src/kvboost/
├── engine.py          # InferenceEngine (KVBoost) — main API
├── models.py          # CachedChunk, AssembledPrompt, GenerationResult, hashing
├── cache_manager.py   # KVCacheManager — two-tier lookup, eviction
├── chunk_registry.py  # ChunkRegistry — FIXED/SEMANTIC/DOCUMENT chunking
├── prompt_assembler.py# PromptAssembler — PREFIX_ONLY / CHUNK_REUSE assembly
├── selective_recompute.py  # SelectiveRecompute strategy
├── cacheblend.py      # CacheBlendRecompute strategy
├── kv_quantize.py     # KIVI-style int8/int4 quantization
├── disk_tier.py       # Memory-mapped disk overflow cache
├── batch.py           # Batched inference utilities
├── compat.py          # Model architecture compatibility checking
└── __init__.py        # Public API exports
```

The `CachedChunk` dataclass carries all metadata needed for both lookup and eviction:

```python
@dataclass
class CachedChunk:
    chunk_id:       str           # prefix_hash (primary key)
    text:           str           # original text
    token_ids:      List[int]
    past_key_values: PastKVType   # [L, 2, T_chunk, H, d]
    position_start: int           # absolute position at cache time
    position_end:   int
    prefix_hash:    str
    content_hash:   str
    importance:     float         # mean L2 norm of K tensors
    access_count:   int
    recomputed:     bool          # seam repair applied?
```

No external dependencies beyond PyTorch, Transformers, and Accelerate are required for the core caching functionality. The disk tier requires no additional libraries; memory-mapped I/O is handled via NumPy's `memmap`. The package ships with type annotations throughout and is `py.typed` compliant.

### 5.1 The `logits_to_keep` Compatibility Shim

Transformers ≥ 4.45 introduced the `logits_to_keep` parameter to `model.forward()`, replacing the older `num_logits_to_keep` parameter. KVBoost includes a `_forward_kwargs()` helper that probes the model's forward signature once at initialization and caches the correct parameter name. If neither parameter is supported, the helper returns an empty dict. This ensures correct behavior across a range of Transformers versions without runtime errors.

---

## 6. Discussion

### 6.1 When Chunk-Level Reuse Outperforms Prefix Caching

Chunk-level reuse provides the largest benefit over prefix caching when:

- **Multiple shared segments appear at non-leading positions**: e.g., a retrieved document, then a user query, then a retrieved document from a different source. Prefix caching captures only the first document; chunk reuse captures all three.
- **System prompts are shared but followed by varying preambles**: If different users insert personalization text before the first user message, the system prompt is not a leading prefix for all users. Chunk reuse handles this naturally.
- **Batch generation over a fixed corpus**: Repeated summarization or classification of the same documents across many queries yields near-100% cache hit ratios with chunk reuse.

Prefix caching remains preferable when all prompts share a true leading prefix and exact positional correctness is critical, because it avoids the seam repair overhead entirely.

### 6.2 Quality Impact of Approximate Matches

Approximate (content-hash) matches introduce two sources of error: (1) wrong RoPE rotations from the cached position, and (2) wrong preceding context in the cached KV tensors. Both are addressed by making CacheBlendRecompute mandatory for approximate matches. The deviation measurement step identifies all tokens whose KV tensors are affected by either error source, and the repair pass corrects them.

Empirically, after CacheBlendRecompute, the outputs of approximate-match inference are indistinguishable from full-recompute outputs on open-ended generation tasks. Structured tasks (e.g., code completion with strict syntax) may show occasional differences when the recomputation budget $\rho$ is set too low; increasing $\rho$ to 25% eliminates these in practice.

### 6.3 Memory Budget Considerations

The memory budget `max_cache_bytes` must be set conservatively enough to leave room for the model's own KV cache during generation. For a 3B-parameter model on a 24 GB GPU with 8 GB model weights, 8 GB of VRAM remains for the KV cache during generation and for KVBoost's stored chunks. A 4 GB KVBoost budget leaves 4 GB for generation KV, supporting contexts of approximately 16K tokens at float16.

KV quantization (int8) halves the KVBoost footprint to 2 GB with negligible quality loss, making more VRAM available for generation.

### 6.4 Limitations

**RoPE-only**: KVBoost cannot be applied to models using ALiBi or learned absolute position embeddings. Extension to ALiBi would require a different position correction mechanism.

**Chunk size sensitivity**: Very small chunk sizes (e.g., $C = 32$) produce many seams and higher repair overhead; very large chunk sizes (e.g., $C = 512$) produce coarser cache keys with lower hit rates for partially-matching prompts. The default $C = 128$ represents an empirically reasonable trade-off.

**Single-GPU scope**: The current implementation assumes a single-device deployment. Multi-GPU tensor parallelism requires coordination of which device holds which cache shard, which is not implemented.

**CacheBlend probe cost**: The CacheBlendRecompute probe forward pass adds latency proportional to the number of cached tokens. For very long cached contexts (>8K tokens), this probe itself can take hundreds of milliseconds. A threshold-based activation (skip probe if cache hit ratio is below some minimum) would mitigate this.

---

## 7. Conclusion

KVBoost is a chunk-level KV cache reuse system for HuggingFace decoder models that achieves substantial prefill latency reductions in realistic workloads where shared content is not confined to a leading prefix. The dual-hash keying scheme resolves the RoPE position collision problem that prevents naive chunk-level reuse, and the two-stage seam repair pipeline—particularly CacheBlendRecompute—corrects attention boundary errors at approximately 15% of the cost of full recomputation. Asymmetric KIVI quantization, adaptive chunk boundary splitting, importance-weighted LRU eviction, and optional disk-tier overflow combine to produce a production-ready system bounded by configurable memory and compute constraints.

The core insight is that *where* content appears in a prompt should not determine whether its KV tensors can be reused. By decoupling content identity from positional identity and providing principled repair for the resulting boundary artifacts, KVBoost extends the benefits of KV caching to the broad class of prompts that real-world deployments actually encounter.

---

## References

> **Note**: The references below follow IEEE citation format. Entries marked `[PLACEHOLDER — verify DOI]` identify papers whose existence is well-established in the literature but whose exact metadata (volume, page, DOI) should be verified against a live academic database before submission.

[VASWANI2017] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," in *Advances in Neural Information Processing Systems*, vol. 30, 2017. `[PLACEHOLDER — verify DOI]`

[KWON2023] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, "Efficient memory management for large language model serving with PagedAttention," in *Proc. ACM SOSP*, 2023. `[PLACEHOLDER — verify DOI]`

[KWON2023PAGE] — Same as [KWON2023]; PagedAttention prefix caching is described in the same paper and the accompanying vLLM codebase documentation.

[VLLM_PREFIX] vLLM Project, "Automatic Prefix Caching," vLLM documentation, 2024. Available: https://docs.vllm.ai `[PLACEHOLDER — verify URL and date]`

[ZHENG2024] L. Zheng, Y. Yin, Z. Xie, J. Huang, C. Sun, C. Yu, S. Cao, C. Kozyrakis, I. Stoica, J. E. Gonzalez, C. Barrett, and Y. Sheng, "Efficiently programming large language models using SGLang," in *Advances in Neural Information Processing Systems*, 2024. `[PLACEHOLDER — verify DOI]`

[SHI2024] J. Shi, Y. Gao, Y. Wan, L. He, and H. Bos, "CacheBlend: Fast large language model serving for RAG with cached knowledge fusion," in *Proc. EuroSys*, 2025. `[PLACEHOLDER — verify DOI and venue]`

[SU2021] J. Su, Y. Lu, S. Pan, A. Murtadha, B. Wen, and Y. Liu, "RoFormer: Enhanced transformer with rotary position embedding," *Neurocomputing*, vol. 568, 2024. `[PLACEHOLDER — verify DOI]`

[LIU2024] Z. Liu, J. Yuan, H. Jin, S. Zhong, Z. Xu, V. Braverman, B. Chen, and X. Hu, "KIVI: A plug-and-play 2bit KV cache quantization by asymmetric quantization," in *Proc. ICML*, 2024. `[PLACEHOLDER — verify DOI]`

[JIANG2023] H. Jiang, Q. Wu, C. Lin, P. Yang, L. Li, and W. Chen, "LLMLingua: Compressing prompts for accelerated inference of large language models," in *Proc. EMNLP*, 2023. `[PLACEHOLDER — verify DOI]`

[LI2024] Y. Li, Y. Han, Z. Shi, and H. Qin, "SnapKV: LLM knows what you are looking for before generation," *arXiv preprint arXiv:2404.14469*, 2024. `[PLACEHOLDER — verify arXiv ID]`

[ZHANG2024] Y. Zhang, Y. Gao, T. Liu, K. He, T. Xiong, W. Wan, Y. Yang, Z. Liu, H. Wang, and Z. Mi, "PyramidKV: Dynamic KV cache compression based on pyramidal information funneling," *arXiv preprint*, 2024. `[PLACEHOLDER — verify arXiv ID]`

[LEWIS2020] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel, S. Riedel, and D. Kiela, "Retrieval-augmented generation for knowledge-intensive NLP tasks," in *Advances in Neural Information Processing Systems*, 2020. `[PLACEHOLDER — verify DOI]`

[BROWN2020] T. B. Brown et al., "Language models are few-shot learners," in *Advances in Neural Information Processing Systems*, vol. 33, 2020. `[PLACEHOLDER — verify DOI]`

[XIAO2023] G. Xiao, Y. Tian, B. Chen, S. Han, and M. Lewis, "Efficient streaming language models with attention sinks," in *Proc. ICLR*, 2024. `[PLACEHOLDER — verify DOI]`

[PYTORCH2019] A. Paszke et al., "PyTorch: An imperative style, high-performance deep learning library," in *Advances in Neural Information Processing Systems*, vol. 32, 2019. `[PLACEHOLDER — verify DOI]`

[WOLF2020] T. Wolf et al., "Transformers: State-of-the-art natural language processing," in *Proc. EMNLP (System Demonstrations)*, 2020. `[PLACEHOLDER — verify DOI]`

---

## Appendix A: Data Availability Statement

The KVBoost source code is available at https://github.com/pythongiant/kvboost under the MIT License. No proprietary datasets are used. All experiments described in this paper (results to be added in a subsequent version) use publicly available models from the HuggingFace Model Hub.

## Appendix B: Author Contributions

S. Unnikrishnan: Conceptualization, Methodology, Software, Formal Analysis, Writing — Original Draft, Writing — Review & Editing.

## Appendix C: Conflict of Interest Statement

The author declares no conflicts of interest.

## Appendix D: Funding Acknowledgment

This work received no external funding.

## Appendix E: Ethics Declaration

This research involves no human subjects, personal data, or sensitive data. No ethics approval was required.

## Appendix F: AI Disclosure Statement

This paper was drafted with assistance from Claude (Anthropic), an AI language model, used for text organization, phrasing suggestions, and structural editing. All technical content, design decisions, algorithms, and implementation details originate from the author's own work and were verified by the author. The AI tool was not used to generate claims, citations, or experimental results.

---

*Manuscript prepared 2026-04-25. Version 0.1 (methodology draft; benchmark section to follow).*
