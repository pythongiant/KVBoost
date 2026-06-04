"""
KVBoost inference server CLI.

Usage
-----
    # Minimum
    python -m kvboost.server --model Qwen/Qwen2.5-3B

    # Full options
    python -m kvboost.server \
        --model Qwen/Qwen2.5-3B \
        --host 0.0.0.0 \
        --port 8000 \
        --max-cache-bytes 2e9 \
        --chunk-size 128 \
        --recompute-strategy cacheblend \
        --kv-cache-bits 8 \
        --batch-window-ms 20 \
        --max-batch-size 8 \
        --max-queue-size 256 \
        --warm "You are a helpful assistant." \
        --device cuda \
        --dtype float16 \
        --workers 1

    # CPU paged attention backend
    python -m kvboost.server \
        --model Qwen/Qwen2.5-3B \
        --backend cpu-paged \
        --block-size 16 \
        --num-blocks 4096

OpenAI client example
---------------------
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8000/v1", api_key="kvboost")
    resp = client.chat.completions.create(
        model="Qwen/Qwen2.5-3B",
        messages=[{"role": "user", "content": "Hello!"}],
        max_tokens=128,
    )
    print(resp.choices[0].message.content)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger("kvboost.server")


def parse_args():
    p = argparse.ArgumentParser(
        prog="python -m kvboost.server",
        description="KVBoost OpenAI-compatible inference server",
    )

    # Model
    p.add_argument("--model", required=True, help="HuggingFace model name or local path")
    p.add_argument("--gguf-file", default=None,
                   help="GGUF filename inside --model repo (e.g. 'Qwen3-8B-Q4_K_M.gguf'). "
                        "When set, transformers loads weights+tokenizer from the GGUF blob "
                        "(dequantized to --dtype in memory).")
    p.add_argument("--model-name", default=None, help="Override model id shown in /v1/models")
    p.add_argument("--device", default=None, help="Device: cuda | mps | cpu (auto-detected if omitted)")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"],
                   help="Model weight dtype (default: float16)")
    p.add_argument("--attn-impl", default="auto",
                   choices=["auto", "flash_attention_2", "flashinfer",
                            "sage", "triton_flash", "sdpa", "eager"],
                   help="Attention backend. 'auto' (default) uses "
                        "flash_attention_2 if installed (faster/lower-memory "
                        "prefill -> better TTFT; Ampere+ e.g. RTX 3060) and "
                        "falls back to sdpa otherwise. 'flash_attention_2' "
                        "requires it (errors if missing). 'flashinfer' routes "
                        "DECODE attention through FlashInfer's CUDA kernel "
                        "(SDPA prefill + fallback; helps long-context decode). "
                        "'sage' runs INT8 SageAttention on PREFILL via a Triton "
                        "kernel (INT8 tensor-core QK^T on Ampere; no nvcc/build; "
                        "SDPA fallback for decode/unsupported shapes). "
                        "'triton_flash' is the FP16 Triton flash baseline. "
                        "'sdpa'/'eager' force.")
    p.add_argument("--compile", action="store_true", default=False,
                   help="torch.compile(mode='reduce-overhead') on the model: "
                        "CUDA graphs + pointwise fusion to erase per-token "
                        "launch overhead (closes most of the eager-decode gap "
                        "to the bandwidth ceiling). EXPERIMENTAL — compiles "
                        "lazily on first request; drop the flag if a run "
                        "errors. First request pays a one-time compile cost. "
                        "Ignored if --cuda-graph-decode is set.")
    p.add_argument("--cuda-graph-decode", action="store_true", default=False,
                   help="Capture the single-token DECODE step into a CUDA graph "
                        "(over a static KV cache) and replay it — removes the "
                        "eager loop's per-token launch overhead, the dominant "
                        "decode cost on bandwidth-bound GPUs (e.g. RTX 3060). "
                        "Reuse-based prefill is preserved; self-checked vs eager "
                        "with eager fallback. Stacks with Marlin int4 weights.")
    p.add_argument("--backend", default="default", choices=["default", "cpu-paged"],
                   help="Inference backend (default: standard KVBoost)")
    p.add_argument("--quantization", default="none",
                   choices=["none", "bnb-4bit", "bnb-8bit", "hqq-4bit", "hqq-2bit"],
                   help="On-the-fly weight quantization. "
                        "bnb-4bit (NF4) / bnb-8bit: bitsandbytes, ~4x / 2x VRAM reduction. "
                        "hqq-4bit / hqq-2bit: HQQ, no calibration, lower load-time memory than bnb. "
                        "Pre-quantized AWQ/GPTQ checkpoints are detected automatically — "
                        "leave this 'none' and just point --model at e.g. Qwen/Qwen3-8B-AWQ.")
    p.add_argument("--use-slow-tokenizer", action="store_true",
                   help="Force the SentencePiece-based slow tokenizer. "
                        "Workaround for fast-tokenizer builds whose byte-level "
                        "decoder is missing/broken (symptom: decoded text drops "
                        "spaces/newlines or shows literal 'Ġ' and 'Ċ'). "
                        "Seen on some Llama-3 / DeepSeek-R1-Distill checkpoints.")
    p.add_argument("--max-memory", default=None,
                   help="Per-device memory cap for CPU offload, JSON dict. "
                        'Example: \'{"0": "7GiB", "cpu": "32GiB"}\'. When set, uses '
                        "device_map='auto' so transformers spills overflow layers to CPU RAM. "
                        "Slower but lets bigger models run on small GPUs.")

    # KVBoost cache
    p.add_argument("--max-cache-bytes", type=float, default=2e9,
                   help="KV cache memory budget in bytes (default: 2 GB)")
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--chunk-boundary-window", type=int, default=0,
                   help="Content-aligned chunking: split chunks at content "
                        "boundaries (newlines etc.) within +/- this many tokens "
                        "of the fixed size, instead of pure fixed-size cuts. "
                        "0=off. Set >0 so a block of content (e.g. a file) that "
                        "RECURS AT A DIFFERENT POSITION still chunks identically "
                        "→ content-hash matches → CacheBlend reuse. Needed for "
                        "the reshuffled multi-turn / RAG reuse workloads.")
    p.add_argument("--recompute-strategy", default="cacheblend",
                   choices=["selective", "cacheblend", "cacheblend_sparse", "none"],
                   help="KV seam-repair strategy on chunk reuse. "
                        "'cacheblend_sparse' is the faithful CacheBlend — "
                        "recomputes only high-deviation tokens layer-by-layer "
                        "(paper's 2.2-3.3× TTFT vs full recompute), falls back "
                        "to 'cacheblend' on unsupported architectures.")
    p.add_argument("--kv-cache-bits", type=int, default=16, choices=[4, 8, 16],
                   help="KV quantization bits (16=off, 8=int8, 4=int4)")
    p.add_argument("--sink-tokens", type=int, default=0)
    p.add_argument("--overlap-k", type=int, default=0)
    p.add_argument("--prefill-chunk-size", type=int, default=0,
                   help="Process the prompt in slices of N tokens during prefill, "
                        "growing past_key_values between iterations. 0 = single-shot "
                        "(legacy). Set to e.g. 512 or 1024 to fit long prompts on "
                        "small GPUs by capping peak FFN/attention activation memory.")

    # CPU paged backend
    p.add_argument("--block-size", type=int, default=16, help="Tokens per paged block")
    p.add_argument("--num-blocks", type=int, default=4096, help="Number of paged blocks")

    # AWQ Layer Streaming — load AWQ weights to host RAM and DMA per layer
    # so models larger than GPU VRAM still run. Layers in keep_first_k +
    # keep_last_k stay resident; the rest stream from pinned host RAM.
    # Composes with chunk reuse: KV-cache hits work the same on streamed
    # weights, so a warm prefix still skips prefill.
    p.add_argument("--awq-streaming", action="store_true",
                   help="Enable AWQ layer streaming. Use for AWQ-quantized "
                        "models that don't fit fully in GPU VRAM.")
    p.add_argument("--streaming-mode", default="partial_resident",
                   choices=["full_resident", "partial_resident",
                            "ffn_only_stream", "full_stream"],
                   help="Residency policy when --awq-streaming is set "
                        "(default: partial_resident)")
    p.add_argument("--keep-first-k", type=int, default=4,
                   help="Decoder layers at the head of the network that stay "
                        "resident in VRAM (default: 4)")
    p.add_argument("--keep-last-k", type=int, default=4,
                   help="Decoder layers at the tail of the network that stay "
                        "resident in VRAM (default: 4)")
    p.add_argument("--streaming-quant-kernel", default="auto",
                   choices=["auto", "marlin", "exllama_v2", "torch"],
                   help="AWQ GEMM kernel preference (default: auto — "
                        "probes Marlin → ExLlamaV2 → pure-torch fallback)")

    # Speculative decoding — small draft model proposes K tokens per
    # round, target verifies in one batched forward. Disabled by default.
    p.add_argument("--speculative-draft-model", default=None,
                   help="HuggingFace model id of the draft model. When set, "
                        "speculative decoding is enabled. The draft must share "
                        "a tokenizer family with the target (e.g. Qwen2.5-1.5B "
                        "draft against Qwen2.5-32B target).")
    p.add_argument("--speculative-gamma", type=int, default=5,
                   help="Number of tokens the draft proposes per verification "
                        "round (default: 5). Higher K = larger speedup when "
                        "acceptance is high, but more wasted work when low.")
    p.add_argument("--speculative-mode", default="greedy",
                   choices=["greedy", "sampling"],
                   help="Verification strategy. 'greedy' matches non-speculative "
                        "greedy decode bit-for-bit. 'sampling' uses rejection "
                        "sampling (Leviathan et al. 2023) for temperature > 0.")
    p.add_argument("--speculative-temperature", type=float, default=1.0,
                   help="Temperature applied to target logits in sampling mode "
                        "(default: 1.0). Ignored in greedy mode.")

    # SpecBlock-inspired tree speculative decoding. Requires the flat
    # speculative drafter to be set (uses the same draft model with a
    # tree-drafting wrapper). The ``ModeSelector`` then picks per-request
    # between flat-K and tree-(B,D) by expected wall-time tokens/s.
    p.add_argument("--speculative-tree", action="store_true", default=False,
                   help="Enable SpecBlock-inspired tree speculative "
                        "decoding alongside flat. Requires --speculative-"
                        "draft-model. Per-request mode is auto-selected "
                        "by the cost model unless --speculative-mode-policy "
                        "overrides.")
    p.add_argument("--speculative-mode-policy", default=None,
                   choices=["auto", "flat", "tree", "none"],
                   help="Force one speculative mode per request. Default "
                        "is 'auto' when --speculative-tree is set, else "
                        "'flat'. 'none' disables speculation entirely.")
    p.add_argument("--speculative-tree-max-branching", type=int, default=4,
                   help="Cap on per-node children in the draft tree "
                        "(default: 4). Higher = wider tree.")
    p.add_argument("--speculative-tree-max-depth", type=int, default=6,
                   help="Cap on tree depth (default: 6). Deeper trees "
                        "win more when acceptance is high.")
    p.add_argument("--speculative-tree-node-budget", type=int, default=32,
                   help="Total node-count cap for the tree (default: 32). "
                        "Hard-bounds the target verifier's cost.")
    p.add_argument("--speculative-tree-cold-accept", type=float, default=0.5,
                   help="Seed acceptance prior for the tree EWMA (default: "
                        "0.5). Used until 16+ samples per (B,D) cohort.")

    # Server
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--workers", type=int, default=1,
                   help="Engine thread-pool size (keep 1 for GPU)")

    # Batching
    p.add_argument("--batch-window-ms", type=float, default=20.0,
                   help="Request collection window before dispatch (ms)")
    p.add_argument("--max-batch-size", type=int, default=8)
    p.add_argument("--max-queue-size", type=int, default=256)
    p.add_argument("--release-cache-after-request", action="store_true",
                   help="Run torch.cuda.empty_cache() between requests. "
                        "Recommended on 8 GB-class GPUs where PyTorch's "
                        "allocator cache from request N can prevent prefill "
                        "from fitting in request N+1. Costs ~5-20 ms per request.")

    # OOM planning — proactively pick a (prefill_chunk_size, kv_cache_bits)
    # combo per request so that peak memory is predicted to fit. Replaces
    # the older reactive OOMRecovery loop, which caught OOMs mid-request
    # and cascaded through knob shrinks (slow failure, mutated global state).
    p.add_argument("--oom-planning", action="store_true", default=True,
                   help="Pre-flight every request: estimate peak VRAM and "
                        "pick chunk_size/kv_bits that fit (default: on). "
                        "Use --no-oom-planning to disable.")
    p.add_argument("--no-oom-planning", action="store_false", dest="oom_planning",
                   help="Disable proactive planning. CUDA OOMs propagate to "
                        "the client unchanged. Use this only for debugging.")
    p.add_argument("--auto-truncate", action="store_true", default=False,
                   help="If a request's predicted peak exceeds available "
                        "VRAM even at the most aggressive plan (smallest "
                        "chunk + int4 KV), silently truncate the prompt "
                        "to the largest prefix that fits. Default: off "
                        "(server returns HTTP 413 with an explanation).")
    p.add_argument("--planner-safety-margin", type=float, default=0.15,
                   help="Fraction of free VRAM reserved as headroom above "
                        "the planner's prediction. Covers allocator "
                        "fragmentation + model approximation error. "
                        "Default: 0.15 (15%%). Lower for tighter packing, "
                        "higher if you see surprise OOMs slip through.")

    # Server-side max_tokens cap. Schema-level cap is 131072 (Qwen3-YaRN
    # ceiling) but real-world VRAM rarely supports that. This flag is the
    # operator's say-so: any incoming request's max_tokens is clamped down
    # to this value before reaching the engine. Set to a value that fits
    # comfortably in your KV-cache budget at int8.
    p.add_argument("--max-tokens", type=int, default=None,
                   help="Server-side ceiling on request max_tokens (1..131072). "
                        "Incoming requests with higher values are clamped down. "
                        "Default: no clamp (schema default of 4096 still applies "
                        "when the client doesn't send max_tokens).")

    # Tool / function calling
    p.add_argument("--enable-auto-tool-choice", action="store_true",
                   help="Enable OpenAI-compatible tool/function calling. When set, "
                        "the server forwards 'tools' to the chat template and parses "
                        "tool calls out of the model output using --tool-call-parser.")
    p.add_argument("--tool-call-parser", default="hermes",
                   choices=["hermes", "json_codeblock", "qwen3_coder",
                            "llama", "mistral", "auto"],
                   help="Format used by the model to emit tool calls.\n"
                        "  hermes         = <tool_call>{json}</tool_call> (Qwen2.5/3, Hermes 2/3)\n"
                        "  json_codeblock = ```json\\n{...}\\n``` (Qwen2.5-Coder agent, mixed-format models)\n"
                        "  qwen3_coder    = <function=X><parameter=K>V</parameter> XML attr style\n"
                        "  llama          = <|python_tag|>{json}<|eom_id|> (Llama 3.1/3.2/3.3)\n"
                        "  mistral        = [TOOL_CALLS][...] prefix + JSON array\n"
                        "  auto           = try each parser in order; first match wins.\n"
                        "Recommended: 'auto' when serving mixed/unknown formats; pin to a specific one\n"
                        "for the model you're serving.")

    # Pre-warm
    p.add_argument("--warm", default=None,
                   help="Text to pre-warm the KV cache before accepting requests")
    p.add_argument("--always-warm", default=None,
                   help="Like --warm, but also re-runs the warm after every cache "
                        "release (i.e. with --release-cache-after-request). Use to "
                        "keep a system prompt's KV resident across requests on tight "
                        "GPUs. Adds the warm latency to each request boundary. "
                        "If both --warm and --always-warm are set, --always-warm wins.")

    # Logging
    p.add_argument("--log-level", default="info",
                   choices=["debug", "info", "warning", "error"])

    return p.parse_args()


def load_engine(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    log.info("Loading model %s ...", args.model)
    if args.gguf_file:
        log.info("Using GGUF file: %s", args.gguf_file)
    gguf_kwargs = {"gguf_file": args.gguf_file} if args.gguf_file else {}

    quant_config = None
    if args.quantization in ("bnb-4bit", "bnb-8bit"):
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401  — fail-fast if not installed
        except ImportError:
            raise SystemExit(
                "ERROR: --quantization bnb-* requires bitsandbytes.\n"
                "Run: pip install bitsandbytes"
            )
        if args.quantization == "bnb-4bit":
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        else:
            quant_config = BitsAndBytesConfig(load_in_8bit=True)
        log.info("Quantization: %s", args.quantization)
    elif args.quantization in ("hqq-4bit", "hqq-2bit"):
        try:
            from transformers import HqqConfig
        except ImportError:
            raise SystemExit(
                "ERROR: --quantization hqq-* requires a recent transformers + hqq.\n"
                "Run: pip install -U transformers hqq"
            )
        nbits = 4 if args.quantization == "hqq-4bit" else 2
        quant_config = HqqConfig(nbits=nbits, group_size=64)
        log.info("Quantization: %s (HQQ %d-bit)", args.quantization, nbits)

    max_memory = None
    if args.max_memory:
        try:
            raw = json.loads(args.max_memory)
        except json.JSONDecodeError as e:
            raise SystemExit(f"ERROR: --max-memory must be valid JSON: {e}")
        # Keys may be int (GPU index) or "cpu" / "disk"
        max_memory = {(int(k) if k.isdigit() else k): v for k, v in raw.items()}
        log.info("CPU/GPU offload max_memory=%s", max_memory)

    tokenizer_kwargs = dict(**gguf_kwargs)
    if args.use_slow_tokenizer:
        tokenizer_kwargs["use_fast"] = False
        log.info("Loading slow (SentencePiece) tokenizer.")
    tokenizer = AutoTokenizer.from_pretrained(args.model, **tokenizer_kwargs)

    if args.backend == "cpu-paged":
        if args.gguf_file:
            raise SystemExit("--gguf-file is not supported with --backend cpu-paged.")
        if quant_config is not None:
            raise SystemExit("--quantization is not supported with --backend cpu-paged.")
        from ..cpu_paged import CPUPagedEngine
        engine = CPUPagedEngine.from_pretrained(
            args.model,
            max_cache_bytes=int(args.max_cache_bytes),
            chunk_size=args.chunk_size,
            recompute_strategy=args.recompute_strategy,
            kv_cache_bits=args.kv_cache_bits,
            sink_tokens=args.sink_tokens,
            overlap_k=args.overlap_k,
            block_size=args.block_size,
            num_blocks=args.num_blocks,
        )
    else:
        from ..engine import InferenceEngine
        from ..compat import default_device
        device = args.device or default_device()

        # ── AWQ Layer Streaming path ─────────────────────────────────────
        # When --awq-streaming is set, the engine's factory constructs the
        # model via StreamingCausalLM (pinned-host AWQ + per-forward DMA).
        # Chunk reuse, KV-cache management, and SSE token streaming work
        # exactly the same on top of it — InferenceEngine doesn't care that
        # the model's weights are being shuffled in and out.
        if args.awq_streaming:
            if args.gguf_file:
                raise SystemExit(
                    "--gguf-file is incompatible with --awq-streaming "
                    "(GGUF weights don't have AWQ projection layout)."
                )
            if quant_config is not None:
                raise SystemExit(
                    "--quantization is incompatible with --awq-streaming. "
                    "Streaming reads AWQ tensors directly from safetensors; "
                    "drop --quantization (the model already has its own "
                    "AWQ quantization_config in config.json)."
                )
            from ..streaming import StreamingConfig
            streaming_config = StreamingConfig(
                residency_mode=args.streaming_mode,
                keep_first_k=args.keep_first_k,
                keep_last_k=args.keep_last_k,
                quant_kernel=args.streaming_quant_kernel,
            )
            log.info(
                "AWQ streaming enabled: %s, keep_first_k=%d, keep_last_k=%d, "
                "kernel=%s",
                args.streaming_mode, args.keep_first_k, args.keep_last_k,
                args.streaming_quant_kernel,
            )
            speculative_cfg = _build_speculative_config(args)
            tree_speculative_cfg = _build_tree_speculative_config(args)
            engine = InferenceEngine.from_pretrained(
                args.model,
                streaming_config=streaming_config,
                max_cache_bytes=int(args.max_cache_bytes),
                chunk_size=args.chunk_size,
                chunk_boundary_window=args.chunk_boundary_window,
                recompute_strategy=args.recompute_strategy,
                kv_cache_bits=args.kv_cache_bits,
                sink_tokens=args.sink_tokens,
                overlap_k=args.overlap_k,
                prefill_chunk_size=args.prefill_chunk_size,
                device=device,
                speculative_config=speculative_cfg,
                tree_speculative_config=tree_speculative_cfg,
                # attn_impl is ignored on the streaming path (StreamingCausalLM
                # owns attention); compile flows through to __init__.
                attn_implementation=args.attn_impl,
                compile_model=args.compile,
                cuda_graph_decode=args.cuda_graph_decode,
            )
            log.info("Model loaded.")
            return engine

        # ── Standard load path ───────────────────────────────────────────
        # By default load directly onto the target device. With --max-memory we
        # opt into device_map="auto" + accelerate offload; InferenceEngine
        # detects this (via hf_device_map / quantization) and skips its .to().
        from_pretrained_kwargs = dict(**gguf_kwargs)
        # Bypass accelerate's meta-init path. With low_cpu_mem_usage=True,
        # any submodule that dispatch can't place stays on `meta`, which
        # crashes gptqmodel's Marlin post_init. False forces real tensors
        # so we either succeed or get a real CUDA-OOM error.
        from_pretrained_kwargs["low_cpu_mem_usage"] = False
        if max_memory is not None:
            from_pretrained_kwargs["device_map"] = "auto"
            from_pretrained_kwargs["max_memory"] = max_memory
        else:
            # Use the explicit-dict form — passing a bare string ("cuda") can
            # leave some buffers on `meta`, which breaks AWQ-Marlin's post_init
            # ("Expected a cuda device, but got: meta"). The {"": dev} form
            # forces every leaf onto the target device.
            target = device if ":" in device or device in ("cpu", "mps") else f"{device}:0"
            from_pretrained_kwargs["device_map"] = {"": target}
        if quant_config is not None:
            # bnb/HQQ set compute dtype themselves; passing dtype here
            # is ignored (and would warn), so omit it.
            from_pretrained_kwargs["quantization_config"] = quant_config
        else:
            from_pretrained_kwargs["dtype"] = dtype
        # Attention backend. 'auto' tries FA2 (better TTFT on Ampere+, e.g.
        # RTX 3060) then falls back to sdpa; an explicit choice is honored.
        # 'flashinfer' is registered with HF here (falls back to sdpa if the
        # package is absent) before it can be used as an impl key.
        from ..kernels import resolve_attn_impl
        _impl = resolve_attn_impl(args.attn_impl)
        _want_fa2 = _impl in ("auto", "flash_attention_2")
        from_pretrained_kwargs["attn_implementation"] = (
            "flash_attention_2" if _want_fa2 else _impl
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, **from_pretrained_kwargs,
            )
            log.info("Attention backend: %s",
                     from_pretrained_kwargs["attn_implementation"])
        except Exception as e:
            if args.attn_impl != "auto":
                raise  # explicit backend requested — don't mask the failure
            log.info("flash_attention_2 unavailable (%s); using sdpa", e)
            from_pretrained_kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(
                args.model, **from_pretrained_kwargs,
            )
            log.info("Attention backend: sdpa")
        engine = InferenceEngine(
            model=model,
            tokenizer=tokenizer,
            max_cache_bytes=int(args.max_cache_bytes),
            chunk_size=args.chunk_size,
            chunk_boundary_window=args.chunk_boundary_window,
            recompute_strategy=args.recompute_strategy,
            kv_cache_bits=args.kv_cache_bits,
            sink_tokens=args.sink_tokens,
            overlap_k=args.overlap_k,
            prefill_chunk_size=args.prefill_chunk_size,
            device=device,
            speculative_config=_build_speculative_config(args),
            tree_speculative_config=_build_tree_speculative_config(args),
            compile_model=args.compile,
            cuda_graph_decode=args.cuda_graph_decode,
        )

    log.info("Model loaded.")
    return engine


def _build_speculative_config(args):
    """Build a SpeculativeConfig from parsed CLI args, or return None when
    speculative decoding is disabled (no --speculative-draft-model)."""
    if not getattr(args, "speculative_draft_model", None):
        return None
    from ..speculative import SpeculativeConfig
    return SpeculativeConfig(
        draft_model_id=args.speculative_draft_model,
        draft_k=args.speculative_gamma,
        mode=args.speculative_mode,
        temperature=args.speculative_temperature,
    )


def _build_tree_speculative_config(args):
    """Build a TreeSpeculativeConfig from parsed CLI args, or return None
    when tree mode is disabled.

    Requires the flat drafter (we reuse the same draft model wrapped
    by ``TreeDraftModel``). When ``--speculative-tree`` is set but no
    drafter is configured, raise a SystemExit with a clear message —
    silently disabling tree mode would mask a misconfiguration.
    """
    if not getattr(args, "speculative_tree", False):
        return None
    if not getattr(args, "speculative_draft_model", None):
        raise SystemExit(
            "ERROR: --speculative-tree requires --speculative-draft-model "
            "(the tree drafter wraps the same small model). Pass both, "
            "or drop --speculative-tree."
        )
    from ..speculative import TreeSpeculativeConfig
    policy = getattr(args, "speculative_mode_policy", None) or "auto"
    return TreeSpeculativeConfig(
        max_branching=args.speculative_tree_max_branching,
        max_depth=args.speculative_tree_max_depth,
        node_budget=args.speculative_tree_node_budget,
        cold_accept=args.speculative_tree_cold_accept,
        policy=policy,
    )


def main():
    args = parse_args()

    logging.getLogger().setLevel(args.log_level.upper())

    try:
        import uvicorn
        import fastapi  # noqa
    except ImportError:
        print(
            "ERROR: server dependencies not installed.\n"
            "Run: pip install 'kvboost[server]'",
            file=sys.stderr,
        )
        sys.exit(1)

    engine = load_engine(args)

    from .engine_worker import EngineWorker
    from .app import build_app

    # Don't pre-create a loop here — uvicorn will create its own. EngineWorker
    # captures the running loop in start() (the FastAPI startup hook).
    # --always-warm wins if both are set; it's a superset of --warm.
    warm_text = args.always_warm or args.warm
    rewarm_text = args.always_warm

    planner = None
    if args.oom_planning:
        from ..oom_planner import OOMPlanner
        from ..cost_model import probe_cost_coefficients

        # Probe once at startup: VRAM, per-layer bytes, PCIe & HBM bandwidth,
        # model shape. The planner uses these to predict peak transient
        # memory for each request and pick (chunk_size, kv_bits) accordingly.
        # Bounded probe time (~2 s); falls back to defaults on non-CUDA
        # devices or if any individual probe fails.
        try:
            cost_coefficients = probe_cost_coefficients(engine)
        except Exception as e:
            log.warning(
                "Cost-coefficient probe failed (%s); planning will use "
                "conservative defaults.", e,
            )
            from ..cost_model import CostCoefficients
            cost_coefficients = CostCoefficients(
                total_vram_mb=0.0, per_layer_mb=150.0, num_layers=32,
                pcie_h2d_gibps=4.0, hbm_bandwidth_gibps=200.0,
                step_latency_ms=50.0,
            )

        planner = OOMPlanner(
            engine,
            cost_coefficients,
            auto_truncate=args.auto_truncate,
            safety_margin_frac=args.planner_safety_margin,
        )
        log.info(
            "OOM planning enabled: auto_truncate=%s, safety_margin=%.0f%%",
            args.auto_truncate, args.planner_safety_margin * 100,
        )
        # Same coefficients drive tree-shape selection. The engine
        # already constructed its tree engine with defaults; this
        # writes the calibrated values in.
        engine.set_cost_coefficients(cost_coefficients)

    worker = EngineWorker(
        engine=engine,
        max_workers=args.workers,
        batch_window_ms=args.batch_window_ms,
        max_batch_size=args.max_batch_size,
        max_queue_size=args.max_queue_size,
        release_cache_after_request=args.release_cache_after_request,
        rewarm_text=rewarm_text,
        planner=planner,
    )

    if args.max_tokens is not None:
        if not (1 <= args.max_tokens <= 131072):
            raise SystemExit(
                f"ERROR: --max-tokens must be in [1, 131072], got {args.max_tokens}"
            )
        log.info("Server-side max_tokens cap: %d", args.max_tokens)

    app = build_app(
        worker,
        model_name=args.model_name or args.model,
        enable_auto_tool_choice=args.enable_auto_tool_choice,
        tool_call_parser=args.tool_call_parser,
        max_tokens_cap=args.max_tokens,
    )

    # Pre-warm synchronously before accepting requests
    if warm_text:
        mode = "always-warm" if args.always_warm else "warm"
        log.info("Pre-warming KV cache (%s) ...", mode)
        engine.warm(warm_text)
        log.info("Pre-warm complete.")

    log.info("Starting server on %s:%d", args.host, args.port)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
