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
    p.add_argument("--recompute-strategy", default="cacheblend",
                   choices=["selective", "cacheblend", "none"])
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

    # OOM recovery — catch CUDA OOM mid-request, lower the right knob, retry.
    # Default on: the server should survive single-request OOMs by trimming
    # the KV-cache budget (if cache is "high") or shrinking AWQ-streaming
    # residency (if cache is "low"). See kvboost.oom_recovery for the policy.
    p.add_argument("--oom-recovery", action="store_true", default=True,
                   help="Catch CUDA OOM mid-request, lower KV cache or streaming "
                        "residency, retry (default: on). Use --no-oom-recovery to disable.")
    p.add_argument("--no-oom-recovery", action="store_false", dest="oom_recovery",
                   help="Disable OOM recovery (originally-emitted CUDA OOM errors "
                        "will propagate to the client unchanged).")
    p.add_argument("--oom-max-retries", type=int, default=None,
                   help="Max OOM recovery attempts per request. Default: unbounded "
                        "— recovery keeps shrinking knobs until the call succeeds "
                        "or every knob hits its floor (capped at SAFETY_CAP=16 "
                        "attempts as an absolute safety limit). Set to an integer "
                        "to bound it tighter. Mid-stream requests never retry; the "
                        "knob still gets adjusted so the NEXT request benefits.")

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
            engine = InferenceEngine.from_pretrained(
                args.model,
                streaming_config=streaming_config,
                max_cache_bytes=int(args.max_cache_bytes),
                chunk_size=args.chunk_size,
                recompute_strategy=args.recompute_strategy,
                kv_cache_bits=args.kv_cache_bits,
                sink_tokens=args.sink_tokens,
                overlap_k=args.overlap_k,
                prefill_chunk_size=args.prefill_chunk_size,
                device=device,
                speculative_config=speculative_cfg,
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
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            **from_pretrained_kwargs,
        )
        engine = InferenceEngine(
            model=model,
            tokenizer=tokenizer,
            max_cache_bytes=int(args.max_cache_bytes),
            chunk_size=args.chunk_size,
            recompute_strategy=args.recompute_strategy,
            kv_cache_bits=args.kv_cache_bits,
            sink_tokens=args.sink_tokens,
            overlap_k=args.overlap_k,
            prefill_chunk_size=args.prefill_chunk_size,
            device=device,
            speculative_config=_build_speculative_config(args),
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

    oom_recovery = None
    if args.oom_recovery:
        from ..oom_recovery import OOMRecovery
        oom_recovery = OOMRecovery(
            engine,
            initial_max_cache_bytes=int(args.max_cache_bytes),
            initial_keep_first_k=args.keep_first_k if args.awq_streaming else None,
            initial_keep_last_k=args.keep_last_k if args.awq_streaming else None,
            streaming_enabled=args.awq_streaming,
            max_retries=args.oom_max_retries,
        )
        log.info(
            "OOM recovery enabled: max_retries=%s, streaming=%s",
            args.oom_max_retries if args.oom_max_retries is not None else "default",
            args.awq_streaming,
        )

    worker = EngineWorker(
        engine=engine,
        max_workers=args.workers,
        batch_window_ms=args.batch_window_ms,
        max_batch_size=args.max_batch_size,
        max_queue_size=args.max_queue_size,
        release_cache_after_request=args.release_cache_after_request,
        rewarm_text=rewarm_text,
        oom_recovery=oom_recovery,
    )

    app = build_app(
        worker,
        model_name=args.model_name or args.model,
        enable_auto_tool_choice=args.enable_auto_tool_choice,
        tool_call_parser=args.tool_call_parser,
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
