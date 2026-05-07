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

    # CPU paged backend
    p.add_argument("--block-size", type=int, default=16, help="Tokens per paged block")
    p.add_argument("--num-blocks", type=int, default=4096, help="Number of paged blocks")

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

    # Pre-warm
    p.add_argument("--warm", default=None,
                   help="Text to pre-warm the KV cache before accepting requests")

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
    torch_dtype = dtype_map[args.dtype]

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
                bnb_4bit_compute_dtype=torch_dtype,
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

    tokenizer = AutoTokenizer.from_pretrained(args.model, **gguf_kwargs)

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
            # bnb/HQQ set compute dtype themselves; passing torch_dtype here
            # is ignored (and would warn), so omit it.
            from_pretrained_kwargs["quantization_config"] = quant_config
        else:
            from_pretrained_kwargs["torch_dtype"] = torch_dtype
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
            device=device,
        )

    log.info("Model loaded.")
    return engine


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
    worker = EngineWorker(
        engine=engine,
        max_workers=args.workers,
        batch_window_ms=args.batch_window_ms,
        max_batch_size=args.max_batch_size,
        max_queue_size=args.max_queue_size,
    )

    app = build_app(worker, model_name=args.model_name or args.model)

    # Pre-warm synchronously before accepting requests
    if args.warm:
        log.info("Pre-warming KV cache ...")
        engine.warm(args.warm)
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
