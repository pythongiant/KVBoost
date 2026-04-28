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
import asyncio
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
    p.add_argument("--model-name", default=None, help="Override model id shown in /v1/models")
    p.add_argument("--device", default=None, help="Device: cuda | mps | cpu (auto-detected if omitted)")
    p.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32"],
                   help="Model weight dtype (default: float16)")
    p.add_argument("--backend", default="default", choices=["default", "cpu-paged"],
                   help="Inference backend (default: standard KVBoost)")

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
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.backend == "cpu-paged":
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
        device = args.device
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch_dtype,
            device_map=device or "auto",
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

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    from .engine_worker import EngineWorker
    from .app import build_app

    worker = EngineWorker(
        engine=engine,
        loop=loop,
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
        loop="none",  # use our own event loop
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
