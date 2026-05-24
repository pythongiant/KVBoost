#!/usr/bin/env python3
"""
3-way ShareGPT benchmark — vLLM runner.

Stack under test:
  * Qwen2.5-7B-Instruct target  (HF, fp16)
  * Qwen2.5-1.5B-Instruct draft (HF, fp16) — vLLM speculative
  * vLLM automatic prefix caching across conversation turns
  * speculative decoding via vLLM's draft model (gamma drafted per round)

We use AsyncLLMEngine + streaming so TTFT is measured as wall-clock time
from request submission to the first token chunk, matching the existing
vllm_sharegpt_replay benchmark.

vLLM's speculative kwargs changed across versions:
  * v0.5.x / v0.6 (legacy): speculative_model=..., num_speculative_tokens=N,
                            use_v2_block_manager=True
  * v0.7+:                  speculative_config={"model": ..., "num_speculative_tokens": N}

We try the new form first and fall back to the legacy kwargs.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import _common as common
from _common import (
    add_common_args, capture_run_metadata, checkpoint_key, compute_metrics,
    is_run_complete, load_sharegpt, print_summary, replay_conversations,
    setup_logging,
)
from dataclasses import asdict
from datetime import datetime, timezone

RESULTS_DIR    = Path(__file__).resolve().parent / "results"
CHECKPOINT_DIR = Path(__file__).resolve().parent / ".checkpoints"

log = logging.getLogger("sharegpt_3way.vllm")


_CACHED_TOKEN_ATTRS = (
    "num_cached_tokens",
    "num_prefix_cache_tokens",
    "cache_hit_tokens",
    "num_computed_tokens",
)


def build_engine_args(args) -> "AsyncEngineArgs":
    from vllm import AsyncEngineArgs

    base_kwargs = dict(
        model=args.model,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=True,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
    )

    # First try the modern speculative_config dict form (vLLM ≥ 0.7).
    try:
        return AsyncEngineArgs(
            **base_kwargs,
            speculative_config={
                "model": args.draft_model,
                "num_speculative_tokens": args.gamma,
            },
        )
    except TypeError:
        log.info("vLLM: falling back to legacy speculative kwargs.")
    except Exception as e:
        log.warning("vLLM: speculative_config rejected (%s); trying legacy.", e)

    # Legacy kwargs path (vLLM 0.5/0.6).
    return AsyncEngineArgs(
        **base_kwargs,
        speculative_model=args.draft_model,
        num_speculative_tokens=args.gamma,
        use_v2_block_manager=True,
    )


class VLLMRunner:
    def __init__(self, args):
        from vllm import AsyncLLMEngine, SamplingParams
        from transformers import AutoTokenizer

        self.SamplingParams = SamplingParams
        self.max_new_tokens = args.max_new_tokens
        self.engine = AsyncLLMEngine.from_engine_args(build_engine_args(args))

        logging.getLogger("vllm.engine.async_llm_engine").setLevel(logging.WARNING)
        logging.getLogger("vllm.core.scheduler").setLevel(logging.WARNING)

        self._tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._request_counter = 0
        self._cached_attr = self._discover_cached_attr()

    @property
    def tokenizer(self):
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=True))

    def _discover_cached_attr(self) -> Optional[str]:
        try:
            from vllm.engine.metrics_types import RequestMetrics as RM
        except ImportError:
            return None
        fields = set()
        try:
            fields.update(f.name for f in dataclasses.fields(RM))
        except Exception:
            pass
        if hasattr(RM, "__dataclass_fields__"):
            fields.update(RM.__dataclass_fields__.keys())
        for a in _CACHED_TOKEN_ATTRS:
            if a in fields or hasattr(RM, a):
                return a
        return None

    def _get_cached_tokens(self, final_output) -> int:
        if not final_output:
            return 0
        m = getattr(final_output, "metrics", None)
        if m is None:
            return 0
        if self._cached_attr:
            return int(getattr(m, self._cached_attr, 0) or 0)
        for a in _CACHED_TOKEN_ATTRS:
            v = getattr(m, a, None)
            if v is not None:
                self._cached_attr = a
                return int(v)
        return 0

    def run_turn(self, prompt: str) -> dict:
        return self._loop.run_until_complete(self._run_turn_async(prompt))

    async def _run_turn_async(self, prompt: str) -> dict:
        self._request_counter += 1
        request_id = f"req-{self._request_counter}"
        params = self.SamplingParams(max_tokens=self.max_new_tokens, temperature=0.0)

        t0 = time.perf_counter()
        first_chunk_seen = False
        wall_ttft_ms: Optional[float] = None
        final_output = None
        async for output in self.engine.generate(prompt, params, request_id=request_id):
            if not first_chunk_seen:
                wall_ttft_ms = (time.perf_counter() - t0) * 1000.0
                first_chunk_seen = True
            final_output = output
        total_ms = (time.perf_counter() - t0) * 1000.0

        ttft_ms = wall_ttft_ms or total_ms
        if final_output is not None:
            m = getattr(final_output, "metrics", None)
            if m is not None:
                ftl = getattr(m, "first_token_latency", None)
                if ftl is not None and ftl > 0:
                    ttft_ms = ftl * 1000.0

        output_text = ""
        output_tokens = 0
        if final_output and getattr(final_output, "outputs", None):
            output_text = final_output.outputs[0].text or ""
            output_tokens = len(getattr(final_output.outputs[0], "token_ids", []) or [])
        prompt_tokens = (
            len(final_output.prompt_token_ids)
            if final_output and getattr(final_output, "prompt_token_ids", None)
            else self.count_tokens(prompt)
        )
        cached_tokens = self._get_cached_tokens(final_output)

        # Stop reason from vLLM's finish_reason ("stop" → eos, "length" → max_tokens).
        stop_reason = None
        cumulative_logprob = None
        finish_reason = None
        if final_output and getattr(final_output, "outputs", None):
            out0 = final_output.outputs[0]
            finish_reason = getattr(out0, "finish_reason", None)
            cumulative_logprob = getattr(out0, "cumulative_logprob", None)
            if finish_reason == "stop":
                stop_reason = "eos"
            elif finish_reason == "length":
                stop_reason = "max_tokens"
            else:
                stop_reason = finish_reason

        # Surface the full RequestMetrics dict as backend telemetry. Field names
        # vary across vLLM versions, so we dump everything we can introspect.
        backend_telemetry: dict = {}
        if final_output is not None:
            m = getattr(final_output, "metrics", None)
            if m is not None:
                metrics_dict = {}
                try:
                    import dataclasses
                    for f in dataclasses.fields(m):
                        v = getattr(m, f.name, None)
                        if isinstance(v, (int, float, str, bool)) or v is None:
                            metrics_dict[f.name] = v
                except Exception:
                    pass
                # Catch-all for non-dataclass metrics shapes.
                if not metrics_dict:
                    for attr in dir(m):
                        if attr.startswith("_"):
                            continue
                        try:
                            v = getattr(m, attr)
                            if isinstance(v, (int, float, str, bool)):
                                metrics_dict[attr] = v
                        except Exception:
                            pass
                backend_telemetry["request_metrics"] = metrics_dict
            backend_telemetry["finish_reason"] = finish_reason
            backend_telemetry["cumulative_logprob"] = cumulative_logprob
            backend_telemetry["request_id"] = request_id
            backend_telemetry["num_cached_tokens"] = cached_tokens

        # vLLM does not expose per-request speculative acceptance counters in
        # its public API. spec_* stays None — the speedup shows up in ITL/tps.
        return {
            "ttft_ms":       ttft_ms,
            "total_ms":      total_ms,
            "output_text":   output_text,
            "output_tokens": output_tokens,
            "prompt_tokens": prompt_tokens,
            "cached_tokens": cached_tokens,
            "stop_reason":   stop_reason,
            "backend_telemetry": backend_telemetry,
        }

    def close(self):
        if not self._loop.is_closed():
            self._loop.close()


def main():
    parser = argparse.ArgumentParser(description="vLLM 3-way ShareGPT runner")
    add_common_args(parser)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--draft-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()

    setup_logging(args.verbose, args.debug)
    print(f"\n{'=' * 72}\n  vLLM (prefix-cache + spec) — ShareGPT 3-way\n{'=' * 72}")
    print(f"  target={args.model}")
    print(f"  draft ={args.draft_model}  gamma={args.gamma}")
    print(f"  n_samples={args.n_samples}  turns={args.min_turns}-{args.max_turns}")
    print(f"{'=' * 72}\n")

    out_path = Path(args.output) if args.output else RESULTS_DIR / "vllm.json"
    if not args.no_checkpoint and is_run_complete(out_path, args.n_samples):
        print(f"[skip] {out_path} already covers {args.n_samples} conversations; "
              "delete it or pass --no-checkpoint to force re-run.")
        return

    runner = VLLMRunner(args)
    conversations = load_sharegpt(
        n_conversations=args.n_samples,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        max_tokens_per_turn=args.max_tokens_per_turn,
        tokenizer=runner.tokenizer,
        max_context_tokens=args.max_context_tokens,
    )
    if not conversations:
        sys.exit("No conversations after filtering.")

    ck_path = CHECKPOINT_DIR / f"vllm_{checkpoint_key('vllm', args.model, args.n_samples, args.max_turns)}.json"
    meta = {"backend": "vllm", "model": args.model, "draft": args.draft_model, "gamma": args.gamma}

    config = {
        "gamma": args.gamma,
        "prefix_caching": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_new_tokens": args.max_new_tokens,
        "n_samples": args.n_samples,
        "min_turns": args.min_turns,
        "max_turns": args.max_turns,
        "max_context_tokens": args.max_context_tokens,
        "max_tokens_per_turn": args.max_tokens_per_turn,
        "save_output_text": args.save_output_text,
    }
    run_metadata = capture_run_metadata("vllm", config)

    t0 = time.perf_counter()
    # NB: do NOT reset between conversations — vLLM's prefix cache benefits
    # from shared system/preamble tokens across conversations too.
    results = replay_conversations(
        run_turn=runner.run_turn,
        count_tokens=runner.count_tokens,
        reset_between_convs=None,
        conversations=conversations,
        ck_path=ck_path,
        meta=meta,
        run_metadata=run_metadata,
        no_checkpoint=args.no_checkpoint,
        save_output_text=args.save_output_text,
        on_error=args.error_mode,
        progress_every=args.progress_every,
        max_new_tokens=args.max_new_tokens,
    )
    wall_s = time.perf_counter() - t0
    run_metadata.end_iso = datetime.now(timezone.utc).isoformat()

    metrics = compute_metrics(results, total_wall_s=wall_s)
    print_summary("vllm", metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": "vllm",
        "model": args.model,
        "draft_model": args.draft_model,
        "config": config,
        "run_metadata": asdict(run_metadata),
        "wall_s": wall_s,
        "metrics": metrics,
        "results": [asdict(r) for r in results],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Results written: {out_path}")

    runner.close()
    if ck_path.exists() and not args.no_checkpoint:
        ck_path.unlink()
    live = ck_path.with_name(ck_path.stem + ".live.json")
    if live.exists():
        live.unlink()


if __name__ == "__main__":
    main()
