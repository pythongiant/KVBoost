#!/usr/bin/env python3
"""
3-way ShareGPT benchmark — llama.cpp runner.

Stack under test:
  * Qwen2.5-7B-Instruct target  (GGUF, Q4_K_M)
  * Qwen2.5-1.5B-Instruct draft (GGUF, Q4_K_M) — llama.cpp speculative
  * llama.cpp implicit KV-prefix retention across calls (n_past matching)
  * speculative decoding via Llama(..., draft_model=Llama(draft_gguf))

Implementation notes
--------------------
* llama-cpp-python keeps the KV cache around between calls on the same
  Llama instance. When a new prompt shares a prefix with the previous
  prompt, `eval_tokens` re-uses the matching prefix and only evaluates
  the suffix. We surface this as `cached_tokens` by reading
  `llm.n_tokens` *before* the call (= length of prefix retained from the
  previous call, conditional on prefix match).
* Speculative is enabled via the `draft_model` kwarg on `Llama`. The
  installed version of llama-cpp-python must support model-based
  speculative (≥ 0.2.50 roughly). On older builds this benchmark falls
  back to non-speculative llama.cpp and logs a warning.
* TTFT is measured against the first streamed token from
  `create_completion(stream=True)`.
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import _common as common
from _common import (
    add_common_args, capture_run_metadata, checkpoint_key, compute_metrics,
    load_sharegpt, print_summary, replay_conversations, setup_logging,
)
from dataclasses import asdict
from datetime import datetime, timezone

RESULTS_DIR    = Path(__file__).resolve().parent / "results"
CHECKPOINT_DIR = Path(__file__).resolve().parent / ".checkpoints"

log = logging.getLogger("sharegpt_3way.llamacpp")


def _llama_supports_draft() -> bool:
    try:
        from llama_cpp import Llama
    except ImportError:
        return False
    return "draft_model" in inspect.signature(Llama.__init__).parameters


class LlamaCppRunner:
    def __init__(self, args):
        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise SystemExit(
                "llama-cpp-python is required. Install with:\n"
                "  CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python --no-binary llama-cpp-python"
            ) from e

        from transformers import AutoTokenizer

        # HF tokenizer is used ONLY for ShareGPT filtering / count_tokens.
        # llama.cpp internally tokenizes with its own GGUF tokenizer.
        self._tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id, trust_remote_code=True)
        self.max_new_tokens = args.max_new_tokens

        common_kwargs = dict(
            n_ctx=args.n_ctx,
            n_gpu_layers=args.n_gpu_layers,
            verbose=False,
            logits_all=False,
        )

        log.info("llama.cpp: loading draft model %s ...", args.draft_model_path)
        draft = Llama(model_path=args.draft_model_path, **common_kwargs)

        log.info("llama.cpp: loading target model %s ...", args.model_path)
        if _llama_supports_draft():
            try:
                self.llm = Llama(
                    model_path=args.model_path,
                    draft_model=draft,
                    **common_kwargs,
                )
                self._spec_enabled = True
                log.info("Speculative enabled via draft_model kwarg.")
            except TypeError as e:
                log.warning("draft_model kwarg rejected (%s); using non-spec llama.cpp.", e)
                self._spec_enabled = False
                self.llm = Llama(model_path=args.model_path, **common_kwargs)
        else:
            log.warning(
                "Installed llama-cpp-python lacks `draft_model`. "
                "Running non-speculative llama.cpp (this is still a fair "
                "prefix-cache baseline, just without spec-decoding speedup)."
            )
            self._spec_enabled = False
            self.llm = Llama(model_path=args.model_path, **common_kwargs)

        self._last_prompt_tokens: list[int] = []

    @property
    def tokenizer(self):
        return self._tokenizer

    def count_tokens(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=True))

    def _gguf_tokenize(self, text: str) -> list[int]:
        # llama-cpp-python's tokenize wants bytes and returns ids.
        return self.llm.tokenize(text.encode("utf-8"), add_bos=True, special=True)

    def _prefix_match_len(self, new_ids: list[int]) -> int:
        prev = self._last_prompt_tokens
        n = min(len(prev), len(new_ids))
        i = 0
        while i < n and prev[i] == new_ids[i]:
            i += 1
        return i

    def _capture_perf(self) -> dict:
        """Pull llama.cpp's internal perf counters when available. Returns
        a plain dict so it's JSON-serializable regardless of API shape."""
        snap: dict = {}
        # llama-cpp-python ≥ 0.2.50: llm.context_perf() / llm._ctx fields
        try:
            ctx = getattr(self.llm, "_ctx", None)
            if ctx is not None:
                # Many builds expose llama_perf_context_data via a helper.
                perf_fn = getattr(ctx, "get_perf", None) or getattr(ctx, "perf", None)
                if callable(perf_fn):
                    perf = perf_fn()
                    for attr in (
                        "t_start_ms", "t_load_ms", "t_p_eval_ms",
                        "t_eval_ms", "n_p_eval", "n_eval", "n_sample_ms",
                    ):
                        v = getattr(perf, attr, None)
                        if v is not None:
                            snap[attr] = v
        except Exception:
            pass
        # llm.n_tokens — current KV occupancy (= retained prefix tokens)
        try:
            snap["n_tokens"] = int(getattr(self.llm, "n_tokens", 0))
        except Exception:
            pass
        return snap

    def run_turn(self, prompt: str) -> dict:
        new_ids = self._gguf_tokenize(prompt)
        cached_before = self._prefix_match_len(new_ids)
        perf_before = self._capture_perf()

        t0 = time.perf_counter()
        first_token_seen = False
        ttft_ms: Optional[float] = None
        output_text_chunks: list[str] = []
        output_token_count = 0
        finish_reason: Optional[str] = None

        # stream=True yields one delta per token, so we can stamp TTFT cleanly.
        stream = self.llm.create_completion(
            prompt=prompt,
            max_tokens=self.max_new_tokens,
            temperature=0.0,
            stream=True,
        )
        for chunk in stream:
            choice = chunk["choices"][0]
            piece = choice.get("text", "")
            if piece:
                if not first_token_seen:
                    ttft_ms = (time.perf_counter() - t0) * 1000.0
                    first_token_seen = True
                output_text_chunks.append(piece)
                output_token_count += 1
            fr = choice.get("finish_reason")
            if fr is not None:
                finish_reason = fr
                break

        total_ms = (time.perf_counter() - t0) * 1000.0
        if ttft_ms is None:
            ttft_ms = total_ms

        perf_after = self._capture_perf()
        # Remember the served prompt for next-turn prefix accounting.
        self._last_prompt_tokens = new_ids

        stop_reason = None
        if finish_reason == "stop":
            stop_reason = "eos"
        elif finish_reason == "length":
            stop_reason = "max_tokens"
        else:
            stop_reason = finish_reason

        backend_telemetry = {
            "finish_reason": finish_reason,
            "perf_before": perf_before,
            "perf_after": perf_after,
            "spec_enabled": self._spec_enabled,
            "prefix_match_len_tokens": cached_before,
        }

        return {
            "ttft_ms":       ttft_ms,
            "total_ms":      total_ms,
            "output_text":   "".join(output_text_chunks),
            "output_tokens": output_token_count,
            "prompt_tokens": len(new_ids),
            "cached_tokens": cached_before,
            "stop_reason":   stop_reason,
            "backend_telemetry": backend_telemetry,
        }

    def reset_between_convs(self):
        # Start each conversation with a fresh prefix-match baseline.
        self._last_prompt_tokens = []


def main():
    parser = argparse.ArgumentParser(description="llama.cpp 3-way ShareGPT runner")
    add_common_args(parser)
    parser.add_argument(
        "--model-path", required=True,
        help="Path to target GGUF (e.g. Qwen2.5-7B-Instruct-Q4_K_M.gguf)",
    )
    parser.add_argument(
        "--draft-model-path", required=True,
        help="Path to draft GGUF (e.g. Qwen2.5-1.5B-Instruct-Q4_K_M.gguf)",
    )
    parser.add_argument(
        "--tokenizer-id", default="Qwen/Qwen2.5-7B-Instruct",
        help="HF tokenizer ID used for ShareGPT filtering only.",
    )
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--n-gpu-layers", type=int, default=-1,
                        help="-1 = offload all layers to GPU (default).")
    args = parser.parse_args()

    setup_logging(args.verbose, args.debug)
    print(f"\n{'=' * 72}\n  llama.cpp (prefix-cache + spec) — ShareGPT 3-way\n{'=' * 72}")
    print(f"  target={args.model_path}")
    print(f"  draft ={args.draft_model_path}  gamma={args.gamma}")
    print(f"  n_samples={args.n_samples}  turns={args.min_turns}-{args.max_turns}")
    print(f"{'=' * 72}\n")

    runner = LlamaCppRunner(args)
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

    ck_path = CHECKPOINT_DIR / f"llamacpp_{checkpoint_key('llamacpp', args.model_path, args.n_samples, args.max_turns)}.json"
    meta = {"backend": "llamacpp", "model": args.model_path, "draft": args.draft_model_path, "gamma": args.gamma}

    config = {
        "gamma": args.gamma,
        "n_ctx": args.n_ctx,
        "n_gpu_layers": args.n_gpu_layers,
        "max_new_tokens": args.max_new_tokens,
        "n_samples": args.n_samples,
        "min_turns": args.min_turns,
        "max_turns": args.max_turns,
        "max_context_tokens": args.max_context_tokens,
        "max_tokens_per_turn": args.max_tokens_per_turn,
        "speculative_enabled": runner._spec_enabled,
        "save_output_text": args.save_output_text,
    }
    run_metadata = capture_run_metadata("llamacpp", config)

    t0 = time.perf_counter()
    results = replay_conversations(
        run_turn=runner.run_turn,
        count_tokens=runner.count_tokens,
        reset_between_convs=runner.reset_between_convs,
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
    print_summary("llamacpp", metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else RESULTS_DIR / "llamacpp.json"
    payload = {
        "backend": "llamacpp",
        "model": args.model_path,
        "draft_model": args.draft_model_path,
        "config": config,
        "run_metadata": asdict(run_metadata),
        "wall_s": wall_s,
        "metrics": metrics,
        "results": [asdict(r) for r in results],
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Results written: {out_path}")

    if ck_path.exists() and not args.no_checkpoint:
        ck_path.unlink()
    live = ck_path.with_name(ck_path.stem + ".live.json")
    if live.exists():
        live.unlink()


if __name__ == "__main__":
    main()
