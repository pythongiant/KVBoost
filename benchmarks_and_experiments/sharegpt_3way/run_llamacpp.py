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
    add_common_args, checkpoint_key, compute_metrics, load_sharegpt,
    print_summary, replay_conversations, setup_logging,
)

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

    def run_turn(self, prompt: str) -> dict:
        new_ids = self._gguf_tokenize(prompt)
        cached_before = self._prefix_match_len(new_ids)

        t0 = time.perf_counter()
        first_token_seen = False
        ttft_ms: Optional[float] = None
        output_text_chunks: list[str] = []
        output_token_count = 0

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
            if choice.get("finish_reason") is not None:
                break

        total_ms = (time.perf_counter() - t0) * 1000.0
        if ttft_ms is None:
            ttft_ms = total_ms

        # Remember the served prompt for next-turn prefix accounting.
        self._last_prompt_tokens = new_ids

        return {
            "ttft_ms":       ttft_ms,
            "total_ms":      total_ms,
            "output_text":   "".join(output_text_chunks),
            "output_tokens": output_token_count,
            "prompt_tokens": len(new_ids),
            "cached_tokens": cached_before,
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

    t0 = time.perf_counter()
    results = replay_conversations(
        run_turn=runner.run_turn,
        count_tokens=runner.count_tokens,
        reset_between_convs=runner.reset_between_convs,
        conversations=conversations,
        ck_path=ck_path,
        meta=meta,
        no_checkpoint=args.no_checkpoint,
    )
    wall_s = time.perf_counter() - t0

    metrics = compute_metrics(results, total_wall_s=wall_s)
    print_summary("llamacpp", metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output) if args.output else RESULTS_DIR / "llamacpp.json"
    payload = {
        "backend": "llamacpp",
        "model": args.model_path,
        "draft_model": args.draft_model_path,
        "config": {
            "gamma": args.gamma,
            "n_ctx": args.n_ctx,
            "n_gpu_layers": args.n_gpu_layers,
            "max_new_tokens": args.max_new_tokens,
            "n_samples": args.n_samples,
            "min_turns": args.min_turns,
            "max_turns": args.max_turns,
            "max_context_tokens": args.max_context_tokens,
            "speculative_enabled": runner._spec_enabled,
        },
        "wall_s": wall_s,
        "metrics": metrics,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"Results written: {out_path}")

    if ck_path.exists():
        ck_path.unlink()


if __name__ == "__main__":
    main()
