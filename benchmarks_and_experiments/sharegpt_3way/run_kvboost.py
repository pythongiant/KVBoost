#!/usr/bin/env python3
"""
3-way ShareGPT benchmark — KVBoost runner.

Stack under test:
  * Qwen2.5-7B-Instruct target  (HF, fp16)
  * Qwen2.5-1.5B-Instruct draft (HF, fp16)
  * CacheBlend KV reuse across conversation turns
  * Speculative decoding (gamma=5 by default)

Outputs results/kvboost.json in the schema shared with run_vllm.py and
run_llamacpp.py so compare.py can plot all three side-by-side.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import _common as common
from _common import (
    ConvResult, TurnResult, add_common_args, capture_run_metadata,
    checkpoint_key, compute_metrics, is_run_complete, load_sharegpt,
    print_summary, replay_conversations, setup_logging,
)
from dataclasses import asdict
from datetime import datetime, timezone

RESULTS_DIR    = Path(__file__).resolve().parent / "results"
CHECKPOINT_DIR = Path(__file__).resolve().parent / ".checkpoints"

log = logging.getLogger("sharegpt_3way.kvboost")


# ── OOM recovery ────────────────────────────────────────────────────────
#
# On CUDA OOM we inspect *which* knob is most likely the cause and adjust
# the one that frees the most VRAM with the least quality hit:
#
#   * KV cache "high" (used > HIGH_FRAC × budget) → lower max_cache_bytes
#     (and evict). This is the cheap fix: trims residency without touching
#     the model.
#   * KV cache "low"  → lower streaming residency (keep_first_k /
#     keep_last_k). Layers fall out of VRAM, paying a streaming-overhead
#     tax but unblocking the run.
#
# A second OOM on the same knob flips to the other knob; further OOMs
# halve again until we hit floors (MIN_CACHE_BYTES / MIN_KEEP), at which
# point the turn is recorded as an error and we move on.
class OOMRecovery:
    HIGH_FRAC = 0.5
    CACHE_SHRINK = 0.7
    STREAM_SHRINK = 0.5
    MIN_CACHE_BYTES = int(2.5e8)   # 250 MB floor
    MIN_KEEP = 16                  # at least 16 fully-resident layers each side

    def __init__(
        self,
        engine,
        *,
        initial_max_cache_bytes: int,
        initial_keep_first_k: int | None,
        initial_keep_last_k: int | None,
        streaming_enabled: bool,
        max_retries: int = 2,
    ):
        self.engine = engine
        self.max_cache_bytes = int(initial_max_cache_bytes)
        self.keep_first_k = initial_keep_first_k
        self.keep_last_k = initial_keep_last_k
        self.streaming_enabled = streaming_enabled
        self.max_retries = max_retries
        self.events: List[Dict[str, Any]] = []
        self._last_action: str | None = None  # to alternate when one knob is exhausted

    # ── Introspection ──
    def _cache_bytes_used(self) -> int:
        try:
            cm = self.engine.cache_manager
            fn = getattr(cm, "current_bytes", None)
            if callable(fn):
                return int(fn())
            if isinstance(fn, (int, float)):
                return int(fn)
            # Fallback: sum of per-chunk sizes if exposed
            chunks = getattr(cm, "_chunks", None)
            if chunks:
                total = 0
                for c in (chunks.values() if isinstance(chunks, dict) else chunks):
                    nb = getattr(c, "nbytes", None)
                    if nb is not None:
                        total += int(nb)
                return total
        except Exception:
            pass
        return 0

    def _cache_high(self) -> bool:
        used = self._cache_bytes_used()
        budget = max(self.max_cache_bytes, 1)
        return used > self.HIGH_FRAC * budget

    # ── Knob adjusters (return whatever we managed to change) ──
    def _lower_cache(self) -> Dict[str, Any] | None:
        old_bytes = self.max_cache_bytes
        new_bytes = int(old_bytes * self.CACHE_SHRINK)
        if new_bytes < self.MIN_CACHE_BYTES:
            return None
        cm = self.engine.cache_manager
        # Apply to whichever attribute the cache manager exposes.
        for attr in ("max_cache_bytes", "_max_cache_bytes", "max_bytes"):
            if hasattr(cm, attr):
                try:
                    setattr(cm, attr, new_bytes)
                except Exception:
                    pass
        self.max_cache_bytes = new_bytes
        try:
            cm.clear()
        except Exception:
            pass
        return {
            "action": "lower_cache",
            "old_max_cache_bytes": old_bytes,
            "new_max_cache_bytes": new_bytes,
        }

    def _lower_streaming(self) -> Dict[str, Any] | None:
        if not self.streaming_enabled or self.keep_first_k is None:
            return None
        old_first, old_last = self.keep_first_k, self.keep_last_k
        new_first = max(self.MIN_KEEP, int(old_first * self.STREAM_SHRINK))
        new_last = max(self.MIN_KEEP, int((old_last or 0) * self.STREAM_SHRINK))
        if new_first == old_first and new_last == old_last:
            return None

        # Push into whatever exposes the streaming config. Hierarchy varies
        # across kvboost versions — try the documented spots, swallow errors.
        applied = False
        model = self.engine.model
        for owner in (
            getattr(model, "streaming_model", None),
            model,
            getattr(model, "config", None),
        ):
            if owner is None:
                continue
            cfg = getattr(owner, "streaming_config", None) or owner
            if hasattr(cfg, "keep_first_k") and hasattr(cfg, "keep_last_k"):
                try:
                    cfg.keep_first_k = new_first
                    cfg.keep_last_k = new_last
                    applied = True
                except Exception:
                    pass
            # If the streaming engine has a rebalance hook, fire it.
            for hook_name in ("rebalance_residency", "refresh_residency", "_recompute_residency"):
                hook = getattr(owner, hook_name, None)
                if callable(hook):
                    try:
                        hook()
                    except Exception:
                        pass
        if not applied:
            return None

        self.keep_first_k = new_first
        self.keep_last_k = new_last
        return {
            "action": "lower_streaming",
            "old_keep_first_k": old_first,
            "old_keep_last_k": old_last,
            "keep_first_k": new_first,
            "keep_last_k": new_last,
        }

    # ── Driver ──
    def attempt(self, run_turn_fn, prompt: str) -> dict:
        import torch
        last_err: Exception | None = None
        oom_events: List[Dict[str, Any]] = []

        for attempt_idx in range(self.max_retries + 1):
            try:
                result = run_turn_fn(prompt)
                if oom_events:
                    bt = result.setdefault("backend_telemetry", {})
                    bt["oom_events"] = oom_events
                return result
            except torch.cuda.OutOfMemoryError as e:
                last_err = e
            except RuntimeError as e:
                msg = str(e).lower()
                if "out of memory" not in msg and "cuda oom" not in msg:
                    raise
                last_err = e

            cache_high = self._cache_high()
            cache_used = self._cache_bytes_used()
            cache_frac = cache_used / max(self.max_cache_bytes, 1)
            reason = (
                f"cache HIGH ({cache_used / 1e9:.2f}/{self.max_cache_bytes / 1e9:.2f} GB, "
                f"{cache_frac:.0%} of budget ≥ {self.HIGH_FRAC:.0%} → cache is the suspect)"
                if cache_high else
                f"cache LOW  ({cache_used / 1e9:.2f}/{self.max_cache_bytes / 1e9:.2f} GB, "
                f"{cache_frac:.0%} of budget < {self.HIGH_FRAC:.0%} → resident layers are the suspect)"
            )
            last_action_str = (
                f"previous action was '{self._last_action}'"
                if self._last_action else "first OOM this turn"
            )
            log.warning(
                "OOM #%d on attempt %d: %s — %s.",
                len(self.events) + 1, attempt_idx, reason, last_action_str,
            )

            # Free everything we can before trying again.
            try:
                self.engine.reset_cache()
            except Exception:
                pass
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

            # Pick the knob. If the preferred knob was already adjusted last
            # round (and didn't help), flip to the other knob.
            primary_name, secondary_name = (
                ("lower_cache", "lower_streaming") if cache_high
                else ("lower_streaming", "lower_cache")
            )
            primary, secondary = (
                (self._lower_cache, self._lower_streaming)
                if cache_high
                else (self._lower_streaming, self._lower_cache)
            )
            change = primary()
            flipped_reason: str | None = None
            if change is None:
                flipped_reason = f"{primary_name} unavailable (floor reached or disabled)"
            elif change.get("action") == self._last_action:
                flipped_reason = f"{primary_name} was already tried last attempt without recovering"
            if flipped_reason is not None:
                log.warning(
                    "OOM recovery: skipping primary knob '%s' — %s; trying secondary '%s'.",
                    primary_name, flipped_reason, secondary_name,
                )
                alt = secondary()
                if alt is not None:
                    change = alt

            if change is None:
                log.error(
                    "OOM recovery EXHAUSTED at attempt %d: "
                    "cache=%.2f GB (floor=%.2f GB), keep=%s/%s (floor=%d). "
                    "Re-raising — replay loop will record this turn as errored.",
                    attempt_idx,
                    self.max_cache_bytes / 1e9, self.MIN_CACHE_BYTES / 1e9,
                    self.keep_first_k, self.keep_last_k, self.MIN_KEEP,
                )
                break

            # Annotate + log the recovery action.
            change["attempt"] = attempt_idx
            change["reason"] = "cache_high" if cache_high else "cache_low"
            change["cache_used_gb"] = round(cache_used / 1e9, 3)
            change["cache_budget_gb"] = round(self.max_cache_bytes / 1e9, 3)
            change["flipped_to_secondary"] = flipped_reason is not None
            self.events.append(change)
            oom_events.append(change)
            self._last_action = change["action"]

            if change["action"] == "lower_cache":
                log.warning(
                    "OOM recovery → lower_cache: max_cache_bytes %.2f GB → %.2f GB "
                    "(×%.2f, reason=%s, attempt=%d). Will retry turn.",
                    change["old_max_cache_bytes"] / 1e9,
                    change["new_max_cache_bytes"] / 1e9,
                    self.CACHE_SHRINK, change["reason"], attempt_idx,
                )
            else:
                log.warning(
                    "OOM recovery → lower_streaming: keep_first_k %d→%d, "
                    "keep_last_k %d→%d (×%.2f, reason=%s, attempt=%d). Will retry turn.",
                    change["old_keep_first_k"], change["keep_first_k"],
                    change["old_keep_last_k"] or 0, change["keep_last_k"],
                    self.STREAM_SHRINK, change["reason"], attempt_idx,
                )

        # All retries failed: re-raise so the replay loop records an error.
        assert last_err is not None
        raise last_err


def build_engine(args):
    from kvboost import KVBoost
    from kvboost.speculative import SpeculativeConfig

    spec_cfg = SpeculativeConfig(
        draft_model_id=args.draft_model,
        draft_k=args.gamma,
        mode="greedy",   # bit-identical to non-spec greedy
    )

    # AWQ targets need the streaming load path (or transformers will route to
    # gptqmodel and fall over). On a small GPU (e.g. RTX 3060 12 GB), this is
    # also what lets a 7B AWQ target fit alongside a 1.5B AWQ draft.
    streaming_cfg = None
    if args.awq_streaming:
        from kvboost.streaming import StreamingConfig
        streaming_cfg = StreamingConfig(
            residency_mode=args.streaming_mode,
            keep_first_k=args.keep_first_k,
            keep_last_k=args.keep_last_k,
        )

    return KVBoost.from_pretrained(
        args.model,
        streaming_config=streaming_cfg,
        # ── KV reuse + CacheBlend prefill ──
        chunk_size=args.chunk_size,
        recompute_strategy="cacheblend",
        chunk_boundary_window=16,
        overlap_k=16,
        sink_tokens=32,
        recompute_overlap=16,
        max_cache_bytes=int(args.max_cache_bytes),
        recency_window_chunks=args.recency_window_chunks,
        kv_cache_bits=args.kv_cache_bits,
        # ── Speculative decoding ──
        speculative_config=spec_cfg,
    )


def make_run_turn(engine, max_new_tokens: int, oom_recovery: "OOMRecovery | None" = None):
    from kvboost import GenerationMode

    def _full_spec_snapshot() -> dict:
        return dict(engine.speculative_stats() or {})

    def _cache_snapshot() -> dict:
        """Best-effort snapshot of KV cache state. All fields optional —
        attribute names may shift across kvboost versions."""
        snap: dict = {}
        try:
            cm = getattr(engine, "cache_manager", None)
            if cm is not None:
                chunks = getattr(cm, "_chunks", None)
                if chunks is not None:
                    snap["num_chunks"] = len(chunks)
                bytes_used = getattr(cm, "current_bytes", None)
                if callable(bytes_used):
                    snap["bytes_used"] = int(bytes_used())
                elif isinstance(bytes_used, (int, float)):
                    snap["bytes_used"] = int(bytes_used)
        except Exception:
            pass
        try:
            cr = getattr(engine, "chunk_registry", None)
            if cr is not None:
                snap["chunk_size"] = getattr(cr, "chunk_size", None)
        except Exception:
            pass
        return snap

    def _do_run_turn(prompt: str) -> dict:
        engine.warm_chunks(prompt, position_offset=0)
        pre = _full_spec_snapshot()
        cache_pre = _cache_snapshot()

        t0 = time.perf_counter()
        result = engine.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            mode=GenerationMode.CHUNK_KV_REUSE,
            do_sample=False,
        )
        wall_total_ms = (time.perf_counter() - t0) * 1000.0

        post = _full_spec_snapshot()
        cache_post = _cache_snapshot()

        d_rounds   = post.get("rounds", 0) - pre.get("rounds", 0)
        d_accepted = post.get("accepted_total", 0) - pre.get("accepted_total", 0)
        d_proposed = post.get("draft_forwards", 0) - pre.get("draft_forwards", 0)
        d_committed = post.get("committed_total", 0) - pre.get("committed_total", 0)
        d_target_fwd = post.get("target_forwards", 0) - pre.get("target_forwards", 0)
        d_draft_time = post.get("draft_time_s", 0.0) - pre.get("draft_time_s", 0.0)
        d_verify_time = post.get("verify_time_s", 0.0) - pre.get("verify_time_s", 0.0)
        d_rollback_time = post.get("rollback_time_s", 0.0) - pre.get("rollback_time_s", 0.0)

        spec_telemetry: dict = {}
        if d_rounds > 0:
            spec_telemetry = {
                "rounds": int(d_rounds),
                "accepted": int(d_accepted),
                "proposed": int(d_proposed),
                "committed": int(d_committed),
                "target_forwards": int(d_target_fwd),
                "acceptance_rate": (d_accepted / d_proposed) if d_proposed else None,
                "avg_committed_per_round": d_committed / d_rounds,
                "draft_time_ms": d_draft_time * 1000.0,
                "verify_time_ms": d_verify_time * 1000.0,
                "rollback_time_ms": d_rollback_time * 1000.0,
            }

        backend_telemetry = {
            "kv_reuse_ratio": float(getattr(result, "kv_reuse_ratio", 0.0) or 0.0),
            "ttft_engine_ms": float(result.ttft_ms),
            "total_engine_ms": float(result.total_ms) if result.total_ms else None,
            "wall_total_ms": wall_total_ms,
            "cache_pre": cache_pre,
            "cache_post": cache_post,
            "spec": spec_telemetry,
        }

        return {
            "ttft_ms":        float(result.ttft_ms),
            "total_ms":       float(result.total_ms) if result.total_ms else wall_total_ms,
            "output_text":    result.output_text,
            "output_tokens":  int(result.generated_tokens),
            "prompt_tokens":  int(result.prompt_tokens),
            "cached_tokens":  int(result.cached_tokens),
            "spec_accepted":  int(d_accepted) if d_rounds > 0 else None,
            "spec_proposed":  int(d_proposed) if d_rounds > 0 else None,
            "spec_rounds":    int(d_rounds)   if d_rounds > 0 else None,
            "backend_telemetry": backend_telemetry,
        }

    if oom_recovery is None:
        return _do_run_turn

    def run_turn_with_oom(prompt: str) -> dict:
        return oom_recovery.attempt(_do_run_turn, prompt)

    return run_turn_with_oom


def main():
    parser = argparse.ArgumentParser(description="KVBoost 3-way ShareGPT runner")
    add_common_args(parser)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--draft-model", default="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
                        help="DraftModel always routes through StreamingCausalLM, "
                             "so this must be an AWQ checkpoint.")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--max-cache-bytes", type=float, default=3.0e9)
    parser.add_argument("--recency-window-chunks", type=int, default=16)
    parser.add_argument("--kv-cache-bits", type=int, default=16, choices=[4, 8, 16],
                        help="KV quantization bits (16=off, 8=int8, 4=int4). Saves "
                             "VRAM and a bit of decode bandwidth at long contexts.")
    parser.add_argument("--awq-streaming", action="store_true",
                        help="Load the TARGET via AWQ streaming. Required when --model "
                             "is an AWQ checkpoint (transformers otherwise routes to "
                             "gptqmodel). On small GPUs this also lets a 7B AWQ target "
                             "fit alongside the draft.")
    parser.add_argument("--streaming-mode", default="partial_resident",
                        choices=["full_resident", "partial_resident",
                                 "ffn_only_stream", "full_stream"])
    parser.add_argument("--keep-first-k", type=int, default=1024)
    parser.add_argument("--keep-last-k", type=int, default=1024)
    parser.add_argument("--oom-recovery", action="store_true", default=True,
                        help="Catch CUDA OOM, lower KV cache or streaming residency, retry. "
                             "Default on; pass --no-oom-recovery to disable.")
    parser.add_argument("--no-oom-recovery", action="store_false", dest="oom_recovery")
    parser.add_argument("--oom-max-retries", type=int, default=2,
                        help="Max OOM retries per turn before giving up (default: 2).")
    args = parser.parse_args()

    setup_logging(args.verbose, args.debug)
    print(f"\n{'=' * 72}\n  KVBoost (cacheblend + spec) — ShareGPT 3-way\n{'=' * 72}")
    print(f"  target={args.model}")
    print(f"  draft ={args.draft_model}  gamma={args.gamma}")
    print(f"  n_samples={args.n_samples}  turns={args.min_turns}-{args.max_turns}")
    print(f"{'=' * 72}\n")

    out_path = Path(args.output) if args.output else RESULTS_DIR / "kvboost.json"
    if not args.no_checkpoint and is_run_complete(out_path, args.n_samples):
        print(f"[skip] {out_path} already covers {args.n_samples} conversations; "
              "delete it or pass --no-checkpoint to force re-run.")
        return

    engine = build_engine(args)
    conversations = load_sharegpt(
        n_conversations=args.n_samples,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
        max_tokens_per_turn=args.max_tokens_per_turn,
        tokenizer=engine.tokenizer,
        max_context_tokens=args.max_context_tokens,
    )
    if not conversations:
        sys.exit("No conversations after filtering.")

    ck_path = CHECKPOINT_DIR / f"kvboost_{checkpoint_key('kvboost', args.model, args.n_samples, args.max_turns)}.json"
    meta = {"backend": "kvboost", "model": args.model, "draft": args.draft_model, "gamma": args.gamma}

    config = {
        "gamma": args.gamma,
        "recompute_strategy": "cacheblend",
        "chunk_size": args.chunk_size,
        "max_cache_bytes": int(args.max_cache_bytes),
        "recency_window_chunks": args.recency_window_chunks,
        "kv_cache_bits": args.kv_cache_bits,
        "awq_streaming": args.awq_streaming,
        "streaming_mode": args.streaming_mode if args.awq_streaming else None,
        "keep_first_k": args.keep_first_k if args.awq_streaming else None,
        "keep_last_k": args.keep_last_k if args.awq_streaming else None,
        "max_new_tokens": args.max_new_tokens,
        "n_samples": args.n_samples,
        "min_turns": args.min_turns,
        "max_turns": args.max_turns,
        "max_context_tokens": args.max_context_tokens,
        "max_tokens_per_turn": args.max_tokens_per_turn,
        "save_output_text": args.save_output_text,
        "oom_recovery": args.oom_recovery,
        "oom_max_retries": args.oom_max_retries,
    }
    run_metadata = capture_run_metadata("kvboost", config)

    oom_recovery = None
    if args.oom_recovery:
        oom_recovery = OOMRecovery(
            engine,
            initial_max_cache_bytes=int(args.max_cache_bytes),
            initial_keep_first_k=args.keep_first_k if args.awq_streaming else None,
            initial_keep_last_k=args.keep_last_k if args.awq_streaming else None,
            streaming_enabled=args.awq_streaming,
            max_retries=args.oom_max_retries,
        )

    t0 = time.perf_counter()
    results = replay_conversations(
        run_turn=make_run_turn(engine, args.max_new_tokens, oom_recovery=oom_recovery),
        count_tokens=lambda s: len(engine.tokenizer.encode(s, add_special_tokens=True)),
        reset_between_convs=engine.reset_cache,
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
    print_summary("kvboost", metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "backend": "kvboost",
        "model": args.model,
        "draft_model": args.draft_model,
        "config": config,
        "run_metadata": asdict(run_metadata),
        "wall_s": wall_s,
        "metrics": metrics,
        "oom_recovery": {
            "enabled": args.oom_recovery,
            "n_events": len(oom_recovery.events) if oom_recovery else 0,
            "final_max_cache_bytes": oom_recovery.max_cache_bytes if oom_recovery else int(args.max_cache_bytes),
            "final_keep_first_k": oom_recovery.keep_first_k if oom_recovery else (args.keep_first_k if args.awq_streaming else None),
            "final_keep_last_k": oom_recovery.keep_last_k if oom_recovery else (args.keep_last_k if args.awq_streaming else None),
            "events": oom_recovery.events if oom_recovery else [],
        },
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
