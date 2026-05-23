#!/usr/bin/env python3
"""
Measure PyTorch overhead during normal generation vs speculative decoding.

What "PyTorch overhead" means here
----------------------------------
For each call into ``model.forward`` we capture:

  * ``wall_ms``    — time the CPU spent inside the forward (Python dispatch +
                     C++ dispatcher + kernel launch queueing). CUDA kernels
                     run asynchronously, so this excludes GPU compute time.
  * ``gpu_ms``     — time the GPU actually spent on the kernels that this
                     forward enqueued, measured with ``torch.cuda.Event``s
                     that are queued (not synced) at the forward boundaries.

End-to-end we also capture:

  * ``end_to_end_wall_ms`` — total wall time of the generation loop, including
                             a single ``cuda.synchronize`` at the end.
  * ``end_to_end_gpu_ms``  — sum of ``gpu_ms`` across all forwards inside the
                             generation loop.

Headline metric:

  ``framework_overhead_ms = end_to_end_wall_ms - end_to_end_gpu_ms``

If positive, the run was CPU-bound — PyTorch (and the Python loop around it)
was the bottleneck and the GPU sat idle between forwards. If close to zero,
GPU compute fully overlapped CPU dispatch.

For speculative decoding we additionally pull per-phase totals (draft /
verify / rollback) out of ``engine.speculative_stats()`` so we can see where
the overhead concentrates, and we separate forwards by model (target vs
draft) so the per-call mean is meaningful.

Output
------
Prints a comparison table and writes a JSON report to ``results/``.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import statistics
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

# Allow running directly from this directory without installing kvboost.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from kvboost import KVBoost, GenerationMode  # noqa: E402
from kvboost.speculative import SpeculativeConfig  # noqa: E402

log = logging.getLogger("pytorch_overhead")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


# ─────────────────────────────────────────────────────────────────────────────
# Probe: per-forward CPU/GPU timing via monkey-patched .forward
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ForwardRecord:
    """One ``model.forward`` invocation."""
    wall_ms: float                            # CPU-side dispatch time
    start: Optional[torch.cuda.Event] = None  # queued, sync at end to read
    end: Optional[torch.cuda.Event] = None
    gpu_ms: Optional[float] = None            # filled in by ``finalize``


class ForwardTimer:
    """Monkey-patches ``model.forward`` to capture per-call CPU and GPU time.

    Uses CUDA events *without* per-call sync, so the probe itself doesn't
    serialize the host with the GPU — the only sync is in ``finalize`` after
    the whole run. That way wall-time numbers reflect the genuine cost of
    PyTorch dispatch and Python orchestration.

    On non-CUDA devices, GPU timing is unavailable and ``gpu_ms`` defaults to
    ``wall_ms`` (a conservative lower bound; on CPU the two coincide).
    """

    def __init__(self, model: torch.nn.Module, name: str) -> None:
        self.model = model
        self.name = name
        self._orig_forward: Optional[Callable] = None
        self.records: List[ForwardRecord] = []
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            self.device = torch.device("cpu")
        self._is_cuda = self.device.type == "cuda"

    def __enter__(self) -> "ForwardTimer":
        self._orig_forward = self.model.forward
        orig = self._orig_forward
        records = self.records
        is_cuda = self._is_cuda

        def wrapped(*args, **kwargs):
            if is_cuda:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                t0 = time.perf_counter()
                out = orig(*args, **kwargs)
                wall_ms = (time.perf_counter() - t0) * 1000.0
                end.record()
                records.append(ForwardRecord(wall_ms=wall_ms, start=start, end=end))
            else:
                t0 = time.perf_counter()
                out = orig(*args, **kwargs)
                wall_ms = (time.perf_counter() - t0) * 1000.0
                records.append(ForwardRecord(wall_ms=wall_ms))
            return out

        self.model.forward = wrapped
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._orig_forward is not None:
            self.model.forward = self._orig_forward
            self._orig_forward = None

    def finalize(self) -> None:
        """Sync GPU once and read all queued CUDA events."""
        if self._is_cuda:
            torch.cuda.synchronize(self.device)
        for r in self.records:
            if r.start is not None and r.end is not None:
                r.gpu_ms = r.start.elapsed_time(r.end)
            else:
                r.gpu_ms = r.wall_ms

    def summary(self, skip_first: int = 1) -> Dict[str, Any]:
        """Aggregate. ``skip_first`` drops warm-up forwards from per-call stats.

        Totals always include every recorded forward so they line up with
        end-to-end wall numbers; only per-call mean/median exclude warm-up.
        """
        if not self.records:
            return {
                "name": self.name,
                "device": str(self.device),
                "n_forwards": 0,
                "wall_ms_total": 0.0,
                "gpu_ms_total": 0.0,
                "overhead_ms_total": 0.0,
                "wall_ms_mean": 0.0,
                "wall_ms_median": 0.0,
                "wall_ms_p95": 0.0,
                "gpu_ms_mean": 0.0,
                "gpu_ms_median": 0.0,
                "gpu_ms_p95": 0.0,
            }

        wall = [r.wall_ms for r in self.records]
        gpu = [r.gpu_ms or 0.0 for r in self.records]
        wall_total = float(sum(wall))
        gpu_total = float(sum(gpu))

        eff = self.records[skip_first:] if len(self.records) > skip_first else self.records
        eff_wall = [r.wall_ms for r in eff]
        eff_gpu = [r.gpu_ms or 0.0 for r in eff]

        def pct(xs, q):
            xs = sorted(xs)
            if not xs:
                return 0.0
            k = int(q * (len(xs) - 1))
            return xs[k]

        return {
            "name": self.name,
            "device": str(self.device),
            "n_forwards": len(self.records),
            "wall_ms_total": round(wall_total, 3),
            "gpu_ms_total": round(gpu_total, 3),
            "overhead_ms_total": round(wall_total - gpu_total, 3),
            "wall_ms_mean": round(statistics.mean(eff_wall), 4) if eff_wall else 0.0,
            "wall_ms_median": round(statistics.median(eff_wall), 4) if eff_wall else 0.0,
            "wall_ms_p95": round(pct(eff_wall, 0.95), 4) if eff_wall else 0.0,
            "gpu_ms_mean": round(statistics.mean(eff_gpu), 4) if eff_gpu else 0.0,
            "gpu_ms_median": round(statistics.median(eff_gpu), 4) if eff_gpu else 0.0,
            "gpu_ms_p95": round(pct(eff_gpu, 0.95), 4) if eff_gpu else 0.0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end run wrapper
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    label: str
    prompt_tokens: int
    generated_tokens: int
    end_to_end_wall_ms: float          # measured around the whole generate()
    end_to_end_gpu_ms: float           # sum of every model's gpu_ms_total
    framework_overhead_ms: float       # wall - gpu_total (the headline number)
    overhead_per_token_ms: float
    forwards: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    spec_stats: Optional[Dict[str, Any]] = None


def _device_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def time_generation(
    timers: List[ForwardTimer],
    device: torch.device,
) -> Any:
    """Context manager that yields a dict to fill with (wall_ms, ...).

    Synchronizes once before starting and once at the end so wall time is
    a true end-to-end measurement (Python loop + dispatch + GPU + final wait).
    """
    out: Dict[str, float] = {}
    _device_sync(device)
    t0 = time.perf_counter()
    try:
        yield out
    finally:
        _device_sync(device)
        out["wall_ms"] = (time.perf_counter() - t0) * 1000.0
        for t in timers:
            t.finalize()


# ─────────────────────────────────────────────────────────────────────────────
# Three measurement scenarios
# ─────────────────────────────────────────────────────────────────────────────

def measure_normal(
    engine: KVBoost,
    prompt: str,
    max_new_tokens: int,
    label: str = "normal",
) -> RunResult:
    """Normal autoregressive generation, no speculative decoding.

    We use ``GenerationMode.BASELINE`` (no prefix cache, no chunk reuse) so
    this isolates the per-token decode loop. The engine's speculative_engine
    must be ``None`` for this run — instantiate a separate engine if needed.
    """
    if engine.speculative_engine is not None:
        raise ValueError(
            "measure_normal: this engine has speculative decoding enabled; "
            "create a separate engine without speculative_config for a fair "
            "normal-decoding measurement."
        )

    device = torch.device(engine.device)
    timer_target = ForwardTimer(engine.model, name="target")
    timers = [timer_target]

    prompt_tokens = len(engine.tokenizer.encode(prompt))

    with timer_target, time_generation(timers, device) as wall:
        result = engine.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            mode=GenerationMode.BASELINE,
            do_sample=False,
        )

    end_to_end_wall_ms = wall["wall_ms"]
    summaries = {t.name: t.summary() for t in timers}
    gpu_total = sum(s["gpu_ms_total"] for s in summaries.values())
    overhead = end_to_end_wall_ms - gpu_total
    n_gen = max(result.generated_tokens, 1)

    return RunResult(
        label=label,
        prompt_tokens=prompt_tokens,
        generated_tokens=result.generated_tokens,
        end_to_end_wall_ms=round(end_to_end_wall_ms, 3),
        end_to_end_gpu_ms=round(gpu_total, 3),
        framework_overhead_ms=round(overhead, 3),
        overhead_per_token_ms=round(overhead / n_gen, 4),
        forwards=summaries,
    )


def measure_speculative(
    engine: KVBoost,
    prompt: str,
    max_new_tokens: int,
    label: str = "speculative",
) -> RunResult:
    """Speculative decoding. ``engine`` must have ``speculative_engine``."""
    if engine.speculative_engine is None:
        raise ValueError(
            "measure_speculative: engine has no speculative_config; rebuild "
            "with SpeculativeConfig(...) to measure speculative decoding."
        )

    device = torch.device(engine.device)

    target_model = engine.model
    draft_model = engine.speculative_engine.draft.model

    timer_target = ForwardTimer(target_model, name="target")
    timer_draft = ForwardTimer(draft_model, name="draft")
    timers = [timer_target, timer_draft]

    if engine._speculative_stats is not None:
        engine._speculative_stats.reset()

    prompt_tokens = len(engine.tokenizer.encode(prompt))

    # Must use CHUNK_KV_REUSE (or PREFIX_CACHE) — the speculative handoff lives
    # in engine._decode_with_kv, which BASELINE bypasses. With CHUNK_KV_REUSE
    # on a fresh cold prompt there's no actual reuse, just the chunk-walk
    # bookkeeping (a few ms), so the measurement is still dominated by the
    # spec loop itself.
    with timer_target, timer_draft, time_generation(timers, device) as wall:
        result = engine.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            mode=GenerationMode.CHUNK_KV_REUSE,
            do_sample=False,
        )

    end_to_end_wall_ms = wall["wall_ms"]
    summaries = {t.name: t.summary() for t in timers}
    gpu_total = sum(s["gpu_ms_total"] for s in summaries.values())
    overhead = end_to_end_wall_ms - gpu_total
    n_gen = max(result.generated_tokens, 1)

    spec_stats = engine.speculative_stats() if engine._speculative_stats else None
    if spec_stats and spec_stats.get("rounds", 0) == 0:
        raise RuntimeError(
            f"Speculative decoding produced 0 rounds for {label}. The spec engine "
            "did not run — check that engine.speculative_engine is not None and "
            "that the generation mode reaches engine._decode_with_kv (BASELINE "
            "skips it)."
        )

    return RunResult(
        label=label,
        prompt_tokens=prompt_tokens,
        generated_tokens=result.generated_tokens,
        end_to_end_wall_ms=round(end_to_end_wall_ms, 3),
        end_to_end_gpu_ms=round(gpu_total, 3),
        framework_overhead_ms=round(overhead, 3),
        overhead_per_token_ms=round(overhead / n_gen, 4),
        forwards=summaries,
        spec_stats=spec_stats,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Optional torch.profiler pass — per-op CPU vs CUDA time
# ─────────────────────────────────────────────────────────────────────────────

def profile_with_profiler(
    engine: KVBoost,
    prompt: str,
    max_new_tokens: int,
    label: str,
    out_dir: Path,
) -> Dict[str, Any]:
    """Run a single generation under ``torch.profiler`` and dump a chrome trace.

    Returns a small dict of headline op aggregates; trace file is written to
    ``out_dir`` for inspection in chrome://tracing or perfetto.
    """
    from torch.profiler import profile, ProfilerActivity

    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)

    out_dir.mkdir(parents=True, exist_ok=True)
    trace_path = out_dir / f"trace_{label}.json"

    with profile(activities=activities, record_shapes=False) as prof:
        engine.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            mode=GenerationMode.BASELINE,
            do_sample=False,
        )

    try:
        prof.export_chrome_trace(str(trace_path))
    except Exception as e:
        log.warning("Could not export trace for %s: %s", label, e)

    key_avgs = prof.key_averages()
    total_cpu_us = sum(getattr(k, "cpu_time_total", 0) for k in key_avgs)
    total_cuda_us = sum(getattr(k, "cuda_time_total", 0) for k in key_avgs) if torch.cuda.is_available() else 0

    top_cpu = sorted(key_avgs, key=lambda k: -getattr(k, "cpu_time_total", 0))[:10]
    top_cuda = (
        sorted(key_avgs, key=lambda k: -getattr(k, "cuda_time_total", 0))[:10]
        if torch.cuda.is_available() else []
    )

    return {
        "trace_file": str(trace_path),
        "total_cpu_ms": round(total_cpu_us / 1000.0, 3),
        "total_cuda_ms": round(total_cuda_us / 1000.0, 3),
        "top_cpu_ops": [
            {"op": k.key, "cpu_ms": round(getattr(k, "cpu_time_total", 0) / 1000.0, 3),
             "count": k.count}
            for k in top_cpu
        ],
        "top_cuda_ops": [
            {"op": k.key, "cuda_ms": round(getattr(k, "cuda_time_total", 0) / 1000.0, 3),
             "count": k.count}
            for k in top_cuda
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Engine factories
# ─────────────────────────────────────────────────────────────────────────────

def build_normal_engine(
    model_id: str,
    max_cache_bytes: int = 2_000_000_000,
) -> KVBoost:
    log.info("Loading normal engine: %s", model_id)
    return KVBoost.from_pretrained(
        model_name=model_id,
        max_cache_bytes=max_cache_bytes,
        # everything else defaulted; BASELINE mode bypasses chunk reuse anyway
    )


def build_speculative_engine(
    target_id: str,
    draft_id: str,
    draft_k: int,
    max_cache_bytes: int = 2_000_000_000,
) -> KVBoost:
    log.info("Loading speculative engine: target=%s  draft=%s  k=%d",
             target_id, draft_id, draft_k)
    spec_cfg = SpeculativeConfig(
        draft_model_id=draft_id,
        draft_k=draft_k,
        mode="greedy",
    )
    try:
        return KVBoost.from_pretrained(
            model_name=target_id,
            max_cache_bytes=max_cache_bytes,
            speculative_config=spec_cfg,
        )
    except FileNotFoundError as e:
        if "AWQ quantization config" in str(e):
            raise SystemExit(
                f"Draft model {draft_id!r} is not an AWQ checkpoint. "
                "kvboost's DraftModel always loads via StreamingCausalLM, which "
                "requires AWQ. Pass --draft-model with an AWQ variant, e.g. "
                "Qwen/Qwen2.5-1.5B-Instruct-AWQ or Qwen/Qwen2.5-0.5B-Instruct-AWQ."
            ) from e
        raise


# ─────────────────────────────────────────────────────────────────────────────
# Prompts
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PROMPTS: Dict[str, str] = {
    "short": (
        "Explain in one paragraph how a transformer's self-attention "
        "mechanism differs from convolution."
    ),
    "medium": (
        "You are an experienced systems engineer. The following is a code "
        "review request.\n\n"
        "```python\n"
        "def merge_intervals(intervals):\n"
        "    if not intervals:\n"
        "        return []\n"
        "    intervals.sort(key=lambda x: x[0])\n"
        "    result = [intervals[0]]\n"
        "    for current in intervals[1:]:\n"
        "        if current[0] <= result[-1][1]:\n"
        "            result[-1][1] = max(result[-1][1], current[1])\n"
        "        else:\n"
        "            result.append(current)\n"
        "    return result\n"
        "```\n\n"
        "Walk through this implementation step by step, identify any edge cases "
        "that could cause incorrect behavior, propose a more robust version, "
        "and explain the time and space complexity of both. Be thorough."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# Output formatting
# ─────────────────────────────────────────────────────────────────────────────

def print_run(r: RunResult) -> None:
    print(f"\n── {r.label} ──")
    print(f"  prompt tokens         : {r.prompt_tokens}")
    print(f"  generated tokens      : {r.generated_tokens}")
    print(f"  end-to-end wall       : {r.end_to_end_wall_ms:>10.2f} ms")
    print(f"  GPU compute (sum)     : {r.end_to_end_gpu_ms:>10.2f} ms")
    print(f"  framework overhead    : {r.framework_overhead_ms:>10.2f} ms"
          f"   ({100.0 * r.framework_overhead_ms / max(r.end_to_end_wall_ms, 1e-6):.1f}% of wall)")
    print(f"  overhead per token    : {r.overhead_per_token_ms:>10.3f} ms")
    for name, s in r.forwards.items():
        if not s:
            continue
        print(f"  ── {name} forwards ({s.get('n_forwards', 0)}) ──")
        print(f"     wall mean / p95   : {s.get('wall_ms_mean', 0):.3f} / {s.get('wall_ms_p95', 0):.3f} ms")
        print(f"     gpu  mean / p95   : {s.get('gpu_ms_mean', 0):.3f} / {s.get('gpu_ms_p95', 0):.3f} ms")
        print(f"     wall_total / gpu_total : {s.get('wall_ms_total', 0):.2f} / {s.get('gpu_ms_total', 0):.2f} ms")
    if r.spec_stats:
        s = r.spec_stats
        print(f"  ── speculative phase totals ──")
        print(f"     rounds                : {s.get('rounds', 0)}")
        print(f"     acceptance_rate       : {s.get('acceptance_rate', 0):.3f}")
        print(f"     avg_committed/round   : {s.get('avg_committed_per_round', 0):.3f}")
        print(f"     draft total           : {s.get('draft_time_s', 0) * 1000:.2f} ms"
              f"   (avg {s.get('avg_draft_ms_per_forward', 0):.3f} ms / draft forward)")
        print(f"     verify total          : {s.get('verify_time_s', 0) * 1000:.2f} ms"
              f"   (avg {s.get('avg_verify_ms_per_forward', 0):.3f} ms / verify forward)")
        print(f"     rollback total        : {s.get('rollback_time_s', 0) * 1000:.2f} ms"
              f"   (avg {s.get('avg_rollback_ms_per_round', 0):.3f} ms / round)")
        engine_loop_ms = (
            r.end_to_end_wall_ms
            - (s.get("draft_time_s", 0) + s.get("verify_time_s", 0) + s.get("rollback_time_s", 0)) * 1000
        )
        print(f"     residual loop/setup   : {engine_loop_ms:.2f} ms"
              f"   (wall − draft − verify − rollback; includes prefill & sampling)")


def print_comparison(results: List[RunResult]) -> None:
    if not results:
        return
    print("\n" + "=" * 100)
    print("  COMPARISON")
    print("=" * 100)
    print(f"  {'label':<28} {'wall ms':>10} {'gpu ms':>10} {'overhead ms':>12} {'over/tok ms':>12} {'gen toks':>9}")
    print(f"  {'-' * 28} {'-' * 10} {'-' * 10} {'-' * 12} {'-' * 12} {'-' * 9}")
    for r in results:
        pct = 100.0 * r.framework_overhead_ms / max(r.end_to_end_wall_ms, 1e-6)
        print(f"  {r.label:<28} {r.end_to_end_wall_ms:>10.2f} {r.end_to_end_gpu_ms:>10.2f} "
              f"{r.framework_overhead_ms:>9.2f}({pct:>3.0f}%) {r.overhead_per_token_ms:>12.3f} "
              f"{r.generated_tokens:>9d}")
    print("=" * 100)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target-model", default="Qwen/Qwen2.5-3B",
                        help="HF model id for normal generation and the speculative target. "
                             "Any fp16/bf16 HF causal LM works (loaded via AutoModelForCausalLM).")
    parser.add_argument("--draft-model", default="Qwen/Qwen2.5-1.5B-Instruct-AWQ",
                        help="HF model id for the speculative draft model. MUST be an AWQ "
                             "checkpoint — kvboost's DraftModel always routes through "
                             "StreamingCausalLM (see src/kvboost/speculative/draft.py:77). "
                             "Tokenizer family must match the target (vocab parity is asserted).")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--draft-k", type=int, default=5,
                        help="Number of speculative draft tokens per round.")
    parser.add_argument("--prompt", default=None,
                        help="Override prompt text. If unset, runs both 'short' and 'medium' fixtures.")
    parser.add_argument("--prompt-name", choices=list(DEFAULT_PROMPTS.keys()),
                        default=None, help="Run only one named fixture prompt.")
    parser.add_argument("--warmup-runs", type=int, default=1,
                        help="Number of warmup generations per scenario before measurement.")
    parser.add_argument("--measure-runs", type=int, default=3,
                        help="Number of measurement runs per scenario (results are averaged in JSON; individual runs printed).")
    parser.add_argument("--profiler", action="store_true",
                        help="Also run a torch.profiler pass for each scenario (slower, writes chrome traces).")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path (default: results/overhead_<timestamp>.json).")
    parser.add_argument("--skip-normal", action="store_true")
    parser.add_argument("--skip-speculative", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        log.warning("CUDA not available — GPU timing falls back to wall time; "
                    "the overhead delta will read as ~0 on CPU/MPS.")

    # Build prompts list
    if args.prompt:
        prompts = [("custom", args.prompt)]
    elif args.prompt_name:
        prompts = [(args.prompt_name, DEFAULT_PROMPTS[args.prompt_name])]
    else:
        prompts = list(DEFAULT_PROMPTS.items())

    all_results: List[RunResult] = []
    profiler_results: Dict[str, Any] = {}

    # ── Normal generation ──────────────────────────────────────────────
    if not args.skip_normal:
        engine = build_normal_engine(args.target_model)
        try:
            for pname, prompt in prompts:
                # Warmup
                for w in range(args.warmup_runs):
                    log.info("[normal/%s] warmup %d/%d", pname, w + 1, args.warmup_runs)
                    engine.generate(prompt, max_new_tokens=min(16, args.max_new_tokens),
                                    mode=GenerationMode.BASELINE, do_sample=False)
                # Measure
                for run_i in range(args.measure_runs):
                    label = f"normal/{pname}/run{run_i + 1}"
                    log.info("[%s] measuring ...", label)
                    r = measure_normal(engine, prompt, args.max_new_tokens, label=label)
                    print_run(r)
                    all_results.append(r)
                if args.profiler:
                    pkey = f"normal/{pname}"
                    log.info("[%s] running torch.profiler pass ...", pkey)
                    profiler_results[pkey] = profile_with_profiler(
                        engine, prompt, args.max_new_tokens, pkey, RESULTS_DIR / "traces"
                    )
        finally:
            del engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # ── Speculative decoding ───────────────────────────────────────────
    if not args.skip_speculative:
        engine = build_speculative_engine(args.target_model, args.draft_model, args.draft_k)
        try:
            for pname, prompt in prompts:
                # Warmup
                for w in range(args.warmup_runs):
                    log.info("[spec/%s] warmup %d/%d", pname, w + 1, args.warmup_runs)
                    engine.generate(prompt, max_new_tokens=min(16, args.max_new_tokens),
                                    mode=GenerationMode.BASELINE, do_sample=False)
                # Measure
                for run_i in range(args.measure_runs):
                    label = f"spec/{pname}/run{run_i + 1}"
                    log.info("[%s] measuring ...", label)
                    r = measure_speculative(engine, prompt, args.max_new_tokens, label=label)
                    print_run(r)
                    all_results.append(r)
                if args.profiler:
                    pkey = f"spec/{pname}"
                    log.info("[%s] running torch.profiler pass ...", pkey)
                    profiler_results[pkey] = profile_with_profiler(
                        engine, prompt, args.max_new_tokens, pkey, RESULTS_DIR / "traces"
                    )
        finally:
            del engine
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print_comparison(all_results)

    out_path = args.output or RESULTS_DIR / f"overhead_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "target_model": args.target_model,
            "draft_model": args.draft_model,
            "max_new_tokens": args.max_new_tokens,
            "draft_k": args.draft_k,
            "warmup_runs": args.warmup_runs,
            "measure_runs": args.measure_runs,
            "cuda_available": torch.cuda.is_available(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "runs": [asdict(r) for r in all_results],
        "profiler": profiler_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
