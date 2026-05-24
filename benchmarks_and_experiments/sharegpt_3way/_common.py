"""
Shared scaffolding for the 3-way ShareGPT benchmark.

Every backend runner (KVBoost / vLLM / llama.cpp) writes the SAME schema:
``RunMetadata`` (one per run) + ``ConvResult`` (one per conversation) +
``TurnResult`` (one per human turn). ``compare.py`` reads them side-by-side.

Per-turn metrics captured
-------------------------
Core latency:
- ttft_ms                 : time-to-first-token (engine-reported when available)
- total_ms                : full request wall time
- decode_ms               : total_ms - ttft_ms
- itl_ms                  : mean inter-token latency
- decode_tps              : output_tokens / (decode_ms / 1000)

Token accounting:
- prompt_tokens           : tokenized full prompt
- output_tokens           : decoded token count
- cached_tokens           : tokens served from cache (KV reuse / prefix hit)
- cache_hit_ratio         : cached_tokens / prompt_tokens
- history_tokens          : tokens in conversation history at turn start

Speculative (None outside of spec runs):
- spec_accepted / spec_proposed / spec_rounds

Output:
- stop_reason             : "eos" | "max_tokens" | "error"
- output_text_preview     : first 200 chars (always)
- output_text             : full text (opt-in: --save-output-text)
- error                   : exception message if the turn raised

System / hardware telemetry:
- turn_start_iso          : wall clock at turn start (UTC ISO)
- vram_mb_before / after  : torch.cuda.memory_allocated snapshots
- vram_mb_peak            : torch.cuda.max_memory_allocated during turn
- host_rss_mb             : process RSS via psutil
- backend_telemetry       : free-form dict of backend-specific data

Per-conversation also: start_iso, end_iso, wall_s, error_count, peak_vram_mb.

Per-run also (captured once at startup): hostname, GPU, CUDA / torch / backend
versions, command line, git SHA — in ``RunMetadata``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import signal
import socket
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger("sharegpt_3way")


# ── Data containers ─────────────────────────────────────────────────────

@dataclass
class TurnResult:
    # Identity
    conv_id: str
    turn_idx: int
    n_turns_total: int

    # Token accounting
    history_tokens: int
    prompt_tokens: int
    cached_tokens: int
    cache_hit_ratio: float

    # Core latency
    ttft_ms: float
    total_ms: float
    decode_ms: float
    output_tokens: int
    itl_ms: float
    decode_tps: float

    # Speculative (None outside spec runs)
    spec_accepted: Optional[int] = None
    spec_proposed: Optional[int] = None
    spec_rounds: Optional[int] = None

    # Outcome / output
    error: Optional[str] = None
    stop_reason: Optional[str] = None
    output_text_preview: str = ""
    output_text: Optional[str] = None

    # System telemetry
    turn_start_iso: str = ""
    vram_mb_before: Optional[float] = None
    vram_mb_after: Optional[float] = None
    vram_mb_peak: Optional[float] = None
    host_rss_mb: Optional[float] = None

    # Free-form backend dict (spec stats deltas, scheduler stats, perf counters)
    backend_telemetry: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConvResult:
    conv_id: str
    n_turns: int
    turns: List[TurnResult] = field(default_factory=list)
    start_iso: str = ""
    end_iso: str = ""
    wall_s: float = 0.0
    error_count: int = 0
    peak_vram_mb: Optional[float] = None


@dataclass
class RunMetadata:
    backend: str
    start_iso: str
    end_iso: str = ""
    hostname: str = ""
    platform: str = ""
    python_version: str = ""
    torch_version: Optional[str] = None
    transformers_version: Optional[str] = None
    cuda_version: Optional[str] = None
    cudnn_version: Optional[str] = None
    gpu_name: Optional[str] = None
    gpu_mem_total_mb: Optional[float] = None
    gpu_compute_capability: Optional[str] = None
    driver_version: Optional[str] = None
    cpu_count: int = 0
    cpu_model: str = ""
    ram_total_mb: float = 0.0
    kvboost_version: Optional[str] = None
    vllm_version: Optional[str] = None
    llama_cpp_version: Optional[str] = None
    command_line: str = ""
    git_sha: str = ""
    git_branch: str = ""
    git_dirty: bool = False
    backend_config: Dict[str, Any] = field(default_factory=dict)


# ── Telemetry helpers ───────────────────────────────────────────────────

def _gpu_mem_mb() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return None


def _gpu_peak_mem_mb() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return None


def _reset_gpu_peak_mem() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _host_rss_mb() -> Optional[float]:
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 ** 2)
    except Exception:
        return None


def _try_subprocess(cmd: List[str]) -> Optional[str]:
    try:
        out = subprocess.check_output(
            cmd, stderr=subprocess.DEVNULL, timeout=3.0,
        ).decode().strip()
        return out if out else None
    except Exception:
        return None


def _cpu_model() -> str:
    try:
        if sys.platform == "linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            out = _try_subprocess(["sysctl", "-n", "machdep.cpu.brand_string"])
            if out:
                return out
    except Exception:
        pass
    return platform.processor() or ""


def capture_run_metadata(backend: str, backend_config: Dict[str, Any]) -> RunMetadata:
    """One-shot hardware + software fingerprint at run start.

    Every field is best-effort — anything that probes for a missing dependency
    silently falls back to None / empty so this works on Linux, macOS, with or
    without torch/CUDA/nvidia-smi/git.
    """
    md = RunMetadata(
        backend=backend,
        start_iso=datetime.now(timezone.utc).isoformat(),
        hostname=socket.gethostname(),
        platform=platform.platform(),
        python_version=sys.version.split()[0],
        cpu_count=os.cpu_count() or 0,
        cpu_model=_cpu_model(),
        command_line=" ".join(sys.argv),
        backend_config=backend_config,
    )

    try:
        import torch
        md.torch_version = torch.__version__
        if torch.cuda.is_available():
            md.cuda_version = torch.version.cuda
            md.cudnn_version = str(torch.backends.cudnn.version() or "")
            props = torch.cuda.get_device_properties(0)
            md.gpu_name = props.name
            md.gpu_mem_total_mb = props.total_memory / (1024 ** 2)
            md.gpu_compute_capability = f"{props.major}.{props.minor}"
    except Exception:
        pass

    try:
        import transformers
        md.transformers_version = transformers.__version__
    except Exception:
        pass

    try:
        import psutil
        md.ram_total_mb = psutil.virtual_memory().total / (1024 ** 2)
    except Exception:
        pass

    driver = _try_subprocess(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
    )
    if driver:
        md.driver_version = driver.splitlines()[0].strip()

    sha = _try_subprocess(["git", "rev-parse", "HEAD"])
    if sha:
        md.git_sha = sha
        md.git_branch = _try_subprocess(["git", "rev-parse", "--abbrev-ref", "HEAD"]) or ""
        diff = _try_subprocess(["git", "status", "--porcelain"])
        md.git_dirty = bool(diff)

    if backend == "kvboost":
        try:
            import kvboost
            md.kvboost_version = getattr(kvboost, "__version__", "unknown")
        except Exception:
            pass
    elif backend == "vllm":
        try:
            import vllm
            md.vllm_version = getattr(vllm, "__version__", "unknown")
        except Exception:
            pass
    elif backend == "llamacpp":
        try:
            import llama_cpp
            md.llama_cpp_version = getattr(llama_cpp, "__version__", "unknown")
        except Exception:
            pass

    return md


# ── ShareGPT loading ────────────────────────────────────────────────────

def _maybe_tqdm(iterable, desc: str, total: Optional[int] = None):
    """Wrap with tqdm if available, otherwise return as-is."""
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total, mininterval=1.0)
    except ImportError:
        return iterable


def _batch_token_lengths(tokenizer, texts: List[str], batch_size: int = 256) -> List[int]:
    """Tokenize ``texts`` in batches; return per-text token counts.

    HF fast tokenizers process lists much faster than one-string-at-a-time
    calls (single Rust call instead of N Python→Rust hops). For 50k×8 turns
    this cuts the filter pass from ~60s to ~5s.
    """
    lengths: List[int] = []
    n = len(texts)
    last_log = time.perf_counter()
    for start in range(0, n, batch_size):
        chunk = texts[start:start + batch_size]
        try:
            enc = tokenizer(chunk, add_special_tokens=True, return_length=False)
            for ids in enc["input_ids"]:
                lengths.append(len(ids))
        except Exception:
            # Slow path for tokenizers that choke on batched input.
            for t in chunk:
                lengths.append(len(tokenizer.encode(t)))
        now = time.perf_counter()
        if now - last_log > 5.0:
            log.info("    tokenized %d/%d strings ...", start + len(chunk), n)
            last_log = now
    return lengths


def load_sharegpt(
    n_conversations: int,
    min_turns: int,
    max_turns: int,
    max_tokens_per_turn: int,
    tokenizer,
    max_context_tokens: Optional[int] = None,
    seed: int = 42,
) -> List[dict]:
    """Random-walk + per-conversation filter; stop once ``n_conversations`` accepted.

    Old behavior filtered the entire corpus first (770K+ tokenize calls for the
    full 94K Vicuna dump) and then sampled. Since accept-rates are ~50% on
    typical filter settings, we only need to scan ~2× n_conversations on
    average. We shuffle indices with ``seed`` for reproducibility, then walk
    one row at a time, tokenizing all turns of a candidate in a single batched
    tokenizer call — so for n_conversations=500 we hit ~1000 conversations and
    ~5000 tokenize calls instead of hundreds of thousands.
    """
    from datasets import load_dataset

    log.info("Loading anon8231489123/ShareGPT_Vicuna_unfiltered ...")
    ds = load_dataset(
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
        split="train",
    )
    n_raw = len(ds)
    log.info(f"  Raw conversations: {n_raw}")

    rng = np.random.RandomState(seed)
    order = rng.permutation(n_raw)

    log.info(
        "  Walking shuffled corpus, filtering on the fly "
        "(target=%d  filters: min_turns=%d max_turns=%d max_tokens_per_turn=%d max_context_tokens=%s)",
        n_conversations, min_turns, max_turns, max_tokens_per_turn,
        max_context_tokens,
    )

    conversations: List[dict] = []
    rej_struct = rej_turn = rej_ctx = 0
    scanned = 0
    last_log = time.perf_counter()

    pbar = _maybe_tqdm(
        range(n_raw),
        desc="sample+filter",
        total=n_conversations,
    )
    pbar_iter = iter(pbar) if hasattr(pbar, "__iter__") else None

    for idx in order:
        if len(conversations) >= n_conversations:
            break
        scanned += 1

        raw = ds[int(idx)]
        turns = raw.get("conversations", [])
        if not turns:
            rej_struct += 1
            continue

        human_count = sum(1 for t in turns if t.get("from") == "human")
        if human_count < min_turns:
            rej_struct += 1
            continue

        capped: List[dict] = []
        n_human = 0
        for t in turns:
            capped.append(t)
            if t.get("from") == "human":
                n_human += 1
                if n_human >= max_turns:
                    break

        # Single batched tokenize call per candidate conversation.
        all_texts = [t["value"] for t in capped]
        try:
            enc = tokenizer(all_texts, add_special_tokens=True)
            all_lens = [len(ids) for ids in enc["input_ids"]]
        except Exception:
            all_lens = [len(tokenizer.encode(t)) for t in all_texts]

        human_lens = [
            all_lens[i] for i, t in enumerate(capped) if t.get("from") == "human"
        ]
        if any(l > max_tokens_per_turn for l in human_lens):
            rej_turn += 1
            continue

        if max_context_tokens is not None and sum(all_lens) > max_context_tokens:
            rej_ctx += 1
            continue

        conversations.append({
            "id": raw.get("id", f"conv_{len(conversations)}"),
            "turns": capped,
        })

        if pbar_iter is not None:
            try:
                next(pbar_iter)
            except StopIteration:
                pbar_iter = None

        now = time.perf_counter()
        if now - last_log > 5.0:
            log.info(
                "    accepted %d/%d  (scanned %d, rejected struct=%d turn=%d ctx=%d)",
                len(conversations), n_conversations, scanned,
                rej_struct, rej_turn, rej_ctx,
            )
            last_log = now

    log.info(
        "  Sampled: %d/%d conversations  (scanned %d/%d  rejected struct=%d turn=%d ctx=%d)",
        len(conversations), n_conversations, scanned, n_raw,
        rej_struct, rej_turn, rej_ctx,
    )

    if len(conversations) < n_conversations:
        log.warning(
            "Only %d/%d conversations met all filters; raise max_tokens_per_turn "
            "or max_context_tokens, or lower n_samples.",
            len(conversations), n_conversations,
        )

    if conversations:
        turn_counts = [
            sum(1 for t in c["turns"] if t.get("from") == "human")
            for c in conversations
        ]
        log.info(
            f"  Turn distribution: min={min(turn_counts)} max={max(turn_counts)} "
            f"mean={np.mean(turn_counts):.1f} median={np.median(turn_counts):.1f}"
        )
    return conversations


# ── Checkpointing ───────────────────────────────────────────────────────

def checkpoint_key(backend: str, model: str, n_conversations: int, max_turns: int) -> str:
    return hashlib.md5(f"{backend}_{model}_{n_conversations}_{max_turns}".encode()).hexdigest()[:8]


def is_run_complete(out_path: Path, n_samples: int) -> bool:
    """True iff the final results JSON at ``out_path`` already covers
    ``n_samples`` conversations. Used by runners to skip engine load when
    the previous run already finished."""
    if not out_path.exists():
        return False
    try:
        with open(out_path) as f:
            data = json.load(f)
        if data.get("config", {}).get("n_samples", 0) < n_samples:
            return False
        if len(data.get("results", [])) < n_samples:
            return False
        return True
    except Exception:
        return False


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    tmp.replace(path)


def save_checkpoint(
    results: List[ConvResult],
    processed_ids: List[str],
    path: Path,
    meta: dict,
    *,
    run_metadata: Optional[RunMetadata] = None,
) -> None:
    """Atomically write checkpoint to ``path``. Safe to call frequently."""
    payload = {
        "meta": meta,
        "run_metadata": asdict(run_metadata) if run_metadata else None,
        "saved_iso": datetime.now(timezone.utc).isoformat(),
        "timestamp": time.time(),
        "processed_ids": processed_ids,
        "results": [asdict(r) for r in results],
    }
    _atomic_write_json(path, payload)
    log.debug(f"Checkpoint: {len(processed_ids)} conversations → {path.name}")


def load_checkpoint(path: Path) -> Tuple[List[ConvResult], List[str]]:
    if not path.exists():
        return [], []
    try:
        with open(path) as f:
            data = json.load(f)
        results = []
        turn_fields = set(TurnResult.__dataclass_fields__.keys())
        conv_fields = set(ConvResult.__dataclass_fields__.keys())
        for cr in data.get("results", []):
            turns = [
                TurnResult(**{k: v for k, v in t.items() if k in turn_fields})
                for t in cr.get("turns", [])
            ]
            kwargs = {k: v for k, v in cr.items() if k in conv_fields and k != "turns"}
            kwargs["turns"] = turns
            results.append(ConvResult(**kwargs))
        return results, data.get("processed_ids", [])
    except Exception as e:
        log.warning(f"Checkpoint load failed: {e} — starting fresh")
        return [], []


def _write_live_progress(
    ck_path: Path,
    conv_result: ConvResult,
    conv_idx: int,
    n_total: int,
) -> None:
    """Forensic file written after every turn — captures mid-conversation state
    so a crash mid-turn leaves at least the last completed turn visible. The
    main checkpoint (`save_checkpoint`) only fires at conversation boundaries
    because conversation-level cache state can't be safely resumed mid-stream.
    """
    live = ck_path.with_name(ck_path.stem + ".live.json")
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "conv_idx": conv_idx,
        "n_total": n_total,
        "conv_id": conv_result.conv_id,
        "turns_done": len(conv_result.turns),
        "last_turn": asdict(conv_result.turns[-1]) if conv_result.turns else None,
    }
    try:
        _atomic_write_json(live, payload)
    except Exception:
        pass


# ── Replay loop (generic over backend.run_turn) ─────────────────────────

def replay_conversations(
    *,
    run_turn: Callable[[str], dict],
    count_tokens: Callable[[str], int],
    reset_between_convs: Optional[Callable[[], None]],
    conversations: List[dict],
    ck_path: Path,
    meta: dict,
    run_metadata: Optional[RunMetadata] = None,
    no_checkpoint: bool = False,
    save_output_text: bool = False,
    on_error: str = "continue",
    progress_every: int = 5,
    max_new_tokens: int = 128,
) -> List[ConvResult]:
    """Drive ``run_turn`` over the loaded conversations.

    Per-turn flow:
      1. Read pre-turn telemetry (VRAM, RSS, wall ISO).
      2. Call backend.run_turn(prompt). On exception: record error, abort conv.
      3. Read post-turn telemetry; compute derived metrics.
      4. Persist live-progress file.

    Per-conversation flow:
      1. Reset GPU peak counter, optionally backend cache state.
      2. Run all turns.
      3. Save main checkpoint (atomic).
      4. Log progress every ``progress_every`` conversations.

    SIGINT / SIGTERM trigger a final checkpoint write before exit.
    """
    if no_checkpoint:
        all_results: List[ConvResult] = []
        processed_ids: List[str] = []
    else:
        all_results, processed_ids = load_checkpoint(ck_path)
    processed_set = set(processed_ids)

    interrupted = {"flag": False, "signal": None}

    def _save_now(reason: str) -> None:
        try:
            save_checkpoint(
                all_results, processed_ids, ck_path, meta,
                run_metadata=run_metadata,
            )
            log.info("Checkpoint saved on %s (%d conversations)", reason, len(all_results))
        except Exception as e:
            log.error("Failed to save checkpoint on %s: %s", reason, e)

    def _signal_handler(signum, frame):  # noqa: ARG001
        if interrupted["flag"]:
            # Second hit — let it through, the user really wants out
            log.warning("Second signal %s received; raising KeyboardInterrupt.", signum)
            raise KeyboardInterrupt(f"signal {signum}")
        interrupted["flag"] = True
        interrupted["signal"] = signum
        log.warning("Signal %s caught; will checkpoint and exit after current turn.", signum)

    old_sigint = signal.signal(signal.SIGINT, _signal_handler)
    try:
        old_sigterm = signal.signal(signal.SIGTERM, _signal_handler)
    except Exception:
        old_sigterm = None

    total_start = time.perf_counter()
    try:
        for conv_idx, conv in enumerate(conversations):
            if interrupted["flag"]:
                log.info("Interrupted before conversation %d — stopping.", conv_idx)
                break

            conv_id = conv["id"]
            if conv_id in processed_set:
                continue

            if reset_between_convs is not None:
                reset_between_convs()
            _reset_gpu_peak_mem()

            turns = conv["turns"]
            n_human = sum(1 for t in turns if t.get("from") == "human")

            conv_start_perf = time.perf_counter()
            conv_result = ConvResult(
                conv_id=conv_id,
                n_turns=n_human,
                start_iso=datetime.now(timezone.utc).isoformat(),
            )
            conv_peak_vram = 0.0

            history = ""
            human_turn_idx = 0
            for turn in turns:
                if turn.get("from") != "human":
                    continue
                if interrupted["flag"]:
                    break

                prompt = history + f"Human: {turn['value']}\nAssistant:"
                history_tokens = count_tokens(prompt)

                turn_start_iso = datetime.now(timezone.utc).isoformat()
                vram_before = _gpu_mem_mb()
                rss_before = _host_rss_mb()
                _reset_gpu_peak_mem()  # peak now reflects this turn only

                error_msg: Optional[str] = None
                r: Dict[str, Any] = {}
                try:
                    r = run_turn(prompt)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    error_msg = f"{type(e).__name__}: {e}"
                    log.error(
                        "Turn %s/%d failed: %s",
                        conv_id, human_turn_idx, error_msg,
                    )
                    if on_error == "raise":
                        raise

                vram_after = _gpu_mem_mb()
                vram_peak = _gpu_peak_mem_mb()
                if vram_peak is not None and vram_peak > conv_peak_vram:
                    conv_peak_vram = vram_peak

                if error_msg is not None:
                    tr = TurnResult(
                        conv_id=conv_id,
                        turn_idx=human_turn_idx,
                        n_turns_total=n_human,
                        history_tokens=history_tokens,
                        prompt_tokens=history_tokens,
                        cached_tokens=0,
                        cache_hit_ratio=0.0,
                        ttft_ms=0.0,
                        total_ms=0.0,
                        decode_ms=0.0,
                        output_tokens=0,
                        itl_ms=0.0,
                        decode_tps=0.0,
                        error=error_msg,
                        stop_reason="error",
                        turn_start_iso=turn_start_iso,
                        vram_mb_before=vram_before,
                        vram_mb_after=vram_after,
                        vram_mb_peak=vram_peak,
                        host_rss_mb=rss_before,
                    )
                    conv_result.turns.append(tr)
                    conv_result.error_count += 1
                    _write_live_progress(ck_path, conv_result, conv_idx, len(conversations))
                    # History is now broken — abort the rest of this conversation.
                    break

                decode_ms = max(r["total_ms"] - r["ttft_ms"], 0.0)
                out_tok = int(r.get("output_tokens", 0))
                itl_ms = decode_ms / max(out_tok - 1, 1)
                decode_tps = (out_tok / (decode_ms / 1000.0)) if decode_ms > 0 else 0.0
                prompt_tokens = int(r.get("prompt_tokens", history_tokens))
                cached = int(r.get("cached_tokens", 0))
                output_text = r.get("output_text", "") or ""
                preview = (
                    output_text[:200] + "…"
                    if len(output_text) > 200
                    else output_text
                )

                # Stop reason: backend's choice wins; else heuristic.
                stop_reason = r.get("stop_reason")
                if not stop_reason:
                    stop_reason = "max_tokens" if out_tok >= max_new_tokens else "eos"

                tr = TurnResult(
                    conv_id=conv_id,
                    turn_idx=human_turn_idx,
                    n_turns_total=n_human,
                    history_tokens=history_tokens,
                    prompt_tokens=prompt_tokens,
                    cached_tokens=cached,
                    cache_hit_ratio=cached / max(prompt_tokens, 1),
                    ttft_ms=float(r["ttft_ms"]),
                    total_ms=float(r["total_ms"]),
                    decode_ms=decode_ms,
                    output_tokens=out_tok,
                    itl_ms=itl_ms,
                    decode_tps=decode_tps,
                    spec_accepted=r.get("spec_accepted"),
                    spec_proposed=r.get("spec_proposed"),
                    spec_rounds=r.get("spec_rounds"),
                    stop_reason=stop_reason,
                    output_text_preview=preview,
                    output_text=output_text if save_output_text else None,
                    turn_start_iso=turn_start_iso,
                    vram_mb_before=vram_before,
                    vram_mb_after=vram_after,
                    vram_mb_peak=vram_peak,
                    host_rss_mb=rss_before,
                    backend_telemetry=dict(r.get("backend_telemetry") or {}),
                )
                conv_result.turns.append(tr)

                history = prompt + output_text + "\n"
                human_turn_idx += 1

                _write_live_progress(ck_path, conv_result, conv_idx, len(conversations))

            conv_result.end_iso = datetime.now(timezone.utc).isoformat()
            conv_result.wall_s = time.perf_counter() - conv_start_perf
            conv_result.peak_vram_mb = conv_peak_vram if conv_peak_vram > 0 else None

            all_results.append(conv_result)
            processed_ids.append(conv_id)
            processed_set.add(conv_id)

            if (conv_idx + 1) % progress_every == 0 or conv_idx == 0:
                elapsed = time.perf_counter() - total_start
                ttfts = [
                    t.ttft_ms for r2 in all_results
                    for t in r2.turns if not t.error
                ]
                ttft_p50 = float(np.percentile(ttfts, 50)) if ttfts else 0.0
                log.info(
                    "  [%d/%d] conv=%s turns=%d errs=%d ttft_p50=%.0fms "
                    "vram_peak=%.0fMB elapsed=%.0fs",
                    len(all_results), len(conversations), conv_id, n_human,
                    conv_result.error_count, ttft_p50,
                    conv_result.peak_vram_mb or 0.0, elapsed,
                )

            # Main checkpoint at every conversation boundary.
            _save_now("conversation boundary") if not no_checkpoint else None
    finally:
        signal.signal(signal.SIGINT, old_sigint)
        if old_sigterm is not None:
            try:
                signal.signal(signal.SIGTERM, old_sigterm)
            except Exception:
                pass
        if not no_checkpoint and all_results:
            _save_now("finalize")

    log.info(
        "Replay completed: %d conversations in %.1fs",
        len(all_results), time.perf_counter() - total_start,
    )
    return all_results


# ── Metrics ─────────────────────────────────────────────────────────────

def compute_metrics(results: List[ConvResult], total_wall_s: float) -> dict:
    all_turns = [t for r in results for t in r.turns]
    ok_turns = [t for t in all_turns if not t.error]
    if not ok_turns:
        return {
            "n_conversations": len(results),
            "n_turns_total": len(all_turns),
            "n_turns_errored": sum(1 for t in all_turns if t.error),
            "overall": {},
            "by_turn": {},
        }

    ttfts        = [t.ttft_ms for t in ok_turns]
    itls         = [t.itl_ms for t in ok_turns if t.output_tokens > 1]
    decode_tps   = [t.decode_tps for t in ok_turns if t.decode_tps > 0]
    out_tokens   = [t.output_tokens for t in ok_turns]
    hit_ratios   = [t.cache_hit_ratio for t in ok_turns]
    cached_tok   = [t.cached_tokens for t in ok_turns]
    prompt_tok   = [t.prompt_tokens for t in ok_turns]
    vram_peaks   = [t.vram_mb_peak for t in ok_turns if t.vram_mb_peak is not None]
    rss_values   = [t.host_rss_mb for t in ok_turns if t.host_rss_mb is not None]

    total_out_tok = sum(out_tokens)
    total_req_time_s = sum(t.total_ms for t in ok_turns) / 1000.0

    by_turn: Dict[int, dict] = defaultdict(lambda: {
        "ttfts": [], "itls": [], "decode_tps": [], "hit": [], "hist": [], "out": []
    })
    for t in ok_turns:
        by_turn[t.turn_idx]["ttfts"].append(t.ttft_ms)
        if t.output_tokens > 1:
            by_turn[t.turn_idx]["itls"].append(t.itl_ms)
        if t.decode_tps > 0:
            by_turn[t.turn_idx]["decode_tps"].append(t.decode_tps)
        by_turn[t.turn_idx]["hit"].append(t.cache_hit_ratio)
        by_turn[t.turn_idx]["hist"].append(t.history_tokens)
        by_turn[t.turn_idx]["out"].append(t.output_tokens)

    turn_metrics = {}
    for k in sorted(by_turn.keys()):
        d = by_turn[k]
        turn_metrics[k] = {
            "n": len(d["ttfts"]),
            "ttft_p50": float(np.percentile(d["ttfts"], 50)),
            "ttft_p90": float(np.percentile(d["ttfts"], 90)),
            "itl_p50":  float(np.percentile(d["itls"], 50)) if d["itls"] else None,
            "decode_tps_mean": float(np.mean(d["decode_tps"])) if d["decode_tps"] else None,
            "cache_hit_ratio_mean": float(np.mean(d["hit"])),
            "avg_history_tokens": float(np.mean(d["hist"])),
            "avg_output_tokens":  float(np.mean(d["out"])),
        }

    spec_acc = [t.spec_accepted for t in ok_turns if t.spec_accepted is not None]
    spec_prop = [t.spec_proposed for t in ok_turns if t.spec_proposed is not None]
    spec_acceptance_rate = (
        float(sum(spec_acc) / max(sum(spec_prop), 1)) if spec_acc and spec_prop else None
    )

    # Stop-reason distribution.
    stop_reasons: Dict[str, int] = defaultdict(int)
    for t in all_turns:
        stop_reasons[t.stop_reason or "unknown"] += 1

    return {
        "n_conversations": len(results),
        "n_turns_total":   len(all_turns),
        "n_turns_errored": sum(1 for t in all_turns if t.error),
        "stop_reasons":    dict(stop_reasons),
        "overall": {
            "ttft_p50_ms":      float(np.percentile(ttfts, 50)),
            "ttft_p90_ms":      float(np.percentile(ttfts, 90)),
            "ttft_p99_ms":      float(np.percentile(ttfts, 99)),
            "ttft_mean_ms":     float(np.mean(ttfts)),
            "itl_p50_ms":       float(np.percentile(itls, 50)) if itls else None,
            "itl_p90_ms":       float(np.percentile(itls, 90)) if itls else None,
            "decode_tps_mean":  float(np.mean(decode_tps)) if decode_tps else None,
            "decode_tps_p50":   float(np.percentile(decode_tps, 50)) if decode_tps else None,
            "avg_cache_hit_ratio": float(np.mean(hit_ratios)),
            "avg_cached_tokens": float(np.mean(cached_tok)),
            "avg_prompt_tokens": float(np.mean(prompt_tok)),
            "avg_output_tokens": float(np.mean(out_tokens)),
            "request_throughput_rps":   round(len(ok_turns) / max(total_wall_s, 1e-6), 4),
            "output_token_throughput":  round(total_out_tok / max(total_wall_s, 1e-6), 2),
            "per_request_tps_mean":     round(total_out_tok / max(total_req_time_s, 1e-6), 2),
            "spec_acceptance_rate":     spec_acceptance_rate,
            "peak_vram_mb_max":  float(max(vram_peaks)) if vram_peaks else None,
            "peak_vram_mb_mean": float(np.mean(vram_peaks)) if vram_peaks else None,
            "host_rss_mb_max":   float(max(rss_values)) if rss_values else None,
            "host_rss_mb_mean":  float(np.mean(rss_values)) if rss_values else None,
        },
        "by_turn": turn_metrics,
    }


# ── Pretty-printing ─────────────────────────────────────────────────────

def print_summary(backend: str, metrics: dict):
    ov = metrics.get("overall") or {}
    if not ov:
        print(f"\n{'=' * 72}\n  {backend.upper()} — no successful turns recorded\n{'=' * 72}\n")
        return
    print(f"\n{'=' * 72}")
    print(f"  {backend.upper()} — ShareGPT replay summary")
    print(f"{'=' * 72}")
    print(f"  Conversations: {metrics['n_conversations']}   Turns: {metrics['n_turns_total']}   Errors: {metrics.get('n_turns_errored', 0)}")
    print()
    print(f"  {'Metric':<32}{'Value':>20}")
    print(f"  {'-' * 32} {'-' * 19}")
    rows = [
        ("TTFT p50 (ms)",            f"{ov['ttft_p50_ms']:.1f}"),
        ("TTFT p90 (ms)",            f"{ov['ttft_p90_ms']:.1f}"),
        ("TTFT p99 (ms)",            f"{ov['ttft_p99_ms']:.1f}"),
        ("ITL p50 (ms/tok)",         f"{ov['itl_p50_ms']:.2f}" if ov.get('itl_p50_ms') else "—"),
        ("ITL p90 (ms/tok)",         f"{ov['itl_p90_ms']:.2f}" if ov.get('itl_p90_ms') else "—"),
        ("Decode tok/s (mean)",      f"{ov['decode_tps_mean']:.2f}" if ov.get('decode_tps_mean') else "—"),
        ("Cache hit ratio (mean)",   f"{ov['avg_cache_hit_ratio']:.1%}"),
        ("Avg cached tokens",        f"{ov['avg_cached_tokens']:.0f}"),
        ("Avg prompt tokens",        f"{ov['avg_prompt_tokens']:.0f}"),
        ("Avg output tokens",        f"{ov['avg_output_tokens']:.0f}"),
        ("Request throughput (rps)", f"{ov['request_throughput_rps']:.3f}"),
        ("Output tok/s (wall)",      f"{ov['output_token_throughput']:.2f}"),
        ("Spec acceptance rate",     f"{ov['spec_acceptance_rate']:.1%}" if ov.get('spec_acceptance_rate') is not None else "—"),
        ("Peak VRAM max (MB)",       f"{ov['peak_vram_mb_max']:.0f}" if ov.get('peak_vram_mb_max') else "—"),
        ("Peak VRAM mean (MB)",      f"{ov['peak_vram_mb_mean']:.0f}" if ov.get('peak_vram_mb_mean') else "—"),
        ("Host RSS max (MB)",        f"{ov['host_rss_mb_max']:.0f}" if ov.get('host_rss_mb_max') else "—"),
    ]
    for k, v in rows:
        print(f"  {k:<32}{v:>20}")
    print(f"{'=' * 72}\n")


def setup_logging(verbose: bool = False, debug: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of conversations to replay (default: 500)")
    parser.add_argument("--min-turns", type=int, default=3)
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--max-tokens-per-turn", type=int, default=512)
    parser.add_argument("--max-context-tokens", type=int, default=6000)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--gamma", type=int, default=5,
                        help="Speculative draft tokens per round (default: 5)")
    parser.add_argument("--no-checkpoint", action="store_true",
                        help="Don't read or write the checkpoint file.")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--save-output-text", action="store_true",
                        help="Persist full generated text per turn (otherwise just first 200 chars).")
    parser.add_argument("--error-mode", default="continue", choices=["continue", "raise"],
                        help="On turn failure: 'continue' records error & advances (default), "
                             "'raise' aborts the run immediately.")
    parser.add_argument("--progress-every", type=int, default=5,
                        help="Log a progress line every N conversations (default: 5).")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: results/<backend>.json)")
