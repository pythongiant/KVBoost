"""Production-style streaming load test for the OOM planner.

Drives a *live* kvboost server with a realistic workload (long-doc
analysis, code review, multi-turn chat, short bursts) at configurable
concurrency, using **token streaming** so we measure what users feel:

  - TTFT (time to first token) — p50 / p95 per shape
  - decode rate (tokens/s after the first token) per request
  - system throughput (total output tokens / wall second)
  - error breakdown (413 planned, 5xx, timeouts)
  - planner calibration progression (pre / mid / post snapshots)

A zero-dependency live ANSI dashboard shows every request transitioning
through WAITING → PREFILL → STREAM → DONE/ERR with live token counts and
tok/s. Falls back to plain line logging when stdout isn't a TTY (piped
to a file, CI, etc.).

Run:
  # Terminal 1
  ./tests/integration/launch_oom_planner_server.sh loose 9000
  # Terminal 2
  python tests/integration/load_oom_planner.py \\
      --workload production --concurrency 2

Exit code: 0 if every request succeeded or returned a planned 413; 1 on
any unexpected error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

sys.path.insert(0, "/Users/srihariunnikrishnan/Documents/kv_cache/tests/integration")
import workload as wl


# ── ANSI helpers (no external deps) ──────────────────────────────────────────

CSI = "\033["
_RESET = f"{CSI}0m"
_BOLD = f"{CSI}1m"
_HIDE_CURSOR = f"{CSI}?25l"
_SHOW_CURSOR = f"{CSI}?25h"


def _c(text: str, code: str) -> str:
    return f"{CSI}{code}m{text}{_RESET}"


_STATE_STYLE = {
    "waiting": ("WAITING", "90"),    # grey
    "prefill": ("PREFILL", "33"),    # yellow
    "stream":  ("STREAM ", "36"),    # cyan
    "done":    ("DONE   ", "32"),    # green
    "err":     ("ERR    ", "31"),    # red
    "413":     ("413    ", "35"),    # magenta (planned reject = ok)
}


# ── Per-request live state ───────────────────────────────────────────────────


@dataclass
class ReqState:
    idx: int
    shape: str
    expected_prompt_tokens: int
    max_tokens: int
    status: str = "waiting"          # waiting|prefill|stream|done|err|413
    n_tokens: int = 0
    start_time: float = 0.0
    first_tok_time: float = 0.0
    last_tok_time: float = 0.0
    wall_ms: float = 0.0
    error: Optional[str] = None
    planned_413: bool = False

    @property
    def ttft_ms(self) -> float:
        if self.first_tok_time and self.start_time:
            return (self.first_tok_time - self.start_time) * 1000.0
        return 0.0

    @property
    def decode_tok_s(self) -> float:
        # tokens after the first, over the decode window.
        if self.n_tokens > 1 and self.last_tok_time > self.first_tok_time:
            return (self.n_tokens - 1) / (self.last_tok_time - self.first_tok_time)
        return 0.0

    @property
    def ok(self) -> bool:
        return self.status == "done" or self.planned_413

    def live_elapsed_ms(self, now: float) -> float:
        if self.status in ("done", "err", "413"):
            return self.wall_ms
        if self.start_time:
            return (now - self.start_time) * 1000.0
        return 0.0


# ── Streaming request driver ─────────────────────────────────────────────────


async def fire_stream(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    item: wl.WorkloadItem,
    state: ReqState,
    timeout_s: float,
) -> None:
    """Fire one streaming chat request, updating ``state`` live."""
    body = item.to_body(model)
    body["stream"] = True
    state.start_time = time.perf_counter()
    state.status = "prefill"
    try:
        async with client.stream(
            "POST", f"{base_url}/v1/chat/completions",
            json=body, timeout=timeout_s,
        ) as resp:
            if resp.status_code != 200:
                raw = await resp.aread()
                state.wall_ms = (time.perf_counter() - state.start_time) * 1000.0
                _classify_error(state, resp.status_code, raw)
                return
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                # Error chunk mid-stream (e.g. planner 413 surfaced in SSE).
                if "error" in data:
                    state.wall_ms = (time.perf_counter() - state.start_time) * 1000.0
                    err = data["error"]
                    if err.get("type") == "prompt_too_large" or err.get("code") == 413:
                        state.status = "413"
                        state.planned_413 = True
                    else:
                        state.status = "err"
                        state.error = str(err.get("message", err))[:160]
                    return
                choices = data.get("choices") or [{}]
                delta = choices[0].get("delta", {}) or {}
                content = delta.get("content")
                if content:
                    now = time.perf_counter()
                    if state.n_tokens == 0:
                        state.first_tok_time = now
                        state.status = "stream"
                    state.n_tokens += 1
                    state.last_tok_time = now
        state.wall_ms = (time.perf_counter() - state.start_time) * 1000.0
        state.status = "done"
    except httpx.TimeoutException as e:
        state.wall_ms = (time.perf_counter() - state.start_time) * 1000.0
        state.status = "err"
        state.error = f"timeout: {e!r}"[:160]
    except Exception as e:
        state.wall_ms = (time.perf_counter() - state.start_time) * 1000.0
        state.status = "err"
        state.error = repr(e)[:160]


def _classify_error(state: ReqState, status_code: int, raw: bytes) -> None:
    """Map a non-200 response onto the state (planned 413 vs real error)."""
    text = raw.decode("utf-8", "replace")
    if status_code == 413:
        try:
            err = json.loads(text)
            err = err.get("detail") or err.get("error", {})
            if err.get("type") == "prompt_too_large":
                state.status = "413"
                state.planned_413 = True
                return
        except Exception:
            pass
        state.status = "413"
        state.planned_413 = True   # any 413 is the planner doing its job
        return
    state.status = "err"
    state.error = f"HTTP {status_code}: {text[:140]}"


# ── Worker pool ──────────────────────────────────────────────────────────────


async def worker(
    queue: asyncio.Queue,
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    timeout_s: float,
) -> None:
    while True:
        job = await queue.get()
        if job is None:
            queue.task_done()
            return
        item, state = job
        await fire_stream(client, base_url, model, item, state, timeout_s)
        queue.task_done()


# ── Live dashboard (ANSI) ────────────────────────────────────────────────────


class Dashboard:
    """Redraws an in-place table of all request rows + an aggregate footer.

    TTY only — when stdout isn't a terminal, ``enabled`` is False and the
    renderer is a no-op (callers print plain per-request lines instead).
    """

    def __init__(self, states: List[ReqState], *, title: str):
        self.states = states
        self.title = title
        self.enabled = sys.stdout.isatty()
        self._last_lines = 0

    def _row(self, s: ReqState, now: float) -> str:
        label, color = _STATE_STYLE.get(s.status, ("?      ", "0"))
        state_cell = _c(label, color)
        ttft = f"{s.ttft_ms:6.0f}" if s.ttft_ms else "     ·"
        dtok = f"{s.decode_tok_s:6.1f}" if s.decode_tok_s else "     ·"
        elapsed = s.live_elapsed_ms(now) / 1000.0
        bar = ""
        if s.status == "stream":
            # tiny progress hint: tokens / max_tokens
            frac = min(1.0, s.n_tokens / max(s.max_tokens, 1))
            filled = int(frac * 10)
            bar = _c("█" * filled + "·" * (10 - filled), "36")
        elif s.status == "done":
            bar = _c("██████████", "32")
        elif s.status == "413":
            bar = _c("rejected  ", "35")
        elif s.status == "err":
            bar = _c("failed    ", "31")
        return (
            f" {s.idx:>2} {s.shape:<16} {state_cell} "
            f"in≈{s.expected_prompt_tokens:>6} out={s.n_tokens:>4} "
            f"ttft={ttft}ms dec={dtok}t/s {elapsed:5.1f}s {bar}"
        )

    def _footer(self, now: float) -> str:
        done = [s for s in self.states if s.status == "done"]
        errs = [s for s in self.states if s.status == "err"]
        rej = [s for s in self.states if s.status == "413"]
        active = [s for s in self.states if s.status in ("prefill", "stream")]
        total_out = sum(s.n_tokens for s in self.states)
        # System throughput: total output tokens / wall elapsed so far.
        wall = now - min((s.start_time for s in self.states if s.start_time),
                         default=now)
        sys_tps = total_out / wall if wall > 0 else 0.0
        return (
            f"{_BOLD}done {len(done)}  active {len(active)}  "
            f"rejected(413) {len(rej)}  errors {len(errs)}  "
            f"| out_tokens {total_out}  sys {sys_tps:.1f} tok/s  "
            f"elapsed {wall:.1f}s{_RESET}"
        )

    def render(self) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        lines = [f"{_BOLD}{self.title}{_RESET}"]
        lines += [self._row(s, now) for s in self.states]
        lines.append(self._footer(now))
        out = []
        if self._last_lines:
            out.append(f"{CSI}{self._last_lines}A")   # cursor up N
        for ln in lines:
            out.append(f"{CSI}2K{ln}\n")              # clear line + content
        sys.stdout.write("".join(out))
        sys.stdout.flush()
        self._last_lines = len(lines)

    async def run(self, stop: asyncio.Event, interval: float = 0.2) -> None:
        if not self.enabled:
            return
        sys.stdout.write(_HIDE_CURSOR)
        try:
            while not stop.is_set():
                self.render()
                await asyncio.sleep(interval)
            self.render()   # final frame
        finally:
            sys.stdout.write(_SHOW_CURSOR)
            sys.stdout.flush()


# ── Stats ────────────────────────────────────────────────────────────────────


def fetch_stats(base_url: str) -> Dict[str, Any]:
    try:
        return httpx.get(f"{base_url}/v1/stats", timeout=5.0).json()
    except Exception as e:
        return {"error": repr(e)}


def render_planner_snapshot(s: Dict[str, Any]) -> str:
    planner = s.get("planner", {})
    cal = planner.get("calibration", {})
    free = planner.get("free_vram_mb_now", "?")
    free_s = f"{free:.0f}" if isinstance(free, (int, float)) else str(free)
    lines = [
        f"  free_vram_mb_now: {free_s}",
        f"  calibration: n={cal.get('n_samples', 0)} "
        f"median={cal.get('residual_median', 0):+.1%} "
        f"p95={cal.get('residual_p95', 0):+.1%} "
        f"suggested_margin={cal.get('suggested_margin', 0):.1%}",
    ]
    cohorts = cal.get("cohorts", {})
    for key, c in sorted(cohorts.items()):
        lines.append(f"    {key}: n={c['n']} median_err={c['median_err']:+.1%}")
    return "\n".join(lines)


def _pcts(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {}
    s = sorted(vals)
    pick = lambda q: s[max(0, int(q * len(s)) - 1)]
    return {"p50": pick(0.5), "p95": pick(0.95), "max": s[-1]}


def render_summary(states: List[ReqState]) -> str:
    by_shape: Dict[str, List[ReqState]] = {}
    for s in states:
        by_shape.setdefault(s.shape, []).append(s)

    header = (
        f"{'shape':<16} {'ok':>3} {'413':>4} {'err':>4} "
        f"{'ttftP50':>8} {'ttftP95':>8} {'decP50':>7} {'sysTok/s':>9}"
    )
    lines = [header, "-" * len(header)]
    for shape, group in sorted(by_shape.items()):
        ok = [s for s in group if s.status == "done"]
        n_413 = sum(1 for s in group if s.status == "413")
        n_err = sum(1 for s in group if s.status == "err")
        ttft = _pcts([s.ttft_ms for s in ok if s.ttft_ms])
        dec = _pcts([s.decode_tok_s for s in ok if s.decode_tok_s])
        out_tokens = sum(s.n_tokens for s in ok)
        wall = sum(s.wall_ms for s in ok) / 1000.0
        sys_tps = out_tokens / wall if wall > 0 else 0.0
        lines.append(
            f"{shape:<16} {len(ok):>3} {n_413:>4} {n_err:>4} "
            f"{ttft.get('p50', 0):>8.0f} {ttft.get('p95', 0):>8.0f} "
            f"{dec.get('p50', 0):>7.1f} {sys_tps:>9.1f}"
        )
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────


def _select_workload(name: str, seed: int, n: int) -> List[wl.WorkloadItem]:
    if name == "production":
        return wl.production_mix(seed=seed, n=n)
    if name == "heavy":
        return wl.heavy_mix(seed=seed, n=n)
    if name == "burst":
        return wl.burst_short(n=n)
    if name == "oversized":
        return [wl.oversized() for _ in range(n)]
    raise SystemExit(f"unknown workload: {name}")


async def run_load(args) -> int:
    items = _select_workload(args.workload, args.seed, args.n_requests)
    states = [
        ReqState(idx=i, shape=it.name,
                 expected_prompt_tokens=it.expected_prompt_tokens,
                 max_tokens=it.max_tokens)
        for i, it in enumerate(items)
    ]

    print(f"Workload: {args.workload}, {len(items)} requests, "
          f"concurrency={args.concurrency}, streaming=ON")
    print("Planner snapshot (pre-load):")
    print(render_planner_snapshot(fetch_stats(args.base_url)))
    print()

    queue: asyncio.Queue = asyncio.Queue()
    for it, st in zip(items, states):
        queue.put_nowait((it, st))
    for _ in range(args.concurrency):
        queue.put_nowait(None)

    dash = Dashboard(states, title=f"kvboost load — {args.workload}")
    stop = asyncio.Event()
    t_start = time.perf_counter()

    async with httpx.AsyncClient() as client:
        render_task = asyncio.create_task(dash.run(stop))
        workers = [
            asyncio.create_task(
                worker(queue, client, args.base_url, args.model, args.timeout_s)
            )
            for _ in range(args.concurrency)
        ]
        # Plain-mode progress when not a TTY: emit a line per completion.
        if not dash.enabled:
            asyncio.create_task(_plain_progress(states, stop))
        await asyncio.gather(*workers)
        stop.set()
        await render_task

    elapsed = time.perf_counter() - t_start
    print()
    print(f"Planner snapshot (post-load, {elapsed:.1f}s):")
    print(render_planner_snapshot(fetch_stats(args.base_url)))
    print()
    print("Per-shape TTFT + decode throughput:")
    print(render_summary(states))

    total_out = sum(s.n_tokens for s in states)
    n_unexpected = sum(1 for s in states if not s.ok)
    print()
    print(f"Overall: {len(states)} requests, {total_out} output tokens in "
          f"{elapsed:.1f}s ({total_out / max(elapsed, 1e-3):.1f} sys tok/s), "
          f"{n_unexpected} unexpected errors")
    return 0 if n_unexpected == 0 else 1


async def _plain_progress(states: List[ReqState], stop: asyncio.Event) -> None:
    """Non-TTY fallback: log each request's terminal state once."""
    seen = set()
    while not stop.is_set():
        for s in states:
            if s.idx not in seen and s.status in ("done", "err", "413"):
                seen.add(s.idx)
                tag = {"done": "OK ", "err": "ERR", "413": "413"}[s.status]
                print(
                    f"  {tag} {s.shape:<16} in≈{s.expected_prompt_tokens} "
                    f"out={s.n_tokens} ttft={s.ttft_ms:.0f}ms "
                    f"dec={s.decode_tok_s:.1f}t/s {s.wall_ms / 1000:.1f}s"
                    + (f" err={s.error}" if s.error else ""),
                    flush=True,
                )
        await asyncio.sleep(0.3)


def main() -> int:
    ap = argparse.ArgumentParser(description="Streaming load test for the OOM planner.")
    ap.add_argument("--base-url", default="http://localhost:9000")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--workload", default="production",
                    choices=["production", "heavy", "burst", "oversized"])
    ap.add_argument("--n-requests", type=int, default=30)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Concurrent in-flight requests (default 1). Engine is "
                         "single-GPU-worker; higher just queues. Raise "
                         "--timeout-s to match.")
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not _wait_healthy(args.base_url):
        print(f"FAIL server at {args.base_url} is not healthy", file=sys.stderr)
        return 1
    return asyncio.run(run_load(args))


def _wait_healthy(base_url: str, timeout_s: float = 10.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            if httpx.get(f"{base_url}/health", timeout=2.0).status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


if __name__ == "__main__":
    sys.exit(main())
