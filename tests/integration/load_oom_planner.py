"""Production-style load test for the OOM planner.

Drives a *live* kvboost server with a realistic workload (long-doc
analysis, code review, multi-turn chat, short conversational replies)
at configurable concurrency. Reports:

  - per-shape latency (p50, p95, p99, max)
  - throughput (tokens/s, requests/s)
  - error rate (413, 5xx, timeouts) broken out
  - planner calibration progression (snapshots at start, midpoint, end)
  - per-shape predicted-vs-actual peak residuals
  - mode-selection distribution (when tree speculative is enabled)

Run pattern:

  # Terminal 1: launch server (see launch_oom_planner_server.sh)
  ./tests/integration/launch_oom_planner_server.sh tight 9000

  # Terminal 2: drive the load
  python tests/integration/load_oom_planner.py \\
      --base-url http://localhost:9000 \\
      --model Qwen/Qwen2.5-3B-Instruct \\
      --workload production \\
      --concurrency 4 \\
      --duration 120

Exit code: 0 if every request either succeeded or returned a planned
413 (the planner deliberately rejecting an oversized prompt is success).
1 if any request returned an unexpected error.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx

# Local import — keep this script + workload.py side-by-side under
# tests/integration/ so the relative import works when run as a script.
sys.path.insert(0, "/Users/srihariunnikrishnan/Documents/kv_cache/tests/integration")
import workload as wl


# ── Result records ──────────────────────────────────────────────────────────


@dataclass
class RequestResult:
    shape: str
    status: int
    expected_prompt_tokens: int
    completion_tokens: int
    wall_ms: float
    error: Optional[str] = None
    planned_413: bool = False    # 413 with prompt_too_large is "success"

    @property
    def ok(self) -> bool:
        return self.status == 200 or self.planned_413


@dataclass
class ShapeStats:
    shape: str
    latencies_ms: List[float] = field(default_factory=list)
    completion_tokens: List[int] = field(default_factory=list)
    n_ok: int = 0
    n_413_planned: int = 0
    n_413_unplanned: int = 0
    n_5xx: int = 0
    n_timeout: int = 0
    n_other_err: int = 0

    def absorb(self, r: RequestResult) -> None:
        if r.status == 200:
            self.n_ok += 1
            self.latencies_ms.append(r.wall_ms)
            self.completion_tokens.append(r.completion_tokens)
        elif r.planned_413:
            self.n_413_planned += 1
        elif r.status == 413:
            self.n_413_unplanned += 1
        elif r.status >= 500:
            self.n_5xx += 1
        elif r.status == 0:    # client-side (timeout, network)
            if r.error and "timeout" in r.error.lower():
                self.n_timeout += 1
            else:
                self.n_other_err += 1
        else:
            self.n_other_err += 1

    def percentiles(self) -> Dict[str, float]:
        if not self.latencies_ms:
            return {}
        s = sorted(self.latencies_ms)
        def pick(q):
            idx = max(0, int(q * len(s)) - 1)
            return s[idx]
        return {
            "p50": pick(0.50), "p95": pick(0.95), "p99": pick(0.99),
            "max": s[-1], "min": s[0], "mean": statistics.mean(s),
        }

    def tokens_per_s(self) -> float:
        if not self.latencies_ms or not self.completion_tokens:
            return 0.0
        total_tokens = sum(self.completion_tokens)
        total_s = sum(self.latencies_ms) / 1000.0
        if total_s <= 0:
            return 0.0
        return total_tokens / total_s


# ── HTTP driver ──────────────────────────────────────────────────────────────


async def fire_one(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    item: wl.WorkloadItem,
    timeout_s: float,
) -> RequestResult:
    body = item.to_body(model)
    t0 = time.perf_counter()
    try:
        r = await client.post(
            f"{base_url}/v1/chat/completions",
            json=body, timeout=timeout_s,
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        if r.status_code == 200:
            data = r.json()
            tok = int(data.get("usage", {}).get("completion_tokens", 0))
            return RequestResult(
                shape=item.name, status=200,
                expected_prompt_tokens=item.expected_prompt_tokens,
                completion_tokens=tok, wall_ms=wall_ms,
            )
        if r.status_code == 413:
            # Planner rejection is operationally a success — the server
            # correctly identified an unfittable prompt up-front.
            try:
                err = r.json().get("detail") or r.json().get("error", {})
                planned = err.get("type") == "prompt_too_large"
            except Exception:
                planned = False
            return RequestResult(
                shape=item.name, status=413,
                expected_prompt_tokens=item.expected_prompt_tokens,
                completion_tokens=0, wall_ms=wall_ms,
                planned_413=planned,
                error=r.text[:200] if not planned else None,
            )
        return RequestResult(
            shape=item.name, status=r.status_code,
            expected_prompt_tokens=item.expected_prompt_tokens,
            completion_tokens=0, wall_ms=wall_ms,
            error=r.text[:200],
        )
    except httpx.TimeoutException as e:
        return RequestResult(
            shape=item.name, status=0,
            expected_prompt_tokens=item.expected_prompt_tokens,
            completion_tokens=0,
            wall_ms=(time.perf_counter() - t0) * 1000.0,
            error=f"timeout: {e!r}",
        )
    except Exception as e:
        return RequestResult(
            shape=item.name, status=0,
            expected_prompt_tokens=item.expected_prompt_tokens,
            completion_tokens=0,
            wall_ms=(time.perf_counter() - t0) * 1000.0,
            error=repr(e),
        )


async def worker(
    queue: asyncio.Queue, results: List[RequestResult],
    client: httpx.AsyncClient, base_url: str, model: str, timeout_s: float,
    worker_id: int, verbose: bool,
) -> None:
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            return
        r = await fire_one(client, base_url, model, item, timeout_s)
        results.append(r)
        if verbose:
            tag = "OK " if r.ok else "ERR"
            print(
                f"  [w{worker_id}] {tag} {r.shape:<20} "
                f"status={r.status} prompt≈{r.expected_prompt_tokens} "
                f"out={r.completion_tokens} {r.wall_ms:.0f}ms"
                + (f" err={r.error[:80]}" if r.error else ""),
                flush=True,
            )
        queue.task_done()


# ── Stats helpers ────────────────────────────────────────────────────────────


def fetch_stats(base_url: str) -> Dict[str, Any]:
    try:
        return httpx.get(f"{base_url}/v1/stats", timeout=5.0).json()
    except Exception as e:
        return {"error": repr(e)}


def render_planner_snapshot(s: Dict[str, Any]) -> str:
    planner = s.get("planner", {})
    cal = planner.get("calibration", {})
    cohorts = cal.get("cohorts", {})
    lines = [
        f"  free_vram_mb_now: {planner.get('free_vram_mb_now', '?'):.0f}",
        f"  calibration:",
        f"    n_samples:        {cal.get('n_samples', 0)}",
        f"    residual_median:  {cal.get('residual_median', 0):+.2%}",
        f"    residual_p95:     {cal.get('residual_p95', 0):+.2%}",
        f"    residual_max:     {cal.get('residual_max', 0):+.2%}",
        f"    residual_min:     {cal.get('residual_min', 0):+.2%}",
        f"    suggested_margin: {cal.get('suggested_margin', 0):.2%}",
    ]
    if cohorts:
        lines.append(f"    cohorts ({len(cohorts)}):")
        for key, c in sorted(cohorts.items()):
            lines.append(
                f"      {key}: n={c['n']} median_err={c['median_err']:+.2%}"
            )
    return "\n".join(lines)


def render_summary(results: List[RequestResult]) -> str:
    by_shape: Dict[str, ShapeStats] = {}
    for r in results:
        by_shape.setdefault(r.shape, ShapeStats(shape=r.shape)).absorb(r)

    lines = []
    header = (
        f"{'shape':<22} {'n_ok':>5} {'413p':>5} {'413u':>5} {'5xx':>4} "
        f"{'tout':>4} {'p50ms':>7} {'p95ms':>7} {'p99ms':>7} {'tok/s':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for shape, st in sorted(by_shape.items()):
        p = st.percentiles()
        lines.append(
            f"{shape:<22} {st.n_ok:>5} {st.n_413_planned:>5} "
            f"{st.n_413_unplanned:>5} {st.n_5xx:>4} {st.n_timeout:>4} "
            f"{p.get('p50', 0):>7.0f} {p.get('p95', 0):>7.0f} "
            f"{p.get('p99', 0):>7.0f} {st.tokens_per_s():>7.1f}"
        )

    n_total = len(results)
    n_ok = sum(1 for r in results if r.ok)
    n_unexpected = n_total - n_ok
    lines.append("")
    lines.append(
        f"totals: {n_total} requests, {n_ok} ok / planned-413, "
        f"{n_unexpected} unexpected errors"
    )
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────


async def run_load(args) -> int:
    if args.workload == "production":
        items = wl.production_mix(seed=args.seed, n=args.n_requests)
    elif args.workload == "heavy":
        items = wl.heavy_mix(seed=args.seed, n=args.n_requests)
    elif args.workload == "burst":
        items = wl.burst_short(n=args.n_requests)
    elif args.workload == "oversized":
        items = [wl.oversized() for _ in range(args.n_requests)]
    else:
        raise SystemExit(f"unknown workload: {args.workload}")

    print(f"Workload: {args.workload}, {len(items)} requests, "
          f"concurrency={args.concurrency}")
    print(f"Per-shape counts:")
    counts: Dict[str, int] = {}
    for it in items:
        counts[it.name] = counts.get(it.name, 0) + 1
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print()

    print(f"Planner snapshot (pre-load):")
    print(render_planner_snapshot(fetch_stats(args.base_url)))
    print()

    queue: asyncio.Queue = asyncio.Queue()
    for it in items:
        queue.put_nowait(it)
    for _ in range(args.concurrency):
        queue.put_nowait(None)   # sentinel per worker

    results: List[RequestResult] = []
    t_start = time.perf_counter()

    async with httpx.AsyncClient() as client:
        # Periodic mid-load snapshot.
        async def midpoint_snapshot():
            await asyncio.sleep(args.duration_s / 2 if args.duration_s > 0 else 5.0)
            print()
            print(f"Planner snapshot (midpoint, t≈{time.perf_counter() - t_start:.0f}s):")
            print(render_planner_snapshot(fetch_stats(args.base_url)))
            print()

        mid_task = asyncio.create_task(midpoint_snapshot())

        workers = [
            asyncio.create_task(worker(
                queue, results, client, args.base_url, args.model,
                args.timeout_s, worker_id=i, verbose=args.verbose,
            ))
            for i in range(args.concurrency)
        ]
        await asyncio.gather(*workers)
        mid_task.cancel()

    elapsed = time.perf_counter() - t_start

    print()
    print(f"Planner snapshot (post-load, {elapsed:.1f}s total):")
    print(render_planner_snapshot(fetch_stats(args.base_url)))
    print()
    print(f"Per-shape latency + throughput:")
    print(render_summary(results))

    n_unexpected = sum(1 for r in results if not r.ok)
    rps = len(results) / max(elapsed, 1e-3)
    print()
    print(f"Overall: {len(results)} requests in {elapsed:.1f}s "
          f"({rps:.2f} req/s), {n_unexpected} unexpected errors")
    return 0 if n_unexpected == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Production load test for the OOM planner.",
    )
    ap.add_argument(
        "--base-url", default="http://localhost:9000",
        help="Server URL (default: http://localhost:9000)",
    )
    ap.add_argument(
        "--model", default="Qwen/Qwen2.5-3B-Instruct",
        help="Model id (must match server's --model)",
    )
    ap.add_argument(
        "--workload", default="production",
        choices=["production", "heavy", "burst", "oversized"],
        help="Workload mix. 'production' = realistic; 'heavy' = bias "
             "toward long contexts; 'burst' = all short (overhead test); "
             "'oversized' = all 80K-token rejects",
    )
    ap.add_argument("--n-requests", type=int, default=30)
    ap.add_argument(
        "--concurrency", type=int, default=1,
        help="Concurrent in-flight requests (default 1). The engine is "
             "single-GPU-worker, so higher concurrency mostly piles up "
             "queue wait — raise it only to test back-pressure / batching, "
             "and raise --timeout-s to match.",
    )
    ap.add_argument(
        "--timeout-s", type=float, default=600.0,
        help="Per-request HTTP timeout (default 600s). Long-context "
             "completions queued behind others on a single GPU worker can "
             "take minutes wall-clock; 600s avoids spurious client-side "
             "ReadTimeouts that look like failures but are really the "
             "server still working.",
    )
    ap.add_argument(
        "--duration-s", type=float, default=0.0,
        help="Used only to schedule the midpoint stats snapshot "
             "(default: schedule it at the 5s mark)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--verbose", action="store_true",
        help="Print each request's outcome as it completes",
    )
    args = ap.parse_args()

    if not _wait_healthy(args.base_url):
        print(f"FAIL server at {args.base_url} is not healthy", file=sys.stderr)
        return 1

    return asyncio.run(run_load(args))


def _wait_healthy(base_url: str, timeout_s: float = 10.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base_url}/health", timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


if __name__ == "__main__":
    sys.exit(main())
