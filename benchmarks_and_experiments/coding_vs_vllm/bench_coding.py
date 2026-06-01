"""Coding-benchmark: kvboost vs vLLM — TTFT, throughput, and OOM survival.

Both servers are OpenAI-compatible, so one streaming client drives both.
Two measurements:

  THROUGHPUT  — a realistic coding-agent mix (repo-context prompts). Per
                backend: TTFT p50/p95, decode tok/s, system tok/s, errors.

  OOM         — ramp prompt context length upward and record, per backend,
                the OUTCOME at each size:
                  COMPLETED         — streamed to finish (record TTFT/tok)
                  REJECTED_GRACEFUL — clean 4xx (vLLM "max context length"
                                      400, or kvboost planner 413): the
                                      server said no without crashing
                  OOM_FAIL          — 5xx with a CUDA/OOM message
                  CONN_DROP         — connection died mid-request (the usual
                                      symptom of a server-side OOM crash)
                  TIMEOUT           — exceeded --timeout-s
                The headline: the largest context each backend COMPLETED,
                and HOW each one fails past that. The kvboost story is
                "chunked prefill + per-request kv-bits complete prompts that
                exceed a fixed KV budget, or 413 cleanly — never crash,"
                vs vLLM crashing/OOMing past its configured ceiling.

Honest framing: vLLM's continuous batching usually wins raw throughput.
This benchmark is about (a) measuring that gap fairly and (b) showing the
OOM-survival difference. The script reports raw numbers; it does not
editorialize.

Usage
-----
    # start each server (see README), then:
    python bench_coding.py \\
        --kvboost-url http://localhost:9000 --kvboost-model Qwen/Qwen2.5-3B-Instruct \\
        --vllm-url    http://localhost:8001 --vllm-model    Qwen/Qwen2.5-3B-Instruct \\
        --mode both

Give only one of --kvboost-url / --vllm-url to bench a single backend.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import httpx

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import coding_workload as cw


# ── Outcome classification ────────────────────────────────────────────────────

COMPLETED = "COMPLETED"
REJECTED = "REJECTED_GRACEFUL"
OOM_FAIL = "OOM_FAIL"
CONN_DROP = "CONN_DROP"
TIMEOUT = "TIMEOUT"
SERVER_ERR = "SERVER_ERROR"

_OOM_MARKERS = ("out of memory", "cuda oom", "outofmemory", "cuda error",
                "no memory", "kv cache", "out of available memory")


@dataclass
class Outcome:
    backend: str
    label: str            # the shape label (code-12k …)
    target_tokens: int
    status: str           # COMPLETED / REJECTED_GRACEFUL / OOM_FAIL / ...
    http_code: int = 0
    prompt_tokens: int = 0
    cached_tokens: int = 0   # from usage.prompt_tokens_details.cached_tokens (vLLM)
    out_tokens: int = 0
    ttft_ms: float = 0.0
    wall_ms: float = 0.0
    first_tok_time: float = 0.0
    last_tok_time: float = 0.0
    detail: str = ""

    @property
    def decode_tok_s(self) -> float:
        if self.out_tokens > 1 and self.last_tok_time > self.first_tok_time:
            return (self.out_tokens - 1) / (self.last_tok_time - self.first_tok_time)
        return 0.0


def _classify_http(code: int, body_text: str) -> Tuple[str, str]:
    low = body_text.lower()
    if code == 200:
        return COMPLETED, ""
    if code in (400, 413, 422):
        return REJECTED, body_text[:200]
    if code >= 500:
        if any(m in low for m in _OOM_MARKERS):
            return OOM_FAIL, body_text[:200]
        return SERVER_ERR, body_text[:200]
    return SERVER_ERR, f"HTTP {code}: {body_text[:160]}"


# ── Streaming request ──────────────────────────────────────────────────────────


async def run_prompt(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    backend: str,
    prompt: "cw.CodingPrompt",
    timeout_s: float,
) -> Outcome:
    o = Outcome(backend=backend, label=prompt.name,
                target_tokens=prompt.target_tokens, status=SERVER_ERR)
    body = prompt.to_body(model, stream=True)
    t0 = time.perf_counter()
    try:
        async with client.stream(
            "POST", f"{base_url}/v1/chat/completions", json=body, timeout=timeout_s,
        ) as resp:
            o.http_code = resp.status_code
            if resp.status_code != 200:
                raw = (await resp.aread()).decode("utf-8", "replace")
                o.status, o.detail = _classify_http(resp.status_code, raw)
                o.wall_ms = (time.perf_counter() - t0) * 1000
                return o
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
                if "error" in data:
                    err = json.dumps(data["error"])
                    o.status, o.detail = _classify_http(
                        data["error"].get("code", 500) or 500, err)
                    o.wall_ms = (time.perf_counter() - t0) * 1000
                    return o
                ch = (data.get("choices") or [{}])[0]
                delta = (ch.get("delta") or {}).get("content")
                # capture usage if the server appends it
                usage = data.get("usage")
                if usage:
                    o.prompt_tokens = usage.get("prompt_tokens", o.prompt_tokens)
                    details = usage.get("prompt_tokens_details") or {}
                    o.cached_tokens = details.get("cached_tokens", o.cached_tokens)
                if delta:
                    now = time.perf_counter()
                    if o.out_tokens == 0:
                        o.first_tok_time = now
                        o.ttft_ms = (now - t0) * 1000
                    o.out_tokens += 1
                    o.last_tok_time = now
            o.wall_ms = (time.perf_counter() - t0) * 1000
            o.status = COMPLETED
            return o
    except httpx.TimeoutException as e:
        o.status, o.detail = TIMEOUT, repr(e)[:160]
    except (httpx.RemoteProtocolError, httpx.ConnectError, httpx.ReadError) as e:
        # Connection dropped mid-flight — the classic server-OOM-crash symptom.
        o.status, o.detail = CONN_DROP, repr(e)[:160]
    except Exception as e:
        o.status, o.detail = SERVER_ERR, repr(e)[:160]
    o.wall_ms = (time.perf_counter() - t0) * 1000
    return o


# ── Throughput mode ──────────────────────────────────────────────────────────


async def bench_throughput(
    base_url: str, model: str, backend: str, *,
    concurrency: int, n: int, timeout_s: float, seed: int,
) -> List[Outcome]:
    prompts = cw.throughput_mix(seed=seed, n=n)
    q: asyncio.Queue = asyncio.Queue()
    for p in prompts:
        q.put_nowait(p)
    for _ in range(concurrency):
        q.put_nowait(None)
    results: List[Outcome] = []

    async def worker(client):
        while True:
            p = await q.get()
            if p is None:
                return
            r = await run_prompt(client, base_url, model, backend, p, timeout_s)
            results.append(r)
            tag = "ok " if r.status == COMPLETED else r.status
            print(f"  [{backend}] {tag:<10} {r.label:<10} "
                  f"ttft={r.ttft_ms:7.0f}ms out={r.out_tokens:4d} "
                  f"dec={r.decode_tok_s:5.1f}t/s {r.wall_ms/1000:5.1f}s",
                  flush=True)

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*[worker(client) for _ in range(concurrency)])
    return results


# ── OOM ramp mode ──────────────────────────────────────────────────────────────


async def bench_oom_ramp(
    base_url: str, model: str, backend: str, *,
    contexts: List[int], timeout_s: float,
) -> List[Outcome]:
    """Sequential ramp (one at a time) so an OOM crash on one prompt doesn't
    corrupt the timing of others. Records the outcome at each context size."""
    prompts = cw.oom_ramp(contexts)
    results: List[Outcome] = []
    async with httpx.AsyncClient() as client:
        for p in prompts:
            r = await run_prompt(client, base_url, model, backend, p, timeout_s)
            results.append(r)
            mark = {
                COMPLETED: "✓ completed", REJECTED: "▲ rejected(graceful)",
                OOM_FAIL: "✗ OOM", CONN_DROP: "✗ conn-drop(crash?)",
                TIMEOUT: "✗ timeout", SERVER_ERR: "✗ error",
            }.get(r.status, r.status)
            print(f"  [{backend}] ~{p.target_tokens:>6} tok  {mark:<22} "
                  f"prompt={r.prompt_tokens or '?':>6} ttft={r.ttft_ms:7.0f}ms "
                  f"{('+'+r.detail) if r.detail else ''}", flush=True)
            # Give the server a beat to recover / GC between heavy prompts.
            await asyncio.sleep(1.0)
    return results


# ── Reporting ──────────────────────────────────────────────────────────────────


def _pcts(vals: List[float]) -> Dict[str, float]:
    if not vals:
        return {}
    s = sorted(vals)
    return {"p50": s[len(s) // 2], "p95": s[max(0, int(0.95 * len(s)) - 1)]}


def report_throughput(by_backend: Dict[str, List[Outcome]], wall_by_backend: Dict[str, float]) -> None:
    print("\n" + "=" * 78)
    print("THROUGHPUT — coding-agent mix")
    print("=" * 78)
    hdr = (f"{'backend':<10} {'ok':>3} {'err':>4} {'ttftP50':>8} {'ttftP95':>8} "
           f"{'decP50':>7} {'sysTok/s':>9} {'totTok':>7} {'wall_s':>7}")
    print(hdr)
    print("-" * len(hdr))
    for backend, outs in by_backend.items():
        ok = [o for o in outs if o.status == COMPLETED]
        err = [o for o in outs if o.status != COMPLETED]
        ttft = _pcts([o.ttft_ms for o in ok if o.ttft_ms])
        dec = _pcts([o.decode_tok_s for o in ok if o.decode_tok_s])
        tot = sum(o.out_tokens for o in ok)
        wall = wall_by_backend.get(backend, 0.0)
        sys_tps = tot / wall if wall > 0 else 0.0
        print(f"{backend:<10} {len(ok):>3} {len(err):>4} "
              f"{ttft.get('p50', 0):>8.0f} {ttft.get('p95', 0):>8.0f} "
              f"{dec.get('p50', 0):>7.1f} {sys_tps:>9.1f} {tot:>7} {wall:>7.1f}")


def report_oom(by_backend: Dict[str, List[Outcome]]) -> None:
    print("\n" + "=" * 78)
    print("OOM SURVIVAL — context ramp")
    print("=" * 78)
    backends = list(by_backend.keys())
    # union of context sizes
    sizes = sorted({o.target_tokens for outs in by_backend.values() for o in outs})
    short = {COMPLETED: "✓ ok", REJECTED: "▲ reject", OOM_FAIL: "✗ OOM",
             CONN_DROP: "✗ crash", TIMEOUT: "✗ t/o", SERVER_ERR: "✗ err"}
    hdr = f"{'ctx~tok':>8} " + " ".join(f"{b:>18}" for b in backends)
    print(hdr)
    print("-" * len(hdr))
    for sz in sizes:
        cells = []
        for b in backends:
            o = next((x for x in by_backend[b] if x.target_tokens == sz), None)
            if o is None:
                cells.append(f"{'-':>18}")
            else:
                s = short.get(o.status, o.status)
                extra = f" {o.ttft_ms/1000:.0f}s" if o.status == COMPLETED else ""
                cells.append(f"{s + extra:>18}")
        print(f"{sz:>8} " + " ".join(cells))
    print()
    for b in backends:
        outs = by_backend[b]
        completed = [o.target_tokens for o in outs if o.status == COMPLETED]
        crashed = [o.target_tokens for o in outs if o.status in (OOM_FAIL, CONN_DROP)]
        max_ok = max(completed) if completed else 0
        first_crash = min(crashed) if crashed else None
        print(f"  {b}: largest COMPLETED ≈{max_ok} tok; "
              + (f"first hard-fail (OOM/crash) ≈{first_crash} tok"
                 if first_crash else "no hard OOM/crash observed"))


# ── Main ─────────────────────────────────────────────────────────────────────


def _wait_healthy(url: str, timeout_s: float = 10.0) -> bool:
    import time as _t
    deadline = _t.time() + timeout_s
    while _t.time() < deadline:
        for path in ("/health", "/v1/models"):
            try:
                if httpx.get(f"{url}{path}", timeout=3.0).status_code < 500:
                    return True
            except Exception:
                pass
        _t.sleep(0.5)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="kvboost vs vLLM coding benchmark")
    ap.add_argument("--kvboost-url", default=None)
    ap.add_argument("--vllm-url", default=None)
    ap.add_argument("--kvboost-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--vllm-model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--mode", choices=["throughput", "oom", "both"], default="both")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--n", type=int, default=24, help="throughput request count")
    ap.add_argument("--contexts", type=int, nargs="+",
                    default=[2000, 8000, 16000, 32000, 64000, 96000],
                    help="OOM-ramp context sizes (approx tokens)")
    ap.add_argument("--timeout-s", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    targets = []
    if args.kvboost_url:
        targets.append(("kvboost", args.kvboost_url, args.kvboost_model))
    if args.vllm_url:
        targets.append(("vllm", args.vllm_url, args.vllm_model))
    if not targets:
        print("ERROR: give at least one of --kvboost-url / --vllm-url", file=sys.stderr)
        return 2

    for name, url, _ in targets:
        if not _wait_healthy(url):
            print(f"ERROR: {name} at {url} not healthy", file=sys.stderr)
            return 1

    tput: Dict[str, List[Outcome]] = {}
    tput_wall: Dict[str, float] = {}
    oom: Dict[str, List[Outcome]] = {}

    if args.mode in ("throughput", "both"):
        print("Running THROUGHPUT mix ...")
        for name, url, model in targets:
            t0 = time.perf_counter()
            tput[name] = asyncio.run(bench_throughput(
                url, model, name, concurrency=args.concurrency,
                n=args.n, timeout_s=args.timeout_s, seed=args.seed))
            tput_wall[name] = time.perf_counter() - t0

    if args.mode in ("oom", "both"):
        print("\nRunning OOM context ramp ...")
        for name, url, model in targets:
            oom[name] = asyncio.run(bench_oom_ramp(
                url, model, name, contexts=args.contexts, timeout_s=args.timeout_s))

    if tput:
        report_throughput(tput, tput_wall)
    if oom:
        report_oom(oom)
    return 0


if __name__ == "__main__":
    sys.exit(main())
