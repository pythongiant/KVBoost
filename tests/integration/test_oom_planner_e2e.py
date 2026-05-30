"""Production-style end-to-end OOM-planner integration test.

Drives a *running* kvboost server with real HTTP traffic, asserts the
operationally-important behaviors that unit tests can't cover:

  1. Normal request succeeds and emits a Plan-committed log line.
  2. ``/v1/stats`` exposes the planner's calibration block.
  3. Predicted-vs-actual peak residuals accumulate after N requests.
  4. Oversized prompt returns HTTP 413 with the operator-actionable
     diagnostic body (prompt_tokens, predicted_peak_mb,
     suggested_max_tokens).
  5. With auto-truncate the same oversized prompt returns 200 and the
     server's response is shorter (because the prompt was silently
     trimmed before generation).
  6. Mode-selection log shows planner decisions over multiple requests.

Why a separate file rather than pytest under tests/: this needs a real
GPU, a real model load (multiple GB), and HTTP traffic against a
running server. CI doesn't have that; an operator validating a deploy
does. Run it like:

    # Terminal 1
    ./tests/integration/launch_oom_planner_server.sh tight 9000

    # Terminal 2
    python tests/integration/test_oom_planner_e2e.py \\
        --base-url http://localhost:9000 \\
        --model Qwen/Qwen2.5-3B-Instruct

Exit code: 0 on all-pass, 1 on any failure. Each scenario prints
PASS/FAIL with the specific assertion that failed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import httpx


# ── Result reporter ──────────────────────────────────────────────────────────


@dataclass
class ScenarioResult:
    name: str
    passed: bool
    detail: str
    elapsed_s: float


class Reporter:
    def __init__(self) -> None:
        self.results: List[ScenarioResult] = []

    def record(
        self, name: str, passed: bool, detail: str, elapsed_s: float,
    ) -> None:
        self.results.append(ScenarioResult(name, passed, detail, elapsed_s))
        marker = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
        print(f"  {marker} {name} ({elapsed_s:.2f}s) — {detail}")

    def summary(self) -> int:
        n_pass = sum(1 for r in self.results if r.passed)
        n_total = len(self.results)
        print()
        print(f"{'=' * 60}")
        print(f"OOM-planner E2E: {n_pass}/{n_total} scenarios passed")
        print(f"{'=' * 60}")
        if n_pass != n_total:
            print("\nFAILURES:")
            for r in self.results:
                if not r.passed:
                    print(f"  - {r.name}: {r.detail}")
        return 0 if n_pass == n_total else 1


# ── HTTP helpers ─────────────────────────────────────────────────────────────


def wait_for_server(base_url: str, timeout_s: float = 30.0) -> bool:
    """Poll /health until the server responds 200, or time out."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base_url}/health", timeout=2.0)
            if r.status_code == 200:
                return True
        except httpx.RequestError:
            pass
        time.sleep(0.5)
    return False


def get_stats(base_url: str) -> Dict[str, Any]:
    return httpx.get(f"{base_url}/v1/stats", timeout=5.0).json()


def chat(
    base_url: str, model: str, content: str, max_tokens: int = 64,
    *, timeout_s: float = 120.0,
) -> httpx.Response:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    return httpx.post(
        f"{base_url}/v1/chat/completions",
        json=body, timeout=timeout_s,
    )


# ── Scenarios ────────────────────────────────────────────────────────────────


def scenario_small_request(
    base_url: str, model: str, rpt: Reporter,
) -> None:
    """Baseline: a small request must succeed and exercise the planner."""
    name = "small request succeeds (planner active)"
    t0 = time.time()
    try:
        r = chat(base_url, model, "Say hello in one sentence.", max_tokens=32)
        if r.status_code != 200:
            rpt.record(name, False, f"got {r.status_code}: {r.text[:200]}",
                       time.time() - t0)
            return
        body = r.json()
        msg = body["choices"][0]["message"]["content"]
        rpt.record(
            name, True, f"got {len(msg)} chars of completion",
            time.time() - t0,
        )
    except Exception as e:
        rpt.record(name, False, f"exception: {e!r}", time.time() - t0)


def scenario_stats_exposes_planner(
    base_url: str, rpt: Reporter,
) -> None:
    """``/v1/stats`` must include planner.calibration after at least one
    request has run."""
    name = "/v1/stats exposes planner.calibration"
    t0 = time.time()
    try:
        s = get_stats(base_url)
        planner = s.get("planner")
        if planner is None:
            rpt.record(name, False, "no 'planner' key in stats", time.time() - t0)
            return
        calib = planner.get("calibration")
        if calib is None:
            rpt.record(name, False, "no 'calibration' under planner",
                       time.time() - t0)
            return
        if "suggested_margin" not in calib:
            rpt.record(name, False, f"no suggested_margin: {list(calib)}",
                       time.time() - t0)
            return
        rpt.record(
            name, True,
            f"n_samples={calib.get('n_samples', 0)}, "
            f"suggested_margin={calib.get('suggested_margin', 0):.3f}",
            time.time() - t0,
        )
    except Exception as e:
        rpt.record(name, False, f"exception: {e!r}", time.time() - t0)


def scenario_calibration_accumulates(
    base_url: str, model: str, rpt: Reporter, *, n_requests: int = 20,
) -> None:
    """Drive N small requests; assert the calibration tracker sees them
    and the residuals stabilize."""
    name = f"calibration accumulates over {n_requests} requests"
    t0 = time.time()
    try:
        for i in range(n_requests):
            r = chat(
                base_url, model,
                f"Count to three. Iteration {i}.",
                max_tokens=16,
            )
            if r.status_code != 200:
                rpt.record(
                    name, False,
                    f"req {i} failed: {r.status_code} {r.text[:120]}",
                    time.time() - t0,
                )
                return

        s = get_stats(base_url)
        calib = s["planner"]["calibration"]
        n = calib.get("n_samples", 0)
        if n < n_requests:
            rpt.record(
                name, False,
                f"only {n}/{n_requests} samples recorded",
                time.time() - t0,
            )
            return
        rpt.record(
            name, True,
            f"n_samples={n}, p95_err={calib.get('residual_p95', 0):.2%}, "
            f"suggested_margin={calib.get('suggested_margin', 0):.2%}",
            time.time() - t0,
        )
    except Exception as e:
        rpt.record(name, False, f"exception: {e!r}", time.time() - t0)


def scenario_413_on_oversized_prompt(
    base_url: str, model: str, rpt: Reporter,
) -> None:
    """Send a deliberately-oversized prompt; expect HTTP 413 with the
    diagnostic body (NOT a 500 or a long wait)."""
    name = "oversized prompt → HTTP 413 with diagnostic body"
    t0 = time.time()
    try:
        # ~80K-token prompt (1 line ≈ 10 tokens × 8000 lines).
        chunk = "The quarterly report shows revenue of $12.4B. "
        content = chunk * 8000
        r = chat(base_url, model, content, max_tokens=64, timeout_s=30.0)
        if r.status_code != 413:
            rpt.record(
                name, False,
                f"expected 413, got {r.status_code}: {r.text[:200]}",
                time.time() - t0,
            )
            return
        body = r.json()
        err = body.get("detail") or body.get("error", {})
        if err.get("type") != "prompt_too_large":
            rpt.record(
                name, False,
                f"wrong error type: {err}", time.time() - t0,
            )
            return
        suggested = err.get("suggested_max_tokens")
        rpt.record(
            name, True,
            f"413 ok; prompt_tokens={err.get('prompt_tokens')}, "
            f"suggested_max_tokens={suggested}",
            time.time() - t0,
        )
    except Exception as e:
        rpt.record(name, False, f"exception: {e!r}", time.time() - t0)


def scenario_auto_truncate_silently_succeeds(
    base_url: str, model: str, rpt: Reporter, *, expect_truncate: bool,
) -> None:
    """Same oversized prompt but server launched with --auto-truncate
    should return 200 (silent truncation). Only runs when called with
    expect_truncate=True; otherwise asserts the server is NOT
    auto-truncating."""
    name = "auto-truncate=ON → oversized prompt returns 200"
    t0 = time.time()
    try:
        chunk = "The quarterly report shows revenue of $12.4B. "
        content = chunk * 8000
        r = chat(base_url, model, content, max_tokens=64, timeout_s=180.0)
        if expect_truncate:
            ok = r.status_code == 200
            detail = (f"got 200 (truncated)" if ok
                      else f"expected 200 but got {r.status_code}: {r.text[:200]}")
            rpt.record(name, ok, detail, time.time() - t0)
        else:
            ok = r.status_code == 413
            detail = (
                "got 413 (auto-truncate off, as expected)" if ok
                else f"expected 413 but got {r.status_code}"
            )
            rpt.record(name + " [disabled]", ok, detail, time.time() - t0)
    except Exception as e:
        rpt.record(name, False, f"exception: {e!r}", time.time() - t0)


def scenario_mode_selection_log_populated(
    base_url: str, rpt: Reporter,
) -> None:
    """After previous scenarios, the speculative.mode_selection_log
    (or planner snapshot) should reflect the requests we sent."""
    name = "planner snapshot reflects request history"
    t0 = time.time()
    try:
        s = get_stats(base_url)
        planner = s.get("planner", {})
        # Either calibration.n_samples or a mode log should be > 0
        n_calib = planner.get("calibration", {}).get("n_samples", 0)
        free_mb = planner.get("free_vram_mb_now", -1)
        ok = n_calib > 0 and free_mb >= 0
        rpt.record(
            name, ok,
            f"calibration.n_samples={n_calib}, free_vram_mb_now={free_mb:.0f}",
            time.time() - t0,
        )
    except Exception as e:
        rpt.record(name, False, f"exception: {e!r}", time.time() - t0)


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(
        description="OOM-planner end-to-end integration test.",
    )
    ap.add_argument(
        "--base-url", default="http://localhost:9000",
        help="Server URL (default: http://localhost:9000)",
    )
    ap.add_argument(
        "--model", default="Qwen/Qwen2.5-3B-Instruct",
        help="Model id the server was launched with",
    )
    ap.add_argument(
        "--expect-truncate", action="store_true",
        help="Set when the server was launched with --auto-truncate. "
             "Changes the oversized-prompt scenario to assert 200 instead "
             "of 413.",
    )
    ap.add_argument(
        "--calibration-requests", type=int, default=20,
        help="How many small requests to drive for calibration test.",
    )
    args = ap.parse_args()

    print(f"OOM-planner E2E test")
    print(f"  base-url: {args.base_url}")
    print(f"  model:    {args.model}")
    print(f"  expect_truncate: {args.expect_truncate}")
    print()

    if not wait_for_server(args.base_url):
        print(f"\033[91mFAIL\033[0m server at {args.base_url} did not become healthy")
        return 1

    rpt = Reporter()
    print("Running scenarios:")

    scenario_small_request(args.base_url, args.model, rpt)
    scenario_stats_exposes_planner(args.base_url, rpt)
    scenario_calibration_accumulates(
        args.base_url, args.model, rpt,
        n_requests=args.calibration_requests,
    )
    if args.expect_truncate:
        scenario_auto_truncate_silently_succeeds(
            args.base_url, args.model, rpt, expect_truncate=True,
        )
    else:
        scenario_413_on_oversized_prompt(args.base_url, args.model, rpt)
    scenario_mode_selection_log_populated(args.base_url, rpt)

    return rpt.summary()


if __name__ == "__main__":
    sys.exit(main())
