"""One-shot profiling wrapper around :mod:`demo_partial_8b`.

Forces ``KVBOOST_PROFILE=1`` for this process, runs the streaming demo
end-to-end, flushes the trace, and prints the
:func:`kvboost.streaming.analyze_profile.summarize` table.

Usage::

    python -m kvboost.streaming.profile_run \\
        --model Qwen/Qwen2.5-32B-Instruct-AWQ \\
        --keep-first-k 4 --keep-last-k 4 \\
        --prompt "Explain entropy in two sentences." \\
        --max-new-tokens 16 --quiet-stream

The flag forwarding is intentionally just ``sys.argv`` pass-through:
this script is a wrapper around the demo's argparse, not a re-parser.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

_DEFAULT_TRACE = "/tmp/kvboost_trace.jsonl"


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run demo_partial_8b with KVBOOST_PROFILE enabled, then print the breakdown.",
        add_help=False,  # let the demo's --help win when forwarded
    )
    parser.add_argument(
        "--profile-out",
        default=os.environ.get("KVBOOST_PROFILE_OUT", _DEFAULT_TRACE),
        help=f"JSONL trace path (default: {_DEFAULT_TRACE}).",
    )
    parser.add_argument(
        "--keep-trace",
        action="store_true",
        help="Don't delete the existing trace before this run "
             "(default: reset so the file only contains this run).",
    )
    parser.add_argument(
        "--include-first",
        action="store_true",
        help="Include the first iteration (TTFT) in the steady-state aggregate.",
    )
    known, forwarded = parser.parse_known_args(argv)

    # Set env BEFORE importing anything that touches the profiler.
    os.environ["KVBOOST_PROFILE"] = "1"
    os.environ["KVBOOST_PROFILE_OUT"] = known.profile_out

    from kvboost.streaming.analyze_profile import summarize  # noqa: WPS433
    from kvboost.streaming.demo_partial_8b import main as demo_main  # noqa: WPS433
    from kvboost.streaming.profile import get_profiler  # noqa: WPS433

    prof = get_profiler()
    if not prof.enabled:
        # Most likely cause: this module imported earlier in the same
        # process under different env. Profiler singleton is sticky.
        print(
            "[profile_run] profiler did not pick up KVBOOST_PROFILE=1 — "
            "is this a fresh process?",
            file=sys.stderr,
        )
        return 2

    if not known.keep_trace:
        prof.reset()

    print(f"[profile_run] tracing to {known.profile_out}", file=sys.stderr)
    rc = demo_main(forwarded)

    # Flush after the demo's torch.cuda.synchronize() at the end of
    # generate(); CUDA events are guaranteed reached, so materialize is
    # cheap.
    written = prof.flush()
    print(
        f"[profile_run] wrote {written} records to {known.profile_out}",
        file=sys.stderr,
    )

    print()
    print(summarize(known.profile_out, drop_first_iteration=not known.include_first))

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
