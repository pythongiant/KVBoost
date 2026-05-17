"""Tests for the streaming profiler.

Covers the CPU/MPS fallback path (no CUDA required). The CUDA-event
path is exercised by the live ``profile_run`` demo, not here — mocking
``torch.cuda.Event`` is brittle and the hot path it gates is plain
arithmetic on event handles.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from kvboost.streaming.profile import StreamingProfiler


@pytest.fixture
def profiler(tmp_path):
    """Fresh profiler with a clean JSONL path; resets the singleton so
    later tests don't see this one's flag state.
    """
    out = tmp_path / "trace.jsonl"
    StreamingProfiler._reset_singleton()
    os.environ["KVBOOST_PROFILE"] = "1"
    os.environ["KVBOOST_PROFILE_OUT"] = str(out)
    yield StreamingProfiler.instance()
    StreamingProfiler._reset_singleton()
    os.environ.pop("KVBOOST_PROFILE", None)
    os.environ.pop("KVBOOST_PROFILE_OUT", None)


def test_disabled_profiler_is_noop(tmp_path):
    StreamingProfiler._reset_singleton()
    os.environ.pop("KVBOOST_PROFILE", None)
    prof = StreamingProfiler.instance()
    assert not prof.enabled

    with prof.region("never.runs"):
        pass
    handle = prof.start("also.never")
    prof.end(handle)
    assert prof.records_snapshot() == []
    assert prof.flush() == 0


def test_region_records_dt_and_metadata(profiler):
    profiler.bump_iteration()  # iteration -> 1
    with profiler.region("alpha", layer_idx=3, sub_path="self_attn.q_proj"):
        time.sleep(0.005)

    snap = profiler.records_snapshot()
    assert len(snap) == 1
    rec = snap[0]
    assert rec["name"] == "alpha"
    assert rec["layer_idx"] == 3
    assert rec["sub_path"] == "self_attn.q_proj"
    assert rec["iteration"] == 1
    # CPU path uses time.perf_counter — should clear ~5 ms with slack.
    assert rec["dt_ms"] is not None
    assert rec["dt_ms"] >= 4.0


def test_iteration_bumps_separate_records(profiler):
    for _ in range(3):
        profiler.bump_iteration()
        with profiler.region("forward.total"):
            pass

    snap = profiler.records_snapshot()
    iters = sorted(r["iteration"] for r in snap)
    assert iters == [1, 2, 3]


def test_split_start_end_pairs_across_hooks(profiler):
    """Mirrors the pre-hook / post-hook pattern in model_shell."""
    profiler.bump_iteration()
    h = profiler.start("model.forward.total")
    time.sleep(0.002)
    profiler.end(h)

    snap = profiler.records_snapshot()
    assert len(snap) == 1
    assert snap[0]["name"] == "model.forward.total"
    assert snap[0]["dt_ms"] is not None


def test_flush_writes_jsonl_and_clears(profiler, tmp_path):
    profiler.bump_iteration()
    with profiler.region("alpha"):
        pass
    with profiler.region("beta", layer_idx=7):
        pass

    written = profiler.flush()
    assert written == 2
    assert profiler.records_snapshot() == []  # buffer drained

    with open(profiler.out_path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    assert [r["name"] for r in lines] == ["alpha", "beta"]
    assert lines[1]["layer_idx"] == 7


def test_reset_drops_buffer_and_file(profiler):
    profiler.bump_iteration()
    with profiler.region("alpha"):
        pass
    profiler.flush()
    assert os.path.exists(profiler.out_path)

    profiler.reset()
    assert not os.path.exists(profiler.out_path)
    assert profiler.current_iteration == 0
    assert profiler.records_snapshot() == []
