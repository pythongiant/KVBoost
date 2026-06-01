"""Real coding prompts from a HuggingFace dataset. No synthetic content.

Builds two prompt sets from a real code corpus:

  * REUSE set — a coding-agent pattern: a shared *repo context* (several
    real source files / functions concatenated) as the prompt prefix, then
    a varying real coding task. Replayed sequentially, the shared prefix's
    KV is reused across requests (kvboost chunk-reuse + CacheBlend; vLLM
    prefix caching), so TTFT drops after the first request — this is the
    "faster TTFT" measurement.

  * OOM set — the same real corpus concatenated to increasing target token
    counts, to probe the memory ceiling (kvboost survives via chunked
    prefill / per-request kv-bits; vLLM OOMs/crashes past its budget).

Requires ``pip install datasets`` and downloads a real dataset — there is
no synthetic fallback by design. Default ``openai_humaneval`` (small, no
auth, real Python). For long-context coding agents, point ``--dataset`` at
a repo-level set (e.g. ``repobench``) — the adapter pulls code text from
common field names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Prompt:
    name: str
    system: str
    user: str
    max_tokens: int
    target_tokens: int  # approximate; server reports exact prompt_tokens

    def to_body(self, model: str, stream: bool = True) -> dict:
        return {
            "model": model,
            "messages": [
                {"role": "system", "content": self.system},
                {"role": "user", "content": self.user},
            ],
            "max_tokens": self.max_tokens,
            "temperature": 0.2,
            "stream": stream,
            # request usage on the final stream chunk → input/decode throughput
            "stream_options": {"include_usage": True},
        }


_SYS = ("You are a senior software engineer working in the codebase below. "
        "Answer using the provided code; write correct, idiomatic code with "
        "type hints.")

# Field names we try, in order, to extract real code text from common
# coding datasets (HumanEval, MBPP, RepoBench, The Stack, CodeSearchNet, …).
_TEXT_FIELDS = ("text", "content", "func_code_string", "whole_func_string",
                "code", "canonical_solution", "prompt", "problem_statement")


def _extract_code(row: dict) -> str:
    parts = []
    # HumanEval: prompt (signature+docstring) + canonical_solution (body).
    if "prompt" in row and "canonical_solution" in row:
        return (row["prompt"] or "") + (row["canonical_solution"] or "")
    # MBPP: text (NL spec) + code.
    if "text" in row and "code" in row:
        return f"# {row['text']}\n{row['code']}"
    for f in _TEXT_FIELDS:
        v = row.get(f)
        if isinstance(v, str) and v.strip():
            parts.append(v)
            break
    return "\n".join(parts)


def load_corpus(dataset: str, *, split: str, n_units: int) -> List[str]:
    """Return up to ``n_units`` real code strings from ``dataset``.

    Raises SystemExit with an install hint if ``datasets`` is missing — there
    is intentionally no synthetic fallback.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit(
            "ERROR: this benchmark uses a real HuggingFace dataset.\n"
            "Run: pip install datasets"
        )
    try:
        ds = load_dataset(dataset, split=split)
    except Exception as e:
        raise SystemExit(
            f"ERROR: could not load dataset '{dataset}' (split={split}): {e}\n"
            "Pick another with --dataset (e.g. openai_humaneval, mbpp) or "
            "check network / HF auth."
        )
    units: List[str] = []
    for row in ds:
        code = _extract_code(dict(row))
        if code and len(code) > 80:   # skip trivially short rows
            units.append(code.strip())
        if len(units) >= n_units:
            break
    if not units:
        raise SystemExit(
            f"ERROR: dataset '{dataset}' yielded no usable code text. "
            "Try --dataset openai_humaneval."
        )
    return units


def _repo_context(units: List[str], n_files: int) -> str:
    chosen = units[:n_files]
    return "\n\n".join(
        f"# ── file {i+1} ──\n{u}" for i, u in enumerate(chosen)
    )


_TASK = (
    "\n\n# ── TASK ──\n"
    "Add complete, typed docstrings and inline type hints to every function "
    "and class in `file {0}` above, preserving behavior. Return only the "
    "rewritten file {0}."
)


def reuse_prompts(
    corpus: List[str], *, n: int, n_files: int = 6, max_tokens: int = 256,
) -> List[Prompt]:
    """Coding-agent reuse workload: shared real repo-context prefix, varying
    real task. Sequential replay → prefix KV reused across requests."""
    ctx = _repo_context(corpus, n_files)
    target = len(ctx) // 4
    out = []
    for i in range(n):
        file_no = (i % n_files) + 1   # vary which file the task targets
        out.append(Prompt(
            name=f"reuse-{i}",
            system=_SYS,
            user=ctx + _TASK.format(file_no),
            max_tokens=max_tokens,
            target_tokens=target,
        ))
    return out


def oom_prompts(
    corpus: List[str], *, contexts: List[int], max_tokens: int = 128,
) -> List[Prompt]:
    """One prompt per target token size, built by concatenating real code
    until the size is reached. Ascending — for the OOM ramp."""
    # ~4 chars/token. Repeat the corpus (real code) to fill the target.
    joined = "\n\n".join(corpus)
    out = []
    for tgt in contexts:
        need_chars = tgt * 4
        reps = max(1, need_chars // max(len(joined), 1) + 1)
        body = ("\n\n".join([joined] * reps))[:need_chars]
        user = (
            f"# ── repository snapshot ──\n{body}\n\n"
            "# ── TASK ──\nList the three most important functions in this "
            "codebase and explain what each does in one sentence."
        )
        out.append(Prompt(
            name=f"oom-{tgt//1000}k" if tgt >= 1000 else f"oom-{tgt}",
            system=_SYS, user=user, max_tokens=max_tokens, target_tokens=tgt,
        ))
    return out
