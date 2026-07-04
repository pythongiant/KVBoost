#!/usr/bin/env python3
"""
Decode-backend A/B: is an fp16 megakernel (MPK) worth adopting on this GPU?

This settles the ONE question before anyone writes an MPKDecoder: on a
bandwidth-bound GPU (RTX 3060 / sm_86), does a *fp16* Mirage Persistent Kernel
beat kvboost's existing *int4-Marlin* decode? MPK is fp16/bf16-only (no int4),
so adopting it means forfeiting weight quant — the biggest decode-bandwidth
lever on this box. The bar MPK must clear is therefore the int4 number, not the
fp16 one.

What it measures: steady-state single-token *decode* latency (ms/token) at
batch=1, greedy, across kvboost's real decode backends:

    fp16-eager        plain fp16 model, eager autoregressive loop
    fp16-cudagraph    fp16 + CUDAGraphDecoder (torch.compile reduce-overhead)
    int4-eager        AWQ/GPTQ checkpoint -> Marlin int4 GEMM, eager
    int4-cudagraph    int4 + CUDAGraphDecoder (best-effort; may self-disable)

It drives the *actual* engine.generate() path (not a reimplementation), isolates
decode from prefill via per-token timestamps, and reports the median inter-token
gap (robust to a warmup/GC outlier).

MPK arm: this script does NOT reimplement MPK's frontend. Run MPK's own shipped
demo on the box to get its ms/token, then pass it via --mpk-ms-per-token to fold
it into the verdict. If `import mirage` fails or MPK won't build on sm_86, that
failure IS the answer (question moot -> stay on int4).

    # kvboost baselines only:
    python decode_backend_benchmark.py \
        --fp16-model Qwen/Qwen2.5-3B-Instruct \
        --awq-model  Qwen/Qwen2.5-3B-Instruct-AWQ

    # fold in MPK's number from its demo:
    python decode_backend_benchmark.py --awq-model ...-AWQ --fp16-model ... \
        --mpk-ms-per-token 11.8

Run this ON THE 3060 BOX (needs CUDA). It is meaningless on a CUDA-less dev box.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

log = logging.getLogger("decode_bench")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# A neutral filler paragraph; repeated/truncated to hit the target prompt length.
_FILLER = (
    "In a distributed inference system the decode phase is memory-bandwidth "
    "bound: each token streams the full weight matrix from HBM, so the wall "
    "time per step is dominated by bytes moved, not by arithmetic. "
)


@dataclass
class DecodeResult:
    label: str
    model: str
    cuda_graph_flag: bool          # was cuda_graph_decode requested
    cgd_active: bool               # did CUDAGraphDecoder.applicable() hold
    cgd_disabled_after: bool       # did the graph path self-disable (self-check/graph-break)
    prompt_tokens: int
    gen_tokens: int
    ttft_ms: float                 # prefill + first token
    decode_ms_per_token: float     # median steady-state inter-token gap
    decode_tokens_per_s: float
    error: Optional[str] = None


def _build_prompt(tokenizer, target_tokens: int) -> str:
    ids = tokenizer.encode(_FILLER)
    if not ids:
        return _FILLER * 8
    reps = target_tokens // len(ids) + 1
    ids = (ids * reps)[:target_tokens]
    return tokenizer.decode(ids)


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def _measure_one(label: str, model_name: str, cuda_graph: bool,
                 *, attn: str, max_cache_bytes: int, prompt_tokens: int,
                 gen_tokens: int) -> DecodeResult:
    """Load one engine config, warm it up, then time steady-state decode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kvboost.engine import InferenceEngine

    log.info("=== %s: loading %s (cuda_graph_decode=%s) ===",
             label, model_name, cuda_graph)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Explicit-dict device_map, mirroring server/__main__.py's standard load
    # path: the bare-string form can leave AWQ-Marlin buffers on `meta`.
    target = device if (":" in device or device in ("cpu", "mps")) else f"{device}:0"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # DO NOT use InferenceEngine.from_pretrained here: it loads on CPU with no
    # device_map, and __init__ skips .to(device) for quantized models (they
    # can't be moved post-load) — so an AWQ checkpoint stays on CPU and the
    # forward dies with a device mismatch (embed_tokens on cpu, ids on cuda).
    # Replicate the server's real load: device_map={"": target} +
    # low_cpu_mem_usage=False places AWQ/GPTQ-Marlin weights on the GPU at load
    # time. transformers auto-detects a pre-quantized checkpoint's
    # quantization_config; a plain fp16 checkpoint loads the same way.
    want_fa2 = attn in ("auto", "flash_attention_2")
    load_kwargs = dict(
        dtype=torch.float16,
        low_cpu_mem_usage=False,
        device_map={"": target},
        attn_implementation=("flash_attention_2" if want_fa2 else attn),
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    except Exception as e:
        if not want_fa2:
            raise
        log.info("flash_attention_2 unavailable (%s); using sdpa", e)
        load_kwargs["attn_implementation"] = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.eval()

    engine = InferenceEngine(
        model=model,
        tokenizer=tokenizer,
        max_cache_bytes=max_cache_bytes,
        device=device,
        cuda_graph_decode=cuda_graph,
    )
    try:
        prompt = _build_prompt(tokenizer, prompt_tokens)
        p_tokens = len(tokenizer.encode(prompt))
        cgd_active = bool(getattr(engine, "_cgd", None) is not None
                          and engine._cgd.applicable())

        # Warmup: triggers compile / self-check / CUDA-graph capture + context.
        engine.generate(prompt, max_new_tokens=8, do_sample=False)
        _free()

        stamps: List[float] = []

        def _cb(_tok_id: int) -> None:
            stamps.append(time.perf_counter())

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        engine.generate(prompt, max_new_tokens=gen_tokens,
                        do_sample=False, on_token=_cb)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        if len(stamps) < 3:
            raise RuntimeError(
                f"only {len(stamps)} tokens emitted; need >=3 for a stable "
                "median (raise --max-new-tokens or check the model emitted EOS "
                "early)")

        ttft_ms = (stamps[0] - t0) * 1000.0
        gaps_ms = np.diff(stamps) * 1000.0
        # Drop the first inter-token gap: it can carry a one-off transient
        # (first captured graph replay, allocator warmup). Median of the rest
        # is the steady-state decode cost.
        steady = gaps_ms[1:] if len(gaps_ms) > 1 else gaps_ms
        decode_ms = float(np.median(steady))

        cgd_disabled = bool(getattr(engine, "_cgd", None) is not None
                            and getattr(engine._cgd, "_disabled", False))
        res = DecodeResult(
            label=label, model=model_name, cuda_graph_flag=cuda_graph,
            cgd_active=cgd_active, cgd_disabled_after=cgd_disabled,
            prompt_tokens=p_tokens, gen_tokens=len(stamps),
            ttft_ms=round(ttft_ms, 2),
            decode_ms_per_token=round(decode_ms, 3),
            decode_tokens_per_s=round(1000.0 / decode_ms, 2),
        )
        log.info("%-16s  %.3f ms/tok  (%.1f tok/s)  ttft=%.1f ms  "
                 "cgd_active=%s disabled_after=%s",
                 label, res.decode_ms_per_token, res.decode_tokens_per_s,
                 res.ttft_ms, cgd_active, cgd_disabled)
        return res
    finally:
        del engine
        _free()


def _mpk_status() -> str:
    try:
        import mirage  # noqa: F401
        ver = getattr(mirage, "__version__", "unknown")
        return f"installed (mirage {ver})"
    except Exception as e:
        return f"NOT installed ({type(e).__name__}: {e})"


def _verdict(results: List[DecodeResult], mpk_ms: Optional[float]) -> List[str]:
    ok = [r for r in results if r.error is None]
    lines: List[str] = []
    fp16 = [r for r in ok if r.label.startswith("fp16")]
    int4 = [r for r in ok if r.label.startswith("int4")]
    best_fp16 = min((r.decode_ms_per_token for r in fp16), default=None)
    best_int4 = min((r.decode_ms_per_token for r in int4), default=None)

    if best_fp16 and best_int4:
        lines.append(f"int4-Marlin lever: {best_fp16:.2f} -> {best_int4:.2f} "
                     f"ms/tok ({best_fp16 / best_int4:.2f}x faster than fp16).")
    for tag, rs in (("fp16", fp16), ("int4", int4)):
        eager = next((r for r in rs if r.label.endswith("eager")), None)
        cg = next((r for r in rs if r.label.endswith("cudagraph")), None)
        if eager and cg:
            spd = eager.decode_ms_per_token / cg.decode_ms_per_token
            note = " (graph self-disabled)" if cg.cgd_disabled_after else ""
            lines.append(f"{tag} cudagraph vs eager: {spd:.2f}x{note}.")

    if mpk_ms is None:
        lines.append("MPK arm: no --mpk-ms-per-token given. Run MPK's demo on "
                     "this box and re-run with the number to get the verdict.")
        return lines

    lines.append("")
    lines.append(f"MPK (fp16 megakernel, external): {mpk_ms:.2f} ms/tok.")
    if best_fp16:
        lines.append(f"  vs your best fp16 ({best_fp16:.2f}): "
                     f"{best_fp16 / mpk_ms:.2f}x — the apples-to-apples number.")
    if best_int4:
        worth = mpk_ms < best_int4
        lines.append(f"  vs your best int4 ({best_int4:.2f}): "
                     f"{best_int4 / mpk_ms:.2f}x — THE DECISION BAR.")
        lines.append("")
        if worth:
            lines.append("VERDICT: MPK beats int4 even at fp16 -> worth "
                         "prototyping an MPKDecoder. (Then ask whether int4 "
                         "support could close the remaining gap.)")
        else:
            lines.append("VERDICT: MPK is fp16-only and does NOT beat int4 -> "
                         "NOT worth adopting on this GPU. Stay on int4-Marlin "
                         "(+cudagraph). Revisit only if MPK gains int4 support.")
    return lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fp16-model", default=None,
                    help="HF id/path of the plain fp16 checkpoint")
    ap.add_argument("--awq-model", default=None,
                    help="HF id/path of the AWQ/GPTQ checkpoint (int4 Marlin)")
    ap.add_argument("--mpk-ms-per-token", type=float, default=None,
                    help="MPK's decode ms/token from its own demo (fp16)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--prompt-tokens", type=int, default=512)
    ap.add_argument("--attn", default="sdpa",
                    help="attn impl for prefill (sdpa is safe/no-build; decode "
                         "runs sdpa regardless)")
    ap.add_argument("--max-cache-bytes", type=int, default=3 * 1024 ** 3)
    ap.add_argument("--allow-cpu", action="store_true",
                    help="run even without CUDA (numbers are meaningless)")
    args = ap.parse_args()

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit("No CUDA device. Run this on the 3060 box, or pass "
                         "--allow-cpu to smoke-test the harness (meaningless "
                         "timings).")
    if not args.fp16_model and not args.awq_model:
        raise SystemExit("Give at least one of --fp16-model / --awq-model.")

    log.info("GPU: %s | MPK: %s",
             torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
             _mpk_status())

    configs = []
    if args.fp16_model:
        configs += [("fp16-eager", args.fp16_model, False),
                    ("fp16-cudagraph", args.fp16_model, True)]
    if args.awq_model:
        configs += [("int4-eager", args.awq_model, False),
                    ("int4-cudagraph", args.awq_model, True)]

    results: List[DecodeResult] = []
    for label, model_name, cg in configs:
        try:
            results.append(_measure_one(
                label, model_name, cg, attn=args.attn,
                max_cache_bytes=args.max_cache_bytes,
                prompt_tokens=args.prompt_tokens, gen_tokens=args.max_new_tokens))
        except Exception as e:  # one config's OOM/error must not kill the run
            log.exception("%s FAILED: %s", label, e)
            results.append(DecodeResult(
                label=label, model=model_name, cuda_graph_flag=cg,
                cgd_active=False, cgd_disabled_after=False, prompt_tokens=0,
                gen_tokens=0, ttft_ms=0.0, decode_ms_per_token=float("nan"),
                decode_tokens_per_s=0.0, error=f"{type(e).__name__}: {e}"))
        _free()

    # ---- report ----
    print("\n" + "=" * 74)
    print(f"{'backend':<16}{'ms/token':>12}{'tok/s':>10}{'ttft ms':>12}"
          f"{'prompt tok':>12}")
    print("-" * 74)
    for r in results:
        if r.error:
            print(f"{r.label:<16}{'ERROR':>12}   {r.error[:40]}")
        else:
            print(f"{r.label:<16}{r.decode_ms_per_token:>12.3f}"
                  f"{r.decode_tokens_per_s:>10.1f}{r.ttft_ms:>12.1f}"
                  f"{r.prompt_tokens:>12}")
    print("=" * 74 + "\n")

    for line in _verdict(results, args.mpk_ms_per_token):
        print(line)
    print()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"decode_backend_{stamp}.json"
    payload = {
        "timestamp": stamp,
        "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else "cpu"),
        "mpk_status": _mpk_status(),
        "mpk_ms_per_token": args.mpk_ms_per_token,
        "args": vars(args),
        "results": [asdict(r) for r in results],
    }
    out.write_text(json.dumps(payload, indent=2))
    log.info("Wrote %s", out)


if __name__ == "__main__":
    main()
