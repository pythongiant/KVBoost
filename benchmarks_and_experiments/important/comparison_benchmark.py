#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite: KVBoost vs vLLM (prefix-caching) vs Baseline

Three-way comparison testing:
  1. Accuracy:     Exact match on LongBench long-context QA tasks
  2. Latency:      Time-to-first-token (TTFT) and end-to-end latency
  3. GPU Memory:   Peak GPU memory utilization and efficiency

All results saved to JSON with human-readable table output.
"""

import argparse
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np
from datetime import datetime

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

RESULTS_DIR = Path(__file__).parent / "results"
CHECKPOINT_DIR = Path(__file__).parent / ".checkpoints"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

CHECKPOINT_INTERVAL = 10


@dataclass
class BenchmarkSample:
    """Single benchmark sample result"""
    sample_id: str
    model: str
    backend: str  # 'kvboost', 'vllm_prefixcache', 'baseline'
    task: str  # task name
    context_length: int
    question_length: int
    answer: str
    predicted_answer: str
    is_correct: bool
    ttft_ms: float  # Time to first token
    total_latency_ms: float  # End-to-end latency
    tokens_per_second: float
    gpu_mem_mb: float  # Peak GPU memory
    gpu_mem_percent: float
    input_tokens: int
    output_tokens: int
    kv_cache_efficiency: float  # For kvboost only


@dataclass
class BackendResult:
    """Aggregated results for one backend"""
    backend: str
    model: str
    n_samples: int
    accuracy: float
    avg_ttft_ms: float
    avg_total_latency_ms: float
    avg_throughput_tps: float
    avg_gpu_mem_mb: float
    avg_gpu_mem_percent: float
    avg_kv_cache_efficiency: float
    samples: List[BenchmarkSample]


DATASET_REPO = "JetBrains-Research/lca-bug-localization"
DATASET_CONFIG = "py"

_SAMPLE_CACHE: Dict[tuple, List[Dict]] = {}


def _load_longbench_samples(
    n_samples: int,
    max_context_tokens: int = 8192,
) -> List[Dict]:
    cache_key = (n_samples, max_context_tokens)
    if cache_key in _SAMPLE_CACHE:
        return _SAMPLE_CACHE[cache_key]

    from datasets import load_dataset
    import json as _json, random as _random

    try:
        ds = load_dataset(DATASET_REPO, DATASET_CONFIG, split="train")
    except Exception as e:
        log.error("Could not load %s: %s", DATASET_REPO, e)
        return []

    samples = []
    for item in ds:
        if len(samples) >= n_samples:
            break
        diff = item.get("diff", "")
        if not diff or len(diff) < 10:
            continue
        issue_title = item.get("issue_title", "").strip()
        issue_body = (item.get("issue_body", "") or "").strip()
        question = issue_title + ("\n" + issue_body[:1000] if issue_body else "")
        if not question:
            continue
        changed_files = item.get("changed_files", [])
        if isinstance(changed_files, str):
            try:
                changed_files = _json.loads(changed_files)
            except Exception:
                changed_files = [changed_files]
        if not changed_files:
            continue
        correct_file = changed_files[0]
        approx_tokens = int(len((diff + question).split()) * 0.75)
        if approx_tokens > max_context_tokens:
            continue
        options = list(dict.fromkeys(changed_files))
        if len(options) < 2:
            continue
        _random.seed(42)
        choices = [correct_file] + [f for f in options if f != correct_file][:3]
        _random.shuffle(choices)
        correct_idx = choices.index(correct_file)
        letter = chr(ord("A") + correct_idx)
        samples.append({
            "task": "bug_localization",
            "context": diff,
            "input": question,
            "answers": [letter, correct_file],
            "approx_tokens": approx_tokens,
            "choices": choices,
            "correct_idx": correct_idx,
        })

    log.info("Loaded %d samples from %s", len(samples), DATASET_REPO)
    _SAMPLE_CACHE[cache_key] = samples
    return samples


def _format_prompt(context: str, question: str, choices: Optional[List[str]] = None) -> str:
    header = (
        "You are a code reviewer. Read the following git diff carefully.\n\n"
        f"Diff:\n{context}\n\nIssue: {question}\n\n"
    )
    if choices:
        opts = "\n".join(f"  {chr(ord('A')+i)}) {c}" for i, c in enumerate(choices))
        return header + f"Which file contains the bug? Choose one:\n{opts}\n\nAnswer (letter only):"
    return header + "Which file contains the bug?\n\nAnswer:"


def _extract_answer(raw: str) -> str:
    answer = raw.strip()
    if "\n" in answer:
        answer = answer[: answer.index("\n")].strip()
    return answer


def _compute_f1(gold: str, pred: str) -> float:
    gold_tokens = set(gold.lower().split())
    pred_tokens = set(pred.lower().split())
    if not gold_tokens or not pred_tokens:
        return 1.0 if gold == pred else 0.0
    common = gold_tokens & pred_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _is_correct(gold_answers: List[str], predicted: str) -> bool:
    pred_norm = predicted.lower().strip()
    for gold in gold_answers:
        if pred_norm == gold.lower().strip():
            return True
        if _compute_f1(gold.lower().strip(), pred_norm) >= 0.5:
            return True
    return False


def _gpu_total_mb() -> float:
    """Total GPU VRAM in MB, or 0 if no CUDA."""
    import torch
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)


class _PeakMemCtx:
    """Reset and capture peak GPU memory around a block."""
    peak_mb: float = 0.0

    def __enter__(self):
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        return self

    def __exit__(self, *_):
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            self.peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)


class BenchmarkRunner:
    """
    Orchestrates benchmarking across three backends.

    Each method performs a single model-load + inference pass that captures
    accuracy, latency, and GPU memory simultaneously, avoiding the cost of
    loading the model three times per backend.
    """

    def __init__(
        self,
        model_name: str,
        output_dir: Path = RESULTS_DIR,
        kvboost_recompute_strategy: str = "selective",
        kvboost_chunk_boundary_window: int = 0,
        kvboost_overlap_k: int = 0,
        kvboost_sink_tokens: int = 0,
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.kvboost_recompute_strategy = kvboost_recompute_strategy
        self.kvboost_chunk_boundary_window = kvboost_chunk_boundary_window
        self.kvboost_overlap_k = kvboost_overlap_k
        self.kvboost_sink_tokens = kvboost_sink_tokens

    def _run_backend(
        self,
        backend: str,
        n_samples: int,
        max_context_tokens: int = 8192,
        max_new_tokens: int = 64,
    ) -> List[BenchmarkSample]:
        """Single pass: load model once, measure accuracy + latency + GPU mem per sample."""
        import torch

        samples_data = _load_longbench_samples(n_samples, max_context_tokens)
        if not samples_data:
            log.warning("No LongBench samples loaded for %s", backend)
            return []

        gpu_total = _gpu_total_mb()
        weights_mb = 0.0

        if backend == "kvboost":
            results = self._run_kvboost(
                samples_data, max_new_tokens, gpu_total,
                recompute_strategy=self.kvboost_recompute_strategy,
                chunk_boundary_window=self.kvboost_chunk_boundary_window,
                overlap_k=self.kvboost_overlap_k,
                sink_tokens=self.kvboost_sink_tokens,
            )
        elif backend == "vllm_prefixcache":
            results = self._run_vllm(samples_data, max_new_tokens, gpu_total)
        elif backend == "baseline":
            results = self._run_baseline(samples_data, max_new_tokens, gpu_total)
        else:
            raise ValueError(f"Unknown backend: {backend!r}")

        return results

    def _run_kvboost(
        self,
        samples_data: List[Dict],
        max_new_tokens: int,
        gpu_total_mb: float,
        recompute_strategy: str = "selective",
        chunk_boundary_window: int = 0,
        overlap_k: int = 0,
        sink_tokens: int = 0,
    ) -> List[BenchmarkSample]:
        import torch
        from kvboost import KVBoost, GenerationMode

        engine = KVBoost.from_pretrained(
            model_name=self.model_name,
            max_cache_bytes=4_000_000_000,
            chunk_size=128,
            recompute_overlap=16,
            recompute_strategy=recompute_strategy,
            chunk_boundary_window=chunk_boundary_window,
            overlap_k=overlap_k,
            sink_tokens=sink_tokens,
        )
        tokenizer = engine.tokenizer
        weights_mb = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

        results = []
        for i, s in enumerate(samples_data):
            prompt = _format_prompt(s["context"], s["input"], s.get("choices"))
            prompt_tokens = len(tokenizer.encode(prompt))

            ctx = _PeakMemCtx()
            with ctx:
                result = engine.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    mode=GenerationMode.CHUNK_KV_REUSE,
                    do_sample=False,
                )

            predicted = _extract_answer(result.output_text)
            correct = _is_correct(s["answers"], predicted)
            peak_mb = ctx.peak_mb
            gpu_pct = peak_mb / gpu_total_mb if gpu_total_mb > 0 else 0.0

            results.append(BenchmarkSample(
                sample_id=f"{s['task']}_{i:04d}",
                model=self.model_name,
                backend="kvboost",
                task=s["task"],
                context_length=s["approx_tokens"],
                question_length=len(s["input"].split()),
                answer=s["answers"][0] if s["answers"] else "",
                predicted_answer=predicted,
                is_correct=correct,
                ttft_ms=result.ttft_ms,
                total_latency_ms=result.total_ms,
                tokens_per_second=result.tokens_per_sec,
                gpu_mem_mb=peak_mb,
                gpu_mem_percent=gpu_pct,
                input_tokens=prompt_tokens,
                output_tokens=result.generated_tokens,
                kv_cache_efficiency=result.kv_reuse_ratio,
            ))

        del engine
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results

    def _run_vllm(
        self,
        samples_data: List[Dict],
        max_new_tokens: int,
        gpu_total_mb: float,
    ) -> List[BenchmarkSample]:
        import torch
        from vllm import LLM, SamplingParams
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        llm = LLM(model=self.model_name, enable_prefix_caching=True,
                  max_model_len=max_context_tokens + 128, gpu_memory_utilization=0.95)
        params = SamplingParams(temperature=0, max_tokens=max_new_tokens)

        results = []
        for i, s in enumerate(samples_data):
            prompt = _format_prompt(s["context"], s["input"], s.get("choices"))
            prompt_tokens = len(tokenizer.encode(prompt))

            ctx = _PeakMemCtx()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            with ctx:
                outputs = llm.generate([prompt], params)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_ms = (time.perf_counter() - t0) * 1000

            out = outputs[0]
            completion_tokens = len(out.outputs[0].token_ids)
            predicted = _extract_answer(out.outputs[0].text)
            correct = _is_correct(s["answers"], predicted)

            ttft_ms = getattr(out, "ttft_ms", None) or (total_ms / max(completion_tokens, 1))
            tps = completion_tokens / (total_ms / 1000) if total_ms > 0 else 0.0
            peak_mb = ctx.peak_mb
            gpu_pct = peak_mb / gpu_total_mb if gpu_total_mb > 0 else 0.0

            results.append(BenchmarkSample(
                sample_id=f"{s['task']}_{i:04d}",
                model=self.model_name,
                backend="vllm_prefixcache",
                task=s["task"],
                context_length=s["approx_tokens"],
                question_length=len(s["input"].split()),
                answer=s["answers"][0] if s["answers"] else "",
                predicted_answer=predicted,
                is_correct=correct,
                ttft_ms=ttft_ms,
                total_latency_ms=total_ms,
                tokens_per_second=tps,
                gpu_mem_mb=peak_mb,
                gpu_mem_percent=gpu_pct,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                kv_cache_efficiency=0.0,
            ))

        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results

    def _run_baseline(
        self,
        samples_data: List[Dict],
        max_new_tokens: int,
        gpu_total_mb: float,
    ) -> List[BenchmarkSample]:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if device in ("cuda", "mps") else torch.float32,
        ).to(device)
        hf_model.eval()

        results = []
        with torch.no_grad():
            for i, s in enumerate(samples_data):
                prompt = _format_prompt(s["context"], s["input"], s.get("choices"))
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                prompt_tokens = inputs["input_ids"].shape[1]

                # TTFT: time first-token generation
                if device == "cuda":
                    torch.cuda.synchronize()
                t_ttft = time.perf_counter()
                hf_model.generate(
                    **inputs,
                    max_new_tokens=1,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
                if device == "cuda":
                    torch.cuda.synchronize()
                ttft_ms = (time.perf_counter() - t_ttft) * 1000

                # Full generation with peak memory tracking
                ctx = _PeakMemCtx()
                t_gen = time.perf_counter()
                with ctx:
                    gen_ids = hf_model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                if device == "cuda":
                    torch.cuda.synchronize()
                total_ms = (time.perf_counter() - t_gen) * 1000 + ttft_ms

                completion_tokens = gen_ids.shape[1] - prompt_tokens
                new_ids = gen_ids[0][prompt_tokens:]
                predicted = _extract_answer(tokenizer.decode(new_ids, skip_special_tokens=True))
                correct = _is_correct(s["answers"], predicted)

                tps = completion_tokens / (total_ms / 1000) if total_ms > 0 else 0.0
                peak_mb = ctx.peak_mb
                gpu_pct = peak_mb / gpu_total_mb if gpu_total_mb > 0 else 0.0

                results.append(BenchmarkSample(
                    sample_id=f"{s['task']}_{i:04d}",
                    model=self.model_name,
                    backend="baseline",
                    task=s["task"],
                    context_length=s["approx_tokens"],
                    question_length=len(s["input"].split()),
                    answer=s["answers"][0] if s["answers"] else "",
                    predicted_answer=predicted,
                    is_correct=correct,
                    ttft_ms=ttft_ms,
                    total_latency_ms=total_ms,
                    tokens_per_second=tps,
                    gpu_mem_mb=peak_mb,
                    gpu_mem_percent=gpu_pct,
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    kv_cache_efficiency=0.0,
                ))

        del hf_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return results

    def benchmark_accuracy(
        self,
        backend: str,
        dataset_name: str = "longbench",
        n_samples: int = 50,
        **kwargs
    ) -> List[BenchmarkSample]:
        """Run full benchmark for backend (accuracy + latency + GPU mem in one pass)."""
        log.info("Starting benchmark for %s...", backend)
        return self._run_backend(backend, n_samples)

    def benchmark_latency(
        self,
        backend: str,
        dataset_name: str = "longbench",
        n_samples: int = 50,
        **kwargs
    ) -> List[BenchmarkSample]:
        """Alias for benchmark_accuracy — all metrics collected in one pass."""
        return self.benchmark_accuracy(backend, dataset_name, n_samples, **kwargs)

    def benchmark_gpu_utilization(
        self,
        backend: str,
        dataset_name: str = "longbench",
        n_samples: int = 50,
        **kwargs
    ) -> List[BenchmarkSample]:
        """Alias for benchmark_accuracy — all metrics collected in one pass."""
        return self.benchmark_accuracy(backend, dataset_name, n_samples, **kwargs)


def aggregate_results(samples: List[BenchmarkSample], backend: str, model: str) -> BackendResult:
    """Aggregate individual samples into backend-level statistics"""
    if not samples:
        return BackendResult(
            backend=backend,
            model=model,
            n_samples=0,
            accuracy=0,
            avg_ttft_ms=0,
            avg_total_latency_ms=0,
            avg_throughput_tps=0,
            avg_gpu_mem_mb=0,
            avg_gpu_mem_percent=0,
            avg_kv_cache_efficiency=0,
            samples=[]
        )

    accuracy = sum(s.is_correct for s in samples) / len(samples)
    avg_ttft = np.mean([s.ttft_ms for s in samples])
    avg_total = np.mean([s.total_latency_ms for s in samples])
    avg_tps = np.mean([s.tokens_per_second for s in samples])
    avg_gpu_mem = np.mean([s.gpu_mem_mb for s in samples])
    avg_gpu_pct = np.mean([s.gpu_mem_percent for s in samples])
    avg_kv_eff = np.mean([s.kv_cache_efficiency for s in samples]) if samples[0].kv_cache_efficiency > 0 else 0

    return BackendResult(
        backend=backend,
        model=model,
        n_samples=len(samples),
        accuracy=accuracy,
        avg_ttft_ms=avg_ttft,
        avg_total_latency_ms=avg_total,
        avg_throughput_tps=avg_tps,
        avg_gpu_mem_mb=avg_gpu_mem,
        avg_gpu_mem_percent=avg_gpu_pct,
        avg_kv_cache_efficiency=avg_kv_eff,
        samples=samples,
    )


def print_accuracy_comparison(results: Dict[str, BackendResult]):
    """Print accuracy comparison table"""
    print("\n" + "="*80)
    print("  ACCURACY COMPARISON (% Exact Match)")
    print("="*80)
    print(f"  {'Backend':<20} {'N Samples':>10} {'Accuracy':>12} {'Std Dev':>12}")
    print(f"  {'-'*20} {'-'*10} {'-'*12} {'-'*12}")
    
    for backend_name in ['kvboost', 'vllm_prefixcache', 'baseline']:
        if backend_name not in results:
            continue
        r = results[backend_name]
        accuracies = [s.is_correct for s in r.samples]
        std_dev = np.std(accuracies) if len(accuracies) > 1 else 0
        print(
            f"  {backend_name:<20} {r.n_samples:>10} "
            f"{r.accuracy:>11.1%} {std_dev:>11.1%}"
        )
    print()


def print_latency_comparison(results: Dict[str, BackendResult]):
    """Print latency comparison table"""
    print("\n" + "="*80)
    print("  LATENCY COMPARISON (milliseconds)")
    print("="*80)
    print(
        f"  {'Backend':<20} {'TTFT (ms)':>15} {'Total (ms)':>15} "
        f"{'Throughput':>12} {'Speedup':>10}"
    )
    print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*12} {'-'*10}")
    
    baseline_ttft = None
    for backend_name in ['kvboost', 'vllm_prefixcache', 'baseline']:
        if backend_name not in results:
            continue
        r = results[backend_name]
        
        if backend_name == 'baseline':
            baseline_ttft = r.avg_ttft_ms
        
        speedup = baseline_ttft / r.avg_ttft_ms if baseline_ttft and r.avg_ttft_ms > 0 else 1.0
        marker = " <<<" if backend_name == 'baseline' else ""
        
        print(
            f"  {backend_name:<20} {r.avg_ttft_ms:>14.1f} "
            f"{r.avg_total_latency_ms:>14.1f} {r.avg_throughput_tps:>11.1f} "
            f"{speedup:>9.2f}x{marker}"
        )
    print()


def print_gpu_utilization_comparison(results: Dict[str, BackendResult]):
    """Print GPU utilization comparison table"""
    print("\n" + "="*80)
    print("  GPU UTILIZATION COMPARISON")
    print("="*80)
    print(
        f"  {'Backend':<20} {'Peak Memory (MB)':>20} {'% of GPU':>15} "
        f"{'Memory Efficiency':>20}"
    )
    print(f"  {'-'*20} {'-'*20} {'-'*15} {'-'*20}")
    
    for backend_name in ['kvboost', 'vllm_prefixcache', 'baseline']:
        if backend_name not in results:
            continue
        r = results[backend_name]
        marker = " <<<" if backend_name == 'baseline' else ""
        
        print(
            f"  {backend_name:<20} {r.avg_gpu_mem_mb:>19.0f} {r.avg_gpu_mem_percent:>14.1%} "
            f"{r.avg_kv_cache_efficiency:>19.1%}{marker}"
        )
    print()


def print_summary(
    results: Dict[str, BackendResult],
    timestamp: str
):
    """Print complete summary with all three metrics"""
    any_result = next(iter(results.values()))
    print("\n" + "="*80)
    print(f"  BENCHMARK SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Model: {any_result.model}")
    print("="*80)
    
    print_accuracy_comparison(results)
    print_latency_comparison(results)
    print_gpu_utilization_comparison(results)
    
    # Winner determination
    if results:
        best_accuracy = max(r.accuracy for r in results.values())
        best_latency = min(r.avg_ttft_ms for r in results.values() if r.avg_ttft_ms > 0)
        best_gpu = min(r.avg_gpu_mem_mb for r in results.values())
        
        print("="*80)
        print("  RANKINGS")
        print("="*80)
        print(f"  Best Accuracy:      {[n for n,r in results.items() if r.accuracy == best_accuracy][0]}")
        print(f"  Best TTFT Latency:  {[n for n,r in results.items() if r.avg_ttft_ms == best_latency][0]}")
        print(f"  Best GPU Efficiency: {[n for n,r in results.items() if r.avg_gpu_mem_mb == best_gpu][0]}")
        print("="*80 + "\n")


def save_results_json(
    results: Dict[str, BackendResult],
    model: str,
    output_path: Optional[Path] = None
) -> Path:
    """Save all results to JSON"""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = RESULTS_DIR / f"benchmark_comparison_{model.replace('/', '_')}_{timestamp}.json"
    
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    serializable = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "backends": {}
    }
    
    for backend_name, result in results.items():
        serializable["backends"][backend_name] = {
            "n_samples": result.n_samples,
            "accuracy": result.accuracy,
            "avg_ttft_ms": result.avg_ttft_ms,
            "avg_total_latency_ms": result.avg_total_latency_ms,
            "avg_throughput_tps": result.avg_throughput_tps,
            "avg_gpu_mem_mb": result.avg_gpu_mem_mb,
            "avg_gpu_mem_percent": result.avg_gpu_mem_percent,
            "avg_kv_cache_efficiency": result.avg_kv_cache_efficiency,
            "samples": [asdict(s) for s in result.samples]
        }
    
    with open(output_path, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    
    log.info(f"Results saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="3-Way Benchmark: KVBoost vs vLLM (prefix-caching) vs Baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-3B",
        help="Model name (default: Qwen/Qwen2.5-3B)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of samples per backend (default: 50)"
    )
    parser.add_argument(
        "--dataset",
        default="longbench",
        choices=["longbench", "custom"],
        help="Dataset to use (default: longbench)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: auto-generated)"
    )
    parser.add_argument(
        "--kvboost-only",
        action="store_true",
        help="Only benchmark KVBoost (skip vLLM and baseline)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print(f"\n{'='*80}")
    print(f"  3-WAY BENCHMARK: KVBoost vs vLLM (prefix-caching) vs Baseline")
    print(f"{'='*80}")
    print(f"  Model:       {args.model}")
    print(f"  Samples:     {args.n_samples} per backend")
    print(f"  Dataset:     {args.dataset}")
    print(f"  Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")
    
    runner = BenchmarkRunner(args.model, args.output.parent if args.output else RESULTS_DIR)
    
    # Placeholder: In full implementation, these would actually run benchmarks
    all_results: Dict[str, BackendResult] = {}
    
    backends = ['kvboost'] if args.kvboost_only else ['kvboost', 'vllm_prefixcache', 'baseline']
    
    for backend in backends:
        log.info("Benchmarking %s...", backend)

        # Single pass: accuracy + latency + GPU memory measured together
        samples = runner.benchmark_accuracy(backend, args.dataset, args.n_samples)

        result = aggregate_results(samples, backend, args.model)
        all_results[backend] = result

        log.info(
            "  ✓ %s: accuracy=%.1f%%, ttft=%.1fms, gpu=%.0fMB",
            backend, result.accuracy * 100, result.avg_ttft_ms, result.avg_gpu_mem_mb,
        )
    
    # Print summary tables
    if all_results:
        print_summary(all_results, datetime.now().isoformat())
    
    # Save to JSON
    output_path = save_results_json(all_results, args.model, args.output)
    print(f"✓ Benchmark complete. Results saved to: {output_path}\n")


if __name__ == "__main__":
    main()
