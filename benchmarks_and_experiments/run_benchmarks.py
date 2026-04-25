#!/usr/bin/env python3
"""
Main Benchmark Runner

Executes all three benchmarks (accuracy, latency, GPU memory) and produces:
1. Console output with formatted tables
2. JSON files with detailed results
3. Unified report comparing all three backends
"""

import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import sys

# Import benchmark modules
try:
    from accuracy_benchmark import benchmark_accuracy, aggregate_accuracy_results, print_accuracy_table, save_accuracy_results
except ImportError:
    benchmark_accuracy = None

try:
    from latency_benchmark import benchmark_latency, aggregate_latency_results, print_latency_table, save_latency_results
except ImportError:
    benchmark_latency = None

try:
    from memory_benchmark import benchmark_gpu_memory, aggregate_gpu_memory_results, print_gpu_memory_table, print_memory_breakdown, save_gpu_memory_results
except ImportError:
    benchmark_gpu_memory = None

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


def run_all_benchmarks(
    model: str,
    n_samples: int,
    output_dir: Path = RESULTS_DIR,
    backends: list = None,
) -> Dict[str, Any]:
    """Run all three benchmarks"""
    
    if backends is None:
        backends = ['kvboost', 'vllm_prefixcache', 'baseline']
    
    print(f"\n{'='*100}")
    print(f"  COMPREHENSIVE 3-WAY BENCHMARK SUITE")
    print(f"  KVBoost vs vLLM (prefix-caching) vs Baseline")
    print(f"{'='*100}")
    print(f"  Model:          {model}")
    print(f"  Samples/backend: {n_samples}")
    print(f"  Backends:        {', '.join(backends)}")
    print(f"  Timestamp:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*100}\n")
    
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "n_samples": n_samples,
        "backends": backends,
    }
    
    # === ACCURACY BENCHMARK ===
    if benchmark_accuracy:
        print("\n► RUNNING ACCURACY BENCHMARK...")
        print(f"  Testing exact-match accuracy on LongBench tasks\n")
        
        accuracy_results = {}
        for backend in backends:
            print(f"  [{backend}] Running...")
            accuracy_results[backend] = benchmark_accuracy(backend, model, n_samples)
        
        accuracy_agg = aggregate_accuracy_results(accuracy_results)
        print_accuracy_table(accuracy_agg)
        
        accuracy_output = save_accuracy_results(accuracy_agg, model)
        all_results["accuracy"] = accuracy_agg
        log.info(f"  ✓ Accuracy benchmark complete: {accuracy_output}")
    else:
        log.warning("  ⚠ Accuracy benchmark module not available")
    
    # === LATENCY BENCHMARK ===
    if benchmark_latency:
        print("\n► RUNNING LATENCY BENCHMARK...")
        print(f"  Measuring TTFT and throughput\n")
        
        latency_results = {}
        for backend in backends:
            print(f"  [{backend}] Running...")
            vllm_pc = True if backend == 'vllm_prefixcache' else False
            latency_results[backend] = benchmark_latency(backend, model, n_samples, vllm_prefix_caching=vllm_pc)
        
        latency_agg = aggregate_latency_results(latency_results)
        print_latency_table(latency_agg)
        
        latency_output = save_latency_results(latency_agg, model)
        all_results["latency"] = latency_agg
        log.info(f"  ✓ Latency benchmark complete: {latency_output}")
    else:
        log.warning("  ⚠ Latency benchmark module not available")
    
    # === GPU MEMORY BENCHMARK ===
    if benchmark_gpu_memory:
        print("\n► RUNNING GPU MEMORY BENCHMARK...")
        print(f"  Measuring peak GPU memory and cache efficiency\n")
        
        memory_results = {}
        for backend in backends:
            print(f"  [{backend}] Running...")
            memory_results[backend] = benchmark_gpu_memory(backend, model, n_samples)
        
        memory_agg = aggregate_gpu_memory_results(memory_results)
        print_gpu_memory_table(memory_agg)
        print_memory_breakdown(memory_agg)
        
        memory_output = save_gpu_memory_results(memory_agg, model)
        all_results["gpu_memory"] = memory_agg
        log.info(f"  ✓ GPU memory benchmark complete: {memory_output}")
    else:
        log.warning("  ⚠ GPU memory benchmark module not available")
    
    return all_results


def save_unified_report(all_results: Dict[str, Any], output_path: Path = None) -> Path:
    """Save unified report with all results"""
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model = all_results.get("model", "unknown").replace("/", "_")
        output_path = RESULTS_DIR / f"unified_benchmark_report_{model}_{timestamp}.json"
    
    output_path.parent.mkdir(exist_ok=True, parents=True)
    
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    log.info(f"Unified report saved to {output_path}")
    return output_path


def print_final_summary(all_results: Dict[str, Any]):
    """Print final summary and rankings"""
    print(f"\n{'='*100}")
    print(f"  BENCHMARK SUMMARY & RANKINGS")
    print(f"{'='*100}\n")
    
    if "accuracy" in all_results:
        print("  ACCURACY RANKINGS:")
        acc_results = all_results["accuracy"]
        for backend, data in sorted(acc_results.items(), key=lambda x: x[1]["exact_match_accuracy"], reverse=True):
            if isinstance(data, dict) and "exact_match_accuracy" in data:
                print(f"    {backend:20} {data['exact_match_accuracy']:>6.1%} exact match")
    
    if "latency" in all_results:
        print("\n  LATENCY RANKINGS (lower TTFT is better):")
        lat_results = all_results["latency"]
        for backend, data in sorted(lat_results.items(), key=lambda x: x[1]["ttft_ms"]["mean"]):
            if isinstance(data, dict) and "ttft_ms" in data:
                print(f"    {backend:20} {data['ttft_ms']['mean']:>7.1f}ms avg TTFT")
    
    if "gpu_memory" in all_results:
        print("\n  GPU MEMORY RANKINGS (lower peak memory is better):")
        mem_results = all_results["gpu_memory"]
        for backend, data in sorted(mem_results.items(), key=lambda x: x[1]["gpu_memory_mb"]["mean"]):
            if isinstance(data, dict) and "gpu_memory_mb" in data:
                print(f"    {backend:20} {data['gpu_memory_mb']['mean']:>7.1f}MB peak GPU memory")
    
    print(f"\n{'='*100}\n")


def main():
    parser = argparse.ArgumentParser(
        description="3-Way Benchmark: KVBoost vs vLLM (prefix-caching) vs Baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full benchmark with all backends
  python run_benchmarks.py --model Qwen/Qwen2.5-3B --n-samples 50
  
  # KVBoost only
  python run_benchmarks.py --model Qwen/Qwen2.5-3B --backends kvboost --n-samples 100
  
  # Compare vLLM with baseline
  python run_benchmarks.py --backends vllm_prefixcache baseline --n-samples 50
        """
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
        help="Number of samples per benchmark (default: 50)"
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        default=None,
        choices=["kvboost", "vllm_prefixcache", "baseline"],
        help="Backends to benchmark (default: all)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Output directory (default: {RESULTS_DIR})"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run benchmarks
    all_results = run_all_benchmarks(
        model=args.model,
        n_samples=args.n_samples,
        output_dir=args.output_dir,
        backends=args.backends or ['kvboost', 'vllm_prefixcache', 'baseline'],
    )
    
    # Save unified report
    report_path = save_unified_report(all_results, output_path=args.output_dir / f"unified_benchmark_report_{args.model.replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    # Print final summary
    print_final_summary(all_results)
    
    print(f"✓ All benchmarks complete!")
    print(f"  Reports saved to: {args.output_dir}/\n")


if __name__ == "__main__":
    main()
