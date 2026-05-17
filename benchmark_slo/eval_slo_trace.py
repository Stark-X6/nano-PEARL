#!/usr/bin/env python3
"""
SLO Trace Benchmark — Mode B: Poisson/Trace arrival, per-SLO metrics.

Emits requests over time (Poisson process or from trace file).
Collects per-SLO-group latency and attainment metrics.

Usage:
    python eval_slo_trace.py \
        --draft-model /path/to/draft \
        --target-model /path/to/target \
        --target-tp 3 \
        --rps 2.0 \
        --num-requests 50
"""

import argparse
import json
import os
import random
import sys
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_slo.generate_trace import generate_trace
from benchmark_slo.run_workload import run_workload


LEGACY_WARNING = (
    "Legacy benchmark entrypoint: eval_slo_trace.py is kept only for compatibility. "
    "Formal experiments should use exps/test.sh and benchmark_slo/run_workload.py."
)


def parse_args():
    parser = argparse.ArgumentParser(description='SLO Trace Benchmark - Mode B')

    # Model paths
    parser.add_argument('--draft-model', '-d', type=str, required=True)
    parser.add_argument('--target-model', '-t', type=str, required=True)

    # GPU layout
    parser.add_argument('--draft-tp', type=int, default=1)
    parser.add_argument('--target-tp', type=int, default=3)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9)

    # Generation
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max-tokens', type=int, default=128)

    # Emission
    parser.add_argument('--rps', type=float, default=2.0,
                        help='Requests per second (Poisson mode)')
    parser.add_argument('--num-requests', type=int, default=50)
    parser.add_argument('--input-len', type=int, default=1024)
    parser.add_argument('--trace', type=str, default=None,
                        help='Trace JSON file (overrides --rps)')

    # SLO
    parser.add_argument('--max-gamma', type=int, default=16)
    parser.add_argument('--min-gamma', type=int, default=1)
    parser.add_argument('--baseline-latency', type=float, default=-1.0)
    parser.add_argument('--correction-factor', type=float, default=1.0)

    # Other
    parser.add_argument('--seed', type=int, default=0)

    return parser.parse_args()


def generate_random_inputs(num, input_len, seed=0):
    rng = random.Random(seed)
    return [[rng.randint(0, 10000) for _ in range(input_len)] for _ in range(num)]


def run_trace_benchmark(slo_config, inputs, sampling_params, emission, slo_ratios_dist):
    """Run SLO benchmark with time-based request emission."""
    engine = SLOPearlEngine(slo_config)

    results = []
    slo_ratios_used = []

    for idx, inp in enumerate(inputs):
        slo_ratio = emission.sample_slo_ratio()
        slo_ratios_used.append(slo_ratio)

        emission.wait_until_next_request()
        arrival_time = time.time()

        engine.add_request(inp, copy.deepcopy(sampling_params), slo_ratio=slo_ratio)

        # Generate immediately (Step 1: single batch mode)
        output_text, num_tokens, num_acc_tokens, elapsed, slo_metrics = (
            engine.slo_generate()
        )
        finish_time = time.time()

        results.append({
            'idx': idx,
            'slo_ratio': slo_ratio,
            'num_tokens': num_tokens[0] if num_tokens else 0,
            'elapsed': elapsed,
            'arrival_time': arrival_time,
            'finish_time': finish_time,
            'latency_ms': (finish_time - arrival_time) * 1000,
            'per_token_latency_ms': ((finish_time - arrival_time) * 1000 / max(num_tokens[0], 1)) if num_tokens else 0,
        })

        logger.info(
            f"[{idx}/{len(inputs)}] slo={slo_ratio:.1f}, "
            f"tokens={results[-1]['num_tokens']}, "
            f"latency={results[-1]['latency_ms']:.0f}ms"
        )

    engine.exit()

    # Compute per-SLO-group metrics
    slo_groups = {}
    for ratio, _ in slo_ratios_dist:
        slo_groups[ratio] = {
            'count': 0,
            'total_tokens': 0,
            'total_latency_ms': 0,
            'total_per_token_ms': 0,
            'attained': 0,
        }

    baseline_ms = slo_config.baseline_latency_ms
    for r in results:
        nearest = min(slo_groups.keys(), key=lambda x: abs(x - r['slo_ratio']))
        g = slo_groups[nearest]
        g['count'] += 1
        g['total_tokens'] += r['num_tokens']
        g['total_latency_ms'] += r['latency_ms']
        g['total_per_token_ms'] += r['per_token_latency_ms']
        if baseline_ms > 0:
            constraint = nearest * baseline_ms
            if r['per_token_latency_ms'] <= constraint:
                g['attained'] += 1

    return results, slo_groups


def print_trace_report(results, slo_groups, slo_ratios_dist):
    """Print trace benchmark report."""
    total_toks = sum(r['num_tokens'] for r in results)
    total_time = sum(r['elapsed'] for r in results)

    print("\n" + "=" * 60)
    print("SLO Trace Benchmark Report - Mode B")
    print("=" * 60)
    print(f"Total requests: {len(results)}")
    print(f"Total tokens: {total_toks}")
    print(f"Total time: {total_time:.2f}s")
    print(f"Throughput: {total_toks / total_time:.1f} tok/s" if total_time > 0 else "N/A")

    print()
    print(f"{'SLO Group':>15} {'Count':>7} {'Avg Latency':>12} {'Avg tok/ms':>12} {'Attainment':>12}")
    print("-" * 60)

    for ratio, _ in slo_ratios_dist:
        g = slo_groups[ratio]
        if g['count'] == 0:
            continue
        avg_lat = g['total_latency_ms'] / g['count']
        avg_pt = g['total_per_token_ms'] / g['count']
        attainment = g['attained'] / g['count'] * 100

        label = f"slo={ratio}"
        print(f"{label:>15} {g['count']:>7} {avg_lat:>10.1f}ms {avg_pt:>10.2f}ms/tok {attainment:>10.1f}%")

    print("=" * 60)


def main():
    args = parse_args()
    print(LEGACY_WARNING, file=sys.stderr)
    random.seed(args.seed)
    if args.trace:
        trace_path = args.trace
        cleanup_path = None
    else:
        prompts = [f"Trace prompt {i}" for i in range(args.num_requests)]
        tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
        tmp.close()
        generate_trace(
            tmp.name,
            num_requests=args.num_requests,
            rps=args.rps,
            output_length=args.max_tokens,
            prompts=prompts,
            seed=args.seed,
        )
        trace_path = tmp.name
        cleanup_path = tmp.name

    try:
        runner_args = argparse.Namespace(
            system="adaserve",
            input_file=trace_path,
            draft_model=args.draft_model,
            target_model=args.target_model,
            draft_tp=args.draft_tp,
            target_tp=args.target_tp,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_num_batched_tokens=8192,
            max_num_seqs=128,
            temperature=args.temperature,
            ignore_eos=True,
            num_pearl_steps=100,
            baseline_latency_per_token_ms=args.baseline_latency,
            max_gamma=args.max_gamma,
            min_gamma=args.min_gamma,
            correction_factor=args.correction_factor,
            enforce_eager=True,
        )
        result = run_workload(runner_args)
        print(result["result_text"])
    finally:
        if cleanup_path is not None:
            os.unlink(cleanup_path)


if __name__ == "__main__":
    main()
