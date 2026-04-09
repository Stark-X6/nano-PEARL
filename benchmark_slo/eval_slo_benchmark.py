#!/usr/bin/env python3
"""
SLO Benchmark — Mode A: Random SLO ratio, batch comparison.

Compares SLO-PEARL (per-seq gamma) vs Original PEARL (uniform gamma).
All requests arrive at once with random SLO ratios.

Usage:
    python eval_slo_benchmark.py \
        --draft-model /path/to/draft \
        --target-model /path/to/target \
        --target-tp 3 \
        --bs 32 \
        --num-pearl-steps 100
"""

import argparse
import copy
import os
import random
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nano_pearl import PEARLConfig, PEARLEngine, SamplingParams, logger
from nano_pearl.slo_config import SLOConfig
from nano_pearl.pearl_engine_slo.slo_pearl_engine import SLOPearlEngine
from nano_pearl.emission import RandomSLOEmission


def parse_args():
    parser = argparse.ArgumentParser(description='SLO Benchmark — Mode A')

    # Model paths
    parser.add_argument('--draft-model', '-d', type=str, required=True)
    parser.add_argument('--target-model', '-t', type=str, required=True)

    # GPU layout
    parser.add_argument('--draft-tp', type=int, default=1)
    parser.add_argument('--target-tp', type=int, default=3)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9)

    # Generation
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--max-tokens', type=int, default=200)
    parser.add_argument('--num-pearl-steps', type=int, default=100)

    # Benchmark
    parser.add_argument('--num-samples', type=int, default=100)
    parser.add_argument('--input-len', type=int, default=1024)
    parser.add_argument('--bs', type=int, default=32)

    # SLO
    parser.add_argument('--max-gamma', type=int, default=16)
    parser.add_argument('--min-gamma', type=int, default=1)
    parser.add_argument('--baseline-latency', type=float, default=-1.0,
                        help='Baseline latency in ms (-1=auto)')
    parser.add_argument('--correction-factor', type=float, default=1.0)

    # Other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warmup-iters', type=int, default=1)
    parser.add_argument('--skip-original', action='store_true',
                        help='Skip original PEARL benchmark')

    return parser.parse_args()


def cleanup_stale_shm():
    """Remove stale shared memory segments from previous runs."""
    import glob
    for path in glob.glob("/dev/shm/*group*"):
        try:
            os.unlink(path)
        except OSError:
            pass


def generate_random_inputs(num_samples, input_len, seed=0):
    rng = random.Random(seed)
    return [[rng.randint(0, 10000) for _ in range(input_len)] for _ in range(num_samples)]


def run_original_pearl(config, inputs, sampling_params, bs, num_pearl_steps):
    cleanup_stale_shm()
    """Run original PEARL with uniform gamma as baseline."""
    logger.info("=" * 60)
    logger.info("Running Original PEARL (uniform gamma)")
    logger.info("=" * 60)

    engine = PEARLEngine(config)

    # Warmup
    if True:
        sp = SamplingParams(temperature=0, ignore_eos=False, max_tokens=512)
        engine.add_request("Benchmark:", sp)
        engine.generate()

    all_num_tokens = []
    all_num_acc_tokens = []
    total_time = 0

    num_complete = len(inputs) // bs
    total = num_complete * bs

    for i in range(0, total, bs):
        batch = inputs[i : i + bs]
        for inp in batch:
            engine.add_request(inp, copy.deepcopy(sampling_params))
        output_text, num_tokens, num_acc_tokens, elapsed = engine.bench_generate(
            num_pearl_steps=num_pearl_steps
        )
        all_num_tokens.extend(num_tokens)
        all_num_acc_tokens.extend(num_acc_tokens)
        total_time += elapsed

    # Compute metrics
    mat_list = [sum(n) / len(n) for n in all_num_acc_tokens if n]
    mat = sum(mat_list) / len(mat_list) if mat_list else 0
    throughput = sum(all_num_tokens) / total_time if total_time > 0 else 0

    engine.exit()

    return {
        'throughput': throughput,
        'mat': mat,
        'total_tokens': sum(all_num_tokens),
        'total_time': total_time,
        'num_acc_tokens': all_num_acc_tokens,
    }


def run_slo_pearl(slo_config, inputs, sampling_params, bs, num_pearl_steps,
                  slo_ratios_dist, seed):
    """Run SLO-PEARL with per-seq gamma."""
    cleanup_stale_shm()
    logger.info("=" * 60)
    logger.info("Running SLO-PEARL (per-seq gamma)")
    logger.info("=" * 60)

    engine = SLOPearlEngine(slo_config)

    # Warmup
    sp = SamplingParams(temperature=0, ignore_eos=False, max_tokens=512)
    engine.add_request("Benchmark:", sp, slo_ratio=1.0)
    engine.slo_generate()

    emission = RandomSLOEmission(slo_ratios=slo_ratios_dist, seed=seed)

    all_num_tokens = []
    all_num_acc_tokens = []
    all_slo_ratios = []
    total_time = 0

    num_complete = len(inputs) // bs
    total = num_complete * bs

    for i in range(0, total, bs):
        batch = inputs[i : i + bs]
        for inp in batch:
            slo_ratio = emission.sample_slo_ratio()
            all_slo_ratios.append(slo_ratio)
            engine.add_request(inp, copy.deepcopy(sampling_params), slo_ratio=slo_ratio)

        output_text, num_tokens, num_acc_tokens, elapsed, slo_metrics = (
            engine.slo_bench_generate(num_pearl_steps=num_pearl_steps)
        )
        all_num_tokens.extend(num_tokens)
        all_num_acc_tokens.extend(num_acc_tokens)
        total_time += elapsed

    # Compute metrics
    mat_list = [sum(n) / len(n) for n in all_num_acc_tokens if n]
    mat = sum(mat_list) / len(mat_list) if mat_list else 0
    throughput = sum(all_num_tokens) / total_time if total_time > 0 else 0

    # Per-SLO-group breakdown
    slo_groups = {}
    for ratio, _ in slo_ratios_dist:
        slo_groups[ratio] = {'tokens': 0, 'count': 0}

    for idx, slo_ratio in enumerate(all_slo_ratios):
        # Find nearest SLO group
        nearest = min(slo_groups.keys(), key=lambda r: abs(r - slo_ratio))
        slo_groups[nearest]['tokens'] += all_num_tokens[idx]
        slo_groups[nearest]['count'] += 1

    engine.exit()

    return {
        'throughput': throughput,
        'mat': mat,
        'total_tokens': sum(all_num_tokens),
        'total_time': total_time,
        'num_acc_tokens': all_num_acc_tokens,
        'slo_ratios': all_slo_ratios,
        'slo_groups': slo_groups,
    }


def print_comparison(original_metrics, slo_metrics, slo_ratios_dist):
    """Print comparison table."""
    print("\n" + "=" * 60)
    print("SLO Benchmark Report - Mode A (Random SLO, batch comparison)")
    print("=" * 60)

    orig_tp = original_metrics['throughput']
    slo_tp = slo_metrics['throughput']
    tp_change = (slo_tp - orig_tp) / orig_tp * 100 if orig_tp > 0 else 0

    print(f"{'':>25} {'Original PEARL':>15} {'SLO-PEARL':>15} {'Change':>10}")
    print("-" * 65)
    print(f"{'Overall throughput':>25} {orig_tp:>12.1f} tok/s {slo_tp:>12.1f} tok/s {tp_change:>+8.1f}%")
    print(f"{'MAT':>25} {original_metrics['mat']:>15.2f} {slo_metrics['mat']:>15.2f}")

    # Per SLO group
    if 'slo_groups' in slo_metrics:
        print()
        print("Per SLO group:")
        for ratio, _ in slo_ratios_dist:
            g = slo_metrics['slo_groups'].get(ratio, {'tokens': 0, 'count': 0})
            count = g['count']
            tokens = g['tokens']
            label = f"slo={ratio}"
            if ratio <= 0.8:
                label += " (urgent)"
            elif ratio >= 1.4:
                label += " (relaxed)"
            print(f"  {label}:")
            print(f"    count: {count}, tokens: {tokens}")

    print("=" * 60)


def main():
    args = parse_args()
    random.seed(args.seed)

    slo_ratios_dist = [
        (0.6, 0.25),
        (1.0, 0.25),
        (1.4, 0.25),
        (1.8, 0.25),
    ]

    # Generate inputs
    inputs = generate_random_inputs(args.num_samples, args.input_len, args.seed)
    sampling_params = SamplingParams(
        temperature=args.temperature,
        ignore_eos=True,
        max_tokens=args.max_tokens,
    )

    # Original PEARL config
    pearl_config = PEARLConfig(
        draft_model_path=args.draft_model,
        target_model_path=args.target_model,
        draft_tensor_parallel_size=args.draft_tp,
        target_tensor_parallel_size=args.target_tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
    )

    # SLO config
    slo_config = SLOConfig(
        draft_model_path=args.draft_model,
        target_model_path=args.target_model,
        draft_tensor_parallel_size=args.draft_tp,
        target_tensor_parallel_size=args.target_tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        max_gamma=args.max_gamma,
        min_gamma=args.min_gamma,
        baseline_latency_ms=args.baseline_latency,
        correction_factor=args.correction_factor,
        slo_ratios=slo_ratios_dist,
    )

    # Run original PEARL
    original_metrics = None
    if not args.skip_original:
        original_metrics = run_original_pearl(
            pearl_config, inputs, sampling_params, args.bs, args.num_pearl_steps
        )

    # Run SLO-PEARL
    slo_metrics = run_slo_pearl(
        slo_config, inputs, sampling_params, args.bs, args.num_pearl_steps,
        slo_ratios_dist, args.seed,
    )

    # Print comparison
    if original_metrics:
        print_comparison(original_metrics, slo_metrics, slo_ratios_dist)
    else:
        print(f"\nSLO-PEARL: throughput={slo_metrics['throughput']:.1f} tok/s, MAT={slo_metrics['mat']:.2f}")


if __name__ == "__main__":
    main()
