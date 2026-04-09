#!/usr/bin/env python3
"""
Generate test trace JSON for Mode B benchmarks.

Output format: [{"emission_time_ms": float, "slo_ratio": float, "prompt_len": int}, ...]

Usage:
    python generate_trace.py --num-requests 100 --rps 2.0 --output trace.json
"""

import argparse
import json
import math
import random


def generate_trace(output_path, num_requests=100, rps=2.0,
                   slo_ratios=None, prompt_len=1024, seed=0):
    """Generate a trace file with Poisson arrivals and random SLO ratios."""
    if slo_ratios is None:
        slo_ratios = [(0.6, 0.25), (1.0, 0.25), (1.4, 0.25), (1.8, 0.25)]

    rng = random.Random(seed)

    # Build cumulative SLO distribution
    cum_ratios = []
    cum = 0.0
    for ratio, weight in slo_ratios:
        cum += weight
        cum_ratios.append((ratio, cum))

    trace = []
    current_time_ms = 0.0

    for i in range(num_requests):
        # Sample inter-arrival time from exponential distribution
        u = rng.random()
        if u <= 0:
            u = 1e-10
        interval_ms = -math.log(u) / rps * 1000.0

        current_time_ms += interval_ms

        # Sample SLO ratio
        r = rng.random()
        slo_ratio = cum_ratios[-1][0]
        for ratio, cum_prob in cum_ratios:
            if r <= cum_prob:
                slo_ratio = ratio
                break

        trace.append({
            'emission_time_ms': round(current_time_ms, 2),
            'slo_ratio': slo_ratio,
            'prompt_len': prompt_len,
        })

    with open(output_path, 'w') as f:
        json.dump(trace, f, indent=2)

    print(f"Generated {num_requests} requests to {output_path}")
    print(f"  RPS: {rps}")
    print(f"  Duration: {current_time_ms / 1000:.1f}s")
    print(f"  SLO ratios: {[r for r, _ in slo_ratios]}")

    return trace


def main():
    parser = argparse.ArgumentParser(description='Generate test trace JSON')
    parser.add_argument('--output', '-o', type=str, default='trace.json')
    parser.add_argument('--num-requests', type=int, default=100)
    parser.add_argument('--rps', type=float, default=2.0)
    parser.add_argument('--prompt-len', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    generate_trace(
        args.output, args.num_requests, args.rps,
        prompt_len=args.prompt_len, seed=args.seed,
    )


if __name__ == "__main__":
    main()
