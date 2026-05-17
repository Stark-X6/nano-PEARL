#!/usr/bin/env python3
"""
Legacy compatibility wrapper for trace generation.

This script now writes the canonical AdaServe-style schema:
[{emission_time_ms, prompt, output_length, slo_ratio}, ...]
"""

import argparse
import json

from benchmark_slo.trace_builder import build_trace


def generate_trace(
    output_path,
    num_requests=100,
    rps=2.0,
    slo_ratios=None,
    output_length=128,
    prompts=None,
    seed=0,
):
    if slo_ratios is None:
        slo_ratios = [(0.6, 0.25), (1.0, 0.25), (1.4, 0.25), (1.8, 0.25)]
    if prompts is None:
        prompts = [f"prompt-{i}" for i in range(num_requests)]
    if len(prompts) != num_requests:
        raise ValueError("num_requests must match len(prompts)")

    trace = build_trace(
        prompts=prompts,
        output_length=output_length,
        rps=rps,
        slo_ratios=slo_ratios,
        seed=seed,
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trace, f, indent=2)

    return trace


def main():
    parser = argparse.ArgumentParser(description="Generate canonical test trace JSON")
    parser.add_argument("--output", "-o", type=str, default="trace.json")
    parser.add_argument("--num-requests", type=int, default=100)
    parser.add_argument("--rps", type=float, default=2.0)
    parser.add_argument("--output-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    generate_trace(
        args.output,
        num_requests=args.num_requests,
        rps=args.rps,
        output_length=args.output_length,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
