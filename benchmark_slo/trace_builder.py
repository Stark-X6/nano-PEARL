from __future__ import annotations

import math
import random


def build_trace(
    *,
    prompts: list[str],
    output_length: int,
    rps: float,
    slo_ratios: list[tuple[float, float]],
    seed: int = 0,
) -> list[dict]:
    rng = random.Random(seed)

    cumulative = []
    total = 0.0
    for ratio, weight in slo_ratios:
        total += weight
        cumulative.append((ratio, total))

    if cumulative:
        last_ratio, _ = cumulative[-1]
        cumulative[-1] = (last_ratio, 1.0)

    current_time_ms = 0.0
    trace = []
    for prompt in prompts:
        u = rng.random()
        if u <= 0:
            u = 1e-10
        interval_ms = -math.log(u) / rps * 1000.0 if rps > 0 else 0.0
        current_time_ms += interval_ms

        ratio_draw = rng.random()
        slo_ratio = cumulative[-1][0]
        for ratio, cum_prob in cumulative:
            if ratio_draw <= cum_prob:
                slo_ratio = ratio
                break

        trace.append(
            {
                "emission_time_ms": round(current_time_ms, 2),
                "prompt": prompt,
                "output_length": output_length,
                "slo_ratio": slo_ratio,
            }
        )

    return trace
