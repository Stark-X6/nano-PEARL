from __future__ import annotations


def format_result_text(system_name: str, metrics: dict) -> str:
    scale_entries = []
    for slo_ratio, stats in metrics["slo_attainment_by_scale"].items():
        scale_entries.append(
            f"{slo_ratio:.3f} : {stats['rate'] * 100:.3f}% "
            f"({stats['attained']}/{stats['total']})"
        )

    lines = [
        f"system({system_name})",
        f"completed_requests({metrics['completed_requests']})",
        f"total_generated_tokens({metrics['total_generated_tokens']})",
        f"slo_attainment({metrics['slo_attainment'] * 100:.3f}%)",
        f"goodput({metrics['goodput']:.3f})",
        f"total_run_time_s({metrics['total_run_time_s']:.3f})",
        "slo_attainment_by_scale(" + " ".join(scale_entries) + ")",
    ]
    return "\n".join(lines)
