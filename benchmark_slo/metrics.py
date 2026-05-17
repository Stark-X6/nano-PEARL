from __future__ import annotations

from dataclasses import dataclass


@dataclass(eq=True, frozen=True)
class RequestRecord:
    request_id: int
    slo_ratio: float
    arrival_time_ms: float
    decode_start_time_ms: float
    finish_time_ms: float
    num_generated_tokens: int
    attained: bool


def compute_metrics(records: list[RequestRecord], total_run_time_s: float) -> dict:
    completed_requests = len(records)
    total_generated_tokens = sum(record.num_generated_tokens for record in records)
    attained_records = [record for record in records if record.attained]
    attained_request_count = len(attained_records)
    attained_tokens = sum(record.num_generated_tokens for record in attained_records)

    slo_attainment_by_scale: dict[float, dict[str, float | int]] = {}
    for record in records:
        bucket = slo_attainment_by_scale.setdefault(
            record.slo_ratio,
            {"attained": 0, "total": 0, "rate": 0.0},
        )
        bucket["total"] += 1
        if record.attained:
            bucket["attained"] += 1

    for bucket in slo_attainment_by_scale.values():
        total = bucket["total"]
        bucket["rate"] = bucket["attained"] / total if total else 0.0

    return {
        "completed_requests": completed_requests,
        "total_generated_tokens": total_generated_tokens,
        "goodput": attained_tokens / total_run_time_s if total_run_time_s > 0 else 0.0,
        "slo_attainment": attained_request_count / completed_requests if completed_requests > 0 else 0.0,
        "slo_attainment_by_scale": slo_attainment_by_scale,
        "total_run_time_s": total_run_time_s,
    }
