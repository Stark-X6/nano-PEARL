from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(eq=True, frozen=True)
class WorkloadRequest:
    request_id: int
    emission_time_ms: float
    prompt: str
    output_length: int
    slo_ratio: float


_REQUIRED_FIELDS = (
    "emission_time_ms",
    "prompt",
    "output_length",
    "slo_ratio",
)


def load_workload(path: str) -> list[WorkloadRequest]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError("workload must be a list of request objects")

    requests: list[WorkloadRequest] = []
    for request_id, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"request at index {request_id} must be an object")
        for field in _REQUIRED_FIELDS:
            if field not in item:
                raise ValueError(f"request at index {request_id} is missing required field '{field}'")
        requests.append(
            WorkloadRequest(
                request_id=request_id,
                emission_time_ms=float(item["emission_time_ms"]),
                prompt=str(item["prompt"]),
                output_length=int(item["output_length"]),
                slo_ratio=float(item["slo_ratio"]),
            )
        )
    return requests
