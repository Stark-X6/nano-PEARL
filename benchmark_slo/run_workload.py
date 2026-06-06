from __future__ import annotations

import argparse
from typing import Any

from benchmark_slo.metrics import RequestRecord, compute_metrics
from benchmark_slo.systems import SUPPORTED_SYSTEMS, create_system
from benchmark_slo.workload_loader import WorkloadRequest, load_workload


def _make_token_counter(model_path: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

    def count_tokens(prompt: str | list[int]) -> int:
        if isinstance(prompt, list):
            return len(prompt)
        templated = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        return len(tokenizer.encode(templated))

    return count_tokens


def build_workload_batches(
    workload: list[WorkloadRequest],
    max_num_seqs: int,
    max_num_batched_tokens: int,
    token_count_fn,
) -> list[list[WorkloadRequest]]:
    batches: list[list[WorkloadRequest]] = []
    current_batch: list[WorkloadRequest] = []
    current_tokens = 0

    for request in workload:
        request_tokens = token_count_fn(request.prompt)
        would_overflow = (
            current_batch
            and (
                len(current_batch) >= max_num_seqs
                or current_tokens + request_tokens > max_num_batched_tokens
            )
        )
        if would_overflow:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0

        current_batch.append(request)
        current_tokens += request_tokens

    if current_batch:
        batches.append(current_batch)
    return batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Formal AdaServe-aligned workload runner")
    parser.add_argument("--system", choices=SUPPORTED_SYSTEMS, required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-tp", type=int, default=1)
    parser.add_argument("--target-tp", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--num-pearl-steps", type=int, default=100)
    parser.add_argument("--baseline-latency-per-token-ms", type=float, default=-1.0)
    parser.add_argument("--max-gamma", type=int, default=16)
    parser.add_argument("--min-gamma", type=int, default=1)
    parser.add_argument("--correction-factor", type=float, default=1.0)
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args()


def build_request_records(
    workload: list[WorkloadRequest],
    raw_output: list[tuple[int, list[int], list[int]]],
    seq_id_to_request_id: dict[int, int],
    total_run_time_s: float,
    baseline_latency_per_token_ms: float,
) -> list[RequestRecord]:
    total_run_time_ms = total_run_time_s * 1000.0
    records: list[RequestRecord] = []

    for seq_id, token_ids, _ in raw_output:
        request_id = seq_id_to_request_id[seq_id]
        request = workload[request_id]
        generated_tokens = len(token_ids)
        per_token_latency_ms = total_run_time_ms / max(generated_tokens, 1)
        if request.slo_ratio > 0:
            slo_constraint_ms = request.slo_ratio * baseline_latency_per_token_ms
        else:
            slo_constraint_ms = -request.slo_ratio

        records.append(
            RequestRecord(
                request_id=request.request_id,
                slo_ratio=request.slo_ratio,
                arrival_time_ms=request.emission_time_ms,
                decode_start_time_ms=request.emission_time_ms,
                finish_time_ms=request.emission_time_ms + total_run_time_ms,
                num_generated_tokens=generated_tokens,
                attained=per_token_latency_ms <= slo_constraint_ms,
            )
        )

    records.sort(key=lambda record: record.request_id)
    return records


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


def run_workload(
    args: Any,
    *,
    load_workload_fn=load_workload,
    create_system_fn=create_system,
    compute_metrics_fn=compute_metrics,
    sampling_params_cls=None,
    token_count_fn=None,
) -> dict[str, Any]:
    if sampling_params_cls is None:
        from nano_pearl import SamplingParams

        sampling_params_cls = SamplingParams

    workload = load_workload_fn(args.input_file)
    if token_count_fn is None:
        token_count_fn = _make_token_counter(args.draft_model)
    batches = build_workload_batches(
        workload,
        max_num_seqs=getattr(args, 'max_num_seqs', 128),
        max_num_batched_tokens=getattr(args, 'max_num_batched_tokens', 8192),
        token_count_fn=token_count_fn,
    )
    system = None
    all_records = []
    all_raw_output = []
    total_elapsed_time_s = 0.0

    try:
        for batch in batches:
            if system is None:
                system = create_system_fn(args.system, args)

            local_seq_id_to_request_id: dict[int, int] = {}
            for request in batch:
                sampling_params = sampling_params_cls(
                    temperature=getattr(args, "temperature", 0.0),
                    ignore_eos=getattr(args, "ignore_eos", True),
                    max_tokens=request.output_length,
                )
                seq_id = system.add_request(
                    request.prompt,
                    sampling_params,
                    request.slo_ratio,
                    request_id=request.request_id,
                )
                if seq_id is not None:
                    local_seq_id_to_request_id[seq_id] = request.request_id

            raw_output, elapsed_time_s = system.run()
            total_elapsed_time_s += elapsed_time_s
            seq_id_to_request_id = getattr(system, "seq_id_to_request_id", None) or local_seq_id_to_request_id
            batch_records = build_request_records(
                workload=workload,
                raw_output=raw_output,
                seq_id_to_request_id=seq_id_to_request_id,
                total_run_time_s=elapsed_time_s,
                baseline_latency_per_token_ms=getattr(args, "baseline_latency_per_token_ms", -1.0),
            )
            all_records.extend(batch_records)
            all_raw_output.extend(raw_output)

            if not getattr(system, "supports_batch_reuse", True):
                system.exit()
                system = None

        metrics = compute_metrics_fn(all_records, total_elapsed_time_s)
        result_text = format_result_text(args.system, metrics)
        return {
            "records": all_records,
            "metrics": metrics,
            "result_text": result_text,
            "raw_output": all_raw_output,
        }
    finally:
        if system is not None:
            system.exit()


def main() -> None:
    args = parse_args()
    result = run_workload(args)
    print(result["result_text"])


if __name__ == "__main__":
    main()
