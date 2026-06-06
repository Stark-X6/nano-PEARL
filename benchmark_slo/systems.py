from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable


SUPPORTED_SYSTEMS = ("vllm-spec", "pearl-spec", "adaserve", "slopearl")


@dataclass
class BenchmarkSystem:
    name: str
    engine: Any
    run_method_name: str
    num_pearl_steps: int
    needs_slo_ratio: bool
    seq_id_to_request_id: dict[int, int]
    supports_batch_reuse: bool = True

    def add_request(self, prompt: str | list[int], sampling_params: Any, slo_ratio: float, request_id: int | None = None) -> int:
        sampling_params = copy.deepcopy(sampling_params)
        if self.needs_slo_ratio:
            seq_id = self.engine.add_request(prompt, sampling_params, slo_ratio=slo_ratio)
        else:
            seq_id = self.engine.add_request(prompt, sampling_params)
        if request_id is None:
            request_id = len(self.seq_id_to_request_id)
        self.seq_id_to_request_id[seq_id] = request_id
        return seq_id

    def run(self):
        run_method = getattr(self.engine, self.run_method_name)
        return run_method(num_pearl_steps=self.num_pearl_steps)

    def exit(self) -> None:
        self.engine.exit()


def _default_engine_factory(system_name: str, args: Any):
    from nano_pearl import PEARLConfig, PEARLEngine
    from nano_pearl.slo_config import SLOConfig
    from nano_pearl.pearl_engine_slo.slo_pearl_engine import SLOPearlEngine

    if system_name in ("vllm-spec", "pearl-spec"):
        config = PEARLConfig(
            draft_model_path=args.draft_model,
            target_model_path=args.target_model,
            draft_tensor_parallel_size=getattr(args, "draft_tp", 1),
            target_tensor_parallel_size=getattr(args, "target_tp", 3),
            max_num_batched_tokens=getattr(args, "max_num_batched_tokens", 8192),
            max_num_seqs=getattr(args, "max_num_seqs", 128),
            gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.9),
            enforce_eager=getattr(args, "enforce_eager", True),
        )
        return PEARLEngine(config)

    config = SLOConfig(
        draft_model_path=args.draft_model,
        target_model_path=args.target_model,
        draft_tensor_parallel_size=getattr(args, "draft_tp", 1),
        target_tensor_parallel_size=getattr(args, "target_tp", 3),
        max_num_batched_tokens=getattr(args, "max_num_batched_tokens", 8192),
        max_num_seqs=getattr(args, "max_num_seqs", 128),
        gpu_memory_utilization=getattr(args, "gpu_memory_utilization", 0.9),
        enforce_eager=getattr(args, "enforce_eager", True),
        max_gamma=getattr(args, "max_gamma", 16),
        min_gamma=getattr(args, "min_gamma", 1),
        baseline_latency_ms=getattr(args, "baseline_latency_per_token_ms", -1.0),
        correction_factor=getattr(args, "correction_factor", 1.0),
        enable_double_buffering=(system_name == "slopearl"),
    )
    return SLOPearlEngine(config)


def create_system(
    system_name: str,
    args: Any,
    engine_factory: Callable[[str, Any], Any] | None = None,
) -> BenchmarkSystem:
    if system_name not in SUPPORTED_SYSTEMS:
        raise ValueError(f"unsupported system '{system_name}'")

    if engine_factory is None:
        engine_factory = _default_engine_factory

    engine = engine_factory(system_name, args)

    if system_name == "vllm-spec":
        return BenchmarkSystem(
            name=system_name,
            engine=engine,
            run_method_name="vllm_spec_bench_generate_raw",
            num_pearl_steps=args.num_pearl_steps,
            needs_slo_ratio=False,
            seq_id_to_request_id={},
        )
    if system_name == "pearl-spec":
        return BenchmarkSystem(
            name=system_name,
            engine=engine,
            run_method_name="bench_generate_raw",
            num_pearl_steps=args.num_pearl_steps,
            needs_slo_ratio=False,
            seq_id_to_request_id={},
        )
    if system_name == "adaserve":
        return BenchmarkSystem(
            name=system_name,
            engine=engine,
            run_method_name="slo_bench_generate_raw",
            num_pearl_steps=args.num_pearl_steps,
            needs_slo_ratio=True,
            seq_id_to_request_id={},
        )

    return BenchmarkSystem(
        name=system_name,
        engine=engine,
        run_method_name="slo_bench_generate_double_buffer_raw",
        num_pearl_steps=args.num_pearl_steps,
        needs_slo_ratio=True,
        seq_id_to_request_id={},
        supports_batch_reuse=False,
    )
