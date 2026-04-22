"""
SLOConfig — extends PEARLConfig with SLO-aware budget allocation parameters.
对应 AdaServe RequestManager 中的配置 (baseline_latency_ms, correction_factor, etc.)
"""

from dataclasses import dataclass
from typing import Optional
from nano_pearl.pearl_config import PEARLConfig
from nano_pearl.utils.pearl_logger import logger


@dataclass
class SLOConfig:
    """SLO-aware PEARL configuration.

    Composes an internal PEARLConfig for model/GPU setup,
    and adds SLO-specific budget allocation parameters.
    """

    # Model paths (passed through to PEARLConfig)
    draft_model_path: str = ""
    target_model_path: str = ""

    # GPU layout: 1 draft GPU + 3 verify GPUs (TP=3)
    draft_tensor_parallel_size: int = 1
    target_tensor_parallel_size: int = 3

    # PEARLConfig pass-through
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 128
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    kvcache_block_size: int = 256
    enforce_eager: bool = True   # True for SLO (verify has variable-length inputs)

    # --- SLO-specific parameters ---
    # baseline_latency_ms: per-token latency for target model at bs=1.
    #   -1 means auto-profile from auto_set_gamma().
    #   对应 AdaServe: set_baseline_latency(double)
    baseline_latency_ms: float = -1.0

    # max_gamma: maximum draft tokens per request per step.
    #   对应 AdaServe: max_tree_depth
    max_gamma: int = 16

    # min_gamma: minimum draft tokens per request per step.
    min_gamma: int = 1

    # total_draft_budget: total draft tokens available per PEARL step.
    #   -1 means auto (max_gamma * max_num_seqs).
    total_draft_budget: int = -1

    # correction_factor: scaling factor for SLO budget calculation.
    #   对应 AdaServe: correction_factor
    correction_factor: float = 1.0

    # slo_ratio distribution for random sampling: [(ratio, weight), ...]
    #   weights should sum to 1.0. Used by emission machines.
    #   对应 AdaServe: slo_ratios in EmissionMachine
    slo_ratios: Optional[list] = None  # list[tuple[float, float]]
    enable_double_buffering: bool = False

    # Latency estimates for draft and verify (auto-profiled if not set)
    # 对应 AdaServe: ssm_spec_latency_ms / max_tree_depth
    draft_step_latency_ms: float = -1.0
    # 对应 AdaServe: batch_size_2_latency_ms_map
    verify_step_latency_ms: float = -1.0

    def __post_init__(self):
        if self.slo_ratios is None:
            self.slo_ratios = [
                (0.6, 0.25),
                (1.0, 0.25),
                (1.4, 0.25),
                (1.8, 0.25),
            ]

        if self.total_draft_budget == -1:
            self.total_draft_budget = self.max_gamma * self.max_num_seqs

        # Build internal PEARLConfig
        self.pearl_config = PEARLConfig(
            draft_model_path=self.draft_model_path,
            target_model_path=self.target_model_path,
            draft_tensor_parallel_size=self.draft_tensor_parallel_size,
            target_tensor_parallel_size=self.target_tensor_parallel_size,
            max_num_batched_tokens=self.max_num_batched_tokens,
            max_num_seqs=self.max_num_seqs,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            kvcache_block_size=self.kvcache_block_size,
            enforce_eager=self.enforce_eager,
            gamma=-1,  # gamma is managed by SLOScheduler, not PEARL
        )

        logger.info("=" * 50)
        logger.info("SLO_Config:")
        logger.info(f"Baseline_Latency={self.baseline_latency_ms} ms/tok [-1=auto]")
        logger.info(f"Max_Gamma={self.max_gamma}")
        logger.info(f"Min_Gamma={self.min_gamma}")
        logger.info(f"Total_Draft_Budget={self.total_draft_budget}")
        logger.info(f"Correction_Factor={self.correction_factor}")
        logger.info(f"SLO_Ratios={self.slo_ratios}")
        logger.info(f"Enable_Double_Buffering={self.enable_double_buffering}")
        logger.info(f"Draft_Step_Latency={self.draft_step_latency_ms} ms [-1=auto]")
        logger.info(f"Verify_Step_Latency={self.verify_step_latency_ms} ms [-1=auto]")
        logger.info("=" * 50)
