"""
SLOPearlEngine — controller for SLO-aware speculative decoding.

Mirrors PEARLEngine structure but spawns SLODraftRunner / SLOTargetRunner
instead of DraftModelRunner / TargetModelRunner.

Provides:
  - add_request() with optional slo_ratio
  - slo_generate() for SLO-aware generation
  - slo_bench_generate() for benchmarking
  - profile_baseline_latency() for auto-profiling
"""

import atexit
import pickle
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.multiprocessing as mp
from multiprocessing.shared_memory import SharedMemory
from transformers import AutoTokenizer

from nano_pearl.pearl_engine.sequence import Sequence
from nano_pearl.layers.sampler import SamplingParams
from nano_pearl.pearl_engine.pearl_engine import Controller
from nano_pearl.pearl_engine_slo.slo_draft_runner import SLODraftRunner
from nano_pearl.pearl_engine_slo.slo_target_runner import SLOTargetRunner
from nano_pearl.pearl_engine_slo.slo_sequence import SLOSequence
from nano_pearl.utils.pearl_logger import logger


@dataclass
class SLOGenerationMetrics:
    """Metrics collected per SLO generation run."""
    # Per-sequence metrics
    seq_id: int = 0
    slo_ratio: float = 1.0
    num_tokens: int = 0
    num_acc_tokens: list = None  # list[int]
    decode_latency_ms: float = 0.0
    slo_constraint_ms: float = 0.0
    slo_attained: bool = False

    # Aggregate metrics
    total_throughput: float = 0.0
    goodput: float = 0.0
    slo_attainment_rate: float = 0.0
    mat: float = 0.0  # mean acceptance tokens

    def __post_init__(self):
        if self.num_acc_tokens is None:
            self.num_acc_tokens = []


class SLOController(Controller):
    """Extended Controller that also stores SLO config for runner initialization."""

    def __init__(self, config, control_event):
        super().__init__(config.pearl_config, control_event)
        self.slo_config = config


class SLOPearlEngine:
    """SLO-aware PEARL engine.

    Composes SLOConfig and spawns SLODraftRunner / SLOTargetRunner processes.
    Provides SLO-aware request submission and generation.
    """

    def __init__(self, slo_config):
        """Initialize SLO engine with SLOConfig.

        Args:
            slo_config: SLOConfig instance with model paths, GPU layout, and SLO parameters.
        """
        self.slo_config = slo_config
        self.config = slo_config.pearl_config
        self.ps = []

        ctx = mp.get_context("spawn")
        self.control_event = ctx.Event()

        # Store SLO parameters on PEARLConfig for runners to access
        # (runners receive PEARLConfig in __init__)
        self.config._slo_max_gamma = slo_config.max_gamma
        self.config._slo_min_gamma = slo_config.min_gamma
        self.config._slo_correction_factor = slo_config.correction_factor
        self.config._slo_baseline_latency_ms = slo_config.baseline_latency_ms
        self.config._slo_draft_step_latency_ms = slo_config.draft_step_latency_ms
        self.config._slo_verify_step_latency_ms = slo_config.verify_step_latency_ms
        self.config._slo_total_budget = slo_config.total_draft_budget

        self.controller = SLOController(slo_config, self.control_event)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.draft_config.model, use_fast=True
        )
        self.config.eos = self.config.draft_config.eos

        logger.info(
            f"[SLO Engine] EOS token id: {self.config.eos}, "
            f"EOS tokens: {self.tokenizer.decode(self.config.eos)}"
        )

        # Spawn processes
        for i in range(self.config.world_size):
            event = ctx.Event()
            runner_cls = SLODraftRunner if i in self.config.draft_config.devices else SLOTargetRunner
            process = ctx.Process(
                target=runner_cls,
                args=(self.config, i, event, self.control_event),
            )
            process.daemon = True
            process.start()
            self.ps.append(process)
            self.controller.add_event(i, event)

        logger.info(
            "[SLO Engine] Waiting for model initialization...", color="red"
        )
        self.control_event.wait()
        self.control_event.clear()

        atexit.register(self.exit)

    def add_request(self, prompt, sampling_params=None, slo_ratio=1.0):
        """Add a request with SLO ratio.

        Args:
            prompt: str or list[int] token ids
            sampling_params: SamplingParams (default: greedy, 128 max tokens)
            slo_ratio: SLO ratio (0.6=urgent, 1.0=normal, 1.4=relaxed, 1.8=very relaxed)
        """
        if isinstance(prompt, str):
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt = self.tokenizer.encode(prompt)

        if sampling_params is None:
            sampling_params = SamplingParams(max_tokens=128)

        seq = Sequence(prompt, sampling_params)
        slo_seq = SLOSequence(seq, slo_ratio=slo_ratio)

        self.controller.write_draft_shm("add_request", slo_seq)
        self.controller.write_target_shm("add_request", slo_seq)
        self.control_event.wait()
        self.control_event.clear()
        return seq.seq_id

    def slo_generate(self):
        """Run SLO-aware PEARL generation.

        Returns:
            output_text, num_tokens, num_acc_tokens, elapsed_time, slo_metrics
        """
        self.controller.write_draft_shm("slo_generate")
        self.controller.write_target_shm("slo_generate")
        self.control_event.wait()
        self.control_event.clear()

        output, elapsed_time = self.controller.read_output()
        output = sorted(output, key=lambda x: x[0])
        seq_id, token_ids, num_acc_tokens = zip(*output)
        output_text = [
            self.tokenizer.decode(tids, skip_special_tokens=False)
            for tids in token_ids
        ]
        num_tokens = [len(t) for t in token_ids]

        # Compute SLO metrics
        slo_metrics = self._compute_metrics(
            output, elapsed_time, num_tokens, num_acc_tokens
        )

        return output_text, num_tokens, num_acc_tokens, elapsed_time, slo_metrics

    def slo_bench_generate(self, num_pearl_steps=100):
        """Benchmark: fixed steps, collect per-seq metrics.

        Returns:
            output_text, num_tokens, num_acc_tokens, elapsed_time, slo_metrics
        """
        self.controller.write_draft_shm("slo_bench_generate", num_pearl_steps)
        self.controller.write_target_shm("slo_bench_generate", num_pearl_steps)
        self.control_event.wait()
        self.control_event.clear()

        output, elapsed_time = self.controller.read_output()
        output = sorted(output, key=lambda x: x[0])
        seq_id, token_ids, num_acc_tokens = zip(*output)
        output_text = [
            self.tokenizer.decode(tids, skip_special_tokens=False)
            for tids in token_ids
        ]
        num_tokens = [len(t) for t in token_ids]

        slo_metrics = self._compute_metrics(
            output, elapsed_time, num_tokens, num_acc_tokens
        )

        return output_text, num_tokens, num_acc_tokens, elapsed_time, slo_metrics

    def slo_bench_generate_raw(self, num_pearl_steps=100):
        self.controller.write_draft_shm("slo_bench_generate", num_pearl_steps)
        self.controller.write_target_shm("slo_bench_generate", num_pearl_steps)
        self.control_event.wait()
        self.control_event.clear()
        output, elapsed_time = self.controller.read_output()
        output = sorted(output, key=lambda x: x[0])
        return output, elapsed_time

    def _compute_metrics(self, output, elapsed_time, num_tokens, num_acc_tokens):
        """Compute SLO attainment metrics.

        Args:
            output: list of (seq_id, token_ids, acc_tokens)
            elapsed_time: total generation time in seconds
            num_tokens: list of completion token counts
            num_acc_tokens: list of per-step acceptance counts

        Returns:
            SLOGenerationMetrics with aggregate statistics
        """
        total_toks = sum(num_tokens)
        total_time_s = elapsed_time

        metrics = SLOGenerationMetrics()
        metrics.total_throughput = total_toks / total_time_s if total_time_s > 0 else 0

        # MAT: mean acceptance tokens per step
        all_acc = []
        for acc_list in num_acc_tokens:
            if acc_list:
                all_acc.extend(acc_list)
        metrics.mat = sum(all_acc) / len(all_acc) if all_acc else 0

        # SLO attainment per sequence (based on slo_ratio distribution)
        baseline_ms = self.slo_config.baseline_latency_ms
        if baseline_ms > 0:
            # Per-request SLO check: decode_latency <= slo_ratio * baseline_latency
            # For bench mode, estimate per-token latency
            n_seqs = len(num_tokens)
            per_seq_latency_ms = [
                (total_time_s * 1000) / max(nt, 1)
                for nt in num_tokens
            ]

            # Get slo_ratios from config distribution
            slo_ratios = [r for r, _ in self.slo_config.slo_ratios]
            slo_attained_count = 0
            goodput_toks = 0

            for i, latency in enumerate(per_seq_latency_ms):
                # Assign SLO ratio based on sequence index (round-robin from distribution)
                ratio = slo_ratios[i % len(slo_ratios)]
                constraint = ratio * baseline_ms
                if latency <= constraint:
                    slo_attained_count += 1
                    goodput_toks += num_tokens[i]

            metrics.slo_attainment_rate = slo_attained_count / n_seqs if n_seqs > 0 else 0
            metrics.goodput = goodput_toks / total_time_s if total_time_s > 0 else 0
        else:
            metrics.slo_attainment_rate = 1.0
            metrics.goodput = metrics.total_throughput

        return metrics

    def profile_baseline_latency(self):
        """Profile baseline latency using auto_set_gamma.

        Extracts baseline_latency_ms = 1000 / target_speed[bs=1].
        """
        self.controller.write_draft_shm("profile_slo_latency")
        self.controller.write_target_shm("profile_slo_latency")
        self.control_event.wait()
        self.control_event.clear()

        logger.info("[SLO Engine] Baseline latency profiling complete.")

    def exit(self):
        """Shutdown all runner processes."""
        self.controller.write_draft_shm("exit")
        self.controller.write_target_shm("exit")
        for p in self.ps:
            p.join()
        self.controller.draft_shm.close()
        self.controller.target_shm.close()
        self.controller.draft_shm.unlink()
        self.controller.target_shm.unlink()

    def slo_generate_double_buffer(self):
        """
        [Double Buffering] 运行双缓冲 SLO 感知投机采样推理。
        逐行解释：
        1. 向控制器发送指令，让 Draft 和 Target 运行对应的 _double_buffer 函数。
        2. 等待控制事件信号（由 Rank 0 在完成任务后设置）。
        3. 从共享内存读取生成结果和总耗时。
        4. 对结果进行排序、解码，并计算 SLO 指标。
        """
        # 通过共享内存命令字触发 Runner 执行新函数
        self.controller.write_draft_shm("slo_generate_double_buffer")
        self.controller.write_target_shm("slo_generate_double_buffer")
         
        # 阻塞等待子进程完成
        self.control_event.wait()
        self.control_event.clear()

        # 读取并解析输出
        output, elapsed_time = self.controller.read_output()
        output = sorted(output, key=lambda x: x[0])
        seq_id, token_ids, num_acc_tokens = zip(*output)
         
        # 解码 Token 为文本
        output_text = [
            self.tokenizer.decode(tids, skip_special_tokens=False)
            for tids in token_ids
        ]
        num_tokens = [len(t) for t in token_ids]

        # 调用现有的 _compute_metrics 计算 SLO 达标率等实验数据
        slo_metrics = self._compute_metrics(
            output, elapsed_time, num_tokens, num_acc_tokens
        )

        return output_text, num_tokens, num_acc_tokens, elapsed_time, slo_metrics

    def slo_bench_generate_double_buffer(self, num_pearl_steps=100):
        """
        [Double Buffering] 用于 Benchmark 的双缓冲函数。
        """
        # 发送带参数的指令（指定运行步数）
        self.controller.write_draft_shm("slo_bench_generate_double_buffer", num_pearl_steps)
        self.controller.write_target_shm("slo_bench_generate_double_buffer", num_pearl_steps)
        
        self.control_event.wait()
        self.control_event.clear()

        output, elapsed_time = self.controller.read_output()
        output = sorted(output, key=lambda x: x[0])
        seq_id, token_ids, num_acc_tokens = zip(*output)
         
        output_text = [
            self.tokenizer.decode(tids, skip_special_tokens=False)
            for tids in token_ids
        ]
        num_tokens = [len(t) for t in token_ids]

        slo_metrics = self._compute_metrics(
            output, elapsed_time, num_tokens, num_acc_tokens
        )

        return output_text, num_tokens, num_acc_tokens, elapsed_time, slo_metrics

    def slo_bench_generate_double_buffer_raw(self, num_pearl_steps=100):
        self.controller.write_draft_shm("slo_bench_generate_double_buffer", num_pearl_steps)
        self.controller.write_target_shm("slo_bench_generate_double_buffer", num_pearl_steps)
        self.control_event.wait()
        self.control_event.clear()
        output, elapsed_time = self.controller.read_output()
        output = sorted(output, key=lambda x: x[0])
        return output, elapsed_time
