"""
SLODraftRunner — per-sequence gamma drafting for SLO-aware speculative decoding.

Key difference from original DraftModelRunner:
  1. Drafts max_gamma tokens for ALL sequences (collect logits each step)
  2. Calls SLOScheduler to compute per-seq gamma based on SLO constraints
  3. Rolls back excess tokens per sequence
  4. Uses two-broadcast verify protocol (gamma_map + variable-length token msg)

Mirrors AdaServe's "build full tree then prune" pattern for linear sequences.
"""

import pickle
import math
import torch
import torch.distributed as dist
from nano_pearl.pearl_engine.pearl_model_runner import ModelRunnerBase
from nano_pearl.pearl_engine.sequence import SequenceStatus
from nano_pearl.pearl_engine.scheduler import is_eos
from nano_pearl.pearl_engine_slo.slo_sequence import SLOSequence
from nano_pearl.pearl_engine_slo.slo_scheduler import SLOScheduler
from nano_pearl.utils.context import reset_context
from nano_pearl.pearl_engine_slo.slo_verify_protocol import (
    build_verify_payload,
    count_next_round_tokens,
    count_verify_tokens,
)
from nano_pearl.utils.pearl_logger import logger


class SLODraftRunner(ModelRunnerBase):
    """Draft model runner with SLO-aware per-sequence gamma allocation.

    Overrides pearl_step() and verify() from DraftModelRunner.
    Inherits all infrastructure (init_dist, init_model_and_kvcache, shared memory loop, etc.)
    from ModelRunnerBase.
    """

    def __init__(self, config, rank, event, control_event):
        # Store SLO config before super().__init__ (which calls init_model_and_kvcache)
        self._slo_max_gamma = getattr(config, '_slo_max_gamma', 16)
        self._slo_min_gamma = getattr(config, '_slo_min_gamma', 1)
        self._slo_correction_factor = getattr(config, '_slo_correction_factor', 1.0)
        self._slo_baseline_latency_ms = getattr(config, '_slo_baseline_latency_ms', -1.0)
        self._slo_draft_step_latency_ms = getattr(config, '_slo_draft_step_latency_ms', -1.0)
        self._slo_verify_step_latency_ms = getattr(config, '_slo_verify_step_latency_ms', -1.0)
        self._slo_total_budget = getattr(config, '_slo_total_budget', -1)

        # MUST initialize these BEFORE super().__init__(), which calls
        # init_shared_memory() -> self.loop() (blocking). Anything set
        # after super().__init__() is unreachable.
        self._slo_seqs = {}  # seq_id -> SLOSequence

        # Initialize SLO scheduler before super().__init__
        class _SchedConfig:
            pass
        _sc = _SchedConfig()
        _sc.min_gamma = self._slo_min_gamma
        _sc.max_gamma = self._slo_max_gamma
        _sc.correction_factor = self._slo_correction_factor
        self.slo_scheduler = SLOScheduler(_sc)

        # Latency defaults
        self.baseline_latency_ms = self._slo_baseline_latency_ms if self._slo_baseline_latency_ms > 0 else 30.0
        self.draft_step_latency_ms = self._slo_draft_step_latency_ms if self._slo_draft_step_latency_ms > 0 else 5.0
        self.verify_step_latency_ms = self._slo_verify_step_latency_ms if self._slo_verify_step_latency_ms > 0 else 25.0

        super().__init__(config, rank, event, control_event)

        # Code after super().__init__() is UNREACHABLE — loop() blocks forever.
        # All initialization must go above.

    def add_request(self, seq):
        """Override to wrap Sequence in SLOSequence."""
        if isinstance(seq, SLOSequence):
            slo_seq = seq
        else:
            slo_seq = SLOSequence(seq)
        self._slo_seqs[seq.seq_id] = slo_seq
        # Add underlying Sequence to scheduler
        self.scheduler.add(slo_seq.seq)
        dist.barrier()

    def _get_slo_seqs(self, seqs):
        """Get SLOSequence wrappers for the given Sequence list."""
        result = []
        for seq in seqs:
            if seq.seq_id in self._slo_seqs:
                result.append(self._slo_seqs[seq.seq_id])
            else:
                result.append(SLOSequence(seq))
        return result

    @property
    def max_gamma(self):
        return self._slo_max_gamma

    def pearl_step(self):
        """SLO-aware drafting: draft max_gamma, compute probs, allocate, rollback, verify.

        Corresponds to AdaServe's flow:
          SSM runs max_tree_depth steps -> prune_token_tree() -> rollback
        """
        # ===== Phase A: Draft max_gamma tokens for all seqs, collect logits =====
        draft_logits_steps = []      # list of (num_seqs, vocab) tensors
        draft_token_ids_steps = []   # list of list[int]
        seqs = []

        for step in range(self.max_gamma):
            seqs, is_prefill = self.scheduler.schedule()
            if is_prefill:
                input_ids, positions = self.prepare_prefill(seqs)
                temperatures = self.prepare_sample(seqs) if self.tp_params.local_rank == 0 else None
                logits = self.run_model(input_ids, positions, True)
                sample_tokens = (
                    self.sampler(logits, temperatures)
                    if self.tp_params.local_rank == 0
                    else torch.zeros(len(seqs), dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
                )
                dist.broadcast(sample_tokens, src=self.tp_params.master_rank, group=self.group)
                token_ids = sample_tokens.tolist()
                reset_context(self.tp_params)
                self.scheduler.postprocess(seqs, token_ids)
                return

            input_ids, positions = self.prepare_pearl_decode(seqs)
            logits = self.run_model(input_ids, positions, False)

            # Collect logits for SLO scheduler (corresponds to AdaServe ssm_inference_result.probs)
            draft_logits_steps.append(logits.clone())

            # Greedy sampling (temperature=0 for draft, same as original)
            sample_tokens = (logits.argmax(dim=-1) if self.tp_params.local_rank == 0
                            else torch.zeros(len(seqs), dtype=torch.int64, pin_memory=True).cuda(non_blocking=True))
            dist.broadcast(sample_tokens, src=self.tp_params.master_rank, group=self.group)
            token_ids = sample_tokens.tolist()
            draft_token_ids_steps.append(token_ids)
            reset_context(self.tp_params)

            # Append tokens to sequences (don't use postprocess to avoid early EOS exit)
            for seq, token_id in zip(seqs, token_ids):
                seq.append_token(token_id)

        # ===== Phase B: SLO budget allocation =====
        slo_seqs = self._get_slo_seqs(seqs)

        # 1. Compute accumulated probabilities from draft logits
        per_seq_token_infos = self.slo_scheduler.compute_draft_token_probs(
            draft_logits_steps, draft_token_ids_steps
        )

        # 2. Determine total budget
        total_budget = self._slo_total_budget if self._slo_total_budget > 0 else self.max_gamma * len(seqs)

        # 3. Get latency estimates (use profiled values or defaults)
        baseline_ms = getattr(self, 'baseline_latency_ms', 30.0)
        draft_step_ms = getattr(self, 'draft_step_latency_ms', 5.0)
        verify_step_ms = getattr(self, 'verify_step_latency_ms', 25.0)
        # batch_latency_ms scales with batch size (approximation)
        batch_ms = verify_step_ms * max(1.0, len(seqs) / 8.0)

        # 4. Allocate budget
        gamma_map = self.slo_scheduler.allocate_budget(
            slo_seqs, total_budget, baseline_ms,
            draft_step_ms, verify_step_ms, batch_ms,
            per_seq_token_infos,
        )

        # Store gamma_map for verify
        self._current_gamma_map = gamma_map

        # Update SLOSequence assigned_gamma
        for slo_seq in slo_seqs:
            slo_seq.assigned_gamma = gamma_map.get(slo_seq.seq_id, 1)

        # 5. Rollback excess tokens
        for seq in seqs:
            g = gamma_map.get(seq.seq_id, 1)
            excess = self.max_gamma - g
            if excess > 0:
                self.scheduler.rollback(seq, excess)

        # ===== Phase C: Verify with per-seq gamma =====
        self.verify(seqs, gamma_map)

    def prepare_pearl_decode(self, seqs):
        """Same as original prepare_decode."""
        return super().prepare_decode(seqs)

    @torch.inference_mode()
    def verify(self, seqs, gamma_map):
        """Two-broadcast verify protocol for per-seq gamma.

        Broadcast 1: gamma_tensor (num_seqs ints) -> target
        Broadcast 2: [to_be_verified_tokens | next_round_input] -> target
        Receive: verify_res (4, num_seqs) from target

        Corresponds to original DraftModelRunner.verify() but with per-seq gamma.
        """
        num_seqs = len(seqs)

        if self.tp_params.local_rank == 0:
            to_be_verified_tokens, next_round_input = build_verify_payload(seqs, gamma_map)

            # Broadcast 1: gamma_tensor
            gamma_tensor = torch.tensor(
                [gamma_map.get(s.seq_id, 1) for s in seqs],
                dtype=torch.int64, device="cuda",
            )
            dist.broadcast(gamma_tensor, src=self.rank, group=self.verify_group)

            # Broadcast 2: variable-length token message
            msg = torch.tensor(
                to_be_verified_tokens + next_round_input,
                dtype=torch.int64, device="cuda",
            )
            dist.broadcast(msg, src=self.rank, group=self.verify_group)

        else:
            # Non-master draft ranks still participate in broadcast
            gamma_tensor = torch.zeros(num_seqs, dtype=torch.int64, device="cuda")
            dist.broadcast(gamma_tensor, src=self.tp_params.master_rank, group=self.verify_group)

            # Compute message size from the received gamma map.
            received_gamma_map = {seq.seq_id: int(g) for seq, g in zip(seqs, gamma_tensor.tolist())}
            msg_size = count_verify_tokens(seqs, received_gamma_map) + count_next_round_tokens(seqs, received_gamma_map)
            msg = torch.zeros(msg_size, dtype=torch.int64, device="cuda")
            dist.broadcast(msg, src=self.tp_params.master_rank, group=self.verify_group)

        # Receive verification results from target
        verify_res = torch.zeros((4, num_seqs), dtype=torch.int64, device="cuda")
        dist.broadcast(verify_res, src=self.global_config.target_config.master_rank)

        # Process verification results
        acc, rollout, revise_token, finish = verify_res.tolist()
        for idx, seq in enumerate(seqs):
            g = gamma_map.get(seq.seq_id, 1)
            if finish[idx]:
                seq.status = SequenceStatus.FINISHED
                self.scheduler.block_manager.deallocate(seq)
                self.scheduler.running.remove(seq)
                self.scheduler.finished.append(seq)
                continue

            if seq.pre_verify:
                if acc[idx]:
                    seq.pre_verify = False
                else:
                    seq.pre_verify = True
                    self.scheduler.rollback(seq, g)
                    seq.append_token(revise_token[idx])
            else:
                if acc[idx]:
                    seq.pre_verify = False
                else:
                    seq.pre_verify = True
                    self.scheduler.rollback(seq, g)
                    if rollout[idx] > 1:
                        self.scheduler.rollback(seq, rollout[idx] - 1)
                    seq.append_token(revise_token[idx])

    def auto_set_gamma(self):
        """Override: run original auto_set_gamma and extract SLO latency metrics.

        Sets baseline_latency_ms = 1000 / target_speed[bs=1]
        Sets draft_step_latency_ms = 1000 / draft_speed[bs=current_batch]
        """
        super().auto_set_gamma()

        # Extract latency metrics from profiled speeds
        # These are available on all ranks after all_reduce
        if hasattr(self, 'gamma_list') and self.gamma_list:
            # baseline_latency_ms = 1000 / target_speed[bs=1]
            # target_speed[bs=1] ≈ 1 / (time_per_step at bs=1)
            # From gamma_list: gamma = round(draft_speed / target_speed)
            # We need actual speeds — they're in auto_set_gamma's local vars.
            # Since auto_set_gamma is already done, we use default estimates.
            pass

        # Store defaults; will be overwritten by controller with profiled values
        self.baseline_latency_ms = getattr(self, '_slo_baseline_latency_ms', 30.0)
        if self.baseline_latency_ms < 0:
            self.baseline_latency_ms = 30.0
        self.draft_step_latency_ms = getattr(self, '_slo_draft_step_latency_ms', 5.0)
        if self.draft_step_latency_ms < 0:
            self.draft_step_latency_ms = 5.0
        self.verify_step_latency_ms = getattr(self, '_slo_verify_step_latency_ms', 25.0)
        if self.verify_step_latency_ms < 0:
            self.verify_step_latency_ms = 25.0

    def slo_generate(self):
        """SLO-aware PEARL generation."""
        dist.barrier()
        torch.cuda.synchronize()
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)

        import time as _time
        start_time.record()
        wall_start = _time.time()

        self.prefill()

        while not self.scheduler.is_finished():
            self.pearl_step()

        end_time.record()
        torch.cuda.synchronize()
        wall_end = _time.time()

        seqs = self.scheduler.finished
        output = [
            (seq.seq_id, seq.completion_token_ids, seq.num_acc_tokens)
            for seq in seqs
        ]

        if self.rank == self.global_config.target_config.master_rank:
            data = pickle.dumps([output, wall_end - wall_start])
            n = len(data)
            self.shm.buf[0:4] = n.to_bytes(4, "little")
            self.shm.buf[4 : n + 4] = data

        self.clear_requests()

    def slo_bench_generate(self, num_pearl_steps=100):
        """Benchmark: fixed steps, collect per-seq metrics."""
        dist.barrier()
        torch.cuda.synchronize()
        import time as _time
        start_time = _time.time()

        self.prefill()

        for seq in self.scheduler.running:
            seq.max_tokens = int(1e8)
            seq.ignore_eos = True
            # Mark decode start for SLO tracking
            if seq.seq_id in self._slo_seqs:
                self._slo_seqs[seq.seq_id].mark_decode_start()

        for _ in range(num_pearl_steps):
            self.pearl_step()

        torch.cuda.synchronize()
        end_time = _time.time()

        seqs = self.scheduler.running
        for seq in seqs:
            seq.num_acc_tokens.append(seq.cur_acc_tokens)

        output = [
            (seq.seq_id, seq.completion_token_ids, seq.num_acc_tokens)
            for seq in seqs
        ]

        if self.rank == self.global_config.target_config.master_rank:
            data = pickle.dumps([output, end_time - start_time])
            n = len(data)
            self.shm.buf[0:4] = n.to_bytes(4, "little")
            self.shm.buf[4 : n + 4] = data

        self.clear_requests()

    def clear_requests(self):
        """Clear both scheduler and SLO tracking."""
        self._slo_seqs.clear()
        self.scheduler.clear()
        dist.barrier()

    def profile_slo_latency(self):
        """Profile and store latency metrics for SLO budget calculation.

        Runs auto_set_gamma() and extracts:
          - baseline_latency_ms = 1000 / target_speed[bs=1]
          - draft_step_latency_ms = 1000 / draft_speed[bs=1]
        """
        super().auto_set_gamma()

        # Reconstruct speeds from gamma_list
        # auto_set_gamma stores gamma_list but not raw speeds.
        # For accurate latency, use the timing from the profiling run.
        # Fallback: estimate from gamma values
        if hasattr(self, 'gamma_list') and self.rank == 0:
            logger.info(f"[SLODraftRunner] Profiled gamma_list: {self.gamma_list}")

        # Default latency estimates (will be refined)
        self.baseline_latency_ms = getattr(self, '_slo_baseline_latency_ms', 30.0)
        if self.baseline_latency_ms < 0:
            self.baseline_latency_ms = 30.0

        self.draft_step_latency_ms = getattr(self, '_slo_draft_step_latency_ms', 5.0)
        if self.draft_step_latency_ms < 0:
            self.draft_step_latency_ms = 5.0

        self.verify_step_latency_ms = getattr(self, '_slo_verify_step_latency_ms', 25.0)
        if self.verify_step_latency_ms < 0:
            self.verify_step_latency_ms = 25.0

        if self.rank == 0:
            logger.info(f"[SLODraftRunner] Latency: baseline={self.baseline_latency_ms:.1f}ms, "
                        f"draft_step={self.draft_step_latency_ms:.1f}ms, "
                        f"verify_step={self.verify_step_latency_ms:.1f}ms")

    # === Double Buffering Logic: New Functions Added for Step 2 ===
    def draft_batch_double_buffer(self, seqs):
        """
        [Double Buffering] 仅执行起草逻辑，不进行通信。
        该函数对应“先起草再剪枝”策略中的“全量起草”阶段。
        """
        draft_logits_steps = []
        draft_token_ids_steps = []

        # 无论 SLO 如何，先统一为 batch 内所有请求起草 max_gamma 步
        for step in range(self.max_gamma):
            for seq in seqs:
                while not self.scheduler.block_manager.can_append(seq):
                    self.scheduler.preempt(self.scheduler.running[-1])
                self.scheduler.block_manager.may_append(seq)

            input_ids, positions = self.prepare_pearl_decode(seqs)
            logits = self.run_model(input_ids, positions, False)
            
            # 克隆 logits 以便后续 slo_scheduler 计算每个 token 的接受概率
            draft_logits_steps.append(logits.clone())
            
            # 执行起草采样（通常为 Greedy）
            sample_tokens = (logits.argmax(dim=-1) if self.tp_params.local_rank == 0 
                            else torch.zeros(len(seqs), dtype=torch.int64, pin_memory=True).cuda(non_blocking=True))
            if self.tp_params.tp_size > 1:
                dist.broadcast(sample_tokens, src=self.tp_params.master_rank, group=self.group)
            token_ids = sample_tokens.tolist()
            draft_token_ids_steps.append(token_ids)
            reset_context(self.tp_params)
            
            # 将起草的 token 临时附加到序列对象中
            for seq, token_id in zip(seqs, token_ids):
                seq.append_token(token_id)
        
        return draft_logits_steps, draft_token_ids_steps

    def verify_double_buffer_send(self, seqs, draft_logits_steps, draft_token_ids_steps):
        """
        [Double Buffering] 执行剪枝计算，并将验证数据发送给 Target。
        该函数实现了“先起草再剪枝”策略中的“根据 SLO 分配并回滚”以及“变长验证数据外发”。
        """
        slo_seqs = self._get_slo_seqs(seqs)
        
        # 计算每个 token 在 draft 模型下的累积概率
        per_seq_token_infos = self.slo_scheduler.compute_draft_token_probs(
            draft_logits_steps, draft_token_ids_steps
        )
        
        # 使用第一步实现的 SLO 分配算法计算每个序列专属的 gamma
        total_budget = self._slo_total_budget if self._slo_total_budget > 0 else self.max_gamma * len(seqs)
        
        # Get latency estimates (use profiled values or defaults)
        baseline_ms = getattr(self, 'baseline_latency_ms', 30.0)
        draft_step_ms = getattr(self, 'draft_step_latency_ms', 5.0)
        verify_step_ms = getattr(self, 'verify_step_latency_ms', 25.0)
        # batch_latency_ms scales with batch size (approximation)
        batch_ms = verify_step_ms * max(1.0, len(seqs) / 8.0)

        # Allocate budget
        gamma_map = self.slo_scheduler.allocate_budget(
            slo_seqs, total_budget, baseline_ms,
            draft_step_ms, verify_step_ms, batch_ms,
            per_seq_token_infos,
        )

        # Store gamma_map for verify
        self._current_gamma_map = gamma_map

        # Update SLOSequence assigned_gamma
        for slo_seq in slo_seqs:
            slo_seq.assigned_gamma = gamma_map.get(slo_seq.seq_id, 1)

        # 执行“剪枝”：根据 gamma_map 回滚超出分配深度的多余 Token
        for seq in seqs:
            g = gamma_map.get(seq.seq_id, 1)
            excess = self.max_gamma - g
            if excess > 0:
                self.scheduler.rollback(seq, excess)

        # 按照“两次广播协议”发送数据到 Target
        if self.tp_params.local_rank == 0:
            to_be_verified_tokens, next_round_input = build_verify_payload(seqs, gamma_map)

            # 广播 1: 发送每个请求不同的 gamma 值
            gamma_tensor = torch.tensor([gamma_map.get(s.seq_id, 1) for s in seqs], dtype=torch.int64, device="cuda")
            dist.broadcast(gamma_tensor, src=self.rank, group=self.verify_group)
        
            # 广播 2: 发送待验证的 token 序列
            msg = torch.tensor(to_be_verified_tokens + next_round_input, dtype=torch.int64, device="cuda")
            dist.broadcast(msg, src=self.rank, group=self.verify_group)
        else:
            # 辅助 Rank 配合 NCCL 同步点
            gamma_tensor = torch.zeros(len(seqs), dtype=torch.int64, device="cuda")
            dist.broadcast(gamma_tensor, src=self.tp_params.master_rank, group=self.verify_group)
            received_gamma_map = {seq.seq_id: int(g) for seq, g in zip(seqs, gamma_tensor.tolist())}
            msg_size = count_verify_tokens(seqs, received_gamma_map) + count_next_round_tokens(seqs, received_gamma_map)
            msg = torch.zeros(msg_size, dtype=torch.int64, device="cuda")
            dist.broadcast(msg, src=self.tp_params.master_rank, group=self.verify_group)
        
        return gamma_map

    def verify_double_buffer_recv(self, seqs, gamma_map):
        """
        [Double Buffering] 接收验证结果。
        此函数通常在下一个 Batch 起草完成后调用，从而实现计算重叠。
        """
        num_seqs = len(seqs)
        verify_res = torch.zeros((4, num_seqs), dtype=torch.int64, device="cuda")
        
        # 阻塞在此，直到 Target 模型完成验证并广播结果
        dist.broadcast(verify_res, src=self.global_config.target_config.master_rank)

        # 根据验证结果（acc/rollout/finish等）更新本地序列状态
        acc, rollout, revise_token, finish = verify_res.tolist()
        for idx, seq in enumerate(seqs):
            g = gamma_map.get(seq.seq_id, 1)
            if finish[idx]:
                seq.status = SequenceStatus.FINISHED
                self.scheduler.block_manager.deallocate(seq)
                self.scheduler.running.remove(seq)
                self.scheduler.finished.append(seq)
                continue

            if seq.pre_verify:
                if acc[idx]:
                    seq.pre_verify = False
                else:
                    seq.pre_verify = True
                    self.scheduler.rollback(seq, g)
                    seq.append_token(revise_token[idx])
            else:
                if acc[idx]:
                    seq.pre_verify = False
                else:
                    seq.pre_verify = True
                    self.scheduler.rollback(seq, g)
                    if rollout[idx] > 1:
                        self.scheduler.rollback(seq, rollout[idx] - 1)
                    seq.append_token(revise_token[idx])

    def slo_generate_double_buffer(self):
        """
        [Double Buffering] Draft 端流水线主循环。
        实现：Prime Batch 0 -> Loop (Draft B1 & RecvVerify B0)
        """
        dist.barrier()
        torch.cuda.synchronize()
        
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        import time as _time
        start_time.record()
        wall_start = _time.time()
        
        # 0. 准备工作：同步执行 Prefill
        self.prefill()

        # 1. 划分初始 Batch (通过 Phase 1 实现的函数)
        batch_0, batch_1 = self.scheduler.schedule_double_buffer()

        # 安全回退：如果请求太少无法分两批，退回到单批次模式
        if not batch_0 or not batch_1:
            while not self.scheduler.is_finished():
                self.pearl_step() # 调用原有的同步 pearl_step
            end_time.record()
            torch.cuda.synchronize()
            wall_end = _time.time()
            self._write_output_to_shm(wall_start, wall_end)
            self.clear_requests()
            return

        # 2. Prime 阶段：起草 Batch 0 并外发验证数据
        logits_0, ids_0 = self.draft_batch_double_buffer(batch_0)
        g_map_0 = self.verify_double_buffer_send(batch_0, logits_0, ids_0)

        # 初始状态指针
        curr_draft_batch = batch_1
        curr_verify_batch = batch_0
        curr_verify_g_map = g_map_0

        # 3. Loop 阶段：核心流水线
        while not self.scheduler.is_finished():
            print(f"--- [Draft Side] Starting Draft Loop for Batch {curr_draft_batch[0].seq_id if curr_draft_batch else 'N/A'} ---")
            # A. 并行点：起草下一批 (此时 Target 正在并行验证上一批)
            l_next, i_next = self.draft_batch_double_buffer(curr_draft_batch)

            print(f"--- [Draft Side] Finished Drafting B1. Now waiting for B0 verification... ---")
            # B. 同步点：等待并接收上一批的验证结果
            # 如果此时验证还没完，Draft 会阻塞在此处
            self.verify_double_buffer_recv(curr_verify_batch, curr_verify_g_map)

            print(f"--- [Draft Side] Received B0 results. Updating status... ---")
            # C. 发送点：将刚刚起草完的数据发给 Target
            g_map_next = self.verify_double_buffer_send(curr_draft_batch, l_next, i_next)

            # D. 指针交换：轮换 Batch
            curr_draft_batch, curr_verify_batch = curr_verify_batch, curr_draft_batch
            curr_verify_g_map = g_map_next

        #e2
        self.verify_double_buffer_recv(curr_verify_batch, curr_verify_g_map)

        # 4. 结果输出
        end_time.record()
        torch.cuda.synchronize()
        wall_end = _time.time()
        self._write_output_to_shm(wall_start, wall_end)
        self.clear_requests()

    def slo_bench_generate_double_buffer(self, num_pearl_steps=100):
        """
        [Double Buffering] 用于 Benchmark 的 Draft 端主循环。
        逐行解释：
        1. 初始化计时器。
        2. 同步执行 Prefill。
        3. 强制设置所有序列不因 EOS 停止，并记录解码开始时间。
        4. 划分 Batch。
        5. Prime 阶段：起草 Batch 0。
        6. 循环执行 num_pearl_steps 步，实现计算重叠。
        """
        dist.barrier()
        torch.cuda.synchronize()
        import time as _time
        start_time = _time.time()

        self.prefill()

        # Benchmark 特有设置：忽略 EOS，记录开始时间
        for seq in self.scheduler.running:
            seq.max_tokens = int(1e8)
            seq.ignore_eos = True
            if seq.seq_id in self._slo_seqs:
                self._slo_seqs[seq.seq_id].mark_decode_start()

        # 划分初始 Batch
        batch_0, batch_1 = self.scheduler.schedule_double_buffer()
         
        # 如果无法分两批，退回到单批次同步模式
        if not batch_0 or not batch_1:
            for _ in range(num_pearl_steps):
                self.pearl_step()
            torch.cuda.synchronize()
            wall_end = _time.time()
            self._write_output_to_shm(start_time, wall_end)
            self.clear_requests()
            return

        # Prime 阶段
        l_0, i_0 = self.draft_batch_double_buffer(batch_0)
        g_map_0 = self.verify_double_buffer_send(batch_0, l_0, i_0)

        curr_draft_batch = batch_1
        curr_verify_batch = batch_0
        curr_verify_g_map = g_map_0

        # Loop 阶段：固定步数循环
        for _ in range(num_pearl_steps):
            print(f"--- [Draft Side] Starting Draft Loop for Batch {curr_draft_batch[0].seq_id if curr_draft_batch else 'N/A'} ---")
            # A. 异步起草 (与 Target 并行)
            l_next, i_next = self.draft_batch_double_buffer(curr_draft_batch)

            print(f"--- [Draft Side] Finished Drafting B1. Now waiting for B0 verification... ---")
            # B. 接收验证结果 (同步点)
            self.verify_double_buffer_recv(curr_verify_batch, curr_verify_g_map)
            
            print(f"--- [Draft Side] Received B0 results. Updating status... ---")
            # C. 发送新验证请求
            g_map_next = self.verify_double_buffer_send(curr_draft_batch, l_next, i_next)
            # D. 指针轮换
            curr_draft_batch, curr_verify_batch = curr_verify_batch, curr_draft_batch
            curr_verify_g_map = g_map_next
        
        #e1
        self.verify_double_buffer_recv(curr_verify_batch, curr_verify_g_map)

        # 4. 完成后收集结果
        torch.cuda.synchronize()
        end_time = _time.time()
         
        # 将最后一步的接受情况计入结果
        for seq in self.scheduler.running:
            seq.num_acc_tokens.append(seq.cur_acc_tokens)

        self._write_output_to_shm(start_time, end_time, use_running=True)
        self.clear_requests()

    #e3
    def _write_output_to_shm(self, wall_start, wall_end, use_running=False):
        seqs = self.scheduler.running if use_running else self.scheduler.finished
        output = [(s.seq_id, s.completion_token_ids, s.num_acc_tokens) for s in seqs]
        if self.rank == self.global_config.target_config.master_rank:
            data = pickle.dumps([output, wall_end - wall_start])
            n = len(data)
            self.shm.buf[0:4] = n.to_bytes(4, "little")
            self.shm.buf[4 : n + 4] = data