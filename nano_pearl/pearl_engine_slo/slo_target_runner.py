"""
SLOTargetRunner — target model runner with per-sequence gamma verification.

Key difference from original TargetModelRunner:
  1. prepare_pearl_decode uses per-seq gamma from gamma_map
  2. verify receives two broadcasts (gamma_map + variable-length tokens)
  3. Verification uses per-seq gamma for token checking

Uses two-broadcast protocol:
  Broadcast 1: gamma_tensor (num_seqs ints) from draft
  Broadcast 2: [to_be_verified_tokens | next_round_input] from draft
  Broadcast 3: verify_res (4, num_seqs) from target to draft
"""

import pickle
import torch
import torch.distributed as dist
from nano_pearl.pearl_engine.pearl_model_runner import ModelRunnerBase
from nano_pearl.pearl_engine.sequence import SequenceStatus
from nano_pearl.pearl_engine.scheduler import is_eos
from nano_pearl.pearl_engine_slo.slo_sequence import SLOSequence
from nano_pearl.layers.sampler import norm_logits
from nano_pearl.utils.context import reset_context, set_context
from nano_pearl.utils.pearl_logger import logger


class SLOTargetRunner(ModelRunnerBase):
    """Target model runner with per-sequence gamma verification.

    Overrides prepare_pearl_decode(), pearl_step(), and verify().
    """

    def __init__(self, config, rank, event, control_event):
        # Track per-seq gamma map (received from draft via broadcast)
        self._current_gamma_map = {}
        # Track SLOSequences for latency updates
        self._slo_seqs = {}  # seq_id -> SLOSequence

        super().__init__(config, rank, event, control_event)

        if self.rank == 0:
            logger.info("[SLOTargetRunner] Initialized with per-seq gamma verification.", color="green")

    def add_request(self, seq):
        """Override to track SLOSequence."""
        if isinstance(seq, SLOSequence):
            slo_seq = seq
        else:
            slo_seq = SLOSequence(seq)
        self._slo_seqs[seq.seq_id] = slo_seq
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

    def prepare_pearl_decode(self, seqs, gamma_map):
        """Prepare decode inputs with per-seq gamma.

        Same packing approach as original, but num_tokens per seq = gamma_map[seq_id].

        For a sequence in pre-verify: input tokens = 1 (the last token)
        For a sequence in post-verify: input tokens = gamma_map[seq_id]
        """
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        temp_seqs = []

        for seq in seqs:
            g = gamma_map.get(seq.seq_id, 1)
            num_tokens = g if not seq.pre_verify else 1
            to_append_tokens = seq.token_ids[-num_tokens:]
            input_ids.extend(to_append_tokens)
            positions.extend(list(range(len(seq) - num_tokens, len(seq))))
            context_lens.extend(list(range(len(seq) - num_tokens + 1, len(seq) + 1)))
            slot_mapping.extend([
                seq.token_to_slot(token_index)
                for token_index in range(len(seq) - num_tokens, len(seq))
            ])
            temp_seqs.extend([seq] * num_tokens)

        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(temp_seqs)
        set_context(self.tp_params, False, slot_mapping=slot_mapping,
                    context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions, temp_seqs

    def pearl_step(self):
        """Target-side PEARL step: receive gamma_map, prepare decode, verify."""
        seqs, is_prefill = self.scheduler.schedule()
        assert not is_prefill, "wrong match. current stage is prefill."

        # Receive gamma_map from draft (Broadcast 1)
        num_seqs = len(seqs)
        gamma_tensor = torch.zeros(num_seqs, dtype=torch.int64, device="cuda")
        dist.broadcast(gamma_tensor, src=self.global_config.draft_config.master_rank,
                       group=self.verify_group)
        gamma_map = {seq.seq_id: int(g) for seq, g in zip(seqs, gamma_tensor.tolist())}
        self._current_gamma_map = gamma_map

        # Prepare decode with per-seq gamma
        input_ids, positions, temp_seqs = self.prepare_pearl_decode(seqs, gamma_map)
        temperatures = (self.prepare_sample(temp_seqs) if self.tp_params.local_rank == 0
                        else None)
        logits = self.run_model(input_ids, positions, is_prefill)
        self.verify(logits, seqs, temperatures, gamma_map)

    @torch.inference_mode()
    def verify(self, logits, seqs, temperatures, gamma_map):
        """Per-seq gamma verification with two-broadcast protocol.

        Broadcast 2 (receive): [to_be_verified_tokens | next_round_input] from draft
        Broadcast 3 (send): verify_res (4, num_seqs) to draft
        """
        # Compute expected message sizes from gamma_map
        num_to_be_verified_tokens = sum(
            1 if seq.pre_verify else gamma_map.get(seq.seq_id, 1)
            for seq in seqs
        )
        num_next_round_input = sum(gamma_map.get(seq.seq_id, 1) for seq in seqs)

        # Broadcast 2: receive tokens from draft
        msg = torch.zeros(num_to_be_verified_tokens + num_next_round_input,
                          dtype=torch.int64, device="cuda")
        dist.broadcast(msg, src=self.global_config.draft_config.master_rank,
                       group=self.verify_group)

        to_be_verified_tokens = msg[:num_to_be_verified_tokens].tolist()
        next_round_input = msg[num_to_be_verified_tokens:].tolist()

        verify_res = torch.zeros((4, len(seqs)), dtype=torch.int64, device="cuda")

        if self.tp_params.local_rank == 0:
            r = torch.rand(num_to_be_verified_tokens, device="cuda")
            target_logits = norm_logits(logits, temperatures)
            target_prob = target_logits.gather(
                dim=1, index=msg[:num_to_be_verified_tokens].unsqueeze(1)
            ).squeeze(1)
            judge = (r <= target_prob).tolist()

            logits.scatter_(1, msg[:num_to_be_verified_tokens].unsqueeze(1), -float("inf"))
            revised_tokens = self.sampler(logits, temperatures)

            acc, rollout, revise_token, finish = [], [], [], []

            v_idx = 0
            for i, seq in enumerate(seqs):
                g = gamma_map.get(seq.seq_id, 1)

                if seq.pre_verify:
                    # Verify 1 token
                    acc.append(judge[v_idx])
                    rollout.append(0 if judge[v_idx] else g)
                    revise_token.append(revised_tokens[v_idx])

                    if judge[v_idx]:
                        seq.cur_acc_tokens += 1
                        finish.append(
                            (not seq.ignore_eos and is_eos(to_be_verified_tokens[v_idx], self.scheduler.eos))
                            or seq.num_completion_tokens >= seq.max_tokens - 1
                        )
                    else:
                        seq.num_acc_tokens.append(seq.cur_acc_tokens + 1)
                        seq.cur_acc_tokens = 0
                        finish.append(
                            (not seq.ignore_eos and is_eos(revise_token[-1], self.scheduler.eos))
                            or seq.num_completion_tokens >= seq.max_tokens - 1
                        )
                    v_idx += 1
                else:
                    # Verify g tokens
                    finish_flag = False
                    n = g  # number of accepted tokens
                    for j in range(v_idx, v_idx + g):
                        if not seq.ignore_eos and judge[j] and is_eos(
                            to_be_verified_tokens[j], self.scheduler.eos
                        ):
                            finish_flag = True
                        if not judge[j]:
                            n = j - v_idx
                            break

                    acc.append(n == g)
                    rollout.append(g - n)
                    revise_token.append(revised_tokens[n + v_idx] if n < g else -1)
                    finish.append(
                        finish_flag
                        or seq.num_completion_tokens >= seq.max_tokens - min(n + 1, g)
                    )

                    if n == g:
                        seq.cur_acc_tokens += n
                    else:
                        seq.num_acc_tokens.append(seq.cur_acc_tokens + n + 1)
                        seq.cur_acc_tokens = 0

                    v_idx += g

            verify_res = torch.tensor(
                [acc, rollout, revise_token, finish],
                dtype=torch.int64, device="cuda",
            )

        # Broadcast 3: send verify_res to draft
        dist.broadcast(verify_res, src=self.global_config.target_config.master_rank)

        # Post-process: update sequences on target side
        acc, rollout, revise_token, finish = verify_res.tolist()

        for idx, seq in enumerate(seqs):
            g = gamma_map.get(seq.seq_id, 1)

            if seq.pre_verify:
                if acc[idx]:
                    seq.pre_verify = False
                    for token in next_round_input[
                        sum(gamma_map.get(s.seq_id, 1) for s in seqs[:idx]) :
                        sum(gamma_map.get(s.seq_id, 1) for s in seqs[: idx + 1])
                    ]:
                        seq.append_token(token)
                else:
                    seq.pre_verify = True
                    seq.append_token(revise_token[idx])
            else:
                if acc[idx]:
                    seq.pre_verify = False
                    for token in next_round_input[
                        sum(gamma_map.get(s.seq_id, 1) for s in seqs[:idx]) :
                        sum(gamma_map.get(s.seq_id, 1) for s in seqs[: idx + 1])
                    ]:
                        seq.append_token(token)
                else:
                    seq.pre_verify = True
                    if rollout[idx] > 1:
                        self.scheduler.rollback(seq, rollout[idx] - 1)
                    seq.append_token(revise_token[idx])

            if finish[idx]:
                seq.status = SequenceStatus.FINISHED
                seq.num_acc_tokens.append(seq.cur_acc_tokens)
                self.scheduler.block_manager.deallocate(seq)
                self.scheduler.running.remove(seq)
                self.scheduler.finished.append(seq)

    def slo_generate(self):
        """SLO-aware generation on target side."""
        dist.barrier()
        torch.cuda.synchronize()
        import time as _time
        start_time = _time.time()

        self.prefill()

        while not self.scheduler.is_finished():
            self.pearl_step()

        torch.cuda.synchronize()
        end_time = _time.time()

        seqs = self.scheduler.finished
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

    def slo_bench_generate(self, num_pearl_steps=100):
        """Benchmark on target side."""
        dist.barrier()
        torch.cuda.synchronize()
        import time as _time
        start_time = _time.time()

        self.prefill()

        for seq in self.scheduler.running:
            seq.max_tokens = int(1e8)
            seq.ignore_eos = True

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
        self._slo_seqs.clear()
        self.scheduler.clear()
        dist.barrier()
