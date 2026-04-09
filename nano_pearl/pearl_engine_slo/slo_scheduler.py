"""
SLOScheduler — SLO-aware per-sequence gamma allocator.

完整复刻 AdaServe RequestManager 中的 budget 分配机制:
  prune_token_tree → add_tokens_toward_slo → add_tokens_toward_goodput

模块化函数对应关系:
  add_tokens_to_spec_token_tree  → compute_draft_token_probs
  prune_token_tree (计算部分)     → compute_num_tokens_to_decode
  add_tokens_toward_slo          → allocate_budget_toward_slo
  add_tokens_toward_goodput      → allocate_budget_toward_goodput
  prune_token_tree (主入口)      → allocate_budget
"""

import math
import heapq
import torch
import torch.nn.functional as F
from typing import Optional


class DraftTokenInfo:
    """存储单个 draft token 的概率信息。

    对应 AdaServe 中 TokenTreeNode 的部分字段。
    线性序列中每个 token 等价于树中一条单路径上的节点。

    Fields:
        token_id: 选出的 token id
        log_prob: log P_draft(token | prefix)
        log_accumulated_prob: 从起始到本 token 的累积 log 概率
    """

    __slots__ = ('token_id', 'log_prob', 'log_accumulated_prob')

    def __init__(self, token_id, log_prob, log_accumulated_prob):
        self.token_id = token_id
        self.log_prob = log_prob
        self.log_accumulated_prob = log_accumulated_prob

    def __repr__(self):
        return (f"DraftTokenInfo(id={self.token_id}, "
                f"log_p={self.log_prob:.4f}, "
                f"log_acc={self.log_accumulated_prob:.4f})")


class SLOScheduler:
    """SLO-aware per-sequence gamma allocator.

    仿照 AdaServe RequestManager 中的 prune_token_tree / add_tokens_toward_slo /
    add_tokens_toward_goodput 三阶段分配。
    """

    def __init__(self, config):
        self.min_gamma = config.min_gamma
        self.max_gamma = config.max_gamma
        self.correction_factor = config.correction_factor

    # ================================================================
    # 函数1: 对应 AdaServe add_tokens_to_spec_token_tree
    # ================================================================
    def compute_draft_token_probs(self, draft_logits_steps, draft_token_ids_steps):
        """从 draft 模型每步的 logits 计算每个 token 的累积 log 概率。

        对应 AdaServe 中 add_tokens_to_spec_token_tree() 对 SSM 推理结果的处理:
          log_prob = log(ssm_inference_result.probs[result_idx])
          accumulated_log_prob = log_prob + parent_log_prob

        线性序列版本: 每步选 argmax token, 取其 softmax 概率, 沿序列累积。

        Args:
            draft_logits_steps: 每步 draft 的原始 logits, list of (num_seqs, vocab)
            draft_token_ids_steps: 每步选择的 token id, list of list[int]

        Returns:
            per_seq_token_infos: list[list[DraftTokenInfo]], 每个请求每步的 token 概率信息
        """
        num_seqs = draft_logits_steps[0].shape[0]
        per_seq_token_infos = [[] for _ in range(num_seqs)]
        log_acc_probs = [0.0] * num_seqs

        for step_idx, (logits, token_ids) in enumerate(
            zip(draft_logits_steps, draft_token_ids_steps)
        ):
            # 对应 AdaServe: probs = softmax(logits), 取所选 token 的概率
            probs = F.softmax(logits, dim=-1)  # (num_seqs, vocab)

            for seq_idx in range(num_seqs):
                token_id = token_ids[seq_idx]
                # 对应 AdaServe: double log_prob = log(ssm_inference_result.probs[result_idx])
                token_prob = probs[seq_idx, token_id].item()
                log_prob = math.log(max(token_prob, 1e-10))

                # 对应 AdaServe: accumulated_log_prob = log_prob + parent_log_prob
                log_acc_probs[seq_idx] += log_prob

                info = DraftTokenInfo(
                    token_id=token_id,
                    log_prob=log_prob,
                    log_accumulated_prob=log_acc_probs[seq_idx],
                )
                per_seq_token_infos[seq_idx].append(info)

        return per_seq_token_infos

    # ================================================================
    # 函数2: 对应 AdaServe prune_token_tree 中的计算部分
    # ================================================================
    def compute_num_tokens_to_decode(self, slo_seqs, baseline_latency_ms,
                                     draft_step_latency_ms, verify_step_latency_ms,
                                     batch_latency_ms):
        """计算每个请求需要多少个 token 才能满足其 SLO。

        对应 AdaServe prune_token_tree() L4268-4301:
          num_tokens_to_decode_per_step = (ssm_spec_latency + batch_latency) / slo_constraint
          expected_num_tokens_decoded = decode_latency_ms / slo_constraint
          num_tokens_to_decode = max(1.0,
              (per_step + expected) * correction_factor - decode_length)

        Args:
            slo_seqs: SLOSequence 列表
            baseline_latency_ms: 单 token 基准延迟 (从 auto_set_gamma 获取)
            draft_step_latency_ms: draft 模型单步延迟
            verify_step_latency_ms: target 模型验证步延迟 (unused, kept for interface parity)
            batch_latency_ms: 当前 batch 大小对应的验证延迟

        Returns:
            [(num_tokens_to_decode, seq_index), ...] 按 num_tokens_to_decode 降序排列
        """
        results = []
        for idx, seq in enumerate(slo_seqs):
            # 对应 AdaServe: get_slo_constraint(request) = slo_ratio * baseline_latency_ms
            slo_constraint = seq.slo_ratio * baseline_latency_ms

            # 对应 AdaServe: (ssm_spec_latency_estimated + batch_latency_ms) / slo_constraint
            # 线性序列中 ssm_spec_latency = gamma * draft_step_latency
            num_tokens_to_decode_per_step = (
                (draft_step_latency_ms + batch_latency_ms) / slo_constraint
            )

            # 对应 AdaServe: request.decode_latency_ms / get_slo_constraint(request)
            expected_num_tokens_decoded = seq.decode_latency_ms / slo_constraint

            # 对应 AdaServe: max(1.0, (...) * correction_factor - decode_length)
            decode_length = max(getattr(seq, 'num_completion_tokens', 1), 1)
            num_tokens_to_decode = max(
                1.0,
                (num_tokens_to_decode_per_step + expected_num_tokens_decoded)
                * self.correction_factor - decode_length,
            )
            # 对应 AdaServe: min(num_tokens_to_decode, ssm_tree_depth + 1)
            num_tokens_to_decode = min(num_tokens_to_decode, self.max_gamma + 1)

            results.append((num_tokens_to_decode, idx))

        # 对应 AdaServe: sort by num_tokens_to_decode DESC
        results.sort(key=lambda x: x[0], reverse=True)
        return results

    # ================================================================
    # 函数3: 对应 AdaServe add_tokens_toward_slo
    # ================================================================
    def allocate_budget_toward_slo(self, seq_idx, budget, num_tokens_to_decode,
                                   num_req_with_slo, token_infos):
        """为单个请求分配 gamma, 以满足其 SLO 约束。

        对应 AdaServe add_tokens_toward_slo() L4540-4582:
          double current_added = 1.0;  // root already counted
          while (budget > 0 && current_added < num_tokens_to_decode) {
              auto [node, log_acc_prob] = pq.top();
              double prob = exp(log_acc_prob);
              if (prob < 0.04) break;
              node->included = true;
              current_added += prob;
              budget--;
          }

        线性序列适配: 按序列顺序累加每个 token 的累积概率 (exp(log_acc_prob)),
        直到期望接受数 >= num_tokens_to_decode。

        Args:
            seq_idx: 请求索引
            budget: 当前剩余 budget (int, will be decreased)
            num_tokens_to_decode: 需要达到的目标 token 数 (float)
            num_req_with_slo: 有 SLO 约束的请求数 (用于限制单请求最大分配)
            token_infos: 该请求的 DraftTokenInfo 列表

        Returns:
            (assigned_gamma, remaining_budget)
        """
        current_added = 1.0  # 对应 AdaServe: root 已计入
        assigned_gamma = 0

        # max_token_toward_slo: 对应 AdaServe 中限制单个请求最大分配
        max_token_toward_slo = int(budget * 1.2 / max(num_req_with_slo, 1))

        for info in token_infos:
            if budget <= 0 or max_token_toward_slo <= 0:
                break
            if current_added >= num_tokens_to_decode:
                break

            # 对应 AdaServe: double prob = exp(log_acc_prob)
            prob = math.exp(info.log_accumulated_prob)

            # 对应 AdaServe: if (prob < 4e-2) break
            if prob < 4e-2:
                break

            current_added += prob
            assigned_gamma += 1
            budget -= 1
            max_token_toward_slo -= 1

        return assigned_gamma, budget

    # ================================================================
    # 函数4: 对应 AdaServe add_tokens_toward_goodput
    # ================================================================
    def allocate_budget_toward_goodput(self, budget, slo_seqs,
                                       all_token_infos):
        """全局贪心分配剩余 budget 以最大化吞吐。

        对应 AdaServe add_tokens_toward_goodput() L4655-4723:
          全局 PQ, 每次弹出所有请求中概率最高的节点
          while (budget > 0):
              pop global max prob node
              prob = exp(acc_log_prob)
              if prob < 0.02: break
              include node, budget--

        线性序列适配:
          对每个请求, 计算其下一个 token 的概率 (基于已有 token 的累积概率)
          选概率最高的请求, 给它多分配一个 gamma
          重复直到 budget 耗尽或所有概率 < 2%

        Args:
            budget: 剩余可用 budget
            slo_seqs: SLOSequence 列表
            all_token_infos: 每个请求的 DraftTokenInfo 列表

        Returns:
            {seq_idx: extra_gamma} 超出 SLO 分配的额外 gamma
        """
        gamma_bonus = {}  # seq_idx -> extra gamma beyond SLO allocation

        # 用堆模拟 AdaServe 的全局 PQ
        # 每个条目: (-log_acc_prob, seq_idx)
        heap = []
        for idx, infos in enumerate(all_token_infos):
            if not infos:
                continue
            # 取最后一个 token 的累积概率 (代表该请求继续扩展的信心)
            last_prob = infos[-1].log_accumulated_prob
            heapq.heappush(heap, (-last_prob, idx))

        while budget > 0 and heap:
            neg_log_acc_prob, seq_idx = heapq.heappop(heap)
            log_acc_prob = -neg_log_acc_prob

            # 对应 AdaServe: prob = exp(acc_log_prob); if (prob < 2e-2 && budget % 32 == 0) break
            prob = math.exp(log_acc_prob)
            if prob < 2e-2 and budget % 32 == 0:
                break

            # 给这个请求多分配 1 个 gamma
            gamma_bonus[seq_idx] = gamma_bonus.get(seq_idx, 0) + 1
            budget -= 1

            # 对应 AdaServe: 从该请求取下一个候选入队
            # 概率衰减估计: 每多一个 token, 累积概率乘以平均 token 概率
            new_log_acc_prob = log_acc_prob + math.log(max(prob ** 0.1, 1e-10))
            heapq.heappush(heap, (-new_log_acc_prob, seq_idx))

        return gamma_bonus

    # ================================================================
    # 主入口: 对应 AdaServe prune_token_tree
    # ================================================================
    def allocate_budget(self, slo_seqs, total_budget, baseline_latency_ms,
                        draft_step_latency_ms, verify_step_latency_ms,
                        batch_latency_ms,
                        all_token_infos=None):
        """主分配函数, 对应 AdaServe prune_token_tree() L4246-4353。

        完整复刻 AdaServe 的三阶段流程:
          1. 每个请求至少分配 min_gamma (= root token, budget -= num_requests)
          2. Phase 1 (SLO): compute_num_tokens_to_decode -> allocate_budget_toward_slo
          3. Phase 2 (Goodput): allocate_budget_toward_goodput

        Args:
            slo_seqs: SLOSequence 列表
            total_budget: 总 draft token budget
            baseline_latency_ms: 单 token 基准延迟
            draft_step_latency_ms: draft 模型单步延迟
            verify_step_latency_ms: target 模型验证步延迟
            batch_latency_ms: 当前 batch 的验证延迟
            all_token_infos: 每个请求的 DraftTokenInfo 列表 (from compute_draft_token_probs)

        Returns:
            {seq.seq_id: assigned_gamma}
        """
        num_seqs = len(slo_seqs)
        if num_seqs == 0:
            return {}

        gamma_map = {}

        # Phase 0: 每个请求至少分配 min_gamma
        # 对应 AdaServe: budget -= (num_available_requests - prefilling_requests.size())
        remaining_budget = total_budget - num_seqs * self.min_gamma
        for seq in slo_seqs:
            gamma_map[seq.seq_id] = self.min_gamma

        if remaining_budget <= 0:
            return gamma_map

        # Phase 1: 计算每个请求需要多少 token 满足 SLO
        num_tokens_list = self.compute_num_tokens_to_decode(
            slo_seqs, baseline_latency_ms,
            draft_step_latency_ms, verify_step_latency_ms,
            batch_latency_ms,
        )

        # 对每个请求 (按紧迫度降序), 分配 SLO budget
        for num_tokens_to_decode, seq_idx in num_tokens_list:
            token_infos = all_token_infos[seq_idx] if all_token_infos else []

            extra_gamma, remaining_budget = self.allocate_budget_toward_slo(
                seq_idx, remaining_budget, num_tokens_to_decode,
                len(num_tokens_list), token_infos,
            )
            gamma_map[slo_seqs[seq_idx].seq_id] += extra_gamma

        # Phase 2: 全局贪心分配剩余 budget
        if remaining_budget > 0 and all_token_infos:
            gamma_bonus = self.allocate_budget_toward_goodput(
                remaining_budget, slo_seqs, all_token_infos,
            )
            for seq_idx, bonus in gamma_bonus.items():
                gamma_map[slo_seqs[seq_idx].seq_id] += bonus

        # Clamp to [min_gamma, max_gamma]
        for seq_id in gamma_map:
            gamma_map[seq_id] = max(
                self.min_gamma, min(self.max_gamma, gamma_map[seq_id])
            )

        return gamma_map
