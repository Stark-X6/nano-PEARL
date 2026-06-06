# Slopearl Formal Fix Design

## Goal

将 `slopearl` 从当前“外层按 workload batch 重建 engine”的 workaround 恢复成正式实验可用版本：单个 `SLOPearlEngine` 处理完整 workload，双缓冲循环在 engine 内部稳定运行，不再依赖 harness 层重建进程组来规避状态错误。

## Current Problems

当前 `slopearl` 存在两类问题：

1. `benchmark_slo/run_workload.py` 为了绕开运行时错误，把 `slopearl` 标成 `supports_batch_reuse=False`，导致每个 workload batch 都重新构造 engine，反复加载模型，性能不可接受，也不是正式实验语义。
2. `slo_bench_generate_double_buffer()` / `slo_generate_double_buffer()` 使用静态 `batch_0` / `batch_1` 列表驱动整个 benchmark 生命周期，而真实 scheduler 的 `running` / `waiting` / `finished` 状态会在运行中变化。静态 batch 快照与真实调度状态漂移后，会表现为：
   - draft / target 双缓冲两侧处理的序列集合不再一致
   - block / rollback / preempt 后的序列状态与双缓冲指针不一致
   - 长时间运行后卡在 NCCL collective 或后续 `add_request` / `exit` 生命周期交互

## Formal Direction

正式版本应遵循以下原则：

- `run_workload()` 不负责为 `slopearl` 做额外生命周期管理。它只负责把 workload 喂给 system，然后调用一次 `system.run()`。
- `slopearl` 的双缓冲必须由 engine / runner 内部自己维持一致性，不能依赖固定的初始两半 batch 快照跑满全部 step。
- 双缓冲循环中的“下一批 draft”和“当前批 verify”必须建立在 scheduler 当前仍然有效的 seq 集合上；如果一侧 batch 在验证后发生 finish / rollback / preempt，下一轮使用的 batch 也要相应更新。
- 现有已经确认正确的修复应保留：
  - SLO verify payload 长度统一 helper
  - SLO prefill 分支不再 assert
  - draft double-buffer 路径的 block reservation
  - 移除 engine 构造中的 `atexit.register(self.exit)`

## Proposed Implementation

### 1. 恢复单-engine workload 语义

- 删除 `supports_batch_reuse=False` 的 slopearl 特判。
- 恢复 `run_workload()` 使用单个 system 处理整个 workload 的逻辑。
- 对应 regression test：`slopearl` 不再要求跨 batch 重建 engine。

### 2. 让双缓冲循环使用可刷新的 batch 状态

当前 `schedule_double_buffer()` 只在 prefill 后把 `running` 列表一分为二，之后所有循环都只是在 `curr_*_batch` 和 `next_*_batch` 之间交换，不会重新和 scheduler 对齐。

正式修法：

- 在 draft / target 双缓冲 benchmark 循环中，引入一个小的“batch refresh”层：
  - 每轮在发送/接收验证结果后，基于当前仍处于 `scheduler.running` 的 seq 集合过滤掉已经 finished / preempted 的旧 batch 成员。
  - 如果一侧 batch 为空，或者双缓冲不再满足两批都非空，则安全退化到单批 `pearl_step()` 路径继续跑剩余 steps，而不是继续使用失效的静态双缓冲状态。
- 目标不是在运行时重新切整个 `running` 队列，而是保证双缓冲循环内部使用的 seq 引用始终仍然合法。

### 3. 明确 finish/preempt 后的双缓冲契约

- draft 端在 `verify_double_buffer_recv()` 后，任何被标记 finished 的 seq 都会从 `scheduler.running` 移除；后续循环不能再继续把这些 seq 留在 `curr_draft_batch` / `curr_verify_batch`。
- target 端在 `verify_double_buffer_send()` 后也会做相同的本地状态更新；它必须和 draft 一样对下一轮 batch 做过滤。
- 如果过滤后双缓冲失效（某一批为空），后续 step 退回单批同步 PEARL 路径，直到本次 benchmark step 用完。

## Testing Strategy

- Source-level regression test：验证双缓冲 benchmark 循环不再只依赖不可刷新的静态初始 batch。
- Unit regression：恢复 `run_workload()` 对非 reusable system 的假设移除，保证 slopearl 路径不再要求跨 batch 重建 system。
- Lightweight test suite：继续跑已有 `tests.test_run_workload`、`tests.test_slo_*`、`tests.test_pearl_*`。
- GPU validation 仍需用户在目标机器上执行真实 `slopearl` 命令确认。

## Non-Goals

- 这次不做性能优化。
- 这次不重写整个 SLO scheduler。
- 这次不引入新的 experiment shell interface。
