# AdaServe-Aligned Test File Checklist

## Goal

这份清单把正式实验重构到“文件级”。

目标不是继续修补当前 `benchmark_slo/`，而是把 `nano-PEARL` 的正式实验入口、workload、baseline 切换方式、指标输出方式全部改成 AdaServe 风格。

当前实验范围固定为：

- 仅支持 Qwen 模型
- 仅支持 4 张 GPU
- GPU 拓扑固定：
  - 1 张 draft
  - 3 张 target
- TP 固定：
  - `draft_tp=1`
  - `target_tp=3`
- 模型 checkpoint 已经下载在数据盘，实验脚本只负责接收模型路径

---

## A. 直接保留，不作为正式实验修改重点

这些文件可以继续保留，作为现有运行时实现或本地调试工具：

- `nano_pearl/pearl_engine/block_manager.py`
- `nano_pearl/pearl_engine/sequence.py`
- `nano_pearl/pearl_engine/scheduler.py`
- `nano_pearl/pearl_engine_slo/slo_sequence.py`
- `nano_pearl/pearl_engine_slo/slo_scheduler.py`
- `nano_pearl/slo_config.py`
- `nano_pearl/emission.py`
- `benchmark/data/*.jsonl`

说明：

- `benchmark/data/*.jsonl` 不再作为正式论文实验输入
- 但这些文件不需要删除

---

## B. 正式实验入口层：新增 AdaServe 风格 `exps/`

### 1. 新增 `exps/test.sh`

**Action:** 新增  
**Role:** 统一 baseline switchboard，完全对应 AdaServe 的 [exps/test.sh](/root/ykxia/AdaServe-Artifact-Evaluation/exps/test.sh)

需要实现的职责：

- 校验公共环境变量：
  - `LLM_MODEL`
  - `SSM_MODEL`
  - `DATASETS_FILE`
- 提供四个系统开关：
  - `ENABLE_VLLM_SPEC`
  - `ENABLE_PEARL_SPEC`
  - `ENABLE_ADASERVE`
  - `ENABLE_SLOPEARL`
- 对每个系统调用同一个 Python runner：
  - `benchmark_slo/run_workload.py`
- 负责：
  - 传递 workload 路径
  - 传递系统模式
  - 传递模型与 TP 配置
  - 传递结果文件路径

不应负责：

- 计算指标
- 解析 trace
- 组织请求生命周期

---

### 2. 新增 `exps/fig8,9/run_qwen_rps.sh`

**Action:** 新增  
**Role:** RPS sweep，Qwen

需要实现的职责：

- 对齐 AdaServe `fig8,9`
- sweep：
  - `RPS_MIN`
  - `RPS_MAX`
  - `RPS_STEP`
- 对每个 `RPS`：
  - 设置 `DATASETS_FILE=exps/fig8,9/emission_rps${RPS}_ol${OUTPUT_LENGTH}.json`
  - 设置四个系统各自的输出目录：
    - `results/fig8,9/qwen/vllm-spec/`
    - `results/fig8,9/qwen/pearl-spec/`
    - `results/fig8,9/qwen/adaserve/`
    - `results/fig8,9/qwen/slopearl/`
  - 调用 `exps/test.sh`

---

### 3. 新增 `exps/fig10/run_qwen_prop.sh`

**Action:** 新增  
**Role:** strict-SLO proportion sweep，Qwen

需要实现：

- sweep：
  - `PROP_MIN`
  - `PROP_MAX`
  - `PROP_STEP`
- 对每个比例：
  - 读取 `exps/fig10/emission_rps${RPS}_prop${PROP}.json`
  - 调用 `exps/test.sh`

---

### 4. 新增 `exps/fig11/run_qwen_slo.sh`

**Action:** 新增  
**Role:** SLO scale sweep，Qwen

需要实现：

- sweep：
  - `SLO_SCALE_MIN`
  - `SLO_SCALE_MAX`
  - `SLO_SCALE_STEP`
- workload 文件：
  - `exps/fig11/emission_slo${SLO_SCALE}_ol${OUTPUT_LENGTH}.json`

---

### 5. 新增 `exps/fig14/run_qwen_fluc.sh`

**Action:** 新增  
**Role:** fluctuating-load workload，Qwen

需要实现：

- 固定读取：
  - `exps/fig14/emission_fluc_6m_peak4.0.json`
- 切四个系统模式

---

## C. 正式 workload 层：新增或拷贝 AdaServe 风格 trace

### 10. 新增目录 `exps/fig8,9/`

**Action:** 新增目录 + 导入 JSON trace  
**Role:** 存放 RPS sweep 的 workload

建议直接导入与 AdaServe 同 schema 的文件，例如：

- `emission_rps2.6_ol256.json`
- `emission_rps2.8_ol256.json`
- `emission_rps3.0_ol256.json`
- ...

要求：

- schema 必须与 AdaServe 一致
- 字段至少包含：
  - `emission_time_ms`
  - `prompt`
  - `output_length`
  - `slo_ratio`

---

### 11. 新增目录 `exps/fig10/`

**Action:** 新增目录 + 导入 JSON trace  
**Role:** strict-SLO proportion sweep 的 workload

例如：

- `emission_rps4.0_prop0.1.json`
- `emission_rps4.0_prop0.2.json`
- ...

---

### 12. 新增目录 `exps/fig11/`

**Action:** 新增目录 + 导入 JSON trace  
**Role:** SLO scale sweep 的 workload

例如：

- `emission_slo0.6_ol256.json`
- `emission_slo0.8_ol256.json`
- ...

---

### 13. 新增目录 `exps/fig14/`

**Action:** 新增目录 + 导入 JSON trace  
**Role:** fluctuating-load workload

例如：

- `emission_fluc_6m_peak4.0.json`

---

## D. Python 正式 runner 层：新增统一实验后端

### 14. 新增 `benchmark_slo/run_workload.py`

**Action:** 新增  
**Role:** 正式唯一 workload runner

这是整个正式实验框架的核心 Python 入口。

职责：

- 解析 CLI 参数：
  - `--system`
  - `--input-file`
  - `--draft-model`
  - `--target-model`
  - `--draft-tp`
  - `--target-tp`
  - `--baseline-latency-per-token-ms`
  - `--max-num-seqs`
  - `--max-num-batched-tokens`
- 调用 `workload_loader.py` 读取 trace
- 调用 `systems.py` 选择系统实现
- 进行 open-loop trace replay
- 调用 `metrics.py` 计算：
  - `goodput`
  - `slo_attainment`
  - `slo_attainment_by_scale`
- 将结果输出为 AdaServe 风格文本

不应实现：

- baseline 具体算法
- 低层 engine 逻辑

---

### 15. 新增 `benchmark_slo/workload_loader.py`

**Action:** 新增  
**Role:** 读取 AdaServe 风格 trace，并转为统一 request record

职责：

- 读取 JSON trace
- 校验字段：
  - `emission_time_ms`
  - `prompt`
  - `output_length`
  - `slo_ratio`
- 规范化为内部结构，例如：
  - `request_id`
  - `arrival_time_ms`
  - `prompt`
  - `output_length`
  - `slo_ratio`

不应实现：

- 请求发射逻辑
- 指标统计逻辑

---

### 16. 新增 `benchmark_slo/metrics.py`

**Action:** 新增  
**Role:** 严格复刻 AdaServe 指标定义

职责：

- 计算 per-request SLO 是否达成
- 计算：
  - `slo_attainment`
  - `slo_attainment_by_scale`
  - `goodput`
  - `total_generated_tokens`
  - `completed_requests`
  - `total_run_time_s`

输入应基于真实 per-request lifecycle：

- `seq_id`
- `slo_ratio`
- `arrival_time_ms`
- `decode_start_time_ms`
- `finish_time_ms`
- `num_generated_tokens`

这里不能再用当前 `_compute_metrics()` 那种 batch 平均估计。

---

### 17. 新增 `benchmark_slo/systems.py`

**Action:** 新增  
**Role:** 统一封装四个正式系统

需要暴露四种系统模式：

- `vllm-spec`
- `pearl-spec`
- `adaserve`
- `slopearl`

职责：

- 根据 `--system` 返回相应的 driver
- 屏蔽底层 engine 差异
- 提供统一接口，例如：
  - `submit_request(...)`
  - `run_until(time_ms)`
  - `drain()`
  - `collect_request_records()`

---

### 18. 新增 `benchmark_slo/trace_builder.py`

**Action:** 新增  
**Role:** 本地生成与 AdaServe 同 schema 的 trace

用途：

- 仅在需要再造自定义 trace 时使用

输出 schema 必须是：

```json
{
  "emission_time_ms": ...,
  "prompt": "...",
  "output_length": ...,
  "slo_ratio": ...
}
```

它不能再输出当前 `generate_trace.py` 的简化版 `prompt_len` schema。

---

## E. 系统实现层：新增正式 baseline 接口

### 19. 修改 `nano_pearl/pearl_engine/pearl_model_runner.py`

**Action:** 修改  
**Role:** 增加 `vllm-spec` 所需的新执行路径

需要新增的能力：

- 一个“串行 speculative decode”路径
- 语义：
  - 仍然统一 `gamma`
  - 仍然 speculative
  - 但禁止 draft/verify overlap

建议新增方法：

- `serial_pearl_step()`
- `serial_pearl_generate()`
- `serial_pearl_bench_generate()`

要求：

- 不破坏现有 `pearl_step()` 和 `pearl_bench_generate()`
- `pearl-spec` 继续复用旧路径
- `vllm-spec` 走新路径

---

### 20. 修改 `nano_pearl/pearl_engine/pearl_engine.py`

**Action:** 修改  
**Role:** 对外暴露 `vllm-spec` 入口

建议新增方法：

- `serial_generate()`
- `serial_bench_generate()`

职责：

- 将 shared-memory 控制命令发给 draft/target runner
- 区分：
  - `bench_generate()` -> `pearl-spec`
  - `serial_bench_generate()` -> `vllm-spec`

---

### 21. 修改 `nano_pearl/pearl_engine_slo/slo_draft_runner.py`

**Action:** 修改  
**Role:** 明确 `adaserve` 与 `slopearl` 的正式入口职责

当前已有：

- `slo_bench_generate()`：单 batch，SLO-aware，不开 double buffering
- `slo_bench_generate_double_buffer()`：双 batch，SLO-aware

正式实验映射应写清：

- `slo_bench_generate()` -> `adaserve`
- `slo_bench_generate_double_buffer()` -> `slopearl`

这部分不要求为 `adaserve` 再新增一条 serial 路径。

---

### 22. 修改 `nano_pearl/pearl_engine_slo/slo_target_runner.py`

**Action:** 修改  
**Role:** 明确 `adaserve` 与 `slopearl` 的 target 侧职责

要求：

- 保持：
  - `slo_bench_generate()` 对应 `adaserve`
  - `slo_bench_generate_double_buffer()` 对应 `slopearl`
- 不要求为 `adaserve` 额外新增 serial target 路径

---

### 23. 修改 `nano_pearl/pearl_engine_slo/slo_pearl_engine.py`

**Action:** 修改  
**Role:** 对外暴露正式的三种 SLO 系统入口

保留：

- `slo_bench_generate()`：正式 `adaserve` 方法
- `slo_bench_generate_double_buffer()`：正式主方法

同时建议修改：

- `_compute_metrics()` 不再承担正式论文实验指标计算

它可以保留作为本地 quick summary，但正式指标应转移到 `benchmark_slo/metrics.py`。

---

## F. 现有测试脚本层：降级为 smoke test / 废弃正式角色

### 24. 修改 `benchmark_slo/eval_slo_benchmark.py`

**Action:** 修改  
**Role:** 降级为 smoke test，或直接改成 thin wrapper

当前问题：

- 用随机 token ids
- 不读 AdaServe trace
- 不具备正式 open-loop replay 语义

建议改法二选一：

- 方案 A：保留，但在文件头标明“非正式实验，仅本地 smoke”
- 方案 B：直接变成对 `benchmark_slo/run_workload.py` 的兼容包装

正式实验不再从这里启动。

---

### 25. 修改 `benchmark_slo/eval_slo_trace.py`

**Action:** 修改  
**Role:** 不再作为正式 trace runner

当前问题：

- 每次只提交一个请求
- 立即跑一次 `slo_generate()`
- 不符合 AdaServe open-loop serving 模式

建议：

- 标记 deprecated
- 或改成对 `run_workload.py` 的轻量 wrapper

---

### 26. 修改 `benchmark_slo/generate_trace.py`

**Action:** 修改  
**Role:** 替换为 AdaServe 同 schema 的 trace 生成器，或让位给 `trace_builder.py`

当前问题：

- 只输出：
  - `emission_time_ms`
  - `slo_ratio`
  - `prompt_len`
- 正式实验不够用

建议：

- 若保留此文件，就改成输出完整 schema
- 若新增 `trace_builder.py`，则把此文件标记 deprecated

---

## G. 输出目录层：运行时创建，不需要提交空结果

### 27. 新增运行时目录 `results/`

**Action:** 运行时创建  
**Role:** 存放正式实验输出

建议结构：

- `results/fig8,9/<model>/<system>/`
- `results/fig10/<model>/<system>/`
- `results/fig11/<model>/<system>/`
- `results/fig14/<model>/<system>/`

不需要提交结果文件到仓库，只需要让脚本自动 `mkdir -p`。

---

## H. 实施顺序建议

按这个顺序改最稳：

1. `benchmark_slo/workload_loader.py`
2. `benchmark_slo/metrics.py`
3. `benchmark_slo/systems.py`
4. `nano_pearl/pearl_engine/pearl_model_runner.py`
5. `nano_pearl/pearl_engine/pearl_engine.py`
6. `nano_pearl/pearl_engine_slo/slo_draft_runner.py`
7. `nano_pearl/pearl_engine_slo/slo_target_runner.py`
8. `nano_pearl/pearl_engine_slo/slo_pearl_engine.py`
9. `benchmark_slo/run_workload.py`
10. `exps/test.sh`
11. `exps/fig8,9/*`
12. `exps/fig10/*`
13. `exps/fig11/*`
14. `exps/fig14/*`
15. 最后再处理 `benchmark_slo/eval_slo_benchmark.py` / `eval_slo_trace.py` / `generate_trace.py`

---

## I. 最终文件状态总表

### 新增

- `exps/test.sh`
- `exps/fig8,9/run_qwen_rps.sh`
- `exps/fig10/run_qwen_prop.sh`
- `exps/fig11/run_qwen_slo.sh`
- `exps/fig14/run_qwen_fluc.sh`
- `exps/fig8,9/*.json`
- `exps/fig10/*.json`
- `exps/fig11/*.json`
- `exps/fig14/*.json`
- `benchmark_slo/run_workload.py`
- `benchmark_slo/workload_loader.py`
- `benchmark_slo/metrics.py`
- `benchmark_slo/systems.py`
- `benchmark_slo/trace_builder.py`

### 修改

- `nano_pearl/pearl_engine/pearl_model_runner.py`
- `nano_pearl/pearl_engine/pearl_engine.py`
- `nano_pearl/pearl_engine_slo/slo_draft_runner.py`
- `nano_pearl/pearl_engine_slo/slo_target_runner.py`
- `nano_pearl/pearl_engine_slo/slo_pearl_engine.py`
- `benchmark_slo/eval_slo_benchmark.py`
- `benchmark_slo/eval_slo_trace.py`
- `benchmark_slo/generate_trace.py`

### 保留不动

- `nano_pearl/pearl_engine/block_manager.py`
- `nano_pearl/pearl_engine/sequence.py`
- `nano_pearl/pearl_engine_slo/slo_sequence.py`
- `nano_pearl/pearl_engine_slo/slo_scheduler.py`
- `nano_pearl/slo_config.py`
- `nano_pearl/emission.py`（除非你想把 trace helper 也并入正式 runner）

### 不再作为正式实验入口

- `benchmark/eval_random.py`
- `benchmark/eval_benchmark.py`
- `benchmark_slo/eval_slo_benchmark.py`
- `benchmark_slo/eval_slo_trace.py`
