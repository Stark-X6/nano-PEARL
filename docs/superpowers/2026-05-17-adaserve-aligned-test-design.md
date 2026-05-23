# AdaServe-Aligned Test Design for nano-PEARL

## Goal

本设计文件定义 `nano-PEARL` 后续正式实验框架的唯一基准：

- 外层实验组织方式与 AdaServe 对齐
- workload schema 与 AdaServe 对齐
- 指标定义与 AdaServe 对齐
- baseline 语义与本文方法拆分清晰

当前实验约束：

- 当前只支持 Qwen 模型
- 当前固定使用 4 张 GPU
- GPU 拓扑固定为：
  - 1 张 GPU 跑 draft
  - 3 张 GPU 跑 target
- TP 固定为：
  - `draft_tp=1`
  - `target_tp=3`
- 模型 checkpoint 已经下载在数据盘，实验脚本只需要接收路径，不负责下载模型

本文件不沿用当前 `benchmark_slo/` 的随机输入测试设计。当前 `benchmark_slo/` 只保留为本地调试脚手架，不作为正式实验入口。

---

## 1. AdaServe 参考实现

### 1.1 实验组织

AdaServe 的实验框架分两层：

- 外层 shell switchboard: [exps/test.sh](/root/ykxia/AdaServe-Artifact-Evaluation/exps/test.sh)
- 各 figure 的 sweep 脚本: `exps/fig8,9/`, `exps/fig10/`, `exps/fig11/`, `exps/fig14/`, `exps/fig15/`

外层 shell 只负责三件事：

- 选择 baseline
- 注入 workload 文件
- 设置实验超参和结果文件路径

真正的 workload 消费、在线请求处理和指标统计全部在系统内核里完成。

### 1.2 Workload 形式

AdaServe 的 canonical workload 是 JSON trace。样例见：

- [fig8,9/emission_rps3.6_ol256.json](/root/ykxia/AdaServe-Artifact-Evaluation/exps/fig8,9/emission_rps3.6_ol256.json)

每条请求至少包含：

- `emission_time_ms`
- `prompt`
- `output_length`
- `slo_ratio`

`slo_ratio` 的语义与 AdaServe 内核一致：

- `slo_ratio > 0`：相对 baseline latency 的倍数
- `slo_ratio < 0`：绝对每 token 时延约束，单位 ms

### 1.3 Baseline 组织

AdaServe 的 `exps/test.sh` 用开关区分多个系统：

- `ENABLE_SPECSCHEDULER_SPEC_INFER`
- `ENABLE_VLLM_SERVER_BENCHMARK`
- `ENABLE_VLLM_SPEC_INFER`
- `ENABLE_VLLM_SARATHI_SERVE`
- `ENABLE_SPECSCHEDULER_SPEC_INFER_OVERHEAD_BREAKDOWN`

也就是说，正式实验比较的是“系统模式”，不是“脚本模式”。

### 1.4 指标定义

AdaServe 正式输出的关键指标在：

- [request_manager.cc](/root/ykxia/AdaServe-Artifact-Evaluation/adaserve/src/runtime/request_manager.cc:3895)
- [request_manager.cc](/root/ykxia/AdaServe-Artifact-Evaluation/adaserve/src/runtime/request_manager.cc:3942)

核心定义如下：

- `slo_attainment_rate`：
  完成请求中，满足 SLO 的请求占比
- `goodput`：
  满足 SLO 的请求所生成 token 总数 / 整个 serving run 的总 wall-clock 时间

AdaServe 不是用 batch 平均值近似每请求指标，而是基于每个请求的真实完成状态和真实 decode latency 计算。

---

## 2. 当前 nano-PEARL 测试层现状

### 2.1 当前文件

当前测试相关文件主要是：

- [benchmark/eval_benchmark.py](/root/ykxia/nano-PEARL/benchmark/eval_benchmark.py)
- [benchmark/eval_random.py](/root/ykxia/nano-PEARL/benchmark/eval_random.py)
- [benchmark_slo/eval_slo_benchmark.py](/root/ykxia/nano-PEARL/benchmark_slo/eval_slo_benchmark.py)
- [benchmark_slo/eval_slo_trace.py](/root/ykxia/nano-PEARL/benchmark_slo/eval_slo_trace.py)
- [benchmark_slo/generate_trace.py](/root/ykxia/nano-PEARL/benchmark_slo/generate_trace.py)

### 2.2 与 AdaServe 的关键偏差

| Topic | AdaServe | Current nano-PEARL | Required Alignment |
|---|---|---|---|
| 实验入口 | `exps/test.sh` + figure sweep scripts | `benchmark*.py` 单脚本入口 | 改成 AdaServe 风格 `exps/` |
| workload | JSON trace，含真实 `prompt/output_length/slo_ratio/emission_time_ms` | 随机 token ids 或简化 trace | 使用 AdaServe 同 schema trace |
| 请求模式 | open-loop trace replay | 随机 batch 或逐请求串行 | 改成持续在线发射 |
| baseline 粒度 | 系统模式切换 | benchmark 类型切换 | 改成 system switchboard |
| 指标 | per-request 真实统计 | 估算版 aggregate | 改成真实 request record 统计 |
| goodput | attained tokens / total wall time | 当前仅近似实现 | 严格复刻 AdaServe 定义 |
| SLO attainment | 基于真实 request completion | 当前按平均值近似 | 严格复刻 AdaServe 定义 |

### 2.3 当前 baseline 覆盖情况

当前仓库并没有完整覆盖正式实验需要的四个系统。

已存在：

- `PEARLEngine.bench_generate()`：
  单 batch，统一 gamma，带 draft/verify overlap
- `SLOPearlEngine.slo_bench_generate()`：
  单 batch，SLO-aware gamma，不开 double buffering
- `SLOPearlEngine.slo_bench_generate_double_buffer()`：
  双 batch，SLO-aware gamma，双缓冲 overlap

未存在：

- 串行 speculative decoding，统一 gamma，无 SLO

因此：

- 当前 `PEARLEngine.bench_generate()` 只能作为 `pearl-spec`
- 当前 `SLOPearlEngine.slo_bench_generate()` 可直接作为 `adaserve`
- 当前 `parallel_generate()` 是 AR，不是 speculative baseline

---

## 3. 正式实验应比较的系统

正式实验统一比较四个系统，其中前三个是 baseline，第四个是本文方法。

| Name | Budget Policy | Execution Mode | Batch Topology | Meaning |
|---|---|---|---|---|
| `vllm-spec` | uniform gamma | serial speculative decode | one batch | Baseline 1 |
| `pearl-spec` | uniform gamma | original nano-PEARL speculative path | one batch | Baseline 2 |
| `adaserve` | per-seq gamma | current single-batch SLO-aware path | one batch | Baseline 3 |
| `slopearl` | per-seq gamma | double-buffered SLO-aware path | two batches | Our method |

### 3.1 这四个系统的语义

`vllm-spec`

- 对应用户定义的“GPU 上实现的串行推测解码，统一 gamma，无 SLO 感知”
- 语义上接近 vLLM-style speculative decode
- 必须显式禁止 draft 与 verify 计算重叠

`pearl-spec`

- 对应原始 `nano-PEARL`
- 单 batch
- 统一 gamma
- draft 与 target 在一个 step 内重叠执行

`adaserve`

- 对应当前代码里已经实现的 single-batch SLO-aware speculative decoding
- 单 batch
- per-seq gamma
- `double_buffering=False`
- 正式入口对应当前 `SLOPearlEngine.slo_bench_generate()`

`slopearl`

- 对应本文方法
- 两个 batch 交替起草/验证
- per-seq gamma
- 开启双缓冲重叠

### 3.2 不作为正式 baseline 的系统

`single_batch_slo_pipelined`

- 已经并入 `adaserve`
- 不再单独作为额外消融命名

---

## 4. 新实验框架总体设计

### 4.1 目录布局

新的正式实验框架应镜像 AdaServe，但第一版只保留 Qwen 相关脚本：

```text
nano-PEARL/
├── exps/
│   ├── test.sh
│   ├── fig8,9/
│   │   ├── emission_rps*.json
│   │   └── run_qwen_rps.sh
│   ├── fig10/
│   │   ├── emission_rps*_prop*.json
│   │   └── run_qwen_prop.sh
│   ├── fig11/
│   │   ├── emission_slo*.json
│   │   └── run_qwen_slo.sh
│   └── fig14/
│       ├── emission_fluc_*.json
│       └── run_qwen_fluc.sh
├── results/
│   ├── fig8,9/
│   ├── fig10/
│   ├── fig11/
│   └── fig14/
└── benchmark_slo/
    ├── run_workload.py
    ├── workload_loader.py
    ├── metrics.py
    ├── systems.py
    └── trace_builder.py
```

说明：

- `exps/` 是正式实验入口，完全对齐 AdaServe
- `benchmark_slo/` 退化为 Python helper 层，不再作为对外 benchmark CLI

### 4.2 外层 switchboard

`exps/test.sh` 应模仿 AdaServe `exps/test.sh`，但 baseline 开关改成本文系统：

- `ENABLE_VLLM_SPEC`
- `ENABLE_PEARL_SPEC`
- `ENABLE_ADASERVE`
- `ENABLE_SLOPEARL`

它只负责：

- 检查模型和 workload 参数
- 选择一个系统模式
- 调用统一 Python workload runner
- 将 stdout/stderr 重定向到结果文件

它不负责计算指标。

### 4.3 统一 Python runner

新的正式 runner 只保留一个主入口，例如：

- `benchmark_slo/run_workload.py`

它接收：

- `--system`
- `--input-file`
- `--draft-model`
- `--target-model`
- `--draft-tp`
- `--target-tp`
- `--max-num-seqs`
- `--baseline-latency-per-token-ms`

并完成：

- 读取 AdaServe trace
- 按 emission time 持续注入请求
- 调用对应系统接口
- 记录 per-request lifecycle
- 统一计算 goodput 和 SLO attainment
- 输出 AdaServe-style 结果文本

---

## 5. Workload 对齐策略

### 5.1 Canonical workload source

正式实验直接采用 AdaServe 的 trace schema，优先直接复用或镜像以下 workload：

- `fig8,9`: RPS sweep
- `fig10`: strict-SLO proportion sweep
- `fig11`: SLO scale sweep
- `fig14`: fluctuating load

现有 `benchmark/data/*.jsonl` 不再直接作为正式评测输入。

### 5.2 Trace schema

新的 canonical trace schema 必须与 AdaServe 一致：

```json
{
  "emission_time_ms": 0.0,
  "prompt": "...",
  "output_length": 256,
  "slo_ratio": 1.2
}
```

如果本地需要再生 trace，`trace_builder.py` 也必须输出这个 schema，而不是当前的简化版 `prompt_len` schema。

### 5.3 Trace replay semantics

runner 必须采用 open-loop replay：

- 请求到达由 `emission_time_ms` 决定
- 请求被注入系统后，系统持续服务
- 不允许“一个请求跑完再提交下一个请求”

这点是当前 [eval_slo_trace.py](/root/ykxia/nano-PEARL/benchmark_slo/eval_slo_trace.py:80) 最大的结构性问题，也是必须重写的部分。

---

## 6. 指标对齐策略

### 6.1 基础统计字段

每个请求必须至少记录以下字段：

- `seq_id`
- `slo_ratio`
- `arrival_time_ms`
- `decode_start_time_ms`
- `finish_time_ms`
- `num_generated_tokens`
- `attained`

### 6.2 SLO 约束定义

对齐 AdaServe：

- 若 `slo_ratio > 0`
  - `slo_constraint_per_token_ms = slo_ratio * baseline_latency_per_token_ms`
- 若 `slo_ratio < 0`
  - `slo_constraint_per_token_ms = -slo_ratio`

总 decode SLO 判定：

```text
decode_latency_ms <= slo_constraint_per_token_ms * num_generated_tokens
```

这与 AdaServe `get_request_expected_latency(request)` 的逻辑保持一致。

### 6.3 Goodput 定义

对齐 AdaServe：

```text
goodput = sum(num_generated_tokens for attained requests) / total_run_time_s
```

### 6.4 SLO attainment rate 定义

对齐 AdaServe：

```text
slo_attainment_rate = attained_request_count / completed_request_count
```

### 6.5 输出格式

正式结果文件建议至少包含以下 key，尽量保持 AdaServe 风格：

- `system(...)`
- `completed_requests(...)`
- `slo_attainment(...)`
- `slo_attainment_by_scale(...)`
- `goodput(...)`
- `total_generated_tokens(...)`
- `total_run_time_s(...)`

---

## 7. 各 figure 的对齐策略

### 7.1 Figure 8/9: RPS sweep

保持 AdaServe 结构：

- 输入：`emission_rps*_ol256.json`
- sweep 变量：`RPS`
- 输出：`results/fig8,9/qwen/<system>/rps*_ol256.txt`
- 指标：`goodput`, `slo_attainment_rate`

### 7.2 Figure 10: strict-SLO proportion sweep

保持 AdaServe 结构：

- 输入：`emission_rps4.0_prop*.json`
- sweep 变量：严格 SLO 请求比例
- 输出：`results/fig10/qwen/<system>/prop*.txt`

### 7.3 Figure 11: SLO scale sweep

保持 AdaServe 结构：

- 输入：`emission_slo*_ol256.json`
- sweep 变量：SLO scale
- 输出：`results/fig11/qwen/<system>/slo*_ol256.txt`

### 7.4 Figure 14: fluctuating load

保持 AdaServe 结构：

- 输入：`emission_fluc_*.json`
- 输出：`results/fig14/qwen/<system>/ol*.txt`

### 7.5 Figure 15

当前用户只要求 `goodput` 和 `SLO attainment rate`，因此不需要先对齐 AdaServe 的 overhead breakdown。

`fig15` 不纳入第一版正式测试设计。

---

## 8. 当前代码与新设计的关系

### 8.1 可直接复用

- 原始 `PEARLEngine.bench_generate()` 可直接作为 `pearl-spec`
- 当前 `SLOPearlEngine.slo_bench_generate()` 可直接作为 `adaserve`
- 当前 `SLOPearlEngine.slo_bench_generate_double_buffer()` 可直接作为 `slopearl`
- `SLOScheduler` 可直接复用到 `adaserve` 和 `slopearl`
- `emission.py` 的分布采样逻辑可以保留，但 `TraceEmission` 需要只作为 timing helper，而不是 workload 定义本身

### 8.2 必须重定义

- `vllm-spec`
- 正式 trace replay runner
- 正式 per-request metrics collector
- 正式 AdaServe-style `exps/` shell harness

### 8.3 不再作为正式实验入口

- `benchmark/eval_random.py`
- `benchmark_slo/eval_slo_benchmark.py`
- `benchmark_slo/eval_slo_trace.py`

这些脚本可以保留用于 smoke test，但不再产出论文主结果。

---

## 9. 最终对齐结论

新的正式测试设计应遵循以下原则：

1. 不再以“随机输入 batch benchmark”为主。
2. 不再以“逐请求串行 generate”近似在线 serving。
3. 正式实验入口完全镜像 AdaServe 的 `exps/` 组织方式。
4. 正式 workload 完全采用 AdaServe trace schema。
5. 正式指标只使用真实 per-request 统计得到的 `goodput` 和 `SLO attainment rate`。
6. 正式主比较系统固定为四个：`vllm-spec`、`pearl-spec`、`adaserve`、`slopearl`。

补充说明：

- 其中 `adaserve` 在当前代码里已经存在实现，对应“不开 double buffering 的 SLO-aware 单 batch 路径”
- 只有 `vllm-spec` 仍然缺少明确的源代码开关与执行路径

如果以上六条不同时满足，那么测试只能算功能验证或局部 benchmark，不能算“和 AdaServe 对齐的正式实验”。

---

## 10. Acceptance Criteria

当以下条件全部成立时，认为测试设计完成对齐：

- 新实验入口位于 `exps/`，并能像 AdaServe 一样按 figure 运行
- trace 文件 schema 与 AdaServe 一致
- 能在同一 workload 上切换四个系统模式
- 输出结果文件中包含 `goodput(...)` 和 `slo_attainment(...)`
- 指标基于真实 per-request 完成记录统计，而非 batch 平均估算
- 当前 `benchmark_slo/*.py` 不再承担正式论文实验角色
