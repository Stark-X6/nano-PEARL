# GPU Validation Checklist for AdaServe-Aligned Qwen Test Framework

## Current Status

测试框架代码已经基本完成，当前分支上的相关提交包括：

- `66e1c03` `chore: ignore local test-design docs`
- `146262e` `docs: add qwen test framework implementation plan`
- `23dddf1` `feat(test): add vllm-spec engine entrypoints`
- `1dbb4e4` `feat(test): add workload loader and metrics core`
- `cfb6a1b` `feat(test): add systems mapping and result formatter`
- `186043c` `feat(test): implement formal workload runner`
- `b9d9253` `feat(test): add qwen experiment shell entrypoints`
- `d39a408` `feat(test): add canonical qwen workload traces`
- `ca6c54d` `feat(test): align workload results by seq_id`
- `d43487e` `chore(test): wrap legacy benchmark entrypoints`

当前已经具备的能力：

- 正式系统名统一：
  - `vllm-spec`
  - `pearl-spec`
  - `adaserve`
  - `slopearl`
- 正式 workload schema 对齐 AdaServe：
  - `emission_time_ms`
  - `prompt`
  - `output_length`
  - `slo_ratio`
- 正式 Python runner：
  - `benchmark_slo/workload_loader.py`
  - `benchmark_slo/metrics.py`
  - `benchmark_slo/systems.py`
  - `benchmark_slo/run_workload.py`
- 正式 shell 入口：
  - `exps/test.sh`
  - `exps/fig8,9/run_qwen_rps.sh`
  - `exps/fig10/run_qwen_prop.sh`
  - `exps/fig11/run_qwen_slo.sh`
  - `exps/fig14/run_qwen_fluc.sh`
- canonical workload traces 已经复制到：
  - `exps/fig8,9/`
  - `exps/fig10/`
  - `exps/fig11/`
  - `exps/fig14/`

本地单测已经全部通过：

```bash
python -m unittest \
  tests.test_workload_loader \
  tests.test_metrics \
  tests.test_systems \
  tests.test_run_workload \
  tests.test_exps \
  tests.test_trace_builder \
  tests.test_generate_trace \
  tests.test_legacy_wrappers
```

结果：

```text
Ran 17 tests in 0.173s

OK
```

---

## Hardware / Runtime Assumptions

当前正式实验固定假设：

- 只支持 Qwen 模型
- 4 张 GPU
- 1 张 GPU 跑 draft
- 3 张 GPU 跑 target
- `draft_tp=1`
- `target_tp=3`
- 模型 checkpoint 已经下载到数据盘

---

## Formal Systems

正式实验比较四个系统：

- `vllm-spec`
  - uniform gamma
  - speculative decoding baseline
- `pearl-spec`
  - 原始 `nano-PEARL`
  - uniform gamma
- `adaserve`
  - 当前 single-batch SLO-aware 路径
  - 对应 `SLOPearlEngine.slo_bench_generate()`
- `slopearl`
  - 当前 double-buffer SLO-aware 路径
  - 对应 `SLOPearlEngine.slo_bench_generate_double_buffer()`

---

## Output Expectations

每个正式输出文件都至少应包含：

- `system(...)`
- `completed_requests(...)`
- `total_generated_tokens(...)`
- `slo_attainment(...)`
- `goodput(...)`
- `total_run_time_s(...)`
- `slo_attainment_by_scale(...)`

---

## Validation Order

推荐的真实 GPU 验证顺序：

1. `pearl-spec`
2. `adaserve`
3. `slopearl`
4. `vllm-spec`

原因：

- 前 3 个更接近当前主线实现
- `vllm-spec` 是后补的 baseline 入口，最需要真机确认

---

## Test Checklist

### 1. Dry Run Check

目的：

- 验证 shell 层参数展开正常
- 验证结果文件路径和 workload 路径正确

命令：

```bash
DRY_RUN=1 \
QWEN_TARGET_MODEL=/path/to/target \
QWEN_DRAFT_MODEL=/path/to/draft \
ENABLE_VLLM_SPEC=ON \
bash exps/fig8,9/run_qwen_rps.sh
```

检查点：

- 输出里出现 `--system vllm-spec`
- 输出里出现真实的 `emission_rps*.json`
- 输出里出现目标结果文件路径

---

### 2. Minimal Smoke Test: `pearl-spec`

目的：

- 验证原始 `nano-PEARL` 路径已经被正式实验框架正确接入

命令：

```bash
LLM_MODEL=/path/to/target \
SSM_MODEL=/path/to/draft \
DATASETS_FILE=exps/fig8,9/emission_rps2.4_ol256.json \
OUTPUT_FILE=/tmp/pearl-spec.txt \
ENABLE_PEARL_SPEC=ON \
bash exps/test.sh
```

检查点：

- 进程正常启动
- 输出文件生成成功
- 输出文件包含：
  - `system(pearl-spec)`
  - `slo_attainment(...)`
  - `goodput(...)`

---

### 3. Minimal Smoke Test: `adaserve`

目的：

- 验证 single-batch SLO-aware 路径

命令：

```bash
LLM_MODEL=/path/to/target \
SSM_MODEL=/path/to/draft \
DATASETS_FILE=exps/fig8,9/emission_rps2.4_ol256.json \
OUTPUT_FILE=/tmp/adaserve.txt \
ENABLE_ADASERVE=ON \
bash exps/test.sh
```

检查点：

- 输出文件包含 `system(adaserve)`
- 输出文件包含 `slo_attainment_by_scale(...)`
- 输出文件包含 `goodput(...)`

---

### 4. Minimal Smoke Test: `slopearl`

目的：

- 验证 double-buffering 路径

命令：

```bash
LLM_MODEL=/path/to/target \
SSM_MODEL=/path/to/draft \
DATASETS_FILE=exps/fig8,9/emission_rps2.4_ol256.json \
OUTPUT_FILE=/tmp/slopearl.txt \
ENABLE_SLOPEARL=ON \
bash exps/test.sh
```

检查点：

- 输出文件包含 `system(slopearl)`
- 结果文件正常生成
- 没有通信死锁

---

### 5. Minimal Smoke Test: `vllm-spec`

目的：

- 验证后补的 baseline 入口

命令：

```bash
LLM_MODEL=/path/to/target \
SSM_MODEL=/path/to/draft \
DATASETS_FILE=exps/fig8,9/emission_rps2.4_ol256.json \
OUTPUT_FILE=/tmp/vllm-spec.txt \
ENABLE_VLLM_SPEC=ON \
bash exps/test.sh
```

检查点：

- 输出文件包含 `system(vllm-spec)`
- 输出文件包含 `goodput(...)`
- 路径行为符合 baseline 预期

---

### 6. Figure-Level Smoke Tests

目的：

- 验证 sweep 脚本和目录组织完整可用

#### 6.1 RPS Sweep

```bash
QWEN_TARGET_MODEL=/path/to/target \
QWEN_DRAFT_MODEL=/path/to/draft \
RPS_MIN=2.4 \
RPS_MAX=2.4 \
ENABLE_PEARL_SPEC=ON \
bash exps/fig8,9/run_qwen_rps.sh
```

检查点：

- `results/fig8,9/qwen/pearl-spec/` 下生成文件

#### 6.2 Strict-SLO Proportion Sweep

```bash
QWEN_TARGET_MODEL=/path/to/target \
QWEN_DRAFT_MODEL=/path/to/draft \
PROP_MIN=0.1 \
PROP_MAX=0.1 \
ENABLE_ADASERVE=ON \
bash exps/fig10/run_qwen_prop.sh
```

检查点：

- `results/fig10/qwen/adaserve/` 下生成文件

#### 6.3 SLO Scale Sweep

```bash
QWEN_TARGET_MODEL=/path/to/target \
QWEN_DRAFT_MODEL=/path/to/draft \
SLO_SCALE_MIN=1.0 \
SLO_SCALE_MAX=1.0 \
ENABLE_SLOPEARL=ON \
bash exps/fig11/run_qwen_slo.sh
```

检查点：

- `results/fig11/qwen/slopearl/` 下生成文件

#### 6.4 Fluctuating Load

```bash
QWEN_TARGET_MODEL=/path/to/target \
QWEN_DRAFT_MODEL=/path/to/draft \
ENABLE_VLLM_SPEC=ON \
bash exps/fig14/run_qwen_fluc.sh
```

检查点：

- `results/fig14/qwen/vllm-spec/` 下生成文件

---

## Final Acceptance Checklist

上卡后，认为正式测试框架真正可用，需要满足：

- `DRY_RUN=1` 时所有 shell 入口展开正确
- 四个系统都至少完成一次最小 smoke test
- 至少一个 figure 脚本跑通最小 sweep
- 每个结果文件都包含 canonical 指标 key
- 没有 GPU OOM
- 没有 NCCL 死锁
- `vllm-spec` 的真实运行行为符合预期 baseline 语义

---

## Remaining Risk

当前最大剩余风险不是测试框架，而是 **真实 GPU 运行语义**：

- `vllm-spec` 是后补 baseline，需要重点看真机行为
- `slopearl` 需要重点看双缓冲情况下是否存在卡住或异常回滚

换句话说：

- **测试框架代码已经基本完成**
- **真正未完成的，是有卡后的真实 smoke 验证**

