# AdaServe-Aligned Qwen Test Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an AdaServe-aligned formal test framework for Qwen-only `nano-PEARL` experiments, including the missing `vllm-spec` baseline and the `exps/`-style workload runner.

**Architecture:** Keep the current runtime skeleton and only add the missing `vllm-spec` execution path. Reuse existing `pearl-spec`, `adaserve`, and `slopearl` engine entrypoints. Move formal experiment orchestration into an AdaServe-style `exps/` shell layer plus a Python `run_workload.py` backend that replays trace workloads and computes real per-request `goodput` and `slo_attainment`.

**Tech Stack:** Python, Bash, PyTorch distributed, existing `nano-PEARL` runtime, Git-based milestone commits.

---

### Task 1: Clean Repo State And Record The Plan

**Files:**
- Create: `docs/superpowers/plans/2026-05-17-adaserve-aligned-qwen-test-framework.md`
- Modify: `.gitignore`

- [ ] **Step 1: Verify the current working tree is clean enough to proceed**

Run:

```bash
git -C /root/ykxia/nano-PEARL status --short --branch
```

Expected:

```text
## feat/double-buffering...origin/feat/double-buffering
 M .gitignore
?? .ipynb_checkpoints/
```

- [ ] **Step 2: Keep local design docs ignored and leave `.ipynb_checkpoints/` untouched**

Required state in `.gitignore`:

```gitignore
__pycache__/
nano_pearl/__pycache__/
nano_pearl/*/__pycache__/

nano_PEARL.egg-info/
.tmp/

~/models/
docs/2026-05-17-adaserve-aligned-test-design.md
docs/2026-05-17-adaserve-aligned-test-file-checklist.md
```

- [ ] **Step 3: Commit the repo hygiene change as the first milestone**

Run:

```bash
git -C /root/ykxia/nano-PEARL add .gitignore
git -C /root/ykxia/nano-PEARL commit -m "chore: ignore local test-design docs"
```

Expected:

```text
[feat/double-buffering ...] chore: ignore local test-design docs
 1 file changed, ...
```

---

### Task 2: Add The Missing `vllm-spec` Runtime Path

**Files:**
- Modify: `nano_pearl/pearl_engine/pearl_model_runner.py`
- Modify: `nano_pearl/pearl_engine/pearl_engine.py`

- [ ] **Step 1: Add a distinct serial speculative path in the draft runner**

Implement new draft-side methods in `nano_pearl/pearl_engine/pearl_model_runner.py`:

```python
class DraftModelRunner(ModelRunnerBase):
    ...
    def vllm_spec_step(self):
        seqs, is_prefill = self.scheduler.schedule()
        assert not is_prefill, "wrong match. current stage is prefill."
        for _ in range(self.gamma):
            input_ids, positions = self.prepare_pearl_decode(seqs)
            logits = self.run_model(input_ids, positions, False)
            sample_tokens = (
                logits.argmax(dim=-1)
                if self.tp_params.local_rank == 0
                else torch.zeros(len(seqs), dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
            )
            dist.broadcast(sample_tokens, src=self.tp_params.master_rank, group=self.group)
            token_ids = sample_tokens.tolist()
            reset_context(self.tp_params)
            for seq, token_id in zip(seqs, token_ids):
                seq.append_token(token_id)
        self.verify(seqs)

    def vllm_spec_generate(self):
        dist.barrier()
        torch.cuda.synchronize()
        start_time = time.time()
        self.prefill()
        if self.gamma == -1:
            self.gamma = self.gamma_list[next(x for x in self.gamma_list if x >= len(self.scheduler.running))]
        while not self.scheduler.is_finished():
            self.vllm_spec_step()
        torch.cuda.synchronize()
        end_time = time.time()
        self.clear_requests()

    def vllm_spec_bench_generate(self, num_pearl_steps: int = 100):
        dist.barrier()
        torch.cuda.synchronize()
        start_time = time.time()
        self.prefill()
        for seq in self.scheduler.running:
            seq.max_tokens = 1e8
            seq.ignore_eos = True
        if self.gamma == -1:
            self.gamma = self.gamma_list[next(x for x in self.gamma_list if x >= len(self.scheduler.running))]
        for _ in range(num_pearl_steps):
            self.vllm_spec_step()
        torch.cuda.synchronize()
        end_time = time.time()
        for seq in self.scheduler.running:
            seq.num_acc_tokens.append(seq.cur_acc_tokens)
        output = [(seq.seq_id, seq.completion_token_ids, seq.num_acc_tokens) for seq in self.scheduler.running]
        if self.rank == self.global_config.target_config.master_rank:
            data = pickle.dumps([output, end_time - start_time])
            n = len(data)
            self.shm.buf[0:4] = n.to_bytes(4, "little")
            self.shm.buf[4:n+4] = data
        self.clear_requests()
```

Notes:

- This is intentionally close to the existing PEARL speculative loop.
- The distinction is interface-level and experiment-level naming: this becomes the formal `vllm-spec` baseline entrypoint.
- Do not modify the existing `pearl_step()` path.

- [ ] **Step 2: Add the target-side dispatch by reusing the current verify path**

Implement target-side methods in `nano_pearl/pearl_engine/pearl_model_runner.py`:

```python
class TargetModelRunner(ModelRunnerBase):
    ...
    def vllm_spec_step(self):
        seqs, is_prefill = self.scheduler.schedule()
        assert not is_prefill, "wrong match. current stage is prefill."
        input_ids, positions, temp_seqs = self.prepare_pearl_decode(seqs)
        temperatures = self.prepare_sample(temp_seqs) if self.tp_params.local_rank == 0 else None
        logits = self.run_model(input_ids, positions, False)
        self.verify(logits, seqs, temperatures)

    def vllm_spec_generate(self):
        dist.barrier()
        torch.cuda.synchronize()
        start_time = time.time()
        self.prefill()
        if self.gamma == -1:
            self.gamma = self.gamma_list[next(x for x in self.gamma_list if x >= len(self.scheduler.running))]
        while not self.scheduler.is_finished():
            self.vllm_spec_step()
        torch.cuda.synchronize()
        end_time = time.time()
        seqs = self.scheduler.finished
        output = [(seq.seq_id, seq.completion_token_ids, seq.num_acc_tokens) for seq in seqs]
        if self.rank == self.global_config.target_config.master_rank:
            data = pickle.dumps([output, end_time - start_time])
            n = len(data)
            self.shm.buf[0:4] = n.to_bytes(4, "little")
            self.shm.buf[4:n+4] = data
        self.clear_requests()

    def vllm_spec_bench_generate(self, num_pearl_steps: int = 100):
        dist.barrier()
        torch.cuda.synchronize()
        start_time = time.time()
        self.prefill()
        for seq in self.scheduler.running:
            seq.max_tokens = 1e8
            seq.ignore_eos = True
        if self.gamma == -1:
            self.gamma = self.gamma_list[next(x for x in self.gamma_list if x >= len(self.scheduler.running))]
        for _ in range(num_pearl_steps):
            self.vllm_spec_step()
        torch.cuda.synchronize()
        end_time = time.time()
        for seq in self.scheduler.running:
            seq.num_acc_tokens.append(seq.cur_acc_tokens)
        output = [(seq.seq_id, seq.completion_token_ids, seq.num_acc_tokens) for seq in self.scheduler.running]
        if self.rank == self.global_config.target_config.master_rank:
            data = pickle.dumps([output, end_time - start_time])
            n = len(data)
            self.shm.buf[0:4] = n.to_bytes(4, "little")
            self.shm.buf[4:n+4] = data
        self.clear_requests()
```

- [ ] **Step 3: Expose the new entrypoints in `PEARLEngine`**

Add the following methods to `nano_pearl/pearl_engine/pearl_engine.py`:

```python
class PEARLEngine:
    ...
    def vllm_spec_generate(self):
        self.controller.write_draft_shm("vllm_spec_generate")
        self.controller.write_target_shm("vllm_spec_generate")
        self.control_event.wait()
        self.control_event.clear()
        output, time = self.controller.read_output()
        output = sorted(output, key=lambda x: x[0])
        seq_id, token_ids, num_acc_tokens = zip(*output)
        output_text = [self.tokenizer.decode(token_ids, skip_special_tokens=False) for token_ids in token_ids]
        num_tokens = [len(t) for t in token_ids]
        return output_text, num_tokens, num_acc_tokens, time

    def vllm_spec_bench_generate(self, num_pearl_steps: int = 100):
        self.controller.write_draft_shm("vllm_spec_bench_generate", num_pearl_steps)
        self.controller.write_target_shm("vllm_spec_bench_generate", num_pearl_steps)
        self.control_event.wait()
        self.control_event.clear()
        output, time = self.controller.read_output()
        output = sorted(output, key=lambda x: x[0])
        seq_id, token_ids, num_acc_tokens = zip(*output)
        output_text = [self.tokenizer.decode(token_ids, skip_special_tokens=False) for token_ids in token_ids]
        num_tokens = [len(t) for t in token_ids]
        return output_text, num_tokens, num_acc_tokens, time
```

- [ ] **Step 4: Run a syntax-only verification on the modified runtime files**

Run:

```bash
python -m py_compile \
  /root/ykxia/nano-PEARL/nano_pearl/pearl_engine/pearl_model_runner.py \
  /root/ykxia/nano-PEARL/nano_pearl/pearl_engine/pearl_engine.py
```

Expected:

```text
[no output]
```

- [ ] **Step 5: Commit the `vllm-spec` runtime milestone**

Run:

```bash
git -C /root/ykxia/nano-PEARL add \
  nano_pearl/pearl_engine/pearl_model_runner.py \
  nano_pearl/pearl_engine/pearl_engine.py
git -C /root/ykxia/nano-PEARL commit -m "feat(test): add vllm-spec runtime path"
```

---

### Task 3: Add A Unified Formal Workload Runner

**Files:**
- Create: `benchmark_slo/workload_loader.py`
- Create: `benchmark_slo/metrics.py`
- Create: `benchmark_slo/systems.py`
- Create: `benchmark_slo/run_workload.py`

- [ ] **Step 1: Implement a canonical AdaServe-style workload loader**

Expected public API in `benchmark_slo/workload_loader.py`:

```python
from dataclasses import dataclass

@dataclass
class WorkloadRequest:
    request_id: int
    emission_time_ms: float
    prompt: str
    output_length: int
    slo_ratio: float

def load_workload(path: str) -> list[WorkloadRequest]:
    ...
```

Validation rules:

- `emission_time_ms` must exist
- `prompt` must exist
- `output_length` must exist
- `slo_ratio` must exist

- [ ] **Step 2: Implement formal per-request metrics**

Expected public API in `benchmark_slo/metrics.py`:

```python
from dataclasses import dataclass

@dataclass
class RequestRecord:
    request_id: int
    slo_ratio: float
    arrival_time_ms: float
    decode_start_time_ms: float
    finish_time_ms: float
    num_generated_tokens: int
    attained: bool

def compute_metrics(records: list[RequestRecord], total_run_time_s: float) -> dict:
    ...
```

Must output:

- `goodput`
- `slo_attainment`
- `slo_attainment_by_scale`
- `completed_requests`
- `total_generated_tokens`
- `total_run_time_s`

- [ ] **Step 3: Implement the system mapper**

Expected mapping in `benchmark_slo/systems.py`:

```python
SYSTEMS = {
    "vllm-spec": ...,
    "pearl-spec": ...,
    "adaserve": ...,
    "slopearl": ...,
}
```

Required runtime mapping:

- `vllm-spec` -> `PEARLEngine.vllm_spec_bench_generate`
- `pearl-spec` -> `PEARLEngine.bench_generate`
- `adaserve` -> `SLOPearlEngine.slo_bench_generate`
- `slopearl` -> `SLOPearlEngine.slo_bench_generate_double_buffer`

- [ ] **Step 4: Implement the formal open-loop workload runner**

Expected behavior in `benchmark_slo/run_workload.py`:

- parse Qwen-only runtime args
- load trace
- instantiate the selected system
- replay requests according to `emission_time_ms`
- submit each request with `SamplingParams(max_tokens=output_length)`
- collect request records
- compute formal metrics
- print AdaServe-style result text

- [ ] **Step 5: Commit the formal runner milestone**

Run:

```bash
git -C /root/ykxia/nano-PEARL add \
  benchmark_slo/workload_loader.py \
  benchmark_slo/metrics.py \
  benchmark_slo/systems.py \
  benchmark_slo/run_workload.py
git -C /root/ykxia/nano-PEARL commit -m "feat(test): add formal workload runner"
```

---

### Task 4: Add AdaServe-Style `exps/` Experiment Entry Points

**Files:**
- Create: `exps/test.sh`
- Create: `exps/fig8,9/run_qwen_rps.sh`
- Create: `exps/fig10/run_qwen_prop.sh`
- Create: `exps/fig11/run_qwen_slo.sh`
- Create: `exps/fig14/run_qwen_fluc.sh`

- [ ] **Step 1: Add the shared `exps/test.sh` switchboard**

Required toggles:

```bash
ENABLE_VLLM_SPEC
ENABLE_PEARL_SPEC
ENABLE_ADASERVE
ENABLE_SLOPEARL
```

Required behavior:

- validate `LLM_MODEL`, `SSM_MODEL`, `DATASETS_FILE`
- call `python benchmark_slo/run_workload.py ...`
- route output to the correct result file

- [ ] **Step 2: Add Qwen-only figure sweep scripts**

Required scripts:

- `exps/fig8,9/run_qwen_rps.sh`
- `exps/fig10/run_qwen_prop.sh`
- `exps/fig11/run_qwen_slo.sh`
- `exps/fig14/run_qwen_fluc.sh`

Each script must:

- define Qwen model paths through environment variables
- define `draft_tp=1`, `target_tp=3`
- sweep only the figure-specific variable
- write results under `results/<figure>/qwen/<system>/...`

- [ ] **Step 3: Commit the experiment entrypoint milestone**

Run:

```bash
git -C /root/ykxia/nano-PEARL add exps
git -C /root/ykxia/nano-PEARL commit -m "feat(test): add exps entrypoints for qwen sweeps"
```

---

### Task 5: Add Canonical AdaServe-Style Qwen Workloads

**Files:**
- Create: `exps/fig8,9/*.json`
- Create: `exps/fig10/*.json`
- Create: `exps/fig11/*.json`
- Create: `exps/fig14/*.json`

- [ ] **Step 1: Copy or generate trace files with canonical schema**

Every trace record must contain:

```json
{
  "emission_time_ms": 0.0,
  "prompt": "...",
  "output_length": 256,
  "slo_ratio": 1.2
}
```

- [ ] **Step 2: Add a helper generator for local custom traces if needed**

Expected helper:

```python
def build_trace(...):
    ...
```

This belongs in `benchmark_slo/trace_builder.py`.

- [ ] **Step 3: Commit the workload milestone**

Run:

```bash
git -C /root/ykxia/nano-PEARL add exps/fig8,9 exps/fig10 exps/fig11 exps/fig14 benchmark_slo/trace_builder.py
git -C /root/ykxia/nano-PEARL commit -m "feat(test): add qwen workload traces"
```

---

### Task 6: Demote Legacy Benchmark Scripts To Smoke-Test Role

**Files:**
- Modify: `benchmark_slo/eval_slo_benchmark.py`
- Modify: `benchmark_slo/eval_slo_trace.py`
- Modify: `benchmark_slo/generate_trace.py`

- [ ] **Step 1: Mark legacy benchmark scripts as non-formal**

Add file header notes clarifying:

- these scripts are for local smoke tests only
- formal experiments now live under `exps/` + `benchmark_slo/run_workload.py`

- [ ] **Step 2: Optionally make them thin wrappers**

If kept callable, they should defer to:

```python
benchmark_slo/run_workload.py
```

and not implement separate metric logic.

- [ ] **Step 3: Commit the legacy demotion milestone**

Run:

```bash
git -C /root/ykxia/nano-PEARL add \
  benchmark_slo/eval_slo_benchmark.py \
  benchmark_slo/eval_slo_trace.py \
  benchmark_slo/generate_trace.py
git -C /root/ykxia/nano-PEARL commit -m "chore(test): demote legacy benchmark scripts"
```

---

### Task 7: End-To-End Verification

**Files:**
- Verify only

- [ ] **Step 1: Smoke-test each system switch**

Run representative commands such as:

```bash
bash exps/fig8,9/run_qwen_rps.sh
```

with exactly one enabled system at a time.

- [ ] **Step 2: Confirm formal output contains the canonical metrics**

Expected keys in result text:

```text
system(...)
slo_attainment(...)
slo_attainment_by_scale(...)
goodput(...)
```

- [ ] **Step 3: Commit only if code changes were made during verification**

```bash
git -C /root/ykxia/nano-PEARL commit -m "fix(test): address verification issues"
```

