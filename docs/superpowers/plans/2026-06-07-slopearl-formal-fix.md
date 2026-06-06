# Slopearl Formal Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore `slopearl` to a formal single-engine experiment path by removing the workload-layer engine recreation workaround and making the double-buffer benchmark loop tolerate runtime batch membership changes.

**Architecture:** Keep the verified SLO protocol fixes, but move correctness back into the engine layer. `run_workload()` should treat `slopearl` like the other systems again, while the draft/target double-buffer loops refresh or safely degrade their local batch views after verify results mutate scheduler state.

**Tech Stack:** Python, unittest, existing `benchmark_slo` harness, `nano_pearl` SLO runners.

---

### Task 1: Restore single-engine workload semantics

**Files:**
- Modify: `benchmark_slo/run_workload.py`
- Modify: `benchmark_slo/systems.py`
- Test: `tests/test_run_workload.py`

- [ ] Remove the `supports_batch_reuse=False` workaround for `slopearl` and restore `run_workload()` to the single-system lifecycle.
- [ ] Update the regression test so `slopearl` no longer expects per-batch system recreation.
- [ ] Run `python -m unittest tests.test_run_workload`.

### Task 2: Make draft double-buffer batch state refreshable

**Files:**
- Modify: `nano_pearl/pearl_engine_slo/slo_draft_runner.py`
- Test: `tests/test_slo_double_buffer_source.py`

- [ ] Add a small helper in the draft runner that filters a batch down to seqs still present in `scheduler.running`.
- [ ] Use that helper after each verify receive/send boundary so finished or preempted seqs do not stay in the next loop iteration.
- [ ] If either side of the double buffer becomes empty, degrade to single-batch `pearl_step()` for the remaining benchmark steps.
- [ ] Extend the source regression to assert the benchmark loop refreshes or filters batch state instead of only swapping static initial batches.
- [ ] Run `python -m unittest tests.test_slo_double_buffer_source`.

### Task 3: Mirror the same batch refresh contract on the target side

**Files:**
- Modify: `nano_pearl/pearl_engine_slo/slo_target_runner.py`
- Test: `tests/test_slo_double_buffer_source.py`

- [ ] Add the same running-set filtering / safe degradation behavior in target double-buffer loops.
- [ ] Ensure draft and target still run the same number of verify exchanges after degradation logic triggers.
- [ ] Extend source regression coverage if needed.
- [ ] Run `python -m unittest tests.test_slo_double_buffer_source tests.test_pearl_step_source tests.test_slo_verify_protocol`.

### Task 4: Regression sweep and local cleanup

**Files:**
- Modify: `tests/test_run_workload.py`
- Modify: `tests/test_slo_double_buffer_source.py`
- Modify: `tests/test_engine_lifecycle_source.py`

- [ ] Run the lightweight regression suite covering workload harness, SLO protocol, double-buffer source behavior, lifecycle behavior, and PEARL sync.
- [ ] Run `git diff --check`.
- [ ] Prepare a local commit with only the formal slopearl fix files.

