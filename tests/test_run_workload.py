import types
import unittest

from benchmark_slo.metrics import RequestRecord
from benchmark_slo.run_workload import (
    build_request_records,
    format_result_text,
    run_workload,
)
from benchmark_slo.workload_loader import WorkloadRequest


class RunWorkloadFormattingTests(unittest.TestCase):
    def test_format_result_text_emits_adaserve_style_keys(self):
        metrics = {
            "completed_requests": 3,
            "total_generated_tokens": 9,
            "goodput": 2.5,
            "slo_attainment": 2 / 3,
            "slo_attainment_by_scale": {
                1.0: {"attained": 1, "total": 1, "rate": 1.0},
                0.6: {"attained": 0, "total": 1, "rate": 0.0},
            },
            "total_run_time_s": 2.0,
        }

        text = format_result_text("adaserve", metrics)

        self.assertIn("system(adaserve)", text)
        self.assertIn("completed_requests(3)", text)
        self.assertIn("total_generated_tokens(9)", text)
        self.assertIn("goodput(2.500)", text)
        self.assertIn("slo_attainment(66.667%)", text)
        self.assertIn("slo_attainment_by_scale(", text)

    def test_build_request_records_uses_positive_and_negative_slo_constraints(self):
        workload = [
            WorkloadRequest(
                request_id=0,
                emission_time_ms=0.0,
                prompt="p0",
                output_length=4,
                slo_ratio=1.0,
            ),
            WorkloadRequest(
                request_id=1,
                emission_time_ms=10.0,
                prompt="p1",
                output_length=4,
                slo_ratio=-50.0,
            ),
        ]

        records = build_request_records(
            workload=workload,
            raw_output=[
                (0, [1] * 20, [1]),
                (1, [2] * 10, [1]),
            ],
            seq_id_to_request_id={0: 0, 1: 1},
            total_run_time_s=0.4,
            baseline_latency_per_token_ms=30.0,
        )

        self.assertEqual(
            records,
            [
                RequestRecord(
                    request_id=0,
                    slo_ratio=1.0,
                    arrival_time_ms=0.0,
                    decode_start_time_ms=0.0,
                    finish_time_ms=400.0,
                    num_generated_tokens=20,
                    attained=True,
                ),
                RequestRecord(
                    request_id=1,
                    slo_ratio=-50.0,
                    arrival_time_ms=10.0,
                    decode_start_time_ms=10.0,
                    finish_time_ms=410.0,
                    num_generated_tokens=10,
                    attained=True,
                ),
            ],
        )

    def test_run_workload_submits_requests_and_uses_formal_metrics(self):
        class FakeSamplingParams:
            def __init__(self, temperature, ignore_eos, max_tokens):
                self.temperature = temperature
                self.ignore_eos = ignore_eos
                self.max_tokens = max_tokens

        class FakeSystem:
            def __init__(self):
                self.requests = []
                self.exited = False
                self.next_seq_id = 10

            def add_request(self, prompt, sampling_params, slo_ratio, request_id=None):
                seq_id = self.next_seq_id
                self.next_seq_id += 1
                self.requests.append((seq_id, prompt, sampling_params.max_tokens, slo_ratio))
                return seq_id

            def run(self):
                return [
                    (10, [101] * 8, [1]),
                    (11, [202] * 4, [1]),
                ], 0.4

            def exit(self):
                self.exited = True

        fake_system = FakeSystem()

        args = types.SimpleNamespace(
            system="adaserve",
            input_file="unused.json",
            baseline_latency_per_token_ms=30.0,
            temperature=0.0,
            ignore_eos=True,
            num_pearl_steps=100,
        )

        workload = [
            WorkloadRequest(0, 0.0, "p0", 8, 1.0),
            WorkloadRequest(1, 10.0, "p1", 4, 0.6),
        ]

        result = run_workload(
            args,
            load_workload_fn=lambda _: workload,
            create_system_fn=lambda *_: fake_system,
            sampling_params_cls=FakeSamplingParams,
            token_count_fn=lambda prompt: len(prompt),
        )

        self.assertEqual(
            fake_system.requests,
            [(10, "p0", 8, 1.0), (11, "p1", 4, 0.6)],
        )
        self.assertTrue(fake_system.exited)
        self.assertEqual(result["metrics"]["completed_requests"], 2)
        self.assertIn("system(adaserve)", result["result_text"])
        self.assertEqual(len(result["records"]), 2)

    def test_run_workload_aligns_outputs_by_seq_id_not_position(self):
        class FakeSamplingParams:
            def __init__(self, temperature, ignore_eos, max_tokens):
                self.temperature = temperature
                self.ignore_eos = ignore_eos
                self.max_tokens = max_tokens

        class FakeSystem:
            def __init__(self):
                self.seq_ids = [41, 99]
                self.calls = []
                self.exited = False

            def add_request(self, prompt, sampling_params, slo_ratio, request_id=None):
                seq_id = self.seq_ids[len(self.calls)]
                self.calls.append((seq_id, prompt, sampling_params.max_tokens, slo_ratio))
                return seq_id

            def run(self):
                # Deliberately reverse the output order to force seq_id-based alignment.
                return [
                    (99, [9] * 4, [1]),
                    (41, [4] * 8, [2]),
                ], 0.4

            def exit(self):
                self.exited = True

        fake_system = FakeSystem()
        args = types.SimpleNamespace(
            system="adaserve",
            input_file="unused.json",
            baseline_latency_per_token_ms=30.0,
            temperature=0.0,
            ignore_eos=True,
            num_pearl_steps=100,
        )
        workload = [
            WorkloadRequest(0, 0.0, "p0", 8, 1.0),
            WorkloadRequest(1, 10.0, "p1", 4, 0.6),
        ]

        result = run_workload(
            args,
            load_workload_fn=lambda _: workload,
            create_system_fn=lambda *_: fake_system,
            sampling_params_cls=FakeSamplingParams,
            token_count_fn=lambda prompt: len(prompt),
        )

        self.assertEqual(
            [record.request_id for record in result["records"]],
            [0, 1],
        )
        self.assertEqual(
            [record.num_generated_tokens for record in result["records"]],
            [8, 4],
        )

    def test_run_workload_recreates_nonreusable_system_between_batches(self):
        class FakeSamplingParams:
            def __init__(self, temperature, ignore_eos, max_tokens):
                self.temperature = temperature
                self.ignore_eos = ignore_eos
                self.max_tokens = max_tokens

        class FakeSystem:
            supports_batch_reuse = False

            def __init__(self, seq_base):
                self.seq_base = seq_base
                self.requests = []
                self.seq_id_to_request_id = {}
                self.exited = False

            def add_request(self, prompt, sampling_params, slo_ratio, request_id=None):
                seq_id = self.seq_base + len(self.requests)
                self.requests.append((seq_id, request_id, prompt, sampling_params.max_tokens, slo_ratio))
                self.seq_id_to_request_id[seq_id] = request_id
                return seq_id

            def run(self):
                return [
                    (seq_id, [seq_id] * max_tokens, [1])
                    for seq_id, _request_id, _prompt, max_tokens, _slo_ratio in self.requests
                ], 0.1

            def exit(self):
                self.exited = True

        created_systems = []

        def create_system(*_args):
            system = FakeSystem(seq_base=100 * (len(created_systems) + 1))
            created_systems.append(system)
            return system

        args = types.SimpleNamespace(
            system="slopearl",
            input_file="unused.json",
            baseline_latency_per_token_ms=30.0,
            temperature=0.0,
            ignore_eos=True,
            num_pearl_steps=100,
            max_num_seqs=8,
            max_num_batched_tokens=4,
        )
        workload = [
            WorkloadRequest(0, 0.0, "aa", 2, 1.0),
            WorkloadRequest(1, 1.0, "bbb", 3, 1.0),
            WorkloadRequest(2, 2.0, "cc", 2, 1.0),
        ]

        result = run_workload(
            args,
            load_workload_fn=lambda _: workload,
            create_system_fn=create_system,
            sampling_params_cls=FakeSamplingParams,
            token_count_fn=lambda prompt: len(prompt),
        )

        self.assertEqual(len(created_systems), 3)
        self.assertTrue(all(system.exited for system in created_systems))
        self.assertEqual(
            [record.request_id for record in result["records"]],
            [0, 1, 2],
        )

    def test_run_workload_accepts_partial_bench_output(self):
        class FakeSamplingParams:
            def __init__(self, temperature, ignore_eos, max_tokens):
                self.temperature = temperature
                self.ignore_eos = ignore_eos
                self.max_tokens = max_tokens

        class FakeSystem:
            def __init__(self):
                self.exited = False

            def add_request(self, prompt, sampling_params, slo_ratio, request_id=None):
                return len(prompt)

            def run(self):
                return [
                    (2, [2] * 6, [1]),
                    (3, [3] * 7, [1]),
                ], 0.5

            def exit(self):
                self.exited = True

        args = types.SimpleNamespace(
            system="pearl-spec",
            input_file="unused.json",
            baseline_latency_per_token_ms=30.0,
            temperature=0.0,
            ignore_eos=True,
            num_pearl_steps=100,
        )
        workload = [
            WorkloadRequest(0, 0.0, "aa", 8, 1.0),
            WorkloadRequest(1, 1.0, "bbb", 8, 1.0),
            WorkloadRequest(2, 2.0, "cccc", 8, 1.0),
        ]

        result = run_workload(
            args,
            load_workload_fn=lambda _: workload,
            create_system_fn=lambda *_: FakeSystem(),
            sampling_params_cls=FakeSamplingParams,
            token_count_fn=lambda prompt: len(prompt),
        )

        self.assertEqual(len(result["records"]), 2)
        self.assertEqual([record.request_id for record in result["records"]], [0, 1])


if __name__ == "__main__":
    unittest.main()
