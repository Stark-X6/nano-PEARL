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
            num_tokens=[20, 10],
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

            def add_request(self, prompt, sampling_params, slo_ratio):
                self.requests.append((prompt, sampling_params.max_tokens, slo_ratio))

            def run(self):
                return ["o0", "o1"], [8, 4], [[1], [1]], 0.4, {"unused": True}

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
        )

        self.assertEqual(
            fake_system.requests,
            [("p0", 8, 1.0), ("p1", 4, 0.6)],
        )
        self.assertTrue(fake_system.exited)
        self.assertEqual(result["metrics"]["completed_requests"], 2)
        self.assertIn("system(adaserve)", result["result_text"])
        self.assertEqual(len(result["records"]), 2)


if __name__ == "__main__":
    unittest.main()
