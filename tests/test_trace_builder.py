import unittest

from benchmark_slo.trace_builder import build_trace


class TraceBuilderTests(unittest.TestCase):
    def test_build_trace_emits_canonical_adaserve_schema(self):
        prompts = ["prompt-a", "prompt-b", "prompt-c"]
        trace = build_trace(
            prompts=prompts,
            output_length=128,
            rps=2.0,
            slo_ratios=[(0.6, 0.5), (1.4, 0.5)],
            seed=0,
        )

        self.assertEqual(len(trace), 3)
        self.assertEqual(set(trace[0].keys()), {"emission_time_ms", "prompt", "output_length", "slo_ratio"})
        self.assertEqual(trace[0]["prompt"], "prompt-a")
        self.assertEqual(trace[1]["prompt"], "prompt-b")
        self.assertEqual(trace[2]["prompt"], "prompt-c")
        self.assertEqual(trace[0]["output_length"], 128)
        self.assertGreaterEqual(trace[1]["emission_time_ms"], trace[0]["emission_time_ms"])
        self.assertGreaterEqual(trace[2]["emission_time_ms"], trace[1]["emission_time_ms"])

    def test_build_trace_uses_given_slo_distribution(self):
        prompts = ["only-prompt"]
        trace = build_trace(
            prompts=prompts,
            output_length=64,
            rps=1.0,
            slo_ratios=[(-50.0, 1.0)],
            seed=123,
        )

        self.assertEqual(trace[0]["slo_ratio"], -50.0)


if __name__ == "__main__":
    unittest.main()
