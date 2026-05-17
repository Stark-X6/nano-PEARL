import json
import tempfile
import unittest
from pathlib import Path

from benchmark_slo.generate_trace import generate_trace


class GenerateTraceCompatibilityTests(unittest.TestCase):
    def test_generate_trace_writes_canonical_adaserve_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "trace.json"
            prompts = ["prompt-a", "prompt-b"]
            trace = generate_trace(
                str(output_path),
                num_requests=2,
                rps=2.0,
                slo_ratios=[(1.0, 1.0)],
                output_length=32,
                prompts=prompts,
                seed=0,
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(trace, payload)
        self.assertEqual(set(payload[0].keys()), {"emission_time_ms", "prompt", "output_length", "slo_ratio"})
        self.assertEqual(payload[0]["prompt"], "prompt-a")
        self.assertEqual(payload[1]["prompt"], "prompt-b")
        self.assertEqual(payload[0]["output_length"], 32)


if __name__ == "__main__":
    unittest.main()
