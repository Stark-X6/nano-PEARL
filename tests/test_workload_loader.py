import json
import tempfile
import unittest
from pathlib import Path

from benchmark_slo.workload_loader import WorkloadRequest, load_workload


class WorkloadLoaderTests(unittest.TestCase):
    def test_load_workload_parses_canonical_adaserve_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workload_path = Path(tmpdir) / "trace.json"
            workload_path.write_text(
                json.dumps(
                    [
                        {
                            "emission_time_ms": 0.0,
                            "prompt": "hello",
                            "output_length": 16,
                            "slo_ratio": 1.2,
                        },
                        {
                            "emission_time_ms": 12.5,
                            "prompt": "world",
                            "output_length": 32,
                            "slo_ratio": -50.0,
                        },
                    ]
                ),
                encoding="utf-8",
            )

            requests = load_workload(str(workload_path))

        self.assertEqual(
            requests,
            [
                WorkloadRequest(
                    request_id=0,
                    emission_time_ms=0.0,
                    prompt="hello",
                    output_length=16,
                    slo_ratio=1.2,
                ),
                WorkloadRequest(
                    request_id=1,
                    emission_time_ms=12.5,
                    prompt="world",
                    output_length=32,
                    slo_ratio=-50.0,
                ),
            ],
        )

    def test_load_workload_rejects_missing_required_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            workload_path = Path(tmpdir) / "bad-trace.json"
            workload_path.write_text(
                json.dumps(
                    [
                        {
                            "emission_time_ms": 0.0,
                            "prompt": "hello",
                            "output_length": 16,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "slo_ratio"):
                load_workload(str(workload_path))


if __name__ == "__main__":
    unittest.main()
