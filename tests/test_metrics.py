import unittest

from benchmark_slo.metrics import RequestRecord, compute_metrics


class MetricsTests(unittest.TestCase):
    def test_compute_metrics_matches_adaserve_goodput_and_attainment(self):
        records = [
            RequestRecord(
                request_id=0,
                slo_ratio=1.0,
                arrival_time_ms=0.0,
                decode_start_time_ms=10.0,
                finish_time_ms=70.0,
                num_generated_tokens=2,
                attained=True,
            ),
            RequestRecord(
                request_id=1,
                slo_ratio=0.6,
                arrival_time_ms=5.0,
                decode_start_time_ms=20.0,
                finish_time_ms=140.0,
                num_generated_tokens=4,
                attained=False,
            ),
            RequestRecord(
                request_id=2,
                slo_ratio=-50.0,
                arrival_time_ms=7.0,
                decode_start_time_ms=25.0,
                finish_time_ms=75.0,
                num_generated_tokens=3,
                attained=True,
            ),
        ]

        metrics = compute_metrics(records, total_run_time_s=2.0)

        self.assertEqual(metrics["completed_requests"], 3)
        self.assertEqual(metrics["total_generated_tokens"], 9)
        self.assertEqual(metrics["goodput"], 2.5)
        self.assertEqual(metrics["slo_attainment"], 2 / 3)
        self.assertEqual(
            metrics["slo_attainment_by_scale"],
            {
                1.0: {"attained": 1, "total": 1, "rate": 1.0},
                0.6: {"attained": 0, "total": 1, "rate": 0.0},
                -50.0: {"attained": 1, "total": 1, "rate": 1.0},
            },
        )


if __name__ == "__main__":
    unittest.main()
