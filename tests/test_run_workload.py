import unittest

from benchmark_slo.run_workload import format_result_text


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


if __name__ == "__main__":
    unittest.main()
