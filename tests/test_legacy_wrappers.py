import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from benchmark_slo import eval_slo_benchmark, eval_slo_trace


class LegacyWrapperTests(unittest.TestCase):
    def test_eval_slo_benchmark_warns_that_it_is_legacy(self):
        self.assertIn("legacy", eval_slo_benchmark.LEGACY_WARNING.lower())

    def test_eval_slo_trace_warns_that_it_is_legacy(self):
        self.assertIn("legacy", eval_slo_trace.LEGACY_WARNING.lower())


if __name__ == "__main__":
    unittest.main()
