import re
import unittest
from pathlib import Path

DRAFT_SOURCE = Path(
    "/root/ykxia/nano-PEARL/nano_pearl/pearl_engine_slo/slo_draft_runner.py"
).read_text()
TARGET_SOURCE = Path(
    "/root/ykxia/nano-PEARL/nano_pearl/pearl_engine_slo/slo_target_runner.py"
).read_text()


class SLODoubleBufferSourceTests(unittest.TestCase):
    def test_draft_batch_double_buffer_reserves_block_capacity_before_decode(self):
        match = re.search(
            r"def draft_batch_double_buffer\(self, seqs\):(.*?)def verify_double_buffer_send",
            DRAFT_SOURCE,
            re.S,
        )
        self.assertIsNotNone(match)
        body = match.group(1)
        self.assertIn("can_append", body)
        self.assertIn("may_append", body)
        self.assertLess(body.index("may_append"), body.index("prepare_pearl_decode"))

    def test_draft_benchmark_loop_filters_active_batches_before_reuse(self):
        match = re.search(
            r"def slo_bench_generate_double_buffer\(self, num_pearl_steps=100\):(.*?)def _write_output_to_shm",
            DRAFT_SOURCE,
            re.S,
        )
        self.assertIsNotNone(match)
        body = match.group(1)
        self.assertIn("_filter_active_batch", body)
        self.assertIn("self.pearl_step()", body)

    def test_target_benchmark_loop_filters_active_batches_before_reuse(self):
        match = re.search(
            r"def slo_bench_generate_double_buffer\(self, num_pearl_steps=100\):(.*?)def _write_output_to_shm",
            TARGET_SOURCE,
            re.S,
        )
        self.assertIsNotNone(match)
        body = match.group(1)
        self.assertIn("_filter_active_batch", body)
        self.assertIn("self.pearl_step()", body)


if __name__ == "__main__":
    unittest.main()
