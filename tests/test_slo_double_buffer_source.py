import re
import unittest
from pathlib import Path

SOURCE = Path(
    "/root/ykxia/nano-PEARL/nano_pearl/pearl_engine_slo/slo_draft_runner.py"
).read_text()


class SLODoubleBufferSourceTests(unittest.TestCase):
    def test_draft_batch_double_buffer_reserves_block_capacity_before_decode(self):
        match = re.search(
            r"def draft_batch_double_buffer\(self, seqs\):(.*?)def verify_double_buffer_send",
            SOURCE,
            re.S,
        )
        self.assertIsNotNone(match)
        body = match.group(1)
        self.assertIn("can_append", body)
        self.assertIn("may_append", body)
        self.assertLess(body.index("may_append"), body.index("prepare_pearl_decode"))


if __name__ == "__main__":
    unittest.main()
