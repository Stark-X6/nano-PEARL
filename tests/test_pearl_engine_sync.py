import re
import unittest
from pathlib import Path

SOURCE = Path('/root/ykxia/nano-PEARL/nano_pearl/pearl_engine/pearl_model_runner.py').read_text()


class PearlEngineSynchronizationTests(unittest.TestCase):
    def test_init_shared_memory_does_not_gate_ack_on_rank_zero(self):
        init_shared_memory_match = re.search(
            r"def init_shared_memory\(self\):(.*?)def init_model_and_kvcache",
            SOURCE,
            re.S,
        )
        self.assertIsNotNone(init_shared_memory_match)
        body = init_shared_memory_match.group(1)
        self.assertIn("self.control_event.set()", body)
        self.assertIn("if self.rank == 0:", body)
        self.assertNotIn(
            'logger.info(f"[Sub-Process] Draft Model and Target Model initialized. Starting to run the model...", color="yellow")\n            self.control_event.set()',
            body,
        )

    def test_loop_does_not_only_signal_rank_zero(self):
        loop_match = re.search(
            r"def loop\(self\):(.*?)def read_shm",
            SOURCE,
            re.S,
        )
        self.assertIsNotNone(loop_match)
        body = loop_match.group(1)
        self.assertNotIn('if self.rank == 0 and method_name != "exit":', body)
        self.assertIn('method_name != "exit"', body)
        self.assertIn('self.control_event.set()', body)

    def test_loop_uses_finally_to_avoid_wait_deadlocks_on_errors(self):
        loop_match = re.search(
            r"def loop\(self\):(.*?)def read_shm",
            SOURCE,
            re.S,
        )
        self.assertIsNotNone(loop_match)
        body = loop_match.group(1)
        self.assertIn('finally:', body)


if __name__ == '__main__':
    unittest.main()
