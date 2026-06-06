import re
import unittest
from pathlib import Path


SOURCE = Path(
    "/root/ykxia/nano-PEARL/nano_pearl/pearl_engine/pearl_model_runner.py"
).read_text()


class PearlStepSourceTests(unittest.TestCase):
    def test_draft_pearl_step_no_longer_asserts_on_prefill(self):
        match = re.search(
            r"class DraftModelRunner.*?def pearl_step\(self\):(.*?)def vllm_spec_step",
            SOURCE,
            re.S,
        )
        self.assertIsNotNone(match)
        body = match.group(1)
        self.assertNotIn('assert not is_prefill', body)

    def test_target_pearl_step_no_longer_asserts_on_prefill(self):
        match = re.search(
            r"class TargetModelRunner.*?def pearl_step\(self\):(.*?)def vllm_spec_step",
            SOURCE,
            re.S,
        )
        self.assertIsNotNone(match)
        body = match.group(1)
        self.assertNotIn('assert not is_prefill', body)


if __name__ == "__main__":
    unittest.main()
