import re
import unittest
from pathlib import Path


SOURCE = Path(
    "/root/ykxia/nano-PEARL/nano_pearl/pearl_engine/pearl_model_runner.py"
).read_text()

SLO_DRAFT_SOURCE = Path(
    "/root/ykxia/nano-PEARL/nano_pearl/pearl_engine_slo/slo_draft_runner.py"
).read_text()
SLO_TARGET_SOURCE = Path(
    "/root/ykxia/nano-PEARL/nano_pearl/pearl_engine_slo/slo_target_runner.py"
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

    def test_slo_draft_pearl_step_no_longer_asserts_on_prefill(self):
        match = re.search(
            r"class SLODraftRunner.*?def pearl_step\(self\):(.*?)def prepare_pearl_decode",
            SLO_DRAFT_SOURCE,
            re.S,
        )
        self.assertIsNotNone(match)
        body = match.group(1)
        self.assertNotIn('assert not is_prefill', body)

    def test_slo_target_pearl_step_no_longer_asserts_on_prefill(self):
        match = re.search(
            r"class SLOTargetRunner.*?def pearl_step\(self\):(.*?)@torch.inference_mode",
            SLO_TARGET_SOURCE,
            re.S,
        )
        self.assertIsNotNone(match)
        body = match.group(1)
        self.assertNotIn('assert not is_prefill', body)

    def test_draft_vllm_spec_step_uses_serialized_path(self):
        match = re.search(
            r"class DraftModelRunner.*?def vllm_spec_step\(self\):(.*?)@torch.inference_mode",
            SOURCE,
            re.S,
        )
        self.assertIsNotNone(match)
        body = match.group(1)
        self.assertIn("self.serialized_pearl_step()", body)
        self.assertNotIn("self.pearl_step()", body)

    def test_target_vllm_spec_step_uses_serialized_path(self):
        match = re.search(
            r"class TargetModelRunner.*?def vllm_spec_step\(self\):(.*?)@torch.inference_mode",
            SOURCE,
            re.S,
        )
        self.assertIsNotNone(match)
        body = match.group(1)
        self.assertIn("self.serialized_pearl_step()", body)
        self.assertNotIn("self.pearl_step()", body)

    def test_runner_source_defines_serialized_pearl_steps(self):
        self.assertEqual(SOURCE.count("def serialized_pearl_step(self):"), 2)


if __name__ == "__main__":
    unittest.main()
