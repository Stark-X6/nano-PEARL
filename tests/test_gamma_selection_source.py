import unittest
from pathlib import Path


class GammaSelectionSourceTests(unittest.TestCase):
    def test_runner_source_does_not_use_stopiteration_prone_next_lookup(self):
        source = Path(
            "/root/ykxia/nano-PEARL/nano_pearl/pearl_engine/pearl_model_runner.py"
        ).read_text()

        self.assertNotIn(
            'next(x for x in self.gamma_list if x >= len(self.scheduler.running))',
            source,
        )

    def test_runner_source_defines_gamma_resolution_helper(self):
        source = Path(
            "/root/ykxia/nano-PEARL/nano_pearl/pearl_engine/pearl_model_runner.py"
        ).read_text()

        self.assertIn("def resolve_batch_gamma", source)


if __name__ == "__main__":
    unittest.main()
