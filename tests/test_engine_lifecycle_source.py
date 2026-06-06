import unittest
from pathlib import Path

PEARL_ENGINE_SOURCE = Path(
    "/root/ykxia/nano-PEARL/nano_pearl/pearl_engine/pearl_engine.py"
).read_text()
SLO_ENGINE_SOURCE = Path(
    "/root/ykxia/nano-PEARL/nano_pearl/pearl_engine_slo/slo_pearl_engine.py"
).read_text()


class EngineLifecycleSourceTests(unittest.TestCase):
    def test_pearl_engine_does_not_register_atexit_exit(self):
        self.assertNotIn("atexit.register(self.exit)", PEARL_ENGINE_SOURCE)

    def test_slo_engine_does_not_register_atexit_exit(self):
        self.assertNotIn("atexit.register(self.exit)", SLO_ENGINE_SOURCE)


if __name__ == "__main__":
    unittest.main()
