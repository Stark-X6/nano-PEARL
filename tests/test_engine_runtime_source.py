import unittest
from pathlib import Path

PEARL_ENGINE_SOURCE = Path(
    "/root/ykxia/nano-PEARL/nano_pearl/pearl_engine/pearl_engine.py"
).read_text()
SLO_ENGINE_SOURCE = Path(
    "/root/ykxia/nano-PEARL/nano_pearl/pearl_engine_slo/slo_pearl_engine.py"
).read_text()
RUNNER_SOURCE = Path(
    "/root/ykxia/nano-PEARL/nano_pearl/pearl_engine/pearl_model_runner.py"
).read_text()


class EngineRuntimeSourceTests(unittest.TestCase):
    def test_pearl_engine_sanitizes_runtime_env_before_spawning(self):
        self.assertIn("_sanitize_runtime_env", PEARL_ENGINE_SOURCE)
        self.assertIn("self._sanitize_runtime_env()", PEARL_ENGINE_SOURCE)
        self.assertIn('os.environ["OMP_NUM_THREADS"] = "1"', PEARL_ENGINE_SOURCE)

    def test_slo_engine_sanitizes_runtime_env_before_spawning(self):
        self.assertIn("_sanitize_runtime_env", SLO_ENGINE_SOURCE)
        self.assertIn("self._sanitize_runtime_env()", SLO_ENGINE_SOURCE)
        self.assertIn('os.environ["OMP_NUM_THREADS"] = "1"', SLO_ENGINE_SOURCE)

    def test_runner_uses_configurable_nccl_timeout_minutes(self):
        self.assertIn('PEARL_NCCL_TIMEOUT_MINUTES', RUNNER_SOURCE)
        self.assertNotIn('timeout=timedelta(minutes=1)', RUNNER_SOURCE)


if __name__ == "__main__":
    unittest.main()
