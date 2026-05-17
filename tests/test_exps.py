import os
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path("/root/ykxia/nano-PEARL")


class ExperimentShellTests(unittest.TestCase):
    def test_exps_test_sh_builds_qwen_runner_command_in_dry_run_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "result.txt"
            env = os.environ.copy()
            env.update(
                {
                    "DRY_RUN": "1",
                    "LLM_MODEL": "/models/qwen-target",
                    "SSM_MODEL": "/models/qwen-draft",
                    "DATASETS_FILE": "/tmp/workload.json",
                    "OUTPUT_FILE": str(output_file),
                    "DRAFT_TP": "1",
                    "TARGET_TP": "3",
                    "BASELINE_LATENCY_PER_TOKEN_MS": "28",
                    "MAX_NUM_SEQS": "80",
                    "MAX_NUM_BATCHED_TOKENS": "4096",
                    "ENABLE_ADASERVE": "ON",
                }
            )

            proc = subprocess.run(
                ["bash", str(REPO_ROOT / "exps/test.sh")],
                cwd=str(REPO_ROOT),
                env=env,
                text=True,
                capture_output=True,
                check=True,
            )

        self.assertIn("--system adaserve", proc.stdout)
        self.assertIn("--input-file /tmp/workload.json", proc.stdout)
        self.assertIn("--draft-model /models/qwen-draft", proc.stdout)
        self.assertIn("--target-model /models/qwen-target", proc.stdout)
        self.assertIn(f"> {output_file}", proc.stdout)

    def test_qwen_rps_script_expands_rps_range_and_system_names_in_dry_run_mode(self):
        env = os.environ.copy()
        env.update(
            {
                "DRY_RUN": "1",
                "QWEN_TARGET_MODEL": "/models/qwen-target",
                "QWEN_DRAFT_MODEL": "/models/qwen-draft",
                "RPS_MIN": "2.4",
                "RPS_MAX": "2.6",
                "RPS_STEP": "0.2",
                "OUTPUT_LENGTH": "256",
                "ENABLE_VLLM_SPEC": "ON",
                "ENABLE_SLOPEARL": "ON",
            }
        )

        proc = subprocess.run(
            ["bash", str(REPO_ROOT / "exps/fig8,9/run_qwen_rps.sh")],
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )

        self.assertIn("rps2.4_ol256.json", proc.stdout)
        self.assertIn("rps2.6_ol256.json", proc.stdout)
        self.assertIn("results/fig8,9/qwen/vllm-spec", proc.stdout)
        self.assertIn("results/fig8,9/qwen/slopearl", proc.stdout)


if __name__ == "__main__":
    unittest.main()
