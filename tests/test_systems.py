import types
import unittest

from benchmark_slo.systems import SUPPORTED_SYSTEMS, create_system


class _FakeEngine:
    def __init__(self):
        self.requests = []
        self.calls = []
        self.exited = False

    def add_request(self, prompt, sampling_params, slo_ratio=None):
        self.requests.append(
            {
                "prompt": prompt,
                "sampling_params": sampling_params,
                "slo_ratio": slo_ratio,
            }
        )

    def vllm_spec_bench_generate(self, num_pearl_steps=100):
        self.calls.append(("vllm_spec_bench_generate", num_pearl_steps))
        return [], [], [], 1.0

    def bench_generate(self, num_pearl_steps=100):
        self.calls.append(("bench_generate", num_pearl_steps))
        return [], [], [], 1.0

    def slo_bench_generate(self, num_pearl_steps=100):
        self.calls.append(("slo_bench_generate", num_pearl_steps))
        return [], [], [], 1.0, {"goodput": 0.0}

    def slo_bench_generate_double_buffer(self, num_pearl_steps=100):
        self.calls.append(("slo_bench_generate_double_buffer", num_pearl_steps))
        return [], [], [], 1.0, {"goodput": 0.0}

    def exit(self):
        self.exited = True


class SystemsTests(unittest.TestCase):
    def test_supported_systems_match_formal_experiment_names(self):
        self.assertEqual(
            SUPPORTED_SYSTEMS,
            ("vllm-spec", "pearl-spec", "adaserve", "slopearl"),
        )

    def test_create_system_routes_to_expected_engine_methods(self):
        args = types.SimpleNamespace(num_pearl_steps=77)

        vllm_engine = _FakeEngine()
        pearl_engine = _FakeEngine()
        adaserve_engine = _FakeEngine()
        slopearl_engine = _FakeEngine()

        vllm_system = create_system(
            "vllm-spec",
            args,
            engine_factory=lambda *_: vllm_engine,
        )
        pearl_system = create_system(
            "pearl-spec",
            args,
            engine_factory=lambda *_: pearl_engine,
        )
        adaserve_system = create_system(
            "adaserve",
            args,
            engine_factory=lambda *_: adaserve_engine,
        )
        slopearl_system = create_system(
            "slopearl",
            args,
            engine_factory=lambda *_: slopearl_engine,
        )

        self.assertEqual(vllm_system.run(), ([], [], [], 1.0))
        self.assertEqual(pearl_system.run(), ([], [], [], 1.0))
        self.assertEqual(adaserve_system.run(), ([], [], [], 1.0, {"goodput": 0.0}))
        self.assertEqual(slopearl_system.run(), ([], [], [], 1.0, {"goodput": 0.0}))

        self.assertEqual(vllm_engine.calls, [("vllm_spec_bench_generate", 77)])
        self.assertEqual(pearl_engine.calls, [("bench_generate", 77)])
        self.assertEqual(adaserve_engine.calls, [("slo_bench_generate", 77)])
        self.assertEqual(slopearl_engine.calls, [("slo_bench_generate_double_buffer", 77)])

    def test_create_system_controls_slo_argument_passing(self):
        args = types.SimpleNamespace(num_pearl_steps=10)

        pearl_engine = _FakeEngine()
        adaserve_engine = _FakeEngine()

        pearl_system = create_system(
            "pearl-spec",
            args,
            engine_factory=lambda *_: pearl_engine,
        )
        adaserve_system = create_system(
            "adaserve",
            args,
            engine_factory=lambda *_: adaserve_engine,
        )

        pearl_system.add_request("prompt-a", sampling_params="sp", slo_ratio=1.4)
        adaserve_system.add_request("prompt-b", sampling_params="sp", slo_ratio=0.6)

        self.assertEqual(pearl_engine.requests[0]["slo_ratio"], None)
        self.assertEqual(adaserve_engine.requests[0]["slo_ratio"], 0.6)


if __name__ == "__main__":
    unittest.main()
