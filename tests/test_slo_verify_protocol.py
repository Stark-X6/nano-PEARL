import importlib.util
import unittest
from dataclasses import dataclass
from pathlib import Path


def _load_protocol_module():
    path = Path(
        "/root/ykxia/nano-PEARL/nano_pearl/pearl_engine_slo/slo_verify_protocol.py"
    )
    spec = importlib.util.spec_from_file_location("slo_verify_protocol", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


_PROTOCOL = _load_protocol_module()
build_verify_payload = _PROTOCOL.build_verify_payload
count_next_round_tokens = _PROTOCOL.count_next_round_tokens
count_verify_tokens = _PROTOCOL.count_verify_tokens


@dataclass
class DummySeq:
    seq_id: int
    token_ids: list[int]
    pre_verify: bool


class SLOVerifyProtocolTests(unittest.TestCase):
    def test_gamma_one_post_preverify_sends_one_verify_token(self):
        seq = DummySeq(seq_id=7, token_ids=[10, 11, 12], pre_verify=False)

        to_verify, next_round = build_verify_payload([seq], {7: 1})

        self.assertEqual(to_verify, [12])
        self.assertEqual(next_round, [12])
        self.assertEqual(count_verify_tokens([seq], {7: 1}), len(to_verify))
        self.assertEqual(count_next_round_tokens([seq], {7: 1}), len(next_round))

    def test_mixed_preverify_states_match_count_helpers(self):
        seqs = [
            DummySeq(seq_id=1, token_ids=[1, 2, 3, 4, 5, 6], pre_verify=False),
            DummySeq(seq_id=2, token_ids=[10, 11, 12, 13], pre_verify=True),
        ]
        gamma_map = {1: 3, 2: 2}

        to_verify, next_round = build_verify_payload(seqs, gamma_map)

        self.assertEqual(to_verify, [2, 3, 4, 12])
        self.assertEqual(next_round, [4, 5, 6, 12, 13])
        self.assertEqual(count_verify_tokens(seqs, gamma_map), len(to_verify))
        self.assertEqual(count_next_round_tokens(seqs, gamma_map), len(next_round))


if __name__ == "__main__":
    unittest.main()
