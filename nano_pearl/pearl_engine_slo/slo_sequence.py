"""
SLOSequence — wraps original Sequence with SLO metadata.
对应 AdaServe 中 Request 的 SLO 相关字段 (slo_ratio, decode_latency_ms).

Uses composition (not inheritance) to avoid issues with Sequence's
custom __getstate__ / __setstate__ for multiprocessing serialization.
"""

import time
from nano_pearl.pearl_engine.sequence import Sequence


class SLOSequence:
    """Wraps a Sequence with SLO metadata for budget-aware scheduling."""

    def __init__(self, seq: Sequence, slo_ratio: float = 1.0):
        self.seq = seq
        # SLO fields (对应 AdaServe Request 的 slo_ratio / decode_latency_ms)
        self.slo_ratio = slo_ratio
        self.decode_start_time: float = 0.0   # ms, set when decode phase begins
        self.decode_latency_ms: float = 0.0   # current accumulated decode latency
        self.assigned_gamma: int = 1           # assigned by SLOScheduler

    # ----- Delegate all Sequence attributes to self.seq -----
    def __getattr__(self, name):
        return getattr(self.seq, name)

    def __len__(self):
        return self.seq.num_tokens

    def __getitem__(self, key):
        return self.seq.token_ids[key]

    def __getstate__(self):
        """Serialize: SLO fields + underlying Sequence state."""
        seq_state = self.seq.__getstate__()
        return {
            'seq_state': seq_state,
            'slo_ratio': self.slo_ratio,
            'decode_start_time': self.decode_start_time,
            'decode_latency_ms': self.decode_latency_ms,
            'assigned_gamma': self.assigned_gamma,
        }

    def __setstate__(self, state):
        """Deserialize: restore Sequence + SLO fields."""
        # Reconstruct the Sequence from its serialized state
        self.seq = Sequence.__new__(Sequence)
        self.seq.__setstate__(state['seq_state'])
        self.slo_ratio = state['slo_ratio']
        self.decode_start_time = state['decode_start_time']
        self.decode_latency_ms = state['decode_latency_ms']
        self.assigned_gamma = state['assigned_gamma']

    # ----- SLO-specific helpers -----

    def update_decode_latency(self):
        """Update decode_latency_ms based on current time since decode start."""
        if self.decode_start_time > 0:
            self.decode_latency_ms = (time.time() * 1000) - self.decode_start_time

    def mark_decode_start(self):
        """Record the timestamp when decode phase begins."""
        self.decode_start_time = time.time() * 1000
