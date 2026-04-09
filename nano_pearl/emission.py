"""
Emission machines for SLO-aware request generation.

Port of AdaServe's EmissionMachine system from C++ (FlexFlow/inference.h).

Provides:
  - EmissionMachine: base class with SLO ratio sampling
  - RandomSLOEmission: all requests arrive at once with random SLO ratio (Mode A)
  - ConstantEmission: requests at fixed intervals with random SLO ratio
  - PoissonEmission: requests per Poisson process with random SLO ratio (Mode B)
  - TraceEmission: requests from JSON trace file with fixed timestamps + SLO (Mode B)
"""

import json
import math
import random
import time
from typing import Optional


class EmissionMachine:
    """Base class for request emission patterns.

    Corresponds to AdaServe's EmissionMachine (inference.h).
    Manages SLO ratio distribution sampling using cumulative probability.
    """

    def __init__(self, slo_ratios=None, seed=None):
        """Initialize with SLO ratio distribution.

        Args:
            slo_ratios: list of (ratio, weight) tuples. Weights should sum to 1.0.
                        Default: [(0.6, 0.25), (1.0, 0.25), (1.4, 0.25), (1.8, 0.25)]
            seed: Random seed for reproducibility.
        """
        if slo_ratios is None:
            slo_ratios = [(0.6, 0.25), (1.0, 0.25), (1.4, 0.25), (1.8, 0.25)]

        self.rng = random.Random(seed)

        # Convert weights to cumulative probabilities
        # Corresponds to AdaServe: for (i = 1; i < slo_ratios.size(); i++) slo_ratios[i].second += slo_ratios[i-1].second
        self.slo_ratios = []
        cum = 0.0
        for ratio, weight in slo_ratios:
            cum += weight
            self.slo_ratios.append((ratio, cum))
        # Normalize last entry to 1.0
        if self.slo_ratios:
            last_ratio, _ = self.slo_ratios[-1]
            self.slo_ratios[-1] = (last_ratio, 1.0)

        self.elapsed_time_ms = 0.0
        self.last_request_time_ms = 0.0

    def sample_slo_ratio(self):
        """Sample an SLO ratio from the cumulative distribution.

        Corresponds to AdaServe EmissionMachine::sample_slo_ratio():
          double r = uniform(0, 1);
          for (auto& [ratio, cum_prob] : slo_ratios):
              if (r <= cum_prob) return ratio;
        """
        r = self.rng.random()
        for ratio, cum_prob in self.slo_ratios:
            if r <= cum_prob:
                return ratio
        return self.slo_ratios[-1][0]  # fallback

    def get_next_interval_ms(self):
        """Time to wait before next request (ms). Override in subclasses."""
        return 0.0

    def wait_until_next_request(self):
        """Wait for the next request emission time."""
        interval_ms = self.get_next_interval_ms()
        if interval_ms > 0:
            time.sleep(interval_ms / 1000.0)
        self.elapsed_time_ms += interval_ms
        self.last_request_time_ms = self.elapsed_time_ms

    def get_elapsed_time_ms(self):
        return self.elapsed_time_ms


class RandomSLOEmission(EmissionMachine):
    """Mode A: all requests arrive at once with random SLO ratio.

    No waiting between requests. All are emitted immediately.
    Used for batch benchmarks where all prompts are ready.
    """

    def __init__(self, slo_ratios=None, seed=None):
        super().__init__(slo_ratios, seed)

    def get_next_interval_ms(self):
        return 0.0


class ConstantEmission(EmissionMachine):
    """Requests emitted at fixed intervals.

    Corresponds to AdaServe's ConstantEmissionMachine:
      interval_ms = 1000 / req_per_s
    """

    def __init__(self, req_per_s, slo_ratios=None, seed=None):
        super().__init__(slo_ratios, seed)
        self.req_per_s = req_per_s
        self.interval_ms = 1000.0 / req_per_s if req_per_s > 0 else 0.0

    def get_next_interval_ms(self):
        return self.interval_ms


class PoissonEmission(EmissionMachine):
    """Mode B: requests arrive per Poisson process with random SLO ratio.

    Inter-arrival times follow exponential distribution with lambda = req_per_s.
    Corresponds to AdaServe's PoissonEmissionMachine.
    """

    def __init__(self, req_per_s, slo_ratios=None, seed=None):
        super().__init__(slo_ratios, seed)
        self.req_per_s = req_per_s

    def get_next_interval_ms(self):
        # Exponential distribution: -ln(U) / lambda
        # Corresponds to AdaServe: exponential_distribution<double>(req_per_s)
        u = self.rng.random()
        if u <= 0:
            u = 1e-10
        return -math.log(u) / self.req_per_s * 1000.0


class TraceEmission(EmissionMachine):
    """Mode B variant: requests from JSON trace file.

    Trace format: [{"emission_time_ms": float, "slo_ratio": float, ...}, ...]

    Corresponds to AdaServe's TraceEmissionMachine:
      Uses pre-recorded timestamps and SLO ratios.
    """

    def __init__(self, trace_path=None, trace_data=None, slo_ratios=None, seed=None):
        super().__init__(slo_ratios, seed)

        if trace_data is not None:
            self.trace = trace_data
        elif trace_path is not None:
            with open(trace_path, 'r') as f:
                self.trace = json.load(f)
        else:
            self.trace = []

        self.trace_idx = 0

    def sample_slo_ratio(self):
        """Return SLO ratio from trace, advancing index.

        Corresponds to AdaServe TraceEmissionMachine::sample_slo_ratio():
          return trace[idx++].slo_ratio;
        """
        if self.trace_idx < len(self.trace):
            ratio = self.trace[self.trace_idx].get('slo_ratio', 1.0)
            self.trace_idx += 1
            return ratio
        return 1.0

    def get_next_interval_ms(self):
        """Compute interval from trace timestamps."""
        if self.trace_idx < len(self.trace):
            next_time = self.trace[self.trace_idx].get('emission_time_ms', 0.0)
            interval = next_time - self.elapsed_time_ms
            return max(0.0, interval)
        return 0.0

    def reset(self):
        """Reset trace to beginning for replay."""
        self.trace_idx = 0
        self.elapsed_time_ms = 0.0
        self.last_request_time_ms = 0.0
