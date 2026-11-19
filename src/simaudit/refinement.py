"""Run-local fixed-interval endpoints for checks sharing the same replay probes."""
import math

import numpy as np

from .adapters import Blowup


def epsilon(value):
    return float(np.finfo(value.dtype).eps) if value.dtype.kind == 'f' else 0.0


def refinement_exponent(coarse, fine):
    """Log ratio for positive finite defects, retaining exact powers of two."""
    ratio = float(coarse) / float(fine)
    if 0 < ratio < math.inf:
        return math.log2(ratio)
    return math.log2(coarse) - math.log2(fine)


def refined_state(adapter, state, action, dt, count):
    x = state
    eps = epsilon(state)
    for _ in range(count):
        x = adapter.step(x, action, dt)
        if not np.all(np.isfinite(x)):
            raise Blowup('Nonfinite refinement trajectory')
        eps = max(eps, epsilon(x))
    return x, eps


class Refinement:
    """One instance per immutable configuration, interval and ordered probe set."""
    def __init__(self):
        self.endpoints = {}

    def get(self, index, adapter, state, action, dt, count):
        key = (index, dt, count)
        if key not in self.endpoints:
            self.endpoints[key] = refined_state(adapter, state, action, dt, count)
        return self.endpoints[key]
