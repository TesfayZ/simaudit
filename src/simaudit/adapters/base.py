"""Simulator-independent callable adapter; no optional simulator dependencies."""
from dataclasses import dataclass, field
import platform
from copy import deepcopy
import numpy as np
from ..report import Feature, Modification


class Unsupported(RuntimeError):
    pass


class Blowup(RuntimeError):
    pass


@dataclass
class Adapter:
    name: str
    step_fn: object = None
    native_step_fn: object = None
    distance_fn: object = None
    observe_fn: object = None
    reference_observe_fn: object = None
    assignment_factory: object = None
    features: list[Feature] = field(default_factory=lambda: [
        Feature("dynamics", "unsupported", "User has not supplied a feature inventory")])
    modifications: list[Modification] = field(default_factory=list)
    versions: dict[str, str] = field(default_factory=dict)
    scope: str = "Only supplied states, actions and configurations"
    limitations: list[str] = field(default_factory=list)
    dtype: str = "float64"
    noise_floor: float = 0.0
    smooth: bool | None = None
    declared_order: float | None = None
    order_blockers: list[str] = field(default_factory=list)
    assignment_metadata: dict = field(default_factory=dict)
    dt: float = 0.01
    energy_fn: object = None
    conservative: bool | None = None

    def __post_init__(self):
        if np.dtype(self.dtype).kind != 'f':
            raise ValueError("Adapter dtype must be floating point")
        if not np.isfinite(self.noise_floor) or self.noise_floor < 0:
            raise ValueError("noise_floor must be finite and nonnegative")
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be positive and finite")
        self.versions = {"python": platform.python_version(), "numpy": np.__version__, **self.versions}

    @property
    def capabilities(self):
        return {name for name, fn in (("step", self.step_fn), ("native_step", self.native_step_fn),
                ("observation", self.observe_fn), ("reference_observation", self.reference_observe_fn),
                ("assignment", self.assignment_factory), ("energy", self.energy_fn)) if callable(fn)}

    def step(self, state, action, dt):
        return np.array(self.step_fn(np.array(state, copy=True), deepcopy(action), dt), copy=True)

    def native_step(self, state, action, dt):
        return np.array(self.native_step_fn(np.array(state, copy=True), deepcopy(action), dt), copy=True)

    def distance(self, a, b):
        a, b = np.asarray(a), np.asarray(b)
        if a.shape != b.shape:
            raise ValueError("State shapes differ")
        value = float(self.distance_fn(a, b) if self.distance_fn else np.linalg.norm(a.astype(float)-b.astype(float)))
        if not np.isfinite(value) or value < 0:
            raise Blowup("Nonfinite or negative state distance")
        return value
