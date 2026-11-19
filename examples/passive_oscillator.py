"""Unit-mass, unit-stiffness oscillator in unforced motion, advanced with RK4."""
import numpy as np

from simaudit import Adapter, Feature, Unsupported


def step(state, action, h):
    if action != 0:
        raise Unsupported('Oscillator energy example supports zero applied force')

    def dynamics(x):
        return np.array([x[1], -x[0]])

    k1 = dynamics(state)
    k2 = dynamics(state + h * k1 / 2)
    k3 = dynamics(state + h * k2 / 2)
    k4 = dynamics(state + h * k3)
    return state + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def native_reference(state, action, h):
    # RK4 update matrix for this linear ODE, evaluated independently.
    position, velocity = state
    diagonal = 1 - h**2 / 2 + h**4 / 24
    off_diagonal = h - h**3 / 6
    return np.array([diagonal * position + off_diagonal * velocity,
                     diagonal * velocity - off_diagonal * position])


def factory():
    return Adapter('Passive oscillator', step, native_reference, dt=0.4,
                   energy_fn=lambda s: 0.5 * np.dot(s, s),
                   smooth=True, conservative=True, declared_order=4,
                   scope='Zero forcing, unit mass and stiffness, E=(x²+v²)/2 in oscillator units',
                   features=[Feature('dynamics', 'tested', 'Linear conservative oscillator', True)])
