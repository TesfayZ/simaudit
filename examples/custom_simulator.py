"""A standalone NumPy simulator, usable without Gymnasium or MuJoCo.

Replace advance and native_reference with your simulator's replay wrappers.
Native reference must bypass any adapter transformation under test.
"""
import numpy as np
from simaudit import Adapter, Feature


def advance(state, action, dt):
    # Explicit Euler for x'=x+u.
    return state + dt*(state+action)


def native_reference(state, action, dt):
    return np.asarray([(1+dt)*state[0]+dt*action])


def factory():
    return Adapter('user exponential simulator', advance, native_reference,
                   features=[Feature('dynamics','tested','Smooth scalar ODE; no hidden state',True)],
                   smooth=True, declared_order=1, dt=0.2,
                   observe_fn=lambda s,u,h:advance(s,u,h),
                   reference_observe_fn=lambda s,u,h:native_reference(s,u,h))
