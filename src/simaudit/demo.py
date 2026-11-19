"""Small standalone oscillator fixture for the installed workflow commands."""
from dataclasses import replace
import sys

import numpy as np

from .adapters import Adapter, Unsupported
from .attribution import Hypothesis, Recovery
from .fit import EnergyFit
from .laws import Law, LawStudy, LogArm
from .report import Feature, Modification
from .workflow import WorkflowInputs


def oscillator(stiffness=1.0):
    def rhs(s):
        return np.array([s[1], -stiffness*s[0]])

    def step(s, u, h):
        if u != 0:
            raise Unsupported('Demo supports passive oscillator probes')
        k1 = rhs(s)
        k2 = rhs(s + h*k1/2)
        k3 = rhs(s + h*k2/2)
        return s + h*(k1 + 2*k2 + 2*k3 + rhs(s + h*k3))/6

    def native(s, u, h):
        z = stiffness*h*h
        return (1-z/2+z*z/24)*s + h*(1-z/6)*rhs(s)

    def assign(name, value):
        if name != 'stiffness':
            raise Unsupported('Only stiffness is supported')
        change = Modification(name, stiffness, value, 'Configure oscillator stiffness')
        return tuple(replace(oscillator(value), modifications=[change]) for _ in range(2))

    return Adapter('demo oscillator', step, native, dt=0.2, smooth=True, conservative=True,
                   declared_order=4, features=[Feature('dynamics','tested','Unit mass oscillator',True)],
                   energy_fn=lambda s: (s[1]**2 + stiffness*s[0]**2)/2, assignment_factory=assign,
                   observe_fn=lambda s,u,h: step(s,u,h)[:1],
                   reference_observe_fn=lambda s,u,h: native(s,u,h)[:1])


def inputs():
    rng = np.random.default_rng(312)
    states = rng.uniform(-1,1,(12,2)).tolist()
    def groups(indices):
        return {str(i): ([states[i]], [0]) for i in indices}
    fit = EnergyFit('stiffness', (0.4,2.0), lambda s: (s[1]**2/2,s[0]**2/2),
                    groups(range(8)), groups(range(8,12)), 1e-6, 0.001, 'Unit mass, oscillator units')
    coords = ('position','velocity')
    def arm(name):
        t = np.arange(24)*0.2
        data = {}
        for i in range(6):
            x,v = rng.uniform(-1,1,2)
            data[f'{name}-{i}'] = np.column_stack((x*np.cos(t)+v*np.sin(t), v*np.cos(t)-x*np.sin(t)))
        return LogArm(name,coords,data,dict(sampling='0.2',integration='analytic',noise='none'),True)
    law = Law('energy','conserved',{(2,0):0.5,(0,2):0.5})
    hypotheses = {name:Hypothesis(1.,(0.4,2.),{e:0.5}) for name,e in [('stiffness',(2,0)),('mass',(0,2))]}
    logs = LawStudy(arm('reference'),arm('candidate'),[law],{'energy':Recovery(coords,2,hypotheses)})
    return WorkflowInputs('unit oscillator demo', oscillator(), states, [0]*len(states),
                          assignment=('stiffness',1.2),
                          release=dict(factory='simaudit.demo:oscillator',python_a=sys.executable,python_b=sys.executable),
                          fit=fit, logs=logs, logs_use_adapter_observations=False)
