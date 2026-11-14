"""Prospective software controls from REQUIREMENTS.md; no optional simulator needed."""
import argparse
import math
from collections import Counter
import json
import numpy as np
from simaudit import *


def factory(method='euler',dtype='float64'):
    def advance(s,u,h):
        factor=1+h if method=='euler' else 1+h+h*h/2+h**3/6+h**4/24
        return np.asarray(s*factor,dtype=dtype)
    # Independently evaluate the stability polynomial in the reference path.
    def native(s,u,h):
        coeff=[1,1] if method=='euler' else [1/24,1/6,.5,1,1]
        return np.asarray(s*np.polyval(coeff,h),dtype=dtype)
    return Adapter('analytic '+method,advance,native,dtype=dtype,
        features=[Feature('dynamics','tested','Scalar linear ODE',True)],
        smooth=True,declared_order=1 if method=='euler' else 4,dt=.2)


def wilson(k,n):
    z=1.959963984540054
    center=(k/n+z*z/(2*n))/(1+z*z/n)
    radius=z*math.sqrt(k/n*(1-k/n)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0,center-radius),min(1,center+radius)]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--report',required=True);parser.add_argument('--summary',required=True)
    args=parser.parse_args()
    rng=np.random.default_rng(20261111); states=rng.uniform(.5,1.5,(100,1)); actions=[0]*100
    results=[]; summaries=[]
    for method in ('euler','rk4'):
        a=factory(method)
        r=check_order(a,states,actions);results.append(r)
        counts=Counter(p['status'] for p in r.evidence['probes'])
        summaries.append(dict(case=method,n=100,counts=dict(counts),false_alarm_wilson95=wilson(counts['FAIL'],100)))
    low=factory(dtype='float32'); results.append(check_order(low,states,actions))
    noisy=factory();noisy.noise_floor=.001;results.append(check_order(noisy,states,actions))
    for dtype in ('float64','float32'):
        a=factory();a.observe_fn=lambda s,u,h:np.asarray(s,dtype=dtype)
        a.reference_observe_fn=lambda s,u,h:np.asarray(s,dtype=dtype)
        results.append(check_observation(a,states,actions))
    Report(results).write(args.report)
    with open(args.summary,'w') as f: json.dump(summaries,f,indent=2,allow_nan=False)
    for r in results: print(r.check,r.status.value,r.reason)


if __name__=='__main__': main()
