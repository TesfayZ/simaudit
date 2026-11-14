"""Run in each chosen interpreter. Writes only the explicitly supplied output file."""
import sys
import numpy as np
from simaudit import *
from simaudit.adapters.cartpole import cartpole
from known_cases import heavy_top

rng=np.random.default_rng(20261111)
results=[]
states=rng.uniform(-.1,.1,(100,4))
for p,v in [('masscart',2),('masspole',.2),('length',1),('gravity',19.6)]:
    results.append(check_assignment(cartpole(),p,v,states,[0,1]*50))
try:
    import mujoco
except ImportError:
    results.append(Result('order',Status.SKIPPED,'MuJoCo not installed','Paper controls',
                          ['Unexecuted engine remains untested'],[Feature('dynamics','unsupported','Missing MuJoCo')]))
else:
    for kind in ('ball','hinges'):
        a=heavy_top(kind)
        q=[1,0,0,0] if kind=='ball' else [0,0,0]
        states=[np.r_[q,rng.uniform(-3,3,3)] for _ in range(24)]
        r=check_order(a,states,[0]*24,interval=.05)
        r.evidence['paper_case']='heavy_top_'+kind
        results.append(r)
    import gymnasium as gym
    from simaudit.adapters.mujoco import reacher
    env=gym.make('Reacher-v5').unwrapped
    states=[]
    for i in range(24):
        env.reset(seed=20261111+i)
        state=np.r_[env.data.qpos.copy(),env.data.qvel.copy()]
        state[env.model.nq:env.model.nq+2]=rng.uniform(-1,1,2)
        states.append(state)
    env.close()
    results.append(check_observation(reacher(),states,rng.uniform(-.5,.5,(24,2))))
report=Report(results)
report.write(sys.argv[1])
for r in results:
    print(r.check,r.status.value,r.reason)
    if r.check=='order' and 'probes' in r.evidence:
        from collections import Counter
        print(r.evidence['paper_case'],dict(Counter(p['status'] for p in r.evidence['probes'])))
