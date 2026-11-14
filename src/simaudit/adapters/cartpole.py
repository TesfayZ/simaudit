"""Optional scalar Gymnasium adapter. Assignment methodology: Gebrekidan (2026)."""
import numpy as np
from .base import Adapter, Unsupported
from ..report import Feature, Modification

PARAMS = ('masscart','masspole','length','gravity','force_mag','tau')


def cartpole(parameters=None, *, consistent=True):
    import gymnasium
    from gymnasium.envs.classic_control.cartpole import CartPoleEnv
    params = dict(parameters or {})
    if set(params)-set(PARAMS): raise Unsupported('Unsupported CartPole parameter')
    if any(not np.isfinite(v) or v <= 0 for v in params.values()):
        raise ValueError('CartPole parameters must be positive and finite')

    def new():
        env = CartPoleEnv()
        env.reset(seed=0)
        for key,value in params.items(): setattr(env,key,value)
        if consistent:
            env.total_mass = env.masscart + env.masspole
            env.polemass_length = env.masspole * env.length
        return env

    env = new()
    defaults = CartPoleEnv()
    modifications = [Modification(k, getattr(defaults,k),v,
        'fresh consistent reference' if consistent else 'direct attribute assignment') for k,v in params.items()]
    if consistent:
        for k in ('total_mass','polemass_length'):
            if getattr(defaults,k)!=getattr(env,k):
                modifications.append(Modification(k,getattr(defaults,k),getattr(env,k),'rebuild dependent constant'))
    defaults.close()

    def run(e,s,u,h):
        e.state = np.array(s,dtype=np.float64,copy=True)
        e.steps_beyond_terminated = None
        e.tau = h
        e.step(int(u))
        return np.array(e.state,copy=True)

    def native(s,u,h):
        fresh = new()
        try: return run(fresh,s,u,h)
        finally: fresh.close()

    def factory(name,value):
        if name not in PARAMS: raise Unsupported(f'Assignment parameter not supported: {name}')
        values = {**params,name:value}
        return cartpole(values,consistent=False), cartpole(values,consistent=True)

    def distance(x,y):
        delta = x-y
        delta[2] = np.arctan2(np.sin(delta[2]),np.cos(delta[2]))
        return np.linalg.norm(delta)

    return Adapter('CartPole-v1',lambda s,u,h:run(env,s,u,h),native,distance,
        assignment_factory=factory, assignment_metadata={'partial_update': 'length' in params and not consistent,
            'basis': 'length is read directly, but polemass_length is cached; adapter-specific source inspection'}, dt=env.tau, smooth=True, declared_order=1,
        features=[Feature('dynamics','tested','Scalar unwrapped CartPole internal float64 state',True),
                  Feature('parameters','present_not_tested','Only the requested parameter is probed; other attributes remain untested',True),
                  Feature('wrappers','present_not_tested','TimeLimit/rendering/reward/termination excluded',False)],
        modifications=modifications, versions={'gymnasium':gymnasium.__version__},
        scope='Scalar CartPole internal one-step dynamics with explicit constant-rebuild reference',
        limitations=['Vector CartPole and static attribute discovery are not implemented.'])
