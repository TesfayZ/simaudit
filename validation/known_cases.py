"""Small paper-derived controls. See THIRD_PARTY.md for method and code credit."""
import numpy as np
from simaudit import Adapter, Feature, Modification


def acrobot(variant='book',mass=1.0):
    import gymnasium
    from gymnasium.envs.classic_control.acrobot import AcrobotEnv
    env=AcrobotEnv(); env.reset(seed=0)
    def step(s,u,h,target=env):
        target.state=np.asarray(s,dtype=float).copy()
        target.dt=h; target.book_or_nips=variant; target.LINK_MASS_2=mass
        target.step(int(u))
        return target.state.copy()
    def native(s,u,h):
        e=AcrobotEnv(); e.reset(seed=0)
        try: return step(s,u,h,e)
        finally: e.close()
    return Adapter('Acrobot paper control',step,native,dt=.2,
        features=[Feature('dynamics','tested','Shipped Acrobot, internal state',True)],
        versions={'gymnasium':gymnasium.__version__},
        modifications=[Modification('book_or_nips','book',variant,'Explicit paper control'),
                       Modification('LINK_MASS_2',1.0,mass,'Explicit paper control')])


def acrobot_book(): return acrobot()
def acrobot_nips(): return acrobot('nips')
def acrobot_mass(): return acrobot(mass=1.3)


def heavy_top(kind):
    from simaudit.adapters.mujoco import mujoco_xml
    joint='<joint type="ball"/>' if kind=='ball' else ''.join(
        f'<joint type="hinge" axis="{a}"/>' for a in ('1 0 0','0 1 0','0 0 1'))
    # Geometry adapted from dafd benchmark_quaternion_mechanism.py (MIT).
    xml=f'''<mujoco><option gravity="0 0 -9.8" integrator="RK4" timestep="0.01"/>
    <worldbody><body>{joint}<geom type="box" size=".2 .1 .05" pos="0 0 .3"
    mass="1" contype="0" conaffinity="0"/></body></worldbody></mujoco>'''
    a=mujoco_xml(xml)
    a.modifications.append(Modification('geometry collision masks',[1,1],[0,0],
        'Paper control: contacts disabled; collision-free geometry compiled explicitly'))
    a.limitations.append('Paper heavy-top geometry with collision masks disabled; not shipped benchmark dynamics.')
    return a
