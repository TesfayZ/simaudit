"""Optional native adapters. No model features are silently disabled."""
import hashlib
import numpy as np
from .base import Adapter, Blowup
from ..report import Feature


def mujoco_xml(xml):
    import mujoco as mj
    model = mj.MjModel.from_xml_string(xml)
    data = mj.MjData(model)
    presence = {
        'contacts':bool(model.npair or np.any(model.geom_contype) or np.any(model.geom_conaffinity)),
        'joint_limits':bool(np.any(model.jnt_limited)), 'actuators':bool(model.nu),
        'damping':bool(np.any(model.dof_damping)), 'friction':bool(np.any(model.dof_frictionloss)),
        'fluid':bool(model.opt.viscosity or model.opt.density), 'tendons':bool(model.ntendon),
        'equality_constraints':bool(model.neq), 'plugins':bool(model.nplugin),
        'flex':bool(model.nflex), 'mocap':bool(model.nmocap),
        'callbacks':any(getattr(mj,'get_mjcb_'+name,lambda:None)() is not None
                        for name in ('control','passive','sensor','act_dyn','act_gain','act_bias')),
    }
    blockers = [k for k,v in presence.items() if v]
    features = [Feature(k,'unsupported' if v else 'present_not_tested',
                        'Present: unsupported by autonomous smooth adapter' if v else 'Absent in compiled model',v)
                for k,v in presence.items()]
    features += [Feature('gravity','tested','Retained as compiled',True),
                 Feature('springs','tested','Retained as compiled',bool(np.any(model.jnt_stiffness)))]
    initial_dt = float(model.opt.timestep)

    def run(target,s,u,h):
        if blockers:
            from .base import Unsupported
            raise Unsupported('Restricted native replay cannot represent: '+', '.join(blockers))
        if s.shape != (model.nq+model.nv,): raise ValueError('Expected qpos followed by qvel')
        if np.any(np.asarray(u)!=0): raise ValueError('Autonomous adapter accepts only zero action')
        old = model.opt.timestep
        try:
            model.opt.timestep = h
            mj.mj_resetData(model,target)
            target.qpos[:] = s[:model.nq]
            target.qvel[:] = s[model.nq:]
            mj.mj_forward(model,target)
            mj.mj_step(model,target)
            if np.any(target.warning.number) or not np.isfinite(target.time) or abs(target.time-h)>1e-12:
                raise Blowup('MuJoCo warning or automatic reset during native step')
            return np.concatenate([target.qpos,target.qvel])
        finally:
            model.opt.timestep = old

    def distance(x,y):
        tangent = np.empty(model.nv)
        mj.mj_differentiatePos(model,tangent,1.0,y[:model.nq],x[:model.nq])
        return np.linalg.norm(np.concatenate([tangent,x[model.nq:]-y[model.nq:]]))

    declared = {int(mj.mjtIntegrator.mjINT_EULER):1, int(mj.mjtIntegrator.mjINT_RK4):4}.get(int(model.opt.integrator))
    if declared is None: blockers.append('unvalidated integrator family')
    return Adapter('MuJoCo XML',lambda s,u,h:run(data,s,u,h),
        lambda s,u,h:run(mj.MjData(model),s,u,h),distance,
        features=features, versions={'mujoco':mj.__version__,'model_sha256':hashlib.sha256(xml.encode()).hexdigest()},
        scope='Native engine step of supplied smooth autonomous XML; qpos/qvel only',
        limitations=['Only autonomous rigid models without listed unsupported features are supported.',
                     'No claim about frame-skip trajectories, contacts, constraints, or control callbacks.'],
        smooth=not blockers,order_blockers=blockers,declared_order=declared,dt=initial_dt)


def reacher():
    import gymnasium as gym
    import mujoco as mj
    env = gym.make('Reacher-v5').unwrapped
    env.reset(seed=0)
    model = env.model
    dt = float(env.dt)
    # Each probe begins from the same complete data snapshot, then specifies qpos/qvel.
    fields = mj.mjtState.mjSTATE_INTEGRATION
    snapshot = np.empty(mj.mj_stateSize(model, fields))
    mj.mj_getState(model, env.data, snapshot, fields)

    def restore(target, s):
        mj.mj_resetData(target.model, target.data)
        mj.mj_setState(target.model, target.data, snapshot, fields)
        target.set_state(s[:model.nq], s[model.nq:])

    def run(s,u,h,refresh=False):
        if h != dt: raise ValueError('Reacher adapter uses its native frame-skip interval')
        if s.shape != (model.nq+model.nv,): raise ValueError('Expected qpos followed by qvel')
        restore(env, s)
        observation,*_ = env.step(np.asarray(u,dtype=float))
        if np.any(env.data.warning.number): raise Blowup('MuJoCo warning during Reacher step')
        state = np.concatenate([env.data.qpos.copy(),env.data.qvel.copy()])
        if refresh:
            mj.mj_forward(model,env.data)
            observation = env._get_obs()
        return state,np.array(observation,copy=True)

    def native(s,u,h):
        other = gym.make('Reacher-v5').unwrapped
        try:
            other.reset(seed=0)
            restore(other, s)
            other.step(np.asarray(u,dtype=float))
            return np.concatenate([other.data.qpos.copy(),other.data.qvel.copy()])
        finally: other.close()

    return Adapter('Reacher-v5',lambda s,u,h:run(s,u,h)[0],native,
        observe_fn=lambda s,u,h:run(s,u,h)[1], reference_observe_fn=lambda s,u,h:run(s,u,h,True)[1],
        features=[Feature('observations','tested','Shipped observation versus mj_forward recomputation',True),
                  Feature('actuators','present_not_tested','Retained native control; actuation correctness untested',True),
                  Feature('dynamics','present_not_tested','Native replay checked; physics correctness untested',True),
                  Feature('wrappers','present_not_tested','Unwrapped; TimeLimit excluded',False)],
        versions={'gymnasium':gym.__version__,'mujoco':mj.__version__},dt=dt,
        limitations=['Reference reuses the shipped observation formula; this checks cache consistency only.',
                     'Fixed reset snapshot; no arbitrary user callbacks or wrappers are represented.'])
