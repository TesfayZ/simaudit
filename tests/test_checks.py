import json
import math
import sys
import numpy as np
import pytest
from simaudit import *
from simaudit.release import collect


def analytic(method='euler', dtype='float64'):
    def step(s,u,h):
        factor = 1+h if method=='euler' else 1+h+h*h/2+h**3/6+h**4/24
        return np.asarray(s*factor,dtype=dtype)
    return Adapter('analytic',step,step,features=[Feature('dynamics','tested','Scalar exponential',True)],
                   smooth=True,declared_order=1 if method=='euler' else 4,dtype=dtype,dt=.2)


@pytest.mark.parametrize('method',['euler','rk4'])
def test_known_order(method):
    r=check_order(analytic(method),[[.5],[1],[1.5]],[0]*3)
    assert r.status==Status.PASS
    assert len(r.modifications)==3
    assert Report.loads(Report([r]).dumps()).counts['PASS']==1


def test_misdeclared_order():
    a=analytic(); a.declared_order=4
    assert check_order(a,[[1]],[0]).status==Status.FAIL


@pytest.mark.parametrize('kind',['float32','floor','noise','smooth','missing'])
def test_non_assessment(kind):
    a=analytic(dtype='float32' if kind=='float32' else 'float64')
    if kind=='floor': a.step_fn=a.native_step_fn=lambda s,u,h:s
    if kind=='noise': a.noise_floor=.01
    if kind=='smooth': a.smooth=None
    if kind=='missing': a.native_step_fn=None
    r=check_order(a,[[1]],[0]); assert r.status in {Status.ABSTAIN,Status.PARTIAL,Status.SKIPPED}
    assert Report([r]).exit_code()==0
    assert Report([r]).exit_code(True)==1
    assert 'not an all-clear' in Report([r]).summary()


def test_native_failure_blocks():
    a=analytic(); a.native_step_fn=lambda s,u,h:s+2
    r=check_order(a,[[1]],[0]); assert r.fault_class=='adapter'
    assert 'probes' not in r.evidence


def test_restoration_failure_blocks():
    a=analytic(); count=[0]
    def contaminated(s,u,h):
        count[0]+=1
        return s*(1+h)+(0 if count[0]==1 else .01)
    a.step_fn=contaminated
    assert check_order(a,[[1]],[0]).fault_class=='adapter'


def test_blowup():
    a=analytic(); a.step_fn=lambda s,u,h:np.full_like(s,np.nan,dtype=float)
    assert check_order(a,[[1]],[0]).status==Status.BLOWUP


@pytest.mark.parametrize('dtype,error,status',[('float64',0,Status.PASS),('float64',.1,Status.FAIL),
                                               ('float32',0,Status.ABSTAIN),('float32',.1,Status.FAIL)])
def test_observation_floor(dtype,error,status):
    a=analytic(); a.observe_fn=lambda s,u,h:np.asarray(s+error,dtype=dtype)
    a.reference_observe_fn=lambda s,u,h:np.asarray(s,dtype=dtype)
    assert check_observation(a,[[1]],[0]).status==status


def test_schema_rejection():
    text=Report([check_order(analytic(),[[1]],[0])]).dumps()
    for mutation in (lambda d:d.update(schema_version='2'),lambda d:d.update(extra=1),
                     lambda d:d['results'][0].update(extra=1),
                     lambda d:d['results'][0]['coverage'][0].update(extra=1)):
        d=json.loads(text); mutation(d)
        with pytest.raises(ValueError): Report.loads(json.dumps(d))
    with pytest.raises(ValueError): Report.loads(text.replace('0.2','NaN',1))
    with pytest.raises(ValueError): Report.loads('{"schema_version":"1.0","schema_version":"1.0","results":[]}')
    assert Report([]).exit_code()==1


def test_unknown_feature_visible():
    a=analytic(); a.features.append(Feature('alien_force','tested','Unrecognised force',True))
    r=check_order(a,[[1]],[0])
    assert r.coverage[-1].status=='unsupported'
    assert 'alien_force' in Report([r]).summary()


def test_invalid_inputs():
    with pytest.raises(ValueError): check_order(analytic(),[],[])
    with pytest.raises(ValueError): check_order(analytic(),[[1]],[0],interval=0)
    with pytest.raises(ValueError): check_order(analytic(),[[1]],[0],atol=-1)


@pytest.mark.parametrize('parameter,value,expected',[('masscart',2,'no_op'),('masspole',.2,'stale'),
                                                     ('length',1,'partly_stale'),('gravity',19.6,'consistent')])
def test_cartpole_known_answers(parameter,value,expected):
    pytest.importorskip('gymnasium')
    from simaudit.adapters.cartpole import cartpole
    s=np.random.default_rng(20261111).uniform(-.1,.1,(10,4))
    r=check_assignment(cartpole(),parameter,value,s,[0,1]*5)
    assert r.evidence['classification']==expected
    assert r.status==(Status.PASS if expected=='consistent' else Status.FAIL)
    assert r.modifications


def test_release_isolation_and_missing():
    spec='simaudit.adapters.cartpole:cartpole'
    pytest.importorskip('gymnasium')
    s=[[.1,.1,.1,.1]]
    r=compare_releases(spec,sys.executable,sys.executable,s,[1],steps=2)
    assert r.status==Status.PASS
    assert len(r.evidence['workers'])==2
    r=compare_releases(spec,'/does/not/exist',sys.executable,s,[1])
    assert r.status==Status.SKIPPED


def test_cli_custom(tmp_path):
    import subprocess,os
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONPATH=str(root/'src')+os.pathsep+str(root/'examples'))
    run=subprocess.run([sys.executable,'-m','simaudit','order','--factory','custom_simulator:factory',
                        '--probes',str(root/'examples/probes.json'),'--strict','--json',str(tmp_path/'r.json')],
                       env=env,text=True,capture_output=True)
    assert run.returncode==0,run.stderr
    assert Report.loads((tmp_path/'r.json').read_text()).counts['PASS']==1


def test_unlogged_assignment_blocks():
    a=analytic()
    a.assignment_factory=lambda k,v:(analytic(),analytic())
    r=check_assignment(a,'mass',2,[[1.]],[0])
    assert r.status==Status.FAIL and r.fault_class=='adapter'


def test_noisy_replay_abstains():
    a=analytic(); a.noise_floor=.1
    a.native_step_fn=lambda s,u,h:s*(1+h)+.01
    assert check_order(a,[[1.]],[0]).status==Status.ABSTAIN


def test_worker_changed_dynamics(tmp_path):
    import os
    import subprocess
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONPATH=str(root/'src')+os.pathsep+str(root/'validation'))
    code="""
import sys
from simaudit import compare_releases,Status
r=compare_releases('known_cases:acrobot_book',sys.executable,sys.executable,
                   [[.1,.2,.1,-.2]],[1],factory_b='known_cases:acrobot_mass')
assert r.status==Status.FAIL,r.reason
"""
    pytest.importorskip('gymnasium')
    run=subprocess.run([sys.executable,'-c',code],env=env,text=True,capture_output=True)
    assert run.returncode==0,run.stdout+run.stderr


def test_float32_independent_rounding_is_not_adapter_fault():
    a=analytic(dtype='float32')
    a.native_step_fn=lambda s,u,h:np.asarray(np.asarray(s,dtype='float32')*(1+h),dtype='float32')
    assert check_order(a,[[1.234567]],[0]).status==Status.ABSTAIN


def test_adapter_copies_mutable_return_buffers():
    buffer=np.zeros(1)
    def step(s,u,h):
        buffer[:]=s*(1+h)
        return buffer
    a=Adapter('mutable buffer',step,step)
    first=a.step(np.array([1.]),0,.1)
    a.step(np.array([9.]),0,.1)
    assert first[0]==pytest.approx(1.1)


def test_float64_cast_does_not_hide_float32_observation_limit():
    a=analytic(dtype='float32')
    a.observe_fn=a.reference_observe_fn=lambda s,u,h:np.asarray(s,dtype='float64')
    assert check_observation(a,[[1.]],[0]).status==Status.ABSTAIN


def test_release_workers_do_not_inherit_callers_site_packages(monkeypatch, tmp_path):
    """Regression: PYTHONPATH must expose only the simaudit package to workers."""
    import subprocess, sys
    from simaudit import release
    seen = {}
    real_run = subprocess.run
    def spy(cmd, **kw):
        seen['pythonpath'] = kw['env']['PYTHONPATH']
        seen['entries'] = sorted(p.name for p in __import__('pathlib').Path(kw['env']['PYTHONPATH'].split(__import__('os').pathsep)[0]).iterdir())
        return real_run(cmd, **kw)
    monkeypatch.setattr(release.subprocess, 'run', spy)
    monkeypatch.delenv('PYTHONPATH', raising=False)
    monkeypatch.setenv('PYTHONPATH', str(__import__('pathlib').Path(release.__file__).resolve().parent.parent))
    release.compare_releases('examples_factory:f', sys.executable, sys.executable, [[0.0]], [0], steps=1)
    assert seen['entries'] == ['simaudit']
