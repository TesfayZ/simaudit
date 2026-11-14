"""Run user factories in separate interpreters with identical inputs, without installing anything."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import numpy as np
from .adapters import Adapter
from .checks import guard, probes, result, tolerance
from .report import Feature, Report, Status


def load_factory(spec):
    module, separator, name = spec.partition(':')
    if not separator or not module or not name:
        raise ValueError('Factory must be module:function (a trusted local Python callable)')
    return getattr(importlib.import_module(module),name)()


def collect(adapter, states, actions, steps=1):
    states, actions = probes(states,actions)
    if type(steps) is not int or steps < 1: raise ValueError('steps must be a positive integer')
    blocked = guard(adapter,'release',states,actions,[adapter.dt])
    if blocked: return Report([blocked])
    trajectories = []
    from .adapters import Blowup
    try:
        for s,u in zip(states,actions):
            path = []
            for _ in range(steps):
                s = adapter.step(s,u,adapter.dt)
                if not np.all(np.isfinite(s)): raise Blowup('Nonfinite release trajectory')
                path.append(s.tolist())
            trajectories.append(path)
    except Blowup as exc:
        return Report([result(adapter,'release',Status.BLOWUP,str(exc))])
    return Report([result(adapter,'release',Status.PASS,'Worker replay self-test passed; trajectory collection completed',
        {'trajectories':trajectories,'dt':adapter.dt,'dtype':adapter.dtype,'noise_floor':adapter.noise_floor})])


def compare_releases(factory, python_a, python_b, states, actions, *, steps=5, atol=1e-13, timeout=60, factory_b=None):
    states, actions = probes(states,actions)
    atol = tolerance(atol)
    payload = dict(factory=factory,states=[s.tolist() for s in states],
                   actions=[np.asarray(u).tolist() for u in actions],steps=steps)
    encoded = json.dumps(payload,allow_nan=False)
    adapter = Adapter('isolated releases',features=[Feature('dynamics','tested','Paired worker first steps',True)],
        scope='Same supplied inputs and factory executed in two explicitly supplied interpreters')
    workers = []
    env = dict(os.environ)
    # Expose ONLY this package to the workers. Putting the caller's site-packages
    # on PYTHONPATH would shadow each interpreter's own numpy/simulator versions.
    isolated = tempfile.TemporaryDirectory(prefix='simaudit-worker-')
    package_dir = Path(__file__).resolve().parent
    shutil.copytree(package_dir, Path(isolated.name) / 'simaudit',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    inherited = env.get('PYTHONPATH')
    env['PYTHONPATH'] = isolated.name + (os.pathsep + inherited if inherited else '')
    try:
        return _run_workers(adapter, payload, encoded, factory, factory_b, python_a, python_b,
                            atol, timeout, env, workers)
    finally:
        isolated.cleanup()


def _run_workers(adapter, payload, encoded, factory, factory_b, python_a, python_b, atol, timeout, env, workers):
    for executable, worker_factory in ((python_a,factory),(python_b,factory_b or factory)):
        worker_input = json.dumps({**payload, "factory":worker_factory},allow_nan=False)
        try:
            run = subprocess.run([str(executable),'-m','simaudit.worker'],input=worker_input,text=True,
                                 capture_output=True,timeout=timeout,env=env)
        except (OSError,subprocess.TimeoutExpired) as exc:
            return result(adapter,'release',Status.SKIPPED,f'Worker unavailable or timed out: {executable}: {exc}',
                          {'untested_interpreter':str(executable)})
        if run.returncode:
            return result(adapter,'release',Status.SKIPPED,f'Worker could not collect: {executable}',
                          {'stderr':run.stderr[-4000:],'untested_interpreter':str(executable)})
        worker = Report.loads(run.stdout).results[0]
        workers.append(worker)
        if worker.status != Status.PASS:
            return result(adapter,'release',worker.status,'Worker did not pass: '+worker.reason,
                          {'worker':json.loads(Report([worker]).dumps())})
    left,right = workers
    adapter.versions = {**{f'a:{k}':v for k,v in left.versions.items()},**{f'b:{k}':v for k,v in right.versions.items()}}
    a,b = [np.asarray(w.evidence['trajectories']) for w in workers]
    evidence = dict(input_sha256=hashlib.sha256(encoded.encode()).hexdigest(),inputs=payload,atol=atol,
                    interpreters=[str(python_a),str(python_b)], factories=[factory,factory_b or factory], workers=[json.loads(Report([w]).dumps()) for w in workers])
    if a.shape!=b.shape or left.evidence['dt']!=right.evidence['dt']:
        return result(adapter,'release',Status.PARTIAL,'Worker dimensions or timesteps differ; comparison invalid',evidence)
    first = float(np.max(np.abs(a[:,0]-b[:,0])))
    evidence['first_step_max_difference'] = first
    evidence['later_max_difference'] = float(np.max(np.abs(a-b)))
    low_precision = any(np.finfo(w.evidence['dtype']).eps>1e-10 or w.evidence['noise_floor']>0 for w in workers)
    evidence['bitwise_identical_recorded_trajectories'] = a.tobytes()==b.tobytes()
    noise_floor = max(atol, *(w.evidence['noise_floor'] for w in workers))
    if low_precision:
        noise_floor = max(noise_floor,*(8*np.finfo(w.evidence['dtype']).eps*(1+np.max(np.abs(a[:,0]))) for w in workers))
    evidence['effective_floor'] = float(noise_floor)
    if first > noise_floor:
        return result(adapter,'release',Status.FAIL,'First-step compatibility change above supplied floor',evidence,'changed_dynamics')
    if first or low_precision:
        return result(adapter,'release',Status.ABSTAIN,'First-step difference at/below precision or noise floor',evidence)
    return result(adapter,'release',Status.PASS,'Recorded first steps agree exactly on supplied probes',evidence)
