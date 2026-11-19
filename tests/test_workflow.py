from dataclasses import replace
import json
import sys

import numpy as np
import pytest

from simaudit import (EnergyFit, Hypothesis, Law, LawStudy, LogArm, Recovery, Report, WorkflowInputs,
                      diagnose, fit_parameter, plan, screen_laws, Status)
from simaudit.demo import inputs
from simaudit.cli import main


@pytest.mark.parametrize('question,expected', [
    ('drift',['energy','order']), ('screen',['screen']), ('attribution',['screen','attribution']),
    *[(s,[s]) for s in ('assignment','release','observation','order','fit','energy')]])
def test_routes_without_simulation(question, expected):
    data = inputs()
    data.adapter.step_fn = lambda *a: pytest.fail('preview stepped simulator')
    assert [b['check'] for b in plan(question,data)['branches']] == expected


def test_cli_and_old_report(tmp_path):
    path = tmp_path/'workflow.json'
    assert main(['diagnose','drift','--inputs','simaudit.demo:inputs','--strict','--json',str(path)]) == 0
    record = json.loads(path.read_text())
    assert record['coverage']=='complete'
    assert Report.loads(json.dumps(record['report'])).counts['PASS'] == 2


def test_full_and_missing():
    data = inputs()
    record = diagnose('full', data)
    assert len(record['branches']) == 8
    assert all(r['status']=='PASS' for r in record['report']['results'])
    assert record['branches'][-1]['outcome'] in ('no flagged law','missing inputs')
    blank = diagnose('full', WorkflowInputs('no inputs'))
    assert blank['coverage']=='partial' and blank['strict_exit_code']==1
    assert all(row['missing'] for row in blank['branches'])


def test_independent_screen_after_replay_failure():
    data = inputs()
    data.adapter.native_step_fn = lambda s,u,h: s+1
    record = diagnose('full',replace(data,assignment=None,release=None))
    assert record['coverage']=='partial'
    branches = {b['check']:b['outcome'] for b in record['branches']}
    assert branches['energy'].startswith('blocked') and branches['screen']=='executed'


def test_mixed_or_hidden_error_never_gates_screen():
    data = inputs()
    data.adapter.step_fn = data.adapter.native_step_fn = lambda s,u,h: s + h*np.array([s[1],-1.01*s[0]])
    record = diagnose('full',replace(data,release=None,assignment=None,fit=None))
    assert 'masked' in record['interpretation']
    assert record['report']['results'][-1]['check']=='screen'


@pytest.mark.parametrize('failure',['boundary','flat','gauge','wrong','heldout','overlap'])
def test_unresolved_fit(failure):
    data = inputs()
    fit = data.fit
    if failure=='boundary':
        fit.bounds=(1.2,2.)
    if failure=='flat':
        fit.components=lambda s:(s@s,0.)
    if failure=='gauge':
        fit.components=lambda s:(-s[0],s[0])
    if failure=='wrong':
        fit.components=lambda s:(s[1],s[0]**2)
    if failure=='heldout':
        # A velocity-dependent term present only at a new held-out start.
        fit.holdout={'other': ([[4.,3.]], [0])}
        fit.components=lambda s:(s[1]**2/2+max(s[0]-2,0),s[0]**2/2)
    if failure=='overlap':
        fit.holdout=fit.train
        with pytest.raises(ValueError,match='disjoint'):
            fit_parameter(data.adapter,fit)
        return
    r=fit_parameter(data.adapter,fit)
    assert r.status==Status.ABSTAIN and r.evidence['estimate'] is None


def reacher(seed=1201, l1=0.1, structural=False):
    # Reacher's horizontal tip coordinate, using independent pose groups.
    rng=np.random.default_rng(seed)
    coords=('cos1','sin1','cos2','sin2','tip_x')
    def arm(name,length,perturb):
        logs={}
        for i in range(6):
            a,b=rng.uniform(-np.pi,np.pi,(2,60))
            x=length*np.cos(a)+0.11*np.cos(a+b)
            if perturb:
                x+=0.02*np.sin(a)
            logs[f'{name}{i}']=np.column_stack((np.cos(a),np.sin(a),np.cos(b),np.sin(b),x))
        return LogArm(name,coords,logs,dict(sampling='uniform poses',integration='kinematic',noise=0),True)
    one=(0,0,0,0,0)
    c1=(1,0,0,0,0)
    c1c2=(1,0,1,0,0)
    s1s2=(0,1,0,1,0)
    tip=(0,0,0,0,1)
    norm1={(2,0,0,0,0):1,(0,2,0,0,0):1,one:-1}
    norm2={(0,0,2,0,0):1,(0,0,0,2,0):1,one:-1}
    laws=[Law('unit norm','vanishing',norm1),
          Law('horizontal FK','vanishing',{tip:1,c1:-0.1,c1c2:-0.11,s1s2:0.11})]
    recovery=Recovery(coords,2,dict(l1=Hypothesis(0.1,(0.04,0.2),{c1:-1}),
                                    l2=Hypothesis(0.11,(0.04,0.2),{c1c2:-1,s1s2:1})),[norm1,norm2])
    return LawStudy(arm('healthy',0.1,False),arm('candidate',l1,structural),laws,{'horizontal FK':recovery})


@pytest.mark.parametrize('seed',[1201,1202,1203,1204])
def test_healthy_and_changed_log_groups(seed):
    assert screen_laws(reacher(seed)).status==Status.PASS
    study=reacher(seed,0.13)
    r=screen_laws(study)
    assert r.evidence['flagged']==['horizontal FK'] and r.status==Status.FAIL
    assert r.evidence['laws']['horizontal FK']['p']==pytest.approx(1/924)


@pytest.mark.parametrize('kind',['noise','dependent','short','duplicate','small'])
def test_invalid_screen_units(kind):
    study=reacher()
    if kind=='noise':
        study.candidate.conditions['noise']=1
    if kind=='dependent':
        study.healthy.independent=False
    if kind=='duplicate':
        study.candidate.groups['fresh']=next(iter(study.healthy.groups.values()))
    if kind=='small':
        study.healthy.groups=dict(list(study.healthy.groups.items())[:2])
        study.candidate.groups=dict(list(study.candidate.groups.items())[:2])
    if kind=='short':
        study.laws[0].kind='conserved'
        study.healthy.groups={k:v[:1] for k,v in study.healthy.groups.items()}
    assert screen_laws(study).status in (Status.PARTIAL,Status.ABSTAIN)


@pytest.mark.parametrize('case',['changed','ambiguous','structural','missing','unexcited'])
def test_attribution(case):
    pytest.importorskip('sympy')
    study=reacher(l1=0.13,structural=case=='structural')
    spec=study.recovery['horizontal FK']
    if case=='ambiguous':
        spec.hypotheses['alias']=spec.hypotheses['l1']
    if case=='missing':
        study.recovery={}
    if case=='unexcited':
        for key,data in study.candidate.groups.items():
            study.candidate.groups[key]=np.tile(data[0], (60,1))
    record=diagnose('attribution', WorkflowInputs('Reacher',logs=study))
    r=record['report']['results'][-1]
    if case=='changed':
        assert r['evidence']['parameter']=='l1' and abs(r['evidence']['estimate']-0.13)<1e-8
    elif case=='missing':
        assert record['coverage']=='partial' and r['check']=='screen'
    else:
        assert r['status']=='ABSTAIN' and r['evidence']['parameter'] is None


def test_clean_screen_needs_no_attribution_extra_or_templates(monkeypatch):
    monkeypatch.setattr('simaudit.workflow.find_spec', lambda name: None)
    data = inputs()
    data.logs.recovery = {}
    r = diagnose('attribution',data)
    assert r['coverage']=='complete' and r['branches'][-1]['outcome']=='no flagged law'


def test_preview_detects_unmatched_conditions():
    data = inputs()
    data.logs.candidate.conditions['noise']='unknown'
    b, = plan('screen',data)['branches']
    assert b['missing']


def test_replay_discrepancy_blocks_only_linked_attribution():
    pytest.importorskip('sympy')
    data = inputs()
    data.adapter.reference_observe_fn = lambda *args: np.array([1e6])
    data.logs = reacher(l1=.13)
    for linked, expected in [(True,'physical attribution blocked by observation discrepancy'),(False,'executed')]:
        r = diagnose('full',replace(data,release=None,assignment=None,logs_use_adapter_observations=linked))
        assert r['branches'][-1]['outcome']==expected


def test_unusable_optional_release_does_not_consume_live_probes():
    data = inputs()
    r = diagnose('release',replace(data,adapter=None,release={'factory':'simaudit.demo:oscillator',
                        'python_a':sys.executable,'python_b':sys.executable}))
    assert r['coverage']=='complete'


def test_changed_assignment_configuration_does_not_block_baseline():
    data = inputs()
    factory=data.adapter.assignment_factory
    def broken(name,value):
        assigned,fresh=factory(name,value)
        assigned.native_step_fn=lambda s,u,h: s+7
        return assigned,fresh
    data.adapter.assignment_factory=broken
    rec=diagnose('full',replace(data,release=None))
    assert rec['report']['results'][0]['evidence']['configuration']=='assigned'
    branches={b['check']:b['outcome'] for b in rec['branches']}
    assert branches['order']=='executed' and branches['energy']=='executed'


def test_flagged_identity_blocks_deflation():
    pytest.importorskip('sympy')
    study=reacher(l1=.13)
    for x in study.candidate.groups.values():
        x[:,0] *= 1.1
    result=diagnose('attribution',WorkflowInputs('changed coordinates',logs=study))
    row=result['report']['results'][-1]
    assert row['status']=='ABSTAIN' and 'recovery assumption' in row['reason']


def test_drift_reuses_guarded_endpoints_only_within_one_run(monkeypatch):
    import simaudit.refinement as ref
    calls=[]
    original=ref.refined_state
    def counted(a,s,u,h,n):
        calls.append((h,n))
        return original(a,s,u,h,n)
    monkeypatch.setattr(ref,'refined_state',counted)
    data=inputs()
    for run in (1,2):
        r=diagnose('drift',data)
        assert r['coverage']=='complete'
        assert len(calls)==run*4*len(data.states)


def test_fit_profile_matches_direct_objective_at_every_knot():
    data = inputs()
    states, actions = data.fit.train['0']
    states.extend([[0.5, 1.2], [0.7, 0.8]])
    actions.extend([0, 0])
    r = fit_parameter(data.adapter, data.fit)
    assert r.status == Status.PASS
    groups = [np.asarray([p['component_rates'] for p in r.evidence['probes']
                         if p['split'] == 'train' and p['group'] == key]) for key in data.fit.train]
    for knot,loss in r.evidence['profile']:
        direct = np.mean([np.mean(np.abs(rates[:,0]+knot*rates[:,1])) for rates in groups])
        assert loss == pytest.approx(direct, abs=1e-14)


def test_unexcited_holdout_cannot_validate_fit():
    data = inputs()
    data.fit.holdout = {'equilibrium': ([[0., 0.]], [0])}
    r = fit_parameter(data.adapter, data.fit)
    assert r.status == Status.ABSTAIN and r.evidence['estimate'] is None


def test_unresolved_order_probes_keep_workflow_coverage_partial():
    data = inputs()
    data.adapter.declared_order = 9
    record = diagnose('order', replace(data, states=[[0., 0.], [1., 0.4]], actions=[0, 0]))
    result, = record['report']['results']
    assert result['status'] == 'FAIL'
    assert [p['status'] for p in result['evidence']['probes']] == ['ABSTAIN', 'FAIL']
    assert record['coverage'] == 'partial'


def test_exact_screen_alpha_boundary_and_near_ties():
    from math import comb
    from simaudit.laws import _permutation

    # Close residual sums count as ties in the paper's exact permutation test.
    assert _permutation([0.] * 5, [1e-14] * 5, comb(10, 5)) == 1.
    study = reacher(l1=.13)
    study.alpha = len(study.laws) / comb(12, 6)
    assert screen_laws(study).status == Status.ABSTAIN


def test_failed_integrity_blocks_attribution_from_linked_logs():
    pytest.importorskip('sympy')
    data = inputs()
    data.adapter.native_step_fn = lambda s, u, h: s + 7
    data.logs = reacher(l1=.13)
    r = diagnose('full', replace(data, release=None, logs_use_adapter_observations=True))
    assert r['branches'][-2]['outcome'] == 'executed'
    assert r['branches'][-1]['outcome'] == 'physical attribution blocked by adapter integrity failure'
