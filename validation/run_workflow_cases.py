"""Local cross-check using an explicitly supplied checkout of the method paper.

Usage: python validation/run_workflow_cases.py /path/to/algebraicRLtest output.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

import numpy as np
import sympy as sp
import simaudit

from simaudit import (EnergyFit, Hypothesis, Law, LawStudy, LogArm, Recovery,
                      WorkflowInputs, diagnose)
from known_cases import acrobot


def polynomial(expr, variables):
    return {tuple(map(int,e)):float(c) for e,c in sp.Poly(expr,*variables).terms()}


def rec_spec(template, variables, known, nominal, params, degree, lag):
    hyps = {}
    for p in params:
        theta = nominal[p]
        assert sp.diff(template,p,2) == 0
        deriv = polynomial(sp.diff(template,p).subs(nominal), variables)
        hyps[str(p)] = Hypothesis(theta,(0.4*theta,1.8*theta),deriv)
    return Recovery(tuple(map(str,variables)),degree,hyps,[polynomial(k,variables) for k in known],lag)


def log_case(b, name, params, seed, acro, noise=0.):
    import ideal_diff
    import recover

    if acro:
        v = b.ACRO_SYMS
        polys,names,kinds = b.acrobot_reference()
        spec = rec_spec(b.acrobot_energy_template(),v,polys[:2],b.acrobot_spec(),
                        [b.P_m1,b.P_m2,b.P_I1,b.P_I2,b.P_g],3,50)
        mapping = {'energy':spec}
    else:
        v = b.REACH_SYMS
        polys,names,kinds = b.reacher_reference(scale=1.)
        mapping = {}
        for axis,idx in [('x',0),('y',1)]:
            sub = (*v[:4],v[4+idx],v[8+idx])
            mapping['fk-'+axis] = rec_spec(b.reacher_fk_template(axis),sub,polys[:2],
                                             b.reacher_spec(),b.REACH_PARAMS,2,1)
    def arm(role, model, arm_seed):
        if acro:
            data = b.acrobot_trajectories(params=model,seed=arm_seed,dt=.005,n_traj=8,n_steps=400,noise=noise)
        else:
            data = [b.reacher_samples(params=model,N=125,seed=arm_seed+i,scale=1.,noise=noise)[0] for i in range(8)]
        conditions = dict(sampling='8 independent trajectories' if acro else '8 independent pose groups',
                          integration='RK4 dt=.005' if acro else 'analytic kinematics', noise=noise)
        return LogArm(role,tuple(map(str,v)),{f'{arm_seed}-{i}':x for i,x in enumerate(data)},conditions,True)
    laws = [Law(n,k,polynomial(p,v)) for n,k,p in zip(names,kinds,polys)]
    study = LawStudy(arm('healthy',{},seed),arm(name,params,seed+100),laws,mapping)
    flow = diagnose('attribution',WorkflowInputs(name,logs=study))
    screen,*attrib = flow['report']['results']
    healthy = list(study.healthy.groups.values())
    candidate = list(study.candidate.groups.values())
    reference = ideal_diff.screen(polys, healthy, candidate, v, names, kinds)
    assert screen['evidence']['flagged'] == reference.localised
    for verdict in reference.verdicts:
        assert screen['evidence']['laws'][verdict.name]['adjusted_p'] == verdict.pvalue
    # The direct standard-monomial dictionary must give the same spectral gap
    # as the paper's explicit orthonormal projection in normalised coordinates.
    for row in attrib:
        law = row['evidence']['law']
        rec = mapping[law]
        cols = [study.candidate.coordinates.index(coord) for coord in rec.coordinates]
        variables = tuple(v[i] for i in cols)
        groups = [x[:,cols] for x in candidate]
        if acro:
            _, smallest, second = recover.recover_conserved_quotient(
                groups, variables, polys[:2], degree=rec.degree, lag=rec.lag)
        else:
            _, smallest, second = recover.recover_vanishing_quotient(
                np.vstack(groups), variables, polys[:2], degree=rec.degree)
        values = row['evidence']['singular_values']
        np.testing.assert_allclose(values[-1]/values[-2], smallest/second, rtol=1e-5, atol=1e-12)
    return dict(case=name,seed=seed,coverage=flow['coverage'],screen=screen['status'],
                flagged=screen['evidence']['flagged'],noise=noise,
                attribution=[dict(law=r['evidence']['law'],status=r['status'],reason=r['reason'],
                                  parameter=r['evidence']['parameter'],estimate=r['evidence'].get('estimate')) for r in attrib])


def mass_fit(envs, mass, seed):
    a = acrobot(mass=mass)
    a.smooth = a.conservative = True
    states = np.random.default_rng(seed).uniform(-1,1,(12,4))
    def groups(indices):
        return {str(i):([states[i]],[1]) for i in indices}
    def components(x):
        offset = envs.acrobot_energy_coords(x,{**envs.ACROBOT,'m2':0})
        return offset, envs.acrobot_energy_coords(x) - offset
    fit = EnergyFit('m2',(.3,2.5),components,groups(range(8)),groups(range(8,12)),1e-5,0.001,
                    'Gymnasium default passive book equations, nominal remaining parameters')
    flow = diagnose('fit',WorkflowInputs('Acrobot',adapter=a,fit=fit,interval=.05))
    r, = flow['report']['results']
    return dict(injected=mass,seed=seed,status=r['status'],estimate=r['evidence']['estimate'],
                reason=r['reason'],heldout_rate=r['evidence'].get('heldout_mean_abs_rate'),
                acceptable_interval=r['evidence'].get('acceptable_interval'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('paper',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    sys.path.insert(0,str(args.paper.resolve()))
    import bugs
    import envs
    import gymnasium
    cases = [('Acrobot healthy',{},True),('Acrobot mass',{'m2':1.3},True),
             ('Acrobot inertia',{'I2':1.5},True),('Acrobot structural',{'variant':'nips'},True),
             ('Reacher healthy',{},False),('Reacher first link',{'l1':.09},False),
             ('Reacher second link',{'l2':.13},False)]
    rows = [log_case(bugs,name,params,3100+200*i,acro) for i,(name,params,acro) in enumerate(cases)]
    noisy = [log_case(bugs,'Reacher healthy noisy',{},10000+200*i,False,1e-5) for i in range(20)]
    fits = [mass_fit(envs,mass,40100+i) for i,mass in enumerate([.7,.9,1.001,1.1,1.3])]
    root = Path(__file__).resolve().parents[1]
    package = Path(simaudit.__file__).resolve().parent
    hashes = {f'src/simaudit/{p.relative_to(package)}': hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(package.rglob('*.py'))}
    for p in (Path(__file__).resolve(), root/'validation/known_cases.py'):
        hashes[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
    receipt = dict(python=platform.python_version(),numpy=np.__version__,sympy=sp.__version__,
                   gymnasium=gymnasium.__version__,logs=rows,noisy_controls=noisy,mass_fits=fits,
                   sha256=hashes, reference_sha256={p:hashlib.sha256((args.paper/p).read_bytes()).hexdigest()
                       for p in ('bugs.py','envs.py','ideal_diff.py','recover.py','discover.py','deflation.py')})
    receipt['summary'] = dict(
        scenarios=len(rows), complete=sum(r['coverage']=='complete' for r in rows),
        expected_injection_localisations=sum(bool(r['flagged']) for r in rows if r['case'] in
                ('Acrobot mass','Acrobot inertia','Acrobot structural','Reacher first link','Reacher second link')),
        noisy_healthy_runs=len(noisy), noisy_false_alarms=sum(r['screen']=='FAIL' for r in noisy),
        noisy_abstentions=sum(r['screen']=='ABSTAIN' for r in noisy),
        attributed_laws=sum(a['parameter'] is not None for r in rows for a in r['attribution']),
        attribution_abstentions=sum(a['status']=='ABSTAIN' for r in rows for a in r['attribution']),
        wrong_attributions=sum(a['parameter'] is not None and a['parameter']!=
            {'Acrobot mass':'m2','Acrobot inertia':'I2','Reacher first link':'ell1','Reacher second link':'ell2'}.get(r['case'])
            for r in rows for a in r['attribution']))
    for row in rows:
        print(row)
    print('Noisy healthy screens:', {s:sum(r['screen']==s for r in noisy) for s in ('PASS','FAIL','ABSTAIN')})
    print(fits)
    expected = [[],['energy'],['energy'],['energy'],[],['fk-x','fk-y'],['fk-x','fk-y']]
    assert [r['flagged'] for r in rows] == expected
    for i,param,value in [(1,'m2',1.3),(2,'I2',1.5),(5,'ell1',.09),(6,'ell2',.13)]:
        assert rows[i]['attribution'] and all(a['parameter']==param and abs(a['estimate']-value)<.001 for a in rows[i]['attribution'])
    assert len(rows[3]['attribution']) == 1 and rows[3]['attribution'][0]['status'] == 'ABSTAIN'
    assert all(r['screen']=='PASS' for r in noisy)
    assert all(r['status']=='PASS' and abs(r['estimate']-r['injected'])<.001 for r in fits)
    args.output.write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
