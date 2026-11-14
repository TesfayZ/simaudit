"""Independent scoped checks. Method attribution: see CITATION.cff and THIRD_PARTY.md."""
from dataclasses import asdict
import math
import numpy as np
from .report import Feature, Modification, Result, Status
from .adapters import Blowup, Unsupported

LIMITS = {
    'assignment': ['Only explicitly supplied parameters and probes are assessed.',
                   'The fresh factory defines the parameter contract; direct mutation may not be a supported simulator API.'],
    'order': ['Convergence does not establish correct equations or accurate trajectories.',
              'Non-smooth, noisy, stiff and unresolved regimes cannot be diagnosed by these slopes.',
              'Only the supplied interval and starts are assessed; no energy or model-fault classification.'],
    'observation': ['Only the supplied reference observation map and probes are assessed.',
                    'A shared error in simulator and reference can be invisible.'],
    'release': ['A changed first step is a compatibility difference, not proof of a simulator defect.',
                'Reward, termination, reset, spaces, RNG and rendering are not compared.',
                'Later separation does not identify chaos; unexecuted releases remain untested.'],
}


def result(a, check, status, reason, evidence=None, fault=None, modifications=()):
    coverage = [Feature(**asdict(f)) for f in a.features]
    if not coverage:
        coverage = [Feature('dynamics', 'unsupported', 'No feature inventory supplied')]
    return Result(check, status, reason, a.scope, LIMITS.get(check, []) + a.limitations + [
        'Adapter inventory and reference correctness remain caller responsibilities.'], coverage,
        [*a.modifications, *modifications], a.versions, evidence or {}, fault)


def probes(states, actions):
    states = [np.asarray(s) for s in states]
    actions = list(actions)
    if not states or len(states) != len(actions):
        raise ValueError('Supply a nonempty state/action pair list with matching lengths')
    if any(not np.all(np.isfinite(s)) for s in states):
        raise ValueError('Input states must be finite')
    return states, actions


def tolerance(value):
    if not np.isfinite(value) or value < 0:
        raise ValueError('Tolerance must be finite and nonnegative')
    return float(value)


def self_test(a, states, actions, dts, atol=1e-12):
    """Compare native steps, replay after interleaving, and two-step continuation."""
    for state, action in zip(states, actions):
        for dt in dts:
            x = a.step(state, action, dt)
            native = a.native_step(state, action, dt)
            if not np.all(np.isfinite(x)) or not np.all(np.isfinite(native)):
                raise Blowup('Nonfinite output during adapter self-test')
            replay_floor, _ = precision(a, [x, native], atol)
            if a.distance(x, native) > replay_floor:
                return False, 'native-step equivalence failed'
            # A different input/continuation must not contaminate a replay.
            y = a.step(x, action, dt)
            replay = a.step(state, action, dt)
            native_y = a.native_step(native, action, dt)
            continuation_floor, _ = precision(a, [y, native_y], atol)
            if a.distance(x, replay) > replay_floor or a.distance(y, native_y) > continuation_floor:
                return False, 'state restoration or continuation equivalence failed'
    return True, 'native-step equivalence and interleaved replay passed'


def guard(a, check, states, actions, dts, needed=()):
    missing = {'step', 'native_step', *needed} - a.capabilities
    if missing:
        return result(a, check, Status.SKIPPED, 'Missing capabilities: ' + ', '.join(sorted(missing)))
    try:
        ok, reason = self_test(a, states, actions, dts)
    except Blowup as exc:
        return result(a, check, Status.BLOWUP, str(exc))
    except Unsupported as exc:
        return result(a, check, Status.SKIPPED, str(exc))
    if not ok and a.noise_floor > 0:
        return result(a, check, Status.ABSTAIN, 'Declared noise prevents verifying deterministic native equivalence/restoration; downstream check blocked')
    if not ok:
        return result(a, check, Status.FAIL, reason + '; downstream check blocked', fault='adapter')
    return None


def precision(a, arrays, atol):
    eps = max([float(np.finfo(a.dtype).eps)] + [float(np.finfo(x.dtype).eps) for x in arrays if x.dtype.kind == 'f'])
    scale = 1 + max(float(np.linalg.norm(x)) for x in arrays)
    return max(atol, a.noise_floor, 50 * eps * scale), eps


def check_assignment(a, parameter, value, states, actions, *, atol=1e-12):
    states, actions = probes(states, actions)
    atol = tolerance(atol)
    blocked = guard(a, 'assignment', states, actions, [a.dt], ['assignment'])
    if blocked: return blocked
    try:
        assigned, fresh = a.assignment_factory(parameter, value)
    except Unsupported as exc:
        return result(a, 'assignment', Status.SKIPPED, str(exc))
    for candidate in (assigned, fresh):
        blocked = guard(candidate, 'assignment', states, actions, [candidate.dt])
        if blocked: return blocked
    if not assigned.modifications or not fresh.modifications:
        return result(a, 'assignment', Status.FAIL, 'Assignment factory omitted modification records', fault='adapter')
    rows = []
    try:
        for s, u in zip(states, actions):
            b, d, f = [adapter.step(s, u, adapter.dt) for adapter in (a, assigned, fresh)]
            floor, eps = precision(a, [b, d, f], atol)
            rows.append(dict(state=s.tolist(), action=np.asarray(u).tolist(), baseline=b.tolist(),
                             assigned=d.tolist(), fresh=f.tolist(), floor=floor,
                             baseline_assigned=a.distance(b,d), baseline_fresh=a.distance(b,f),
                             assigned_fresh=a.distance(d,f), low_precision=eps > 1e-10))
    except Blowup as exc:
        return result(a, 'assignment', Status.BLOWUP, str(exc))
    effective = [r for r in rows if r['baseline_fresh'] > r['floor']]
    mismatch = any(r['assigned_fresh'] > r['floor'] for r in rows)
    no_op = all(r['baseline_assigned'] <= r['floor'] for r in rows)
    if mismatch:
        kind = 'no_op' if no_op else 'stale'
        if not no_op and assigned.assignment_metadata.get('partial_update'):
            kind = 'partly_stale'
        status, reason = Status.FAIL, f'{parameter}: {kind} relative to supplied fresh-reference contract'
    elif not effective or any(r['low_precision'] for r in rows) or a.noise_floor:
        kind, status, reason = 'unresolved', Status.ABSTAIN, 'Insufficient parameter excitation or precision/noise blocks a consistency verdict'
    else:
        kind, status, reason = 'consistent', Status.PASS, f'{parameter}: assignment matches the fresh reference on supplied probes'
    return result(a, 'assignment', status, reason, dict(parameter=parameter, value=value, classification=kind,
                  atol=atol, adapter_assignment_metadata=assigned.assignment_metadata, self_test='passed in baseline, assigned and fresh configurations', probes=rows),
                  kind if mismatch else None, [*assigned.modifications, *fresh.modifications])


def check_order(a, states, actions, *, interval=None, atol=1e-12, order_tolerance=0.5, slope_tolerance=0.5):
    states, actions = probes(states, actions)
    atol, order_tolerance, slope_tolerance = map(tolerance, (atol, order_tolerance, slope_tolerance))
    h = a.dt if interval is None else float(interval)
    if not np.isfinite(h) or h <= 0: raise ValueError('Interval must be positive and finite')
    if a.declared_order is None:
        return result(a, 'order', Status.SKIPPED, 'Missing declared_order')
    if not np.isfinite(a.declared_order) or a.declared_order <= 0:
        raise ValueError('declared_order must be positive and finite')
    modifications = [Modification('timestep', a.dt, h/m, f'Fixed-interval refinement: {m} substeps')
                     for m in (1,2,4,8) if h/m != a.dt]
    unknown = [f.name for f in a.features if f.status=='unsupported' and f.present is not False]
    if a.smooth is not True or a.order_blockers or unknown:
        return result(a, 'order', Status.PARTIAL, 'Smoothness not established or unsupported features: ' +
                      ', '.join(a.order_blockers + unknown or ['smooth']), modifications=modifications)
    blocked = guard(a, 'order', states, actions, [h/m for m in (1,2,4,8)])
    if blocked:
        blocked.modifications.extend(modifications)
        return blocked
    rows = []
    try:
        for s, u in zip(states, actions):
            endpoints = []
            for m in (1,2,4,8):
                x = s.copy()
                for _ in range(m):
                    x = a.step(x, u, h/m)
                    if not np.all(np.isfinite(x)): raise Blowup('Nonfinite refinement trajectory')
                endpoints.append(x)
            floor, eps = precision(a, endpoints, atol)
            d = [a.distance(x,y) for x,y in zip(endpoints, endpoints[1:])]
            p = [math.log2(d[i]/d[i+1]) for i in (0,1)] if min(d)>floor else [None,None]
            if eps > 1e-10 or a.noise_floor or p[0] is None:
                label = 'ABSTAIN'
            elif abs(p[0]-p[1]) > slope_tolerance:
                label = 'ABSTAIN'
            elif p[1] < a.declared_order-order_tolerance:
                label = 'FAIL'
            elif p[1] > a.declared_order+order_tolerance:
                label = 'ABSTAIN'
            else:
                label = 'PASS'
            rows.append(dict(state=s.tolist(), action=np.asarray(u).tolist(), endpoints=[e.tolist() for e in endpoints],
                             differences=d, floor=floor, slopes=p, status=label))
    except Unsupported as exc:
        return result(a, 'order', Status.PARTIAL, str(exc), {'probes': rows}, modifications=modifications)
    except Blowup as exc:
        return result(a, 'order', Status.BLOWUP, str(exc), {'probes': rows}, modifications=modifications)
    labels = {r['status'] for r in rows}
    status = Status.FAIL if 'FAIL' in labels else Status.PASS if labels=={'PASS'} else Status.ABSTAIN
    reason = {Status.FAIL:'Resolved order below declaration on one or more supplied probes',
              Status.PASS:'Stable slopes consistent with declared order on all supplied probes',
              Status.ABSTAIN:'Order unresolved: precision/noise floor, unstable slopes, or order outside assessment band'}[status]
    return result(a, 'order', status, reason, dict(interval=h, declared_order=a.declared_order,
        atol=atol, order_tolerance=order_tolerance, slope_tolerance=slope_tolerance,
        self_test='passed at all four timesteps', probes=rows), 'integrator_order' if status==Status.FAIL else None, modifications)


def check_observation(a, states, actions, *, atol=1e-12):
    states, actions = probes(states, actions)
    atol = tolerance(atol)
    blocked = guard(a, 'observation', states, actions, [a.dt], ['observation','reference_observation'])
    if blocked: return blocked
    rows = []
    try:
        for s,u in zip(states,actions):
            observed = np.asarray(a.observe_fn(s.copy(),u,a.dt))
            reference = np.asarray(a.reference_observe_fn(s.copy(),u,a.dt))
            if observed.shape != reference.shape: raise ValueError('Observation shapes differ')
            if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(reference)):
                raise Blowup('Nonfinite observation')
            eps = max([float(np.finfo(a.dtype).eps)] + [float(np.finfo(x.dtype).eps) for x in (observed,reference) if x.dtype.kind=='f'])
            floor = max(atol, a.noise_floor, 8*eps*(1+float(np.max(np.abs(reference)))))
            error = float(np.max(np.abs(observed.astype(float)-reference.astype(float))))
            rows.append(dict(state=s.tolist(), action=np.asarray(u).tolist(), observed=observed.tolist(),
                             reference=reference.tolist(), error=error, floor=floor, low_precision=eps>1e-10))
    except Unsupported as exc:
        return result(a, 'observation', Status.SKIPPED, str(exc), {'probes':rows})
    except Blowup as exc:
        return result(a, 'observation', Status.BLOWUP, str(exc), {'probes':rows})
    fail = any(r['error']>r['floor'] for r in rows)
    uncertain = any(r['low_precision'] for r in rows) or a.noise_floor > 0
    status = Status.FAIL if fail else Status.ABSTAIN if uncertain else Status.PASS
    return result(a, 'observation', status,
        'Observation differs from reference above floor' if fail else
        'Precision/noise blocks a consistency verdict; above-floor faults can still be flagged' if uncertain else
        'Observation agrees with reference on supplied probes',
        dict(atol=atol, self_test='passed', probes=rows), 'observation' if fail else None)
