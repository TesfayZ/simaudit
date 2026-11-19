"""Bounded one-parameter estimation for E(s;t) = A(s) + t B(s)."""
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .adapters import Blowup, Unsupported
from .checks import guard, probes, result, tolerance
from .refinement import refined_state
from .report import Modification, Status


@dataclass
class EnergyFit:
    parameter: str
    bounds: tuple[float, float]
    components: Callable  # state -> (A, B) at fixed energy units and offset
    train: dict           # independent start ID -> (states, actions)
    holdout: dict
    max_rate: float       # acceptable held-out and training absolute energy rate
    resolution: float     # maximum width of the acceptable parameter interval
    fixed: str            # fixed parameters, units and reference description


def fit_parameter(adapter, spec, *, interval=None):
    lo, hi = map(float, spec.bounds)
    h = float(adapter.dt if interval is None else interval)
    if not np.all(np.isfinite([lo, hi, h])) or lo >= hi or h / 8 <= 0:
        raise ValueError('Require finite ordered bounds and a positive refinement step')
    tolerance(spec.max_rate)
    if tolerance(spec.resolution) <= 0 or not spec.parameter or not spec.fixed:
        raise ValueError('Supply parameter, fixed specification and positive resolution')
    if not spec.train or not spec.holdout or set(spec.train) & set(spec.holdout):
        raise ValueError('Supply nonempty disjoint training and held-out start IDs')
    evidence = dict(parameter=spec.parameter, bounds=[lo, hi], fixed=spec.fixed, interval=h,
                    max_rate=spec.max_rate, resolution=spec.resolution, estimate=None, probes=[])
    changes = [Modification('timestep', adapter.dt, h/8, 'Eight substeps for the parameter fit')]
    def finish(status, reason):
        return result(adapter, 'fit', status, reason, evidence, modifications=changes)

    if (adapter.smooth is not True or adapter.conservative is not True or adapter.order_blockers
            or not adapter.features or any(f.status == 'unsupported' and f.present is not False for f in adapter.features)):
        return finish(Status.PARTIAL, 'Fit requires inventoried smooth passive replay')
    if adapter.noise_floor or np.finfo(adapter.dtype).eps > 1e-10:
        return finish(Status.ABSTAIN, 'Fit requires resolved deterministic internal states')

    datasets = {}
    seen_starts = set()
    try:
        for split, groups in [('train', spec.train), ('holdout', spec.holdout)]:
            rows = []
            probe_weights = []
            for start_id, (states, actions) in groups.items():
                states, actions = probes(states, actions)
                key = tuple(states[0].astype(float).ravel())
                if key in seen_starts:
                    raise ValueError('Repeated initial state across independent groups')
                seen_starts.add(key)
                # The paper averages each start's mean defect over starts.
                probe_weights.extend([1 / (len(groups) * len(states))] * len(states))
                blocked = guard(adapter, 'fit', states, actions, [h / 8])
                if blocked:
                    return blocked
                for s, u in zip(states, actions):
                    end, eps = refined_state(adapter, s, u, h/8, 8)
                    first, last = [np.asarray(spec.components(x.copy())) for x in (s, end)]
                    if any(x.shape != (2,) or x.dtype.kind not in 'fiu' for x in (first,last)):
                        raise ValueError('Energy components must be two real numbers')
                    if eps > 1e-10 or any(x.dtype.kind == 'f' and np.finfo(x.dtype).eps > 1e-10 for x in (first,last)):
                        return finish(Status.ABSTAIN, 'Low precision in replay or energy components')
                    rate = (last.astype(float) - first.astype(float)) / h
                    if not np.all(np.isfinite(rate)):
                        raise Blowup('Nonfinite component rate')
                    rows.append(rate)
                    evidence['probes'].append(dict(split=split, group=start_id, state=s.tolist(),
                        action=np.asarray(u).tolist(), endpoint=end.tolist(), start_components=first.tolist(), component_rates=rate.tolist()))
            datasets[split] = (*np.asarray(rows).T, np.asarray(probe_weights))
    except (Blowup, Unsupported) as exc:
        return finish(Status.BLOWUP if isinstance(exc, Blowup) else Status.PARTIAL, str(exc))

    # The mean absolute affine residual is convex and piecewise linear.
    # Its minimum occurs at a root or boundary. Store every knot so the
    # admissible interval can be inspected and reproduced without an optimiser.
    a, b, train_weights = datasets['train']
    moving = b != 0
    if not np.any(moving):
        return finish(Status.ABSTAIN, 'No parameter excitation on training probes')
    roots = -a[moving] / b[moving]
    order = np.argsort(roots)
    roots = roots[order]
    weights = (np.abs(b[moving]) * train_weights[moving])[order]
    cumulative = np.r_[0., np.cumsum(weights)]
    median_index = np.searchsorted(cumulative[1:], cumulative[-1]/2)
    theta = float(np.clip(roots[median_index], lo, hi))
    knots = np.unique(np.r_[lo, roots[(lo < roots) & (roots < hi)], hi])
    idx = int(np.searchsorted(knots, theta))
    # Between knots the derivative is weight to the left minus weight to the
    # right. Integrate away from the median to avoid subtracting large losses.
    splits = np.searchsorted(roots, knots[:-1], side='right')
    increments = np.diff(knots) * (2*cumulative[splits] - cumulative[-1])
    minimum = train_weights @ np.abs(a + theta*b)
    losses = np.r_[minimum - np.cumsum(increments[:idx][::-1])[::-1], minimum,
                   minimum + np.cumsum(increments[idx:])]
    if not np.all(np.isfinite(losses)):
        return finish(Status.BLOWUP, 'Nonfinite fit objective')
    evidence['profile'] = np.column_stack((knots, losses)).tolist()
    acceptable = np.flatnonzero(losses <= spec.max_rate)
    if not len(acceptable):
        return finish(Status.ABSTAIN, 'No candidate meets the specified training residual')
    left, right = acceptable[[0,-1]]
    if left == 0 or right == len(knots)-1:
        return finish(Status.ABSTAIN, 'Acceptable fit reaches a bound or has insufficient excitation')
    lower = np.interp(spec.max_rate, losses[[left,left-1]], knots[[left,left-1]])
    upper = np.interp(spec.max_rate, losses[[right,right+1]], knots[[right,right+1]])
    evidence['acceptable_interval'] = [float(lower), float(upper)]
    if upper - lower > spec.resolution:
        return finish(Status.ABSTAIN, 'Parameter unresolved at the requested resolution')
    # A pure rescaling can drive the whole reference energy to a constant.
    sample = np.array([r['start_components'] for r in evidence['probes'] if r['split']=='train'], float)
    if np.linalg.matrix_rank(sample - sample.mean(axis=0)) < 2:
        return finish(Status.ABSTAIN, 'Energy family varies only by scale or offset on these starts')
    c, d, holdout_weights = datasets['holdout']
    if not np.any(d):
        return finish(Status.ABSTAIN, 'No parameter excitation on held-out probes')
    loss = float(holdout_weights @ np.abs(c + theta*d))
    if not np.isfinite(loss):
        return finish(Status.BLOWUP, 'Nonfinite held-out residual')
    evidence['heldout_mean_abs_rate'] = loss
    if loss > spec.max_rate:
        return finish(Status.ABSTAIN, 'Held-out energy defects exceed the supplied tolerance')
    evidence['estimate'] = theta
    return finish(Status.PASS, f'{spec.parameter} = {theta:.8g} in the supplied energy family')
