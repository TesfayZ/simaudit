"""Energy-refinement classifier of Gebrekidan (DAFD), adapted to scoped replay."""
import math

import numpy as np

from .adapters import Blowup, Unsupported
from .checks import guard, probes, result, tolerance
from .report import Modification, Status
from .refinement import Refinement, epsilon, refinement_exponent


_LEVELS = (1, 2, 4, 8)


def _energy(adapter, state):
    value = np.asarray(adapter.energy_fn(state.copy()))
    if value.ndim != 0 or value.dtype.kind not in 'fiu':
        raise ValueError('energy_fn must return one real scalar')
    if not math.isfinite(float(value)):
        raise Blowup('Nonfinite reference energy')
    return value


def _classify(rates, floor, rate_limit, q):
    if rates[-1] <= floor or min(rates) <= 0:
        return 'undetermined', [None] * 3, Status.ABSTAIN
    if rates[-1] > rate_limit:
        return 'unstable', [None] * 3, Status.ABSTAIN
    slopes = [refinement_exponent(x, y) for x, y in zip(rates, rates[1:])]
    a, b, c = slopes
    threshold = q - (1 if q >= 3 else 0.5)
    mean = sum(slopes) / 3
    if min(slopes) >= threshold:
        return 'consistent', slopes, Status.PASS
    if c < 0.5 or (c < min(a, b) - 0.4 and c < threshold) or mean <= 0.5:
        return 'model', slopes, Status.FAIL
    k = round(mean)
    # A fallback label can round to the declaration even after a failed threshold.
    return f'order-{k}', slopes, Status.FAIL if k < q else Status.ABSTAIN


def check_energy(adapter, states, actions, *, interval=None, atol=0.0, energy_noise_floor=0.0):
    """Pool absolute local energy-defect rates at four refinements of one interval.

    energy_noise_floor and atol are in energy per unit model time. The adapter
    declares conservation of its reference energy for the requested probes.
    """
    return _check_energy(adapter, states, actions, Refinement(), interval=interval,
                         atol=atol, energy_noise_floor=energy_noise_floor)


def _check_energy(adapter, states, actions, refinement, *, interval=None, atol=0.0, energy_noise_floor=0.0):
    states, actions = probes(states, actions)
    atol, energy_noise_floor = map(tolerance, (atol, energy_noise_floor))
    h = adapter.dt if interval is None else float(interval)
    if not math.isfinite(h) or h / 8 <= 0:
        raise ValueError('Interval and substeps must be positive and finite')
    q = adapter.declared_order
    if 'energy' not in adapter.capabilities or q is None:
        return result(adapter, 'energy', Status.SKIPPED, 'Requires energy_fn and declared_order')
    if not math.isfinite(q) or q <= 0:
        raise ValueError('declared_order must be positive and finite')
    blockers = [f.name for f in adapter.features if f.status == 'unsupported' and f.present is not False]
    blockers.extend(adapter.order_blockers)
    if not adapter.features or adapter.smooth is not True or adapter.conservative is not True or blockers:
        return result(adapter, 'energy', Status.PARTIAL,
                      'Requires inventoried, smooth, conservative probes' + (': ' + ', '.join(blockers) if blockers else ''))
    dts = [h / m for m in _LEVELS]
    changes = [Modification('timestep', adapter.dt, dt, f'Energy refinement over {m} substeps')
               for m, dt in zip(_LEVELS, dts) if dt != adapter.dt]
    blocked = guard(adapter, 'energy', states, actions, dts)
    if blocked:
        blocked.modifications.extend(changes)
        return blocked

    rows = []
    evidence = dict(interval=h, subdivisions=list(_LEVELS), declared_order=q, atol=atol,
                    energy_noise_floor=energy_noise_floor, self_test='passed at all four timesteps', probes=rows)
    # Residual arithmetic uses Python floats even when the callback uses a wider dtype.
    eps = max(float(np.finfo(adapter.dtype).eps), float(np.finfo(float).eps))
    try:
        for index, (state, action) in enumerate(zip(states, actions)):
            start = _energy(adapter, state)
            eps = max(eps, epsilon(state), epsilon(start))
            energies, rates = [], []
            for m, dt in zip(_LEVELS, dts):
                x, step_eps = refinement.get(index, adapter, state, action, dt, m)
                eps = max(eps, step_eps)
                end = _energy(adapter, x)
                eps = max(eps, epsilon(end))
                energies.append(float(end))
                rate = (float(end) - float(start)) / h
                if not math.isfinite(rate):
                    raise Blowup('Nonfinite energy-defect rate')
                rates.append(rate)
            rows.append(dict(state=state.tolist(), action=np.asarray(action).tolist(),
                             start_energy=float(start), end_energies=energies, signed_rates=rates))
    except (Blowup, Unsupported) as exc:
        status = Status.BLOWUP if isinstance(exc, Blowup) else Status.PARTIAL
        return result(adapter, 'energy', status, str(exc), evidence, modifications=changes)

    count = len(rows)
    scale = 1 + sum(abs(row['start_energy']) / count for row in rows)
    means = [sum(abs(row['signed_rates'][i]) / count for row in rows) for i in range(len(_LEVELS))]
    floor = max(atol, energy_noise_floor, 50 * eps * scale / h)
    if not all(map(math.isfinite, [scale, floor, *means])):
        return result(adapter, 'energy', Status.BLOWUP, 'Nonfinite aggregate or roundoff floor', evidence, modifications=changes)
    rate_limit = 0.5 * scale
    label, slopes, status = _classify(means, floor, rate_limit, q)
    reason = {
        'consistent': 'Energy defects decay consistently with the declared order',
        'model': 'Energy-defect pattern suggests a model or reference-energy mismatch',
        'unstable': 'Finest energy-defect rate exceeds the large-rate guard',
        'undetermined': 'Zero or unresolved energy-defect rate',
    }.get(label, f'Energy fingerprint {label} with declared order {q}')
    if eps > 1e-10 or adapter.noise_floor > 0:
        label, slopes, status = 'undetermined', [None] * 3, Status.ABSTAIN
        reason = 'Limited arithmetic precision or declared state noise blocks energy classification'
    evidence.update(mean_abs_rates=means, energy_scale=scale, floor=floor, rate_limit=rate_limit,
                    epsilon=eps, exponents=slopes, classification=label)
    return result(adapter, 'energy', status, reason, evidence,
                  label if status == Status.FAIL else None, changes)
