"""Named polynomial laws on independent matched trajectory groups."""
from dataclasses import dataclass, field
from itertools import combinations, islice
import hashlib
import math

import numpy as np

from .report import Feature, Result, Status

Polynomial = dict[tuple[int, ...], float]


@dataclass
class LogArm:
    name: str
    coordinates: tuple[str, ...]
    groups: dict  # independent start ID -> (time, coordinates) array
    conditions: dict  # sampling, integration and noise
    independent: bool


@dataclass
class Law:
    name: str
    kind: str   # vanishing or conserved
    terms: Polynomial


@dataclass
class LawStudy:
    healthy: LogArm
    candidate: LogArm
    laws: list[Law]
    recovery: dict = field(default_factory=dict)
    alpha: float = 0.01
    floor: float = 1e-8


def log_result(study, check, status, reason, evidence):
    return Result(check, status, reason, f'{study.candidate.name} compared with {study.healthy.name}',
                  ['Inference depends on declared independent sampling, matched conditions and coordinate definitions.'],
                  [Feature('observations', 'tested' if status in (Status.PASS, Status.FAIL) else 'present_not_tested',
                           'Supplied trajectory groups and named polynomial laws', True)],
                  versions={'numpy': np.__version__}, evidence=evidence)


def terms(poly, n):
    if not poly or any(len(e) != n or any(type(k) is not int or k < 0 for k in e)
                       or not math.isfinite(c) for e,c in poly.items()):
        raise ValueError('Supply finite polynomial coefficients and nonnegative integer powers per coordinate')
    if not any(poly.values()):
        raise ValueError('Zero polynomial supplies no law')
    return list(poly.items())


def monomials(x, powers):
    return np.column_stack([np.prod(x ** e, axis=1) for e in powers])


def _permutation(healthy, candidate, count):
    pooled = np.r_[healthy, candidate]
    target = float(np.sum(candidate))
    # The paper's trajectory test treats sums within this tolerance as ties.
    slack = 1e-12 * max(1., abs(target))
    splits = combinations(range(len(pooled)), len(candidate))
    hits = 0
    while batch := list(islice(splits, 1024)):
        hits += np.count_nonzero(pooled[np.asarray(batch)].sum(axis=1) >= target - slack)
    return float(hits / count)


def log_requirements(study):
    arms = [study.healthy, study.candidate]
    if arms[0].coordinates != arms[1].coordinates or arms[0].conditions != arms[1].conditions:
        return 'Coordinate order or sampling, integration and noise conditions differ'
    coords = arms[0].coordinates
    if not coords or len(set(coords)) != len(coords):
        raise ValueError('Supply unique coordinate names')
    if not {'sampling','integration','noise'} <= set(arms[0].conditions):
        return 'Missing sampling, integration or noise metadata'
    if not all(a.independent is True for a in arms) or set(arms[0].groups) & set(arms[1].groups):
        return 'Screen requires independent groups within and across arms'
    return None


def screen_laws(study):
    arms = [study.healthy, study.candidate]
    names = [law.name for law in study.laws]
    if not names or not all(names) or len(set(names)) != len(names):
        raise ValueError('Supply unique named laws')
    if not 0 < study.alpha < 1 or not math.isfinite(study.floor) or study.floor < 0:
        raise ValueError('Invalid screen alpha or raw residual floor')
    evidence = dict(alpha=study.alpha, raw_floor=study.floor, correction='Bonferroni',
                    flagged=[], laws={}, sources=[])
    def finish(status, reason):
        return log_result(study, 'screen', status, reason, evidence)
    issue = log_requirements(study)
    if issue:
        return finish(Status.PARTIAL, issue)
    coords = arms[0].coordinates
    data = []
    seen = set()
    for arm in arms:
        arrays = []
        hashes = {}
        for key, values in arm.groups.items():
            x = np.asarray(values, dtype=float)
            if x.ndim != 2 or x.shape[1] != len(coords) or not len(x):
                raise ValueError('Each log group must be a nonempty time-by-coordinate matrix')
            if not np.isfinite(x).all():
                return finish(Status.BLOWUP, 'Nonfinite logged coordinates')
            digest = hashlib.sha256(x.tobytes()).hexdigest()
            if digest in seen:
                return finish(Status.PARTIAL, 'Repeated trajectory data across independent units')
            seen.add(digest)
            hashes[key] = digest
            arrays.append(x)
        evidence['sources'].append(dict(name=arm.name, groups=hashes, conditions=arm.conditions, coordinates=list(coords)))
        data.append(arrays)
    sizes = [len(arm) for arm in data]
    if min(sizes) < 2:
        return finish(Status.ABSTAIN, 'Insufficient independent log units')
    count = math.comb(sum(sizes), sizes[1])
    evidence['permutations'] = count
    if len(names)/count >= study.alpha:
        return finish(Status.ABSTAIN, 'No attainable corrected p-value is below alpha')
    if count > 200000:
        return finish(Status.ABSTAIN, 'Exact permutation enumeration exceeds the supported 200000 splits')
    unresolved = []
    for law in study.laws:
        if law.kind not in ('vanishing', 'conserved'):
            raise ValueError('Unknown law kind')
        powers, coeff = zip(*terms(law.terms, len(coords)))
        raw, relative = [[], []], [[], []]
        for i, arrays in enumerate(data):
            for x in arrays:
                if law.kind == 'conserved' and len(x) < 2:
                    break
                values = monomials(x, powers) * coeff
                signal = values.sum(axis=1)
                magnitude = np.std(signal) if law.kind == 'conserved' else np.sqrt(np.mean(signal**2))
                scale = np.sqrt(np.mean(np.sum(np.abs(values), axis=1)**2))
                if not np.isfinite([magnitude, scale]).all():
                    return finish(Status.BLOWUP, 'Nonfinite polynomial residual or scale')
                raw[i].append(float(magnitude))
                relative[i].append(float(magnitude / scale) if scale else 0.)
        if list(map(len, raw)) != sizes:
            unresolved.append(law.name)
            continue
        suppressed = max(map(max, raw)) <= study.floor
        p = 1.0 if suppressed else _permutation(*relative, count)
        adjusted = min(1., p * len(names))
        evidence['laws'][law.name] = dict(kind=law.kind, terms=[[list(e), c] for e,c in zip(powers,coeff)],
            raw=raw, relative=relative, below_floor=suppressed, p=p, adjusted_p=adjusted)
        if adjusted < study.alpha:
            evidence['flagged'].append(law.name)
    evidence['unresolved'] = unresolved
    if evidence['flagged']:
        return finish(Status.FAIL, 'Elevated residuals for ' + ', '.join(evidence['flagged']))
    if unresolved:
        return finish(Status.PARTIAL, 'Conserved quantities require multi-step trajectories')
    return finish(Status.PASS, 'No named law rejected on the supplied trajectory groups')
