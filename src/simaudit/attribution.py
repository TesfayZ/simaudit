"""Recover a flagged law in a bounded-degree polynomial quotient."""
from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from math import comb

import numpy as np

from .laws import Polynomial, log_result, monomials, terms
from .report import Status


@dataclass
class Hypothesis:
    nominal: float
    bounds: tuple[float, float]
    derivative: Polynomial  # nominal law + (value - nominal) * derivative


@dataclass
class Recovery:
    coordinates: tuple[str, ...]
    degree: int
    hypotheses: dict[str, Hypothesis]
    known: list[Polynomial] = field(default_factory=list)
    lag: int = 1


def _match(base, slope, direction, hypothesis):
    """Minimise the relative residual after projection onto the invariant ray."""
    nominal = hypothesis.nominal
    lo, hi = hypothesis.bounds
    if not np.isfinite([lo, hi, nominal]).all() or not lo < nominal < hi:
        raise ValueError('Each hypothesis requires finite bounds enclosing its nominal value')
    # Maximise (direction.T @ template)^2 / (template.T @ template).
    # With affine coefficients the interior maximum solves a linear equation.
    a, b = direction @ base, direction @ slope
    u, v, w = base @ base, base @ slope, slope @ slope
    values = [lo, nominal, hi]
    if b*v != a*w:
        t = nominal + (a*v - b*u) / (b*v - a*w)
        if lo < t < hi:
            values.append(float(t))
    candidates = []
    for t in values:
        vec = base + (t - nominal) * slope
        size = np.linalg.norm(vec)
        if size:
            residual = np.linalg.norm(vec - direction * (direction @ vec)) / size
            candidates.append((float(residual), t))
    if not candidates:
        raise ValueError('Template vanishes throughout the quotient')
    residual, estimate = min(candidates)
    return dict(residual=residual, estimate=estimate, nominal=nominal, bounds=list(hypothesis.bounds),
                identifiable=bool(np.linalg.matrix_rank(np.stack([base,slope])) == 2))


def attribute(study, law, spec, flagged):
    """Called only for a law flagged by this workflow's screen on the same logs."""
    evidence = dict(law=law.name, parameter=None, exploratory=True)
    def finish(status, reason):
        return log_result(study, 'attribution', status, reason, evidence)
    try:
        import sympy as sp
    except ImportError:
        return finish(Status.SKIPPED, 'Install the algebra extra for polynomial attribution')
    if len(spec.hypotheses) < 2:
        return finish(Status.ABSTAIN, 'Supply competing parameter hypotheses')
    if type(spec.degree) is not int or spec.degree < 1 or type(spec.lag) is not int or spec.lag < 1:
        raise ValueError('Require positive integer recovery degree and lag')
    coords = spec.coordinates
    all_coords = study.candidate.coordinates
    if not coords or len(set(coords)) != len(coords) or not set(coords) <= set(all_coords):
        raise ValueError('Recovery coordinates must be a unique subset of log coordinates')
    n = len(coords)
    if comb(spec.degree+n, n) > 2000:
        return finish(Status.PARTIAL, 'Requested dictionary exceeds 2000 monomials')
    cols = [all_coords.index(name) for name in coords]
    nominal = {}
    for e,c in law.terms.items():
        if any(k and i not in cols for i,k in enumerate(e)):
            raise ValueError('Recovery coordinates omit variables used in the screened law')
        nominal[tuple(e[i] for i in cols)] = c
    symbols = sp.symbols(f'x:{n}')
    def expr(poly):
        return sum(sp.Rational(str(c))*sp.prod(v**k for v,k in zip(symbols,e)) for e,c in terms(poly,n))
    basis = sp.groebner([expr(p) for p in spec.known], *symbols, order='grevlex', domain=sp.QQ)
    leading = [p.LM(order=basis.order).exponents for p in basis.polys]
    # A rejected identity cannot be treated as exact when removing known
    # polynomial consequences. Check ideal membership in the recovery coordinates.
    for other in study.laws:
        if not leading or other.kind != 'vanishing' or other.name not in flagged:
            continue
        if all(all(k == 0 or i in cols for i,k in enumerate(e)) for e in other.terms):
            projected = {tuple(e[i] for i in cols):c for e,c in other.terms.items()}
            if basis.reduce(expr(projected))[1] == 0:
                return finish(Status.ABSTAIN, f'Flagged law {other.name} contradicts a recovery assumption')
    powers = []
    conserved = law.kind == 'conserved'
    for degree in range(int(conserved), spec.degree+1):
        for idx in combinations_with_replacement(range(n), degree):
            e = tuple(idx.count(i) for i in range(n))
            if not any(all(i >= j for i,j in zip(e,lm)) for lm in leading):
                powers.append(e)
    if len(powers) < 2:
        return finish(Status.ABSTAIN, 'Quotient has insufficient independent directions')
    def vector(poly):
        remainder = basis.reduce(expr(poly))[1] if leading else expr(poly)
        coefficients = sp.Poly(remainder, *symbols).as_dict()
        if conserved:
            coefficients.pop((0,)*n, None)
        if set(coefficients) - set(powers):
            raise ValueError('Template exceeds the specified quotient dictionary')
        return np.array([float(coefficients.get(e,0)) for e in powers])
    blocks = []
    for group in study.candidate.groups.values():
        x = np.asarray(group, dtype=float)[:,cols]
        phi = monomials(x, powers)
        if conserved:
            if len(x) <= spec.lag:
                return finish(Status.ABSTAIN, 'Trajectory too short for the requested lag')
            phi = phi[spec.lag:] - phi[:-spec.lag]
        blocks.append(phi)
    matrix = np.vstack(blocks)
    scales = np.linalg.norm(matrix, axis=0)
    if not np.isfinite(scales).all():
        return finish(Status.BLOWUP, 'Nonfinite recovery dictionary')
    if min(scales) == 0 or matrix.shape[0] < len(powers):
        return finish(Status.ABSTAIN, 'Insufficient excitation for quotient recovery')
    try:
        _, s, vt = np.linalg.svd(matrix / scales, full_matrices=False)
    except np.linalg.LinAlgError:
        return finish(Status.ABSTAIN, 'Singular value decomposition did not converge')
    evidence.update(coordinates=list(coords), degree=spec.degree, lag=spec.lag, singular_values=s.tolist(),
                    powers=[list(e) for e in powers], sympy=sp.__version__,
                    known_constraints=[[[list(e),c] for e,c in p.items()] for p in spec.known],
                    template_derivatives={name:[[list(e),c] for e,c in h.derivative.items()]
                                          for name,h in spec.hypotheses.items()})
    if s[-2] <= np.finfo(float).eps * max(matrix.shape) * s[0] or s[-1] >= 0.1*s[-2]:
        return finish(Status.ABSTAIN, 'Recovery lacks an isolated invariant direction')
    direction = vt[-1] / scales
    direction /= np.linalg.norm(direction)
    base = vector(nominal)
    fits = {name: _match(base, vector(h.derivative), direction, h) for name,h in spec.hypotheses.items()}
    evidence.update(coefficients=direction.tolist(), hypotheses=fits)
    best, second = sorted(fits, key=lambda k: fits[k]['residual'])[:2]
    hit = fits[best]
    if hit['residual'] >= 1e-3:
        return finish(Status.ABSTAIN, 'No supplied parameter template explains the recovered law')
    if (not hit['identifiable'] or hit['estimate'] in hit['bounds']
            or fits[second]['residual'] <= 10 * max(1e-14, hit['residual'])):
        return finish(Status.ABSTAIN, 'Parameter hypotheses ambiguous or boundary limited')
    if abs(hit['estimate']-hit['nominal']) <= 1e-3 * max(abs(hit['nominal']), 1e-12):
        return finish(Status.ABSTAIN, 'Parameter deviation unresolved at the template threshold')
    evidence.update(parameter=best, estimate=hit['estimate'])
    return finish(Status.FAIL, f'{law.name} is consistent with {best} = {hit["estimate"]:.8g} among supplied hypotheses')
