import json
from pathlib import Path

import numpy as np
import pytest

from simaudit import Feature, Report, Status, Unsupported, check_energy
from simaudit.cli import main
from simaudit.energy import _classify
from simaudit.release import load_factory


EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'
PROBES = json.loads((EXAMPLES / 'oscillator-probes.json').read_text())


@pytest.fixture
def oscillator(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLES))
    return load_factory('passive_oscillator:factory')


def test_energy_cli(oscillator, tmp_path):
    path = tmp_path / 'audit.json'
    assert main(['energy', '--factory', 'passive_oscillator:factory', '--probes',
                 str(EXAMPLES / 'oscillator-probes.json'), '--json', str(path), '--strict']) == 0
    r, = Report.loads(path.read_text()).results
    assert r.evidence['classification'] == 'consistent'
    rates = [row['signed_rates'] for row in r.evidence['probes']]
    np.testing.assert_allclose(r.evidence['mean_abs_rates'], np.mean(np.abs(rates), axis=0))
    assert len(r.modifications) == 3


def test_changed_reference_energy(oscillator):
    oscillator.energy_fn = lambda s: s[0]**2 + s[1]**2 / 2
    r = check_energy(oscillator, **PROBES)
    assert (r.status, r.fault_class) == (Status.FAIL, 'model')


def test_lower_order_energy(oscillator):
    oscillator.step_fn = oscillator.native_step_fn = lambda s, u, h: s + h * np.array([s[1], -s[0]])
    r = check_energy(oscillator, **PROBES)
    assert (r.status, r.fault_class) == (Status.FAIL, 'order-1')


def test_order_fallback_at_declaration():
    # Three exponents 2.9, 4.2, 4.1. The first misses the q=4 threshold.
    rates = 2.0 ** np.array([11.2, 8.3, 4.1, 0])
    label, _, status = _classify(rates, 0, 2, 4)
    assert (label, status) == ('order-4', Status.ABSTAIN)


def test_exact_threshold_at_nonunit_energy_scale():
    rates = [factor * (9 / 103) for factor in (512, 64, 8, 1)]
    label, slopes, status = _classify(rates, 0., 2., 4)
    assert (label, slopes, status) == ('consistent', [3., 3., 3.], Status.PASS)


@pytest.mark.parametrize('field,value,status', [
    ('energy_fn', None, Status.SKIPPED), ('declared_order', None, Status.SKIPPED),
    ('native_step_fn', None, Status.SKIPPED), ('conservative', None, Status.PARTIAL),
    ('smooth', False, Status.PARTIAL), ('features', [], Status.PARTIAL),
    ('features', [Feature('contacts', 'unsupported', 'Contact active', True)], Status.PARTIAL),
    ('native_step_fn', lambda s,u,h: s, Status.FAIL),
    ('energy_fn', lambda s: np.nan, Status.BLOWUP), ('dtype', 'float32', Status.ABSTAIN),
    ('energy_fn', lambda s: np.float32(0.5 * s @ s), Status.ABSTAIN),
    ('noise_floor', 0.01, Status.ABSTAIN),
])
def test_energy_preconditions(oscillator, field, value, status):
    setattr(oscillator, field, value)
    r = check_energy(oscillator, **PROBES)
    assert r.status == status
    if status == Status.FAIL:
        assert r.fault_class == 'adapter'


def test_exact_conservation_and_noise_floor(oscillator):
    noisy = check_energy(oscillator, **PROBES, energy_noise_floor=1e-4)
    assert (noisy.status, noisy.evidence['classification']) == (Status.ABSTAIN, 'undetermined')
    assert noisy.evidence['floor'] == 1e-4
    oscillator.step_fn = oscillator.native_step_fn = lambda s,u,h: s
    exact = check_energy(oscillator, **PROBES)
    assert exact.status == Status.ABSTAIN and exact.evidence['mean_abs_rates'] == [0]*4


def test_large_rate_guard(oscillator):
    oscillator.step_fn = oscillator.native_step_fn = lambda s,u,h: s + h * 10
    r = check_energy(oscillator, **PROBES)
    assert (r.status, r.evidence['classification']) == (Status.ABSTAIN, 'unstable')


def test_unsupported_energy_retains_completed_probes(oscillator):
    def energy(s):
        if s[0] < 0:
            raise Unsupported('Energy undefined in this region')
        return s @ s / 2
    oscillator.energy_fn = energy
    r = check_energy(oscillator, **PROBES)
    assert r.status == Status.PARTIAL and len(r.evidence['probes']) == 2


@pytest.mark.parametrize('mass,seed,n,label,status,expected_rates', [
    (1.0, 1, 150, 'consistent', Status.PASS,
     [0.004357999838236459, 0.0002156297233575799, 1.2461822043086101e-5, 7.545705278388937e-7]),
    (1.01, 412901, 30, 'model', Status.FAIL,
     [0.05841125307236048, 0.04599979277431974, 0.04547425810901604, 0.04544245837700487]),
    (1.3, 11, 30, 'unstable', Status.ABSTAIN,
     [2.2898059157605193, 2.220049342825528, 2.2192464562861804, 2.2191397760013136]),
])
def test_acrobot_paper_trial(monkeypatch, mass, seed, n, label, status, expected_rates):
    """First trials of three DAFD shipped_acrobot_fingerprint_trials.csv cases."""
    pytest.importorskip('gymnasium')
    monkeypatch.syspath_prepend(str(EXAMPLES.parent / 'validation'))
    from known_cases import acrobot

    def nominal_energy(s):
        q1, q2, v1, v2 = s
        coupling = np.cos(q2)
        kinetic = ((3.5 + coupling) * v1**2 + (2.5 + coupling) * v1*v2 + 1.25 * v2**2) / 2
        return kinetic - 14.7 * np.cos(q1) - 4.9 * np.cos(q1 + q2)

    adapter = acrobot(mass=mass)
    adapter.declared_order = 4
    adapter.smooth = adapter.conservative = True
    adapter.energy_fn = nominal_energy
    # Match the paper's coordinate-wise draw, passive action index, and macro trajectory.
    x = np.random.default_rng(seed).uniform(-1.5, 1.5, (4, n))[:, 0]
    states = []
    for _ in range(50):
        states.append(x)
        x = adapter.step(x, 1, adapter.dt)
    r = check_energy(adapter, states, [1] * len(states))
    assert (r.evidence['classification'], r.status) == (label, status)
    np.testing.assert_allclose(r.evidence['mean_abs_rates'], expected_rates, rtol=2e-8, atol=1e-13)
