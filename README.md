# simaudit

Installable diagnostics for **your own simulator**, with explicit limits on what
was assessed. Supply a small Python adapter; NumPy is the only core dependency.
Gymnasium and MuJoCo are optional examples. Version **0.1.0a2** is an alpha,
not a generally calibrated simulator-certification service.

> **Alpha:** APIs may change. Feedback is welcome via
> [GitHub issues](https://github.com/TesfayZ/simaudit/issues).
> Package: [PyPI](https://pypi.org/project/simaudit/).

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24244959.svg)](https://doi.org/10.5281/zenodo.24244959)

## Install

```bash
# Pre-release: pip skips alpha versions unless you ask for them.
python -m pip install --pre simaudit
# Optional adapters and pytest integration:
python -m pip install --pre 'simaudit[gymnasium,mujoco,pytest]'
# Or from a source checkout:
python -m pip install .
```

Python 3.11 or newer (CI matrix 3.11-3.14; local validation on 3.14.6 and 3.12.3). Nothing downloads, runs experiments, or imports an optional
simulator on package import. For development: `python -m pip install '.[dev]'`.

## Test your own simulator

```python
import numpy as np
from simaudit import Adapter, Feature, Report, check_order

# Wrap your simulator so each call restores every dependency of its next step.
def step(state, action, dt):
    return state + dt * (state + action)

def native_reference(state, action, dt):
    return np.asarray([(1 + dt) * state[0] + dt * action])

adapter = Adapter(
    name="my simulator", step_fn=step, native_step_fn=native_reference,
    features=[Feature("dynamics", "tested", "Smooth scalar ODE, no hidden state", True)],
    smooth=True, declared_order=1, dt=0.2,
)
result = check_order(adapter, [[0.5], [1.0], [1.5]], [0, 0, 0])
report = Report([result])
print(report.summary())
report.write("audit.json")
```

The native reference should call the original simulator without the adapter
transformation being tested. A user-provided reference and inventory are trusted
inputs, not automatically verified specifications. See [the adapter guide](docs/ADAPTERS.md)
for mutable simulators, noise, arbitrary state metrics, assignment factories,
observation maps and capability requirements.

The CLI loads a trusted local `module:function` returning an adapter. From this
directory, run the included NumPy-only example:

```bash
PYTHONPATH=examples simaudit order --factory custom_simulator:factory \
  --probes examples/probes.json --strict --json audit.json
```

Python factories execute code with your Python process's privileges. Probe files
contain only `states` and matching `actions`. The same interface works with your
own importable adapter module. `python -m simaudit` is equivalent to `simaudit`.

## Four independent checks

| Check | Python API | Needs | What it reports |
|---|---|---|---|
| Parameter assignment | `check_assignment` | Replay, native reference, assignment factory | Baseline/direct/fresh comparison; no-op, stale, consistent or unresolved |
| Numerical order | `check_order` | Replay, native reference, timestep, declared order, smoothness, distance | Stable refinement slopes, below-declaration order, or inability to diagnose |
| Release/variant comparison | `compare_releases` | Factory and two supplied Python interpreters | First-step differences on identical inputs; later differences recorded separately |
| Observation consistency | `check_observation` | Replay, native reference, observed and reference observation maps | Differences above the precision/noise floor |

The package doesn't implement the papers' energy-fingerprint classifier,
algebraic localisation, parameter identification or automatic equation diagnosis.
A numerical-order pass cannot establish that equations are correct.

## Results and limitations

Every result contains **PASS, FAIL, ABSTAIN, SKIPPED, PARTIAL or BLOWUP**, a reason,
scope, limitations, mandatory feature coverage, modifications, versions and raw
probe measurements. Limitations and untested features are printed beside results.

- Float32 or a declared noise floor blocks a healthy consistency verdict in
  checks A, B and D. A and D can still flag discrepancies above their floors.
- Missing capabilities return `SKIPPED`; unsupported smoothness/features give
  `PARTIAL`; unresolved or unstable slopes give `ABSTAIN`.
- Nonfinite trajectories or detected engine resets give `BLOWUP`.
- A failed adapter replay/native-step self-test blocks downstream diagnosis.
- `PASS` covers the supplied probes under the declared conditions only. A wrong
  reference, shared implementation error or unreported hidden state can escape it.

By default `FAIL` exits 1; other nonpasses print a warning and exit 0. `--strict`
exits 1 on **any** nonpass. Configuration errors exit 2. Empty reports are never
all-clear. [Report contract](docs/REPORTS.md).

## Optional adapters and releases

```bash
simaudit assignment --factory simaudit.adapters.cartpole:cartpole \
  --probes examples/cartpole-probes.json --parameter masscart --value 2 --json cartpole.json
```

This deliberately detects the paper's assignment discrepancy and exits 1. The
fresh CartPole reference explicitly rebuilds dependent constants; it does not
assert that Gymnasium promises arbitrary attribute mutation.

`mujoco_xml(xml)` supplies native replay for smooth autonomous rigid models and
rejects unsupported features. `reacher()` supplies a cache-consistency comparison
against the shipped observation map after `mj_forward`. Both live in
`simaudit.adapters.mujoco`; imports are lazy. Custom adapters are not subject to
these built-in models' limitations but must declare their own.

For a release pair, create two separate environments yourself, with NumPy and
the simulator versions you want, then supply their Python paths:

```bash
simaudit release --factory simaudit.adapters.cartpole:cartpole \
  --probes examples/cartpole-probes.json --python-a /path/a/bin/python \
  --python-b /path/b/bin/python --strict --json releases.json
```

The worker runs the current package source in each interpreter. The package does
not install or upgrade either environment. Your custom factory module must also
be importable there. Missing versions remain untested. A changed first step means
compatibility changed, not necessarily that either release is wrong.

## pytest and verification

Install the `pytest` extra to use the auto-discovered `simaudit_check` fixture:

```python
def test_simulator(simaudit_check):
    simaudit_check(check_order(adapter, [[1.0]], [0]))
```

`FAIL` fails the test; other nonpasses skip with their original status and full
reason. `pytest --simaudit-strict --simaudit-json results.json` fails the session
on any nonpass and preserves the separate audit counts. Reporting is serial;
pytest-xdist aggregation is not supported.

Run software tests with `python -m pytest`. Build with
`python -m build`. See [validation results](VALIDATION.md),
[prospective paper-case protocol](validation/PROTOCOL.md), and
[requirements and acceptance criteria](docs/REQUIREMENTS.md). These are small package tests
and selected reproductions, not a general false-alarm guarantee.

## Method papers

simaudit draws on two papers. Assignment consistency and numerical order follow
*Diagnosing Model and Discretisation Errors in Reinforcement Learning Simulators Using Invariants* (item 1, a Zenodo preprint); release comparison, observation consistency
and the positive-control validation follow *Diagnosing Faults in Reinforcement Learning Simulators with Canonical Polynomial Invariants* (item 2, on arXiv). Item 1's
energy-fingerprint classifier and item 2's algebraic screening and localisation
method are **not** implemented in this alpha. Please cite the papers relevant to
the checks you use. The simaudit software itself has a separate Zenodo record and
DOI (see CITATION.cff).

1. **Tesfay Zemuy Gebrekidan** (2026). *Diagnosing Model and Discretisation Errors
   in Reinforcement Learning Simulators Using Invariants*.
   [Zenodo](https://zenodo.org/records/24114609).
2. **Tesfay Zemuy Gebrekidan and Hadush Hailu Gebrerufael** (2026).
   *Diagnosing Faults in Reinforcement Learning Simulators with Canonical
   Polynomial Invariants*. [arXiv:2609.13194v2](https://arxiv.org/abs/2609.13194v2).

[CITATION.cff](CITATION.cff), [BibTeX](references.bib), and
[reuse and licence notices](THIRD_PARTY.md) preserve the attribution. MIT licensed.

## What it has found so far

From [VALIDATION.md](VALIDATION.md); each row is scoped to the supplied probes.

| Case | Result |
|---|---|
| Gymnasium CartPole `masscart`/`masspole`/`length` assignment | FAIL (no-op / stale / partly stale) vs a freshly constructed reference; `gravity` PASS |
| MuJoCo 3.16.0 and 3.17.0, RK4 heavy top on a ball joint | FAIL: 23/24 probes below declared order 4 (1/24 ABSTAIN), median slope 1.96 |
| MuJoCo **4.0.1**, same model and probes | PASS 24/24, median slope 3.92 |
| Same model with three hinges (control) | PASS in every tested engine |
| Reacher post-step observation vs refreshed map | FAIL, largest discrepancy 2.4e-6 (3.16.0, 3.17.0 and 4.0.1) |

MuJoCo 4.0.0 itself was not installed or tested; no result is inferred for it.

## Roadmap

Not implemented in this alpha, and not claimed anywhere in this package:

- **0.2:** energy-refinement fingerprint (model-versus-discretisation label), a
  decision table combining checks, and a one-parameter fit, each only with its
  negative-control suite passing.
- **0.3:** algebraic invariant screening behind an optional extra.
- Other classic-control assignment adapters, vector environments, and shipped
  Ant/Humanoid with contacts and actuation.

## Documentation

| File | Purpose |
|---|---|
| [docs/ADAPTERS.md](docs/ADAPTERS.md) | Writing an adapter and declaring its features |
| [docs/REPORTS.md](docs/REPORTS.md) | Report schema, statuses and JSON round trip |
| [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) | Acceptance criteria recorded before validation |
| [VALIDATION.md](VALIDATION.md) | What was actually run, and the limits of those runs |
| [validation/PROTOCOL.md](validation/PROTOCOL.md) | Paper-case protocol and scripts |
| [evidence/README.md](evidence/README.md) | Archived result files and the hash receipt |
| [CHANGELOG.md](CHANGELOG.md) | Release history |
