# SimAudit

SimAudit tests changes to simulators on states and actions you supply. It can detect a parameter write that left stale constants, measure numerical order, compare engine versions and check the observations a simulator returns. Version **0.2.0a1** also tests passive energy drift, fits a named parameter on independent starts, and screens trajectory logs for violations of specified physical laws.

The core requires Python 3.11+ and NumPy. Gymnasium, MuJoCo, SymPy and pytest are optional. APIs are in alpha. [PyPI](https://pypi.org/project/simaudit/) · [Changelog](https://github.com/TesfayZ/simaudit/blob/v0.2.0a1/CHANGELOG.md) · [Issues](https://github.com/TesfayZ/simaudit/issues)

## Install and run

From this checkout, install the alpha and run the included oscillator. The demo uses separate implementations of the RK4 step and its analytical update.

```bash
python -m pip install .
simaudit plan drift --inputs simaudit.demo:inputs
simaudit diagnose drift --inputs simaudit.demo:inputs --strict --json drift.json
```

The two checks should both print `PASS` and coverage should be `complete`. The JSON retains the refinement measurements. To exercise all eight branches, run `simaudit diagnose full --inputs simaudit.demo:inputs --strict`. The demo uses the same interpreter twice as a control for release comparison.

For the published package, `python -m pip install --pre simaudit` selects the latest alpha on PyPI. The commands above require 0.2.0a1 or later. Install `simaudit[algebra]` for polynomial attribution, `simaudit[gymnasium]` for CartPole or `simaudit[mujoco]` for the native MuJoCo and Reacher adapters. From a checkout use, for example, `python -m pip install '.[algebra,mujoco]'`.

## Choose a check

| Question | Command after `simaudit diagnose` | Required inputs |
|---|---|---|
| Did a parameter write take effect? | `assignment` | Replay adapter, probes, direct assignment and fresh construction |
| Does a new release take a different first step? | `release` | Two installed interpreters, trusted adapter factories and matched probes |
| Is the observation stale? | `observation` | Replay, emitted observation and a reference observation map |
| Does the integrator reach its declared order? | `order` | Smooth replay, adjustable timestep, state distance and declared order |
| Does passive energy drift agree with the declaration? | `energy` or `drift` | Reference energy and conservative probes. `drift` also measures state order. |
| What value of a named parameter fits these dynamics? | `fit` | Bounded affine energy family, training and held-out start groups, tolerances |
| Which named laws changed? | `screen` | Independent healthy and candidate groups, matched conditions and polynomial laws |
| Which parameter explains a flagged law? | `attribution` | Screen inputs, known constraints and competing bounded affine templates |

Use `--inputs my_module:factory` to provide a trusted Python callable returning `WorkflowInputs`. A plan previews prerequisites without stepping the simulator. A full diagnosis runs eligible checks and reports missing inputs. Factories execute locally with your process permissions. [Workflow guide and examples](https://github.com/TesfayZ/simaudit/blob/v0.2.0a1/docs/DIAGNOSTIC_WORKFLOW.md)

## Test your own simulator

For a single check, wrap the simulator step and its native reference. Each callback must restore everything needed to reproduce the requested state, action and timestep.

```python
import numpy as np
from simaudit import Adapter, Feature, Report, check_order

def step(state, action, dt):
    return state + dt * (state + action)

def native_reference(state, action, dt):
    return np.asarray([(1 + dt) * state[0] + dt * action])

adapter = Adapter(
    name="Euler scalar ODE", step_fn=step, native_step_fn=native_reference,
    features=[Feature("dynamics", "tested", "Smooth scalar ODE", True)],
    smooth=True, declared_order=1, dt=0.2,
)
result = check_order(adapter, [[0.5], [1.0], [1.5]], [0, 0, 0])
Report([result]).write("audit.json")
print(result.status.value)
```

The single-check CLI accepts a trusted adapter factory and a JSON file containing matching `states` and `actions`. From this checkout run the following example. [Adapter contract and energy units](https://github.com/TesfayZ/simaudit/blob/v0.2.0a1/docs/ADAPTERS.md)

```bash
PYTHONPATH=examples simaudit energy --factory passive_oscillator:factory \
  --probes examples/oscillator-probes.json --strict --json energy.json
```

For release comparisons, supply `--python-a` and `--python-b` to `simaudit release`. Each worker uses its own installed simulator and records its version. Both workers must be able to import the factory. The first-step difference determines the verdict. The report also records subsequent divergence.

## Read a verdict

`PASS` means the check met its criterion on the supplied probes. `FAIL` records a discrepancy. `ABSTAIN` means the signal could not resolve the question, `SKIPPED` identifies missing capabilities, `PARTIAL` identifies unsupported conditions, and `BLOWUP` records a numerical failure. Inspect the scope, reference, feature inventory and individual probe results alongside the verdict. A shared error in a simulator and its reference can escape detection. Parameter attribution is conditional on the supplied hypotheses.

A failed check or empty report exits 1. With `--strict`, any nonpass or incomplete workflow also exits 1. Invalid configuration exits 2. The original commands, pytest fixture and report schema 1.0 remain available. Workflow JSON embeds that report under `report`. [Report format](https://github.com/TesfayZ/simaudit/blob/v0.2.0a1/docs/REPORTS.md)

## Validation and development

On the conservative heavy-top model, the ball-joint RK4 check found below-declaration order on 23 of 24 probes with MuJoCo 3.2.7, 3.10.0, 3.16.0 and 3.17.0. One probe abstained in each run. All 24 passed on 4.0.0 and 4.0.1. The three-hinge control passed on every tested engine. On the Acrobot and Reacher workflow controls, six law-level attributions selected the injected parameter. Recovery abstained on the structural Acrobot fault. [Measurements, versions and reproduction commands](https://github.com/TesfayZ/simaudit/blob/v0.2.0a1/VALIDATION.md)

```bash
python -m pip install -e '.[dev,mujoco,algebra]'
python -m pytest -q
python -m build
python -m twine check --strict dist/*
```

The `pytest` extra exposes the `simaudit_check` fixture. Pass a check result to it inside a test. A discrepancy fails the test. Other nonpasses skip with the diagnostic reason. Use `pytest --simaudit-strict --simaudit-json results.json` for strict serial reporting.

## Method papers and licence

- Tesfay Zemuy Gebrekidan, *Diagnosing Model and Discretisation Errors in Reinforcement Learning Simulators Using Invariants* (2026). Source for assignment consistency, energy fingerprints, state refinement and energy-based parameter fitting. [Zenodo](https://zenodo.org/records/24114609)
- Tesfay Zemuy Gebrekidan and Hadush Hailu Gebrerufael, *Diagnosing Faults in Reinforcement Learning Simulators with Canonical Polynomial Invariants* (2026). Source for release and observation controls, trajectory-level law screening and quotient recovery. [arXiv](https://arxiv.org/abs/2609.13194v2)

SimAudit is MIT licensed. The citation file credits the software and both method papers separately. [Citation](https://github.com/TesfayZ/simaudit/blob/v0.2.0a1/CITATION.cff) · [BibTeX](https://github.com/TesfayZ/simaudit/blob/v0.2.0a1/references.bib) · [Reuse notices](https://github.com/TesfayZ/simaudit/blob/v0.2.0a1/THIRD_PARTY.md) · [Earlier software archive](https://doi.org/10.5281/zenodo.24244959)
