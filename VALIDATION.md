# Validation of 0.1.0a1

Local measurements, 11 November 2026. Protocols were written before package
measurements: docs/REQUIREMENTS.md (software controls) and validation/PROTOCOL.md
(paper cases). Full results are in evidence/*.json. These are new package runs,
not imports of the papers' result tables and not an independent scientific
replication of their complete studies.

## Paper cases

Both installed combinations were exercised: **Gymnasium 1.4.0 / MuJoCo 3.16.0**
and **Gymnasium 1.3.0 / MuJoCo 3.17.0**, under Python 3.14.6.

| Case | Result in both combinations | Scope |
|---|---|---|
| CartPole masscart x2 | FAIL: no-op relative to fresh reference | 100 supplied starts, alternate actions |
| CartPole masspole x2 | FAIL: stale constants | Same 100 starts |
| CartPole length x2 | FAIL: partly stale update | Mechanism annotation comes from adapter source inspection |
| CartPole gravity x2 | PASS: consistent | 100 starts, no observed discrepancies |
| RK4 heavy top, ball joint | FAIL: 23/24 probes below declared order; 1/24 ABSTAIN | Median fine-pair slope 1.95918; conservative geometry, h=.05 |
| RK4 heavy top, three hinges | PASS: 24/24 probes | Median fine-pair slope 3.86154; healthy geometry control |
| Reacher post-step observation | FAIL: discrepancy from refreshed map | 24 starts; largest discrepancy 2.39071e-6 |

No 4.0.0 engine was installed or tested. No result is inferred for it, for other
release versions, or for shipped Ant/Humanoid with contacts and actuation.

### Addendum, 14 November 2026: MuJoCo 4.0.1

The same protocol and seed were run unchanged with Gymnasium 1.4.0 / MuJoCo 4.0.1
under Python 3.12.3 (evidence/paper_cases_mj401.json). CartPole results are
identical. The heavy-top **ball-joint case now PASSes on 24/24 probes** (median
fine-pair slope 3.915; it was FAIL with 1.959 on 3.16.0), and the hinge control
still passes (3.862). The Reacher observation check still FAILs (largest
discrepancy 2.39071e-6). This is a measurement on the conservative heavy-top
model only; it does not establish where the change occurred between 3.17.0 and
4.0.1, and 4.0.0 remains untested.

### Addendum: release-worker isolation bug (fixed before first publication)

When simaudit was pip-installed, `compare_releases` put the caller's whole
site-packages directory on the workers' PYTHONPATH, which could make both workers
import the caller's NumPy/simulator instead of each interpreter's own versions.
The validation runs above used a source tree, where only the package directory
was exposed, so they were not affected. Workers now receive a temporary directory
containing only the simaudit package, with a regression test. The Acrobot release
controls were not rerun after this change.
The Reacher reference shares the observation formula and only checks its cache
consistency. CartPole's fresh reference rebuilds constants under an explicitly
supplied parameter contract; direct attribute mutation need not be an upstream
supported API. A FAIL here is scoped to that comparison.

The two runs reuse the same probe design and are not 48 independent starts for
an engine-wide population claim. The ball-joint abstention is retained. No
threshold was relaxed to remove it.

## Release comparison and controls from the joint paper

Using isolated workers, Gymnasium 1.3.0 versus 1.4.0 Acrobot agreed exactly on
all recorded first steps. On the same 1.3.0 version, the book-versus-nips control
was detected (maximum first-step difference 0.00147277) and the mass2=1.3 control
was detected (0.0609411). Each used 24 paired starts, 5 recorded steps, and the
first-step threshold 1e-13. This validates detection, not the joint paper's
algebraic screening, constraint localisation or parameter attribution, which
are deferred features. The full historical 350-pair matrix was not rerun.

## General-purpose software controls

Euler and RK4 scalar ODE controls each had 0 failures and 0 abstentions across
100 starts. The two-sided 95% Wilson false-alarm interval per control is
approximately [0, 0.0370]. This is conditional on those chosen scalar controls,
not calibration across arbitrary simulators. Float32 order and observation
controls abstain; declared-noise order controls abstain. Float64 observation
controls pass. Tests also cover missing capabilities, bad restoration, wrong
native references, nonfinite trajectories, unknown features, unlogged assignment
modifications, schema rejection, CLI, subprocess comparison and pytest statuses.

Implementation testing exposed two issues which were corrected before final
measurements: independent float32 arithmetic needed a dtype-aware replay
comparison floor, and returned mutable buffers needed copying. The frozen
scientific diagnostic thresholds were not changed. A zero-torque constant-speed
hinge test was corrected to use a gravity-loaded axis; exact flow correctly
abstains because there is no truncation error to estimate.

## Packaging and software test result

The final suite has 34 passing tests (33 at the 11 November receipt, plus the isolation regression test; 34 pass on Python 3.12.3 with MuJoCo 4.0.1), including the optional MuJoCo tests. A wheel
and source distribution were built. The wheel was installed without network
access into a new virtual environment using existing system dependencies, and its
CLI ran a copied custom-simulator example from /tmp, outside the source tree.
The installed package imported without loading Gymnasium or MuJoCo, and pytest
discovered its plugin entry point. This is package isolation, not a clean-room
reinstallation of all third-party dependencies.

Run `python validation/verify_receipt.py` from the repository to check retained
hashes. The repository receipt is excluded from distributions to avoid a
self-referential archive hash.

## Reproduce

From this package directory, after installing the package:

```bash
python -m pytest
python -m build
PYTHONPATH=validation python validation/run_controls.py \
  --report /tmp/package_controls.json --summary /tmp/false_alarm_controls.json
PYTHONPATH=src:validation /path/to/simulator/python validation/run_known_cases.py /tmp/paper_cases.json
PYTHONPATH=src:validation python validation/run_release_cases.py \
  --python-a /path/to/version-a/python --python-b /path/to/version-b/python \
  --out /tmp/release_controls.json
```

`evidence/validation_receipt.json` records final execution checks and hashes.
A GitHub Actions workflow is included (.github/workflows/tests.yml); hosted CI had
not been run when this receipt was written. Stable publication still requires
external usability tests, a larger held-out control set, a complete dependency
licence inventory and additional version coverage. The name `simaudit` was
unregistered on PyPI on 14 November 2026.
