# SimAudit 0.2.0a1 validation

Local release review, 19 November 2026. The comparison baseline was the published [0.1.0a2 wheel](https://pypi.org/project/simaudit/0.1.0a2/), SHA-256 `e45640e9ed05c269d98fff1c5afeee8afd75a267cf78000a836824f8d77fc1e4`. All 12 Python modules matched tag `v0.1.0a2` byte for byte. Its 34 tests passed. The release adds the checks and workflow described in the [changelog](CHANGELOG.md).

## Agreement with the method papers

| Measurement | Result and scope |
|---|---|
| DAFD energy decision, Definition 1 | All 1,200 archived shipped-Acrobot trial labels across 36 configurations matched when classified from the four recorded rates and mean start energy. Three independent fresh replays also matched the four recorded rates for the healthy, mass +0.01 and mass 1.3 examples. |
| DAFD named-mass fit | All five injected Acrobot masses (0.7, 0.9, 1.001, 1.1, 1.3) fell within the predeclared 0.001 absolute error tolerance. Each fit used eight training starts and four separate held-out starts. The package adds bounds, excitation checks and an acceptable interval. |
| Polynomial screen | The corrected p-value of every named law and the rejected law sets agreed with the paper implementation on seven clean/injected Acrobot and Reacher comparisons and 20 additional noisy healthy comparisons. None of the 20 noisy comparisons flagged a law or abstained at the tested noise level. |
| Quotient recovery | Seven attempted recoveries matched the paper implementation’s smallest-to-next singular-value ratio within `rtol=1e-5, atol=1e-12`. Six law-level attributions named the injected mass, inertia or link length. The structural Acrobot fault flagged energy and recovery abstained. |

The DAFD comparison uses revision `1e69c46d3769270321e74b23c1ea39715efde3bb`. The polynomial comparison uses `687c868c204b8ec537451b5e046532de80197d9d`. Attribution uses bounded affine templates and the same sample that triggered the screen. The reported counts cover the listed probes and noise setting. [Workflow measurements and source hashes](evidence/workflow_cases.json) · [Method provenance](THIRD_PARTY.md)

## Native simulator controls

The original protocol fixes the seed, 24 heavy-top initial velocities, collision-free geometry and a 0.05 model-time interval with 1, 2, 4 and 8 substeps. The table shows the number of probe verdicts and the median fine-pair slope across the 24 recorded slopes. [Protocol](validation/PROTOCOL.md) · [Per-version results and full report hashes](evidence/native_controls.json)

| MuJoCo | Gymnasium | Ball joint, RK4 declared | Ball slope | Three hinges |
|---|---|---|---:|---|
| 3.2.7 | 1.3.0 | 23 FAIL, 1 ABSTAIN | 1.959 | 24 PASS, median 3.862 |
| 3.10.0 | 1.3.0 | 23 FAIL, 1 ABSTAIN | 1.959 | 24 PASS, median 3.862 |
| 3.16.0 | 1.4.0 | 23 FAIL, 1 ABSTAIN | 1.959 | 24 PASS, median 3.862 |
| 3.17.0 | 1.3.0 | 23 FAIL, 1 ABSTAIN | 1.959 | 24 PASS, median 3.862 |
| 4.0.0 | 1.4.0 | 24 PASS | 3.915 | 24 PASS, median 3.862 |
| 4.0.1 | 1.4.0 | 24 PASS | 3.915 | 24 PASS, median 3.862 |

All six combinations reproduced the CartPole assignment controls on 100 starts. Doubling `masscart` was a no-op, `masspole` left stale constants, `length` was partly stale and `gravity` matched the fresh reference. The partly stale annotation comes from the adapter’s inspection of the update. Direct attribute mutation is evaluated against the supplied fresh-construction contract.

The Reacher post-step observation differed from the refreshed map in all six runs, with a maximum absolute discrepancy of 2.391e-6. This tests cache consistency. On 4.0.1 the published wheel and candidate agreed across the seven native reports, including the individual probes. The archived 0.1 results and their provenance remain available in the [evidence register](evidence/README.md).

The separate Acrobot worker comparison on Gymnasium 1.3.0 versus 1.4.0 gave exactly matching first steps. The book-versus-nips and mass 1.3 controls gave maximum first-step differences of 0.00147277 and 0.06094115. Each comparison used 24 starts and five recorded steps. The healthy scalar Euler and RK4 controls each passed all 100 starts. Float32 and declared-noise controls abstained as specified. The full historical release matrix, shipped contact-rich models and arbitrary simulator laws are outside this validation.

## Reproduce the checks

From the repository root, install the full test extras. The core package and the test suite run independently of the two research repositories. Supply the pinned paper checkouts for the two cross-checks.

```bash
python -m pip install -e '.[dev,algebra,mujoco]'
python -m pytest -q
python validation/check_fingerprints.py /path/to/dafd
OPENBLAS_NUM_THREADS=1 python validation/run_workflow_cases.py /path/to/algebraicRLtest /tmp/workflow-cases.json
python validation/run_controls.py --report /tmp/controls.json --summary /tmp/false-alarms.json
PYTHONPATH=src:validation /path/to/engine/bin/python validation/run_known_cases.py /tmp/native-cases.json
PYTHONPATH=src:validation python validation/run_release_cases.py \
  --python-a /path/to/gymnasium-1.3/bin/python --python-b /path/to/gymnasium-1.4/bin/python \
  --out /tmp/release-controls.json
```

The 105-test suite ran on Python 3.11.9, 3.12.14, 3.13.15 and 3.14.6. The 3.11 environment exercised NumPy 1.24.4, SymPy 1.12 and MuJoCo 3.2.7. Results include optional-dependency cases. Known-fault controls intentionally return diagnostic `FAIL` inside a passing software test.

## Distribution and publication

Build and check both distribution files from a clean checkout. The default build creates the wheel from the source distribution. Run the receipt check from the repository. Use `--dist dist` to verify the bytes of the original locally reviewed files. A separate build can have different archive timestamps.

```bash
python validation/verify_receipt.py
python -m build
python -m twine check --strict dist/*
```

The tag for this release is `v0.2.0a1`. When the maintainer pushes that tag, the release workflow runs the test matrix, checks the receipt and version, builds both distributions and exercises the wheel outside the checkout. It passes those artifacts to the existing `pypi` Trusted Publishing environment. The local release receipt records the reviewed checkout and local archive hashes. [Release workflow](.github/workflows/release.yml)
