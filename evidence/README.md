# Evidence register

The alpha now has freshly executed package validation. See ../VALIDATION.md and
the prospective protocols in ../docs/REQUIREMENTS.md and ../validation/PROTOCOL.md.

| Artifact | Contents |
|---|---|
| paper_cases_mj316.json | Gymnasium 1.4.0 / MuJoCo 3.16.0: assignment, heavy-top order, observation |
| paper_cases_mj317.json | Gymnasium 1.3.0 / MuJoCo 3.17.0: same declared controls |
| paper_cases_mj401.json | Gymnasium 1.4.0 / MuJoCo 4.0.1, Python 3.12.3 (run 14 Nov): same controls |
| paper_release_controls.json | Isolated Acrobot release pair and two injected paper controls |
| package_controls.json | Generic NumPy-only healthy, float32 and noise controls |
| false_alarm_controls.json | Counts and Wilson intervals for the two healthy scalar controls |
| reuse_sources.json | Hashes of inspected method/reuse sources |
| validation_receipt.json | Final package/protocol/output hashes and execution checks |

Raw states, actions, thresholds, versions, modifications and unresolved probes
are preserved in each full report. The reports deliberately include FAIL when a
known discrepancy is detected; this is not a failed package validation.

Remaining design-plan rows are untested here: vector CartPole, MuJoCo 3.2.7,
3.10.0 and 4.0.0 (4.0.1 only was tested); the full historical release matrix; InvertedDoublePendulum
amplification; Ant/Humanoid conservative audits; the corrected development engine;
energy fingerprint and algebraic screening/localisation. None inherit PASS from
the parent projects. No first-fixed release is inferred from gaps in this matrix.

The result files and receipt live in the GitHub repository and its Zenodo archive; they are
not included in the PyPI source distribution. Run `python validation/verify_receipt.py`
from a repository checkout (distribution hashes are skipped if `dist/` is absent).

The archived result files are unedited records of the original runs, so they contain
the original machine's interpreter paths. `reuse_sources.json` keys are paths relative
to the author's workspace, where the `dafd` and `algebraicRLtest` repositories (see
../THIRD_PARTY.md) were checked out beside this one when their source hashes were taken.
