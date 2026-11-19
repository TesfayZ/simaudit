# Validation evidence

## 0.2.0a1

| File | Contents |
|---|---|
| [workflow_cases.json](workflow_cases.json) | Fresh Acrobot fits, clean and noisy law screens, attribution outcomes, fixture seeds, package and paper-source hashes |
| [native_controls.json](native_controls.json) | Six measured engine combinations, per-model verdict counts and median slopes, Reacher discrepancies and raw report hashes |
| [release_receipt.json](release_receipt.json) | SHA-256 of the reviewed checkout files and locally built wheel and source distribution |

The wheel and source distribution include both current measurement files. The remaining evidence is retained in the repository. The commands in [VALIDATION.md](../VALIDATION.md) rerun the measurements. From this checkout, `python validation/verify_receipt.py` checks the release files and the workflow’s recorded source hashes. Add `--dist dist` to require and check both original distribution files.

## Earlier alpha

| File | Contents |
|---|---|
| paper_cases_mj316.json, paper_cases_mj317.json, paper_cases_mj401.json | Original full CartPole, heavy-top and Reacher reports |
| paper_release_controls.json | Original isolated Acrobot release comparison and two injected controls |
| package_controls.json, false_alarm_controls.json | Scalar, precision and noise controls and the Wilson intervals |
| software_tests.xml, validation_receipt.json | Historical software test output and 0.1.0a1 source, distribution and evidence hashes |
| reuse_sources.json | Hashes of inspected source and licence files from the two method repositories |

Historical reports retain the original interpreter paths. The raw measurements contain known-fault verdicts and abstentions. [Historical validation](https://github.com/TesfayZ/simaudit/blob/v0.1.0a2/VALIDATION.md)
