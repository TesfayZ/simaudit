# Report contract 1.0

`Report.results` is a list of `Result`; `schema_version` is exactly `1.0`.
`Report.loads` rejects unknown envelope/result/coverage/modification fields,
unsupported versions, duplicate keys and nonfinite numbers. Every result has
check, status, reason, scope, limitations, coverage, modifications, versions,
evidence and optional fault_class. Evidence is an open JSON object: check-specific
raw measurements, inputs and parameters live here. Unknown evidence keys are
preserved, not silently interpreted as guarantees. Undefined slopes are null.

Coverage entries have name, status, note and present (true/false/null). Status is
one of tested, disabled_for_test, present_not_tested, unsupported. Unknown feature
names are retained as unsupported. Modification entries have name, before, after
and reason. Reports must have nonempty scope, limitations and coverage. JSON
contains all six statuses and raw per-probe outcomes; summaries never call a
skipped or unresolved result a pass.

Report writers do not suppress coverage. The CLI writes only requested files;
it does not create missing parent directories or overwrite research evidence
unless that exact output path is supplied. Choose a fresh path for a new run.
User provenance can be supplied in adapter.versions, including source/model
hashes. Built-in adapters include relevant package versions and XML hashes.
Validation receipts additionally hash package code, protocols and output files.

CLI: exit 0 for no FAIL in normal mode, 1 for FAIL or any nonpass in strict mode,
2 for invalid configuration. An empty report exits 1. BLOWUP is a non-assessment,
not a successful run; use strict mode for automated acceptance.

Release workers have their own nested full reports. Actual worker versions,
input hash, interpreter paths, factories, trajectories and first-step differences
are retained. Exact agreement is a statement about recorded probes, not all
possible inputs or complete release equivalence.
