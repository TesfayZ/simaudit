# Changelog

## 0.2.0a1

Changes since the published 0.1.0a2 alpha.

- Added `energy` and the `plan` and `diagnose` CLI commands. The new workflow record includes branch coverage, missing inputs and the original schema 1.0 report.
- Added passive energy fingerprints, combined energy and state refinement, and bounded affine parameter fitting with equal weight per independent start and a held-out residual check.
- Added named-law screening with the paper’s raw residual floor, numerical tie tolerance and strict Bonferroni decision. Optional SymPy attribution uses graded reverse lexicographic normal forms and competing single-parameter templates.
- Fixed observation comparisons when callbacks share output buffers or mutate actions. Energy exponents preserve exact power-of-two ratios at the classification boundary. Order checks require a feature inventory and workflows identify unresolved probes even when another probe detects a fault.
- Fixed the Reacher optional installation and replay on MuJoCo versions without `mj_copyData`. Explicit contact pairs now block the smooth autonomous XML adapter.
- Retained the four original checks, public report types and pytest integration. The release workflow now tests the package and exercises the publication wheel before upload.

## 0.1.0a2 - 2026-11-15

- Packaging metadata only: added PyPI keywords, classifiers, project URLs
  (Changelog, Documentation) and a README alpha/feedback note.
- Releases are now published from GitHub Actions via PyPI Trusted Publishing.

## 0.1.0a1 - 2026-11-14

First public alpha.

- Checks: parameter-assignment consistency, integrator-order convergence,
  release comparison across interpreters, and observation consistency.
- Six explicit statuses (PASS, FAIL, ABSTAIN, SKIPPED, PARTIAL, BLOWUP) and
  strict JSON reports with stated limitations.
- CLI, pytest fixture, Gymnasium CartPole/Reacher adapters, MuJoCo order adapter.
- Fixed before publication: `compare_releases` exposed the caller's whole
  site-packages to its worker interpreters when pip-installed; workers now see
  only the simaudit package.
- Validation includes MuJoCo 3.16.0, 3.17.0 and 4.0.1 (see VALIDATION.md).
