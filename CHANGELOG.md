# Changelog

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
