# Method provenance and reuse

SimAudit builds on the following papers.

1. **Tesfay Zemuy Gebrekidan.** *Diagnosing Model and Discretisation Errors in
   Reinforcement Learning Simulators Using Invariants* (2026).
   [Paper](https://doi.org/10.5281/zenodo.24114609).
   Source repository: [`dafd`](https://github.com/TesfayZ/dafd). Assignment consistency, refinement diagnostics,
   native integrator controls and precision/coverage limitations inform this package.
   The energy check implements the classifier in Definition 1 and
   the accompanying algorithm (`dafd/fingerprint.py`). Its Acrobot regression
   cases compare fresh trajectories against the first trial of the healthy,
   LINK_MASS_2 +0.01 and CAT m2=1.3 cases in
   `Results/shipped_acrobot_fingerprint_trials.csv` (dafd revision `1e69c46`).
2. **Tesfay Zemuy Gebrekidan and Hadush Hailu Gebrerufael.** *Diagnosing Faults in
   Reinforcement Learning Simulators with Canonical Polynomial Invariants* (2026).
   [Paper](https://arxiv.org/abs/2609.13194v2).
   Source repository: [`algebraicRLtest`](https://github.com/TesfayZ/algebraicRLtest). Paired release comparisons, positive
   controls and observation consistency inform this package. The workflow adapts its
   trajectory-unit permutation screen and quotient-SVD/template attribution.
   Algebraic reference revision `687c868` supplies optional external validation
   fixtures. The installed checks use independent package implementations.

Both source repositories have MIT licences, copyright 2026 Tesfay Zemuy Gebrekidan.
That software licence notice does not replace the papers' scholarly authorship.
The MIT notice is retained in LICENSE. `validation/known_cases.py` adapts the
minimal heavy-top geometry from `dafd/benchmark_quaternion_mechanism.py`; its
collision masks are explicitly changed for the conservative test. The small
checks are new adapter-based implementations of the documented methods. No
research runner, paper prose, dataset, model or complete source module is vendored.
Source file hashes inspected during development appear in evidence/reuse_sources.json.

Runtime dependencies are imported, not bundled: NumPy (BSD-3-Clause), Gymnasium
(MIT, optional), MuJoCo (Apache-2.0, optional), SymPy (BSD-3-Clause, optional), pytest (MIT, optional). A stable
release still needs a full resolved dependency licence inventory, including
transitive dependencies. This list does not claim that audit has been completed.

Official interfaces consulted: [MuJoCo API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html)
and [pytest plugin documentation](https://docs.pytest.org/en/stable/how-to/writing_plugins.html).
