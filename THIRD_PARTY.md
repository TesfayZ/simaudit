# Method provenance and reuse

The package uses two source papers, with their full authorship preserved:

1. **Tesfay Zemuy Gebrekidan.** *Diagnosing Model and Discretisation Errors in
   Reinforcement Learning Simulators Using Invariants* (2026).
   [Zenodo record](https://zenodo.org/records/24114609).
   Source repository: [`dafd`](https://github.com/TesfayZ/dafd). Assignment consistency, refinement diagnostics,
   native integrator controls and precision/coverage limitations inform this package.
2. **Tesfay Zemuy Gebrekidan and Hadush Hailu Gebrerufael.** *Diagnosing Faults in
   Reinforcement Learning Simulators with Canonical Polynomial Invariants* (2026).
   [arXiv:2609.13194v2](https://arxiv.org/abs/2609.13194v2).
   Source repository: [`algebraicRLtest`](https://github.com/TesfayZ/algebraicRLtest). Paired release comparisons, positive
   controls and observation consistency inform this package. Its algebraic
   screening/localisation method is not implemented in this alpha.

Both source repositories have MIT licences, copyright 2026 Tesfay Zemuy Gebrekidan.
That software licence notice does not replace the papers' scholarly authorship.
The MIT notice is retained in LICENSE. `validation/known_cases.py` adapts the
minimal heavy-top geometry from `dafd/benchmark_quaternion_mechanism.py`; its
collision masks are explicitly changed for the conservative test. The small
checks are new adapter-based implementations of the documented methods. No
research runner, paper prose, dataset, model or complete source module is vendored.
Source file hashes inspected during development appear in evidence/reuse_sources.json.

Runtime dependencies are imported, not bundled: NumPy (BSD-3-Clause), Gymnasium
(MIT, optional), MuJoCo (Apache-2.0, optional), pytest (MIT, optional). A stable
release still needs a full resolved dependency licence inventory, including
transitive dependencies. This list does not claim that audit has been completed.

Official interfaces consulted: [MuJoCo API](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html)
and [pytest plugin documentation](https://docs.pytest.org/en/stable/how-to/writing_plugins.html).
