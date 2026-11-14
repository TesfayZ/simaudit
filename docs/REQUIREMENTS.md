# Package requirements and implementation plan

Decision recorded 11 November 2026, before collecting package validation results.
This document resolves the implementation choices; ../VALIDATION.md reports what
actually ran.

## Users and outcome

Researchers changing simulator parameters, upgrading an engine, or writing an
environment need a reproducible, scoped diagnosis before training. The package
must work independently of any other repository.
The first release candidate is 0.1.0a1, with Python >=3.11, NumPy in the core,
and optional Gymnasium, MuJoCo and pytest dependencies. Publication is a separate
release activity; installation, tests, wheels and documentation are deliverables.

## Acceptance and scope

| Requirement | Implementation | Acceptance |
|---|---|---|
| A: parameter assignment | baseline/direct assignment/consistent fresh reference | CartPole masscart no-op, masspole stale, length partly stale, gravity consistent; healthy reference controls |
| B: numerical order | 1/2/4/8 subdivisions of a fixed interval, two adjacent slopes | Euler/RK4 known answers, misdeclared order, floor abstention, unstable/non-smooth rejection |
| C: release comparison | identical JSON inputs, separate supplied Python interpreters | first-step comparisons, actual versions, missing interpreter skip, no automatic installation |
| D: observation consistency | observed output versus refreshed reference | healthy and stale controls; Reacher scoped cache recomputation, dtype floor |
| Adapter integrity | native-step equivalence and interleaved restoration | failed self-test blocks measurements; configurations checked after changes |
| Reporting | six statuses, coverage, modifications, evidence, versions, limitations | strict round trip; unknown fields/versions and nonfinite numbers rejected |
| CLI | human summary plus JSON; default/strict exit status | FAIL exits 1; other nonpasses exit 0 normally/1 in strict mode; usage errors 2 |
| pytest | assertion fixture, separate status summary and optional JSON | nonpasses skip but keep original status; strict session fails |
| Distribution | src layout, wheel/sdist, no imports from other repositories | wheel installed into fresh environment runs documented examples |

The built-in assignment adapter supports scalar CartPole only. Its fresh reference
rebuilds documented dependent constants explicitly, since Gymnasium's constructor
does not accept physical parameters. This reference is a stated model contract,
not a claim that Gymnasium promises arbitrary attribute mutation. Vector adapters,
static attribute scanning and other classic-control factories are extension work.
Unprobed attributes are inventoried as present_not_tested. A no-op fails only when
the supplied reference changes on the same probes; otherwise it abstains.

The MuJoCo order adapter accepts smooth, autonomous rigid models without active
contacts, limits, actuators, dissipation, tendons, equality constraints, plugins,
flexes, mocap or callbacks. Such features block the order check with PARTIAL;
they are not silently disabled. The native engine step is measured. Quaternion
distance uses mj_differentiatePos. Reacher observation checks use shipped stepping
and compare the same observation map before/after mj_forward, so they diagnose
cache consistency, not independent correctness of the observation formula.

Release comparison detects compatibility changes, not necessarily bugs. Only
the first step controls its verdict; later separation is evidence, not a proof
of chaos. Both workers must pass their self-tests and use identical inputs and
configurations. Historical releases absent from the run remain untested.

## Architecture and extension contract

Checks consume adapters with pure replay step/native_step, a distance function,
capabilities, feature inventory, versions, scope, and a modification ledger.
Checks test replay after an interleaved different state and compare native steps
at every requested timestep. Reusable callable adapters support third-party
models; users supply the native reference and truthful inventory. No adapter can
prove its own inventory complete. Unknown inventory features are unsupported.
Known absent features carry explicit presence=false; they are not tested passes.

Reports are versioned dataclasses with strict field validation. Evidence is an
open JSON payload within a closed envelope; consumers must not interpret unknown
evidence keys as guarantees. Undefined estimates are null. No NaN/Infinity.
Every result retains raw probes, options, thresholds, limitations and provenance.
An empty report is not an all-clear. No global simulator objects, hidden writes,
downloads, training or background work occur on import.

## Reuse and credit

The two method sources are:
1. Tesfay Zemuy Gebrekidan, *Diagnosing Model and Discretisation Errors in
   Reinforcement Learning Simulators Using Invariants*, Zenodo 24114609.
2. Tesfay Zemuy Gebrekidan and Hadush Hailu Gebrerufael, *Diagnosing Faults in
   Reinforcement Learning Simulators with Canonical Polynomial Invariants*,
   arXiv:2609.13194v2.

dafd (https://github.com/TesfayZ/dafd): reuse the assignment comparison and refinement design, with explicit
provenance. Its scripts have global environments, pandas/SciPy, output paths and
research-specific classifiers: reimplement the small mechanisms behind adapters
instead of importing runners. algebraicRLtest (https://github.com/TesfayZ/algebraicRLtest): reuse the isolated-release and
observation-reference methodology, crediting both paper authors. Both repository
LICENSE files were inspected and are MIT. No complete source modules are vendored; the small heavy-top validation geometry
is adapted with its MIT notice and source hash retained.
No claims of fresh validation transfer from either source repository. THIRD_PARTY.md records inspected sources and hashes.

## Prospective package validation protocol

Seed 20261111; 100 independent analytic/CartPole starts where applicable. Assignment
states uniform [-0.1,0.1]^4, both actions; factor 2; absolute tolerance 1e-12.
Order controls use y'=y, h=0.2, one fixed macro interval, 1/2/4/8 subdivisions,
positive starts uniform [0.5,1.5]; absolute floor 1e-12 plus 50 eps times state
scale, slope stability tolerance 0.5 and order tolerance 0.5. The tested interval
is a user-specified diagnostic interval, not automatically the shipped timestep.
Observation tolerance is max(user absolute tolerance, 8 dtype eps times scale);
float32 agreement within its floor abstains. Release noise tolerance 1e-13.
Record failure, abstention and exclusion counts separately, with Wilson intervals
for false-alarm proportions. These small synthetic controls establish software
behaviour, not general false-alarm calibration. Do not tune against these results.

## Delivery order and release gates

1. Closed report contract and reusable adapter interface.
2. A plus CartPole and analytic golden controls.
3. B, C, D and restricted native adapters.
4. CLI, pytest integration, documentation, citation files and CI.
5. Tests, prospective control measurements, wheel isolation and evidence receipt.

Before a stable public release: expand independently selected environments,
measure held-out operational false alarms and abstention rates, validate current
and historical release pairs, check package-name availability and dependency
licences, obtain external usability feedback. Energy fingerprint, automatic
diagnostic fusion, parameter fitting and algebraic extras are later milestones.
