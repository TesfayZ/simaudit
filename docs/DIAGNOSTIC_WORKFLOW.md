# SimAudit diagnostic workflow

Use one explicit question per invocation, or request full coverage. A trusted Python input factory supplies typed replay, fit and log inputs. Preview performs capability inspection without calling step functions. The screen uses independent trajectory identifiers and matched condition declarations. Attribution accepts bounded affine coefficient families in a caller-specified polynomial dictionary, with optional SymPy quotient reduction. Many mass and forward-kinematics length parameters enter these families linearly. Nonlinear parameter families need a separate verified optimiser.

The predeclared validation controls use the following design. Use six independent trajectories per arm for synthetic screens at alpha 0.01, including healthy, injected, unresolved, dependent and mismatched-condition cases. Exercise model, order, mixed and exact-conservation replay. Use disjoint starts for named-mass fits, require absolute error below 0.001 on deterministic Acrobot mass injections, and reject ambiguous objectives or excessive validation residual. Check a Reacher length injection against a competing length and an unexplained structural perturbation. Validate routing through the Python and CLI entry points and build a local development distribution. Record fixture seeds and observed counts separately from the paper's archived evidence.

## What the user supplies

Start with a question and a target configuration. Default to a targeted request. Offer a full diagnostic as an explicit choice when the user wants to investigate across properties. Both modes use the same decision rules.

| Request | Inputs to request | First result and stopping point |
|---|---|---|
| Did assigning a parameter take effect? | Parameter, intended value, replay probes, direct assignment and fresh construction | Compare baseline, assigned and freshly constructed configurations. Stop after reporting whether the update took effect on these probes. |
| Did an upgrade change behaviour? | Two configured interpreters or variants, equivalent settings and shared probes | Run the release comparison. Report the first-step difference and later separation. A cause investigation is a further request. |
| Are the observations stale? | Replay probes, emitted observation and trusted refreshed observation map | Compare the two observation paths at the same step. Stop with the discrepancy and its precision floor. |
| Does the integrator attain its declared order? | Smooth replay, native reference, state metric, declared order, interval and probes | Run state-order refinement. This question can be answered without an energy function. |
| Why does passive energy drift? | Reference energy with fixed units and zero, declared order, smooth passive replay at adjustable timesteps, probes | Start with the DAFD energy fingerprint. Where supported, use state-order refinement on the same probes to interpret the pattern. |
| What value of a named parameter explains these dynamics? | One named parameter, bounds, energy family, fixed other parameters, passive replay and independent starts | Fit the parameter in the energy family and validate on held-out starts. The user can ask this directly even when the nominal value is unknown. |
| Which specified physical laws differ in my logs? | Named constraints, observation coordinates, independent healthy and candidate trajectories matched for sampling, integration and noise | Screen the laws. Return the names, residuals, floors and corrected test results. An attribution request adds the next stage. |
| Which parameter could explain a flagged law? | Screen evidence for that law and configuration, recovery dictionary, known constraints and competing physical templates | Recover the invariant and compare template fits. Report a candidate only if recovery, fit and separation support it. |

A full diagnostic inventories these branches, runs the ones supported by the supplied inputs, and records the missing input for every uncovered branch. A targeted report need only discuss the requested question and its prerequisites. A user who supplies only logs can still receive a screen. A simulator factory alone cannot supply a trusted energy, healthy logs or plausible parameter hypotheses.

## How the planner executes a request

1. **Resolve scope.** Record the question, simulator or dataset identity, configuration, operating regime, and desired depth (detect, localise a law, or attribute a cause). The CLI/API accepts explicit choices. For an ambiguous request such as “check the mass,” present the two concrete interpretations: verify a write to a known mass, or estimate an unknown effective mass.
2. **Preview the plan.** List applicable checks, conditional follow-ups, and precise missing inputs. Distinguish access to a live step, adjustable physical timestep, and supplied trajectory logs. Previewing a plan requires no simulation. Request only the missing inputs that could advance the selected question.
3. **Run each eligible branch.** Apply the current native-step and restoration guards before dependent replay measurements. An adapter-integrity failure blocks conclusions from that replay path. Independently supplied, matched logs can still be screened. A discrepancy in an observation map limits physical interpretation of screens using those same coordinates.
4. **Interpret compatible evidence.** Match the model configuration, reference, coordinates, probe group, actions, interval and precision conditions where relevant. Compare healthy and candidate runs only across the intended difference. Reuse refinement endpoints only when their provenance matches. Energy and order share measured endpoints within one workflow invocation and run their own native replay guards. The fit retains endpoint states and evaluates every candidate on those same transitions. It performs its own guarded replay for its declared training and held-out groups.
5. **Stop at the requested depth.** Return the direct answer, supporting check results, competing explanations, coverage and any next input that would resolve uncertainty. Preserve the six existing check statuses. Record out-of-scope and conditionally unnecessary branches in the plan separately from executed results. A full audit with missing branches must make its partial coverage visible even when every executed check passes.

## Decision rules for a dynamics investigation

| Evidence on compatible probes | Interpretation and follow-up |
|---|---|
| Energy shows a model pattern and state order meets the declaration | The trajectory converges at the expected order while the candidate dynamics disagree with the supplied energy. Verify the invariant, units and nominal parameters. Run assignment if a changed parameter and fresh factory are supplied. Fit a named parameter if estimation is in scope. |
| Energy suggests reduced order and state order is below declaration | Report evidence of a discretisation issue and the observed slopes. In a full audit, keep any independently supported law screen eligible. A dominant numerical error can mask a model error. |
| Energy and state order disagree, or model-pattern energy accompanies low state order | Report both observations and leave the cause unresolved or potentially mixed. Additional starts and a predeclared smaller interval can test sensitivity. |
| Energy is consistent, reaches its precision floor, or exceeds the large-rate guard | Report the specific outcome. An exactly energy-preserving update can give no usable fingerprint. A requested observation check or law screen can still detect a discrepancy. |
| A screen flags a law with a physical template | The law name is the localisation result. If attribution is in scope, remove known polynomial consequences, test invariant recovery, then compare eligible one-parameter hypotheses up to the allowed scale and offset. |
| A screen finds no resolved violation, or a flagged law has no template | End that attribution branch. Report the tested laws and detection floors, or the flagged law and missing template. |

For a **direct one-parameter fit**, keep the simulated endpoints fixed and vary the parameter in the reference energy. Compare candidate values on training trajectories and evaluate the chosen value on separate starts. Report the objective profile, bounds, fixed parameters, numerical resolution and held-out residuals. Flat or ambiguous profiles, pure energy rescalings, boundary-limited optima, and failed held-out checks can leave the value unresolved. A successful estimate answers a conditional question about that energy family. Naming a culprit among several parameters requires competing hypotheses and evidence that separates them.

For **log-based attribution**, preserve the paper's distinction between *a law failed*, *no supplied template explains it*, and *several templates explain it equally well*. In a full audit, predeclare the screened laws and correct across the entire family. If the planner chooses a follow-up from the data, label it exploratory or confirm it on independent trajectories.

## Implementation and validation order

| Milestone | Deliverable | Acceptance before moving on |
|---|---|---|
| 1. Request routing | Explicit request and capability descriptions, preview, executor around existing checks, and a versioned workflow record referring to individual reports | Table-driven routing tests cover each request, missing inputs, failed replay, incompatible evidence, and a logs-only request. Existing single-check APIs, JSON readers and exit behaviour still work. Preview and execution agree on the selected branches. |
| 2. Dynamics interpretation and named fit | Shared fixed-interval replay measurements, rules above, bounded one-parameter estimation and held-out evaluation | Healthy, model-only, order-only, mixed and exact-conservation controls. Recover injected Acrobot masses within a predeclared tolerance. Exercise no excitation, wrong template, gauge ambiguity and boundary optima. |
| 3. Named-law screening | Trajectory input format, named vanishing and conserved quantities, per-trajectory residuals, raw floors, one-sided permutation test and correction across laws | Reproduce selected Acrobot and Reacher law names. Measure false alarms on independent healthy runs. Exercise mismatched noise, dependent samples and insufficient attainable test resolution. |
| 4. Algebraic attribution | Optional polynomial dependency, removal of known constraints, invariant recovery and competing template fits | Reproduce selected single-parameter cases and exercise ambiguous parameters, failed recovery, missing templates and a non-parameter equation fault. |
| 5. End-to-end release candidate | Full and targeted examples, standalone distribution, workflow and method validation evidence tied to one revision | Test requests against fresh held-out controls. Record coverage, false alarms, abstentions, localisation accuracy and wrong attributions separately. Build and install the candidate wheel locally and reconstruct the examples from the distribution. Review these results before tagging GitHub, publishing to PyPI and archiving that release on Zenodo. |

Freeze fixtures, tolerances and expected interpretations before evaluating held-out cases. Reproducing archived labels tests implementation fidelity. Fresh end-to-end cases test whether the planner gives the user a justified answer.

## Method sources

The energy fingerprint, the mixed-error masking example, and the named-mass fit come from *Diagnosing Model and Discretisation Errors in Reinforcement Learning Simulators Using Invariants*. [DAFD paper and source](https://github.com/TesfayZ/dafd)

The independent-trajectory screen, invariant recovery and competing-template attribution come from *Diagnosing Faults in Reinforcement Learning Simulators with Canonical Polynomial Invariants*. [Method paper](https://arxiv.org/abs/2609.13194v2)

## Running a workflow

Run `simaudit plan QUESTION --inputs my_inputs:factory` to preview or
`simaudit diagnose QUESTION --inputs my_inputs:factory --json run.json` to
execute. The Python equivalents are `plan(question, inputs)` and
`diagnose(question, inputs)`. Each returns a workflow record. Its `report`
member uses the existing Report 1.0 schema. Branch result indices refer to
that report's results in order. Strict mode exits 1 for a failing check or
incomplete coverage. The default exit follows the single-check report
convention. Plans and checks use the input factory's declared configuration.
The factory is trusted Python code and should load inputs without stepping the
simulator when preview is desired.

The accepted questions are `assignment`, `release`, `observation`, `order`,
`energy`, `drift`, `fit`, `screen`, `attribution` and `full`. Select
`energy` for the fingerprint alone and `drift` for both refinements.
Supply a single adapter and probe group for a combined dynamics interpretation.
The log and release branches retain their own configurations and source records.

### Fitting a named parameter

For an energy of the form `A(state) + t * B(state)`, supply a callable that
returns A and B and fix the remaining physical parameters. Each independent
group maps a start ID to a pair of state and action sequences. Split by start
before computing probe transitions. One fixed interval is replayed using eight
substeps for each probe. Each start contributes its mean absolute defect rate, with equal weight across starts. This objective is piecewise linear in the parameter. Its breakpoints and bounds determine the minimum and the
parameter interval that meets the supplied rate tolerance. The resolution is
the largest acceptable interval width in parameter units. This interval
describes numerical fit tolerance and carries no statistical confidence level.

```python
from simaudit import EnergyFit, WorkflowInputs, diagnose

fit = EnergyFit(
    parameter="stiffness", bounds=(0.4, 2.0),
    components=lambda s: (s[1]**2 / 2, s[0]**2 / 2),
    train=training_groups, holdout=heldout_groups,
    max_rate=1e-6, resolution=0.001, fixed="Unit mass, oscillator units",
)
record = diagnose("fit", WorkflowInputs("oscillator", adapter=adapter, fit=fit))
```

### Screening logs and attributing a flagged law

Use a matrix of shape (samples, coordinates) for each independent trajectory
or pose group. Give the two arms disjoint group IDs. Declare sampling,
integration and noise conditions using JSON-compatible metadata. IDs and
metadata allow checks for explicit mismatches and duplicated trajectories.
Independence and comparability also depend on how the data were collected.

```python
from simaudit import Hypothesis, Law, LawStudy, LogArm, Recovery, WorkflowInputs

columns = ("x1", "x2", "y")
conditions = dict(sampling="independent pose groups", integration="kinematic", noise=0)
healthy = LogArm("reference", columns, healthy_groups, conditions, independent=True)
candidate = LogArm("test", columns, candidate_groups, conditions, independent=True)
# Specification y = 0.10*x1 + 0.11*x2.
law = Law("position", "vanishing", {(0,0,1): 1, (1,0,0): -0.10, (0,1,0): -0.11})
recovery = Recovery(columns, degree=1, hypotheses={
    "length1": Hypothesis(0.10, (0.04, 0.20), {(1,0,0): -1}),
    "length2": Hypothesis(0.11, (0.04, 0.20), {(0,1,0): -1}),
})
study = LawStudy(healthy, candidate, [law], {"position": recovery})
inputs = WorkflowInputs("two-link position", logs=study)
```

Each polynomial maps a tuple of nonnegative integer powers to a coefficient.
Use `conserved` for a law whose value is constant within each trajectory.
Declare the entire family of laws before running the screen. It uses an exact
one-sided permutation test across units, Bonferroni correction, and a raw floor
in generator units. It abstains if the smallest attainable corrected p-value reaches or exceeds
alpha or the number of permutations exceeds 200,000. Rejection requires a corrected p-value strictly below alpha. As in the reference implementation, sums within `1e-12 * max(1, abs(observed_sum))` of the observed candidate sum count as ties.

For each flagged law, `Recovery` specifies its coordinates, maximum degree,
known vanishing polynomials in those coordinates, and lag for conserved laws.
Each competing hypothesis gives the nominal value, bounds, and derivative of
the polynomial coefficients with respect to the parameter. The candidate law
equals the screened nominal law plus this derivative times parameter change.
The algebra extra computes a graded reverse lexicographic Gröbner basis of the known constraints. Recovery
uses standard monomials in the quotient and a column-normalised SVD. It removes
the constant offset for conserved quantities and fits up to coefficient scale.
The recovered direction needs a singular-value ratio below 0.1. Naming a
parameter requires relative fit error below 0.001, tenfold separation from
the runner-up and resolved deviation from nominal. The result is conditional
on the supplied template family and exploratory on the screened sample.

The installed example `simaudit.demo:inputs` supplies every branch.
Its release comparison uses the same interpreter twice as an equivalence
control. For a release audit supply the two installed interpreter paths and
factory names in `WorkflowInputs.release`. Set `logs_use_adapter_observations`
to False only when the logs are independent of the adapter observation path.
