# Known-issue validation protocol

Recorded before running the package against paper cases, 11 November 2026.
The earlier package-control protocol in docs/REQUIREMENTS.md remains unchanged.

Use seed 20261111. CartPole: 100 uniform [-.1,.1]^4 starts, alternating actions,
parameter factor two, tolerance 1e-12, both installed Gymnasium versions.
MuJoCo: the dafd heavy-top ball-joint and three-hinge geometries, 24 starts with
qpos at model defaults and velocities uniform [-3,3], interval .05 seconds,
1/2/4/8 substeps. Compile collision-free geometry (contype=conaffinity=0),
matching the paper's disabled-contact scope, and record this in the model and
adapter modification ledger. Compare installed engines 3.16.0 and 3.17.0.
Do not infer a result for corrected 4.0.0 when it is not installed.
Reacher: 24 reset states with seed 20261111+i, replace joint velocities with
uniform [-1,1] on the first two joints, use uniform [-.5,.5] actions. Tolerance
1e-12 with dtype-aware floor. This tests the paper's cache-consistency mechanism,
not its full historical release matrix or exact aggregate residual.
Acrobot release controls: 24 uniform [-.2,.2]^4 starts, cycling actions 0/1/2,
5 steps, first-step threshold 1e-13. Compare unchanged factories across installed
Gymnasium versions, then book versus nips and mass2=1 versus 1.3 on the same
version. These reproduce the two papers' detection controls, not the deferred
algebraic localisation algorithm. Different factories are explicit in reports.

Expected outcomes: CartPole masscart no-op, masspole stale, length partly stale,
gravity consistent; ball joint below declared RK4 order, three hinges consistent
or explicitly unresolved; observation cache discrepancy detectable where above
floor; unchanged release pair agrees, injected dynamics controls differ.
All failures and abstentions are retained. A disagreement is reported rather than
changing thresholds. SHA256 manifests bind package sources, protocol and outputs.
