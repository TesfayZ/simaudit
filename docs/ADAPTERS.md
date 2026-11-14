# Adapting your simulator

`Adapter` is a callable contract, independent of Gymnasium or MuJoCo. Import it
from `simaudit`. All optional callbacks default to absent. Checks name missing
capabilities and return SKIPPED instead of inferring a result.

| Attribute | Contract |
|---|---|
| `step_fn(state, action, dt)` | Return internal next state; restore every hidden dependency before stepping |
| `native_step_fn(state, action, dt)` | Independent path to the same native configuration, bypassing adapter transforms |
| `distance_fn(a,b)` | Finite nonnegative scalar; default Euclidean; define angle/quaternion semantics yourself |
| `observe_fn(state,action,dt)` | Actual observation emitted after the specified step |
| `reference_observe_fn(state,action,dt)` | Reference observation for that same step, dtype retained |
| `assignment_factory(name,value)` | Return two new adapters: direct assignment and consistently constructed reference |
| `features` | Explicit `Feature(name,status,note,present)` inventory, not a global health claim |
| `modifications` | `Modification(name,before,after,reason)` entries for every change |
| `dtype` | Actual simulator arithmetic/readout precision, not a float64 cast of float32 data |
| `noise_floor` | Conservative absolute noise bound in the relevant distance units; not estimated automatically |
| `smooth` | True only with justified smoothness on tested intervals |
| `declared_order` | Positive formal integration order; required by order check |
| `order_blockers` | Unsupported features/preconditions for this check |
| `dt` | Native timestep, finite and positive |
| `versions`, `scope`, `limitations` | Engine/build/reference identities and honest assessment boundary |

For a mutable simulator, capture/reset the RNG, time, solver/warm-start state,
controllers, caches, forcing, model options, termination state and any other
next-step dependencies. Restore a complete snapshot in *each* step callback,
then set the requested state and action. Do not accept a partial state if hidden
dependencies can affect its future. If replay cannot be established, omit the
capability or raise `Unsupported(reason)`. A stateless deterministic function
needs no snapshot. Distinct instances in the reference path are useful controls.

State arrays must be finite numeric arrays. Actions are passed through (copied)
and can be scalars or numeric arrays. The current JSON CLI requires serializable
arrays/scalars, not arbitrary simulator objects. User factories may encode a
larger state vector to represent hidden dynamics. This alpha has no asynchronous,
remote, GPU-tensor or vector-environment protocol; wrap these explicitly if they
can satisfy deterministic array replay. It does not automatically detect all
noise sources, non-smoothness or hidden state. Self-tests check native one-step
and two-step equivalence and replay after an interleaved continuation on each
supplied probe/timestep; this finite test cannot prove a general restoration claim.

Declare real float32 precision even if your wrapper returns float64. At/below
precision or noise floors, checks abstain; dtype conversion cannot recover lost
information. Stochastic simulators without controlled reproducible RNG cannot
provide the deterministic replay needed here. Noise bounds are caller supplied
and must cover the chosen observable/metric. A declared noise source that blocks
self-test agreement yields ABSTAIN, not a simulator-fault allegation.

An unknown feature name is retained as unsupported. It blocks order assessment
unless explicitly absent. The recognised inventory names are in report.FEATURES.
Known absent features use presence=false and a note, not status=tested. You can
list unsupported features even when another independent capability can be tested.

Use `Blowup(reason)` for engine instability or automatic reset. Nonfinite outputs
are also detected. Unexpected exceptions are programming/configuration errors;
they are not silently turned into successful or inconclusive science results.

Assignment factories must log modifications on both returned adapters. The
library compares baseline, assigned and fresh outputs at each adapter's own
configured timestep, so timestep itself can be a parameter. It reports no-op
only when direct assignment is ineffective while the fresh reference changes.
With no observable reference excitation it abstains. The CartPole adapter adds
an explicit source-based annotation for partial length updates; this detailed
mechanism is not inferred from arbitrary simulator trajectories.

The default thresholds are documented diagnostic heuristics. Heterogeneous state
units require a sensible distance metric/scaling; choose that and your thresholds
before looking at outcomes. Retain unresolved probes instead of selecting only
those that produce favourable slopes.
