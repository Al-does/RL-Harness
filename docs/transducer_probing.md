# Predictive-belief probes for active environments

## Target

The affine probe targets the predictive Bayesian belief over a finite
transducer's memory state. For a row-vector belief `b_t` and an action-outcome
operator

```text
K(y_t | a_t)[i, j] = P(y_t, s_{t+1}=j | a_t, s_t=i),
```

the update is

```text
b_{t+1} = b_t K(y_t | a_t) / (b_t K(y_t | a_t) 1).
```

This is Equation 8 of Rosas et al.,
[AI in a vat](https://arxiv.org/abs/2504.04608), transposed from the paper's
column-vector convention. The action-free update used by Shai et al.,
[Transformers represent belief state geometry in their residual stream](https://arxiv.org/abs/2405.15943),
is the singleton-action special case.

`analysis.probes.predictive_belief_update` accepts the joint substochastic
operator directly. It therefore supports Mealy transducers in which outcome
generation and state transition do not factor. Environment- and
experiment-specific code supplies the operator and defines what counts as the
outcome.

## Timing

Rosas et al. define `b_t` before processing `(a_t, y_t)`. MESS3 supports two
token timings, so its experiment adapter composes the same transition and
outcome factors in different chronological orders.

With delay one, decision `t` receives the outcome emitted from `s_t` before
the action moves the process to `s_{t+1}`. This is the standard I-O Moore
factorization:

```text
K(y_t | a_t) = diag(P(y_t | s_t)) U(a_t),
```

At episode reset, the target is the environment reset distribution before any
operator is applied.

With delay zero, a Gym step executes the action, moves to `s_{t+1}`, and then
emits the observation available at the next decision:

```text
K(y_{t+1} | a_t) = U(a_t) diag(P(y_{t+1} | s_{t+1})).
```

Under this action/post-outcome alignment the kernel is a structured Mealy
presentation; shifting the output index gives an equivalent I-O Moore view.
The reset target must first condition the reset distribution on the initially
visible token. `U(a_t)` is always the transition matrix actually executed.
An impossible action-outcome pair is an error rather than a zero-vector
target.

Equation 8 assumes that the policy chooses actions from observable history,
not from privileged hidden state. Otherwise the action itself provides
additional evidence about the state and this update is incomplete.

## What this target is not

The bidirectional mixed-state matrix (BDMSM) in Rosas et al. is a joint
posterior over initial and current endpoint states. Its predictive marginal
obeys the update above, but probing the full matrix would test a stronger and
different representation hypothesis. Likewise, a minimal epsilon-transducer
state is an equivalence class over controlled future distributions, not a
belief over a selected latent presentation. Both require separate target
adapters and metrics.

As in Shai et al., the implemented neural probe remains unconstrained affine
least squares over every simplex coordinate. Its predictions are not
softmax-constrained; simplex projection is display-only.
