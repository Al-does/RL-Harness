# Evaluating affine belief probes with MSE

## Recommendation

Use mean squared error (MSE) as the primary affine-probe metric. It matches the
units and presentation used by Shai et al.,
[Transformers represent belief state geometry in their residual stream](https://arxiv.org/abs/2405.15943),
and makes checkpoint curves easy to read: lower is better and zero is exact.

MSE is not meaningful without its baseline and sampling distribution. Every
probe result should therefore record:

```text
mse
target_variance
global_mse_ratio
branch_baseline_mse
fine_mse_ratio
fine_mse_improvement
n_evaluated
sampling_distribution
```

`analysis.probes.global_mse_metrics` and
`analysis.probes.conditional_mse_metrics` compute these quantities. Existing
R² helpers remain available for compatibility.

## Global metrics

For target beliefs `b_i` and affine-probe predictions `b_hat_i`,

```text
mse = mean_i,coordinate (b_hat_i - b_i)^2
target_variance = mean_i,coordinate (b_i - mean(b))^2
global_mse_ratio = mse / target_variance
```

The target variance is the MSE of predicting every belief by the global mean.
Consequently:

```text
global R² = 1 - global_mse_ratio
```

Interpret `global_mse_ratio` as follows:

- `0`: exact decoding;
- `1`: no better than the global mean belief;
- greater than `1`: worse than the global mean.

Raw MSE preserves belief-coordinate units. The ratio records where that error
sits relative to the variance available in this evaluation dataset.

## Fine metrics

Global error can be dominated by an observable branch such as the current
token or the two most recent tokens. An experiment supplies one group key per
sample. The conditional helper computes each group's empirical target
centroid and asks how the probe compares with predicting that centroid:

```text
branch_baseline_mse = mean_i,coordinate (b_i - mean(b | group_i))^2
fine_mse_ratio = fine_evaluation_mse / branch_baseline_mse
fine_mse_improvement = branch_baseline_mse - fine_evaluation_mse
```

Interpret the fine ratio as follows:

- `0`: exact decoding within every branch;
- `1`: matches the branch-centroid baseline;
- greater than `1`: worse than the branch-centroid baseline.

The relation to the existing metric is:

```text
conditional residual R² = 1 - fine_mse_ratio
```

There is no distinct literal residualized "fine MSE" under this definition.
Subtracting the same branch centroid from a prediction and its target leaves
their difference unchanged. `fine_evaluation_mse` is therefore ordinary probe
MSE restricted to groups meeting `min_group_size`. The branch baseline, ratio,
and improvement give that MSE its fine-grained interpretation.

The helper uses evaluation-set target centroids to decompose evaluation
variance, just as conditional residual R² does. It is an analysis baseline,
not a deployable predictor. Experiments that need a deployable branch-only
model should fit branch centroids on training data and score them separately
on test data.

## Two sampling distributions

Report both distributions when comparison with exhaustive-context work and
task operation are important. Never combine them into one unnamed score.

### Uniform contexts

Uniform-context evaluation gives every fixed-length token string equal weight.
This matches the core context enumeration used by Shai et al. for finite
alphabets.

```python
from analysis.contexts import iter_discrete_context_batches

for contexts in iter_discrete_context_batches(
    n_symbols=3,
    context_length=10,
    batch_size=4096,
):
    # The experiment adapter computes exact targets and model activations.
    ...
```

If activations and targets are collected at every position, record that
choice. Flattening all positions weights history lengths 1 through
`context_length` equally; evaluating only the terminal position defines a
different distribution. Split whole contexts between probe fit and evaluation
rather than splitting their individual positions.

Enumeration grows exponentially as `n_symbols ** context_length`. The iterator
batches memory use but cannot remove that computational cost.

Suggested label:

```text
sampling_distribution = "uniform_contexts"
```

### Process-weighted rollouts

`analysis.rollouts.collect_rollout_data` and
`analysis.rollouts.collect_batched_rollout_data` sample histories according to
the environment and policy. Common histories receive more weight than rare
ones. Warmup usually removes short reset transients, so this distribution
describes mature beliefs encountered during task operation.

Use independent seed streams for fitting and evaluation, and record policy
mode, warmup, environment count, and step count.

Suggested label:

```text
sampling_distribution = "process_weighted_rollout"
```

## Why raw MSE differs between the paths

The same decoder can have different MSE under the two paths. Uniform contexts
weight every string equally, including strings the process almost never
generates. Rollouts weight an error by the probability of encountering its
history. The target variance and branch-baseline MSE also change with those
weights.

Compare raw MSE only when target definition, model representation, context
position, and sampling distribution are fixed. Across distributions or HMM
parameters, use the recorded baselines and ratios to interpret the absolute
errors; do not treat equal MSE as equal representation quality.

## Recommended workflow

1. Fix the target timing and representation hook.
2. Construct disjoint fit and evaluation samples.
3. Fit one affine probe on fit samples only.
4. Evaluate MSE on held-out samples.
5. Compute global metrics and experiment-defined fine branch metrics.
6. Repeat with an untrained model using the same architecture and samples.
7. Measure simple observation-history baselines separately.
8. Repeat under both sampling distributions when paper comparability matters.
9. Keep the evaluation definition fixed across checkpoint curves.

Example reporting:

```python
from analysis.probes import (
    conditional_mse_metrics,
    global_mse_metrics,
)

metrics = {
    **global_mse_metrics(predicted, target),
    **conditional_mse_metrics(
        predicted,
        target,
        branch_keys,
        min_group_size=50,
    ),
    "sampling_distribution": "process_weighted_rollout",
}
```

A useful table includes absolute MSE, branch-baseline MSE, fine MSE ratio,
untrained MSE, and the sample-path label. Geometry plots and task performance
remain separate evidence: low affine-probe error establishes decodability, not
that the policy causally uses the decoded belief.
