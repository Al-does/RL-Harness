# Evaluating affine belief probes with MSE

## Recommendation

Use mean squared error (MSE) as the primary affine-probe metric. It matches the
units and presentation used by Shai et al.,
[Transformers represent belief state geometry in their residual stream](https://arxiv.org/abs/2405.15943),
and makes checkpoint curves easy to read: lower is better and zero is exact.

For transformer probes, use the post-final-LayerNorm embedding returned by
`TransformerModel.encode_step` or `TransformerModel.encode_chunks`. Shai et
al.'s main result used the pre-final-LayerNorm `blocks.3.hook_resid_post`, but
Supplementary Figure S1 reported slightly lower trained-model MSE after the
final LayerNorm (approximately `0.0003` post-LN versus `0.0004` pre-LN).
`encode_step_pre_final_norm` and `encode_chunks_pre_final_norm` retain the
paper-main location as a separately named robustness control.

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
different distribution. Shai et al.'s 20%/80% control split whole length-10
sequences before flattening positions, but shorter prefixes consequently
appeared in both partitions under different suffixes. For a stronger
generalization test, deduplicate causal prefixes and use
`analysis.probes.split_group_indices` to keep each complete context or
trajectory cluster wholly in one partition.

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

## Required protocol and differences from Shai et al.

### 1. Independent fit and evaluation data

Use independent process seed streams for rollout probes. For exhaustive
contexts, split by unique causal context or another complete dependency group
with `split_group_indices`; do not randomly split correlated timesteps.

Shai et al.'s headline regression was fit and scored in-sample. Their repeated
20%/80% control demonstrated robustness, but shared short prefixes remained
across its sequence-level partitions. Held-out MSE is the primary result here;
an in-sample number is diagnostic only.

### 2. Cluster-bootstrap evaluation uncertainty

Bootstrap episodes, environment seeds, or complete contexts—not individual
correlated timesteps:

```python
from analysis.probes import (
    cluster_bootstrap_statistics,
    mean_squared_error,
    percentile_interval,
)

estimates = cluster_bootstrap_statistics(
    episode_ids,
    lambda indices: mean_squared_error(
        predicted[indices],
        target[indices],
    ),
    n_resamples=1_000,
    seed=42,
)
interval_95 = percentile_interval(estimates)
```

Keep the fitted probe fixed when estimating evaluation-sample uncertainty.
Refit inside the callback only when explicitly estimating probe-fit
uncertainty. Shai et al. did not bootstrap: they repeated random holdouts and
permutations. Bootstrapping improves uncertainty estimates, not the decoder.

### 3. Separate trained-model seed variation

Report the mean, standard deviation, and individual values across independently
initialized and trained models separately from bootstrap intervals. Bootstrap
replicates reuse one trained model and cannot estimate training-seed
variability. Shai et al.'s MESS3 checkpoint curve used one model seed.

### 4. Held-out permutation null

Permute complete training target rows, refit the probe, and score against true
held-out targets:

```python
from analysis.probes import held_out_permutation_null

null_mse = held_out_permutation_null(
    train_target,
    fit_predict_on_test,
    test_target,
    n_permutations=1_000,
    seed=42,
)
```

This preserves the marginal belief cloud while destroying its association
with activations. Shai et al. declared 1,000 label permutations, but fit their
shuffle control on the full dataset. The held-out variant better measures
generalization under the null. It validates the result; it does not improve
the real probe.

### 5. True initialization and log-spaced checkpoints

Probe a checkpoint saved after module initialization and before the first
optimizer step, then iterations `1, 2, 4, 8, ...` and the final checkpoint.
Keep target, representation, sampling distribution, fit budget, and evaluation
budget fixed along the curve. See `docs/checkpoint_strategy.md`.

Shai et al. showed multiple training checkpoints, which is an important
control, but the earliest published MESS3 checkpoint had already trained; it
was not a true step-zero reference.

### 6. Post-final-LayerNorm default

Use `encode_step` for rollout probes and `encode_chunks` for context batches.
These return the same post-final-LayerNorm embedding consumed by policy and
value heads. Shai et al.'s primary figures used the final-block residual before
the final LayerNorm and unembedding, but Supplementary Figure S1 found that
the post-LN representation was marginally more accurate (`~0.0003` MSE versus
`~0.0004` pre-LN) while preserving the same qualitative geometry.

Record the representation as `post_final_layer_norm`. Use
`encode_step_pre_final_norm` or `encode_chunks_pre_final_norm` only for an
explicit `pre_final_layer_norm` robustness control or exact reproduction of
the paper's primary hook. Do not select between them using final test MSE.

### 7. Optional concatenation across layers

Do not concatenate layers by default. Use it when the scientific hypothesis is
that predictive-state information is distributed across depth—for example,
when distinct beliefs have identical next-token predictions, every
single-layer held-out probe is weak, and a preregistered concatenated probe
improves consistently across model seeds.

Report every included layer, the resulting feature width, single-layer
baselines, and a capacity-matched null. Concatenation is a higher-capacity,
different probe specification. Shai et al. needed it for RRXOR but not MESS3;
PCA and center-of-mass points in that analysis were visualization aids, not
probe-accuracy enhancements.

### 8. Exact float64 targets

Keep analytic beliefs and transducer updates in float64 through fitting and
metric computation. Do not round, clip, or simplex-normalize probe targets or
predictions before scoring. Projection is display-only.

Shai et al.'s notebook rounded beliefs to five decimals while constructing
targets. That is unnecessary target coarsening and should not be copied.

### 9. Stable least squares and training-only regularization

`fit_affine_probe` uses centered, SVD-backed least squares. Ridge penalizes
feature weights but not the intercept, avoiding the condition-number squaring
and intercept shrinkage of normal equations.

Shai et al. used unregularized sklearn OLS. Pass `ridge=0.0` for that exact
probe. If ridge is needed for a smaller or collinear fit set, predeclare it or
select it with nested validation using fit data only. Never choose ridge,
activation location, layers, or checkpoints by final test MSE.

## Reporting example

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
untrained MSE, confidence interval, model-seed values, permutation-null
quantiles, representation location, checkpoint step, and sample-path label.
Geometry plots and task performance remain separate evidence: low affine-probe
error establishes decodability, not that the policy causally uses the decoded
belief.
