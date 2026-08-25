# Factored finite HMMs and representation diagnostics

## Model construction

`envs.hmm.compose_hmm_factors` turns ordinary `HMMModel` instances into one
ordinary `HMMModel`. The result therefore works with `HMMEnv` and every existing
task without a second environment runtime.

- Hidden states are the Cartesian product of factor states.
- Each factor emits a sub-token; `token_map` deterministically maps the
  sub-token tuple to the one token visible to the agent.
- The default map is one-to-one mixed-radix encoding. A custom map may merge
  tuples when the observation intentionally aliases factor outputs.
- With no couplings, the prior, transition, and unmerged emission arrays are
  exact Kronecker products.

RLlib recipes use the import-path factory:

```python
env_config = {
    "model": {
        "factory": "envs.hmm:factored_model",
        "kwargs": {
            "factors": [
                {
                    "factory": "envs.mess3.model:passive_model",
                    "kwargs": {"alpha": 0.85},
                },
                {
                    "factory": "envs.mess3.model:passive_model",
                    "kwargs": {"alpha": 0.85},
                },
            ],
        },
    },
    # Select an existing task, observations, diagnostics, and timing normally.
}
```

For two three-token factors, the default observed alphabet has nine tokens.
Token `3 * first_subtoken + second_subtoken` uniquely determines the pair, but
the policy sees only the one nine-way token.

## Directional coupling

`FactorCoupling(parent, child, transition_matrices, strength)` makes the
child's next-state dynamics depend on the parent's next state:

```text
P(s_child' | s_child, s_parent')
  = (1 - strength) * T_child[s_child, s_child']
    + strength * T_child_given_parent[s_parent', s_child, s_child']
```

The parent evolves under its own transition matrix. `strength=0` is exactly the
independent product model; `strength=1` uses the conditional child dynamics.
Factors must be listed in topological order, and each child currently supports
one parent. This deliberately represents a directed dynamic Bayesian network,
not arbitrary pairwise interaction.

This coupling is ordinary latent-state dependence. It is not automatically the
paper's stronger *conditional independence given observed tokens*. That
property concerns token-labelled GHMM operators. An experiment claiming it
must verify that each observed-token operator tensor-factorizes. The
independent product model satisfies the criterion.

## Insights from “Transformers learn factored representations”

The paper distinguishes two lossless belief geometries when the generator
preserves product states:

- a joint simplex of dimension `product(d_i) - 1`;
- a direct sum of factor belief spaces of dimension `sum(d_i - 1)`.

Its main empirical claims are:

1. Transformers linearly encode each factor's predictive vector.
2. Activation CEV compresses toward the direct-sum dimension, even when the
   residual stream can fit the joint geometry.
3. Factor information occupies approximately orthogonal subspaces.
4. Factoring appears early even for mildly indecomposable generators, then may
   give way to extra dimensions under sustained prediction-loss pressure.
5. The token embedding itself can discover the hidden sub-token decomposition.

CEV means **cumulative explained variance**. It measures effective dimension;
it does not identify which factor uses which direction and cannot establish
orthogonality by itself.

## Analysis protocol

`analysis.probes.factorization` implements the paper's reusable operations:

- `variance_geometry`: PCA spectrum, CEV curves, and effective dimensions;
- `regression_factor_subspaces`: one joint affine regression to concatenated
  factor predictive vectors, then one activation-space basis per factor;
- `subspace_overlap`: basis-invariant squared principal-angle overlap;
- `vary_one_subspace`: grouped mean-centering and PCA for controlled vary-one
  datasets;
- `dimension_additivity`: factor-dimension sum versus pooled-union dimension;
- `representation_dimension_predictions`: direct-sum and full-joint baselines.

Use several diagnostics together:

1. Establish held-out linear decodability of every factor belief.
2. Compare activation CEV with factored and joint dimension predictions.
3. Measure pairwise principal-angle overlap of factor subspaces.
4. When factors are independently controllable, confirm the result with
   vary-one datasets.
5. Compare initialization, log-spaced training checkpoints, and the final
   model.

Dimension additivity detects shared directions, but zero excess does not prove
orthogonality: two non-orthogonal subspaces may still have a trivial
intersection. Principal-angle overlap is the direct orthogonality test.

These analyses establish geometry and decodability, not causal policy use.
Interventions or task controls are required for a causal claim.
