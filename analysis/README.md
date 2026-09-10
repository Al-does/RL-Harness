# Belief-geometry analysis

Use this toolkit to test whether sequential-agent representations expose Bayesian
beliefs, and whether simpler predictive or historical features explain the fit.
It measures accessibility, not causal use or a uniquely identified internal model.

## Workflow

1. **Establish the contract.** Read the recipe, run manifest, and saved evaluations.
   Record source revisions, checkpoint selection, observation/action/reward timing,
   reset prior, and the information actually available to the policy. Inspect task
   performance before choosing a checkpoint; disclose retrospective selection.
2. **Verify targets and alignment.** In evaluation mode, align representation sites
   and exact decision-time targets; record policy sampling mode. Validate filtering
   against public diagnostics.
   Keep complete histories through resets, then remove warmup rows. Use the actual
   encoder receptive field, not merely one layer's window. Delayed observations
   may distinguish a filtered source belief from a decision-time arrival belief.
3. **Separate fit and evaluation.** Use independent trajectories or context groups.
   Tune on training groups only; never split correlated timesteps to choose a
   regularizer. Retain float64 targets and raw predictions. Keep final confirmation
   data out of checkpoint, layer, alternative-model, and threshold selection.
4. **Fit the primary probe and controls.** Start with the training mean, current
   observable branch, exact next-outcome probabilities and their logs; record the
   convention for zero log-probabilities. For RL,
   policy probabilities are not environment predictions. Distinguish all-action
   predictions, separate marginals, and joint outcomes. Add short-history targets,
   actual initialization, label permutations, covariance-matched random features,
   and nuisance-matched real features as the question warrants.
5. **Test the informative distinctions.** Supply task-derived belief contrasts or
   compute a prediction-map null basis. Equal immediate marginals need not imply
   equal joint outcomes or future control consequences. Select alternative models
   using training histories and predictive similarity; reject trivial relabelings
   or affine equivalence on the observed support. Compare target-normalized errors,
   not raw errors across different belief clouds.
6. **Report behavior and geometry separately.** Show task trajectories from actual
   initialization when available. Label analytical optima, numerical bounds,
   privileged-state oracles, and simulated feasible controllers distinctly. Match
   information, horizon, policy mode, and sampling. Expected-value references are
   not hard caps on noisy empirical measurements.

## Tools

| Task | API |
|---|---|
| Restore a trusted native module without an Algorithm | `analysis.checkpoints.load_module_only` |
| Restore a portable module export | `analysis.checkpoints.load_portable_module` |
| Collect representations and public targets | `analysis.rollouts.collect_batched_rollout_data` with experiment adapters |
| Filter complete action/outcome histories or suffixes | `analysis.probes.transducer.filter_operator_histories` |
| Run grouped probes, baselines, contrasts, and nulls | `analysis.belief_geometry.evaluate_belief_geometry` |
| Derive belief contrasts invisible to a prediction map | `analysis.belief_geometry.prediction_null_basis` |
| Screen candidate belief models on training data | `analysis.belief_geometry.select_alternative_beliefs` |
| Fit or score an individual probe | `analysis.probes.controls.fit_grouped_affine`, `score_prediction` |
| Compare shared or different targets | `paired_comparison`, `paired_target_comparison` in `analysis.probes.controls` |
| Sample matched or covariance-preserving null features | `matched_feature_null`, `gaussian_feature_null` in that module |
| Compare raw belief clouds or plot task history | `analysis.plots.plot_belief_comparison`, `plot_learning_curve` |
| Inspect variance/subspaces | `analysis.probes.variance_geometry`, `regression_factor_geometry` |

`load_module_only` accepts a native module directory or an Algorithm checkpoint
containing the requested module. It never falls back to full Algorithm restoration.
Load only trusted checkpoints; verify downloaded artifacts through the configured
storage client without exposing credentials. Use full `load_algorithm` only when
Algorithm state is genuinely required.

## Minimal composition

The experiment prepares aligned arrays; the library does not infer their semantics.

```python
from analysis.belief_geometry import evaluate_belief_geometry, prediction_null_basis
from analysis.plots import plot_belief_comparison

basis = prediction_null_basis(prediction_map)
result = evaluate_belief_geometry(
    {"representation": train_activations},
    {"representation": test_activations},
    train_beliefs,
    test_beliefs,
    train_groups=train_trajectory_ids,
    test_groups=test_trajectory_ids,
    nuisance_features={"next_outcome": (train_predictions, test_predictions)},
    contrasts={f"null_{i}": basis[:, i] for i in range(basis.shape[1])},
    seed=seed,
)
figure = plot_belief_comparison(
    test_beliefs, result.predictions["representation"], state_labels=state_labels
)
figure.savefig(output_png)
```

Use `result.report` for JSON-safe metrics and fit diagnostics. Raw primary and
control predictions remain available in `predictions` and `baseline_predictions`.
Optional `initialization_features` must describe the same histories and target
rows, not merely another network's on-policy sample. `matched_keys` describe
observable nuisance cells; inspect donor coverage and fallback rates. Alternative
selection uses mean forward `KL(true || candidate)`: extra candidate support is
allowed within budget, while missing positive-probability outcomes is rejected.

`filter_operator_histories` consumes a selected substochastic operator per row,
with row-vector update `b_next = b @ K / sum(b @ K)`. Groups may be interleaved but
must contain consecutive steps beginning at reset. The adapter chooses reset
operators, action/outcome alignment, and any later predictive transition. Apply
warmup masks only after filtering or forming suffixes.

## Interpretation and customization

- Keep model restoration/extraction, environment semantics, candidate generators,
  task references, sampling budgets, and report layout in experiment adapters.
  Prefer existing public methods and diagnostics over private environment state.
- Report MSE, target variance, normalized error, R², contrast scores, and sample
  coverage. Constant-target R² is undefined. A high global score can hide a lost
  contrast; changing visitation can change both the target variance and the score.
- Bootstrap independent trajectories with fixed fitted probes, not timesteps.
  These intervals do not measure training-seed variation. Row shuffles and matched
  features are descriptive controls, not automatic permutation p-values.
  Shuffling inputs against original labels differs from refiltering shuffled
  histories; neither automatically requires zero R². Never invent uncertainty
  from aggregate means.
- High initialization scores weaken claims that training created the geometry.
  Predictive controls may explain it; high alternative-model scores weaken unique
  model identification. These controls do not prove absence or causal use of belief
  information. Cumulative explained variance is not a belief-dimension certificate.
- Simplex plots retain raw predictions, paired target colors, and shared limits.
  Inspect out-of-simplex rates. Native plots cover low-dimensional simplexes;
  larger targets need explicitly labeled projections or separately justified
  marginals. Supplied axes are reused but reflowed for annotations. Close returned
  figures when done. Curve values are already in their displayed units; the helper
  does not turn variable-length episode returns into success percentages.
- Historical report schemas can be summarized without loading models. New controls
  require aligned data or fresh checkpoint extraction. Never reconstruct missing
  measurements, confidence intervals, or scatter clouds from summary scores.

## Delegation brief

Give each agent the source refs and manifests, selected checkpoints and selection
rule, adapter entry point, target timing, representation locations, split/group
protocol, seeds, requested controls, and a disjoint output directory. Require a
return of compact metrics, figures, provenance, verification commands, and limits.
Parallelize independent runs or targets, not writes to shared files. Independently
check any surprising mathematical or causal claim before adopting a subagent's
interpretation. Do not start new training, paid jobs, or publication without scope.

Verify from the harness checkout with its own environment:
`uv run pytest -q tests/test_belief_geometry.py tests/test_probe_controls.py tests/test_analysis_plots.py tests/test_analysis.py`.
A personal experiment environment can invalidate packaging-isolation tests. See [probe metric details](probes/README.md) for the
existing lower-level conventions.
