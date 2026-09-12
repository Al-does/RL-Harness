# Interactive component belief geometry

Use `analysis.belief_geometry.evaluate_belief_geometry` to fit grouped affine
probes, then `analysis.simplex.build_simplex_run` and `write_simplex_viewer` to
export their held-out predictions. `geometry_metrics` scores full coordinates
and component posterior masses without projection.

Install the optional dependency with `uv sync --extra visualization` in the
harness, or depend on `rl-harness[visualization]` in an experiment project.
Generated HTML, JavaScript, data and Plotly are bundled together. Open
`index.html` directly or serve its directory; no CDN or server API is required.
Assets and this guide are included in wheels as well as editable installs.

## Division of responsibility

The experiment owns checkpoint discovery/verification, native model restoration,
rollout collection, exact filtering, extraction sites, independent fit/test
histories, controls, scientific budgets, and provenance. The shared library does
not infer token delays, actions, BOS positions, private latent state, or the
meaning of the target. Read [the analysis workflow](../README.md) first.

The viewer supports:

- Any number of named runs, checkpoints and representation sites.
- Any number of **three-state component blocks**; each gets a Bayesian and probe
  panel. Supply the full partition of state indices, including noncontiguous
  blocks. No implicit projection is applied to other component sizes:
  `build_simplex_run` rejects them. `geometry_metrics` supports arbitrary sizes.
- Different sequence lengths across runs, with equal-length complete histories
  within a run. The slider, play button and token buttons use supplied lengths.
- Optional state labels, token strings and experiment-owned position notes.
  Defaults use coordinate and position indices. Passive studies can omit notes;
  controlled studies can display correctly aligned preceding/next actions.
- Raw affine predictions, including negative values and masses outside [0,1].
  Colors blend by positive local coordinates and square-root mass **for color
  only**. Geometry is never clipped, renormalized, or rounded. Labels use four
  decimals; tiny posterior movement is naturally invisible on a unit scale.
- Shared axes and linked cameras across all current panels; arbitrary checkpoint
  labels are displayed as text. Marker shapes/colors cycle through four styles.

`build_simplex_run` requires finite targets and predictions with the same
`(episodes, positions, states)` shape. Coordinates must have identical meaning
and row order across checkpoints. Metrics score **all supplied rows**; selected
cloud rows and example episodes change display only. Row indices use C-order
flattening: `episode * positions + t`. Do not concatenate unrelated policies'
rollouts as if they were matched histories. Use separate runs instead.

Choose the primary representation before looking at held-out scores. For studies
requiring warmup, the caller must apply a consistent prefix mask to metric inputs
and disclose the retained position offset in labels/notes; this API does not
silently drop or score different rows. Preserve complete history while filtering
and extracting features, before applying that mask.

## Composition

Here is the data handoff for an experiment that already collected aligned
complete histories. `train_features` and `test_features` are dictionaries of
2D `(rows, activation_width)` arrays; each row order follows its target array.
All variables representing measurements below must come from actual collection,
not summary scores.

```python
from pathlib import Path

import numpy as np

from analysis.belief_geometry import evaluate_belief_geometry
from analysis.simplex import build_simplex_run, write_simplex_viewer

episodes, positions, states = test_beliefs.shape
train_episodes, train_positions, _ = train_beliefs.shape
result = evaluate_belief_geometry(
    train_features, test_features,
    train_beliefs.reshape(-1, states), test_beliefs.reshape(-1, states),
    train_groups=np.repeat(np.arange(train_episodes), train_positions),
    test_groups=np.repeat(np.arange(train_episodes, train_episodes + episodes), positions),
    nuisance_features=predictive_and_observable_controls,
    initialization_features=initialization_feature_pairs,
    contrasts=task_contrasts,
    seed=42,
)
predictions = {
    "Initialization": {
        site: result.baseline_predictions[f"initialization/{site}"].reshape(test_beliefs.shape)
        for site in test_features
    },
    "Final": {
        site: values.reshape(test_beliefs.shape)
        for site, values in result.predictions.items()
    },
}
run = build_simplex_run(
    name="Experiment name",
    description="Checkpoint identifiers, target timing and sampling description",
    targets=test_beliefs,
    predictions=predictions,
    components=component_state_indices,  # e.g. {"A": [0, 1, 2], "B": [3, 4, 5]}
    primary_site=primary_site,
    tokens=visible_token_labels,  # strings, shape (episodes, positions)
    position_notes=position_notes,  # strings, same shape; omit for passive data
    cloud_rows=np.sort(np.random.default_rng(43).choice(
        episodes * positions, min(6144, episodes * positions), replace=False,
    )),
    example_episodes=np.arange(min(8, episodes)),
)
write_simplex_viewer(
    Path("artifacts/simplex-viewer"), [run],
    description="Explain the filtering, split, controls and limitations here.",
    reports={"Probe battery": result.report, "Provenance": provenance},
)
```

`initialization_feature_pairs[site]` is `(train_array, test_array)` from the
actual initialization checkpoint on those same histories. If it is unavailable,
omit initialization from both the battery arguments and predictions, and record
the gap; do not substitute random weights. All checkpoints in a viewer run must
offer the same representation sites.

Reports are JSON-safe dictionaries; generated report filenames do not come from
experiment labels. `write_simplex_viewer` refuses to overwrite its output files
but can add the viewer to a directory containing experiment-owned raw data.
Save full held-out arrays separately if needed. Keep generated clouds/checkpoints
under ignored artifacts rather than committing them.

## Passive versus controlled targets

For a passive edge-emitting HMM, the observer updates with the token-labeled
operator: `b' = (b @ T_token) / sum(b @ T_token)`. The existing generic
`analysis.probes.predictive_belief_sequence` can consume these operators; it
returns the initial prior followed by each posterior. Its operator API also
handles actions when the caller supplies action/outcome-dependent operators.

Actions that only select a guess or reward do not change this target. Delays
still matter: a filtered source belief for a pending emission can differ from
the environment's arrival-state belief. A controlled transducer may additionally
need the preceding executed action's transition and a pending edge kernel.
These timing rules belong in the experiment adapter and must be checked against
public diagnostics. Never condition targets on unseen emissions or rewards.

## Verification

```bash
uv run --extra visualization pytest -q tests/test_simplex.py
node --test tests/test_simplex_viewer.cjs
node --check analysis/simplex_viewer/viewer.js
```

The Node suite executes trace construction and controls with DOM/Plotly mocks;
it does not verify browser rendering. Check camera linking, playback and every
run/site/checkpoint choice in the browser when changing rendering behavior.
