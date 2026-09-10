---
name: belief-geometry
triggers: [user, model]
description: Analyze sequential-agent belief geometry with held-out probes, predictive controls, and calibrated task-performance references.
---

Read the installed harness's `analysis/README.md` first; locate it through the
project Python environment rather than assuming a checkout path.

1. Verify source refs, run status, checkpoint choice, task units, and available
   information. Keep task performance separate from probe fit.
2. Build a small experiment adapter for representations and exact decision-time
   targets. Check action/observation timing, resets, and the full receptive field.
3. Preserve complete histories, then mask warmup. Split independent trajectories;
   tune on training groups only. Keep targets and predictions unprojected.
4. Use `evaluate_belief_geometry` with relevant predictive/history baselines,
   initialization, contrasts from the correct prediction map, and null controls.
   Select alternative models on training histories, not test fit.
5. Report MSE, variance, R², informative contrasts, coverage, and group-bootstrap
   uncertainty. High initialization or predictive-control scores qualify geometry
   claims; decodability is neither unique model identification nor causal use.
6. Plot actual checkpoint history and raw target/decoded beliefs. Label simulated
   references versus certified bounds; never hide empirical overshoots or invent
   missing measurements. Save compact outputs with provenance and verification.

For delegation, supply exact refs, adapter, selection/sampling protocol, seeds,
controls, and disjoint output paths. Require metrics, figures, checks, and caveats.
Do not launch training, paid compute, or publication unless explicitly requested.
