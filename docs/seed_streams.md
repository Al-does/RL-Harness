# Reproducible random streams

Every run keeps one externally supplied root seed. The CLI, run context,
manifest, Tune trial config, and RLlib `.debugging(seed=...)` value continue to
record that root unchanged. Internal code derives independent NumPy
`SeedSequence` children instead of adding numeric offsets to it.

## Stream hierarchy

Experiment recipes name scientific children at their composition root:

- checkpoint probing: `probe_train`, `probe_test`, `greedy_evaluation`;
- passive validation: `training`, `probe_train`, `probe_test`;
- scrambled evaluation: one `paired_condition_evaluation` stream, deliberately
  reused for normal and scrambled conditions, then split into `probe_train`
  and `probe_test`;
- supervised training: `model_initialization`, `training_sampling`,
  `training_data`, and `minibatch_order`;
- action-lattice analysis: `closed_loop_rollouts` and `cell_subsampling`,
  both reused across lattice sides so action resolutions share trajectories.

Generic rollout collection splits each supplied workflow child into
`episode_seeds`, `action_spaces`, and `policy_sampling`. Environment and action
space seeds are then keyed by environment index and episode index. The HMM
environment splits each reset seed into `state`, `emission`, `presentation`,
and `episode_length`. Consequently, extra policy samples cannot move later
environment resets, and presentation scrambling or episode-length
randomization cannot move latent-state or emission streams.

Stochastic post-hoc Torch policies use a local `torch.Generator` seeded from
`policy_sampling`. CPU and CUDA sample on the inference device. When a PyTorch
build lacks local MPS generators, offline probing samples from a local CPU
generator and transfers only the sampled noise or action; it never advances
the global Torch RNG.

## Stable child identifiers

Purpose names map to explicit numeric spawn keys. Unlike
`SeedSequence.spawn()`, these keys do not depend on dictionary or call order.
Changing or reusing an existing key changes reproducible outputs and must be
treated as a result-compatibility change.

To add a stream safely:

1. add a new purpose name with a previously unused key in the owning
   workflow's stream mapping;
2. derive it with `named_seed_sequences()` or `child_seed_sequence()`;
3. pass the `SeedSequence` or a NumPy generator internally;
4. call `seed_sequence_to_int()` only at APIs, such as Gym or Torch seeding,
   that require a plain integer;
5. add a reproducibility test without renumbering existing keys.
