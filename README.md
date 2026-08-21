# RL Harness

Shared RLlib research library for reproducible experiment composition.

- Rapid, reviewable contribution from coding agents (package-level `AGENTS.md`
  files keep generic code generic).
- Provenance for every run (experiment-repo commit, library commit, seed,
  hardware, lockfile).
- Optional vast.ai tooling for cheap parallel GPU runs.

Personal experiment recipes do **not** live here. **Entry point for
researchers:** fork
[`rl-experiments`](https://github.com/Al-does/rl-experiments), clone your fork,
run `./scripts/bootstrap_local.sh` (clones this library beside it). See
[docs/multi_repo.md](docs/multi_repo.md).

## Setup (library development)

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12 or newer.

```bash
uv sync --group dev
source .venv/bin/activate
```

## Run an experiment

From your personal experiment repo (after `uv sync` there):

```bash
uv run rl-harness \
  experiments.mess3_belief_geometry_2026_07.reward_only.experiment \
  --smoke
```

The CLI imports a dotted module path; the experiment package must be installed
in that environment (the personal repo packages `experiments*`).

Runtime-only options include `--seed`, `--smoke`, `--resume-from`,
`--hardware-profile`, and output-directory overrides. Scientific
hyperparameters live in the recipe.

Each run writes compact records under the experiment leaf's
`results/<run-id>/` and large data under ignored `artifacts/<run-id>/`.
Optional Backblaze B2 upload can mirror `artifacts/` and record URIs in
`results/`; see [docs/artifact_storage.md](docs/artifact_storage.md).

## Architecture

- `harness/` — immutable runtime context, provenance, artifacts, hardware,
  direct-RLlib and Tune runners, and the CLI.
- `learners/` — reusable RLModules and on-device PyTorch components.
- `losses/` — reusable objective primitives and cooperative Learner mixins.
- `analysis/` — generic checkpoint, rollout, probe, metric, and plot tools.
- `envs/` — reusable Gymnasium environments and domain logic.
- `devops/` — remote execution and infrastructure mechanics.

Dependencies point from experiment repos into this library. Generic packages
never import named experiments.

### Phasic Policy Gradient

`PPGConfig` implements the single-network ``detach`` architecture from Phasic
Policy Gradient: PPO value gradients are detached from the shared encoder
during policy phases, then periodic auxiliary phases train a second value head
through that encoder while a frozen policy supplies the cloning KL target.

```python
from learners import PPGConfig
from learners.models import PPGAuxiliaryValueHead, TransformerModel

class PPGTransformer(PPGAuxiliaryValueHead, TransformerModel):
    pass

PPGConfig().training(
    policy_iterations_per_aux=32,
    aux_epochs=6,
    aux_minibatch_size=8192,
    aux_lr=3e-4,
    beta_clone=1.0,
    aux_value_loss_coeff=0.01,
    aux_true_value_loss_coeff=0.01,
)
```

The auxiliary value losses are raw half-MSE, so their coefficients are
reward-scale dependent. Keep `beta_clone` near the paper default of `1.0` and
tune the value coefficients so value and cloning gradients are comparable.
Each Learner snapshots its post-connector policy batches on CPU, preserving
fixed value targets while discarding the much larger raw recurrent episodes.
Auxiliary updates shuffle and transfer one processed policy batch at a time,
avoiding a full multi-phase replay buffer on the learner device. PPG currently
supports one local or remote Learner; multi-Learner DDP is rejected explicitly.

### PPO distributional value critics

RLlib 2.56 does not provide IQN for PPO. Compose the reusable value mixin with
an existing actor-critic model and select the matching Learner:

```python
class IQNTransformerModel(IQNValueMixin, TransformerModel):
    pass

PPOConfig().training(vf_loss_coeff=0.0).learners(
    learner_class=IQNPPOTorchLearner,
    learner_config_dict={
        "iqn_value/loss_coefficient": 0.5,
        "iqn_value/huber_kappa": 1.0,
    },
).rl_module(
    rl_module_spec=RLModuleSpec(
        module_class=IQNTransformerModel,
        model_config={
            **base_model_config,
            "iqn_value": {
                "train_quantiles": 32,
                "value_quantiles": 64,
                "n_cosines": 64,
            },
        },
    )
)
```

This is a distributional PPO value critic trained against sampled on-policy
lambda returns. It is not an IQN-DQN implementation.

For QR-DQN-style fixed quantiles, compose `QRValueMixin` instead and select
`QRPPOTorchLearner`. Configure the model with
`"qr_value": {"num_quantiles": 64}`, set
`"qr_value/loss_coefficient"` / `"qr_value/huber_kappa"` in
`learner_config_dict`, and keep `vf_loss_coeff=0.0`. This is likewise a PPO
critic option, not a replay-buffer QR-DQN algorithm. Fixed-quantile regression
learns quantile locations directly, so it has no C51-style `v_min`/`v_max`
support bounds. Scale `huber_kappa` to the task's return units. RLlib still
reports its scalar value-MSE diagnostic for the quantile mean, but
`vf_loss_coeff=0.0` removes that MSE from the training objective.

See [the harness overview](docs/generic_harness_overview.md) for design
guidance and [the refactor specification](docs/generic_harness_refactor.md)
for detailed boundaries.

For affine belief-probe reporting, sampling distributions, and MSE baseline
interpretation, see the
[probe package guide](analysis/probes/README.md).

## Contribute a reusable change

```bash
git checkout -b alex/my-change
# edit learners/, losses/, harness/, …
uv run pytest -q -m "not slow"
git push -u origin HEAD
gh pr create
```

Idiosyncratic science stays in the experiment repo until reuse proves an
abstraction worth promoting here.

## Included domains

Reusable finite-HMM mechanics and the Gymnasium environment live under
`envs/hmm/`. MESS3 supplies probability models and wrappers under `envs/mess3/`.
Concrete MESS3 study recipes live in `alex-rl-experiments`.

`envs/cassandra_machine/` implements Anthony Cassandra's canonical
four-component machine-maintenance POMDP with original observation symbols and
optional full-belief and component-marginal observations. See its package
README for the model semantics and source references.
