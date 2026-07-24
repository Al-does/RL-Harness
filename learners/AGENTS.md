# `learners/` — reusable RLModules and PyTorch components

## Layout

```
learners/
├── models/         RLModule subclasses (the neural nets RLlib calls).
├── components/     Reusable nn.Modules with no RLlib coupling (encoders, heads).
├── optimizer.py    Configurable torch optimizer factory + Learner mixin.
└── {algo}.py       Optional stable algorithm integrations when reuse warrants one.
```

Companion folder: **`losses/`** holds algorithm-agnostic Learner-side loss
mixins. See `losses/AGENTS.md` for their contract.

## The composition contract

The design goal for this folder is that **adding a new auxiliary head or
objective must not touch an unrelated base class or model**. An experiment
composes existing pieces in its own `experiment.py`:

```
experiment.py
  ├── model     : (BaseModel + HeadMixinA + HeadMixinB, model_config={...})
  ├── learner   : (BaseAlgoLearner + LossMixinA + LossMixinB, learner_config={...})
  └── AlgorithmConfig with namespaced hyperparameters
```

Rules that follow:

- Nothing in `learners/` may know MESS3-specific facts (alphabet size,
  observation layout, belief semantics, etc.). Dimensions and scientific
  choices come from experiment config; task adapters stay with their domain or
  experiment.
- No shared module-level constants leak experiment specifics
  (e.g. `AUX_LOGITS`, `N_AUX_CLASSES`). Each mixin owns its own namespaced
  strings.
- Loss math never lives in `learners/{algo}.py`. `learners/{algo}.py` contains
  only reusable algorithm integration or stable composed leaves.
- One-experiment leaf compositions belong in that experiment's
  `experiment.py`. Do not build every model × head × loss × algorithm
  combination in this package.

## RLModule side

Two kinds of building block:

- **`components/*`** — `nn.Module`s that own parameters (encoders, heads).
  Zero RLlib coupling; pure PyTorch. Reusable across models.
- **`models/*`** — `TorchRLModule` subclasses. A concrete model composes one
  encoder + head mixins. A base model provides ONLY policy + value heads.

### Head mixin contract

A head mixin (e.g. a `NextTokenAuxHead` for next-step classification):

1. Override `setup()`: call `super().setup()` FIRST, then build head parameters
   under an attribute name qualified by the mixin namespace
   (e.g. `self.next_token_aux_head`).
2. Override `_forward_train()`: call `super()._forward_train()` FIRST, then
   write outputs into the returned dict under a namespaced key
   (`out["next_token_aux/logits"] = ...`).
3. Read hyperparameters from `self.model_config` under a namespaced sub-dict:
   ```python
   cfg = self.model_config.get("next_token_aux", {})
   ```
4. Do NOT touch `_forward()` (rollout path) unless the head must run at
   inference. Aux heads typically only run in training.

Leaf model classes are the composition:

```python
class TransformerWithNextTokenAux(NextTokenAuxHead, TransformerBase):
    pass
```

`TransformerBase` should provide policy + value heads only. Aux heads are
added by mixin at the leaf, not baked in at the base.

Reusable compositions may live here when they have independent meaning.
Otherwise define the leaf in `experiment.py` so it remains importable to Ray
workers without creating a combinatorial shared API.

## Learner side

Reusable Learner leaves may live in `learners/{algo}.py` and read as:

```python
# learners/ppo.py
from losses.next_token import NextTokenAuxLossMixin
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner

class PPOWithNextTokenAux(NextTokenAuxLossMixin, PPOTorchLearner):
    pass
```

Mixins come BEFORE the base class. One-experiment combinations stay in
`experiment.py`. All loss-mixin contract details live in `losses/AGENTS.md`.

Primary objective replacements are algorithm integrations, not auxiliary loss
mixins. For example, PPO's implicit-quantile value option composes
`IQNValueMixin` with a compatible actor-critic model and selects
`IQNPPOTorchLearner`, while the recipe sets `vf_loss_coeff=0.0`. Its pure
quantile-Huber math remains in `losses/`; the PPO target and metric wiring live
in `learners/ppo_iqn.py`. A future IQN-DQN would require its own Q module,
target-network behavior, and Learner integration rather than reusing the PPO
leaf.

## Optimizer configuration

RLlib's default `TorchLearner` hardcodes `torch.optim.Adam`. To choose
another optimizer, compose `ConfigurableOptimizerMixin` from
`learners.optimizer` (exported as `learners.ConfigurableOptimizerMixin`) and
set namespaced keys on `learner_config_dict`:

```python
from learners import ConfigurableOptimizerMixin
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner

class ExperimentLearner(ConfigurableOptimizerMixin, PPOTorchLearner):
    pass

# ...
.learners(
    learner_class=ExperimentLearner,
    learner_config_dict={
        "optimizer/type": "adamw",  # adam | adamw | sgd | rmsprop | muon | class | factory
        "optimizer/kwargs": {"weight_decay": 0.01},
    },
)
.training(lr=3e-4, ...)
```

Notes:

- The mixin overrides `configure_optimizers_for_module` and does **not** call
  `super()` (calling it would register a second Adam). Put loss mixins first,
  then this mixin, then the algorithm Learner.
- Learning rate still comes from `.training(lr=...)` (fixed value or RLlib
  schedule) via `register_optimizer(..., lr_or_lr_schedule=config.lr)`.
- `optimizer/type="muon"` registers Muon for 2D weights and AdamW for non-2D
  params (biases, norms). Optional `optimizer/aux_kwargs` configures that AdamW
  group. `build_torch_optimizer(..., name_or_cls="muon")` only accepts 2D
  params; prefer the mixin for full modules.
- For other non-RLlib loops, call
  `build_torch_optimizer(params, name_or_cls=..., kwargs=...)`.
- Changing optimizer type mid-run can break checkpoint restore; keep the same
  optimizer family when resuming.

## Configuration flow

Two channels carry per-experiment hyperparameters:

- **Model side**: `RLModuleSpec(module_class=..., model_config={...})` becomes
  `self.model_config` inside every RLModule (including in mixins' `setup`).
- **Learner side**: `.training(learner_config_dict={...})` becomes
  `config.learner_config_dict` inside every Learner (including in loss mixins'
  `compute_loss_for_module`).

Both are plain dicts. Namespace each mixin's keys with a `<mixin>/<field>` or
sub-dict prefix so multiple mixins never collide:

```python
model_config = {
    "encoder": {"d_model": 96, "n_layers": 3, "n_heads": 4},
    "next_token_aux": {"num_classes": 3, "hidden": 128},
}
learner_config_dict = {
    "next_token_aux/lambda": 0.1,
    "reward_aux/lambda":     0.05,
}
```

The experiment writes these dictionaries directly into its fresh
`AlgorithmConfig`; there is no Blueprint schema.

## Component configuration

Typed dataclasses are useful when they validate one reusable component, such
as transformer dimensions. They are not a replacement experiment schema.

Expose broadly useful choices through component config when they naturally
belong to the component. For example, an MLP encoder may accept activation and
normalization choices. Do not add fields for one experiment's observation
meaning, result paths, or training phase.

## Representation access

Do not store probe activations in rollout or replay data by default. Analysis
loads a model and computes representations on demand. Initially, an experiment
supplies a small extraction callable using a model's natural API (for example,
`encode_step`). Do not add a generic named-representation protocol until
multiple incompatible models demonstrate the need.

## Verification expectations

Test reusable components with inline fixtures rather than named experiments.
When changing an RLlib extension point, cover the isolated composition and one
representative RLlib integration. Keep `Columns.EMBEDDINGS` and all forward
and loss tensors device-native and transient.
