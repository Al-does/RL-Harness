# `learners/` — RLModules and Learners composed from mixins

## Layout

```
learners/
├── models/         RLModule subclasses (the neural nets RLlib calls).
├── components/     Reusable nn.Modules with no RLlib coupling (encoders, heads).
└── {algo}.py       Composed leaf Learner classes (ppo.py, sac.py, ...).
```

Companion folder: **`losses/`** holds algorithm-agnostic Learner-side loss
mixins. See `losses/AGENTS.md` for their contract.

## The composition contract

The design goal for this folder is that **adding a new auxiliary head, probe,
or objective must not touch any base class or any existing model**. A new
experiment is one Blueprint that composes existing pieces:

```
Blueprint
  ├── model     : (BaseModel + HeadMixinA + HeadMixinB, model_config={...})
  ├── learner   : (BaseAlgoLearner + LossMixinA + LossMixinB, learner_config={...})
  └── config    : namespaced hyperparameters per mixin
```

Rules that follow:

- Nothing in `learners/` may know MESS3-specific facts (alphabet size,
  next-token semantics, etc.). Those live in mixins whose names encode the
  task, or in Blueprint config.
- No shared module-level constants leak experiment specifics
  (e.g. `AUX_LOGITS`, `N_AUX_CLASSES`). Each mixin owns its own namespaced
  strings.
- Loss math never lives in `learners/{algo}.py`. `learners/{algo}.py` contains
  only the composed leaf: `class X(LossMixinA, ..., BaseAlgoLearner): pass`.

## RLModule side

Two kinds of building block:

- **`components/*`** — `nn.Module`s that own parameters (encoders, heads).
  Zero RLlib coupling; pure PyTorch. Reusable across models.
- **`models/*`** — `TorchRLModule` subclasses. A concrete model composes one
  encoder + head mixins. A base model provides ONLY policy + value heads.

### Head mixin contract

A head mixin (e.g. a `NextTokenAuxHead` for MESS3 next-token prediction):

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

## Learner side

Leaf Learner classes live in `learners/{algo}.py` and read as:

```python
# learners/ppo.py
from losses.next_token import NextTokenAuxLossMixin
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner

class PPOWithNextTokenAux(NextTokenAuxLossMixin, PPOTorchLearner):
    pass
```

Mixins BEFORE the base class in the class list. All mixin contract details
live in `losses/AGENTS.md`.

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

The Blueprint schema should carry a **generic** `aux_config: dict` field (or
similar) that is forwarded verbatim into `learner_config_dict` — never one
first-class Blueprint field per aux loss.

## Known violations in the current codebase

The current layout is a partial abstraction: files were split into
`learners/{models,components,ppo.py}` but MESS3-specific details leaked into
what should be shared code. An agent editing this folder should recognise the
following anti-patterns and, where practical, refactor them into the pattern
above rather than extending them:

1. **Aux head baked into the shared base model.**
   `learners/models/base.py:24-28` — `BaseActorCriticModel.setup()`
   unconditionally builds `ActorCriticHeads(embedding_dim, action_space,
   auxiliary_dim)`. Every model based on this class carries an aux head whether
   the experiment wants one or not, and there is no path to add a *different*
   aux head. Correct: the base model owns policy + value only; aux heads are
   RLModule mixins composed at the leaf class.

2. **`AUX_LOGITS` / `N_AUX_CLASSES` as library-level constants.**
   `learners/models/base.py:16-17` — `AUX_LOGITS = "aux_logits"` is an
   unnamespaced shared string; `N_AUX_CLASSES = 3` is the MESS3 alphabet size
   hardcoded into shared code. Both leak experiment specifics into
   `learners/`. Correct: each head/loss mixin owns its own namespaced constants
   (`next_token_aux/logits`), and dimensional facts about a task come from
   config.

3. **Aux head unconditionally written into `fwd_out`.**
   `learners/models/base.py:53-55` — `_outputs()` always populates
   `AUX_LOGITS` at training time. Correct: only head mixins actually composed
   onto the model contribute to `fwd_out`.

4. **`_build_encoder` returns aux head dimension.**
   `learners/models/base.py:30`, `learners/models/mlp.py:40-45`,
   `learners/models/transformer.py:55-65` — the encoder-building method
   returns `(embedding_dim, auxiliary_dim)`, mixing encoder responsibilities
   with aux head responsibilities. Correct: `_build_encoder()` returns only
   the encoder's embedding dim; aux heads build themselves in a mixin's
   `setup()`.

5. **Model configs default to the MESS3 alphabet.**
   `learners/models/mlp.py:18` (`auxiliary_dim: int = N_AUX_CLASSES`),
   `learners/models/transformer.py:26` (same) — the model configs know about
   the MESS3 alphabet cardinality. Correct: model configs describe the
   encoder; aux head dimensions live in a namespaced sub-dict read by the
   corresponding head mixin.

6. **`ActorCriticHeads` requires `auxiliary_dim`.**
   `learners/components/heads.py:12-13,27,40-41` — the shared heads module
   requires an aux head at construction and exposes `auxiliary_logits(...)`.
   Correct: `components/heads.py` should offer only policy + value heads; aux
   heads are separate mixin-owned `nn.Module`s built in the mixin's `setup()`.

7. **Monolithic `AuxPPOTorchLearner`.**
   `learners/ppo.py:27-71` — one class both selects PPO and inlines the
   next-token loss math. Not stackable with a second aux loss, not reusable
   across SAC. Correct: math in `losses/next_token.py` as
   `NextTokenAuxLossMixin`; `learners/ppo.py` reduces to
   `class AuxPPOTorchLearner(NextTokenAuxLossMixin, PPOTorchLearner): pass`.

8. **Loss math in the algorithm file.**
   `learners/ppo.py:18-24` — `next_token_targets(...)` is task-specific and
   belongs in `losses/next_token.py` next to the mixin that consumes it.

9. **Learner imports model's aux constants.**
   `learners/ppo.py:15` — `from learners.models.base import AUX_LOGITS,
   N_AUX_CLASSES` couples the Learner to the model via a shared global.
   Correct: the loss mixin declares its own namespaced `FWD_KEY`; the paired
   head mixin writes it under the same string. No cross-file shared constant.

10. **Blueprint schema bakes in one specific aux loss weight.**
    `blueprints/base.py:80` — `aux_next_token_lambda: float = 0.0` is a
    first-class Blueprint field. Adding a second aux loss requires editing
    `Blueprint`. Correct: `Blueprint` carries a generic `aux_config: dict`
    field forwarded verbatim into `learner_config_dict`, and each mixin reads
    its own namespaced key from there.

11. **Unnamespaced `learner_config_dict` key.**
    `learners/ppo.py:40-41` and `scripts/train.py:100-102` — the key
    `aux_next_token_lambda` has no mixin namespace prefix. Rename to
    `next_token_aux/lambda` when refactoring.

If you are editing this folder, take any of the above as an invitation to
refactor the surrounding piece; do not extend the current pattern to a new
experiment.
