# `losses/` — Learner-side loss mixins

This folder holds **composable loss extensions** for RLlib Learners. Each file
defines ONE mixin that adds ONE auxiliary term to `compute_loss_for_module`.

Mixins here are **algorithm-agnostic**: they compose onto any RLlib base
Learner (`PPOTorchLearner`, `SACTorchLearner`, ...) at a leaf class in
`learners/`:

```python
# learners/ppo.py
from losses.next_token import NextTokenAuxLossMixin
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner

class PPOWithNextTokenAux(NextTokenAuxLossMixin, PPOTorchLearner):
    pass
```

## The mixin contract

Every mixin in this folder MUST:

1. Override `compute_loss_for_module` with RLlib's exact keyword-only signature:
   ```python
   def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
   ```
2. **Call `super().compute_loss_for_module(...)` FIRST** and store the returned
   scalar. Skipping this breaks every mixin below in the MRO and drops the base
   algorithm's loss.
3. Return `total + weight * aux_term`. Never overwrite `total` in a way that
   discards it.
4. Read hyperparameters from `config.learner_config_dict` with a **namespaced
   key** (see below).
5. Read input tensors from `fwd_out` with a **namespaced key** written by the
   paired RLModule head mixin. Read target tensors from `batch` (standard
   `Columns.*` keys).
6. Log per-iteration metrics via `self.metrics.log_value(...)` /
   `self.metrics.log_dict(...)`, keyed by `module_id`.
7. Fast-path: if the loss coefficient is `0.0` or the expected `fwd_out` key is
   absent, return `total` unchanged after the `super()` call. No wasted compute,
   no silent gradients.
8. Be pure PyTorch on the training device. No `.item()`, `.cpu()`, or
   `.numpy()` in the hot path.

## Namespacing

Each mixin owns a **prefix** used consistently for its config keys, its
`fwd_out` key, and its metric names. The prefix must match between the loss
mixin and the RLModule head mixin that produces its input.

Convention: `<namespace>/<field>` (single-level, forward-slash delimited).

Example for a `NextToken` mixin pair:
- config:   `learner_config_dict["next_token_aux/lambda"]`
- fwd_out:  `fwd_out["next_token_aux/logits"]`
- metrics:  `"next_token_aux/ce"`, `"next_token_aux/accuracy"`

Never introduce a library-wide unnamespaced key (e.g. `"aux_logits"`) — it
prevents two aux heads from coexisting.

## Template

```python
# losses/my_aux.py
"""One-line description of the auxiliary objective."""

from __future__ import annotations

import torch
from ray.rllib.core.columns import Columns

NAMESPACE = "my_aux"
LAMBDA_KEY = f"{NAMESPACE}/lambda"
FWD_KEY = f"{NAMESPACE}/pred"


class MyAuxLossMixin:
    """Reads:
      config.learner_config_dict[LAMBDA_KEY]  - coefficient (default 0.0)
      fwd_out[FWD_KEY]                        - prediction from paired head mixin
      batch[Columns.SOMETHING]                - target tensor
    """
    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
        total = super().compute_loss_for_module(
            module_id=module_id, config=config, batch=batch, fwd_out=fwd_out,
        )
        weight = float(config.learner_config_dict.get(LAMBDA_KEY, 0.0))
        if weight <= 0.0 or FWD_KEY not in fwd_out:
            return total

        pred = fwd_out[FWD_KEY]
        target = batch[Columns.SOMETHING]
        aux = ((pred - target) ** 2).mean()

        self.metrics.log_value((module_id, f"{NAMESPACE}/loss"), aux, window=1)
        return total + weight * aux
```

## Do / Don't

DO
- One aux objective per file. Task-specific helpers (target selection, masks,
  positive/negative sampling) live in the same file as the mixin that uses them.
- Depend only on `config.learner_config_dict` on the `config` object — it is
  universal across algorithms. Never touch algorithm-specific config fields
  (e.g. `config.clip_param`).
- Guard against missing `fwd_out` keys explicitly; a missing key means "no
  matching head mixin was composed on this experiment," which is legal.

DON'T
- Do not import from `ray.rllib.algorithms.ppo.*` (or any single-algo module).
  A mixin that references a specific algorithm is not a mixin — it's a subclass
  in disguise and belongs in `learners/{algo}.py`.
- Do not put loss math in `learners/*.py`. `learners/` holds only the composed
  leaf class (usually `class X(LossMixin, BaseLearner): pass`).
- Do not read module-level constants defined by an RLModule (e.g.
  `learners.models.base.AUX_LOGITS`). Own your keys locally in the mixin file.
- Do not add a new hyperparameter to `Blueprint`. Namespaced keys go into
  `learner_config_dict` (or `aux_config`) verbatim.

## Known violations in the current codebase

An agent editing this repo should recognise and, where practical, refactor the
following into the pattern above (see `learners/AGENTS.md` for the full list).
The violations relevant to loss composition:

- `learners/ppo.py:27-71` — `AuxPPOTorchLearner(PPOTorchLearner)` inlines the
  next-token cross-entropy math instead of being
  `class AuxPPOTorchLearner(NextTokenAuxLossMixin, PPOTorchLearner): pass`
  with the math in `losses/next_token.py`.
- `learners/ppo.py:18-24` — `next_token_targets(...)` is a MESS3-specific helper
  sitting in the PPO algorithm file. It belongs alongside its mixin in
  `losses/next_token.py`.
- `learners/ppo.py:15` — the Learner file imports `AUX_LOGITS, N_AUX_CLASSES`
  from `learners.models.base`, coupling the loss to the model via a shared
  global. The correct decoupling: the loss mixin declares its own namespaced
  `FWD_KEY`; the paired head mixin writes to the same string. No shared module
  constant is required.
- `learners/ppo.py:40-41` — `aux_next_token_lambda` is an unnamespaced key.
  Rename to `next_token_aux/lambda` when refactoring.
