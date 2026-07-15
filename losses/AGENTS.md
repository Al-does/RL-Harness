# `losses/` — reusable objective primitives and Learner extensions

This folder holds reusable tensor loss operations and composable extensions
for RLlib Learners. Prefer pure tensor functions for loss math; use a mixin
when an orthogonal objective must cooperate with RLlib's
`compute_loss_for_module` hook.

Auxiliary mixins advertised as **algorithm-agnostic** must compose onto any
compatible RLlib base Learner (`PPOTorchLearner`, `SACTorchLearner`, ...) at a
leaf class in an experiment or, when independently reusable, `learners/`:

```python
# experiment.py (or a stable shared learner leaf)
from losses.next_token import NextTokenAuxLossMixin
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner

class ExperimentLearner(NextTokenAuxLossMixin, PPOTorchLearner):
    pass
```

Not every objective is algorithm-agnostic. Primary policy/value/Q objectives
and distributional algorithm changes may require an algorithm-specific
Learner integration. Do not disguise such coupling as a universal mixin.

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
- Keep one coherent objective per file/package. Reusable masks, sampling, and
  tensor target transforms may live beside it.
- Keep environment observation-layout knowledge in a domain or experiment
  adapter. A generic loss must not assume that a token is the first one-hot
  slice of `Columns.OBS`, for example.
- Depend only on `config.learner_config_dict` on the `config` object — it is
  universal across algorithms. Never touch algorithm-specific config fields
  (e.g. `config.clip_param`).
- Guard against missing `fwd_out` keys explicitly; a missing key means "no
  matching head mixin was composed on this experiment," which is legal.

DON'T
- Do not import from `ray.rllib.algorithms.ppo.*` (or any single-algo module).
  An extension that references a specific algorithm is algorithm integration,
  not an algorithm-agnostic mixin; place the integration in `learners/` or the
  experiment while keeping reusable tensor math here.
- Do not put reusable loss math in `learners/*.py`.
- Do not read module-level constants defined by an RLModule (e.g.
  `learners.models.base.AUX_LOGITS`). Own your keys locally in the mixin file.
- Do not recreate Blueprint fields. Experiments write namespaced keys directly
  into their fresh `AlgorithmConfig`.

## Verification expectations

Use inline task adapters in generic tests. Test the zero-weight fast path, an
active objective, missing required adapters, and cooperative composition with
the base loss. Keep one-experiment Learner leaves in `experiment.py`, and test
paired head/loss namespaces together so string drift fails early.

All forward and loss math must remain on the training device. Metric logging
may receive tensors, but hot-path code must not call `.item()`, `.cpu()`, or
`.numpy()`.
