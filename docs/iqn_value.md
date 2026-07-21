# IQN value critics

The harness provides an implicit-quantile (IQN) distributional value critic
for PPO. It models return quantiles for the critic while leaving the policy
head and PPO policy objective unchanged. This is not an IQN-DQN
implementation: it predicts scalar state-value quantiles and regresses them
against PPO's on-policy GAE value targets.

## Configure an experiment

Use either `IQNTransformerModel` or `IQNMLPModel` with
`IQNPPOTorchLearner`:

```python
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from learners import (
    HUBER_KAPPA_KEY,
    LOSS_COEFFICIENT_KEY,
    IQNPPOTorchLearner,
)
from learners.models import IQNTransformerModel, IQN_VALUE_NAMESPACE

config = (
    PPOConfig()
    .learners(
        learner_class=IQNPPOTorchLearner,
        learner_config_dict={
            LOSS_COEFFICIENT_KEY: 0.5,
            HUBER_KAPPA_KEY: 1.0,
        },
    )
    .training(vf_loss_coeff=0.0)
    .rl_module(
        rl_module_spec=RLModuleSpec(
            module_class=IQNTransformerModel,
            model_config={
                # Add the normal TransformerModel settings here.
                IQN_VALUE_NAMESPACE: {
                    "train_quantiles": 32,
                    "value_quantiles": 64,
                    "n_cosines": 64,
                },
            },
        )
    )
)
```

The model samples `train_quantiles` uniform quantile fractions for each
training forward pass. `compute_values()` uses deterministic midpoint
fractions and averages `value_quantiles` predictions to produce PPO's scalar
bootstrap and GAE baseline.

Set `vf_loss_coeff=0.0` when the quantile-Huber objective should fully replace
PPO's scalar value MSE, as in the original promoted experiments. A positive
coefficient intentionally trains the quantile mean with PPO's scalar value
loss in addition to quantile regression.

The defaults are 32 training quantiles, 64 value quantiles, 64 cosine
features, loss coefficient 0.5, and Huber kappa 1.0. Experiments should record
these scientific choices explicitly rather than rely on defaults.

## Compose another model head

`IQNValueMixin` cooperates with training-head mixins. Put other head mixins
first, followed by IQN and the encoder model:

```python
from learners.models import IQNValueMixin, NextTokenAuxHead, TransformerModel


class ExperimentModel(
    NextTokenAuxHead,
    IQNValueMixin,
    TransformerModel,
):
    pass
```

Reusable tensor pieces are also available independently:

- `learners.models.IQNValueHead` predicts quantiles from arbitrary embeddings.
- `losses.quantile_huber_loss` computes masked, device-native quantile
  regression.

All forward and loss operations stay on the model device. Training quantile
sampling uses PyTorch's seeded device RNG; value estimation uses fixed
midpoints.
