"""PPO integration for an implicit-quantile distributional value critic."""

from __future__ import annotations

import torch
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing

from learners.models.iqn_value import FWD_QUANTILES, FWD_TAUS, NAMESPACE
from losses.quantile_huber import quantile_huber_loss


LOSS_COEFFICIENT_KEY = f"{NAMESPACE}/loss_coefficient"
HUBER_KAPPA_KEY = f"{NAMESPACE}/huber_kappa"


def _masked_mean(values: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    if valid is None:
        return values.mean()
    weights = valid.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


class IQNPPOTorchLearner(PPOTorchLearner):
    """Train PPO's critic as a return quantile function.

    Recipes must set ``vf_loss_coeff=0.0`` because this learner replaces PPO's
    scalar value regression with quantile regression against on-policy
    lambda-return samples.
    """

    def build(self) -> None:
        if float(self.config.vf_loss_coeff) != 0.0:
            raise ValueError("IQN PPO requires vf_loss_coeff=0.0")
        super().build()

    def compute_loss_for_module(
        self,
        *,
        module_id,
        config,
        batch,
        fwd_out,
    ):
        total = super().compute_loss_for_module(
            module_id=module_id,
            config=config,
            batch=batch,
            fwd_out=fwd_out,
        )
        learner_config = config.learner_config_dict
        coefficient = float(
            learner_config.get(LOSS_COEFFICIENT_KEY, 0.5)
        )
        kappa = float(learner_config.get(HUBER_KAPPA_KEY, 1.0))
        if coefficient <= 0.0:
            raise ValueError("IQN loss coefficient must be positive")

        quantiles = fwd_out[FWD_QUANTILES]
        taus = fwd_out[FWD_TAUS]
        valid = batch.get(Columns.LOSS_MASK)
        iqn_loss = quantile_huber_loss(
            quantiles,
            taus,
            batch[Postprocessing.VALUE_TARGETS],
            kappa=kappa,
            valid=valid,
        )
        mean_spread = _masked_mean(
            quantiles.std(dim=-1, correction=0),
            valid,
        )
        self.metrics.log_dict(
            {
                f"{NAMESPACE}/loss": iqn_loss,
                f"{NAMESPACE}/mean_quantile_spread": mean_spread,
            },
            key=module_id,
            window=1,
        )
        return total + coefficient * iqn_loss
