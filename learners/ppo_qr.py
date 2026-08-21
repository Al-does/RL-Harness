"""PPO integration for a fixed-quantile distributional value critic."""

from __future__ import annotations

import torch
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing

from learners.components.quantile import midpoint_taus
from learners.models.qr_value import FWD_QUANTILES, NAMESPACE
from losses.quantile_huber import quantile_huber_loss


LOSS_COEFFICIENT_KEY = f"{NAMESPACE}/loss_coefficient"
HUBER_KAPPA_KEY = f"{NAMESPACE}/huber_kappa"


def _validate_qr_config(config) -> tuple[float, float]:
    """Return validated QR loss settings from a PPO config."""

    if float(config.vf_loss_coeff) != 0.0:
        raise ValueError("QR PPO requires vf_loss_coeff=0.0")
    learner_config = config.learner_config_dict
    coefficient = float(learner_config.get(LOSS_COEFFICIENT_KEY, 0.5))
    kappa = float(learner_config.get(HUBER_KAPPA_KEY, 1.0))
    if coefficient <= 0.0:
        raise ValueError("QR loss coefficient must be positive")
    if kappa <= 0.0:
        raise ValueError("QR Huber kappa must be positive")
    return coefficient, kappa


def _masked_mean(values: torch.Tensor, valid: torch.Tensor | None) -> torch.Tensor:
    if valid is None:
        return values.mean()
    weights = valid.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


class QRPPOTorchLearner(PPOTorchLearner):
    """Train PPO's critic as a fixed set of return quantiles.

    Recipes must set ``vf_loss_coeff=0.0`` because this learner replaces PPO's
    scalar value regression with QR-DQN-style quantile regression against
    on-policy lambda-return samples.
    """

    def build(self) -> None:
        _validate_qr_config(self.config)
        super().build()

    def compute_loss_for_module(
        self,
        *,
        module_id,
        config,
        batch,
        fwd_out,
    ):
        if FWD_QUANTILES not in fwd_out:
            raise ValueError(
                "QRPPOTorchLearner requires an RLModule that emits "
                f"{FWD_QUANTILES!r}; compose QRValueMixin with the model"
            )
        total = super().compute_loss_for_module(
            module_id=module_id,
            config=config,
            batch=batch,
            fwd_out=fwd_out,
        )
        coefficient, kappa = _validate_qr_config(config)

        quantiles = fwd_out[FWD_QUANTILES]
        valid = batch.get(Columns.LOSS_MASK)
        qr_loss = quantile_huber_loss(
            quantiles,
            midpoint_taus(quantiles, quantiles.shape[-1]),
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
                f"{NAMESPACE}/loss": qr_loss,
                f"{NAMESPACE}/mean_quantile_spread": mean_spread,
            },
            key=module_id,
            window=1,
        )
        return total + coefficient * qr_loss
