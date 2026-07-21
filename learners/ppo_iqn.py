"""PPO integration for an implicit-quantile value critic."""

from __future__ import annotations

from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.evaluation.postprocessing import Postprocessing

from learners.models.iqn_value import (
    FWD_QUANTILES,
    FWD_TAUS,
    NAMESPACE,
)
from losses.quantile import quantile_huber_loss

LOSS_COEFFICIENT_KEY = f"{NAMESPACE}/loss_coefficient"
HUBER_KAPPA_KEY = f"{NAMESPACE}/huber_kappa"


class IQNPPOTorchLearner(PPOTorchLearner):
    """Add quantile regression against PPO's on-policy value targets."""

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
        if FWD_QUANTILES not in fwd_out or FWD_TAUS not in fwd_out:
            raise ValueError(
                "IQNPPOTorchLearner requires an IQN value model that emits "
                f"{FWD_QUANTILES!r} and {FWD_TAUS!r}"
            )

        quantiles = fwd_out[FWD_QUANTILES]
        taus = fwd_out[FWD_TAUS]
        targets = batch[Postprocessing.VALUE_TARGETS]
        valid = batch.get(Columns.LOSS_MASK)
        iqn_loss = quantile_huber_loss(
            quantiles,
            taus,
            targets,
            kappa=kappa,
            valid=valid,
        )

        spread = quantiles.std(dim=-1, correction=0)
        if valid is None:
            mean_spread = spread.mean()
        else:
            valid_float = valid.to(
                device=spread.device,
                dtype=spread.dtype,
            )
            mean_spread = (
                (spread * valid_float).sum()
                / valid_float.sum().clamp_min(1.0)
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
