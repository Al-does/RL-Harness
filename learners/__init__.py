"""Composable models, neural components, and RLlib learner extensions."""

from learners.idaac import IDAAC, IDAACConfig, IDAACTorchLearner
from learners.optimizer import ConfigurableOptimizerMixin, build_torch_optimizer
from learners.ppg import PPG, PPGConfig, PPGTorchLearner
from learners.ppo_iqn import (
    HUBER_KAPPA_KEY,
    LOSS_COEFFICIENT_KEY,
    IQNPPOTorchLearner,
)
from learners.ppo_qr import (
    HUBER_KAPPA_KEY as QR_HUBER_KAPPA_KEY,
    LOSS_COEFFICIENT_KEY as QR_LOSS_COEFFICIENT_KEY,
    QRPPOTorchLearner,
)

__all__ = [
    "ConfigurableOptimizerMixin",
    "HUBER_KAPPA_KEY",
    "IDAAC",
    "IDAACConfig",
    "IDAACTorchLearner",
    "IQNPPOTorchLearner",
    "LOSS_COEFFICIENT_KEY",
    "PPG",
    "PPGConfig",
    "PPGTorchLearner",
    "QR_HUBER_KAPPA_KEY",
    "QR_LOSS_COEFFICIENT_KEY",
    "QRPPOTorchLearner",
    "build_torch_optimizer",
]
