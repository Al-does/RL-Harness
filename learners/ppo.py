"""PPO Learner leaves composed from algorithm-agnostic loss mixins."""

from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner

from losses.next_token import NextTokenAuxLossMixin


class PPOWithNextTokenAux(NextTokenAuxLossMixin, PPOTorchLearner):
    pass


# Transitional import name for scripts outside this repository.
AuxPPOTorchLearner = PPOWithNextTokenAux

__all__ = ["AuxPPOTorchLearner", "PPOWithNextTokenAux"]
