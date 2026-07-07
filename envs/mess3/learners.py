"""PPO learner with an auxiliary next-token cross-entropy loss (Phase-3 aux arms).

The aux head (``AUX_LOGITS`` from ``_forward_train``) predicts the NEXT token
the agent will observe.  With delay=1 the agent's observation at decision time
t carries one-hot(o_{t-1}) in its first 3 slots, so the target for step t is
simply the token slot of the observation at step t+1 — no info-dict plumbing
needed; the target is derived from the zero-padded OBS tensor itself.

Valid positions for the aux loss: both t and t+1 inside the loss mask, and the
t+1 token slot actually populated (it is all-zeros at the very first decision
of an episode, before any token is revealed).  Loss = lambda * CE, added to
the standard PPO loss; ``aux_ce`` and ``aux_accuracy`` are logged so the
"gradients reach the core" verification has a visible training signal.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModuleID, TensorType

from envs.mess3.rlmodules import AUX_LOGITS, N_AUX_CLASSES


def next_token_targets(obs: torch.Tensor, mask: torch.Tensor):
    """(targets, valid) at each position from a zero-padded (B, T, obs) batch.

    targets[b, t] = argmax over obs[b, t+1, :3]; valid requires mask at t and
    t+1 and a populated token slot at t+1.  The last position of each row has
    no successor and is never valid.
    """
    tok = obs[:, :, :N_AUX_CLASSES]
    nxt = tok[:, 1:, :]                          # (B, T-1, 3)
    targets = nxt.argmax(dim=-1)
    populated = nxt.sum(dim=-1) > 0.5
    valid = mask[:, :-1] & mask[:, 1:] & populated
    return targets, valid


class AuxPPOTorchLearner(PPOTorchLearner):
    @override(PPOTorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id: ModuleID,
        config: PPOConfig,
        batch: Dict[str, Any],
        fwd_out: Dict[str, TensorType],
    ) -> TensorType:
        total = super().compute_loss_for_module(
            module_id=module_id, config=config, batch=batch, fwd_out=fwd_out
        )
        lam = float(config.learner_config_dict.get("aux_next_token_lambda", 0.0))
        if lam <= 0.0 or AUX_LOGITS not in fwd_out:
            return total

        obs = batch[Columns.OBS]
        if obs.dim() == 2:  # non-stateful module: single (B, obs) rows, no successor
            return total
        mask = batch.get(Columns.LOSS_MASK)
        if mask is None:
            mask = torch.ones(obs.shape[:2], dtype=torch.bool, device=obs.device)
        targets, valid = next_token_targets(obs, mask)
        logits = fwd_out[AUX_LOGITS][:, :-1, :]
        if valid.any():
            ce = torch.nn.functional.cross_entropy(
                logits[valid], targets[valid], reduction="mean"
            )
            acc = (logits[valid].argmax(-1) == targets[valid]).float().mean()
        else:
            ce = torch.zeros((), device=obs.device)
            acc = torch.zeros((), device=obs.device)
        self.metrics.log_dict(
            {"aux_ce": ce, "aux_accuracy": acc}, key=module_id, window=1
        )
        return total + lam * ce
