"""Single-network Phasic Policy Gradient on RLlib's new API stack.

This implements the ``detach`` architecture from Cobbe et al. (2021): PPO's
value loss trains only the value head during policy phases, while periodic
auxiliary phases train the shared representation through a second value head.
A frozen pre-auxiliary policy supplies the behavioral-cloning KL target.
"""

from __future__ import annotations

import copy
from contextlib import contextmanager
from typing import Any, Collection

import torch
from ray.rllib.algorithms.algorithm_config import NotProvided
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo import (
    LEARNER_RESULTS_KL_KEY,
    LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY,
    LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY,
)
from ray.rllib.algorithms.ppo.ppo_learner import PPOLearner
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.core.learner.learner import (
    DEFAULT_OPTIMIZER,
    ENTROPY_KEY,
    POLICY_LOSS_KEY,
    VF_LOSS_KEY,
)
from ray.rllib.core.learner.training_data import TrainingData
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.evaluation.postprocessing import Postprocessing
from ray.rllib.execution.rollout_ops import synchronous_parallel_sample
from ray.rllib.policy.sample_batch import MultiAgentBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.metrics import (
    ALL_MODULES,
    ENV_RUNNER_RESULTS,
    ENV_RUNNER_SAMPLING_TIMER,
    LEARNER_RESULTS,
    LEARNER_UPDATE_TIMER,
    NUM_ENV_STEPS_SAMPLED_LIFETIME,
    NUM_MODULE_STEPS_TRAINED_LIFETIME,
    SYNCH_WORKER_WEIGHTS_TIMER,
    TIMERS,
)
from ray.rllib.utils.schedules.scheduler import Scheduler
from ray.rllib.utils.torch_utils import explained_variance
from typing_extensions import Self

from learners.models.ppg import AUX_VALUE_PREDICTIONS


POLICY_OPTIMIZER = DEFAULT_OPTIMIZER
AUXILIARY_OPTIMIZER = "ppg_auxiliary"
PPG_STATE_KEY = "ppg"

AUX_POLICY_KL = "ppg/aux_policy_kl"
AUX_VALUE_LOSS = "ppg/aux_value_loss"
AUX_TRUE_VALUE_LOSS = "ppg/aux_true_value_loss"


def _copy_batch_to_cpu(batch: MultiAgentBatch) -> MultiAgentBatch:
    """Return an isolated, CPU-resident snapshot of a processed train batch."""
    return copy.deepcopy(batch).to_device(torch.device("cpu"))


def _masked_mean(values: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return values.mean()
    weights = mask.to(device=values.device, dtype=values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


class PPGConfig(PPOConfig):
    """PPO configuration extended with periodic PPG auxiliary phases."""

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or PPG)
        self.policy_iterations_per_aux = 32
        self.aux_epochs = 6
        self.aux_minibatch_size = 128
        self.aux_lr = 5e-4
        self.beta_clone = 1.0
        self.aux_value_loss_coeff = 1.0
        self.aux_true_value_loss_coeff = 1.0

    @override(PPOConfig)
    def get_default_learner_class(self):
        if self.framework_str != "torch":
            raise ValueError("PPG currently supports only framework='torch'")
        return PPGTorchLearner

    @override(PPOConfig)
    def training(
        self,
        *,
        policy_iterations_per_aux: int = NotProvided,
        aux_epochs: int = NotProvided,
        aux_minibatch_size: int = NotProvided,
        aux_lr: float | list[list[float]] = NotProvided,
        beta_clone: float = NotProvided,
        aux_value_loss_coeff: float = NotProvided,
        aux_true_value_loss_coeff: float = NotProvided,
        **kwargs,
    ) -> Self:
        super().training(**kwargs)
        if policy_iterations_per_aux is not NotProvided:
            self.policy_iterations_per_aux = policy_iterations_per_aux
        if aux_epochs is not NotProvided:
            self.aux_epochs = aux_epochs
        if aux_minibatch_size is not NotProvided:
            self.aux_minibatch_size = aux_minibatch_size
        if aux_lr is not NotProvided:
            self.aux_lr = aux_lr
        if beta_clone is not NotProvided:
            self.beta_clone = beta_clone
        if aux_value_loss_coeff is not NotProvided:
            self.aux_value_loss_coeff = aux_value_loss_coeff
        if aux_true_value_loss_coeff is not NotProvided:
            self.aux_true_value_loss_coeff = aux_true_value_loss_coeff
        return self

    @override(PPOConfig)
    def validate(self) -> None:
        super().validate()
        if self.framework_str != "torch":
            self._value_error("PPG currently supports only framework='torch'")
        if self.num_learners > 1:
            self._value_error(
                "PPG currently supports at most one Learner "
                "(`num_learners` must be 0 or 1); multi-Learner DDP can fail "
                "because phase-specific parameters are unused"
            )
        if not self.use_critic:
            self._value_error("PPG requires use_critic=True")
        for name in (
            "policy_iterations_per_aux",
            "aux_epochs",
            "aux_minibatch_size",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                self._value_error(f"`{name}` must be a positive integer")
        Scheduler.validate(
            fixed_value_or_schedule=self.aux_lr,
            setting_name="aux_lr",
            description="PPG auxiliary learning rate",
        )
        for name in (
            "beta_clone",
            "aux_value_loss_coeff",
            "aux_true_value_loss_coeff",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value < 0.0:
                self._value_error(f"`{name}` must be a non-negative number")
        if self.beta_clone == 0.0 and self.aux_value_loss_coeff > 0.0:
            self._value_error(
                "`beta_clone` must be positive when the auxiliary value loss "
                "updates the policy representation"
            )


class PPGTorchLearner(PPOTorchLearner):
    """PPO Learner with detached policy-phase values and PPG auxiliary loss."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._ppg_phase = "policy"
        self._ppg_frozen_modules: dict[str, torch.nn.Module] = {}
        self._ppg_replay_batches: list[MultiAgentBatch] = []
        self._ppg_pending_replay_batch: MultiAgentBatch | None = None

    @override(TorchLearner)
    def configure_optimizers_for_module(self, module_id, config=None) -> None:
        module = self._module[module_id]
        params = self.get_parameters(module)
        self.register_optimizer(
            module_id=module_id,
            optimizer_name=POLICY_OPTIMIZER,
            optimizer=torch.optim.Adam(params),
            params=params,
            lr_or_lr_schedule=config.lr,
        )
        self.register_optimizer(
            module_id=module_id,
            optimizer_name=AUXILIARY_OPTIMIZER,
            optimizer=torch.optim.Adam(params),
            params=params,
            lr_or_lr_schedule=config.aux_lr,
        )

    def update(
        self,
        *args,
        ppg_phase: str = "policy",
        ppg_aux_start: bool = False,
        ppg_aux_end: bool = False,
        ppg_replay_index: int | None = None,
        **kwargs,
    ):
        if ppg_phase not in {"policy", "auxiliary"}:
            raise ValueError("ppg_phase must be 'policy' or 'auxiliary'")
        if ppg_phase == "policy":
            if ppg_aux_start or ppg_aux_end:
                raise ValueError("auxiliary phase flags require ppg_phase='auxiliary'")
            if ppg_replay_index is not None:
                raise ValueError("ppg_replay_index requires ppg_phase='auxiliary'")
            if self._ppg_frozen_modules:
                raise RuntimeError("cannot run a policy update during an auxiliary phase")
            self._ppg_pending_replay_batch = None
        else:
            if ppg_replay_index is None:
                raise ValueError("auxiliary updates require ppg_replay_index")
            if not 0 <= ppg_replay_index < len(self._ppg_replay_batches):
                raise IndexError(
                    f"PPG replay index {ppg_replay_index} is out of range for "
                    f"{len(self._ppg_replay_batches)} buffered batches"
                )
            if ppg_aux_start:
                if self._ppg_frozen_modules:
                    raise RuntimeError("PPG auxiliary phase is already active")
                self._ppg_frozen_modules = {
                    module_id: copy.deepcopy(module.unwrapped())
                    .requires_grad_(False)
                    .eval()
                    for module_id, module in self.module.items()
                }
            elif not self._ppg_frozen_modules:
                raise RuntimeError(
                    "PPG auxiliary update started without a policy snapshot"
                )

            replay_batch = _copy_batch_to_cpu(
                self._ppg_replay_batches[ppg_replay_index]
            )
            for module_batch in replay_batch.policy_batches.values():
                module_batch.shuffle()
            for data_argument in (
                "batch",
                "batches",
                "batch_refs",
                "episodes",
                "episodes_refs",
                "data_iterators",
            ):
                kwargs.pop(data_argument, None)
            kwargs["training_data"] = TrainingData(batch=replay_batch)

        self._ppg_phase = ppg_phase
        update_succeeded = False
        try:
            result = super().update(*args, **kwargs)
            update_succeeded = True
            if ppg_phase == "policy":
                if self._ppg_pending_replay_batch is None:
                    raise RuntimeError(
                        "PPG policy update did not produce a processed replay batch"
                    )
                self._ppg_replay_batches.append(self._ppg_pending_replay_batch)
            return result
        finally:
            self._ppg_pending_replay_batch = None
            if ppg_aux_end:
                self._ppg_frozen_modules.clear()
                self._ppg_phase = "policy"
                if update_succeeded:
                    self._ppg_replay_batches.clear()

    @override(PPOLearner)
    def _make_batch_if_necessary(self, training_data):
        batch = super()._make_batch_if_necessary(training_data)
        if self._ppg_phase == "policy":
            if self._ppg_pending_replay_batch is not None:
                raise RuntimeError(
                    "PPG policy update produced more than one processed train batch"
                )
            self._ppg_pending_replay_batch = _copy_batch_to_cpu(batch)
        return batch

    @override(PPOLearner)
    def get_state(
        self,
        components: str | Collection[str] | None = None,
        *,
        not_components: str | Collection[str] | None = None,
        **kwargs,
    ):
        state = super().get_state(
            components=components,
            not_components=not_components,
            **kwargs,
        )
        if self._check_component(PPG_STATE_KEY, components, not_components):
            state[PPG_STATE_KEY] = {
                "replay_batches": [
                    _copy_batch_to_cpu(batch)
                    for batch in self._ppg_replay_batches
                ],
            }
        return state

    @override(PPOLearner)
    def set_state(self, state) -> None:
        super().set_state(state)
        if PPG_STATE_KEY in state:
            self._ppg_replay_batches = [
                _copy_batch_to_cpu(batch)
                for batch in state[PPG_STATE_KEY].get("replay_batches", [])
            ]
        self._ppg_pending_replay_batch = None
        self._ppg_frozen_modules.clear()
        self._ppg_phase = "policy"

    @contextmanager
    def _active_optimizer_only(self):
        optimizer_name = (
            POLICY_OPTIMIZER
            if self._ppg_phase == "policy"
            else AUXILIARY_OPTIMIZER
        )
        original = {
            module_id: list(names)
            for module_id, names in self._module_optimizers.items()
        }
        try:
            for module_id in original:
                self._module_optimizers[module_id] = [
                    f"{module_id}_{optimizer_name}"
                ]
            yield
        finally:
            for module_id, names in original.items():
                self._module_optimizers[module_id] = names

    @override(TorchLearner)
    def postprocess_gradients(self, gradients_dict):
        with self._active_optimizer_only():
            return super().postprocess_gradients(gradients_dict)

    @override(TorchLearner)
    def apply_gradients(self, gradients_dict) -> None:
        with self._active_optimizer_only():
            super().apply_gradients(gradients_dict)

    @override(PPOLearner)
    def after_gradient_based_update(self, *, timesteps) -> None:
        with self._active_optimizer_only():
            if self._ppg_phase == "policy":
                super().after_gradient_based_update(timesteps=timesteps)
            else:
                TorchLearner.after_gradient_based_update(
                    self,
                    timesteps=timesteps,
                )

    @override(PPOTorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id,
        config,
        batch,
        fwd_out,
    ):
        if self._ppg_phase == "auxiliary":
            return self._compute_auxiliary_loss(
                module_id=module_id,
                config=config,
                batch=batch,
                fwd_out=fwd_out,
            )
        return self._compute_policy_loss(
            module_id=module_id,
            config=config,
            batch=batch,
            fwd_out=fwd_out,
        )

    def _compute_policy_loss(self, *, module_id, config, batch, fwd_out):
        module = self.module[module_id].unwrapped()
        embeddings = fwd_out.get(Columns.EMBEDDINGS)
        if embeddings is None:
            raise KeyError(
                "PPG requires the RLModule training forward to emit "
                f"{Columns.EMBEDDINGS!r}"
            )
        mask = batch.get(Columns.LOSS_MASK)
        current_distribution = module.get_train_action_dist_cls().from_logits(
            fwd_out[Columns.ACTION_DIST_INPUTS]
        )
        previous_distribution = (
            module.get_exploration_action_dist_cls().from_logits(
                batch[Columns.ACTION_DIST_INPUTS]
            )
        )
        logp_ratio = torch.exp(
            current_distribution.logp(batch[Columns.ACTIONS])
            - batch[Columns.ACTION_LOGP]
        )
        advantages = batch[Postprocessing.ADVANTAGES]
        surrogate = torch.minimum(
            advantages * logp_ratio,
            advantages
            * torch.clamp(
                logp_ratio,
                1.0 - config.clip_param,
                1.0 + config.clip_param,
            ),
        )
        entropy = current_distribution.entropy()
        values = module.compute_values(batch, embeddings=embeddings.detach())
        value_error_squared = (
            values - batch[Postprocessing.VALUE_TARGETS]
        ).square()
        clipped_value_loss = value_error_squared.clamp(
            max=config.vf_clip_param
        )
        total = _masked_mean(
            -surrogate
            + config.vf_loss_coeff * clipped_value_loss
            - (
                self.entropy_coeff_schedulers_per_module[
                    module_id
                ].get_current_value()
                * entropy
            ),
            mask,
        )
        if config.use_kl_loss:
            mean_kl = _masked_mean(
                previous_distribution.kl(current_distribution),
                mask,
            )
            total = (
                total
                + self.curr_kl_coeffs_per_module[module_id] * mean_kl
            )
        else:
            mean_kl = total.new_zeros(())

        mean_value_loss = _masked_mean(clipped_value_loss, mask)
        self.metrics.log_dict(
            {
                POLICY_LOSS_KEY: -_masked_mean(surrogate, mask),
                VF_LOSS_KEY: mean_value_loss,
                LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY: _masked_mean(
                    value_error_squared,
                    mask,
                ),
                LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY: explained_variance(
                    batch[Postprocessing.VALUE_TARGETS],
                    values,
                ),
                ENTROPY_KEY: _masked_mean(entropy, mask),
                LEARNER_RESULTS_KL_KEY: mean_kl,
            },
            key=module_id,
            window=1,
        )
        return total

    def _compute_auxiliary_loss(self, *, module_id, config, batch, fwd_out):
        module = self.module[module_id].unwrapped()
        frozen = self._ppg_frozen_modules[module_id]
        embeddings = fwd_out.get(Columns.EMBEDDINGS)
        if embeddings is None:
            raise KeyError(
                "PPG requires the RLModule training forward to emit "
                f"{Columns.EMBEDDINGS!r}"
            )
        if AUX_VALUE_PREDICTIONS not in fwd_out:
            raise KeyError(
                "PPG auxiliary phases require an RLModule composed with "
                "PPGAuxiliaryValueHead"
            )

        with torch.no_grad():
            frozen_outputs = frozen.forward_train(batch)
        frozen_distribution = frozen.get_train_action_dist_cls().from_logits(
            frozen_outputs[Columns.ACTION_DIST_INPUTS]
        )
        current_distribution = module.get_train_action_dist_cls().from_logits(
            fwd_out[Columns.ACTION_DIST_INPUTS]
        )
        mask = batch.get(Columns.LOSS_MASK)
        clone_kl = _masked_mean(
            frozen_distribution.kl(current_distribution),
            mask,
        )
        targets = batch[Postprocessing.VALUE_TARGETS]
        auxiliary_value_error = (
            fwd_out[AUX_VALUE_PREDICTIONS] - targets
        ).square()
        true_values = module.compute_values(
            batch,
            embeddings=embeddings.detach(),
        )
        true_value_error = (true_values - targets).square()
        auxiliary_value_loss = 0.5 * _masked_mean(
            auxiliary_value_error,
            mask,
        )
        true_value_loss = 0.5 * _masked_mean(true_value_error, mask)
        total = (
            config.beta_clone * clone_kl
            + config.aux_value_loss_coeff * auxiliary_value_loss
            + config.aux_true_value_loss_coeff * true_value_loss
        )
        self.metrics.log_dict(
            {
                AUX_POLICY_KL: clone_kl,
                AUX_VALUE_LOSS: auxiliary_value_loss,
                AUX_TRUE_VALUE_LOSS: true_value_loss,
            },
            key=module_id,
            window=1,
        )
        return total


class PPG(PPO):
    """PPO policy phases interleaved with replayed PPG auxiliary phases."""

    @classmethod
    @override(PPO)
    def get_default_config(cls) -> PPGConfig:
        return PPGConfig()

    @override(PPO)
    def setup(self, config: PPGConfig) -> None:
        super().setup(config)
        self._ppg_policy_iterations = 0

    @override(PPO)
    def training_step(self) -> None:
        with self.metrics.log_time((TIMERS, ENV_RUNNER_SAMPLING_TIMER)):
            sample_kwargs = {
                "worker_set": self.env_runner_group,
                "sample_timeout_s": self.config.sample_timeout_s,
                "_uses_new_env_runners": True,
                "_return_metrics": True,
            }
            if self.config.count_steps_by == "agent_steps":
                sample_kwargs["max_agent_steps"] = self.config.total_train_batch_size
            else:
                sample_kwargs["max_env_steps"] = self.config.total_train_batch_size
            episodes, env_runner_results = synchronous_parallel_sample(
                **sample_kwargs
            )
            if not episodes:
                return
            if not all(isinstance(episode, SingleAgentEpisode) for episode in episodes):
                raise TypeError("PPG currently supports single-agent episodes only")
            self.metrics.aggregate(env_runner_results, key=ENV_RUNNER_RESULTS)

        timesteps = {
            NUM_ENV_STEPS_SAMPLED_LIFETIME: self.metrics.peek(
                (ENV_RUNNER_RESULTS, NUM_ENV_STEPS_SAMPLED_LIFETIME)
            ),
            NUM_MODULE_STEPS_TRAINED_LIFETIME: self.metrics.peek(
                (
                    LEARNER_RESULTS,
                    ALL_MODULES,
                    NUM_MODULE_STEPS_TRAINED_LIFETIME,
                ),
                default=0,
            ),
        }
        with self.metrics.log_time((TIMERS, LEARNER_UPDATE_TIMER)):
            learner_results = self.learner_group.update(
                episodes=episodes,
                timesteps=timesteps,
                num_epochs=self.config.num_epochs,
                minibatch_size=self.config.minibatch_size,
                shuffle_batch_per_epoch=self.config.shuffle_batch_per_epoch,
                ppg_phase="policy",
            )
            self.metrics.aggregate(learner_results, key=LEARNER_RESULTS)

            self._ppg_policy_iterations += 1
            auxiliary_triggered = (
                self._ppg_policy_iterations
                >= self.config.policy_iterations_per_aux
            )
            if auxiliary_triggered:
                learner_results = self._run_auxiliary_phase(timesteps)
                self._ppg_policy_iterations = 0

        modules_to_update = set(learner_results[0]) - {ALL_MODULES}
        with self.metrics.log_time((TIMERS, SYNCH_WORKER_WEIGHTS_TIMER)):
            self.env_runner_group.sync_weights(
                from_worker_or_learner_group=self.learner_group,
                policies=modules_to_update,
                inference_only=True,
            )
        self.metrics.log_value(
            "ppg/aux_phase_triggered",
            int(auxiliary_triggered),
            window=1,
        )
        self.metrics.log_value(
            "ppg/policy_iterations_since_aux",
            self._ppg_policy_iterations,
            window=1,
        )
        self.metrics.log_value(
            "ppg/buffered_env_steps",
            self._ppg_policy_iterations * self.config.total_train_batch_size,
            window=1,
        )

    def _run_auxiliary_phase(self, timesteps) -> list[dict[str, Any]]:
        num_batches = self._ppg_policy_iterations
        total_updates = self.config.aux_epochs * num_batches
        update_index = 0
        latest_results = None
        for _ in range(self.config.aux_epochs):
            for replay_index in range(num_batches):
                update_index += 1
                latest_results = self.learner_group.update(
                    batch=MultiAgentBatch({}, 0),
                    timesteps=timesteps,
                    num_epochs=1,
                    minibatch_size=self.config.aux_minibatch_size,
                    shuffle_batch_per_epoch=False,
                    ppg_phase="auxiliary",
                    ppg_aux_start=update_index == 1,
                    ppg_aux_end=update_index == total_updates,
                    ppg_replay_index=replay_index,
                )
                self.metrics.aggregate(latest_results, key=LEARNER_RESULTS)
        if latest_results is None:
            raise RuntimeError("PPG auxiliary phase had no replay batches")
        return latest_results

    @override(PPO)
    def get_state(
        self,
        components: str | Collection[str] | None = None,
        *,
        not_components: str | Collection[str] | None = None,
        **kwargs,
    ):
        state = super().get_state(
            components=components,
            not_components=not_components,
            **kwargs,
        )
        if self._check_component(PPG_STATE_KEY, components, not_components):
            state[PPG_STATE_KEY] = {
                "policy_iterations": self._ppg_policy_iterations,
            }
        return state

    @override(PPO)
    def set_state(self, state) -> None:
        super().set_state(state)
        ppg_state = state.get(PPG_STATE_KEY, {})
        self._ppg_policy_iterations = int(
            ppg_state.get("policy_iterations", 0)
        )


__all__ = [
    "AUXILIARY_OPTIMIZER",
    "AUX_POLICY_KL",
    "AUX_TRUE_VALUE_LOSS",
    "AUX_VALUE_LOSS",
    "POLICY_OPTIMIZER",
    "PPG",
    "PPGConfig",
    "PPGTorchLearner",
]
