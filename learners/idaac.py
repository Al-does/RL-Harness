"""Invariant Decoupled Advantage Actor-Critic on RLlib's new API stack."""

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

from learners.models.idaac import (
    ADVANTAGE_PREDICTIONS,
    OLD_VALUE_PREDICTIONS,
    ORDER_TARGETS,
    PAIRED_EMBEDDINGS,
    PAIRED_OBSERVATIONS,
    PAIR_POSITIONS,
    PAIR_VALID_MASK,
)
from losses.idaac import (
    advantage_prediction_loss,
    clipped_value_loss,
    discriminator_order_loss,
    encoder_confusion_loss,
    masked_mean,
    ppo_surrogate,
)


POLICY_OPTIMIZER = "idaac_policy"
VALUE_OPTIMIZER = "idaac_value"
DISCRIMINATOR_OPTIMIZER = "idaac_discriminator"
IDAAC_STATE_KEY = "idaac"

ADVANTAGE_LOSS = "idaac/advantage_loss"
ENCODER_INVARIANCE_LOSS = "idaac/encoder_invariance_loss"
DISCRIMINATOR_LOSS = "idaac/discriminator_loss"
DISCRIMINATOR_ACCURACY = "idaac/discriminator_accuracy"


def _copy_batch_to_cpu(batch: MultiAgentBatch) -> MultiAgentBatch:
    """Return an isolated CPU snapshot of a processed train batch."""

    return copy.deepcopy(batch).to_device(torch.device("cpu"))


def add_temporal_order_pairs(module_batch) -> None:
    """Add random, same-episode observation pairs to a processed PPO batch.

    RLlib's GAE connector preserves contiguous episode order plus termination
    and padding masks but intentionally drops episode IDs. This vectorized
    operation reconstructs episode segments without transferring data to CPU.
    Pairing happens once before minibatch shuffling, matching IDAAC's rollout
    storage behavior.
    """

    observations = module_batch[Columns.OBS]
    terminated = module_batch[Columns.TERMINATEDS].to(dtype=torch.bool)
    truncated = module_batch[Columns.TRUNCATEDS].to(dtype=torch.bool)
    leading_shape = terminated.shape
    if observations.shape[: len(leading_shape)] != leading_shape:
        raise ValueError(
            "IDAAC observations and termination flags have incompatible shapes"
        )
    valid = module_batch.get(Columns.LOSS_MASK)
    if valid is None:
        valid = torch.ones_like(terminated, dtype=torch.bool)
    else:
        valid = valid.to(device=terminated.device, dtype=torch.bool)

    count = terminated.numel()
    indices = torch.arange(count, device=terminated.device)
    flat_valid = valid.reshape(-1)
    flat_done = (terminated | truncated).reshape(-1)

    row_start = torch.zeros(count, dtype=torch.bool, device=terminated.device)
    row_end = torch.zeros_like(row_start)
    if terminated.ndim > 1:
        row_width = leading_shape[-1]
        row_start = indices.remainder(row_width) == 0
        row_end = indices.remainder(row_width) == row_width - 1
    else:
        row_start[0] = True
        row_end[-1] = True

    previous_break = torch.zeros_like(row_start)
    previous_break[1:] = flat_done[:-1] | ~flat_valid[:-1]
    starts_here = row_start | previous_break
    start_candidates = torch.where(starts_here, indices, torch.zeros_like(indices))
    starts = torch.cummax(start_candidates, dim=0).values

    next_invalid = torch.zeros_like(row_end)
    next_invalid[:-1] = ~flat_valid[1:]
    ends_here = row_end | flat_done | next_invalid
    end_candidates = torch.where(
        ends_here,
        indices,
        torch.full_like(indices, count),
    )
    ends = torch.flip(
        torch.cummin(torch.flip(end_candidates, dims=(0,)), dim=0).values,
        dims=(0,),
    )

    lengths = (ends - starts + 1).clamp_min(1)
    positions = indices - starts
    choices = (
        torch.rand(count, device=terminated.device)
        * (lengths - 1).clamp_min(1).to(dtype=torch.float32)
    ).to(dtype=torch.long)
    paired_positions = choices + (choices >= positions).to(dtype=torch.long)
    paired_indices = (starts + paired_positions).clamp(0, count - 1)
    pair_valid = (
        flat_valid
        & (lengths > 1)
        & flat_valid.gather(0, paired_indices)
        & (paired_indices != indices)
    )

    flat_observations = observations.reshape(count, *observations.shape[len(leading_shape) :])
    paired_observations = flat_observations.index_select(0, paired_indices)
    module_batch[PAIRED_OBSERVATIONS] = paired_observations.reshape_as(observations)
    module_batch[PAIR_VALID_MASK] = pair_valid.reshape(leading_shape)
    module_batch[PAIR_POSITIONS] = (
        paired_indices.remainder(leading_shape[-1])
        if terminated.ndim > 1
        else paired_indices
    ).reshape(leading_shape)
    # Match the released implementation: label 1 means the first input came
    # later than the second. Reversing this convention would be equivalent.
    module_batch[ORDER_TARGETS] = (indices > paired_indices).reshape(leading_shape)


class IDAACConfig(PPOConfig):
    """PPO configuration extended with decoupled IDAAC optimization phases."""

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or IDAAC)
        # Paper-wide Procgen defaults (Appendix C).
        self.lr = 5e-4
        self.gamma = 0.999
        self.lambda_ = 0.95
        self.entropy_coeff = 0.01
        self.clip_param = 0.2
        self.vf_clip_param = 0.2
        self.vf_loss_coeff = 0.5
        self.train_batch_size_per_learner = 16_384
        self.minibatch_size = 2_048
        self.num_epochs = 1
        self.use_kl_loss = False
        self.grad_clip = 0.5
        self.grad_clip_by = "global_norm"

        self.value_num_epochs = 9
        self.value_update_frequency = 1
        self.value_minibatch_size = 2_048
        self.advantage_loss_coeff = 0.25
        self.invariance_loss_coeff = 0.001
        self.adam_epsilon = 1e-5

    @override(PPOConfig)
    def get_default_learner_class(self):
        if self.framework_str != "torch":
            raise ValueError("IDAAC currently supports only framework='torch'")
        return IDAACTorchLearner

    @override(PPOConfig)
    def training(
        self,
        *,
        value_num_epochs: int = NotProvided,
        value_update_frequency: int = NotProvided,
        value_minibatch_size: int = NotProvided,
        advantage_loss_coeff: float = NotProvided,
        invariance_loss_coeff: float = NotProvided,
        adam_epsilon: float = NotProvided,
        **kwargs,
    ) -> Self:
        super().training(**kwargs)
        if value_num_epochs is not NotProvided:
            self.value_num_epochs = value_num_epochs
        if value_update_frequency is not NotProvided:
            self.value_update_frequency = value_update_frequency
        if value_minibatch_size is not NotProvided:
            self.value_minibatch_size = value_minibatch_size
        if advantage_loss_coeff is not NotProvided:
            self.advantage_loss_coeff = advantage_loss_coeff
        if invariance_loss_coeff is not NotProvided:
            self.invariance_loss_coeff = invariance_loss_coeff
        if adam_epsilon is not NotProvided:
            self.adam_epsilon = adam_epsilon
        return self

    @override(PPOConfig)
    def validate(self) -> None:
        super().validate()
        if self.framework_str != "torch":
            self._value_error("IDAAC currently supports only framework='torch'")
        if self.num_learners > 1:
            self._value_error(
                "IDAAC currently supports at most one Learner "
                "(`num_learners` must be 0 or 1)"
            )
        if not self.use_critic:
            self._value_error("IDAAC requires use_critic=True")
        for name in (
            "value_num_epochs",
            "value_update_frequency",
            "value_minibatch_size",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                self._value_error(f"`{name}` must be a positive integer")
        for name in (
            "advantage_loss_coeff",
            "invariance_loss_coeff",
            "vf_loss_coeff",
        ):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or value < 0.0:
                self._value_error(f"`{name}` must be a non-negative number")
        Scheduler.validate(
            fixed_value_or_schedule=self.lr,
            setting_name="lr",
            description="IDAAC optimizer learning rate",
        )
        if not isinstance(self.adam_epsilon, (int, float)) or not (
            0.0 < self.adam_epsilon < 1.0
        ):
            self._value_error("`adam_epsilon` must be between zero and one")


class IDAACTorchLearner(PPOTorchLearner):
    """Learner with isolated policy, value, and order-classifier optimizers."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._idaac_phase = "policy"
        self._idaac_optimization_step = "policy"
        self._idaac_store_value_batch = False
        self._idaac_pending_value_batch: MultiAgentBatch | None = None

    @override(TorchLearner)
    def configure_optimizers_for_module(self, module_id, config=None) -> None:
        module = self._module[module_id].unwrapped()
        required = (
            "policy_encoder",
            "policy_head",
            "advantage_head",
            "value_encoder",
            "value_head",
            "order_classifier",
        )
        missing = [name for name in required if not hasattr(module, name)]
        if missing:
            raise TypeError(
                "IDAACTorchLearner requires an IDAAC-compatible RLModule; "
                f"missing {', '.join(missing)}"
            )

        policy_params = [
            *module.policy_encoder.parameters(),
            *module.policy_head.parameters(),
            *module.advantage_head.parameters(),
        ]
        if module.policy_log_std is not None:
            policy_params.append(module.policy_log_std)
        value_params = [
            *module.value_encoder.parameters(),
            *module.value_head.parameters(),
        ]
        discriminator_params = list(module.order_classifier.parameters())
        optimizer_kwargs = {"lr": config.lr, "eps": config.adam_epsilon}
        self.register_optimizer(
            module_id=module_id,
            optimizer_name=POLICY_OPTIMIZER,
            optimizer=torch.optim.Adam(policy_params, **optimizer_kwargs),
            params=policy_params,
            lr_or_lr_schedule=config.lr,
        )
        self.register_optimizer(
            module_id=module_id,
            optimizer_name=VALUE_OPTIMIZER,
            optimizer=torch.optim.Adam(value_params, **optimizer_kwargs),
            params=value_params,
            lr_or_lr_schedule=config.lr,
        )
        self.register_optimizer(
            module_id=module_id,
            optimizer_name=DISCRIMINATOR_OPTIMIZER,
            optimizer=torch.optim.Adam(discriminator_params, **optimizer_kwargs),
            params=discriminator_params,
            lr_or_lr_schedule=config.lr,
        )

    def update(
        self,
        *args,
        idaac_phase: str = "policy",
        idaac_store_value_batch: bool = False,
        **kwargs,
    ):
        if idaac_phase not in {"policy", "value"}:
            raise ValueError("idaac_phase must be 'policy' or 'value'")
        if idaac_phase == "policy":
            self._idaac_pending_value_batch = None
            self._idaac_store_value_batch = idaac_store_value_batch
        else:
            if idaac_store_value_batch:
                raise ValueError("value phases cannot store another value batch")
            if self._idaac_pending_value_batch is None:
                raise RuntimeError("IDAAC value update has no processed policy batch")
            replay_batch = _copy_batch_to_cpu(self._idaac_pending_value_batch)
            for argument in (
                "batch",
                "batches",
                "batch_refs",
                "episodes",
                "episodes_refs",
                "data_iterators",
            ):
                kwargs.pop(argument, None)
            kwargs["training_data"] = TrainingData(batch=replay_batch)

        self._idaac_phase = idaac_phase
        succeeded = False
        try:
            result = super().update(*args, **kwargs)
            succeeded = True
            if idaac_phase == "policy" and idaac_store_value_batch:
                if self._idaac_pending_value_batch is None:
                    raise RuntimeError(
                        "IDAAC policy update did not retain its processed value batch"
                    )
            return result
        finally:
            self._idaac_store_value_batch = False
            if idaac_phase == "value":
                self._idaac_pending_value_batch = None
                self._idaac_phase = "policy"
            elif not succeeded:
                self._idaac_pending_value_batch = None

    @override(PPOLearner)
    def _make_batch_if_necessary(self, training_data):
        batch = super()._make_batch_if_necessary(training_data)
        if self._idaac_phase == "policy":
            for module_id, module_batch in batch.policy_batches.items():
                add_temporal_order_pairs(module_batch)
                if self._idaac_store_value_batch:
                    module = self.module[module_id].unwrapped()
                    with torch.no_grad():
                        module_batch[OLD_VALUE_PREDICTIONS] = module.compute_values(
                            module_batch
                        ).detach()
            if self._idaac_store_value_batch:
                self._idaac_pending_value_batch = _copy_batch_to_cpu(batch)
        return batch

    @contextmanager
    def _active_optimizers_only(self):
        if self._idaac_phase == "value":
            active = [VALUE_OPTIMIZER]
        elif self._idaac_optimization_step == "discriminator":
            active = [DISCRIMINATOR_OPTIMIZER]
        else:
            active = [POLICY_OPTIMIZER]
        original = {
            module_id: list(names)
            for module_id, names in self._module_optimizers.items()
        }
        try:
            for module_id in original:
                self._module_optimizers[module_id] = [
                    f"{module_id}_{name}" for name in active
                ]
            yield
        finally:
            for module_id, names in original.items():
                self._module_optimizers[module_id] = names

    @override(TorchLearner)
    def _uncompiled_update(self, batch, **kwargs):
        """Run IDAAC's classifier step before the policy step per minibatch."""

        if self._idaac_phase == "value":
            return super()._uncompiled_update(batch, **kwargs)

        self._compute_off_policyness(batch)
        fwd_out = self.module.forward_train(batch)

        if self.config.invariance_loss_coeff > 0.0:
            self._idaac_optimization_step = "discriminator"
            discriminator_losses = self.compute_losses(
                fwd_out=fwd_out,
                batch=batch,
            )
            discriminator_gradients = self.compute_gradients(discriminator_losses)
            discriminator_gradients = self.postprocess_gradients(
                discriminator_gradients
            )
            self.apply_gradients(discriminator_gradients)

        # The discriminator has now stepped. The encoder-confusion objective
        # evaluates its updated parameters while keeping them detached.
        self._idaac_optimization_step = "policy"
        policy_losses = self.compute_losses(fwd_out=fwd_out, batch=batch)
        policy_gradients = self.compute_gradients(policy_losses)
        policy_gradients = self.postprocess_gradients(policy_gradients)
        self.apply_gradients(policy_gradients)
        return fwd_out, policy_losses, {}

    @override(TorchLearner)
    def postprocess_gradients(self, gradients_dict):
        with self._active_optimizers_only():
            return super().postprocess_gradients(gradients_dict)

    @override(TorchLearner)
    def apply_gradients(self, gradients_dict) -> None:
        with self._active_optimizers_only():
            super().apply_gradients(gradients_dict)

    @override(PPOLearner)
    def after_gradient_based_update(self, *, timesteps) -> None:
        with self._active_optimizers_only():
            if self._idaac_phase == "policy":
                super().after_gradient_based_update(timesteps=timesteps)
            else:
                TorchLearner.after_gradient_based_update(self, timesteps=timesteps)

    @override(PPOTorchLearner)
    def compute_loss_for_module(
        self,
        *,
        module_id,
        config,
        batch,
        fwd_out,
    ):
        if self._idaac_phase == "value":
            return self._compute_value_loss(
                module_id=module_id,
                config=config,
                batch=batch,
            )
        if self._idaac_optimization_step == "discriminator":
            return self._compute_discriminator_loss(
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
        if embeddings is None or ADVANTAGE_PREDICTIONS not in fwd_out:
            raise KeyError(
                "IDAAC policy training requires embeddings and advantage predictions"
            )
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
        targets = batch[Postprocessing.ADVANTAGES]
        mask = batch.get(Columns.LOSS_MASK)
        surrogate = ppo_surrogate(
            logp_ratio,
            targets,
            clip_param=config.clip_param,
        )
        entropy = current_distribution.entropy()
        advantage_loss = advantage_prediction_loss(
            fwd_out[ADVANTAGE_PREDICTIONS],
            targets,
            mask=mask,
        )
        total = (
            -masked_mean(surrogate, mask)
            - (
                self.entropy_coeff_schedulers_per_module[
                    module_id
                ].get_current_value()
                * masked_mean(entropy, mask)
            )
            + config.advantage_loss_coeff * advantage_loss
        )

        if config.use_kl_loss:
            mean_kl = masked_mean(
                previous_distribution.kl(current_distribution),
                mask,
            )
            total = total + self.curr_kl_coeffs_per_module[module_id] * mean_kl
        else:
            mean_kl = total.new_zeros(())

        encoder_loss = total.new_zeros(())
        if (
            config.invariance_loss_coeff > 0.0
            and PAIRED_EMBEDDINGS in fwd_out
            and ORDER_TARGETS in batch
        ):
            paired_embeddings = fwd_out[PAIRED_EMBEDDINGS]
            pair_mask = batch.get(PAIR_VALID_MASK)
            if mask is not None:
                pair_mask = (
                    mask.to(dtype=torch.bool)
                    if pair_mask is None
                    else pair_mask.to(dtype=torch.bool) & mask.to(dtype=torch.bool)
                )
            encoder_logits = module.order_logits(
                embeddings,
                paired_embeddings,
                detach_classifier=True,
            )
            encoder_loss = encoder_confusion_loss(encoder_logits, mask=pair_mask)
            total = total + config.invariance_loss_coeff * encoder_loss

        self.metrics.log_dict(
            {
                POLICY_LOSS_KEY: -masked_mean(surrogate, mask),
                ENTROPY_KEY: masked_mean(entropy, mask),
                LEARNER_RESULTS_KL_KEY: mean_kl,
                ADVANTAGE_LOSS: advantage_loss,
                ENCODER_INVARIANCE_LOSS: encoder_loss,
            },
            key=module_id,
            window=1,
        )
        return total

    def _compute_discriminator_loss(self, *, module_id, config, batch, fwd_out):
        del config
        module = self.module[module_id].unwrapped()
        embeddings = fwd_out.get(Columns.EMBEDDINGS)
        paired_embeddings = fwd_out.get(PAIRED_EMBEDDINGS)
        if embeddings is None or paired_embeddings is None:
            raise KeyError(
                "IDAAC discriminator training requires paired policy embeddings"
            )
        mask = batch.get(PAIR_VALID_MASK)
        loss_mask = batch.get(Columns.LOSS_MASK)
        if loss_mask is not None:
            mask = (
                loss_mask.to(dtype=torch.bool)
                if mask is None
                else mask.to(dtype=torch.bool) & loss_mask.to(dtype=torch.bool)
            )
        logits = module.order_logits(
            embeddings,
            paired_embeddings,
            detach_embeddings=True,
        )
        discriminator_loss = discriminator_order_loss(
            logits,
            batch[ORDER_TARGETS],
            mask=mask,
        )
        correct = (
            (logits >= 0) == batch[ORDER_TARGETS].to(dtype=torch.bool)
        ).to(dtype=logits.dtype)
        accuracy = masked_mean(correct, mask)
        self.metrics.log_dict(
            {
                DISCRIMINATOR_LOSS: discriminator_loss,
                DISCRIMINATOR_ACCURACY: accuracy,
            },
            key=module_id,
            window=1,
        )
        return discriminator_loss

    def _compute_value_loss(self, *, module_id, config, batch):
        module = self.module[module_id].unwrapped()
        if OLD_VALUE_PREDICTIONS not in batch:
            raise KeyError("IDAAC value phase requires fixed old value predictions")
        predictions = module.compute_values(batch)
        mask = batch.get(Columns.LOSS_MASK)
        value_loss, unclipped_value_loss = clipped_value_loss(
            predictions,
            batch[OLD_VALUE_PREDICTIONS],
            batch[Postprocessing.VALUE_TARGETS],
            clip_param=config.vf_clip_param,
            mask=mask,
        )
        total = config.vf_loss_coeff * value_loss
        self.metrics.log_dict(
            {
                VF_LOSS_KEY: value_loss,
                LEARNER_RESULTS_VF_LOSS_UNCLIPPED_KEY: unclipped_value_loss,
                LEARNER_RESULTS_VF_EXPLAINED_VAR_KEY: explained_variance(
                    batch[Postprocessing.VALUE_TARGETS],
                    predictions,
                ),
            },
            key=module_id,
            window=1,
        )
        return total


class IDAAC(PPO):
    """PPO sampling with separate IDAAC policy and value optimization."""

    @classmethod
    @override(PPO)
    def get_default_config(cls) -> IDAACConfig:
        return IDAACConfig()

    @override(PPO)
    def setup(self, config: IDAACConfig) -> None:
        super().setup(config)
        self._idaac_policy_updates = 0

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
                raise TypeError("IDAAC currently supports single-agent episodes only")
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
        value_phase_triggered = (
            self._idaac_policy_updates % self.config.value_update_frequency == 0
        )
        with self.metrics.log_time((TIMERS, LEARNER_UPDATE_TIMER)):
            learner_results = self.learner_group.update(
                episodes=episodes,
                timesteps=timesteps,
                num_epochs=self.config.num_epochs,
                minibatch_size=self.config.minibatch_size,
                shuffle_batch_per_epoch=self.config.shuffle_batch_per_epoch,
                idaac_phase="policy",
                idaac_store_value_batch=value_phase_triggered,
            )
            self.metrics.aggregate(learner_results, key=LEARNER_RESULTS)
            modules_to_update = set(learner_results[0]) - {ALL_MODULES}

            if value_phase_triggered:
                learner_results = self.learner_group.update(
                    batch=MultiAgentBatch({}, 0),
                    timesteps=timesteps,
                    num_epochs=self.config.value_num_epochs,
                    minibatch_size=self.config.value_minibatch_size,
                    shuffle_batch_per_epoch=True,
                    idaac_phase="value",
                )
                self.metrics.aggregate(learner_results, key=LEARNER_RESULTS)
                modules_to_update.update(set(learner_results[0]) - {ALL_MODULES})
            self._idaac_policy_updates += 1

        with self.metrics.log_time((TIMERS, SYNCH_WORKER_WEIGHTS_TIMER)):
            self.env_runner_group.sync_weights(
                from_worker_or_learner_group=self.learner_group,
                policies=modules_to_update,
                inference_only=True,
            )
        self.metrics.log_value(
            "idaac/value_phase_triggered",
            int(value_phase_triggered),
            window=1,
        )
        self.metrics.log_value(
            "idaac/policy_updates",
            self._idaac_policy_updates,
            window=1,
        )

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
        if self._check_component(IDAAC_STATE_KEY, components, not_components):
            state[IDAAC_STATE_KEY] = {
                "policy_updates": self._idaac_policy_updates,
            }
        return state

    @override(PPO)
    def set_state(self, state) -> None:
        super().set_state(state)
        idaac_state = state.get(IDAAC_STATE_KEY, {})
        self._idaac_policy_updates = int(
            idaac_state.get("policy_updates", 0)
        )


__all__ = [
    "ADVANTAGE_LOSS",
    "DISCRIMINATOR_ACCURACY",
    "DISCRIMINATOR_LOSS",
    "DISCRIMINATOR_OPTIMIZER",
    "ENCODER_INVARIANCE_LOSS",
    "IDAAC",
    "IDAACConfig",
    "IDAAC_STATE_KEY",
    "IDAACTorchLearner",
    "POLICY_OPTIMIZER",
    "VALUE_OPTIMIZER",
    "add_temporal_order_pairs",
]
