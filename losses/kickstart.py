"""Frozen-teacher forward-KL kickstarting objective for RLlib Learners."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.lambda_defaultdict import LambdaDefaultDict
from ray.rllib.utils.metrics import NUM_ENV_STEPS_SAMPLED_LIFETIME
from ray.rllib.utils.schedules.scheduler import Scheduler

NAMESPACE = "kickstart"
COEFF_KEY = f"{NAMESPACE}/coeff"
TEACHER_CHECKPOINT_KEY = f"{NAMESPACE}/teacher_checkpoint"
KL_METRIC_KEY = f"{NAMESPACE}/kl"
COEFF_METRIC_KEY = f"{NAMESPACE}/coeff"


def masked_forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return masked mean KL(p_teacher || p_student) over the last logits axis."""
    log_p_teacher = F.log_softmax(teacher_logits, dim=-1)
    log_p_student = F.log_softmax(student_logits, dim=-1)
    kl = (log_p_teacher.exp() * (log_p_teacher - log_p_student)).sum(dim=-1)
    if mask is None:
        return kl.mean()
    valid = mask.to(dtype=torch.bool)
    count = valid.sum().clamp_min(1)
    return kl.masked_select(valid).sum() / count


class KickstartTeacherKLMixin:
    """Distill a frozen teacher policy's action distribution into the student.

    Adds ``coeff * KL(p_teacher || p_student)`` over per-timestep policy logits
    to the base Learner loss. The teacher is a frozen ``RLModule`` restored via
    ``RLModule.from_checkpoint`` and evaluated on the same learner batch, so
    trajectories, critics, and environments are untouched.

    Reads from ``config.learner_config_dict``:

    - ``kickstart/coeff``: float or ``[[env_steps, value], ...]`` piecewise
      schedule keyed on ``NUM_ENV_STEPS_SAMPLED_LIFETIME`` (same stepping as
      RLlib's entropy/lr schedulers). ``0.0`` disables the term entirely.
    - ``kickstart/teacher_checkpoint``: path to an RLModule checkpoint
      (required when the coefficient is positive).

    Metrics: ``kickstart/kl`` and ``kickstart/coeff``, keyed by module.
    """

    def build(self):
        super().build()
        self._kickstart_schedulers = LambdaDefaultDict(
            lambda module_id: Scheduler(
                self.config.get_config_for_module(
                    module_id
                ).learner_config_dict.get(COEFF_KEY, 0.0),
                framework=self.framework,
                device=self._device,
            )
        )
        self._kickstart_coeffs = {}
        self._kickstart_teachers = {}

    def remove_module(self, module_id, **kwargs):
        spec = super().remove_module(module_id, **kwargs)
        self._kickstart_schedulers.pop(module_id, None)
        self._kickstart_coeffs.pop(module_id, None)
        self._kickstart_teachers.pop(module_id, None)
        return spec

    def after_gradient_based_update(self, *, timesteps):
        super().after_gradient_based_update(timesteps=timesteps)
        steps = timesteps.get(NUM_ENV_STEPS_SAMPLED_LIFETIME, 0)
        for module_id in self.module._rl_modules:
            scheduler = self._kickstart_schedulers[module_id]
            if scheduler.use_schedule:
                self._kickstart_coeffs[module_id] = scheduler.update(
                    timestep=steps
                )
                self.metrics.log_value(
                    (module_id, COEFF_METRIC_KEY),
                    self._kickstart_coeffs[module_id],
                    window=1,
                )

    def _kickstart_teacher(self, module_id, config):
        teacher = self._kickstart_teachers.get(module_id)
        if teacher is None:
            checkpoint = config.learner_config_dict.get(TEACHER_CHECKPOINT_KEY)
            if not checkpoint:
                raise ValueError(
                    f"active {NAMESPACE} loss requires "
                    f"{TEACHER_CHECKPOINT_KEY!r} in learner_config_dict"
                )
            teacher = RLModule.from_checkpoint(str(checkpoint))
            teacher.eval()
            teacher.requires_grad_(False)
            teacher.to(self._device)
            self._kickstart_teachers[module_id] = teacher
        return teacher

    def compute_loss_for_module(self, *, module_id, config, batch, fwd_out):
        total = super().compute_loss_for_module(
            module_id=module_id,
            config=config,
            batch=batch,
            fwd_out=fwd_out,
        )

        spec = config.learner_config_dict.get(COEFF_KEY, 0.0)
        coeff = self._kickstart_coeffs.get(module_id)
        if coeff is None:
            coeff = spec[0][1] if isinstance(spec, (list, tuple)) else float(spec)
        if coeff <= 0.0:
            return total

        teacher = self._kickstart_teacher(module_id, config)
        with torch.no_grad():
            teacher_out = teacher.forward_train(batch)
        kl = masked_forward_kl(
            fwd_out[Columns.ACTION_DIST_INPUTS],
            teacher_out[Columns.ACTION_DIST_INPUTS],
            mask=batch.get(Columns.LOSS_MASK),
        )

        self.metrics.log_value(
            (module_id, KL_METRIC_KEY), kl, window=1
        )
        return total + coeff * kl
