"""Unit tests for the frozen-teacher kickstart KL loss mixin."""

from types import SimpleNamespace

import pytest
import torch

from losses.kickstart import (
    COEFF_KEY,
    KickstartTeacherKLMixin,
    masked_forward_kl,
)
from ray.rllib.core.columns import Columns
from ray.rllib.utils.lambda_defaultdict import LambdaDefaultDict
from ray.rllib.utils.metrics import NUM_ENV_STEPS_SAMPLED_LIFETIME
from ray.rllib.utils.schedules.scheduler import Scheduler


class RecordingMetrics:
    def __init__(self):
        self.values = []

    def log_value(self, key, value, *, window=None, **kwargs):
        self.values.append((key, value))

    def log_dict(self, values, *, key, window):
        for name, value in values.items():
            self.values.append(((key, name), value))


class FakeTeacher:
    def __init__(self, logits):
        self.logits = logits
        self.forward_calls = 0

    def forward_train(self, batch):
        self.forward_calls += 1
        return {Columns.ACTION_DIST_INPUTS: self.logits}


class BaseLearner:
    def compute_loss_for_module(
        self, *, module_id, config, batch, fwd_out
    ):
        self.base_calls += 1
        return self.base_loss

    def after_gradient_based_update(self, *, timesteps):
        pass


class ComposedLearner(KickstartTeacherKLMixin, BaseLearner):
    pass


def make_learner(teacher_logits=None, base_loss=2.0):
    learner = ComposedLearner()
    learner.base_calls = 0
    learner.base_loss = torch.tensor(base_loss)
    learner.metrics = RecordingMetrics()
    learner._device = torch.device("cpu")
    learner._kickstart_coeffs = {}
    learner._kickstart_teachers = {}
    learner._kickstart_schedulers = LambdaDefaultDict(
        lambda module_id: Scheduler(0.0, framework="torch")
    )
    if teacher_logits is not None:
        learner._kickstart_teachers["policy"] = FakeTeacher(teacher_logits)
    return learner


def make_config(**kwargs):
    return SimpleNamespace(learner_config_dict=dict(kwargs))


def make_batch(masked_last=False):
    mask = torch.ones(1, 2, dtype=torch.bool)
    if masked_last:
        mask[0, 1] = False
    return {Columns.OBS: torch.zeros(1, 2, 4), Columns.LOSS_MASK: mask}


def test_masked_forward_kl_matches_manual_computation():
    student = torch.tensor([[0.0, 2.0], [1.0, 1.0]])
    teacher = torch.tensor([[2.0, 0.0], [1.0, 1.0]])
    p_t = torch.softmax(teacher, dim=-1)
    expected = (
        p_t
        * (torch.log_softmax(teacher, dim=-1) - torch.log_softmax(student, dim=-1))
    ).sum(dim=-1)
    assert torch.allclose(masked_forward_kl(student, teacher), expected.mean())


def test_masked_forward_kl_excludes_invalid_steps():
    student = torch.tensor([[[0.0, 2.0], [5.0, -5.0]]])
    teacher = torch.tensor([[[2.0, 0.0], [-5.0, 5.0]]])
    mask = torch.tensor([[True, False]])
    only_first = masked_forward_kl(student, teacher, mask=mask)
    manual = masked_forward_kl(student[:, :1], teacher[:, :1])
    assert torch.allclose(only_first, manual)


def test_kickstart_adds_forward_kl_to_base_loss():
    student_logits = torch.tensor([[[0.0, 2.0], [1.0, 1.0]]], requires_grad=True)
    teacher_logits = torch.tensor([[[2.0, 0.0], [1.0, 1.0]]])
    learner = make_learner(teacher_logits)
    config = make_config(**{COEFF_KEY: 1.0})

    total = learner.compute_loss_for_module(
        module_id="policy",
        config=config,
        batch=make_batch(),
        fwd_out={Columns.ACTION_DIST_INPUTS: student_logits},
    )

    expected = masked_forward_kl(student_logits, teacher_logits)
    assert torch.allclose(total, torch.tensor(2.0) + expected)
    assert learner.base_calls == 1
    total.backward()
    assert student_logits.grad is not None


def test_zero_coeff_fast_path_skips_teacher_and_metrics():
    learner = make_learner()
    total = learner.compute_loss_for_module(
        module_id="policy",
        config=make_config(**{COEFF_KEY: 0.0}),
        batch=make_batch(),
        fwd_out={Columns.ACTION_DIST_INPUTS: torch.zeros(1, 2, 2)},
    )
    assert total is learner.base_loss
    assert learner.base_calls == 1
    assert learner.metrics.values == []
    assert learner._kickstart_teachers == {}


def test_active_loss_requires_teacher_checkpoint():
    learner = make_learner()
    with pytest.raises(ValueError, match="teacher_checkpoint"):
        learner.compute_loss_for_module(
            module_id="policy",
            config=make_config(**{COEFF_KEY: 1.0}),
            batch=make_batch(),
            fwd_out={Columns.ACTION_DIST_INPUTS: torch.zeros(1, 2, 2)},
        )


def test_schedule_updates_coefficient_on_lifetime_env_steps():
    learner = make_learner()
    schedule = [[0, 1.0], [100, 0.0]]
    config = make_config(**{COEFF_KEY: schedule})
    learner._kickstart_schedulers = LambdaDefaultDict(
        lambda module_id: Scheduler(schedule, framework="torch")
    )
    learner.module = SimpleNamespace(_rl_modules={"policy": object()})
    learner.config = SimpleNamespace(get_config_for_module=lambda mid: config)

    learner.after_gradient_based_update(
        timesteps={NUM_ENV_STEPS_SAMPLED_LIFETIME: 50}
    )
    assert learner._kickstart_coeffs["policy"] == pytest.approx(0.5)

    learner.after_gradient_based_update(
        timesteps={NUM_ENV_STEPS_SAMPLED_LIFETIME: 200}
    )
    assert learner._kickstart_coeffs["policy"] == pytest.approx(0.0)


def test_annealed_coefficient_reaches_zero_and_stops_distilling():
    student_logits = torch.zeros(1, 2, 2, requires_grad=True)
    teacher = FakeTeacher(torch.ones(1, 2, 2))
    learner = make_learner()
    learner._kickstart_teachers["policy"] = teacher
    learner._kickstart_coeffs["policy"] = 0.0

    total = learner.compute_loss_for_module(
        module_id="policy",
        config=make_config(**{COEFF_KEY: [[0, 1.0], [100, 0.0]]}),
        batch=make_batch(),
        fwd_out={Columns.ACTION_DIST_INPUTS: student_logits},
    )
    assert total is learner.base_loss
    assert teacher.forward_calls == 0
