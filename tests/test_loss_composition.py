"""Unit tests for algorithm-agnostic Learner loss composition."""

from types import SimpleNamespace

import torch

from losses.next_token import NextTokenAuxLossMixin
from ray.rllib.core.columns import Columns

FWD_KEY = "next_token_aux/logits"
LAMBDA_KEY = "next_token_aux/lambda"


class RecordingMetrics:
    def __init__(self):
        self.calls = []

    def log_dict(self, values, *, key, window):
        self.calls.append((values, key, window))


class BaseLearner:
    def compute_loss_for_module(
        self, *, module_id, config, batch, fwd_out
    ):
        self.base_calls += 1
        return self.base_loss


class ComposedLearner(NextTokenAuxLossMixin, BaseLearner):
    pass


class AdditiveLossMixin:
    def compute_loss_for_module(
        self, *, module_id, config, batch, fwd_out
    ):
        return super().compute_loss_for_module(
            module_id=module_id,
            config=config,
            batch=batch,
            fwd_out=fwd_out,
        ) + 1.0


class StackedLearner(
    NextTokenAuxLossMixin, AdditiveLossMixin, BaseLearner
):
    pass


def make_learner(base_loss=2.0):
    learner = ComposedLearner()
    learner.base_calls = 0
    learner.base_loss = torch.tensor(base_loss)
    learner.metrics = RecordingMetrics()
    return learner


def make_batch():
    obs = torch.zeros(1, 3, 5)
    obs[0, 0, 0] = 1.0
    obs[0, 1, 1] = 1.0
    obs[0, 2, 2] = 1.0
    return {
        Columns.OBS: obs,
        Columns.LOSS_MASK: torch.ones(1, 3, dtype=torch.bool),
    }


def test_loss_mixin_preserves_base_loss_and_adds_namespaced_metrics():
    learner = make_learner()
    logits = torch.zeros(1, 3, 3, requires_grad=True)
    config = SimpleNamespace(
        learner_config_dict={LAMBDA_KEY: 0.5}
    )

    total = learner.compute_loss_for_module(
        module_id="policy",
        config=config,
        batch=make_batch(),
        fwd_out={FWD_KEY: logits},
    )

    expected_ce = torch.log(torch.tensor(3.0))
    assert torch.allclose(total, torch.tensor(2.0) + 0.5 * expected_ce)
    assert learner.base_calls == 1
    values, key, window = learner.metrics.calls[0]
    assert set(values) == {
        "next_token_aux/ce",
        "next_token_aux/accuracy",
    }
    assert key == "policy"
    assert window == 1

    total.backward()
    assert logits.grad is not None


def test_loss_mixin_fast_paths_still_call_super_first():
    learner = make_learner()
    batch = make_batch()

    zero_weight = learner.compute_loss_for_module(
        module_id="policy",
        config=SimpleNamespace(
            learner_config_dict={LAMBDA_KEY: 0.0}
        ),
        batch=batch,
        fwd_out={FWD_KEY: torch.zeros(1, 3, 3)},
    )
    missing_head = learner.compute_loss_for_module(
        module_id="policy",
        config=SimpleNamespace(
            learner_config_dict={LAMBDA_KEY: 1.0}
        ),
        batch=batch,
        fwd_out={},
    )

    assert zero_weight is learner.base_loss
    assert missing_head is learner.base_loss
    assert learner.base_calls == 2
    assert learner.metrics.calls == []


def test_loss_mixin_cooperates_with_another_loss_in_the_mro():
    learner = StackedLearner()
    learner.base_calls = 0
    learner.base_loss = torch.tensor(2.0)
    learner.metrics = RecordingMetrics()

    total = learner.compute_loss_for_module(
        module_id="policy",
        config=SimpleNamespace(
            learner_config_dict={LAMBDA_KEY: 0.0}
        ),
        batch=make_batch(),
        fwd_out={},
    )

    assert torch.equal(total, torch.tensor(3.0))
    assert learner.base_calls == 1


def test_loss_mixin_handles_an_all_invalid_mask_without_device_sync():
    learner = make_learner()
    batch = make_batch()
    batch[Columns.LOSS_MASK].zero_()
    logits = torch.randn(1, 3, 3, requires_grad=True)

    total = learner.compute_loss_for_module(
        module_id="policy",
        config=SimpleNamespace(
            learner_config_dict={LAMBDA_KEY: 1.0}
        ),
        batch=batch,
        fwd_out={FWD_KEY: logits},
    )

    assert torch.equal(total, learner.base_loss)
    assert torch.isfinite(total)
    values, _, _ = learner.metrics.calls[0]
    assert values["next_token_aux/ce"].item() == 0.0
    assert values["next_token_aux/accuracy"].item() == 0.0
