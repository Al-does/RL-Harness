"""Unit and composition tests for configurable torch optimizers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from learners.optimizer import (
    AUX_KWARGS_KEY,
    KWARGS_KEY,
    MUON_AUX_NAME,
    MUON_NAME,
    TYPE_KEY,
    ConfigurableOptimizerMixin,
    build_torch_optimizer,
    partition_muon_params,
)


class TinyModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)


def test_build_torch_optimizer_builtin_names():
    cases = {
        "adam": torch.optim.Adam,
        "AdamW": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
    }
    for name, expected_cls in cases.items():
        module = TinyModule()
        opt = build_torch_optimizer(module.parameters(), name_or_cls=name)
        assert isinstance(opt, expected_cls)

    # Muon only accepts 2D parameters.
    module = TinyModule()
    weight = module.linear.weight
    opt = build_torch_optimizer([weight], name_or_cls="muon")
    assert isinstance(opt, torch.optim.Muon)


def test_build_torch_optimizer_muon_rejects_non_2d_with_clear_error():
    module = TinyModule()
    with pytest.raises(ValueError, match="only accepts 2D"):
        build_torch_optimizer(module.parameters(), name_or_cls="muon")


def test_partition_muon_params():
    module = TinyModule()
    muon_params, other_params = partition_muon_params(module.parameters())
    assert len(muon_params) == 1
    assert muon_params[0].ndim == 2
    assert len(other_params) == 1
    assert other_params[0].ndim == 1


def test_build_torch_optimizer_forwards_kwargs():
    module = TinyModule()
    opt = build_torch_optimizer(
        module.parameters(),
        name_or_cls="adamw",
        kwargs={"weight_decay": 0.05, "lr": 1e-3},
    )
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["weight_decay"] == 0.05
    assert opt.param_groups[0]["lr"] == 1e-3


def test_build_torch_optimizer_accepts_class():
    module = TinyModule()
    opt = build_torch_optimizer(
        module.parameters(),
        name_or_cls=torch.optim.SGD,
        kwargs={"lr": 0.1, "momentum": 0.9},
    )
    assert isinstance(opt, torch.optim.SGD)
    assert opt.param_groups[0]["momentum"] == 0.9


def test_build_torch_optimizer_accepts_factory():
    module = TinyModule()
    created = []

    def factory(params):
        opt = torch.optim.Adam(params, lr=2e-4)
        created.append(opt)
        return opt

    opt = build_torch_optimizer(
        module.parameters(),
        name_or_cls=factory,
        kwargs={"weight_decay": 1.0},  # ignored for callables
    )
    assert opt is created[0]
    assert opt.param_groups[0]["lr"] == 2e-4


def test_build_torch_optimizer_unknown_name():
    module = TinyModule()
    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_torch_optimizer(module.parameters(), name_or_cls="lion")


class RecordingLearner(ConfigurableOptimizerMixin):
    """Minimal stand-in that records register_optimizer calls."""

    def __init__(self, module: nn.Module):
        self._module = {"default_policy": module}
        self.registrations = []

    def get_parameters(self, module):
        return list(module.parameters())

    def register_optimizer(self, **kwargs):
        self.registrations.append(kwargs)


def test_configurable_optimizer_mixin_defaults_to_adam():
    module = TinyModule()
    learner = RecordingLearner(module)
    config = SimpleNamespace(lr=3e-4, learner_config_dict={})

    learner.configure_optimizers_for_module("default_policy", config)

    assert len(learner.registrations) == 1
    reg = learner.registrations[0]
    assert isinstance(reg["optimizer"], torch.optim.Adam)
    assert reg["lr_or_lr_schedule"] == 3e-4
    assert reg["module_id"] == "default_policy"


def test_configurable_optimizer_mixin_reads_namespaced_config():
    module = TinyModule()
    learner = RecordingLearner(module)
    config = SimpleNamespace(
        lr=1e-3,
        learner_config_dict={
            TYPE_KEY: "sgd",
            KWARGS_KEY: {"momentum": 0.9},
        },
    )

    learner.configure_optimizers_for_module("default_policy", config)

    opt = learner.registrations[0]["optimizer"]
    assert isinstance(opt, torch.optim.SGD)
    assert opt.param_groups[0]["momentum"] == 0.9
    assert learner.registrations[0]["lr_or_lr_schedule"] == 1e-3


def test_configurable_optimizer_mixin_rejects_non_dict_kwargs():
    module = TinyModule()
    learner = RecordingLearner(module)
    config = SimpleNamespace(
        lr=1e-3,
        learner_config_dict={TYPE_KEY: "adam", KWARGS_KEY: ["bad"]},
    )

    with pytest.raises(TypeError, match="must be a dict"):
        learner.configure_optimizers_for_module("default_policy", config)


def test_configurable_optimizer_mixin_registers_muon_and_adamw():
    module = TinyModule()
    learner = RecordingLearner(module)
    config = SimpleNamespace(
        lr=2e-3,
        learner_config_dict={
            TYPE_KEY: "muon",
            KWARGS_KEY: {"momentum": 0.9},
            AUX_KWARGS_KEY: {"weight_decay": 0.01},
        },
    )

    learner.configure_optimizers_for_module("default_policy", config)

    assert len(learner.registrations) == 2
    by_name = {reg["optimizer_name"]: reg for reg in learner.registrations}
    assert isinstance(by_name[MUON_NAME]["optimizer"], torch.optim.Muon)
    assert by_name[MUON_NAME]["optimizer"].param_groups[0]["momentum"] == 0.9
    assert isinstance(by_name[MUON_AUX_NAME]["optimizer"], torch.optim.AdamW)
    assert by_name[MUON_AUX_NAME]["optimizer"].param_groups[0]["weight_decay"] == 0.01
    assert by_name[MUON_NAME]["lr_or_lr_schedule"] == 2e-3
    assert by_name[MUON_AUX_NAME]["lr_or_lr_schedule"] == 2e-3
