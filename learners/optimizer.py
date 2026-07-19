"""Configurable torch optimizers for RLlib Learners and plain training loops."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch
from torch.optim import Optimizer

NAMESPACE = "optimizer"
TYPE_KEY = f"{NAMESPACE}/type"
KWARGS_KEY = f"{NAMESPACE}/kwargs"
AUX_KWARGS_KEY = f"{NAMESPACE}/aux_kwargs"

MUON_NAME = "muon"
MUON_AUX_NAME = "muon_aux"

_BUILTIN_OPTIMIZERS: dict[str, type[Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
    "rmsprop": torch.optim.RMSprop,
    "muon": torch.optim.Muon,
}

OptimizerFactory = Callable[[Iterable[torch.Tensor]], Optimizer]


def partition_muon_params(
    params: Iterable[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Split parameters into Muon-eligible (2D) and auxiliary (non-2D) groups."""
    param_list = list(params)
    muon_params = [p for p in param_list if p.ndim == 2]
    other_params = [p for p in param_list if p.ndim != 2]
    return muon_params, other_params


def build_torch_optimizer(
    params: Iterable[torch.Tensor],
    *,
    name_or_cls: str | type[Optimizer] | OptimizerFactory = "adam",
    kwargs: dict[str, Any] | None = None,
) -> Optimizer:
    """Construct a torch optimizer by name, class, or factory callable.

    Built-in string names (case-insensitive): ``adam``, ``adamw``, ``sgd``,
    ``rmsprop``, ``muon``. A subclass of :class:`torch.optim.Optimizer` is
    instantiated with ``params`` and ``**kwargs``. A callable is invoked as
    ``factory(params)`` and must return an optimizer (``kwargs`` are ignored).

    Note:
        ``torch.optim.Muon`` only accepts 2D parameters. For full modules that
        also have biases / norms, use :class:`ConfigurableOptimizerMixin` with
        ``optimizer/type="muon"``, which registers Muon for 2D weights and
        AdamW for the rest.
    """
    opt_kwargs = dict(kwargs or {})

    if isinstance(name_or_cls, str):
        key = name_or_cls.strip().lower()
        if key not in _BUILTIN_OPTIMIZERS:
            supported = ", ".join(sorted(_BUILTIN_OPTIMIZERS))
            raise ValueError(
                f"Unknown optimizer {name_or_cls!r}. "
                f"Supported names: {supported}."
            )
        try:
            return _BUILTIN_OPTIMIZERS[key](params, **opt_kwargs)
        except ValueError as exc:
            if key == "muon":
                raise ValueError(
                    "Muon only accepts 2D parameters. Pass 2D weights only, "
                    "or use ConfigurableOptimizerMixin with "
                    "optimizer/type='muon' to pair Muon (2D) with AdamW "
                    "(non-2D)."
                ) from exc
            raise

    if isinstance(name_or_cls, type) and issubclass(name_or_cls, Optimizer):
        return name_or_cls(params, **opt_kwargs)

    if callable(name_or_cls):
        return name_or_cls(params)

    raise TypeError(
        "name_or_cls must be a string name, Optimizer subclass, or "
        f"callable factory; got {type(name_or_cls)!r}."
    )


def _as_kwargs_dict(value: Any, *, key: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(
            f"{key!r} must be a dict of constructor kwargs; got {type(value)!r}."
        )
    return dict(value)


class ConfigurableOptimizerMixin:
    """Replace RLlib's default Adam with a namespaced optimizer choice.

    Reads from ``config.learner_config_dict``:

    - ``optimizer/type`` — string name, Optimizer subclass, or factory
      (default ``"adam"``)
    - ``optimizer/kwargs`` — constructor kwargs dict (default ``{}``)
    - ``optimizer/aux_kwargs`` — kwargs for the AdamW group used with
      ``optimizer/type="muon"`` (default ``{}``)

    When ``optimizer/type`` is ``"muon"``, registers Muon for 2D parameters and
    AdamW for non-2D parameters (biases, norms, embeddings), matching PyTorch's
    recommended usage.

    Does **not** call ``super().configure_optimizers_for_module`` (that would
    register a second Adam). Place this mixin before the algorithm Learner and
    after loss mixins in the MRO:

    .. code-block:: python

        class ExperimentLearner(
            NextTokenAuxLossMixin,
            ConfigurableOptimizerMixin,
            PPOTorchLearner,
        ):
            pass
    """

    def configure_optimizers_for_module(self, module_id, config=None) -> None:
        module = self._module[module_id]
        params = self.get_parameters(module)
        learner_cfg = getattr(config, "learner_config_dict", None) or {}
        name_or_cls = learner_cfg.get(TYPE_KEY, "adam")
        opt_kwargs = _as_kwargs_dict(
            learner_cfg.get(KWARGS_KEY, {}),
            key=KWARGS_KEY,
        )

        if isinstance(name_or_cls, str) and name_or_cls.strip().lower() == "muon":
            self._register_muon_optimizers(
                module_id=module_id,
                params=params,
                lr_or_lr_schedule=config.lr,
                muon_kwargs=opt_kwargs,
                aux_kwargs=_as_kwargs_dict(
                    learner_cfg.get(AUX_KWARGS_KEY, {}),
                    key=AUX_KWARGS_KEY,
                ),
            )
            return

        optimizer = build_torch_optimizer(
            params,
            name_or_cls=name_or_cls,
            kwargs=opt_kwargs,
        )
        self.register_optimizer(
            module_id=module_id,
            optimizer=optimizer,
            params=params,
            lr_or_lr_schedule=config.lr,
        )

    def _register_muon_optimizers(
        self,
        *,
        module_id,
        params: Sequence[torch.Tensor],
        lr_or_lr_schedule,
        muon_kwargs: dict[str, Any],
        aux_kwargs: dict[str, Any],
    ) -> None:
        muon_params, other_params = partition_muon_params(params)
        if not muon_params and not other_params:
            raise ValueError("No parameters available to optimize with Muon.")

        if muon_params:
            self.register_optimizer(
                module_id=module_id,
                optimizer_name=MUON_NAME,
                optimizer=torch.optim.Muon(muon_params, **muon_kwargs),
                params=muon_params,
                lr_or_lr_schedule=lr_or_lr_schedule,
            )
        if other_params:
            self.register_optimizer(
                module_id=module_id,
                optimizer_name=MUON_AUX_NAME,
                optimizer=torch.optim.AdamW(other_params, **aux_kwargs),
                params=other_params,
                lr_or_lr_schedule=lr_or_lr_schedule,
            )
