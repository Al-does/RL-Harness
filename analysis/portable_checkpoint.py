"""Lightweight portable checkpoint format for module-only offline analysis.

This format captures enough information to restore an RLModule without starting
Ray, EnvironmentRunners, or a full RLlib Algorithm.
"""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator


PORTABLE_CHECKPOINT_DIRNAME = "portable_checkpoint"
PORTABLE_MANIFEST_NAME = "portable_manifest.json"
MODULE_STATE_NAME = "module_state.pt"
SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class PortableCheckpointSpec:
    schema_version: int
    module_class: str
    module_config: dict[str, Any]
    environment_specification: dict[str, Any]
    checkpoint_step: int | None
    experiment_sha: str | None
    harness_sha: str | None
    analysis_protocol: dict[str, Any]
    source_checkpoint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PortableCheckpointSpec":
        return cls(
            schema_version=int(payload["schema_version"]),
            module_class=str(payload["module_class"]),
            module_config=dict(payload.get("module_config") or {}),
            environment_specification=dict(
                payload.get("environment_specification") or {}
            ),
            checkpoint_step=(
                None
                if payload.get("checkpoint_step") is None
                else int(payload["checkpoint_step"])
            ),
            experiment_sha=payload.get("experiment_sha"),
            harness_sha=payload.get("harness_sha"),
            analysis_protocol=dict(payload.get("analysis_protocol") or {}),
            source_checkpoint=payload.get("source_checkpoint"),
        )


def _qualname(obj: Any) -> str:
    cls = obj if isinstance(obj, type) else type(obj)
    return f"{cls.__module__}:{cls.__qualname__}"


def _module_config_dict(module: Any) -> dict[str, Any]:
    config = getattr(module, "config", None)
    if config is None:
        return {}
    if hasattr(config, "to_dict"):
        try:
            return dict(config.to_dict())
        except Exception:  # noqa: BLE001
            pass
    if isinstance(config, dict):
        return dict(config)
    return {"repr": repr(config)}


def write_portable_checkpoint(
    destination: Path,
    *,
    module: Any,
    environment_specification: dict[str, Any],
    checkpoint_step: int | None = None,
    experiment_sha: str | None = None,
    harness_sha: str | None = None,
    analysis_protocol: dict[str, Any] | None = None,
    source_checkpoint: str | Path | None = None,
) -> Path:
    """Serialize module class/config/state plus analysis provenance."""
    import torch

    destination.mkdir(parents=True, exist_ok=True)
    state_path = destination / MODULE_STATE_NAME
    state = module.get_state() if hasattr(module, "get_state") else module.state_dict()
    torch.save(state, state_path)
    spec = PortableCheckpointSpec(
        schema_version=SCHEMA_VERSION,
        module_class=_qualname(module),
        module_config=_module_config_dict(module),
        environment_specification=dict(environment_specification),
        checkpoint_step=checkpoint_step,
        experiment_sha=experiment_sha,
        harness_sha=harness_sha,
        analysis_protocol=dict(analysis_protocol or {}),
        source_checkpoint=(
            None if source_checkpoint is None else str(source_checkpoint)
        ),
    )
    manifest_path = destination / PORTABLE_MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(spec.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    return destination


def read_portable_manifest(path: Path) -> PortableCheckpointSpec:
    payload = json.loads((path / PORTABLE_MANIFEST_NAME).read_text())
    if not isinstance(payload, dict):
        raise ValueError("portable manifest must be a JSON object")
    return PortableCheckpointSpec.from_dict(payload)


def _load_class(qualname: str) -> type:
    module_name, _, class_name = qualname.partition(":")
    if not module_name or not class_name:
        raise ValueError(f"invalid module class {qualname!r}")
    import importlib

    module = importlib.import_module(module_name)
    attr: Any = module
    for part in class_name.split("."):
        attr = getattr(attr, part)
    if not isinstance(attr, type):
        raise TypeError(f"{qualname!r} did not resolve to a class")
    return attr


@contextmanager
def load_portable_module(path: Path) -> Iterator[Any]:
    """Restore only the RLModule weights/config — never starts Ray."""
    import torch

    spec = read_portable_manifest(path)
    cls = _load_class(spec.module_class)
    config = spec.module_config
    module = None
    if hasattr(cls, "from_config"):
        module = cls.from_config(config)
    else:
        try:
            module = cls(config)
        except TypeError:
            module = cls()
    state = torch.load(path / MODULE_STATE_NAME, map_location="cpu", weights_only=False)
    if hasattr(module, "set_state"):
        module.set_state(state)
    elif hasattr(module, "load_state_dict"):
        module.load_state_dict(state)
    else:
        raise TypeError("restored module cannot load state")
    try:
        yield module
    finally:
        del module


def export_portable_from_algorithm_checkpoint(
    checkpoint: Path,
    destination: Path,
    *,
    module_id: str = "default_policy",
    environment_specification: dict[str, Any] | None = None,
    checkpoint_step: int | None = None,
    experiment_sha: str | None = None,
    harness_sha: str | None = None,
    analysis_protocol: dict[str, Any] | None = None,
) -> Path:
    """One-time conversion helper that may use Algorithm restore.

    Offline analysis after export must use ``load_portable_module``.
    """
    from analysis.checkpoints import load_algorithm

    with load_algorithm(checkpoint) as algorithm:
        module = algorithm.get_module(module_id)
        if module is None:
            raise KeyError(f"checkpoint has no RLModule id {module_id!r}")
        env_spec = dict(environment_specification or {})
        if not env_spec:
            config = algorithm.config
            env_spec = {
                "env": getattr(config, "env", None),
                "env_config": getattr(config, "env_config", None),
            }
        return write_portable_checkpoint(
            destination,
            module=module,
            environment_specification=env_spec,
            checkpoint_step=checkpoint_step,
            experiment_sha=experiment_sha,
            harness_sha=harness_sha,
            analysis_protocol=analysis_protocol,
            source_checkpoint=checkpoint,
        )


__all__ = [
    "PORTABLE_CHECKPOINT_DIRNAME",
    "PORTABLE_MANIFEST_NAME",
    "PortableCheckpointSpec",
    "export_portable_from_algorithm_checkpoint",
    "load_portable_module",
    "read_portable_manifest",
    "write_portable_checkpoint",
]
