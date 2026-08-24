"""Pure factories and probability helpers for finite factored HMMs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib
from typing import Any

import numpy as np

from envs.hmm.model import HMMModel


def _row_stochastic(value: np.ndarray, *, name: str) -> np.ndarray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 3:
        raise ValueError(f"{name} must have three dimensions")
    if not np.isfinite(array).all() or (array < 0.0).any():
        raise ValueError(f"{name} must contain finite non-negative values")
    if not np.allclose(array.sum(axis=-1), 1.0, atol=1e-12):
        raise ValueError(f"each row of {name} must sum to one")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class FactorCoupling:
    """A directed parent-to-child dependency in the next-state dynamics.

    ``transition_matrices[p]`` is the child's transition matrix when the
    parent's next state is ``p``. ``strength`` linearly interpolates between
    the child's own transition matrix (zero) and these conditional matrices
    (one).
    """

    parent: int
    child: int
    transition_matrices: np.ndarray
    strength: float = 1.0

    def __post_init__(self) -> None:
        if self.parent < 0 or self.child < 0:
            raise ValueError("coupling factor indices must be non-negative")
        if self.parent == self.child:
            raise ValueError("a factor cannot be coupled to itself")
        if not 0.0 <= float(self.strength) <= 1.0:
            raise ValueError("coupling strength must lie in [0, 1]")
        matrices = _row_stochastic(
            self.transition_matrices,
            name="coupling transition_matrices",
        )
        object.__setattr__(self, "transition_matrices", matrices)
        object.__setattr__(self, "strength", float(self.strength))

    @classmethod
    def from_value(
        cls,
        value: Mapping[str, Any] | FactorCoupling,
    ) -> FactorCoupling:
        if isinstance(value, cls):
            return value
        return cls(**dict(value))


def cartesian_token_map(token_sizes: Sequence[int]) -> np.ndarray:
    """Map factor-token tuples to a contiguous mixed-radix token index."""

    sizes = tuple(int(size) for size in token_sizes)
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("token_sizes must contain positive cardinalities")
    return np.arange(int(np.prod(sizes)), dtype=np.int64).reshape(sizes)


def _validated_token_map(
    token_map: np.ndarray | Sequence[Any] | None,
    *,
    token_sizes: tuple[int, ...],
) -> tuple[np.ndarray, bool]:
    is_cartesian = token_map is None
    mapping = (
        cartesian_token_map(token_sizes)
        if token_map is None
        else np.asarray(token_map)
    )
    if mapping.shape != token_sizes:
        raise ValueError(
            f"token_map must have shape {token_sizes}, got {mapping.shape}"
        )
    if not np.issubdtype(mapping.dtype, np.integer):
        raise ValueError("token_map must contain integer token indices")
    mapping = np.asarray(mapping, dtype=np.int64)
    if (mapping < 0).any():
        raise ValueError("token_map indices must be non-negative")
    used = np.unique(mapping)
    if not np.array_equal(used, np.arange(len(used))):
        raise ValueError("token_map indices must be contiguous from zero")
    return mapping, is_cartesian


def _component_factory(specification: Mapping[str, Any]) -> HMMModel:
    values = dict(specification)
    try:
        path = values.pop("factory")
    except KeyError as error:
        raise ValueError("each factor specification requires 'factory'") from error
    kwargs = dict(values.pop("kwargs", {}))
    if values:
        raise ValueError(f"unknown factor fields: {sorted(values)}")
    if not isinstance(path, str) or ":" not in path:
        raise ValueError("factor factory paths must use 'package.module:Symbol'")
    module_name, qualified_name = path.split(":", 1)
    value: Any = importlib.import_module(module_name)
    for part in qualified_name.split("."):
        value = getattr(value, part)
    model = value(**kwargs)
    if not isinstance(model, HMMModel):
        raise TypeError("each factor factory must return HMMModel")
    return model


def _state_labels(factors: tuple[HMMModel, ...]) -> tuple[str, ...]:
    labels = [
        (
            factor.state_labels
            if factor.state_labels is not None
            else tuple(str(index) for index in range(factor.n_states))
        )
        for factor in factors
    ]
    return tuple("|".join(parts) for parts in np.array(np.meshgrid(
        *labels,
        indexing="ij",
    )).reshape(len(factors), -1).T)


def _token_labels(
    factors: tuple[HMMModel, ...],
    *,
    mapping: np.ndarray,
    is_cartesian: bool,
) -> tuple[str, ...]:
    n_tokens = int(mapping.max()) + 1
    if not is_cartesian:
        return tuple(f"token_{index}" for index in range(n_tokens))
    labels = [
        (
            factor.token_labels
            if factor.token_labels is not None
            else tuple(str(index) for index in range(factor.n_tokens))
        )
        for factor in factors
    ]
    return tuple("|".join(parts) for parts in np.array(np.meshgrid(
        *labels,
        indexing="ij",
    )).reshape(len(factors), -1).T)


def _joint_transition(
    factors: tuple[HMMModel, ...],
    couplings: tuple[FactorCoupling, ...],
) -> np.ndarray:
    if not couplings:
        transition = factors[0].transition_matrix
        for factor in factors[1:]:
            transition = np.kron(transition, factor.transition_matrix)
        return transition

    by_child: dict[int, FactorCoupling] = {}
    for coupling in couplings:
        if coupling.parent >= len(factors) or coupling.child >= len(factors):
            raise ValueError("coupling factor index is out of range")
        if coupling.parent > coupling.child:
            raise ValueError(
                "couplings must follow factor order (parent index < child index)"
            )
        if coupling.child in by_child:
            raise ValueError("each factor may have at most one incoming coupling")
        expected = (
            factors[coupling.parent].n_states,
            factors[coupling.child].n_states,
            factors[coupling.child].n_states,
        )
        if coupling.transition_matrices.shape != expected:
            raise ValueError(
                "coupling transition_matrices must have shape "
                f"{expected}, got {coupling.transition_matrices.shape}"
            )
        by_child[coupling.child] = coupling

    state_sizes = tuple(factor.n_states for factor in factors)
    state_tuples = tuple(np.ndindex(state_sizes))
    transition = np.empty(
        (len(state_tuples), len(state_tuples)),
        dtype=np.float64,
    )
    for source_index, source in enumerate(state_tuples):
        for destination_index, destination in enumerate(state_tuples):
            probability = 1.0
            for factor_index, factor in enumerate(factors):
                base_probability = factor.transition_matrix[
                    source[factor_index],
                    destination[factor_index],
                ]
                coupling = by_child.get(factor_index)
                if coupling is None:
                    factor_probability = base_probability
                else:
                    conditional_probability = coupling.transition_matrices[
                        destination[coupling.parent],
                        source[factor_index],
                        destination[factor_index],
                    ]
                    factor_probability = (
                        (1.0 - coupling.strength) * base_probability
                        + coupling.strength * conditional_probability
                    )
                probability *= factor_probability
            transition[source_index, destination_index] = probability
    return transition


def compose_hmm_factors(
    factors: Sequence[HMMModel],
    *,
    couplings: Sequence[Mapping[str, Any] | FactorCoupling] = (),
    token_map: np.ndarray | Sequence[Any] | None = None,
) -> HMMModel:
    """Compose finite HMM factors into one ordinary finite HMM.

    States use Cartesian-product order. Factor emissions are sampled
    independently given their states, then ``token_map`` deterministically
    combines their sub-tokens. By default every sub-token tuple gets its own
    observed token.

    Directed couplings alter only transition dynamics. A coupling conditions a
    child's transition on its parent's next state; coupling strength zero is
    exactly the independent Kronecker-product model.
    """

    factor_tuple = tuple(factors)
    if not factor_tuple:
        raise ValueError("at least one HMM factor is required")
    if not all(isinstance(factor, HMMModel) for factor in factor_tuple):
        raise TypeError("factors must contain only HMMModel instances")
    coupling_tuple = tuple(FactorCoupling.from_value(value) for value in couplings)
    token_sizes = tuple(factor.n_tokens for factor in factor_tuple)
    mapping, is_cartesian = _validated_token_map(
        token_map,
        token_sizes=token_sizes,
    )

    initial = factor_tuple[0].initial_distribution
    cartesian_emission = factor_tuple[0].emission_matrix
    for factor in factor_tuple[1:]:
        initial = np.kron(initial, factor.initial_distribution)
        cartesian_emission = np.kron(
            cartesian_emission,
            factor.emission_matrix,
        )
    emission = np.zeros(
        (len(initial), int(mapping.max()) + 1),
        dtype=np.float64,
    )
    for cartesian_index, observed_token in enumerate(mapping.reshape(-1)):
        emission[:, observed_token] += cartesian_emission[:, cartesian_index]

    return HMMModel(
        initial_distribution=initial,
        transition_matrix=_joint_transition(factor_tuple, coupling_tuple),
        emission_matrix=emission,
        state_labels=_state_labels(factor_tuple),
        token_labels=_token_labels(
            factor_tuple,
            mapping=mapping,
            is_cartesian=is_cartesian,
        ),
    )


def factored_model(
    *,
    factors: Sequence[Mapping[str, Any]],
    couplings: Sequence[Mapping[str, Any] | FactorCoupling] = (),
    token_map: np.ndarray | Sequence[Any] | None = None,
) -> HMMModel:
    """RLlib-serializable import-path factory for :func:`compose_hmm_factors`."""

    return compose_hmm_factors(
        [_component_factory(specification) for specification in factors],
        couplings=couplings,
        token_map=token_map,
    )


def factor_marginals(
    joint_distribution: np.ndarray,
    factor_sizes: Sequence[int],
) -> tuple[np.ndarray, ...]:
    """Marginalize a joint distribution over each Cartesian-product factor."""

    values = np.asarray(joint_distribution, dtype=np.float64)
    sizes = tuple(int(size) for size in factor_sizes)
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("factor_sizes must contain positive cardinalities")
    if values.shape[-1] != int(np.prod(sizes)):
        raise ValueError("joint_distribution does not match factor_sizes")
    grid = values.reshape(*values.shape[:-1], *sizes)
    prefix_axes = len(values.shape) - 1
    return tuple(
        grid.sum(
            axis=tuple(
                prefix_axes + other
                for other in range(len(sizes))
                if other != factor
            )
        )
        for factor in range(len(sizes))
    )


def product_distribution(marginals: Sequence[np.ndarray]) -> np.ndarray:
    """Return the Cartesian product distribution of aligned factor marginals."""

    values = tuple(np.asarray(value, dtype=np.float64) for value in marginals)
    if not values:
        raise ValueError("at least one marginal is required")
    prefix = values[0].shape[:-1]
    if any(value.ndim < 1 or value.shape[:-1] != prefix for value in values):
        raise ValueError("all marginals must have the same leading shape")
    product = values[0]
    for value in values[1:]:
        product = (
            product[..., :, None] * value[..., None, :]
        ).reshape(*prefix, product.shape[-1] * value.shape[-1])
    return product
