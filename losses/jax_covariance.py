"""Covariance regularization for JAX representations."""

import jax
import jax.numpy as jnp


def off_diagonal_covariance_loss(activations: jax.Array) -> jax.Array:
    """Sum squared off-diagonal sample covariances per feature.

    The last axis indexes features. All leading axes are pooled as samples,
    so callers should pass only positions that contribute to the objective.
    The sample covariance divides by the number of rows minus one.
    """
    if activations.ndim < 2 or activations.shape[-1] == 0:
        raise ValueError("activations need sample axes and a nonempty feature axis")
    width = activations.shape[-1]
    rows = activations.reshape(-1, width)
    if rows.shape[0] < 2:
        raise ValueError("sample covariance requires at least two rows")

    rows = rows.astype(jnp.result_type(rows.dtype, jnp.float32))
    centered = rows - rows.mean(axis=0)
    covariance = centered.T @ centered / (rows.shape[0] - 1)
    diagonal = jnp.arange(width)
    off_diagonal = covariance.at[diagonal, diagonal].set(0)
    return jnp.square(off_diagonal).sum() / width
