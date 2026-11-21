import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp

from losses.jax_covariance import off_diagonal_covariance_loss


def test_matches_sample_covariance_over_all_leading_axes():
    rows = np.array([[0, 1, 2], [1, 2, 2], [2, 8, 2], [3, 4, 0]], dtype=np.float32)
    covariance = np.cov(rows, rowvar=False)
    np.fill_diagonal(covariance, 0)
    expected = np.square(covariance).sum() / rows.shape[-1]
    actual = off_diagonal_covariance_loss(jnp.asarray(rows.reshape(2, 2, 3)))
    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_uncorrelated_dimensions_can_have_nonzero_variance():
    rows = jnp.array([[1, 3], [-1, 3], [1, -3], [-1, -3]], dtype=jnp.float32)
    assert off_diagonal_covariance_loss(rows) == 0


def test_jit_gradient_is_finite_and_translation_invariant():
    rows = jnp.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    value, gradient = jax.jit(jax.value_and_grad(off_diagonal_covariance_loss))(rows)
    assert value == 1
    np.testing.assert_allclose(gradient, [[-1, -1], [0, 0], [1, 1]])
    np.testing.assert_allclose(np.asarray(gradient).sum(axis=0), 0)


@pytest.mark.parametrize("shape", [(3,), (2, 0), (0, 2), (1, 2)])
def test_rejects_inputs_without_sample_covariance(shape):
    with pytest.raises(ValueError):
        off_diagonal_covariance_loss(jnp.zeros(shape))
