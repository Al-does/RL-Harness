import numpy as np
import pytest

from envs.rocksample import (
    Action,
    Observation,
    RockSampleEnv,
    configuration_bits,
    instance_definition,
    joint_from_marginals,
    update_joint,
    update_marginals,
)


def test_worked_example_and_configuration_indexing():
    instance = instance_definition(4, 4, layout_seed=0)
    m = np.full(4, 0.5)
    j = np.full(16, 1 / 16)
    events = [
        ((0, 2), 7, Observation.GOOD),
        ((0, 2), 7, Observation.GOOD),
        ((0, 2), Action.NORTH, Observation.NONE),
        ((0, 3), Action.EAST, Observation.NONE),
        ((1, 3), 7, Observation.GOOD),
        ((1, 3), Action.SAMPLE, Observation.NONE),
    ]
    eta = np.exp(-np.sqrt(2))
    once = (1 + eta) / 2
    twice = once**2 / (once**2 + (1 - once) ** 2)
    for event, expected in zip(events, [once, twice, twice, twice, 1, 0], strict=True):
        position, action, symbol = event
        kwargs = dict(
            rocks=instance.rocks,
            position=position,
            action=action,
            symbol=symbol,
            half_efficiency_distance=instance.sensor_half_efficiency_distance,
        )
        m = update_marginals(m, **kwargs)
        j = update_joint(j, **kwargs)
        assert m[2] == pytest.approx(expected)
        np.testing.assert_array_equal(m[[0, 1, 3]], 0.5)
        np.testing.assert_allclose(j, joint_from_marginals(m), atol=1e-15)
    assert np.all(j[configuration_bits(4)[:, 2]] == 0)
    np.testing.assert_array_equal(
        configuration_bits(2), [[0, 0], [1, 0], [0, 1], [1, 1]]
    )


@pytest.mark.parametrize("random_layout", [False, True])
def test_trajectory_factorization_and_empirical_calibration(random_layout):
    env = RockSampleEnv({"diagnostics": True, "randomize_train_layout": random_layout})
    rng = np.random.default_rng(441)
    instance = env.instance
    bins = np.zeros((10, 4))
    sensor = np.zeros(4)
    for episode in range(250):
        _, info = env.reset(seed=episode)
        m = np.full(7, 0.5)
        j = np.full(128, 1 / 128)
        sampled = np.zeros(7, bool)
        calibration_step = int(rng.integers(10))
        for step in range(100):
            if step == calibration_step:
                for rock in np.flatnonzero(~sampled):
                    b = min(int(m[rock] * 10), 9)
                    bins[b] += [
                        1,
                        m[rock],
                        info["rock_qualities"][rock],
                        m[rock] * (1 - m[rock]),
                    ]
            action = int(rng.integers(12))
            position = info["rover_position"]
            rocks = info["rock_positions"]
            _, _, terminated, truncated, next_info = env.step(action)
            if terminated:
                break
            if action >= 5:
                rock = action - 5
                eta = 2 ** (
                    -np.linalg.norm(position - rocks[rock])
                    / instance.sensor_half_efficiency_distance
                )
                probability = (1 - eta) / 2 + m[rock] * eta
                if step == calibration_step:
                    sensor += [
                        1,
                        probability,
                        next_info["observation_symbol"] == Observation.GOOD,
                        probability * (1 - probability),
                    ]
            if action == Action.SAMPLE:
                sampled |= np.all(rocks == position, axis=1)
            kwargs = dict(
                rocks=rocks,
                position=position,
                action=action,
                symbol=next_info["observation_symbol"],
                half_efficiency_distance=instance.sensor_half_efficiency_distance,
            )
            m = update_marginals(m, **kwargs)
            j = update_joint(j, **kwargs)
            np.testing.assert_allclose(j, joint_from_marginals(m), atol=1e-10)
            assert j.sum() == pytest.approx(1)
            assert np.all((m >= 0) & (m <= 1))
            info = next_info
            if truncated:
                break
    covered = bins[:, 0] > 20
    assert np.all(
        np.abs(bins[covered, 2] - bins[covered, 1]) < 4 * np.sqrt(bins[covered, 3]) + 1
    )
    assert abs(sensor[2] - sensor[1]) < 4 * np.sqrt(sensor[3]) + 1


def test_invariance_sample_and_correlated_joint():
    kwargs = dict(
        rocks=np.array([[0, 0], [1, 1]]),
        position=(0, 1),
        symbol=Observation.NONE,
        half_efficiency_distance=20,
    )
    m = np.array([0.2, 0.7])
    for action in [Action.NORTH, Action.SOUTH, Action.WEST, Action.EAST, Action.SAMPLE]:
        np.testing.assert_array_equal(update_marginals(m, action=action, **kwargs), m)
    j = np.array([0.4, 0.1, 0.2, 0.3])
    kwargs["position"] = (0, 0)
    sampled = update_joint(j, action=Action.SAMPLE, **kwargs)
    np.testing.assert_allclose(sampled, [0.5, 0, 0.5, 0])
    kwargs["symbol"] = Observation.GOOD
    checked = update_joint(j, action=5, **kwargs)
    np.testing.assert_allclose(checked, [0, 0.25, 0, 0.75])


def test_impossible_reading_and_invalid_inputs():
    kwargs = dict(
        rocks=np.array([[0, 0]]),
        position=(0, 0),
        action=5,
        symbol=Observation.GOOD,
        half_efficiency_distance=20,
    )
    with pytest.raises(ValueError, match="zero probability"):
        update_marginals(np.array([0.0]), **kwargs)
    with pytest.raises(ValueError, match="zero probability"):
        update_joint(np.array([1.0, 0.0]), **kwargs)
    kwargs["symbol"] = Observation.NONE
    with pytest.raises(ValueError, match="Check"):
        update_marginals(np.array([0.5]), **kwargs)
    with pytest.raises(ValueError):
        joint_from_marginals(np.array([1.1]))
