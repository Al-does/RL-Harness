from __future__ import annotations

from typing import Any

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.env.single_agent_env_runner import SingleAgentEnvRunner
from ray.rllib.env.single_agent_episode import SingleAgentEpisode
from ray.rllib.utils.annotations import override


class FreshEpisodeSingleAgentEnvRunner(SingleAgentEnvRunner):
    """Complete-episode runner that seeds each vector environment only once.

    RLlib 2.56 marks episode-count sampling as needing an initial reset after
    every call. Its next reset therefore reapplies the worker seed and replays
    the same finite stochastic trajectories. This runner preserves the first
    deterministic reset after environment construction and lets all later
    resets continue each environment's random stream.
    """

    @override(SingleAgentEnvRunner)
    def __init__(self, *, config: AlgorithmConfig, **kwargs: Any):
        if config.batch_mode != "complete_episodes":
            raise ValueError(
                "FreshEpisodeSingleAgentEnvRunner requires "
                "batch_mode='complete_episodes'"
            )
        super().__init__(config=config, **kwargs)

    @override(SingleAgentEnvRunner)
    def make_env(self) -> None:
        super().make_env()
        self._seed_next_reset = True

    @override(SingleAgentEnvRunner)
    def _reset_envs(self, episodes, shared_data, explore):
        if not self._seed_next_reset:
            self._needs_initial_reset = False
        super()._reset_envs(episodes, shared_data, explore)
        self._seed_next_reset = False


class ContinuingSingleAgentEnvRunner(SingleAgentEnvRunner):
    """Opt-in timestep sampler for tasks that never terminate or truncate.

    RLlib 2.56 retains sampled chunks, including recurrent state outputs, in
    its private ongoing-episode metrics cache until termination. Discard only
    those metrics references after each sample, independently of metrics polls.
    Returned training chunks, active episode lookback, connectors, recurrent
    state, and step counters are unchanged. Completed-episode metrics and
    callbacks needing previous episode chunks are not supported by this runner.
    Finite or naturally terminating tasks must use the standard runner instead.
    """

    @override(SingleAgentEnvRunner)
    def __init__(self, *, config: AlgorithmConfig, **kwargs: Any):
        if config.batch_mode != "truncate_episodes":
            raise ValueError("ContinuingSingleAgentEnvRunner requires batch_mode='truncate_episodes'")
        super().__init__(config=config, **kwargs)

    @override(SingleAgentEnvRunner)
    def sample(
        self,
        *,
        num_timesteps: int | None = None,
        num_episodes: int | None = None,
        explore: bool | None = None,
        random_actions: bool = False,
        force_reset: bool = False,
    ) -> list[SingleAgentEpisode]:
        if num_episodes is not None:
            raise ValueError("ContinuingSingleAgentEnvRunner cannot sample num_episodes; use timesteps")
        try:
            samples = super().sample(
                num_timesteps=num_timesteps,
                explore=explore,
                random_actions=random_actions,
                force_reset=force_reset,
            )
            if any(episode.is_done for episode in samples):
                raise ValueError("ContinuingSingleAgentEnvRunner requires non-terminating environments without truncation")
            return samples
        finally:
            self._ongoing_episodes_for_metrics.clear()
