# RLLibHarnesBeta

Local RLlib harness for custom RL experiments.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13+.

```bash
uv sync
```

## Verify

```bash
uv run python -c "import ray; from ray.rllib.algorithms.ppo import PPOConfig; print('RLlib', ray.__version__, 'ready')"
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
