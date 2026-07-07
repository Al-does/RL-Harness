"""Tests for the custom RLModules.

The load-bearing property: the transformer's TRAIN-TIME chunked forward
(context-buffer state + T-step chunk, banded causal attention, RoPE) must
compute exactly the same embeddings as the step-by-step ROLLOUT forward.
If this holds, the policy gradient is computed for the same function the
agent executes, and probe activations can be collected either way.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch

from envs.mess3.rlmodules import (
    AUX_LOGITS,
    Mess3MLPRLModule,
    Mess3TransformerRLModule,
)
from ray.rllib.core.columns import Columns

OBS_DIM = 5
K = 8  # small context for the test


def make_module(action_space, obs_dim=OBS_DIM, context_len=K):
    obs_space = gym.spaces.Box(-5.0, 5.0, shape=(obs_dim,), dtype=np.float32)
    return Mess3TransformerRLModule(
        observation_space=obs_space,
        action_space=action_space,
        model_config={
            "context_len": context_len,
            "d_model": 32,
            "n_layers": 2,
            "n_heads": 2,
            "max_seq_len": 5,
        },
    )


def rollout_embeddings(module, obs_seq):
    """Step-by-step embeddings, threading the recurrent state."""
    state = {
        k: torch.from_numpy(v).unsqueeze(0) for k, v in module.get_initial_state().items()
    }
    embs, states = [], [state]
    for t in range(obs_seq.shape[0]):
        emb, state = module.encode_step(obs_seq[t].unsqueeze(0), state)
        embs.append(emb[0])
        states.append(state)
    return torch.stack(embs), states


@pytest.mark.parametrize("total_len", [3, 13, 20])
def test_chunked_forward_matches_stepwise(total_len):
    torch.manual_seed(0)
    module = make_module(gym.spaces.Box(-5.0, 5.0, (2,), np.float32))
    module.eval()
    obs_seq = torch.randn(total_len, OBS_DIM)

    step_embs, states = rollout_embeddings(module, obs_seq)

    # Emulate the learner connector: chunk into max_seq_len pieces, STATE_IN
    # taken from the rollout state at each chunk boundary.
    T = 5
    with torch.no_grad():
        for start in range(0, total_len, T):
            chunk = obs_seq[start : start + T].unsqueeze(0)
            state_in = states[start]
            batch = {Columns.OBS: chunk, Columns.STATE_IN: state_in}
            out = module._forward_train(batch)
            emb = out[Columns.EMBEDDINGS][0]
            expect = step_embs[start : start + T]
            assert torch.allclose(emb, expect, atol=1e-5), (
                f"chunk at {start}: max err {(emb - expect).abs().max().item()}"
            )


def test_padding_rows_do_not_corrupt_valid_positions():
    """Zero-padding at the END of a chunk must not change earlier embeddings."""
    torch.manual_seed(1)
    module = make_module(gym.spaces.Box(-5.0, 5.0, (2,), np.float32))
    module.eval()
    obs = torch.randn(1, 3, OBS_DIM)
    padded = torch.cat([obs, torch.zeros(1, 2, OBS_DIM)], dim=1)
    state = {
        k: torch.from_numpy(v).unsqueeze(0) for k, v in module.get_initial_state().items()
    }
    with torch.no_grad():
        e1 = module._forward_train({Columns.OBS: obs, Columns.STATE_IN: state})[
            Columns.EMBEDDINGS
        ]
        e2 = module._forward_train({Columns.OBS: padded, Columns.STATE_IN: state})[
            Columns.EMBEDDINGS
        ]
    assert torch.allclose(e1[0], e2[0, :3], atol=1e-5)
    assert torch.isfinite(e2).all()


def test_receptive_field_is_n_layers_times_context_len():
    """Observations more than n_layers*K steps back must not influence output."""
    torch.manual_seed(2)
    module = make_module(gym.spaces.Box(-5.0, 5.0, (2,), np.float32))
    module.eval()
    lookback = module.core.lookback  # n_layers * K = 16 here
    L = lookback + 4
    a = torch.randn(L, OBS_DIM)
    b = a.clone()
    b[0] += 100.0  # beyond the receptive field of the last position
    ea, _ = rollout_embeddings(module, a)
    eb, _ = rollout_embeddings(module, b)
    assert torch.allclose(ea[-1], eb[-1], atol=1e-5)
    assert not torch.allclose(ea[1], eb[1], atol=1e-3)  # nearby steps DO differ


def test_heads_and_outputs_continuous_and_discrete():
    torch.manual_seed(3)
    for space, dist_dim in [
        (gym.spaces.Box(-5.0, 5.0, (2,), np.float32), 4),  # mean(2) + log_std(2)
        (gym.spaces.Discrete(3), 3),
    ]:
        module = make_module(space)
        state = {
            k: torch.from_numpy(v).unsqueeze(0).repeat(2, *([1] * v.ndim))
            for k, v in module.get_initial_state().items()
        }
        batch = {Columns.OBS: torch.randn(2, 5, OBS_DIM), Columns.STATE_IN: state}
        out = module._forward_train(batch)
        assert out[Columns.ACTION_DIST_INPUTS].shape == (2, 5, dist_dim)
        assert out[AUX_LOGITS].shape == (2, 5, 3)
        vals = module.compute_values(batch)
        assert vals.shape == (2, 5)


def test_mlp_module_basic():
    torch.manual_seed(4)
    module = Mess3MLPRLModule(
        observation_space=gym.spaces.Box(-5.0, 5.0, (3,), np.float32),
        action_space=gym.spaces.Discrete(3),
        model_config={"mlp_hidden": (32, 32)},
    )
    batch = {Columns.OBS: torch.randn(7, 3)}
    out = module._forward_train(batch)
    assert out[Columns.ACTION_DIST_INPUTS].shape == (7, 3)
    assert module.compute_values(batch).shape == (7,)
    assert not module.is_stateful()


def test_aux_gradients_reach_core():
    """The program requires verifying that aux-CE gradients reach the core."""
    torch.manual_seed(5)
    module = make_module(gym.spaces.Box(-5.0, 5.0, (2,), np.float32))
    obs = torch.zeros(2, 5, OBS_DIM)
    obs[:, :, 0] = 1.0  # populated token slots at every position
    state = {
        k: torch.from_numpy(v).unsqueeze(0).repeat(2, *([1] * v.ndim))
        for k, v in module.get_initial_state().items()
    }
    out = module._forward_train({Columns.OBS: obs, Columns.STATE_IN: state})
    from envs.mess3.learners import next_token_targets

    mask = torch.ones(2, 5, dtype=torch.bool)
    targets, valid = next_token_targets(obs, mask)
    logits = out[AUX_LOGITS][:, :-1, :]
    ce = torch.nn.functional.cross_entropy(logits[valid], targets[valid])
    ce.backward()
    g = module.core.inp.weight.grad
    assert g is not None and g.abs().max() > 0, "aux loss does not reach the core"
    # ...and the policy head is untouched by the aux loss.
    assert module._pi_mean.weight.grad is None


def test_aux_next_token_targets():
    from envs.mess3.learners import next_token_targets

    B, T = 2, 4
    obs = torch.zeros(B, T, OBS_DIM)
    # row 0: tokens 0,1,2,- ; row 1: -,2,0,1  (- = unpopulated slot)
    obs[0, 0, 0] = obs[0, 1, 1] = obs[0, 2, 2] = 1.0
    obs[1, 1, 2] = obs[1, 2, 0] = obs[1, 3, 1] = 1.0
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, 3] = False  # padded / artificial ts
    targets, valid = next_token_targets(obs, mask)
    # row 0: t=0 -> token@1 = 1 valid; t=1 -> token@2 = 2 valid; t=2 -> mask@3 False
    assert valid[0].tolist() == [True, True, False]
    assert targets[0, 0].item() == 1 and targets[0, 1].item() == 2
    # row 1: t=0 -> token@1 populated (2), valid; t=1 -> 0; t=2 -> 1
    assert valid[1].tolist() == [True, True, True]
    assert targets[1].tolist() == [2, 0, 1]
