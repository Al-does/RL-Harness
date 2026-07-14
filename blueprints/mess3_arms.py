"""All experiment arms of the MESS3-Control program, one blueprint each.

Phase 2 (Environment B ladder), Phase 3 (Environment A main arms) and the
Phase 4 N-scramble arm are declared here so the experiment matrix is fixed
and reviewable from Phase 1 onward.  The Phase-1 operating point
(beta=4, w_max2=5, delay=1) is baked into ``A_ENV``; if review moves the
operating point, change it HERE only.

None of these launch anything: scripts/train.py refuses to run any blueprint
whose gate artifact (results/phase1/GATE_PASSED) is missing.
"""

from __future__ import annotations

from blueprints.base import Blueprint, ModelSpec, PPOSpec, register
from learners.models import (
    MLPModel,
    MLPModelConfig,
    TransformerModel,
    TransformerModelConfig,
    TransformerWithNextTokenAux,
    TransformerWithStateAux,
)
from learners.models.lstm import LSTMModel, LSTMModelConfig
from learners.ppo import PPOWithNextTokenAux

# Phase-1 recommended operating point (see results/phase1/FINDINGS_phase1.md).
OPERATING_POINT = {"beta": 4.0, "w_max2": 5.0, "delay": 1}

A_ENTRY = "envs.mess3.env_continuous:Mess3ContinuousEnv"
B_ENTRY = "envs.mess3.env_stateguess:StateGuessEnv"

A_ENV = dict(OPERATING_POINT, alpha=0.85, episode_length=1024)
B_ENV = dict(alpha=0.85, delay=1, episode_length=1024)

N_TOKENS = 3
N_STATES = 3

TRANSFORMER_CONFIG = TransformerModelConfig(
    d_model=96, n_layers=3, n_heads=4, context_len=64
)
TRANSFORMER = ModelSpec(
    model_class=TransformerModel,
    config=TRANSFORMER_CONFIG,
)
NEXT_TOKEN_TRANSFORMER = ModelSpec(
    model_class=TransformerWithNextTokenAux,
    config=TRANSFORMER_CONFIG,
    mixin_config={"next_token_aux": {"num_classes": N_TOKENS}},
)
STATE_TRANSFORMER = ModelSpec(
    model_class=TransformerWithStateAux,
    config=TRANSFORMER_CONFIG,
    mixin_config={"state_aux": {"num_classes": N_STATES}},
)
MLP = ModelSpec(
    model_class=MLPModel,
    config=MLPModelConfig(hidden_dims=(128, 128)),
)

# --- Phase 2: Environment B ladder ------------------------------------------

register(Blueprint(
    name="b_r1", phase=2, env_entry=B_ENTRY, env_kwargs=dict(B_ENV),
    model=TRANSFORMER, total_steps=2_500_000, n_seeds=3,
    notes="THE measurement: transformer PPO, delay=1, gamma per b_r1_g0/g99 variants.",
))
register(Blueprint(
    name="b_r1_g0", phase=2, env_entry=B_ENTRY, env_kwargs=dict(B_ENV),
    model=TRANSFORMER, ppo=PPOSpec(gamma=0.0), total_steps=2_500_000, n_seeds=2,
    notes="Guesses never influence the future: gamma=0 arm (expected cleaner).",
))
register(Blueprint(
    name="b_r0", phase=2, env_entry=B_ENTRY, env_kwargs=dict(B_ENV, delay=0),
    model=TRANSFORMER, total_steps=2_500_000, n_seeds=3,
    notes="Sanity: should approach the delay=0 filter ceiling (~0.85).",
))
register(Blueprint(
    name="b_m1", phase=2, env_entry=B_ENTRY, env_kwargs=dict(B_ENV),
    model=MLP, ppo=PPOSpec(gamma=0.0), total_steps=2_000_000, n_seeds=3,
    notes="Memoryless MLP; must match analytic memoryless ceiling 0.6593 (plumbing check). "
          "gamma=0 (guesses never influence the future); at gamma=0.99 bootstrap noise "
          "stalls it at ~0.45 greedy — measured, documented in FINDINGS_phase2.",
))
register(Blueprint(
    name="b_sl", phase=2, env_entry=B_ENTRY, env_kwargs=dict(B_ENV),
    model=STATE_TRANSFORMER, total_steps=5_000_000, n_seeds=3,
    rl_loss_enabled=False,
    notes="Supervised twin: identical transformer, cross-entropy on true state, "
          "random-guess rollouts (identical data distribution). Architecture ceiling.",
))

# --- Phase 3: Environment A main arms ----------------------------------------

register(Blueprint(
    name="a_main", phase=3, env_entry=A_ENTRY, env_kwargs=dict(A_ENV),
    model=TRANSFORMER, total_steps=10_000_000, n_seeds=3,
    notes="HEADLINE: reward-only PPO, delay=1. Expected reward ~0.34, fine R^2 0.78-0.83.",
))
register(Blueprint(
    name="a_nodelay", phase=3, env_entry=A_ENTRY, env_kwargs=dict(A_ENV, delay=0),
    model=TRANSFORMER, total_steps=10_000_000, n_seeds=3,
    notes="Premium-removal control. Expected global R^2 ~0.93, fine R^2 ~0.10 (below floor).",
))
register(Blueprint(
    name="a_aux_0p1", phase=3, env_entry=A_ENTRY, env_kwargs=dict(A_ENV),
    model=NEXT_TOKEN_TRANSFORMER, learner_class=PPOWithNextTokenAux,
    aux_config={"next_token_aux/lambda": 0.1},
    total_steps=10_000_000, n_seeds=3,
    notes="A-main + auxiliary next-token CE head, lambda=0.1. Verify grads reach the core.",
))
register(Blueprint(
    name="a_aux_0p5", phase=3, env_entry=A_ENTRY, env_kwargs=dict(A_ENV),
    model=NEXT_TOKEN_TRANSFORMER, learner_class=PPOWithNextTokenAux,
    aux_config={"next_token_aux/lambda": 0.5},
    total_steps=10_000_000, n_seeds=3,
    notes="lambda=0.5. Prior round: 0.354 reward / 0.87 fine R^2.",
))
register(Blueprint(
    name="a_pred", phase=3, env_entry=A_ENTRY, env_kwargs=dict(A_ENV),
    model=NEXT_TOKEN_TRANSFORMER, total_steps=10_000_000, n_seeds=3,
    rl_loss_enabled=False,
    notes="Prediction-only plumbing control (random actions). MUST produce fine geometry (~0.86).",
))
register(Blueprint(
    name="a_oracle", phase=3, env_entry=A_ENTRY,
    env_kwargs=dict(A_ENV, obs_mode="state"),
    model=MLP, total_steps=5_000_000, n_seeds=3,
    notes="True-state observation, MLP. Must match analytic oracle 0.4625.",
))
register(Blueprint(
    name="a_beliefobs", phase=3, env_entry=A_ENTRY,
    env_kwargs=dict(A_ENV, obs_mode="belief"),
    model=MLP, total_steps=5_000_000, n_seeds=3,
    notes="Exact filter posterior as observation, MLP. Decomposes A-main's ceiling gap: "
          "filtering vs PPO optimization.",
))
for k in (2, 4, 8, 16):
    register(Blueprint(
        name=f"a_stack{k}", phase=3, env_entry=A_ENTRY,
        env_kwargs=dict(A_ENV, obs_mode=f"stack{k}"),
        model=MLP, total_steps=5_000_000, n_seeds=2,
        notes=f"MLP frame-stack, last {k} tokens + last {k} actions.",
    ))
register(Blueprint(
    name="a_lstm", phase=3, env_entry=A_ENTRY, env_kwargs=dict(A_ENV),
    model=ModelSpec(LSTMModel, LSTMModelConfig(hidden_dim=128)),
    total_steps=10_000_000, n_seeds=2,
    notes="OPTIONAL architecture contrast; only if budget allows.",
))

# --- Phase 4: the one new arm -------------------------------------------------

register(Blueprint(
    name="n_scramble", phase=4, env_entry=A_ENTRY, env_kwargs=dict(A_ENV),
    model=TRANSFORMER, total_steps=10_000_000, n_seeds=3, scramble_tokens=True,
    notes="Token obs replaced by i.i.d. uniform draws; true chain/filter unchanged. "
          "Expected reward = analytic no-information optimum 0.1415; trained-on-noise "
          "probe floor.",
))
