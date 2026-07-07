# RLLibHarnesBeta

Local RLlib harness for custom RL experiments.  Current program:
**MESS3-Control — belief-state geometry in deep RL** (does reward pressure,
via PPO, carve Bayesian belief geometry into recurrent policies?).

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13+.

```bash
uv sync --group dev
```

## Layout

- `envs/mess3/` — Environment A (`Mess3ContinuousEnv`: MESS3-style POMDP,
  KL-regularized continuous tilt control, one-step observation delay) and
  Environment B (`StateGuessEnv`: pure state estimation in RL clothing),
  plus the exact Bayesian filter, analytic solvers (oracle / belief-MDP VI /
  reactive family / State-Guess ceilings), and the full test suite.
- `blueprints/` — one named, inert blueprint per experiment arm
  (`blueprints/mess3_arms.py`); shared config by composition.
- `scripts/` — phase gates and the gate-enforced training launcher.
- `results/phaseK/` — one results tree per phase: tables, figures, `.npz`
  payloads, `FINDINGS_phaseK.md`, and gate artifacts.
- `analysis/` — shared plotting (barycentric simplex renderings).

## Commands

```bash
# Test suite (fast) / full suite incl. >=1e6-step Monte-Carlo solver checks
uv run pytest envs/mess3/tests -q -m "not slow"
uv run pytest envs/mess3/tests -q

# Phase 1: gate sweep (beta x w_max2 x delay), Env-B table, figures (~8 min)
uv run python scripts/phase1_gate.py
# Phase 1: verify against known-good values; writes results/phase1/GATE_PASSED
uv run python scripts/phase1_findings.py
# Regenerate Phase-1 figures from saved payloads (no re-solve)
uv run python scripts/phase1_gate.py --replot

# Training (any arm x seed). REFUSES to run before GATE_PASSED and
# REVIEW_APPROVED exist in results/phase1/ (the Phase-1 review stop).
uv run python scripts/train.py --blueprint <name> --seed <k>
uv run python scripts/train.py --blueprint a_oracle --seed 0 --smoke  # wiring check
```

## Blueprints

Phase 2 (Environment B ladder): `b_r1`, `b_r1_g0`, `b_r0`, `b_m1`, `b_sl`.
Phase 3 (Environment A arms): `a_main`, `a_nodelay`, `a_aux_0p1`,
`a_aux_0p5`, `a_pred`, `a_oracle`, `a_beliefobs`, `a_stack2`, `a_stack4`,
`a_stack8`, `a_stack16`, `a_lstm` (optional).
Phase 4: `n_scramble` (plus re-analysis of saved checkpoints).

Every result is reproducible from (blueprint name, seed); list them with
`uv run python -c "from blueprints.base import all_blueprints; [print(n) for n in all_blueprints()]"`.

## Program status

Phase 1 (build + analytic gates) is **complete and stopped for review** —
see `results/phase1/FINDINGS_phase1.md`.  All known-good values reproduced;
recommended operating point: beta=4, w_max2=5, delay=1.

## License

Apache 2.0 — see [LICENSE](LICENSE).
