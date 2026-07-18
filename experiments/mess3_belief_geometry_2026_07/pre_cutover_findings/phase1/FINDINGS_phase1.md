# FINDINGS — Phase 1: Build + gates (no training)

Date: 2026-07-05.  All numbers regenerated from scratch in this repo;
"prior" refers to the known-good values carried over from earlier rounds.
Reproduce with:

```
uv run pytest envs/mess3/tests -q          # full unit + regression suite (incl. slow MC)
uv run python scripts/phase1_gate.py       # this sweep (~8 min)
uv run python scripts/phase1_findings.py   # gate verification -> GATE_PASSED
```

## 1. Known-good regression (beta=4, w_max2=5, delay=1)

| quantity        | prior  | this round | MC cross-check (1e6 steps)     | status |
|-----------------|--------|-----------:|--------------------------------|--------|
| reactive        | 0.192  | 0.19145    | 0.192 +/- 0.001                | REPRODUCED |
| stack-2         | 0.295  | 0.30553    | 0.30488 +/- 0.00126 (3e5)      | IMPROVED (see below) |
| belief ceiling  | 0.381  | 0.38140    | 0.38076 +/- 0.00088 (policy MC)| REPRODUCED |
| oracle          | 0.4625 | 0.46250    | within 4 SE (test suite)       | REPRODUCED |
| no-info constant| —      | 0.14154    | 0.14137 +/- 0.00069 (3e5)      | new (N-scramble target) |

Gate thresholds at the operating point: premium over reactive
(0.381 - 0.191)/0.381 = **0.498 >= 0.15**; premium over stack-2
(0.381 - 0.306)/0.381 = **0.199 >= 0.08**.  **GATE PASSES.**

**Stack-2 deviation, investigated.** Our stack-2 solver (exact stationary
value of the induced chain on (state, token, token), globally seeded from the
reactive optimum + corner/random restarts, L-BFGS-B polish) finds 0.3055 vs
prior 0.295.  The table was MC-validated through the actual environment
(0.3049 +/- 0.0013), and the value sits strictly between reactive and the
belief ceiling, so it is a legitimately better stack-2 table — the prior
round's optimizer stopped in a local optimum.  Consequence: the stack-2
premium tightens from ~0.23 to 0.199 (still >2x the 0.08 threshold), and the
A-stack-2 training arm's target moves to 0.3055.

Other required verifications (unit-tested, `envs/mess3/tests/`):
w=0 forever reproduces the (0.45, 0.45, 0.10) stationary distribution with
zero control cost; all tilted rows sum to 1 and stay full-support across the
box; the filter collapses to one-hot at alpha=1.0/delay=0; passive mode
matches the observable-operator (T(o) = diag(E[:,o]) @ M) recursion exactly;
the filter is empirically calibrated at both delays under random continuous
actions (posterior-probability-vs-frequency buckets within 0.02); the
3D-tilt gauge direction is exactly flat (softmax-row invariance), justifying
the 2D gauge-fixed action space.

## 2. Gate sweep: beta x w_max2 x delay (full table: gate_sweep.csv)

Delay=1 rows (ceilings are exact/VI; interior shares under the VI optimum's
own stationary closed loop, 3e5 steps; "any" = at least one coordinate off
the box boundary):

| beta | w_max2 | react | stack2 | belief | oracle | prem_react | prem_stack2 | gate | interior any / per-coord |
|-----:|-------:|------:|-------:|-------:|-------:|-----------:|------------:|------|--------------------------:|
| 2 | 2 | .1262 | .1267 | .1266 | .2470 |  .003 | -.001 | fail | 1.00 / 1.00 (degenerate) |
| 2 | 3 | .1262 | .1267 | .1265 | .2470 |  .003 | -.001 | fail | 1.00 / 1.00 (degenerate) |
| 2 | 5 | .1262 | .1267 | .1253 | .2470 | -.007 | -.011 | fail | 1.00 / 1.00 (degenerate) |
| 2 | 8 | .1262 | .1267 | .1240 | .2470 | -.018 | -.022 | fail | 1.00 / 1.00 (degenerate) |
| 4 | 2 | .1457 | .1548 | .1859 | .3531 |  .216 |  .167 | PASS | .266 / .133 |
| 4 | 3 | .1457 | .1670 | .2697 | .4404 |  .460 |  .381 | PASS | .357 / .178 |
| **4** | **5** | **.1915** | **.3055** | **.3814** | **.4625** | **.498** | **.199** | **PASS** | **.078 / .039** |
| 4 | 8 | .2554 | .3688 | .4223 | .4625 |  .395 |  .127 | PASS | .008 / .005 |
| 8 | 2 | .1834 | .2135 | .2328 | .4116 |  .212 |  .083 | PASS | .119 / .059 |
| 8 | 3 | .2151 | .2577 | .3770 | .5701 |  .429 |  .317 | PASS | .0002 / .0001 |
| 8 | 5 | .3529 | .4941 | .5999 | .6982 |  .412 |  .176 | PASS | .000 / .000 |
| 8 | 8 | .4824 | .5859 | .7040 | .7126 |  .315 |  .168 | PASS | .000 / .000 |

All 12 delay=0 configs FAIL the gate (max premium over stack-2 there is
0.025; over reactive 0.163 but never jointly) — with no delay, one token
nearly closes the information gap and memory buys almost nothing.

Expected sweep behavior, all confirmed:
- **Premiums only at delay=1 with beta in {4, 8}** — beta=2 collapses
  (control too expensive: the belief ceiling cannot even beat the stack-2
  baseline; at w_max2 >= 5 the VI grid value even dips microscopically below
  it, i.e. premium ~ 0 within grid error).  The beta=2 "interior share 1.0"
  is degenerate: optimal tilts are tiny, nothing touches the box.
- **Premium non-monotone in beta** with the sweet spot at beta=4
  (premium_react .498 at beta=4 vs .412 at beta=8, w_max2=5).
- **Prior ~96% boundary saturation at w_max2=5 reproduced**: per-coordinate
  boundary share 96.1%.
- **Smaller boxes raise the interior share at a ceiling cost** (beta=4:
  any-interior .27/.36/.08/.01 for w_max2 = 2/3/5/8; ceilings .186/.270/
  .381/.422).  Jointly-interior actions are essentially measure-zero
  everywhere (<= 0.1%): at least one coordinate is always pinned; hedging
  lives in the remaining coordinate.

Where interior actions concentrate (figures
`fig_interior_vs_time_since_s2.png`, `fig_interior_vs_belief_entropy.png`):
at (4, 5) the visitation mass sits at step 0 (inside/just leaving s2, low
entropy) where actions are corner-pinned, and the interior fraction RISES
with time since ejection (to ~0.5 at steps 3-15) and peaks in the
mid-entropy band (~0.5-0.8 nats) — hedging happens exactly while re-locating
after ejection, which is where the fine belief distinctions live.  At the
tighter boxes (4, 2) and (4, 3) the interior band widens dramatically
(fraction ~0.3-0.9 across most of the visited entropy range) — the broader
hedging demand hypothesized for smaller boxes.

## 3. Recommended operating point

**Primary: beta=4, w_max2=5, delay=1** (the default candidate).
Rationale: passes both gates with the largest reactive premium in the sweep;
every prior-round training expectation (A-main reward ~0.34, fine R^2
0.78-0.83, aux numbers, null floors) is anchored here, so it preserves
comparability.

The tighter box beta=4, w_max2=3 DOES satisfy the spec's alternative
criterion — it passes both gates (premiums .460/.381) with materially higher
interior share (.36 vs .08 any-coordinate) — at a 29% ceiling cost
(0.270 vs 0.381).  Recommendation: keep it as an OPTIONAL secondary arm
(one extra blueprint), not the primary, because (a) prior-round anchors
would all be invalidated, and (b) even at w_max2=5's 96% saturation the
prior round recovered fine fractal structure, so saturation evidently does
not preclude the headline effect.  Decision left to review, per the phase
plan.

Phase-5 caution triggered: heavy box-corner saturation at (4, 5) (jointly
interior ~ 0) means small-N lattices WILL collapse onto corners; the Phase-5
analytic gate (cell-count growth vs N) must be checked before any ladder
training, and a box adjustment for the ladder should be on the table.

## 4. Environment B (State-Guess) analytic table

| delay | random | memoryless (map)      | filter ceiling (sim, 2e6 steps) |
|------:|-------:|-----------------------|---------------------------------|
| 0 | 1/3 | 0.850000 (identity map) — equals alpha EXACTLY, as required | 0.85003 +/- 0.00011 |
| 1 | 1/3 | 0.659250 (map 0->0, 1->1, 2->0) | 0.66912 +/- 0.00006 |

Notes for Phase 2: at delay=1 the filter premium over memoryless is ~0.010
absolute (0.6593 -> 0.6691) — real but small, so B-R1 vs B-M1 needs the
gamma=0 arm's variance reduction and >= 3 seeds to resolve cleanly.  At
delay=0 the filter ceiling equals the memoryless ceiling to within SE
(memory is nearly worthless once the current token is visible), which makes
B-R0 a pure plumbing sanity check.  The delay=1 memoryless map never
guesses state 2 (stationary mass 0.10) — guessing 2 is optimal only with
information the delayed token cannot carry.

## 5. Deliverables and status

- `gate_sweep.csv` / `.json` — the gate + saturation table (24 configs).
- `stateguess_table.csv` — Environment B ceilings, both delays.
- `interior_*.npz` — histogram payloads + closed-loop trajectory samples.
- `fig_attractor_beta*_wmax*.png` — reachable-belief attractor (RGB=belief)
  and the optimal action map over it, per delay=1 config.  At (4, 5) the
  attractor shows the expected fractal branch structure on the 2-simplex.
- `fig_interior_vs_time_since_s2.png`, `fig_interior_vs_belief_entropy.png`.
- `GATE_PASSED` — written by scripts/phase1_findings.py (all 12 checks ok).
- Unit tests: 52 passing (47 fast + 5 slow MC cross-checks).
- Blueprints for ALL Phase 2-4 arms declared and listed in the README;
  scripts/train.py REFUSES to launch phase >= 2 arms until REVIEW_APPROVED
  exists next to GATE_PASSED (MLP arms verified launchable end-to-end via
  --smoke; transformer core + aux/supervised heads are Phase-2 build).

**Phase 1 is complete and STOPPED for review.** To unlock Phase 2/3
training after review: `touch results/phase1/REVIEW_APPROVED`.
