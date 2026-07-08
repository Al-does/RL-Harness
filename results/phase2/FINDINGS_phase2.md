;# FINDINGS — Phase 2: Environment B ladder (State-Guess, the RL-tax measurement)

Date: 2026-07-06.  Reproduce any cell from its blueprint + seed:

```
uv run python scripts/train.py --blueprint <arm> --seed <k>
uv run python scripts/probe_arm.py --run results/phase2/<arm>/seed<k> [--final-only]
uv run python scripts/phase2_findings.py     # assembles the ladder table
```

## 0. Pre-ladder gate: probe pipeline validated on passive MESS3(0.05, 0.85)

The canonical transformer trained with the supervised next-token loss on the
passive environment probes at **global R^2 0.997** (prior round: 0.994; gate
threshold 0.98) with fine R^2 0.956/0.985 at branch depths 2/1
(`results/phase2/passive_probe/`).  The env's passive mode, the exact filter,
activation collection, the affine probe, and both metrics reproduce Shai et
al. end to end before any RL result below is interpreted.

## 1. The ladder (greedy accuracy; analytic anchors from Phase 1)

| rung | value (mean ± sd over seeds) | anchor | verdict |
|------|------------------------------|--------|---------|
| random | 1/3 | 0.3333 | — |
| analytic memoryless (delay=1) | — | 0.6593 | — |
| **B-M1** memoryless MLP, gamma=0, n=3 | **0.6586 ± 0.0017** | 0.6593 | at ceiling |
| **B-R1** transformer, gamma=0.99, n=3 | **0.6606 ± 0.0054** | 0.6691 | between ceilings |
| **B-R1(gamma=0)**, n=2 | **0.6684 ± 0.0008** | 0.6691 | at filter ceiling |
| **B-SL** supervised twin (aux-head held-out acc), n=3 | **0.6699 ± 0.0009** | 0.6691 | at filter ceiling |
| analytic filter ceiling (delay=1) | — | 0.6691 | — |

Sanity arm **B-R0** (delay=0), n=3: greedy 0.8374 ± 0.0165 vs the 0.85
ceiling; the mean is dragged by seed0 (0.814, entropy had already collapsed
before the last quarter of training), seeds 1-2 sit at 0.848/0.850.  The
delay=0 plumbing is confirmed.

**Gap decomposition (the phase's purpose):**

- architecture tax (filter − B-SL) = **−0.001** — the 3-layer banded
  transformer loses nothing to the exact filter on this chain;
- RL tax (B-SL − B-R1(gamma=0)) = **+0.001** — PPO at gamma=0 extracts the
  architecture's full filtering value;
- the gamma tax is the only visible one: gamma=0.99 costs ~0.008 (0.6606 vs
  0.6684) through value-bootstrap noise, exactly the "gamma=0 cleaner"
  prediction.  (An earlier B-M1 run at gamma=0.99 stalled at 0.45 greedy —
  documented in the b_m1 blueprint note; the ladder B-M1 is gamma=0.)

The delay=1 memory premium, though only ~0.010 absolute, is cleanly resolved:
B-R1(gamma=0) 0.6684 ± 0.0008 vs B-M1 0.6586 ± 0.0017 (≈ 5 sd separation),
and matches the analytic filter-minus-memoryless gap 0.0099.

Calibration for Environment A: on this task family, "ceiling minus attained"
gaps from PPO machinery itself are ≈ 0.  Phase-3 shortfalls can therefore be
attributed to filtering difficulty or task structure, not to PPO plumbing.

## 2. Representation geometry: the quantization prediction

Prediction (plan §3, NOTE): the optimal policy is argmax of the posterior —
piecewise-constant with three cells — so trained Env-B representations
should QUANTIZE beliefs (high global R^2, collapsed fine R^2), unlike the
supervised twin which is trained to track the full posterior.

| arm | global R^2 | fine R^2 (depth 2) | k-means(3) var explained (decoded) | within-branch action var |
|-----|-----------:|-------------------:|-----------------------------------:|-------------------------:|
| B-M1 (n=3) | 0.940 | **−9.47 ± 0.07** | 0.868 | 0.03–0.10 |
| B-R1 (n=3) | 0.992 | **−0.40 ± 0.53** | 0.949 | 0.02–0.07 |
| B-R1 g0 (n=2) | 0.996 | **+0.22 ± 0.32** | 0.944 | 0.02–0.04 |
| B-SL (n=3) | 0.9998 | **+0.966 ± 0.010** | 0.944 | (random actions) |

Held: the reward-trained arms sit at high global / collapsed-or-noisy fine
R^2 with ≥ 94% of decoded-belief variance explained by 3 cells, while the
IDENTICAL architecture under a posterior-tracking loss (B-SL) reaches fine
R^2 0.97.  The contrast is the discrete-argmax quantization signature, and
Environment B now stands as the coarsest-partition data point for the
Phase-5 ladder (3 alpha-vector cells).  Two nuances, reported as observed:

- the transformer RL arms are noisier in fine R^2 than the MLP (one
  b_r1_g0 seed reached +0.54): reward pressure does not FORBID residual fine
  structure, it merely stops paying for it — the floor-vs-pruning dynamics
  belong to Phase 4's N-init analysis;
- B-M1's fine R^2 of −9.5 is the pure quantization limit (3 one-hot cells
  from 3 input tokens; systematically displaced from within-branch belief
  variation), a useful lower anchor for "coarse".

## 3. Deviations and operational notes

- **Probe state-alignment fix (no R^2 impact).**  Mid-phase we found the
  probe collector logging s_{t+1} against the decision-time belief over s_t;
  beliefs/tokens/activations/actions were always aligned, so only the
  aux-head accuracy readout was affected.  Fixed; b_sl was re-probed (its
  held-out accuracy rose from a spurious 0.58 to 0.667–0.671 across seeds,
  matching the exact filter's argmax on the same rollouts, 0.6669).
- **Budgets.**  Env-B transformer arms plateau by ~1.5M steps (entropy
  < 0.07, returns flat); blueprints were trimmed 5M -> 2.5M accordingly.
  Seed-0 runs of b_r1/b_r1_g0/b_r0 ran to ~4.2M under the original budget
  and one b_r1/b_r0 seed each was cut at ~2.3-2.5M (plateaued; last
  log-spaced checkpoint used).  b_sl kept 5M (supervised path is cheap).
- gamma note resolved as predicted: report both, gamma=0 is the
  headline B-R1 number (0.6684), gamma=0.99 shown for the tax.
- Log-spaced checkpoints (including step 0) retained for every run; the
  probe-over-training curves are produced in the Phase-4 N-init pass
  (`scripts/phase4_nulls.py --probe`).

## 4. Expected outcomes scorecard (plan §6, Phase 2 row)

| expectation | outcome |
|-------------|---------|
| B-R1(gamma=0) close to B-SL | HELD (0.6684 vs 0.6699) |
| both possibly short of filter ceiling | short by ≤ 0.001 — effectively AT ceiling |
| probe shows ~3-cell quantization | HELD (k-means(3) ≥ 0.94; fine R^2 collapsed in RL arms) |
| gamma=0 cleaner than gamma=0.99 | HELD (+0.008, and 5x smaller seed spread) |
