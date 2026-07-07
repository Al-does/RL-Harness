# FINDINGS — Phase 5 analytic gate (discrete-action-count ladder) — STOPPED FOR REVIEW

Date: 2026-07-06.  `uv run python scripts/phase5_gate.py` regenerates
everything (results/phase5/gate_table.csv, fig_cells_N*.png).

Lattices over the operating-point box (beta=4, w_max2=5, delay=1), same
tilt dynamics and KL cost as A-main; belief-VI restricted to each lattice.

| N | belief ceiling | reactive ceiling | premium | cells (grid / closed loop) | mean cell diameter |
|---|---------------:|-----------------:|--------:|---------------------------:|-------------------:|
| 4 (2x2)  | 0.38139 | 0.1914 | 0.498 | 3 / 3 | 0.607 |
| 9 (3x3)  | 0.38139 | 0.1914 | 0.498 | 5 / 5 | 0.607 |
| 25 (5x5) | 0.38140 | 0.1914 | 0.498 | 7 / 6 | 0.605 |
| 49 (7x7) | 0.38140 | 0.1915 | 0.498 | 9 / 8 | 0.591 |
| continuous (A-main) | 0.38140 | 0.1915 | 0.498 | — | — |

MC cross-check of every discrete VI policy through the actual environment:
0.38099 ± (batch SE ~0.001) at each N — the ceilings are real.

## The Phase-1 caution materialized, in a sharper form than predicted

Cell COUNT does grow with N (3 -> 5 -> 6/7 -> 8/9), so the mechanical gate
check passes.  But the box-corner saturation discovered in Phase 1 (96%
boundary share, jointly-interior actions measure-zero) has a stronger
consequence visible only in this table:

1. **The belief ceiling is flat in N.**  Even the 4-action corner lattice
   attains the FULL continuous ceiling to 5 decimal places.  At this box,
   optimal VALUE is entirely corner-dominated; the continuous optimum's
   interior hedging (7.8% of steps) is worth < 1e-5 average reward.
2. **The resolution statistic barely moves** (mean cell diameter
   0.607 -> 0.591 from N=4 to N=49; the attractor spans diameter ~1).  The
   extra cells at larger N are thin slivers along branch boundaries, not a
   progressively finer partition.

Consequence for the planned headline figure (fine R^2 vs cell resolution):
at THIS box the x-axis has almost no dynamic range, and since reward
pressure is invariant across N, the prediction "fine R^2 climbs
monotonically in N" loses its analytic driver — the task simply does not
demand finer belief resolution at larger N here.  Running the ladder as-is
would measure noise around a flat law.

## Recommendation (decision needed at review)

Run the Phase-5 ladder at the TIGHTER BOX (beta=4, w_max2=3) instead of the
operating point:

- Phase 1 showed it passes both gates (premiums 0.460/0.381) with 36%
  any-coordinate interior share — hedging demand is real there, so lattice
  density should genuinely change attainable value and required belief
  resolution;
- its belief ceiling (0.2697) differs from its stack-2 ceiling (0.1670) by
  a wide margin, leaving room for the per-N ceilings to spread;
- A-main-at-(4,3) would be ONE additional continuous arm (the ladder's limit
  point must share the box, per the plan's confound caution).

Alternative (cheaper, weaker): keep (4,5) and reinterpret the ladder as a
null test — "when the task does not demand resolution, fine R^2 should NOT
climb with N" — which is falsifiable but inverts the headline claim.

Per the plan, NO ladder training has been launched; this table is the
review-stop artifact.
