# Null-bracket table (Phase 4)

Fine R^2 = branch depth 2 (last two visible tokens) unless noted.

| arm | seeds | global R^2 (final) | fine R^2 (final) | fine R^2 (init) |
|-----|------:|-------------------:|-----------------:|----------------:|
| a_aux_0p5 | 2 | 0.920 ± 0.000 | 0.884 ± 0.000 | 0.562 |
| a_beliefobs | 3 | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 |
| a_main | 3 | 0.861 ± 0.019 | 0.802 ± 0.030 | 0.565 |
| a_nodelay | 3 | 0.897 ± 0.020 | 0.013 ± 0.015 | 0.003 |
| a_oracle | 3 | 0.000 ± 0.000 | -0.871 ± 0.011 | 0.476 |
| a_pred | 3 | 0.952 ± 0.002 | 0.950 ± 0.002 | 0.641 |
| a_stack16 | 2 | 0.775 ± 0.010 | 0.635 ± 0.034 | 0.008 |
| a_stack2 | 2 | 0.930 ± 0.027 | 0.884 ± 0.047 | 0.534 |
| a_stack4 | 2 | 0.924 ± 0.003 | 0.866 ± 0.006 | 0.081 |
| a_stack8 | 2 | 0.850 ± 0.031 | 0.758 ± 0.029 | 0.018 |
| b_m1 | 3 | 0.940 ± 0.000 | -9.390 ± 0.061 | -9.425 |
| b_r0 | 3 | 0.989 ± 0.010 | -1.182 ± 2.125 | -5.901 |
| b_r1 | 3 | 0.992 ± 0.003 | -0.405 ± 0.527 | -5.941 |
| b_r1_g0 | 2 | 0.996 ± 0.002 | 0.218 ± 0.327 | -6.469 |
| b_sl | 3 | 1.000 ± 0.000 | 0.966 ± 0.011 | -6.363 |
| n_scramble | 2 | 0.219 ± 0.097 | -0.220 ± 0.136 | 0.003 |

## Null bracket

- initialization floor (mlp): global 0.511, fine -1.327 (range -9.478..1.000)
- initialization floor (transformer): global 0.691, fine -1.718 (range -6.679..0.653)
- trained-on-noise floor (n_scramble): global 0.219, fine -0.220, reward 0.1691 (no-info optima: constant 0.1415, best periodic open-loop 0.1966 at period 4 -- scramble agents retain a clock via action history, so the open-loop value is the right ceiling)
