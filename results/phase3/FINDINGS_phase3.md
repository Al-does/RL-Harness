# Phase 3 findings: Environment A main arms

Operating point beta=4, w_max2=5, delay=1 (alpha=0.85, episodes of 1024).

## Analytic anchors (Phase 1)

- constant: 0.1415
- reactive: 0.1915
- stack2: 0.3055
- belief_ceiling: 0.3814
- oracle: 0.4625
- belief_ceiling_delay0: 0.4054
- openloop_period4: 0.1966

## Arm table (mean ± sd over seeds; rewards are greedy rollouts)

| arm | seeds | reward (greedy) | global R^2 | fine R^2 | within-branch act var |
|-----|------:|----------------:|-----------:|---------:|----------------------:|
| a_main | 3 | 0.337 ± 0.002 | 0.861 ± 0.019 | 0.802 ± 0.030 | 0.703 ± 0.024 |
| a_nodelay | 3 | 0.376 ± 0.015 | 0.897 ± 0.020 | 0.013 ± 0.015 | 0.142 ± 0.010 |
| a_aux_0p1 | 0 | — | — | — | — |
| a_aux_0p5 | 3 | 0.338 ± 0.001 | 0.920 ± 0.001 | 0.884 ± 0.001 | 0.678 ± 0.014 |
| a_pred | 3 | 0.063 ± 0.028 | 0.952 ± 0.002 | 0.950 ± 0.002 | 1.000 ± 0.000 |
| a_oracle | 3 | 0.453 ± 0.000 | 0.000 ± 0.000 | -0.871 ± 0.011 | 1.000 ± 0.000 |
| a_beliefobs | 3 | 0.379 ± 0.001 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.656 ± 0.005 |
| a_stack2 | 2 | 0.330 ± 0.009 | 0.930 ± 0.027 | 0.884 ± 0.047 | 0.455 ± 0.092 |
| a_stack4 | 2 | 0.306 ± 0.000 | 0.924 ± 0.003 | 0.866 ± 0.006 | 0.180 ± 0.004 |
| a_stack8 | 2 | 0.283 ± 0.022 | 0.850 ± 0.031 | 0.758 ± 0.029 | 0.334 ± 0.145 |
| a_stack16 | 2 | 0.274 ± 0.006 | 0.775 ± 0.010 | 0.635 ± 0.034 | 0.445 ± 0.108 |
| a_lstm | 0 | — | — | — | — |

## Readings

- **Headline (a_main)**: greedy reward 0.337 = 76% of the reactive-to-belief-ceiling gap; fine R^2 0.802 with 70% within-branch action variance (policy hedges on decision-relevant belief coordinates).
- **Premium removal (a_nodelay)**: greedy reward 0.376 vs its own ceiling 0.405, but fine R^2 collapses to 0.013 — reward alone does not buy fine belief geometry once the memory premium is gone.
- **Aux dose response (lambda -> fine R^2)**: 0 -> 0.802, 0.5 -> 0.884, 1 -> 0.950 (a_pred is prediction-only, random actions).
- **Stack ladder (k -> greedy reward | fine R^2)**: 2 -> 0.330 | 0.884, 4 -> 0.306 | 0.866, 8 -> 0.283 | 0.758, 16 -> 0.274 | 0.635
- **Oracle sanity (a_oracle)**: 0.453 vs analytic 0.4625; belief-obs MLP 0.379 ± 0.001 vs ceiling 0.3814.
