"""Generic linear-probe fitting and evaluation."""

from analysis.probes.linear import (
    conditional_mse_metrics,
    conditional_residual_r2,
    fit_affine_probe,
    global_mse_metrics,
    mean_squared_error,
    probe_predict,
    r2_score,
    split_indices,
    target_variance,
)
from analysis.probes.transducer import (
    predictive_belief_sequence,
    predictive_belief_update,
)

__all__ = [
    "conditional_mse_metrics",
    "conditional_residual_r2",
    "fit_affine_probe",
    "global_mse_metrics",
    "mean_squared_error",
    "predictive_belief_sequence",
    "predictive_belief_update",
    "probe_predict",
    "r2_score",
    "split_indices",
    "target_variance",
]
