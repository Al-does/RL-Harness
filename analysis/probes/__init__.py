"""Generic linear-probe fitting and evaluation."""

from analysis.probes.linear import (
    conditional_residual_r2,
    fit_affine_probe,
    probe_predict,
    r2_score,
    split_indices,
)
from analysis.probes.transducer import (
    predictive_belief_sequence,
    predictive_belief_update,
)

__all__ = [
    "conditional_residual_r2",
    "fit_affine_probe",
    "predictive_belief_sequence",
    "predictive_belief_update",
    "probe_predict",
    "r2_score",
    "split_indices",
]
