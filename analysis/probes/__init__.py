"""Generic linear-probe fitting and evaluation."""

from analysis.probes.linear import (
    conditional_residual_r2,
    fit_affine_probe,
    probe_predict,
    r2_score,
    split_indices,
)

__all__ = [
    "conditional_residual_r2",
    "fit_affine_probe",
    "probe_predict",
    "r2_score",
    "split_indices",
]
