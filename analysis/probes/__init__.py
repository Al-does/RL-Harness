"""Generic linear-probe fitting and evaluation."""

from analysis.probes.factorization import (
    center_within_groups,
    dimension_additivity,
    effective_dimension,
    orthonormal_basis,
    pairwise_subspace_overlaps,
    principal_component_basis,
    readout_subspace,
    regression_factor_geometry,
    regression_factor_subspaces,
    representation_dimension_predictions,
    subspace_overlap,
    variance_geometry,
    vary_one_subspace,
)
from analysis.probes.linear import (
    conditional_mse_metrics,
    conditional_residual_r2,
    fit_affine_probe,
    global_mse_metrics,
    mean_squared_error,
    probe_predict,
    r2_score,
    split_group_indices,
    split_indices,
    target_variance,
)
from analysis.probes.resampling import (
    cluster_bootstrap_statistics,
    held_out_permutation_null,
    percentile_interval,
)
from analysis.probes.transducer import (
    predictive_belief_sequence,
    predictive_belief_update,
)

__all__ = [
    "center_within_groups",
    "conditional_mse_metrics",
    "conditional_residual_r2",
    "cluster_bootstrap_statistics",
    "dimension_additivity",
    "effective_dimension",
    "fit_affine_probe",
    "global_mse_metrics",
    "held_out_permutation_null",
    "mean_squared_error",
    "orthonormal_basis",
    "pairwise_subspace_overlaps",
    "percentile_interval",
    "principal_component_basis",
    "predictive_belief_sequence",
    "predictive_belief_update",
    "probe_predict",
    "readout_subspace",
    "regression_factor_geometry",
    "regression_factor_subspaces",
    "representation_dimension_predictions",
    "r2_score",
    "split_group_indices",
    "split_indices",
    "subspace_overlap",
    "target_variance",
    "variance_geometry",
    "vary_one_subspace",
]
