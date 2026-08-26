"""50M-step factored MESS3 recipes with configurable transformer width."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
import torch

from envs.hmm import HMMEnv
from experiments.factored_mess3_beliefs_2026_08.analysis import probe_checkpoint
from experiments.factored_mess3_beliefs_2026_08.shared import (
    LARGE_JOINT_MINIBATCH_SIZE,
    LARGE_JOINT_TRAIN_BATCH_SIZE,
    MINIBATCH_SIZE,
    SMOKE_BATCH_SIZE,
    SMOKE_ENV_STEPS,
    SMOKE_MINIBATCH_SIZE,
    TRAIN_BATCH_SIZE,
    _apply_runtime_resources,
    _metric,
    _probe_summary,
    _save_initial_checkpoint,
    environment_config,
)
from experiments.mess3_token_guess_cycle_2.model import (
    PaperActorCriticConfig,
    PaperActorCriticModel,
)
from harness.artifacts import RunArtifacts
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.runners import run_tune
from learners.models.next_token import NextTokenAuxHead
from losses.next_token import NextTokenAuxLossMixin


TOTAL_ENV_STEPS = 50_000_000
ENTROPY_COEFF = 0.008
PREDICTIVE_LOSS_WEIGHT = 1.0

MODEL_CONFIG_64D = PaperActorCriticConfig(
    d_model=64,
    n_layers=4,
    n_heads=1,
    d_head=8,
    d_mlp=256,
    context_length=11,
    max_seq_len=11,
).to_dict()

MODEL_CONFIG_120D = PaperActorCriticConfig(
    d_model=120,
    n_layers=4,
    n_heads=3,
    d_head=40,
    d_mlp=480,
    context_length=11,
    max_seq_len=11,
).to_dict()


class PredictiveModel(NextTokenAuxHead, PaperActorCriticModel):
    """Paper transformer with a training-only joint-token prediction head."""


class PredictiveLearner(NextTokenAuxLossMixin, PPOTorchLearner):
    """Standard PPO plus joint next-token cross-entropy."""


def next_joint_token_targets(
    batch: Mapping[str, Any],
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align action-time activations with the delayed joint token they predict."""

    observations = batch[Columns.OBS]
    if observations.ndim != 3 or logits.ndim != 3:
        raise ValueError("joint-token training expects (B, T, D) tensors")
    num_classes = logits.shape[-1]
    if observations.shape[-1] < num_classes:
        raise ValueError(
            "joint-token observations must contain one channel per class"
        )
    next_tokens = observations[:, 1:, :num_classes]
    targets = next_tokens.argmax(dim=-1)
    populated = next_tokens.sum(dim=-1) > 0.5
    mask = batch.get(Columns.LOSS_MASK)
    if mask is None:
        mask = torch.ones(
            observations.shape[:2],
            dtype=torch.bool,
            device=observations.device,
        )
    else:
        mask = mask.to(device=observations.device, dtype=torch.bool)
    valid = mask[:, :-1] & mask[:, 1:] & populated
    return logits[:, :-1, :], targets, valid


def _resolved_model_config(
    model_config: Mapping[str, Any] | None,
    *,
    n_factors: int,
    predictive_auxiliary: bool,
) -> dict[str, Any]:
    resolved = dict(model_config or MODEL_CONFIG_64D)
    if predictive_auxiliary:
        resolved["next_token_aux"] = {"num_classes": 3**n_factors}
    return resolved


def build_config(
    context: RunContext,
    *,
    n_factors: int = 2,
    model_config: Mapping[str, Any] | None = None,
    predictive_auxiliary: bool = False,
) -> PPOConfig:
    """Build a gamma-zero PPO config for the long-run factored study."""

    resolved_model = _resolved_model_config(
        model_config,
        n_factors=n_factors,
        predictive_auxiliary=predictive_auxiliary,
    )
    large_joint = n_factors >= 6
    batch_size = (
        SMOKE_BATCH_SIZE
        if context.smoke
        else (
            LARGE_JOINT_TRAIN_BATCH_SIZE
            if large_joint
            else TRAIN_BATCH_SIZE
        )
    )
    minibatch_size = (
        SMOKE_MINIBATCH_SIZE
        if context.smoke
        else (
            LARGE_JOINT_MINIBATCH_SIZE
            if large_joint
            else MINIBATCH_SIZE
        )
    )
    profile = context.hardware or PROFILES["cpu"]
    config = (
        PPOConfig()
        .environment(
            HMMEnv,
            env_config=environment_config(n_factors),
        )
        .framework(
            "torch",
            torch_compile_learner=(
                not context.smoke and profile.learner_device == "cuda"
            ),
            torch_compile_learner_what_to_compile="forward_train",
            torch_compile_learner_dynamo_backend="inductor",
            torch_compile_learner_dynamo_mode="reduce-overhead",
            torch_compile_worker=False,
        )
        .training(
            lr=1e-4,
            gamma=0.0,
            lambda_=0.0,
            clip_param=0.2,
            use_kl_loss=False,
            vf_loss_coeff=0.5,
            entropy_coeff=ENTROPY_COEFF,
            train_batch_size_per_learner=batch_size,
            minibatch_size=minibatch_size,
            num_epochs=6,
            shuffle_batch_per_epoch=True,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=(
                    PredictiveModel
                    if predictive_auxiliary
                    else PaperActorCriticModel
                ),
                model_config=resolved_model,
            )
        )
        .callbacks(
            on_algorithm_init=partial(
                _save_initial_checkpoint,
                checkpoint_path=str(
                    context.artifacts_dir / "initial_checkpoint"
                ),
            )
        )
        .debugging(seed=context.seed)
    )
    if predictive_auxiliary:
        config = config.learners(
            learner_class=PredictiveLearner,
            learner_config_dict={
                "next_token_aux/lambda": PREDICTIVE_LOSS_WEIGHT,
                "next_token_aux/target_extractor": next_joint_token_targets,
            },
        )
    return _apply_runtime_resources(
        config,
        context,
        n_factors=n_factors,
    )


def run_independent(
    context: RunContext,
    *,
    n_factors: int = 2,
    model_config: Mapping[str, Any] | None = None,
    predictive_auxiliary: bool = False,
) -> dict[str, Any]:
    """Train one independent factored condition for 50M steps, then probe."""

    resolved_model = _resolved_model_config(
        model_config,
        n_factors=n_factors,
        predictive_auxiliary=predictive_auxiliary,
    )
    if context.seed is None:
        raise ValueError("factored MESS3 training requires a resolved seed")
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    target_steps = SMOKE_ENV_STEPS if context.smoke else TOTAL_ENV_STEPS
    factored_dimension = n_factors * 2
    joint_dimension = 3**n_factors - 1
    joint_states = 3**n_factors
    config_environment = environment_config(n_factors)
    outputs.write_json(
        "resolved_recipe.json",
        {
            "hypothesis": (
                "PPO's transformer will encode each independent MESS3 belief "
                "in a linearly decodable, approximately orthogonal "
                f"two-dimensional subspace, using about {factored_dimension} "
                "activation dimensions rather than the "
                f"{joint_dimension}-dimensional joint simplex."
            ),
            "primary_comparison": (
                "matched_ppo_baseline_vs_ppo_plus_next_token_cross_entropy"
                if predictive_auxiliary
                else "step_zero_initialization_vs_final_checkpoint"
            ),
            "generator": f"{n_factors} independent passive MESS3 HMM factors",
            "observed_token": (
                f"one {joint_states}-way token in one-to-one correspondence "
                "with the Cartesian tuple of three-way factor subtokens, "
                "delivered one decision late"
            ),
            "task": (
                "guess the currently withheld joint emitted token "
                f"({joint_states} actions)"
            ),
            "algorithm": "PPO",
            "objective": (
                "ppo_correctness_plus_joint_next_token_cross_entropy"
                if predictive_auxiliary
                else "ppo_correctness"
            ),
            "predictive_loss_weight": (
                PREDICTIVE_LOSS_WEIGHT if predictive_auxiliary else 0.0
            ),
            "gamma": 0.0,
            "lambda": 0.0,
            "entropy_coeff": ENTROPY_COEFF,
            "environment": config_environment,
            "model": resolved_model,
            "total_env_steps": target_steps,
            "train_batch_size_per_learner": (
                SMOKE_BATCH_SIZE
                if context.smoke
                else (
                    LARGE_JOINT_TRAIN_BATCH_SIZE
                    if n_factors >= 6
                    else TRAIN_BATCH_SIZE
                )
            ),
            "minibatch_size": (
                SMOKE_MINIBATCH_SIZE
                if context.smoke
                else (
                    LARGE_JOINT_MINIBATCH_SIZE
                    if n_factors >= 6
                    else MINIBATCH_SIZE
                )
            ),
            "probe_controls": [
                "exact step-zero checkpoint",
                "disjoint train/test rollout seeds",
                "current-token observation-only affine baseline",
                "joint-versus-factored dimension predictions",
                "Product-Constrained Joint Reconstruction (PCJR)",
                "Correlation-Residual Decodability (CRD)",
                "Joint Readout Excess Subspace (JRES)",
            ],
            "paper": "https://arxiv.org/abs/2602.02385",
        },
    )

    config = build_config(
        context,
        n_factors=n_factors,
        model_config=resolved_model,
        predictive_auxiliary=predictive_auxiliary,
    )
    result_grid = run_tune(
        config,
        context,
        stop={"env_runners/num_env_steps_sampled_lifetime": target_steps},
        run_config_kwargs={
            "checkpoint_config": tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_at_end=True,
            )
        },
    )
    results = list(result_grid)
    if len(results) != 1:
        raise RuntimeError(f"expected one Tune trial, got {len(results)}")
    result = results[0]
    if result.error is not None:
        raise RuntimeError("factored MESS3 PPO training failed") from result.error
    if result.checkpoint is None:
        raise RuntimeError("factored MESS3 PPO produced no final checkpoint")
    initial_checkpoint = context.artifacts_dir / "initial_checkpoint"
    if not (initial_checkpoint / "rllib_checkpoint.json").is_file():
        raise RuntimeError("the exact initialization checkpoint was not saved")
    final_checkpoint = Path(result.checkpoint.path)
    final_steps = int(
        _metric(
            result.metrics or {},
            "env_runners/num_env_steps_sampled_lifetime",
        )
        or target_steps
    )

    initial_metrics = probe_checkpoint(
        context,
        checkpoint=initial_checkpoint,
        agent_steps=0,
        n_factors=n_factors,
    )
    initial_probe_path = context.results_dir / "probe_metrics.json"
    initial_cev_path = context.results_dir / "cev.png"
    initial_dir = context.results_dir / "initial"
    initial_dir.mkdir(parents=True, exist_ok=True)
    initial_probe_path.replace(initial_dir / "probe_metrics.json")
    initial_cev_path.replace(initial_dir / "cev.png")

    final_metrics = probe_checkpoint(
        context,
        checkpoint=final_checkpoint,
        agent_steps=final_steps,
        n_factors=n_factors,
    )
    final_probe_path = context.results_dir / "probe_metrics.json"
    final_cev_path = context.results_dir / "cev.png"
    final_dir = context.results_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    final_probe_path.replace(final_dir / "probe_metrics.json")
    final_cev_path.replace(final_dir / "cev.png")

    initial_summary = _probe_summary(initial_metrics)
    final_summary = _probe_summary(final_metrics)
    summary = {
        "seed": context.seed,
        "smoke": context.smoke,
        "n_factors": n_factors,
        "sampled_env_steps": final_steps,
        "training_episode_return_mean": _metric(
            result.metrics or {},
            "env_runners/episode_return_mean",
        ),
        "initial_probe": initial_summary,
        "final_probe": final_summary,
        "training_change": {
            "joint_belief_r_squared_delta": (
                final_summary["joint_belief_r_squared"]
                - initial_summary["joint_belief_r_squared"]
            ),
            "factor_r_squared_delta": {
                name: (
                    final_summary["factor_r_squared"][name]
                    - initial_summary["factor_r_squared"][name]
                )
                for name in final_summary["factor_r_squared"]
            },
            "cev95_dimension_delta": (
                final_summary["cev95_dimension"]
                - initial_summary["cev95_dimension"]
            ),
            "factor_subspace_overlap_delta": (
                final_summary["factor_subspace_overlap"]
                - initial_summary["factor_subspace_overlap"]
            ),
            "pcjr_product_minus_direct_mse_delta": (
                final_summary["product_constrained_joint_reconstruction"][
                    "product_minus_direct_mse"
                ]
                - initial_summary["product_constrained_joint_reconstruction"][
                    "product_minus_direct_mse"
                ]
            ),
        },
        "conclusion_status": (
            "smoke_diagnostic_only"
            if context.smoke
            else "single_seed_exploratory"
        ),
        "interpretation_rule": (
            "Evidence for native factoring requires all factor beliefs to be "
            "linearly decodable, CEV near the "
            f"{factored_dimension}-dimensional direct-sum prediction rather "
            f"than the {joint_dimension}-dimensional joint prediction, low "
            "factor-subspace overlap, competitive PCJR product reconstruction, "
            "degenerate CRD for this independent generator, and limited JRES "
            "directions beyond the factor union."
        ),
    }
    outputs.write_json("condition_summary.json", summary)
    required = [
        context.results_dir / "resolved_recipe.json",
        context.results_dir / "tune_summary.json",
        context.results_dir / "condition_summary.json",
        initial_dir / "probe_metrics.json",
        initial_dir / "cev.png",
        final_dir / "probe_metrics.json",
        final_dir / "cev.png",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("missing compact outputs: " + ", ".join(missing))
    (context.results_dir / "output_validation.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "required_files": [
                    str(path.relative_to(context.results_dir))
                    for path in required
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return summary
