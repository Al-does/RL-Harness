"""Reproduce the supervised MESS3 residual-stream simplex from Shai et al."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.plots import simplex_scatter
from analysis.probes import r2_score
from envs.mess3.model import passive_model
from experiments.MESS3_supervised.model import PaperTransformer
from harness.artifacts import RunArtifacts, update_run_manifest
from harness.context import RunContext
from harness.hardware import PROFILES
from harness.seeding import (
    SeedSource,
    as_seed_sequence,
    named_seed_sequences,
    seed_sequence_to_int,
)


PAPER_URL = "https://arxiv.org/abs/2405.15943"
VOCAB_SIZE = 3
CONTEXT_LENGTH = 10
PATH_LENGTH = CONTEXT_LENGTH + 1
BATCH_SIZE = 64
TRAIN_UPDATES = 983_140
TRAIN_TOKEN_POSITIONS = TRAIN_UPDATES * BATCH_SIZE * CONTEXT_LENGTH
SMOKE_UPDATES = 8
LEARNING_RATE = 0.01
LOG_EVERY = 1_000
TRAINING_SAMPLE_CHUNK = 2_048
ANALYSIS_BATCH_SIZE = 4_096
PROBE_TRAIN_FRACTION = 0.20
PROBE_TARGET_DECIMALS = 5
FINAL_MSE_THRESHOLD = 1e-3
VALIDATION_RATIO_THRESHOLD = 1.01

VALIDATION_UPDATES = {
    0,
    100,
    1_000,
    4_980,
    10_000,
    30_000,
    100_000,
    300_000,
    600_000,
    TRAIN_UPDATES,
}
CHECKPOINT_UPDATES = {
    0,
    1_000,
    10_000,
    100_000,
    300_000,
    600_000,
    TRAIN_UPDATES,
}

_STREAM_KEYS = {
    "model_initialization": (0,),
    "training_sampling": (1,),
    "probe_split": (2,),
}
# Explicit spawn keys are order-independent; never renumber or reuse a key.



def _device(context: RunContext) -> torch.device:
    profile = context.hardware or PROFILES["cpu"]
    if profile.learner_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA profile selected but CUDA is unavailable")
        return torch.device("cuda")
    if (
        profile.learner_device == "mps"
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")
    return torch.device("cpu")


def _seed_torch(seed: SeedSource) -> None:
    value = seed_sequence_to_int(seed, bits=64)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def paper_transition_matrices() -> np.ndarray:
    """Return token-labeled matrices T[token, old_state, new_state]."""
    model = passive_model(alpha=0.85)
    matrices = np.stack(
        [
            model.transition_matrix * model.emission_matrix[:, token][None, :]
            for token in range(model.n_tokens)
        ]
    )
    matrices.setflags(write=False)
    return matrices


def enumerate_token_sequences(length: int) -> np.ndarray:
    """Enumerate the base-three token strings in lexicographic order."""
    if length <= 0:
        raise ValueError("sequence length must be positive")
    values = np.arange(VOCAB_SIZE**length, dtype=np.int64)
    powers = VOCAB_SIZE ** np.arange(length - 1, -1, -1, dtype=np.int64)
    return (values[:, None] // powers[None, :]) % VOCAB_SIZE


def sequence_probabilities(
    sequences: np.ndarray,
    matrices: np.ndarray | None = None,
) -> np.ndarray:
    """Compute exact stationary probabilities for edge-emission strings."""
    matrices = paper_transition_matrices() if matrices is None else matrices
    state_mass = np.broadcast_to(
        np.full(VOCAB_SIZE, 1.0 / VOCAB_SIZE),
        (len(sequences), VOCAB_SIZE),
    ).copy()
    for position in range(sequences.shape[1]):
        selected = matrices[sequences[:, position]]
        state_mass = np.einsum("bi,bij->bj", state_mass, selected)
    return state_mass.sum(axis=1)


def bayesian_beliefs(
    sequences: np.ndarray,
    matrices: np.ndarray | None = None,
    *,
    decimals: int | None = None,
) -> np.ndarray:
    """Return the exact hidden-state posterior after every observed token."""
    matrices = paper_transition_matrices() if matrices is None else matrices
    belief = np.broadcast_to(
        np.full(VOCAB_SIZE, 1.0 / VOCAB_SIZE),
        (len(sequences), VOCAB_SIZE),
    ).copy()
    history = []
    for position in range(sequences.shape[1]):
        selected = matrices[sequences[:, position]]
        belief = np.einsum("bi,bij->bj", belief, selected)
        belief /= belief.sum(axis=1, keepdims=True)
        history.append(belief.copy())
    result = np.stack(history, axis=1)
    return np.round(result, decimals) if decimals is not None else result


def bayesian_next_token_loss(
    sequences: np.ndarray,
    probabilities: np.ndarray,
    matrices: np.ndarray | None = None,
) -> float:
    """Exact probability-weighted next-token loss for the supplied paths."""
    matrices = paper_transition_matrices() if matrices is None else matrices
    token_probabilities = matrices.sum(axis=2).T
    beliefs = bayesian_beliefs(sequences[:, :-1], matrices)
    losses = np.empty((len(sequences), CONTEXT_LENGTH), dtype=np.float64)
    for position in range(CONTEXT_LENGTH):
        next_distribution = beliefs[:, position] @ token_probabilities
        targets = sequences[:, position + 1]
        losses[:, position] = -np.log(
            next_distribution[np.arange(len(sequences)), targets]
        )
    normalized = probabilities / probabilities.sum()
    return float(np.sum(normalized * losses.mean(axis=1)))


def split_probe_sequences(
    n_sequences: int,
    seed: SeedSource,
) -> tuple[np.ndarray, np.ndarray]:
    """Split whole sequences, keeping their ten positions in one partition."""
    if n_sequences < 2:
        raise ValueError("at least two probe sequences are required")
    rng = np.random.default_rng(as_seed_sequence(seed))
    shuffled = rng.permutation(n_sequences)
    train_size = max(1, int(n_sequences * PROBE_TRAIN_FRACTION))
    return shuffled[:train_size], shuffled[train_size:]


def build_model(seed: SeedSource, device: torch.device | str) -> PaperTransformer:
    """Construct a fresh paper-architecture model."""
    _seed_torch(seed)
    return PaperTransformer().to(device)


@torch.inference_mode()
def _validation_loss(
    model: PaperTransformer,
    sequences: torch.Tensor,
    probabilities: torch.Tensor,
) -> float:
    model.eval()
    total = torch.zeros((), dtype=torch.float64, device=sequences.device)
    for start in range(0, len(sequences), ANALYSIS_BATCH_SIZE):
        paths = sequences[start : start + ANALYSIS_BATCH_SIZE]
        weights = probabilities[start : start + ANALYSIS_BATCH_SIZE]
        logits = model(paths[:, :-1])
        per_token = F.cross_entropy(
            logits.flatten(0, 1),
            paths[:, 1:].flatten(),
            reduction="none",
        ).reshape(len(paths), CONTEXT_LENGTH)
        total += (per_token.mean(dim=1).double() * weights).sum()
    return float(total / probabilities.sum())


def _save_checkpoint(
    directory: Path,
    model: PaperTransformer,
    optimizer: torch.optim.Optimizer,
    step: int,
    *,
    final: bool = False,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    name = "checkpoint_final.pt" if final else f"checkpoint_{step:07d}.pt"
    path = directory / name
    payload: dict[str, Any] = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        payload["cuda_rng_state"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path)
    return path


def _restore_checkpoint(
    path: Path,
    model: PaperTransformer,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> int:
    if not path.is_file():
        raise ValueError("--resume-from must identify a supervised .pt checkpoint")
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state"])
    optimizer.load_state_dict(payload["optimizer_state"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if torch.cuda.is_available() and "cuda_rng_state" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])
    return int(payload["step"])


def _sample_training_chunk(
    support: torch.Tensor,
    cumulative_probability: torch.Tensor,
    *,
    n_batches: int,
) -> torch.Tensor:
    draws = torch.rand(
        n_batches,
        BATCH_SIZE,
        dtype=cumulative_probability.dtype,
        device=cumulative_probability.device,
    )
    indices = torch.searchsorted(cumulative_probability, draws)
    return support[indices]


def _train(
    context: RunContext,
    model: PaperTransformer,
    support: torch.Tensor,
    probabilities: torch.Tensor,
    bayes_loss: float,
    outputs: RunArtifacts,
) -> tuple[list[dict[str, Any]], Path, float]:
    target_updates = SMOKE_UPDATES if context.smoke else TRAIN_UPDATES
    validation_updates = (
        {0, target_updates} if context.smoke else VALIDATION_UPDATES
    )
    checkpoint_updates = (
        {0, target_updates} if context.smoke else CHECKPOINT_UPDATES
    )
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.0,
    )
    start_step = 0
    if context.resume_from is not None:
        start_step = _restore_checkpoint(
            context.resume_from,
            model,
            optimizer,
            support.device,
        )
        if start_step > target_updates:
            raise ValueError("checkpoint is beyond this run's training budget")

    cumulative = probabilities.cumsum(dim=0)
    cumulative = cumulative / cumulative[-1].clone()
    cumulative[-1] = 1.0
    training_model = model
    if support.device.type == "cuda" and not context.smoke:
        training_model = torch.compile(model, mode="reduce-overhead")

    records: list[dict[str, Any]] = []
    training_started = time.monotonic()
    if start_step == 0:
        initial_validation = _validation_loss(model, support, probabilities)
        record = {
            "phase": "validation",
            "optimizer_step": 0,
            "token_positions": 0,
            "validation_loss": initial_validation,
            "bayesian_loss": bayes_loss,
            "normalized_validation_loss": initial_validation / bayes_loss,
            "wall_seconds": 0.0,
        }
        outputs.append_result(record)
        records.append(record)
        _save_checkpoint(outputs.checkpoints_dir, model, optimizer, 0)

    model.train()
    running_loss = torch.zeros((), device=support.device)
    running_count = 0
    sampled_batches: torch.Tensor | None = None
    sample_index = TRAINING_SAMPLE_CHUNK

    for step in range(start_step + 1, target_updates + 1):
        if sampled_batches is None or sample_index >= len(sampled_batches):
            remaining = target_updates - step + 1
            chunk_size = min(TRAINING_SAMPLE_CHUNK, remaining)
            sampled_batches = _sample_training_chunk(
                support,
                cumulative,
                n_batches=chunk_size,
            )
            sample_index = 0
        paths = sampled_batches[sample_index]
        sample_index += 1

        logits = training_model(paths[:, :-1])
        loss = F.cross_entropy(
            logits.flatten(0, 1),
            paths[:, 1:].flatten(),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        running_loss += loss.detach()
        running_count += 1

        if step % (1 if context.smoke else LOG_EVERY) == 0:
            elapsed = time.monotonic() - training_started
            mean_loss = float(running_loss / running_count)
            record = {
                "phase": "training",
                "optimizer_step": step,
                "token_positions": step * BATCH_SIZE * CONTEXT_LENGTH,
                "training_loss": mean_loss,
                "updates_per_second": (step - start_step) / max(elapsed, 1e-9),
                "wall_seconds": elapsed,
            }
            outputs.append_result(record)
            records.append(record)
            running_loss.zero_()
            running_count = 0

        if step in validation_updates:
            value = _validation_loss(model, support, probabilities)
            elapsed = time.monotonic() - training_started
            record = {
                "phase": "validation",
                "optimizer_step": step,
                "token_positions": step * BATCH_SIZE * CONTEXT_LENGTH,
                "validation_loss": value,
                "bayesian_loss": bayes_loss,
                "normalized_validation_loss": value / bayes_loss,
                "wall_seconds": elapsed,
            }
            outputs.append_result(record)
            records.append(record)
            model.train()

        if step in checkpoint_updates and step != target_updates:
            _save_checkpoint(outputs.checkpoints_dir, model, optimizer, step)

    training_seconds = time.monotonic() - training_started
    final_checkpoint = _save_checkpoint(
        outputs.checkpoints_dir,
        model,
        optimizer,
        target_updates,
        final=True,
    )
    return records, final_checkpoint, training_seconds


@torch.inference_mode()
def _collect_activations(
    model: PaperTransformer,
    sequences: np.ndarray,
    device: torch.device,
    layer: int | str,
) -> np.ndarray:
    collected = []
    model.eval()
    for start in range(0, len(sequences), ANALYSIS_BATCH_SIZE):
        tokens = torch.as_tensor(
            sequences[start : start + ANALYSIS_BATCH_SIZE],
            dtype=torch.long,
            device=device,
        )
        _, residuals, normalized = model(tokens, return_residuals=True)
        activation = normalized if layer == "final_norm" else residuals[layer]
        collected.append(
            activation.flatten(0, 1).detach().cpu().numpy().astype(np.float32)
        )
    return np.concatenate(collected)


def _fit_ols(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    augmented = np.concatenate(
        [
            np.asarray(features, dtype=np.float64),
            np.ones((len(features), 1), dtype=np.float64),
        ],
        axis=1,
    )
    coefficient, _, _, _ = np.linalg.lstsq(
        augmented,
        np.asarray(targets, dtype=np.float64),
        rcond=None,
    )
    return coefficient


def _predict_rows(
    coefficient: np.ndarray,
    features: np.ndarray,
    rows: np.ndarray,
) -> np.ndarray:
    predictions = []
    for start in range(0, len(rows), 100_000):
        selected = np.asarray(
            features[rows[start : start + 100_000]],
            dtype=np.float64,
        )
        predictions.append(
            selected @ coefficient[:-1] + coefficient[-1]
        )
    return np.concatenate(predictions)


def _activation_statistics(features: np.ndarray) -> dict[str, Any]:
    values = np.asarray(features, dtype=np.float64)
    centered = values - values.mean(axis=0)
    covariance = centered.T @ centered / max(1, len(centered) - 1)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
    total = eigenvalues.sum()
    proportions = eigenvalues / total if total > 0 else eigenvalues
    positive = proportions[proportions > 0]
    effective_rank = (
        float(np.exp(-(positive * np.log(positive)).sum()))
        if len(positive)
        else 0.0
    )
    per_dimension_std = values.std(axis=0)
    return {
        "activation_rms": float(np.sqrt(np.square(values).mean())),
        "mean_l2_norm": float(np.linalg.norm(values, axis=1).mean()),
        "std_min": float(per_dimension_std.min()),
        "std_median": float(np.median(per_dimension_std)),
        "std_max": float(per_dimension_std.max()),
        "covariance_effective_rank": effective_rank,
        "covariance_eigenvalues": eigenvalues.tolist(),
    }


def _probe(
    context: RunContext,
    model: PaperTransformer,
    device: torch.device,
    *,
    probe_split_seed: SeedSource,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    all_sequences = enumerate_token_sequences(CONTEXT_LENGTH)
    if context.smoke:
        all_sequences = all_sequences[:729]
    beliefs = bayesian_beliefs(
        all_sequences,
        decimals=PROBE_TARGET_DECIMALS,
    ).reshape(-1, VOCAB_SIZE)
    train_sequences, test_sequences = split_probe_sequences(
        len(all_sequences),
        seed=probe_split_seed,
    )
    positions = np.arange(CONTEXT_LENGTH)
    train_rows = (
        train_sequences[:, None] * CONTEXT_LENGTH + positions[None, :]
    ).reshape(-1)
    test_rows = (
        test_sequences[:, None] * CONTEXT_LENGTH + positions[None, :]
    ).reshape(-1)

    layers: list[tuple[str, int | str]] = [
        (f"block_{index}_resid_post", index)
        for index in range(len(model.blocks))
    ]
    layers.append(("final_norm", "final_norm"))
    metrics: dict[str, Any] = {
        "split": {
            "unit": "whole_sequence",
            "train_fraction": PROBE_TRAIN_FRACTION,
            "n_train_sequences": len(train_sequences),
            "n_test_sequences": len(test_sequences),
            "n_train_pairs": len(train_rows),
            "n_test_pairs": len(test_rows),
        },
        "layers": {},
    }
    final_prediction = None
    test_beliefs = beliefs[test_rows]

    for name, layer in layers:
        features = _collect_activations(
            model,
            all_sequences,
            device,
            layer,
        )
        train_features = features[train_rows]
        coefficient = _fit_ols(train_features, beliefs[train_rows])
        predicted = _predict_rows(coefficient, features, test_rows)
        layer_metrics = {
            "mse": float(np.square(predicted - test_beliefs).mean()),
            "r2": r2_score(predicted, test_beliefs),
            "activation_dimension": features.shape[1],
            **_activation_statistics(train_features),
        }
        metrics["layers"][name] = layer_metrics
        if name == "block_3_resid_post":
            final_prediction = predicted

    if final_prediction is None:
        raise RuntimeError("final residual-stream probe was not evaluated")
    return metrics, test_beliefs, final_prediction


def _plot_training(
    records: list[dict[str, Any]],
    bayes_loss: float,
    path: Path,
) -> None:
    training = [record for record in records if record["phase"] == "training"]
    validation = [
        record for record in records if record["phase"] == "validation"
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    if training:
        axis.plot(
            [record["token_positions"] for record in training],
            [record["training_loss"] for record in training],
            alpha=0.55,
            linewidth=0.8,
            label="sampled training loss",
        )
    axis.plot(
        [record["token_positions"] for record in validation],
        [record["validation_loss"] for record in validation],
        marker="o",
        markersize=3,
        label="exact validation loss",
    )
    axis.axhline(
        bayes_loss,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Bayesian floor",
    )
    axis.set_xscale("symlog", linthresh=1_000)
    axis.set_xlabel("token-position losses")
    axis.set_ylabel("cross entropy (nats)")
    axis.set_title("MESS3 next-token training")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_probe_metrics(metrics: dict[str, Any], path: Path) -> None:
    names = list(metrics["layers"])
    mse = [metrics["layers"][name]["mse"] for name in names]
    figure, axis = plt.subplots(figsize=(7.2, 4.2))
    axis.bar(range(len(names)), mse)
    axis.axhline(
        FINAL_MSE_THRESHOLD,
        color="black",
        linestyle="--",
        linewidth=1,
        label="replication threshold",
    )
    axis.set_xticks(
        range(len(names)),
        [name.replace("_resid_post", "") for name in names],
        rotation=20,
        ha="right",
    )
    axis.set_ylabel("held-out belief MSE")
    axis.set_title("Affine probe across the residual stream")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_simplex(
    beliefs: np.ndarray,
    predicted: np.ndarray,
    path: Path,
) -> None:
    colors = np.clip(beliefs, 0.0, 1.0)
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 4.3))
    simplex_scatter(
        axes[0],
        beliefs,
        colors=colors,
        s=0.25,
        alpha=0.35,
        title="true Bayesian belief simplex (held out)",
        labels=("A", "B", "C"),
    )
    simplex_scatter(
        axes[1],
        predicted,
        colors=colors,
        s=0.25,
        alpha=0.35,
        title="decoded final residual stream (held out)",
        labels=("A", "B", "C"),
    )
    figure.tight_layout()
    figure.savefig(path, dpi=220)
    plt.close(figure)


def _write_findings(
    path: Path,
    summary: dict[str, Any],
    probe_metrics: dict[str, Any],
) -> None:
    final = probe_metrics["layers"]["block_3_resid_post"]
    path.write_text(
        "\n".join(
            [
                "# MESS3 supervised replication",
                "",
                f"Paper: {PAPER_URL}",
                "",
                f"- Scientific pass: `{summary['scientific_passed']}`",
                f"- Validation loss: `{summary['final_validation_loss']:.6f}` nats",
                f"- Bayesian floor: `{summary['bayesian_loss']:.6f}` nats",
                f"- Final residual held-out MSE: `{final['mse']:.8f}`",
                f"- Final residual held-out R²: `{final['r2']:.6f}`",
                f"- Training wall time: `{summary['timing_seconds']['training']:.1f}` s",
                f"- Probe and plotting time: `{summary['timing_seconds']['analysis']:.1f}` s",
                f"- Total experiment time: `{summary['timing_seconds']['total']:.1f}` s",
                "",
                "The affine probe was fit on whole sequences comprising 20% of",
                "the exhaustive context set. Every reported point and metric uses",
                "the disjoint 80% held-out sequence partition.",
                "",
            ]
        )
    )


def run(context: RunContext):
    """Train, probe, plot, and enforce the paper-informed replication checks."""
    if context.seed is None:
        raise ValueError("MESS3 supervised training requires an integer seed")
    streams = named_seed_sequences(context.seed, _STREAM_KEYS)
    total_started = time.monotonic()
    outputs = RunArtifacts.from_context(context)
    outputs.prepare()
    device = _device(context)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    model = build_model(streams["model_initialization"], device)
    _seed_torch(streams["training_sampling"])
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    matrices = paper_transition_matrices()
    paths_numpy = enumerate_token_sequences(PATH_LENGTH)
    path_probabilities = sequence_probabilities(paths_numpy, matrices)
    probability_sum = float(path_probabilities.sum())
    if not np.isclose(probability_sum, 1.0, atol=1e-10):
        raise RuntimeError(
            f"exhaustive MESS3 path probabilities sum to {probability_sum}"
        )
    bayes_loss = bayesian_next_token_loss(
        paths_numpy,
        path_probabilities,
        matrices,
    )

    support_numpy = paths_numpy
    support_probabilities = path_probabilities
    support = torch.as_tensor(
        support_numpy,
        dtype=torch.long,
        device=device,
    )
    probabilities = torch.as_tensor(
        support_probabilities,
        dtype=torch.float64,
        device=device,
    )

    recipe = {
        "paper": PAPER_URL,
        "seed": context.seed,
        "smoke": context.smoke,
        "device": str(device),
        "mess3": {
            "transition_matrices": matrices.tolist(),
            "initial_distribution": [1.0 / 3.0] * 3,
        },
        "model": {
            "vocab_size": VOCAB_SIZE,
            "context_length": CONTEXT_LENGTH,
            "d_model": 64,
            "d_head": 8,
            "d_mlp": 256,
            "n_layers": 4,
            "n_heads": 1,
            "activation": "relu",
            "parameter_count": parameter_count,
            "initialization_std": 0.02,
        },
        "training": {
            "optimizer": "SGD",
            "learning_rate": LEARNING_RATE,
            "weight_decay": 0.0,
            "batch_size": BATCH_SIZE,
            "optimizer_updates": SMOKE_UPDATES if context.smoke else TRAIN_UPDATES,
            "token_position_losses": (
                SMOKE_UPDATES if context.smoke else TRAIN_UPDATES
            )
            * BATCH_SIZE
            * CONTEXT_LENGTH,
            "path_support_size": len(paths_numpy),
        },
        "probe": {
            "context_support_size": VOCAB_SIZE**CONTEXT_LENGTH,
            "train_fraction": PROBE_TRAIN_FRACTION,
            "split_unit": "whole_sequence",
            "target_rounding_decimals": PROBE_TARGET_DECIMALS,
        },
    }
    outputs.write_json("resolved_recipe.json", recipe)

    records, final_checkpoint, training_seconds = _train(
        context,
        model,
        support,
        probabilities,
        bayes_loss,
        outputs,
    )
    analysis_started = time.monotonic()
    probe_metrics, held_out_beliefs, held_out_prediction = _probe(
        context,
        model,
        device,
        probe_split_seed=streams["probe_split"],
    )
    outputs.write_json("probe_metrics.json", probe_metrics)
    _plot_training(
        records,
        bayes_loss,
        context.results_dir / "fig_training_curve.png",
    )
    _plot_probe_metrics(
        probe_metrics,
        context.results_dir / "fig_probe_by_layer.png",
    )
    _plot_simplex(
        held_out_beliefs,
        held_out_prediction,
        context.results_dir / "fig_mess3_simplex.png",
    )
    analysis_seconds = time.monotonic() - analysis_started
    final_validation = [
        record
        for record in records
        if record["phase"] == "validation"
    ][-1]["validation_loss"]
    final_probe = probe_metrics["layers"]["block_3_resid_post"]
    scientific_passed = (
        None
        if context.smoke
        else bool(
            final_validation / bayes_loss <= VALIDATION_RATIO_THRESHOLD
            and final_probe["mse"] <= FINAL_MSE_THRESHOLD
        )
    )
    summary = {
        "scientific_passed": scientific_passed,
        "smoke": context.smoke,
        "optimizer_updates": SMOKE_UPDATES if context.smoke else TRAIN_UPDATES,
        "token_position_losses": (
            SMOKE_UPDATES if context.smoke else TRAIN_UPDATES
        )
        * BATCH_SIZE
        * CONTEXT_LENGTH,
        "final_validation_loss": final_validation,
        "bayesian_loss": bayes_loss,
        "normalized_validation_loss": final_validation / bayes_loss,
        "final_probe_mse": final_probe["mse"],
        "final_probe_r2": final_probe["r2"],
        "final_checkpoint": str(final_checkpoint),
        "timing_seconds": {
            "training": training_seconds,
            "analysis": analysis_seconds,
            "total": time.monotonic() - total_started,
        },
    }
    outputs.write_json("summary.json", summary)
    _write_findings(
        context.results_dir / "findings.md",
        summary,
        probe_metrics,
    )
    update_run_manifest(context, scientific_summary=summary)

    if scientific_passed is False:
        raise RuntimeError(
            "MESS3 replication missed its validation/probe thresholds; "
            "the failed Vast run should remain available for debugging"
        )
    return summary
