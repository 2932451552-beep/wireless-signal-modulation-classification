"""Small, explicit training and evaluation loops for classification models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Mapping

import torch
from torch import Tensor, nn


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    """Average loss and accuracy collected over one complete data-loader pass."""

    loss: float
    accuracy: float
    sample_count: int


@dataclass(frozen=True, slots=True)
class EpochRecord:
    """Training and validation measurements for one numbered epoch."""

    epoch: int
    train: EpochMetrics
    validation: EpochMetrics


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """History and best-epoch information returned by a complete training run."""

    history: tuple[EpochRecord, ...]
    best_epoch: int
    best_validation_loss: float
    stopped_early: bool


def _read_batch(batch: Mapping[str, Tensor], device: torch.device) -> tuple[Tensor, Tensor]:
    try:
        signals = batch["signal"]
        labels = batch["label"]
    except KeyError as error:
        raise ValueError("each batch must contain signal and label tensors") from error

    if signals.ndim != 3 or signals.shape[1] != 2:
        raise ValueError("batch signals must have shape (batch, 2, length)")
    if labels.ndim != 1 or labels.shape[0] != signals.shape[0]:
        raise ValueError("batch labels must match the signal batch size")
    return signals.to(device), labels.to(device)


def _finish_metrics(total_loss: float, total_correct: int, sample_count: int) -> EpochMetrics:
    if sample_count == 0:
        raise ValueError("data loader produced no samples")
    return EpochMetrics(
        loss=total_loss / sample_count,
        accuracy=total_correct / sample_count,
        sample_count=sample_count,
    )


def train_one_epoch(
    model: nn.Module,
    data_loader: Iterable[Mapping[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> EpochMetrics:
    """Update model parameters once for every batch in the loader."""

    loss_function = criterion or nn.CrossEntropyLoss()
    model.train()
    total_loss = 0.0
    total_correct = 0
    sample_count = 0

    for batch in data_loader:
        signals, labels = _read_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(signals)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_correct += int((logits.argmax(dim=1) == labels).sum())
        sample_count += batch_size

    return _finish_metrics(total_loss, total_correct, sample_count)


def evaluate(
    model: nn.Module,
    data_loader: Iterable[Mapping[str, Tensor]],
    *,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> EpochMetrics:
    """Measure a model without calculating gradients or changing its parameters."""

    loss_function = criterion or nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    total_correct = 0
    sample_count = 0

    with torch.inference_mode():
        for batch in data_loader:
            signals, labels = _read_batch(batch, device)
            logits = model(signals)
            loss = loss_function(logits, labels)

            batch_size = labels.shape[0]
            total_loss += float(loss) * batch_size
            total_correct += int((logits.argmax(dim=1) == labels).sum())
            sample_count += batch_size

    return _finish_metrics(total_loss, total_correct, sample_count)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    validation_metrics: EpochMetrics,
) -> Path:
    """Save model and optimizer state for the best validation epoch."""

    destination = Path(path)
    if destination.exists() and destination.is_dir():
        raise ValueError("checkpoint path must be a file, not a directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "validation_loss": validation_metrics.loss,
            "validation_accuracy": validation_metrics.accuracy,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        destination,
    )
    return destination


def fit(
    model: nn.Module,
    train_loader: Iterable[Mapping[str, Tensor]],
    validation_loader: Iterable[Mapping[str, Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    epochs: int,
    patience: int | None = None,
    min_delta: float = 0.0,
    checkpoint_path: str | Path | None = None,
    criterion: nn.Module | None = None,
    on_epoch: Callable[[EpochRecord], None] | None = None,
) -> TrainingResult:
    """Train, select the lowest validation-loss epoch, and optionally stop early."""

    if epochs <= 0:
        raise ValueError("epochs must be greater than zero")
    if patience is not None and patience <= 0:
        raise ValueError("patience must be greater than zero when provided")
    if not math.isfinite(min_delta) or min_delta < 0.0:
        raise ValueError("min_delta must be a finite non-negative number")

    history: list[EpochRecord] = []
    best_epoch = 0
    best_validation_loss = math.inf
    best_model_state: dict[str, Tensor] | None = None
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            criterion=criterion,
        )
        validation_metrics = evaluate(
            model,
            validation_loader,
            device=device,
            criterion=criterion,
        )
        record = EpochRecord(epoch, train_metrics, validation_metrics)
        history.append(record)
        if on_epoch is not None:
            on_epoch(record)

        improved = validation_metrics.loss < best_validation_loss - min_delta
        if improved:
            best_epoch = epoch
            best_validation_loss = validation_metrics.loss
            best_model_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
            if checkpoint_path is not None:
                save_checkpoint(
                    checkpoint_path,
                    model,
                    optimizer,
                    epoch=epoch,
                    validation_metrics=validation_metrics,
                )
        else:
            epochs_without_improvement += 1
            if patience is not None and epochs_without_improvement >= patience:
                stopped_early = True
                break

    if best_model_state is None:
        raise RuntimeError("training did not produce a finite best validation result")
    model.load_state_dict(best_model_state)
    return TrainingResult(
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_loss=best_validation_loss,
        stopped_early=stopped_early,
    )
