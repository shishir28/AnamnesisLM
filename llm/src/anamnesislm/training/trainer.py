from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from anamnesislm.data.records import EncodedRecord
from anamnesislm.model.gpt import GPTModel
from anamnesislm.training.config import TrainingConfig

Batch = dict[str, torch.Tensor]  # what ExtractionDataPipeline.collate_batch returns


def compute_loss(model: GPTModel, batch: Batch, ignore_index: int) -> torch.Tensor:
    """Mean cross-entropy over response tokens.

    Logits at position t predict the token at t + 1, so compare logits[:, :-1] with
    labels[:, 1:]. Prompt and padding positions carry ignore_index and are skipped.
    """

    logits = model(batch["input_ids"])
    labels = batch["labels"]
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
        ignore_index=ignore_index,
    )
    return loss


def evaluate_model(
    model: GPTModel,
    loader: DataLoader[EncodedRecord],
    config: TrainingConfig,
    ignore_index: int,
    max_batches: int | None = None,
) -> float:
    """Average loss over the loader in eval mode, without gradients."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in loader:
            if max_batches is not None and num_batches >= max_batches:
                break

            batch = {k: v.to(config.device) for k, v in batch.items()}
            loss = compute_loss(model, batch, ignore_index=ignore_index)
            total_loss += loss.item()
            num_batches += 1
    model.train(was_training)
    return total_loss / num_batches if num_batches > 0 else float("nan")


def save_checkpoint(
    path: Path,
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    validation_loss: float,
    tokenizer: str | None = None,
) -> None:
    """Save model and optimizer state, configs, training position, and validation loss.

    Writes to a temporary file first, so a crash mid-save never corrupts an existing checkpoint.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "validation_loss": validation_loss,
        "model_config": asdict(model.config),
    }
    if tokenizer is not None:
        checkpoint["tokenizer"] = tokenizer

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(
    path: Path, model: GPTModel, optimizer: torch.optim.Optimizer | None = None
) -> dict[str, Any]:
    """Load weights (and optimizer state, if given) into existing objects; return the checkpoint.

    Move the model to its device first: optimizer state is loaded onto the parameters' device.
    """
    checkpoint: dict[str, Any] = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def _evaluate_and_save(
    model: GPTModel,
    optimizer: torch.optim.Optimizer,
    validation_loader: DataLoader[EncodedRecord],
    config: TrainingConfig,
    ignore_index: int,
    epoch: int,
    step: int,
    best_validation_loss: float,
    train_loss: float,
) -> float:
    """Evaluate, log both losses, and save a checkpoint if validation loss improved.

    train_loss is the average over this epoch's batches so far. Returns the best validation loss.
    """
    validation_loss = evaluate_model(
        model,
        validation_loader,
        config,
        ignore_index=ignore_index,
        max_batches=config.eval_iter,
    )
    print(
        f"Epoch {epoch}, Step {step}: "
        f"train loss {train_loss:.4f}, validation loss {validation_loss:.4f}"
    )
    if validation_loss >= best_validation_loss:
        return best_validation_loss

    save_checkpoint(
        path=config.checkpoint_dir / "best.pt",
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        step=step,
        validation_loss=validation_loss,
        tokenizer="gpt2",
    )
    return validation_loss


def train(
    model: GPTModel,
    train_loader: DataLoader[EncodedRecord],
    validation_loader: DataLoader[EncodedRecord],
    config: TrainingConfig,
    ignore_index: int,
) -> None:
    """Train for config.num_epochs, evaluating every config.eval_freq steps on config.eval_iter batches.

    Per batch: move to device -> compute_loss -> zero_grad -> backward -> clip -> step.
    Save a checkpoint whenever validation loss improves.
    """
    torch.manual_seed(config.seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    global_step = 0
    best_validation_loss = float("inf")
    model.to(config.device)
    for epoch in range(config.num_epochs):
        model.train()
        # Summed on the device; calling .item() every step would stall the GPU.
        epoch_loss_sum = torch.zeros((), device=config.device)
        epoch_batches = 0
        for batch in train_loader:
            batch = {k: v.to(config.device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = compute_loss(model, batch, ignore_index=ignore_index)

            if global_step % 50 == 0:
                print(f"Epoch {epoch}, Step {global_step}, Loss: {loss.item():.4f}")

            torch.autograd.backward(loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            global_step += 1
            epoch_loss_sum += loss.detach()
            epoch_batches += 1

            if global_step % config.eval_freq == 0:
                best_validation_loss = _evaluate_and_save(
                    model,
                    optimizer,
                    validation_loader,
                    config,
                    ignore_index,
                    epoch,
                    global_step,
                    best_validation_loss,
                    train_loss=epoch_loss_sum.item() / epoch_batches,
                )

        # End of epoch: short runs never reach eval_freq, so always check here too.
        best_validation_loss = _evaluate_and_save(
            model,
            optimizer,
            validation_loader,
            config,
            ignore_index,
            epoch,
            global_step,
            best_validation_loss,
            train_loss=epoch_loss_sum.item() / max(epoch_batches, 1),
        )
