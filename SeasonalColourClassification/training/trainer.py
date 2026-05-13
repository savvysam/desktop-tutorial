"""Training logic for seasonal color classification models with Neptune integration.

This trainer performs a two-stage training when fine-tuning is enabled:
  1. Initial training for INITIAL_EPOCHS epochs (training only the classifier layers).
  2. Fine-tuning for the remaining epochs (unfreezing the entire network).

If fine-tuning is disabled, the whole network is trained for all epochs.

Resume support
--------------
After every epoch a resume checkpoint is written to ``resume_path`` (passed to
``train()``).  The checkpoint stores enough state to restart training exactly
where it left off:

  - model and optimizer weights
  - training history (losses / accuracies / lr)
  - best-val-loss tracking state
  - which phase (``'initial'`` or ``'finetune'``) the run was in
  - how many epochs had been completed

On restart the caller passes the same ``resume_path`` to ``train()`` and the
method prints::

    Resuming from epoch N (phase: <phase>)
"""

import os
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..config import (
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    FINE_TUNE,
    FINE_TUNE_LR,
    INITIAL_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
)
from ..evaluation.metrics import calculate_metrics
from ..models.base_model import BaseModel


class Trainer:
    """Trainer class for the seasonal color classification model."""

    def __init__(
        self,
        model: BaseModel,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module = nn.CrossEntropyLoss(),
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: str = DEVICE,
        early_stopping_patience: int = EARLY_STOPPING_PATIENCE,
        save_dir: str = "checkpoints",
        neptune_run: Optional[object] = None,
        custom_learning_rate: float = None,
        custom_weight_decay: float = None,
        custom_fine_tune_lr: float = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion

        self.learning_rate = custom_learning_rate if custom_learning_rate is not None else LEARNING_RATE
        self.weight_decay = custom_weight_decay if custom_weight_decay is not None else WEIGHT_DECAY
        self.fine_tune_lr = custom_fine_tune_lr if custom_fine_tune_lr is not None else FINE_TUNE_LR

        self.device = device
        self.early_stop = early_stopping_patience
        self.save_dir = save_dir
        self.neptune_run = neptune_run
        self.model.to(self.device)
        os.makedirs(save_dir, exist_ok=True)

        self.history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "train_acc": [],
            "val_acc": [],
            "lr": [],
        }

        self.best_val_loss = float("inf")
        self.best_epoch = -1
        self.epochs_without_improvement = 0
        self.best_checkpoint_path = os.path.join(self.save_dir, "best_model.pth")

        # For fine-tuning, initially freeze base layers (train classifier only)
        if FINE_TUNE:
            for name, param in self.model.named_parameters():
                if ("fc" in name) or ("classifier" in name) or ("head" in name):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            self.optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )

    # ------------------------------------------------------------------
    # Per-epoch helpers
    # ------------------------------------------------------------------

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        with tqdm(self.train_loader, desc="Training", leave=False) as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                pbar.set_postfix({"loss": loss.item(), "acc": correct / total})

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate_epoch(self) -> Tuple[float, float, Dict]:
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss = running_loss / len(self.val_loader.dataset)
        metrics = calculate_metrics(
            np.array(all_preds),
            np.array(all_labels),
            self.val_loader.dataset.classes,
        )
        return val_loss, metrics["accuracy"], metrics

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        """Save the *best-model* checkpoint (model + optimizer state)."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "epoch": self.best_epoch,
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> None:
        """Load a best-model checkpoint (model + optimizer state)."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.best_val_loss = checkpoint["best_val_loss"]
        self.best_epoch = checkpoint["epoch"]

    def save_resume_checkpoint(
        self,
        path: str,
        completed_epochs: int,
        phase: str,
        total_epochs: int,
        initial_epochs: int,
    ) -> None:
        """Save a per-epoch resume checkpoint with full training state.

        Parameters
        ----------
        path:
            Destination path (e.g. ``checkpoints/finetune_resume.pth``).
        completed_epochs:
            Number of epochs that have been fully completed so far (1-based
            count so the *next* epoch to run is ``completed_epochs + 1``).
        phase:
            ``'initial'`` or ``'finetune'`` — the phase the last epoch
            belonged to.
        total_epochs:
            Total epochs for the full run (passed through for reference).
        initial_epochs:
            Number of head-only epochs (passed through for reference).
        """
        # scheduler_state_dict is None when no scheduler is active; the field is
        # included so that subclasses or future trainers that add a scheduler can
        # persist and restore it without changing the checkpoint schema.
        scheduler_state = (
            self.scheduler.state_dict()
            if hasattr(self, "scheduler") and self.scheduler is not None
            else None
        )
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": scheduler_state,
            "history": self.history,
            "best_val_loss": self.best_val_loss,
            "best_epoch": self.best_epoch,
            "epochs_without_improvement": self.epochs_without_improvement,
            "completed_epochs": completed_epochs,
            "phase": phase,
            "total_epochs": total_epochs,
            "initial_epochs": initial_epochs,
        }
        torch.save(checkpoint, path)

    def _restore_from_resume(self, path: str) -> Tuple[int, str]:
        """Load a resume checkpoint and return (completed_epochs, phase)."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.history = ckpt.get("history", self.history)
        self.best_val_loss = ckpt.get("best_val_loss", self.best_val_loss)
        self.best_epoch = ckpt.get("best_epoch", self.best_epoch)
        self.epochs_without_improvement = ckpt.get("epochs_without_improvement", 0)
        completed_epochs = ckpt.get("completed_epochs", 0)
        phase = ckpt.get("phase", "initial")

        # Restore the optimizer.  When resuming in the fine-tune phase the
        # saved optimizer covers *all* parameters, so we must unfreeze the
        # model before calling load_state_dict to avoid a shape mismatch.
        if phase == "finetune":
            for param in self.model.parameters():
                param.requires_grad = True
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.fine_tune_lr,
                weight_decay=self.weight_decay,
            )
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return completed_epochs, phase

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        num_epochs: int = 50,
        resume_path: Optional[str] = None,
    ) -> Dict[str, list]:
        """Run the two-stage training loop, optionally resuming from a checkpoint.

        Parameters
        ----------
        num_epochs:
            Total number of training epochs.
        resume_path:
            Path to a resume checkpoint written by :meth:`save_resume_checkpoint`.
            When the file exists, training continues from where it left off.
        """
        if FINE_TUNE:
            initial_epochs = INITIAL_EPOCHS
            fine_tune_epochs = num_epochs - INITIAL_EPOCHS
        else:
            initial_epochs = num_epochs
            fine_tune_epochs = 0

        # ---- Resume detection ----------------------------------------
        completed_epochs = 0
        resume_phase = "initial"
        if resume_path and os.path.exists(resume_path):
            completed_epochs, resume_phase = self._restore_from_resume(resume_path)
            print(
                f"Resuming from epoch {completed_epochs + 1} (phase: {resume_phase})",
                flush=True,
            )
        else:
            if FINE_TUNE:
                print(
                    f"Starting initial training for {initial_epochs} epochs "
                    f"(training classifier layers only)...",
                    flush=True,
                )
            else:
                print(
                    f"Starting training for {num_epochs} epochs "
                    f"(training whole network)...",
                    flush=True,
                )

        start_time = time.time()

        # ---- Initial Training Phase ----------------------------------
        if resume_phase == "initial":
            initial_start = completed_epochs  # epochs already done in this phase
            for epoch in range(initial_start, initial_epochs):
                current_epoch = epoch + 1  # 1-based display epoch
                train_loss, train_acc = self.train_epoch()
                val_loss, val_acc, metrics = self.validate_epoch()
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {current_epoch}/{num_epochs} - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                    f"LR: {current_lr:.6f}",
                    flush=True,
                )

                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["train_acc"].append(train_acc)
                self.history["val_acc"].append(val_acc)
                self.history["lr"].append(current_lr)

                self._log_neptune(train_loss, train_acc, val_loss, val_acc, current_lr, metrics)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = current_epoch - 1
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(self.best_checkpoint_path)
                    print(
                        f"Saved best model (initial phase) with val_loss: {val_loss:.4f}",
                        flush=True,
                    )
                else:
                    self.epochs_without_improvement += 1

                # Per-epoch resume checkpoint
                if resume_path:
                    self.save_resume_checkpoint(
                        resume_path,
                        completed_epochs=current_epoch,
                        phase="initial",
                        total_epochs=num_epochs,
                        initial_epochs=initial_epochs,
                    )

                if self.epochs_without_improvement >= self.early_stop:
                    print(
                        f"Early stopping triggered during initial training "
                        f"after {current_epoch} epochs",
                        flush=True,
                    )
                    break

        # ---- Fine-Tuning Phase ---------------------------------------
        if FINE_TUNE and fine_tune_epochs > 0:
            if resume_phase == "initial":
                # Transition: reload best checkpoint before unfreezing
                print(
                    "Reloading best model checkpoint for fine-tuning "
                    "(unfreezing entire network)...",
                    flush=True,
                )
                self.load_checkpoint(self.best_checkpoint_path)
                for param in self.model.parameters():
                    param.requires_grad = True
                self.optimizer = optim.Adam(
                    self.model.parameters(),
                    lr=self.fine_tune_lr,
                    weight_decay=self.weight_decay,
                )
                self.epochs_without_improvement = 0
                ft_start = 0
            else:
                # Resuming mid-fine-tune: optimizer and epochs_without_improvement
                # are already correctly restored by _restore_from_resume; do not reset.
                ft_start = completed_epochs - initial_epochs

            print(f"Starting fine-tuning for {fine_tune_epochs} epochs...", flush=True)

            for ft_epoch in range(ft_start, fine_tune_epochs):
                current_epoch = initial_epochs + ft_epoch + 1
                train_loss, train_acc = self.train_epoch()
                val_loss, val_acc, metrics = self.validate_epoch()
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Fine-tune Epoch {ft_epoch + 1}/{fine_tune_epochs} "
                    f"(Total Epoch {current_epoch}/{num_epochs}) - "
                    f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                    f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                    f"LR: {current_lr:.6f}",
                    flush=True,
                )

                self.history["train_loss"].append(train_loss)
                self.history["val_loss"].append(val_loss)
                self.history["train_acc"].append(train_acc)
                self.history["val_acc"].append(val_acc)
                self.history["lr"].append(current_lr)

                self._log_neptune(train_loss, train_acc, val_loss, val_acc, current_lr, metrics)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_epoch = current_epoch - 1
                    self.epochs_without_improvement = 0
                    self.save_checkpoint(self.best_checkpoint_path)
                    print(
                        f"Saved best model during fine-tuning with val_loss: {val_loss:.4f}",
                        flush=True,
                    )
                else:
                    self.epochs_without_improvement += 1

                # Per-epoch resume checkpoint
                if resume_path:
                    self.save_resume_checkpoint(
                        resume_path,
                        completed_epochs=current_epoch,
                        phase="finetune",
                        total_epochs=num_epochs,
                        initial_epochs=initial_epochs,
                    )

                if self.epochs_without_improvement >= self.early_stop:
                    print(
                        f"Early stopping triggered during fine-tuning "
                        f"after {ft_epoch + 1} epochs",
                        flush=True,
                    )
                    break

        total_time = time.time() - start_time
        print(f"Training completed in {total_time / 60:.2f} minutes", flush=True)
        print(
            f"Best epoch: {self.best_epoch + 1} with val_loss: {self.best_val_loss:.4f}",
            flush=True,
        )
        return self.history

    # ------------------------------------------------------------------
    # Neptune logging (extracted to reduce duplication)
    # ------------------------------------------------------------------

    def _log_neptune(
        self,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        current_lr: float,
        metrics: Dict,
    ) -> None:
        if self.neptune_run is None:
            return
        self.neptune_run["train/loss"].append(train_loss)
        self.neptune_run["train/accuracy"].append(train_acc)
        self.neptune_run["val/loss"].append(val_loss)
        self.neptune_run["val/accuracy"].append(val_acc)
        self.neptune_run["lr"].append(current_lr)
        self.neptune_run["eval/accuracy"].append(metrics["accuracy"])
        self.neptune_run["eval/precision/micro"].append(metrics["precision"]["micro"])
        self.neptune_run["eval/precision/macro"].append(metrics["precision"]["macro"])
        self.neptune_run["eval/precision/weighted"].append(metrics["precision"]["weighted"])
        self.neptune_run["eval/recall/micro"].append(metrics["recall"]["micro"])
        self.neptune_run["eval/recall/macro"].append(metrics["recall"]["macro"])
        self.neptune_run["eval/recall/weighted"].append(metrics["recall"]["weighted"])
        self.neptune_run["eval/f1/micro"].append(metrics["f1"]["micro"])
        self.neptune_run["eval/f1/macro"].append(metrics["f1"]["macro"])
        self.neptune_run["eval/f1/weighted"].append(metrics["f1"]["weighted"])
