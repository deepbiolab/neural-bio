"""Training utilities and functions.

Author: Tim Lin
Organization: DeepBioLab
License: MIT License
"""

import os
from typing import Dict, List, Optional, Tuple, Any, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .inference import simulate

from .models.hybrid_model import HybridModel
from .models.loss import AdaptiveWeightedHybridLoss
from .utils.visualization import save_training_state, create_training_animation_from_history

def save_checkpoint(checkpoint: Dict[str, Any], save_dir: str, model_name: str) -> None:
    """Save a model checkpoint.

    Args:
        checkpoint: Checkpoint dictionary containing model state and metadata
        save_dir: Directory where the checkpoint file will be saved
        model_name: Name of the model file
    """
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{model_name}.pth")
    torch.save(checkpoint, model_path)
    print(f"Saved model with name: {model_path}")


def load_checkpoint(
    model: HybridModel,
    criterion: Optional[AdaptiveWeightedHybridLoss] = None,
    save_dir: Optional[str] = None,
    model_name: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Union[
    nn.Module,
    Tuple[nn.Module, nn.Module, optim.Optimizer, optim.lr_scheduler._LRScheduler],
]:
    """Load model and optionally criterion, optimizer, and scheduler from checkpoint.

    Args:
        model: The model to load weights into
        criterion: The loss function to load state into
        save_dir: Directory where the checkpoint file is stored
        model_name: Name of the model file
        device: Target device to load the model to

    Returns:
        Either the loaded model or a tuple of (model, criterion, optimizer, scheduler)
    """
    checkpoint_path = os.path.join(save_dir, f"{model_name}.pth")

    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)
        model = model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    epoch = checkpoint["epoch"]
    val_loss = checkpoint["val_loss"]
    states_weight = (
        checkpoint["criterion_state_dict"]["states_weight"].detach().cpu().numpy()
    )
    print(f"Loaded best model from epoch {epoch} with validation loss {val_loss:.4f}")
    print(f"Optimal states weights: {states_weight}")

    if criterion is not None:
        criterion.load_state_dict(checkpoint["criterion_state_dict"])
        optimizer = optim.Adam(
            list(model.parameters()) + list(criterion.parameters()),
            lr=0.001,
            weight_decay=1e-5,
        )
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=20,
        )
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        return model, criterion, optimizer, scheduler
    else:
        return model


def train_one_model(
    model: HybridModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: AdaptiveWeightedHybridLoss,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    num_epochs: int = 100,
    time_step: float = 24,
    save_dir: str = "checkpoints",
    log_dir: str = "logs",
    model_name: str = "best_model",
    device: Optional[str] = None,
    animation_freq: int = 10, 
) -> Tuple[List[float], List[float]]:
    """Train a single neural network model.

    Args:
        model: Neural network model to be trained
        train_loader: DataLoader containing training data
        val_loader: DataLoader containing validation data
        criterion: Loss function
        optimizer: Optimizer for model parameter updates
        scheduler: Learning rate scheduler
        num_epochs: Maximum number of training epochs
        time_step: Time step size for numerical integration
        save_dir: Directory to save model checkpoints
        model_name: Name prefix for saved model files
        device: Computing device

    Returns:
        Tuple containing lists of training and validation losses
    """
    if device is None:
        device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )

    device = torch.device(device)
    model.to(device)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    patience_counter = 0

    prediction_history = []
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_data_loss = 0.0
        train_physics_loss = 0.0

        for batch in train_loader:
            X_batch, F_batch, Y_batch, V_batch, Z_batch = [b.to(device) for b in batch]
            batch_size, time_steps, num_vars = X_batch.shape

            if time_steps > 1:
                t = torch.zeros(batch_size, dtype=torch.float32, device=device)
                derivatives_pred = model(
                    t,
                    X_batch[:, :-1],
                    F_batch[:, :-1],
                    V_batch[:, :-1],
                    Z_batch[:, :-1],
                )

                loss, data_loss, physics_loss = criterion(
                    X_batch[:, 1:],
                    X_batch[:, :-1] + derivatives_pred * time_step,
                    derivatives_pred,
                    Y_batch[:, :-1],
                )
            else:
                derivatives_pred = model(t, X_batch, F_batch, V_batch)
                loss, data_loss, physics_loss = criterion(
                    X_batch,
                    X_batch,
                    derivatives_pred,
                    Y_batch,
                )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_data_loss += data_loss.item()
            train_physics_loss += physics_loss.item()

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                X_batch, F_batch, Y_batch, V_batch, Z_batch = [b.to(device) for b in batch]
                t = torch.zeros(X_batch.shape[0], dtype=torch.float32, device=device)
                derivatives_pred = model(t, X_batch, F_batch, V_batch, Z_batch)
                loss, _, _ = criterion(X_batch, X_batch, derivatives_pred, Y_batch)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        train_data_loss /= len(train_loader)
        train_physics_loss /= len(train_loader)
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "criterion_state_dict": criterion.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
            }
            save_checkpoint(checkpoint, save_dir, model_name)
            print(f"Saved new best model with validation loss: {val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= 50:
                print("Early stopping triggered")
                break

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {train_loss:.4f}, "
            f"Data Loss: {train_data_loss:.4f}, "
            f"Physics Loss: {train_physics_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}"
        )

        if epoch % animation_freq == 0 or epoch == num_epochs - 1:
            predictions = save_training_state(
                model=model,
                simulate=simulate,
                val_dataset=val_loader.dataset,
                device=device
            )
            prediction_history.append({
                'epoch': epoch + 1,
                'predictions': predictions
            })

    final_checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "criterion_state_dict": criterion.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": val_loss,
        "train_loss": train_loss,
    }

    save_checkpoint(final_checkpoint, save_dir, "final_model")

    animation_path = os.path.join(log_dir, "training_animation.gif")
    create_training_animation_from_history(
        prediction_history=prediction_history,
        dataset=val_loader.dataset,
        save_path=animation_path,
        fps=3
    )
    return train_losses, val_losses