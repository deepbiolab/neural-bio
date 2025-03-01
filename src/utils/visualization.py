"""Visualization utilities for training and prediction results."""

import os
from typing import List, Optional, Union, Dict, Any
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from IPython.display import HTML

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    title: str = "Training and Validation Loss",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot training and validation loss curves over epochs.

    Args:
        train_losses: List of training loss values for each epoch
        val_losses: List of validation loss values for each epoch
        title: Title of the plot
        save_path: Directory path to save the plot
        show: Whether to display the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(True)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(
            os.path.join(save_path, "loss_curves.png"), 
            dpi=300, 
            bbox_inches="tight"
        )

    if show:
        plt.show()
    else:
        plt.close()

def plot_predicted_profiles(
    X_pred: Union[np.ndarray, torch.Tensor],
    X_columns: List[str],
    X_true: Optional[Union[np.ndarray, torch.Tensor]] = None,
    select_runs: List[int] = [0, 1, 2, 3, 4],
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot predicted profiles with optional comparison to ground truth values.

    Args:
        X_pred: Predicted values array [batch_size, time_steps, num_vars]
        X_columns: List of variable names
        X_true: Optional ground truth values array
        select_runs: List of run indices to plot
        title: Optional title for the figure
        save_path: Optional path to save the figure
        show: Whether to display the plot
    """
    if torch.is_tensor(X_pred):
        X_pred = X_pred.numpy()
    if X_true is not None and torch.is_tensor(X_true):
        X_true = X_true.numpy()

    num_vars = len(X_columns)
    fig, axes = plt.subplots(
        len(select_runs), num_vars, figsize=(16, 4 * len(select_runs))
    )

    time_points = np.arange(0, X_pred.shape[1])

    for i, run_idx in enumerate(select_runs):
        for j, var in enumerate(X_columns):
            if len(select_runs) == 1:
                ax = axes[j] if num_vars > 1 else axes
            else:
                ax = axes[i, j] if num_vars > 1 else axes[i]

            if X_true is not None:
                ax.plot(
                    time_points, 
                    X_true[run_idx, :, j], 
                    "o-",
                    alpha=0.5, 
                    label="True", 
                    color="blue"
                )

            ax.plot(
                time_points,
                X_pred[run_idx, :, j],
                "s--",
                alpha=0.5, 
                label="Predicted" if X_true is not None else "Profile",
                color="red",
            )
            ax.set_title(f"Run {run_idx} - {var}")
            ax.set_xlabel("Time (days)")
            ax.set_ylabel(var)
            ax.grid(True)
            ax.legend()

    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        filename = f"{title}.png" if title else "profiles.png"
        plt.savefig(os.path.join(save_path, filename), dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()


def create_training_animation(
    prediction_history: List[np.ndarray],
    X_true: Union[np.ndarray, torch.Tensor],
    X_columns: List[str],
    epochs: List[int],
    select_runs: List[int] = [4],
    title: str = "Training Progress Animation on Validation Set",
    save_path: Optional[str] = None,
    fps: int = 2,
) -> Optional[HTML]:
    """Create an animation showing how predictions improve during training.

    Args:
        prediction_history: List of prediction arrays from different epochs
        X_true: Ground truth values
        X_columns: List of variable names
        epochs: List of epoch numbers corresponding to predictions
        select_runs: List of run indices to animate
        title: Title for the animation
        save_path: Path to save the animation file
        fps: Frames per second for the animation

    Returns:
        HTML: Animation object for notebook display (if save_path is None)
    """
    if torch.is_tensor(X_true):
        X_true = X_true.numpy()

    # Setup the figure and axes
    num_vars = len(X_columns)
    fig, axes = plt.subplots(
        len(select_runs), num_vars, 
        figsize=(4*num_vars, 3*len(select_runs))
    )
    if len(select_runs) == 1 and num_vars == 1:
        axes = np.array([[axes]])
    elif len(select_runs) == 1:
        axes = axes.reshape(1, -1)
    elif num_vars == 1:
        axes = axes.reshape(-1, 1)

    # Get time points
    time_points = np.arange(X_true.shape[1])

    # Set y-axis limits
    y_mins = np.min(X_true, axis=(0, 1))
    y_maxs = np.max(X_true, axis=(0, 1))
    y_margins = (y_maxs - y_mins) * 0.1
    
    # Initialize plots
    lines = {}
    epoch_text = fig.text(0.02, 0.98, '', fontsize=12)
    
    for i, run_idx in enumerate(select_runs):
        for j, var in enumerate(X_columns):
            ax = axes[i, j]
            # Plot ground truth
            ax.plot(time_points, X_true[run_idx, :, j], 'o-', 
                   label='True', color='blue', alpha=0.5)
            # Initialize prediction line
            pred_line, = ax.plot([], [], 's--', label='Predicted', 
                               color='red', alpha=0.5)
            lines[(i, j)] = pred_line
            
            ax.set_title(f'{var}')
            ax.set_xlabel('Time (days)')
            ax.set_ylabel(var)
            ax.set_ylim(y_mins[j] - y_margins[j], y_maxs[j] + y_margins[j])
            ax.grid(True)
            ax.legend()

    plt.tight_layout()
    
    def init():
        """Initialize animation."""
        for line in lines.values():
            line.set_data([], [])
        epoch_text.set_text('')
        return list(lines.values()) + [epoch_text]
    
    def update(frame):
        """Update animation frame."""
        predictions = prediction_history[frame]
        epoch = epochs[frame]
        
        for i, run_idx in enumerate(select_runs):
            for j in range(num_vars):
                line = lines[(i, j)]
                line.set_data(time_points, predictions[run_idx, :, j])
        
        epoch_text.set_text(f'Epoch: {epoch}')
        return list(lines.values()) + [epoch_text]
    
    # Create animation
    anim = FuncAnimation(
        fig, update, init_func=init, 
        frames=len(prediction_history),
        interval=1000/fps, blit=True
    )
    
    if save_path:
        # Save animation as GIF
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        plt.close()
        return None
    
    # Return HTML object for notebook display
    plt.close()
    return HTML(anim.to_jshtml())


def save_training_state(
    model: torch.nn.Module,
    simulate: callable,
    val_dataset: Any,
    device: str = 'cpu'
) -> np.ndarray:
    """Save model state and generate predictions for animation.

    Args:
        model: Current model state
        epoch: Current epoch number
        val_dataset: Validation dataset
        save_dir: Directory to save predictions
        device: Computing device

    Returns:
        np.ndarray: Current predictions
    """
    model.eval()
    with torch.no_grad():
        sim_data = val_dataset.get_simulation_data()
        # Move all tensors to device
        sim_data = {
            k: v.to(device) if torch.is_tensor(v) else v 
            for k, v in sim_data.items()
        }

        predictions = simulate(model=model, **sim_data)
        
    if torch.is_tensor(predictions):
        predictions = predictions.cpu().numpy()
    
    return predictions


def create_training_animation_from_history(
    prediction_history: List[Dict[str, Any]],
    dataset: Any,
    save_path: Optional[str] = None,
    **kwargs
) -> Optional[HTML]:
    """Create training animation from saved prediction history.

    Args:
        prediction_history: List of dictionaries containing predictions and epochs
        dataset: Dataset containing ground truth values
        save_path: Path to save the animation
        **kwargs: Additional arguments to pass to create_training_animation

    Returns:
        HTML: Animation object for notebook display (if save_path is None)
    """
    predictions = [h['predictions'] for h in prediction_history]
    epochs = [h['epoch'] for h in prediction_history]
    
    return create_training_animation(
        prediction_history=predictions,
        X_true=dataset.X,
        X_columns=dataset.X_columns,
        epochs=epochs,
        save_path=save_path,
        **kwargs
    )