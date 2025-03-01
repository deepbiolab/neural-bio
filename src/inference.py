"""Inference utilities and functions.

Author: Tim Lin
Organization: DeepBioLab
License: MIT License
"""

from typing import Dict, List, Tuple, Any
import torch
import torch.nn as nn

from .utils.visualization import plot_predicted_profiles
from .utils.evaluation import evaluate_predictions
from .models.hybrid_model import HybridModel
from .models.neural_network import NeuralNetwork
from .data.dataset import BioreactorDataset

def simulate(
    model: HybridModel,
    init_conditions: torch.Tensor,
    time_points: torch.Tensor,
    F: torch.Tensor,
    V: torch.Tensor,
    Z: torch.Tensor,
) -> torch.Tensor:
    """Simulate system dynamics using a hybrid neural network model.

    Args:
        model: Hybrid neural network model
        init_conditions: Initial state values [batch_size, num_states]
        time_points: Array of time points [num_time_points]
        F: Feeding rate data [batch_size, time_steps, num_feeds]
        V: Volume data [batch_size, time_steps, 1]
        Z: DOE data [batch_size, time_steps, num_doe_params]

    Returns:
        torch.Tensor: Simulated state trajectories
    """
    device = next(model.parameters()).device
    init_conditions = init_conditions.to(device)
    time_points = time_points.to(device)
    F = F.to(device)
    V = V.to(device)
    Z = Z.to(device)

    model.eval()

    batch_size = init_conditions.shape[0]
    num_time_points = len(time_points)
    num_states = init_conditions.shape[1]

    results = torch.zeros(
        batch_size, num_time_points, num_states, dtype=torch.float32, device=device
    )
    results[:, 0, :] = init_conditions

    dt = time_points[1] - time_points[0]

    with torch.no_grad():
        for i in range(1, num_time_points):
            t = torch.full(
                (batch_size,), time_points[i - 1], dtype=torch.float32, device=device
            )
            current_states = results[:, i - 1, :]
            current_feeds = F[:, i - 1, :]
            current_volumes = V[:, i - 1, :]
            current_z = Z[:, i - 1, :]

            derivatives = model(t, current_states, current_feeds, current_volumes, current_z)
            next_states = current_states + dt * derivatives
            results[:, i, :] = next_states

    return results.detach().cpu()


def load_model(dataset: BioreactorDataset, config: Dict[str, Any]) -> nn.Module:
    """Load a trained hybrid model with specified configuration.

    Args:
        dataset: Dataset object containing feature and state dimensions
        config: Dictionary containing model and training parameters

    Returns:
        nn.Module: Loaded hybrid model with pre-trained weights
    """
    from .trainer import load_checkpoint
    
    neural_model = NeuralNetwork(
        input_dim=dataset.features_dim,
        output_dim=dataset.states_dim,
        hidden_dim=config["model_params"]["hidden_dim"],
        num_layers=config["model_params"]["num_layers"],
    )
    model = HybridModel(neural_model, dataset.sign_mask)

    model = load_checkpoint(
        model,
        save_dir=config["training_params"]["save_dir"],
        model_name=config["training_params"]["model_name"],
        device=config["device"],
    )
    return model


def inference(
    dataset: BioreactorDataset, model: HybridModel, config: Dict[str, Any]
) -> Tuple[torch.Tensor, List[Dict[str, Any]]]:
    """Run inference using the trained model.

    Args:
        dataset: Dataset object containing simulation data
        model: Trained model object
        config: Configuration dictionary

    Returns:
        Tuple containing:
        - Predicted values tensor
        - List of performance metrics dictionaries
    """
    sim_data = dataset.get_simulation_data()
    X_pred = simulate(model=model, **sim_data)

    if config["mode"] == "predict":
        plot_predicted_profiles(
            X_pred=X_pred,
            X_columns=dataset.X_columns,
            title="Predictions",
            save_path=config["training_params"]["log_dir"],
            show=False,
        )
        dataset.save_predictions(X_pred, save_dir=config["results_dir"])
        return X_pred, {}

    info = evaluate_predictions(dataset.X, X_pred, dataset.X_columns)

    for res in info:
        var = res["variable"]
        print(
            f"\n{var}:\n"
            f"R²: {res['r2']:.3f}, "
            f"RMSE: {res['rmse']:.3f}, "
            f"Rel RMSE: {res['rel_rmse']:.3f}"
        )

    plot_predicted_profiles(
        X_true=dataset.X,
        X_pred=X_pred,
        X_columns=dataset.X_columns,
        title="Test Set Predictions",
        save_path=config["training_params"]["log_dir"],
        show=False,
    )

    return X_pred, info