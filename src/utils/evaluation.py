"""Evaluation metrics and utilities."""

from typing import List, Dict, Union, Any

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_predictions(
    X_true: Union[np.ndarray, torch.Tensor],
    X_pred: Union[np.ndarray, torch.Tensor],
    var_names: List[str],
) -> List[Dict[str, Union[str, float]]]:
    """Evaluate model predictions against ground truth values.

    Args:
        X_true: Ground truth values array [batch_size, time_steps, num_vars]
        X_pred: Predicted values array [batch_size, time_steps, num_vars]
        var_names: List of variable names

    Returns:
        List of dictionaries containing performance metrics for each variable
    """
    if torch.is_tensor(X_true):
        X_true = X_true.numpy()
    if torch.is_tensor(X_pred):
        X_pred = X_pred.numpy()

    results = []

    for i, var in enumerate(var_names):
        y_true = X_true[:, :, i].flatten()
        y_pred = X_pred[:, :, i].flatten()

        mean_true = np.mean(y_true)
        std_true = np.std(y_true)

        y_true_norm = (y_true - mean_true) / (std_true + 1e-6)
        y_pred_norm = (y_pred - mean_true) / (std_true + 1e-6)

        r2 = r2_score(y_true_norm, y_pred_norm)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        rel_rmse = rmse / (std_true + 1e-6)

        results.append({
            "variable": var, 
            "r2": r2, 
            "rmse": rmse, 
            "rel_rmse": rel_rmse
        })

    return results