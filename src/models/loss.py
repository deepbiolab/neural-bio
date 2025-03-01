"""Loss functions for hybrid model training.

Author: Tim Lin
Organization: DeepBioLab
License: MIT License
"""

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn


class AdaptiveWeightedHybridLoss(nn.Module):
    """Adaptive Weighted Hybrid Loss combining data-driven and physics-informed losses.

    This loss function combines two components:
    1. Data loss: MSE between predicted and true values
    2. Physics loss: MSE between predicted and true derivatives

    Both components are normalized by their respective standard deviations.
    The weights for each state variable can be either:
    - Fixed weights provided during initialization
    - Learnable parameters that are optimized during training

    Attributes:
        lambda_data (float): Global scaling factor for data loss component
        lambda_physics (float): Global scaling factor for physics loss component
        states_weight (nn.Parameter or torch.Tensor): Weights for each state variable
        adaptive (bool): Whether to use learnable weights
    """

    def __init__(
        self,
        lambda_data: float = 1.0,
        lambda_physics: float = 0.1,
        states_weight: Optional[Union[List[float], torch.Tensor]] = None,
        num_states: Optional[int] = None,
        adaptive: bool = True,
    ) -> None:
        """Initialize the adaptive weighted hybrid loss.

        Args:
            lambda_data: Scaling factor for data loss
            lambda_physics: Scaling factor for physics loss
            states_weight: Fixed weights for each state variable (used if adaptive=False)
            num_states: Number of state variables (required if adaptive=True)
            adaptive: Whether to use learnable weights (default: True)
        """
        super(AdaptiveWeightedHybridLoss, self).__init__()
        self.lambda_data = lambda_data
        self.lambda_physics = lambda_physics
        self.adaptive = adaptive

        if adaptive:
            if num_states is None:
                raise ValueError("num_states must be provided when using adaptive weights")
            # Initialize learnable weights
            self.states_weight = nn.Parameter(torch.zeros(num_states))
        else:
            # Use fixed weights
            if states_weight is not None:
                self.register_buffer(
                    "states_weight", 
                    torch.tensor(states_weight, dtype=torch.float32)
                )
            else:
                self.states_weight = None

    def get_weights(self) -> torch.Tensor:
        """Get weights for loss computation.

        Returns:
            torch.Tensor: Positive weights for each state variable
        """
        if self.adaptive:
            # Transform learnable weights through sigmoid for positive values
            scale = 3.0
            return scale * torch.sigmoid(self.states_weight)
        else:
            return self.states_weight

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        derivatives_pred: torch.Tensor,
        derivatives_true: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the weighted hybrid loss.

        Args:
            y_pred: Predicted values tensor [batch_size, time_steps, num_vars]
            y_true: Ground truth values tensor [batch_size, time_steps, num_vars]
            derivatives_pred: Predicted derivatives tensor [batch_size, time_steps, num_vars]
            derivatives_true: Ground truth derivatives tensor [batch_size, time_steps, num_vars]

        Returns:
            Tuple containing:
            - total_loss: Combined weighted loss
            - data_loss: Data component of the loss
            - physics_loss: Physics component of the loss
        """
        # Compute standard deviations for normalization
        y_std = torch.std(y_true, dim=(0, 1), keepdim=True) + 1e-6
        derivatives_std = torch.std(derivatives_true, dim=(0, 1), keepdim=True) + 1e-6

        # Compute normalized squared errors
        data_losses = ((y_pred - y_true) / y_std) ** 2
        physics_losses = ((derivatives_pred - derivatives_true) / derivatives_std) ** 2

        # Get weights (either learned or fixed)
        weights = self.get_weights()
        if weights is None:
            weights = torch.ones(y_pred.shape[-1], device=y_pred.device)
        else:
            weights = weights.to(y_pred.device)

        # Compute weighted losses
        if self.adaptive:
            weights = weights.view(1, 1, -1)  # Reshape for broadcasting
        
        data_loss = torch.mean(data_losses * weights)
        physics_loss = torch.mean(physics_losses * weights)

        # Combine losses using global scaling factors
        total_loss = self.lambda_data * data_loss + self.lambda_physics * physics_loss

        return total_loss, data_loss, physics_loss