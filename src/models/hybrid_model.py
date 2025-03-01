"""Hybrid model combining neural network with physics-based constraints.

Author: Tim Lin
Organization: DeepBioLab
License: MIT License
"""

from typing import Union
import numpy as np
import torch
import torch.nn as nn

from .neural_network import NeuralNetwork

class HybridModel(nn.Module):
    """A hybrid model combining neural network with physics-based constraints.

    This model implements a hybrid approach where a neural network predicts 
    reaction rates, which are then combined with physical constraints to predict
    the system's temporal evolution.
    """

    def __init__(self, neural_model: NeuralNetwork, sign_mask: np.ndarray) -> None:
        """Initialize the hybrid model.

        Args:
            neural_model: Neural network model that predicts reaction rates
            sign_mask: Array of -1/1 values indicating reaction directions
        """
        super(HybridModel, self).__init__()
        self.neural_model = neural_model
        self.register_buffer("sign_mask", torch.tensor(sign_mask, dtype=torch.float32))

    def to(self, device: Union[str, torch.device]) -> "HybridModel":
        """Move the model to specified device.

        Args:
            device: Target device ('cpu', 'cuda', 'mps' or torch.device)

        Returns:
            self: The model instance on the target device
        """
        super().to(device)
        self.sign_mask = self.sign_mask.to(device)
        return self

    def forward(
        self,
        t: torch.Tensor,
        states: torch.Tensor,
        feeds: torch.Tensor,
        volumes: torch.Tensor,
        z_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the hybrid model.

        Args:
            t: Current time points [batch_size]
            states: Current state variables [batch_size, time_steps, states_dim]
            feeds: Current feeding rates [batch_size, time_steps, states_dim]
            volumes: Current volumes [batch_size, time_steps, 1]
            z_inputs: Experimental design parameters [batch_size, time_steps, z_dim]

        Returns:
            torch.Tensor: Computed derivatives (dX/dt)
        """
        is_time_series = len(states.shape) == 3

        if is_time_series:
            batch_size, time_steps, states_dim = states.shape
            dX_dt = torch.zeros_like(states)

            for step in range(time_steps):
                current_states = states[:, step, :]
                current_feeds = feeds[:, step, :]
                current_z = z_inputs[:, step, :]
                curr_volume = volumes[:, max(0, step - 1), :] if step > 0 else volumes[:, 0, :]
                after_feed_volume = volumes[:, step, :]

                dX_dt[:, step, :] = self._compute_derivatives(
                    current_states,
                    current_feeds,
                    curr_volume,
                    after_feed_volume,
                    current_z,
                )

            return dX_dt
        else:
            return self._compute_derivatives(states, feeds, volumes, volumes, z_inputs)

    def _compute_derivatives(
        self,
        states: torch.Tensor,
        feeds: torch.Tensor,
        curr_volume: torch.Tensor,
        after_feed_volume: torch.Tensor,
        z_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """Helper function to compute derivatives for a single time step.

        Args:
            states: Current state variables [batch_size, states_dim]
            feeds: Current feeding rates [batch_size, states_dim]
            curr_volume: Current volume before feeding [batch_size, 1]
            after_feed_volume: Volume after feeding [batch_size, 1]
            z_inputs: Experimental design parameters [batch_size, z_dim]

        Returns:
            torch.Tensor: Computed derivatives (dX/dt) [batch_size, states_dim]
        """
        states = torch.clamp(states, min=0.0)
        inputs = torch.cat([states, feeds, z_inputs], dim=1)
        predicted_rates = self.neural_model(inputs)

        curr_volume = curr_volume.view(-1, 1)
        after_feed_volume = after_feed_volume.view(-1, 1)

        dX_dt = torch.zeros_like(states)

        for i in range(predicted_rates.shape[-1]):
            reaction = self.sign_mask[i] * predicted_rates[:, i] * curr_volume.squeeze()
            feed = feeds[:, i] * after_feed_volume.squeeze()
            dX_dt[:, i] = (reaction + feed) / after_feed_volume.squeeze()

        return dX_dt