"""Neural network model implementation."""

from typing import List
import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    """Neural network model for predicting reaction rates.
    
    This model consists of multiple parallel neural networks, each predicting
    the reaction rate for one state variable.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 4,
        num_layers: int = 3
    ) -> None:
        """Initialize the neural network.

        Args:
            input_dim: Dimension of input features
            hidden_dim: Number of neurons in hidden layers
            output_dim: Number of state variables to predict
            num_layers: Number of hidden layers in each sub-network
        """
        super(NeuralNetwork, self).__init__()

        self.networks = nn.ModuleList()
        for _ in range(output_dim):
            layers: List[nn.Module] = []
            
            # Input layer
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())

            # Hidden layers
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())

            # Output layer
            layers.append(nn.Linear(hidden_dim, 1))

            self.networks.append(nn.Sequential(*layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the neural network.

        Args:
            x: Input tensor [batch_size, input_dim]

        Returns:
            torch.Tensor: Predicted reaction rates [batch_size, output_dim]
        """
        derivatives = []
        for network in self.networks:
            derivatives.append(network(x))

        return torch.cat(derivatives, dim=1)