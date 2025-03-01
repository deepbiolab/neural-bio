"""Bioreactor dataset implementation.

Author: Tim Lin
Organization: DeepBioLab
License: MIT License
"""

import os
from typing import Tuple, List, Optional, Union
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .data_utils import process_owu_data, process_doe_data, create_empty_owu

class BioreactorDataset(Dataset):
    """Dataset class for bioreactor data handling and processing."""
    
    def __init__(
        self,
        owu_file: str,
        doe_file: str,
        train_path: str = "dataset/interpolation/train",
        test_path: str = "dataset/interpolation/test",
        predict_path: str = "dataset/interpolation/predict",
        t_steps: int = 15,
        time_step: int = 24,
        init_volume: float = 1000,
        Z_columns: List[str] = [],
        X_columns: List[str] = [],
        F_columns: List[str] = [],
        mode: str = "train",
        val_split: float = 0.2,
        random_seed: int = 42,
    ) -> None:
        """Initialize the dataset.

        Args:
            owu_file: Name of the OWU data file
            doe_file: Name of the DOE data file
            train_path: Root path for training dataset
            test_path: Root path for test dataset
            predict_path: Root path for prediction dataset
            t_steps: Number of time steps
            time_step: Duration of each time step in hours
            init_volume: Initial volume in mL
            Z_columns: List of DOE parameter columns
            X_columns: List of state variable columns
            F_columns: List of feeding rate columns
            mode: Dataset mode ('train', 'val', 'test', or 'predict')
            val_split: Validation set ratio (0-1)
            random_seed: Random seed for reproducibility
        """
        self.t_steps = t_steps
        self.time_step = time_step
        self.init_volume = init_volume
        self.mode = mode
        self.train_path = train_path
        self.test_path = test_path
        self.predict_path = predict_path
        self.val_split = val_split
        self.random_seed = random_seed

        self.time_mask = np.ones(t_steps)
        self.sign_mask = np.array([-1 if col in F_columns else 1 for col in X_columns])
        self.feed_mask = np.where(self.sign_mask < 0, 1, 0)

        self.Z_columns = Z_columns
        self.X_columns = [f"X:{col}" for col in X_columns]
        self.F_columns = [f"F:{col}" for col in F_columns]
        
        if mode == "train" or mode == "val":
            self.root_path = train_path
        elif mode == "test":
            self.root_path = test_path
        elif mode == "predict":
            self.root_path = predict_path
        else:
            raise ValueError("Invalid mode. Choose from 'train', 'test', 'predict'.")

        doe_data = self._read_doe(doe_file)

        if mode == "predict":
            owu_data = create_empty_owu(
                owu_file, doe_data, t_steps, self.F_columns, self.X_columns, self.root_path
            )
            self.result_df = owu_data.reset_index().copy()
        else:
            owu_data = self._read_owu(owu_file)

        self._process_data(owu_data, doe_data)

        if mode in ["train", "val"]:
            self._split_data()

        self._init_conditions = None

    @classmethod
    def train_val_split(cls, **kwargs) -> Tuple['BioreactorDataset', 'BioreactorDataset']:
        """Create training and validation dataset instances."""
        train_dataset = cls(mode="train", **kwargs)
        val_dataset = cls(mode="val", **kwargs)
        return train_dataset, val_dataset
    
    def _split_data(self) -> None:
        """Split the data into training and validation sets."""
        np.random.seed(self.random_seed)
        total_size = len(self.X)
        val_size = int(self.val_split * total_size)
        indices = list(range(total_size))
        np.random.shuffle(indices)
        train_indices = indices[val_size:]
        val_indices = indices[:val_size]
        self.indices = train_indices if self.mode == "train" else val_indices
        self.X = self.X[self.indices]
        self.F = self.F[self.indices]
        self.Y = self.Y[self.indices]
        self.V = self.V[self.indices]
        self.Z = self.Z[self.indices]

    def _read_owu(self, file: str) -> pd.DataFrame:
        """Read and process the OWU data file."""
        data = pd.read_csv(f"{self.root_path}/{file}.csv")
        owu_df = data.copy()
        num_runs = len(pd.read_csv(f"{self.root_path}/{file}_doe.csv"))

        if "run" not in owu_df.columns:
            owu_df.index = pd.MultiIndex.from_product(
                [list(range(num_runs)), list(range(self.t_steps))],
                names=["run", "time"],
            )
        else:
            owu_df.set_index(["run", "time"], inplace=True)
        owu_df = owu_df[self.X_columns + self.F_columns]
        return owu_df

    def _read_doe(self, file: str) -> pd.DataFrame:
        """Read the Design of Experiments data file."""
        data = pd.read_csv(
            f"{self.root_path}/{file}.csv",
            usecols=self.Z_columns,
        )
        return data.copy()

    def _process_data(self, owu_data: pd.DataFrame, doe_data: pd.DataFrame) -> None:
        """Process raw data into tensor format."""
        self.X, self.F = process_owu_data(owu_data, self.t_steps, 
                                        self.X_columns, self.F_columns)
        self.Z = process_doe_data(doe_data, self.Z_columns)

        self.V = (
            self.init_volume + (self.F.sum(axis=-1, keepdims=True)).cumsum(axis=1)
        ) / 1000

        self.F = (self.feed_mask[None, None, :] * self.F) / self.time_step
        self.Z = self.time_mask[None, :, None] * self.Z

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.F = torch.tensor(self.F, dtype=torch.float32)
        self.V = torch.tensor(self.V, dtype=torch.float32)
        self.Z = torch.tensor(self.Z, dtype=torch.float32)

        if self.mode != "predict":
            self.Y = self._central_differences(
                self.X.numpy(), self.F.numpy(), self.V.numpy()
            )
            self.Y = torch.tensor(self.Y, dtype=torch.float32)
        else:
            self.Y = torch.zeros_like(self.X)

    def _central_differences(
        self, X: np.ndarray, F: np.ndarray, V: np.ndarray
    ) -> np.ndarray:
        """Calculate central differences for derivatives computation."""
        Y = np.zeros_like(X)

        Y[:, 0, :] = (
            self.sign_mask[None, :]
            * (X[:, 1, :] * V[:, 1, :] - X[:, 0, :] * V[:, 0, :])
        ) / (self.time_step * V[:, 0, :]) + F[:, 0, :]

        Y[:, 1:-1, :] = (
            self.sign_mask[None, None, :]
            * ((X[:, 2:, :] * V[:, 2:, :] - X[:, :-2, :] * V[:, :-2, :]) / 2)
        ) / (self.time_step * V[:, 1:-1, :]) + F[:, 1:-1, :]

        Y[:, -1, :] = (
            self.sign_mask[None, :]
            * (X[:, -1, :] * V[:, -1, :] - X[:, -2, :] * V[:, -2, :])
        ) / (self.time_step * V[:, -1, :]) + F[:, -2, :]

        return Y

    @property
    def features_dim(self) -> int:
        """Total dimension of feature space."""
        return self.X.shape[-1] + self.F.shape[-1] + self.Z.shape[-1]

    @property
    def states_dim(self) -> int:
        """Dimension of state space."""
        return self.X.shape[-1]

    @property
    def init_conditions(self) -> torch.Tensor:
        """Get initial conditions tensor."""
        if self._init_conditions is None:
            self._init_conditions = torch.zeros((len(self), self.states_dim))

            for i, col in enumerate(self.Z_columns):
                if col.endswith("_0") and f"X:{col[:-2]}" in self.X_columns:
                    state_idx = self.X_columns.index(f"X:{col[:-2]}")
                    self._init_conditions[:, state_idx] = self.Z[:, 0, i]

        return self._init_conditions

    def get_simulation_data(self) -> dict:
        """Get all data needed for simulation."""
        return {
            "init_conditions": self.init_conditions,
            "F": self.F,
            "V": self.V,
            "Z": self.Z,
            "time_points": torch.arange(0, self.t_steps * self.time_step, self.time_step),
        }

    def save_predictions(self, X_pred: torch.Tensor, save_dir: str = "results") -> None:
        """Save prediction results."""
        os.makedirs(save_dir, exist_ok=True)
        self.result_df[self.X_columns] = X_pred.reshape(-1, X_pred.shape[-1]).numpy()
        file_path = os.path.join(save_dir, "owu_pred.csv")
        self.result_df.to_csv(file_path, index=False)
        print(f"Predictions saved to {file_path}")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        """Get a single sample from the dataset."""
        if self.mode == "predict":
            return (self.F[idx], self.V[idx], self.Z[idx])
        else:
            return (self.X[idx], self.F[idx], self.Y[idx], self.V[idx], self.Z[idx])