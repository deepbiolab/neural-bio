"""Data processing utilities.

Author: Tim Lin
Organization: DeepBioLab
License: MIT License
"""

from typing import List, Tuple
import numpy as np
import pandas as pd

def process_owu_data(
    owu_raw: pd.DataFrame,
    t_steps: int,
    X_columns: List[str],
    F_columns: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert OWU DataFrame to 3D numpy arrays.

    Args:
        owu_raw: Raw OWU DataFrame
        t_steps: Number of time steps
        X_columns: List of state variable column names
        F_columns: List of feeding variable column names

    Returns:
        Tuple containing:
        - State variables array [batch_size, time_steps, num_vars]
        - Feeding rates array [batch_size, time_steps, num_feeds]
    """
    owu = owu_raw.copy()
    owu = owu.sort_index(level=["run", "time"])
    
    B = owu.index.get_level_values("run").nunique()
    T = t_steps
    C_X = len(X_columns)
    C_F = len(F_columns)

    X = np.zeros((B, T, C_X))
    F = np.zeros((B, T, C_F))

    for i, (run, group) in enumerate(owu.groupby(level="run")):
        X_group = group[X_columns].copy()
        F_group = group[F_columns].copy()

        if len(group) != T:
            raise ValueError(f"Run {run} does not have {T} time steps.")

        X[i, :, :] = X_group.values
        F[i, :, :] = F_group.values

    return X, F


def process_doe_data(
    doe_raw: pd.DataFrame,
    Z_columns: List[str]
) -> np.ndarray:
    """Convert DOE DataFrame to 3D numpy array.

    Args:
        doe_raw: Raw DOE DataFrame
        Z_columns: List of DOE parameter column names

    Returns:
        np.ndarray: 3D array of experimental parameters [batch_size, 1, num_params]
    """
    doe = doe_raw.copy()
    doe = doe.sort_index()

    B = doe.shape[0]
    C_Z = len(Z_columns)
    T = 1

    Z = np.zeros((B, T, C_Z))
    Z[:, 0, :] = doe.values

    return Z


def create_empty_owu(
    file: str,
    doe_data: pd.DataFrame,
    t_steps: int,
    F_columns: List[str],
    X_columns: List[str],
    root_path: str
) -> pd.DataFrame:
    """Create an OWU data framework pre-filled with feeding information.

    Args:
        file: OWU file name
        doe_data: DOE DataFrame containing feed parameters
        t_steps: Number of time steps
        F_columns: List of feeding variable column names
        X_columns: List of state variable column names
        root_path: Root path for data files

    Returns:
        pd.DataFrame: Pre-filled OWU data framework
    """
    header_df = pd.read_csv(f"{root_path}/{file}.csv", nrows=0)

    index = pd.MultiIndex.from_product(
        [list(range(doe_data.shape[0])), list(range(t_steps))],
        names=["run", "time"],
    )

    empty_df = pd.DataFrame(0.0, index=index, columns=header_df.columns)

    for run in range(doe_data.shape[0]):
        feed_start = int(doe_data.iloc[run]["feed_start"])
        feed_end = int(doe_data.iloc[run]["feed_end"])
        feed_rate = doe_data.iloc[run]["Glc_feed_rate"]

        for time in range(feed_start, min(feed_end + 1, t_steps)):
            for col in F_columns:
                empty_df.loc[(run, time), col] = feed_rate

    empty_df = empty_df[X_columns + F_columns]
    return empty_df