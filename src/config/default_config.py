"""Default configuration for the bioreactor model."""

from typing import Dict, Any

def get_default_config() -> Dict[str, Any]:
    """Get default configuration parameters.
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    config = {
        "mode": "predict",  # Operation mode: 'train', 'test', or 'predict'
        "device": "cpu",    # Computing device: 'cuda', 'cpu', 'mps'
        "results_dir": "results",  # Directory for saving results
        "random_seed": 42,  # Random seed for reproducibility
        
        # Dataset configuration parameters
        "dataset_params": {
            "owu_file": "owu",  # Observation Wise Unit data file
            "doe_file": "owu_doe",  # Design of Experiments file
            "train_path": "dataset/interpolation/train",  # Training data path
            "test_path": "dataset/interpolation/test",    # Test data path
            "predict_path": "dataset/interpolation/predict",  # Prediction data path
            "Z_columns": ["feed_start", "feed_end", "Glc_feed_rate", "Glc_0", "VCD_0"], # Design parameters including feeding schedule and initial conditions
            "F_columns": ["Glc"],  # Feeding variables (Glucose)
            "X_columns": ["VCD", "Glc", "Lac", "Titer"], # States: Viable Cell Density, Glucose, Lactate, Product Titer
            "t_steps": 15,  # Number of time steps
            "time_step": 24,  # Hours per time step
            "init_volume": 1000,  # Initial volume in mL
            "val_split": 0.2,  # Validation set ratio
        },
        
        # Neural network model parameters
        "model_params": {
            "hidden_dim": 128,  # Hidden layer dimension
            "num_layers": 3,    # Number of hidden layers
        },
        "loss_params" : {            
            "lambda_data": 1.0,  # Weight for data-driven loss
            "lambda_physics": 1.0,  # Weight for physics-based loss},
            "states_weight": [1.0, 2.0, 1.0, 1.0], # Weight for each state variable
            "adaptive": True, # Whether to use adaptive loss weights
        },
        "cv_params": {
            "n_splits": 5,  # Number of folds for cross-validation
        },
            
        # Training process parameters
        "training_params": {
            "batch_size": 32,  # Mini-batch size
            "learning_rate": 0.001,  # Initial learning rate
            "weight_decay": 1e-5,  # L2 regularization coefficient
            "num_epochs": 1000,  # Maximum number of training epochs
            "patience": 20,  # Early stopping patience
            "save_dir": "checkpoints",  # Directory for model checkpoints
            "model_name": "best_model",  # Name for saved model
            "log_dir": "logs",  # Directory for training logs
        },
    }
    
    return config