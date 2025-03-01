"""Main entry point for the bioreactor hybrid model training and inference.

Author: Tim Lin
Organization: DeepBioLab
License: MIT License
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src import get_default_config
from src import BioreactorDataset
from src import NeuralNetwork, HybridModel, AdaptiveWeightedHybridLoss
from src import train_one_model, inference, load_model
from src import plot_training_curves, plot_predicted_profiles

def train(
    train_dataset: BioreactorDataset, val_dataset: BioreactorDataset, config: dict
) -> nn.Module:
    """Train a hybrid neural network model.

    Args:
        train_dataset: Dataset for training
        val_dataset: Dataset for validation
        config: Configuration dictionary

    Returns:
        nn.Module: Trained hybrid model
    """
    # Set random seeds for reproducibility
    torch.manual_seed(config["random_seed"])
    seed_generator = torch.Generator().manual_seed(config["random_seed"])

    # Initialize data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training_params"]["batch_size"],
        generator=seed_generator,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["training_params"]["batch_size"], shuffle=False
    )

    # Initialize models
    neural_model = NeuralNetwork(
        input_dim=train_dataset.features_dim,
        output_dim=train_dataset.states_dim,
        **config["model_params"],
    )
    model = HybridModel(neural_model, train_dataset.sign_mask)

    # Setup loss function
    criterion = AdaptiveWeightedHybridLoss(
        num_states=train_dataset.states_dim, **config["loss_params"]
    )

    # Initialize optimizer
    optimizer = optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=config["training_params"]["learning_rate"],
        weight_decay=config["training_params"]["weight_decay"],
    )

    # Setup learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=config["training_params"]["patience"],
    )

    # Train the model
    train_losses, val_losses  = train_one_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=config["training_params"]["num_epochs"],
        save_dir=config["training_params"]["save_dir"],
        model_name=config["training_params"]["model_name"],
        device=config["device"],
    )

    # Visualize training progress
    plot_training_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        title="Training and Validation Loss Curves",
        save_path=config["training_params"]["log_dir"],
        show=False,
    )

    # Load best model for final evaluation
    model = load_model(train_dataset, config)

    # Generate and visualize predictions
    train_sim_data = train_dataset.get_simulation_data()
    val_sim_data = val_dataset.get_simulation_data()

    print("\n=== Train Set Results ===")
    X_train_pred, _ = inference(train_dataset, model, config)
    print("\n=== Validation Set Results ===")
    X_val_pred, _ = inference(val_dataset, model, config)

    # Plot predictions
    plot_predicted_profiles(
        X_pred=X_train_pred,
        X_true=train_dataset.X,
        X_columns=train_dataset.X_columns,
        title="Train Set Predictions",
        save_path=config["training_params"]["log_dir"],
        show=False,
    )
    plot_predicted_profiles(
        X_pred=X_val_pred,
        X_true=val_dataset.X,
        X_columns=val_dataset.X_columns,
        title="Validation Set Predictions",
        save_path=config["training_params"]["log_dir"],
        show=False,
    )

    return model


def main(args: argparse.Namespace) -> None:
    """Main function to run the model.

    Args:
        args: Command line arguments
    """
    # Get configuration
    config = get_default_config()
    config["mode"] = args.mode

    # Execute pipeline based on operation mode
    if config["mode"] == "train":
        # Create training and validation datasets
        train_dataset, val_dataset = BioreactorDataset.train_val_split(
            **config["dataset_params"]
        )
        print(
            f"Dataset sizes - Train: {len(train_dataset)}, "
            f"Validation: {len(val_dataset)}"
        )

        # Train the model
        model = train(train_dataset, val_dataset, config)

    elif config["mode"] in ["test", "predict"]:
        # Create dataset for testing or prediction
        dataset = BioreactorDataset(mode=config["mode"], **config["dataset_params"])
        print(f"Dataset size: {len(dataset)}")

        # Load pre-trained model
        model = load_model(dataset, config)

        # Run inference
        X_pred, info = inference(dataset, model, config)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train or run inference on hybrid model"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test", "predict"],
        help="Operation mode: train, test, predict",
    )

    args = parser.parse_args()
    main(args)
