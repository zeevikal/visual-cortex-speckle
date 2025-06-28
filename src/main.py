"""
Main entry point for the Visual Cortex Speckle Recognition project.
"""

import os
import sys
import argparse
import torch
import numpy as np
import random
from typing import Optional

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset import create_data_loaders_from_pickle
from models.base_model import ModelFactory
from training.trainer import Trainer
from training.evaluation import ModelEvaluator
from utils.config import load_config, get_default_config, create_config_templates, print_config
from utils.visualization import create_visualization_report


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device_str: str) -> torch.device:
    """Get the appropriate device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_str)


def train_model(config):
    """Train a model with the given configuration."""
    print("Starting training...")
    print_config(config)
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Get device
    device = get_device(config.device)
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader, data_loader = create_data_loaders_from_pickle(
        pickle_path=config.data.pickle_path,
        subjects=config.data.subjects,
        shape_filter=config.data.shape_filter,
        feature_type=config.data.feature_type,
        train_size=config.data.train_size,
        n_chunks=config.data.n_chunks,
        batch_size=config.training.batch_size,
        validation_split=config.training.validation_split
    )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Number of classes: {data_loader.get_num_classes()}")
    print(f"Class names: {data_loader.get_class_names()}")
    
    # Update number of classes in config
    config.model.num_classes = data_loader.get_num_classes()
    
    # Create model
    print("Creating model...")
    model = ModelFactory.create_model(
        config.model.model_type,
        num_classes=config.model.num_classes,
        input_channels=config.model.input_channels,
        num_filters=config.model.num_filters,
        kernel_size=config.model.kernel_size,
        dropout_rate=config.model.dropout_rate
    )
    
    print(f"Model: {model.__class__.__name__}")
    if hasattr(model, 'get_model_info'):
        model_info = model.get_model_info()
        print(f"Parameters: {model_info['trainable_parameters']:,}")
        print(f"Model size: {model_info['model_size_mb']:.2f} MB")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        device=device,
        save_dir=config.save_dir,
        log_dir=config.log_dir
    )
    
    # Set up optimizer
    if config.training.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    elif config.training.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=config.training.learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")
    
    # Train model
    print("Starting training...")
    history = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.training.epochs,
        optimizer=optimizer,
        scheduler=config.training.scheduler,
        scheduler_kwargs=config.scheduler.__dict__,
        early_stopping={
            'patience': config.training.early_stopping_patience,
            'mode': 'min'
        },
        save_every=config.training.save_every,
        save_best=config.training.save_best
    )
    
    # Evaluate model
    print("Evaluating model...")
    
    # Load best model if available
    best_model_path = os.path.join(config.save_dir, 'best_model.pth')
    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded best model for evaluation")
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        class_names=data_loader.get_class_names()
    )
    
    # Save evaluation results
    evaluator.save_evaluation_report(
        test_loader,
        save_dir=config.results_dir,
        dataset_name='test'
    )
    
    # Create visualizations
    print("Creating visualizations...")
    
    # Get sample data for visualization
    x_train_sample = []
    y_train_sample = []
    for data, target in train_loader:
        x_train_sample.append(data.numpy())
        y_train_sample.append(target.numpy())
        if len(x_train_sample) >= 5:  # Limit samples for visualization
            break
    
    x_test_sample = []
    y_test_sample = []
    for data, target in test_loader:
        x_test_sample.append(data.numpy())
        y_test_sample.append(target.numpy())
        if len(x_test_sample) >= 5:
            break
    
    if x_train_sample and x_test_sample:
        x_train_viz = np.concatenate(x_train_sample, axis=0)
        y_train_viz = np.concatenate(y_train_sample, axis=0)
        x_test_viz = np.concatenate(x_test_sample, axis=0)
        y_test_viz = np.concatenate(y_test_sample, axis=0)
        
        create_visualization_report(
            model=model,
            x_train=x_train_viz,
            y_train=y_train_viz,
            x_test=x_test_viz,
            y_test=y_test_viz,
            history=history,
            class_names=data_loader.get_class_names(),
            save_dir=os.path.join(config.results_dir, 'visualizations')
        )
    
    print("Training completed successfully!")
    return model, history, evaluator


def evaluate_model(config, model_path: str):
    """Evaluate a trained model."""
    print("Starting evaluation...")
    
    # Get device
    device = get_device(config.device)
    
    # Create data loaders
    train_loader, val_loader, test_loader, data_loader = create_data_loaders_from_pickle(
        pickle_path=config.data.pickle_path,
        subjects=config.data.subjects,
        shape_filter=config.data.shape_filter,
        feature_type=config.data.feature_type,
        train_size=config.data.train_size,
        n_chunks=config.data.n_chunks,
        batch_size=config.training.batch_size,
        validation_split=config.training.validation_split
    )
    
    # Load model
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    model = ModelFactory.create_model(
        config.model.model_type,
        num_classes=data_loader.get_num_classes(),
        **config.model.__dict__
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create evaluator
    evaluator = ModelEvaluator(
        model=model,
        device=device,
        class_names=data_loader.get_class_names()
    )
    
    # Evaluate
    evaluator.save_evaluation_report(
        test_loader,
        save_dir=config.results_dir,
        dataset_name='evaluation'
    )
    
    print("Evaluation completed!")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Visual Cortex Speckle Recognition")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "create_configs"], 
                       default="train", help="Mode to run")
    parser.add_argument("--model_path", type=str, help="Path to trained model (for evaluation)")
    
    # Override arguments
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--subjects", type=str, nargs="+", help="Subjects to include")
    parser.add_argument("--shapes", type=str, nargs="+", help="Shapes to include")
    parser.add_argument("--feature_type", type=str, help="Feature type to use")
    
    args = parser.parse_args()
    
    if args.mode == "create_configs":
        create_config_templates()
        return
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        config = get_default_config()
        print("Using default configuration. Use --config to specify a custom config file.")
    
    # Override config with command line arguments
    if args.epochs:
        config.training.epochs = args.epochs
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    if args.subjects:
        config.data.subjects = args.subjects
    if args.shapes:
        config.data.shape_filter = args.shapes
    if args.feature_type:
        config.data.feature_type = args.feature_type
    
    # Create directories
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    if config.log_dir:
        os.makedirs(config.log_dir, exist_ok=True)
    
    # Run mode
    if args.mode == "train":
        train_model(config)
    elif args.mode == "eval":
        if not args.model_path:
            parser.error("--model_path is required for evaluation mode")
        evaluate_model(config, args.model_path)


if __name__ == "__main__":
    main()
