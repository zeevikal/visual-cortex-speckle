"""
Main entry point for the Visual Cortex Speckle Recognition project.
"""

import os
import sys
import argparse
import torch
import numpy as np
import random
import pickle
from typing import Optional
from dataclasses import asdict
import json

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.dataset import create_data_loaders_from_pickle
from models.base_model import ModelFactory
from training.trainer import Trainer
from training.evaluation import ModelEvaluator
from utils.config import load_config, get_default_config, print_config
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


def train_model_kfold(config):
    """Train a model using K-fold cross-validation."""
    print("Starting K-fold cross-validation training...")
    print_config(config)
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Get device
    device = get_device(config.device)
    print(f"Using device: {device}")
    
    # Create full dataset (without train/val split)
    print("Loading data...")
    from .data.preprocessing import prepare_dataset_splits
    from .data.dataset import SpeckleDataset
    
    # Load speckle data
    with open(config.data.pickle_path, 'rb') as f:
        speckle_data = pickle.load(f)
    
    # Prepare full dataset (we'll split it in K-fold)
    x_full, y_full, _, _ = prepare_dataset_splits(
        speckle_data=speckle_data,
        train_size=1.0,  # Use all data for K-fold
        n_chunks=config.data.n_chunks,
        subjects=config.data.subjects,
        shape_filter=config.data.shape_filter,
        feature_type=config.data.feature_type
    )
    
    # Create dataset
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_full)
    
    full_dataset = SpeckleDataset(x_full, y_encoded)
    class_names = label_encoder.classes_.tolist()
    
    print(f"Full dataset size: {len(full_dataset)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")
    
    # Update number of classes in config
    config.model.num_classes = len(class_names)
    
    # Create K-fold trainer
    from .training.kfold_trainer import KFoldTrainer
    
    kfold_trainer = KFoldTrainer(
        model_config=config.model.__dict__,
        training_config=config.training.__dict__,
        device=device,
        save_dir=os.path.join(config.save_dir, 'kfold'),
        log_dir=os.path.join(config.log_dir, 'kfold') if config.log_dir else None,
        stratified=config.training.kfold_stratified,
        verbose=config.verbose
    )
    
    # Perform K-fold cross-validation
    cv_results = kfold_trainer.train_kfold(
        dataset=full_dataset,
        class_names=class_names,
        k_folds=config.training.k_folds,
        random_state=config.training.kfold_random_state
    )
    
    # Save best model
    best_model_path = os.path.join(config.save_dir, 'best_kfold_model.pth')
    kfold_trainer.save_best_model(best_model_path)
    
    # Create comprehensive results summary
    results_summary = {
        'config': config.__dict__,
        'cv_results': cv_results,
        'best_model_path': best_model_path,
        'class_names': class_names
    }
    
    # Save results summary
    summary_path = os.path.join(config.results_dir, 'kfold_training_summary.json')
    with open(summary_path, 'w') as f:
        # Convert to JSON-serializable format
        json_summary = {}
        for key, value in results_summary.items():
            if key == 'config':
                json_summary[key] = asdict(value)
            elif key == 'cv_results':
                # Skip complex nested structures
                json_summary[key] = {
                    'accuracy_mean': value['accuracy']['mean'],
                    'accuracy_std': value['accuracy']['std'],
                    'precision_mean': value['precision']['mean'],
                    'precision_std': value['precision']['std'],
                    'recall_mean': value['recall']['mean'],
                    'recall_std': value['recall']['std'],
                    'f1_score_mean': value['f1_score']['mean'],
                    'f1_score_std': value['f1_score']['std']
                }
            else:
                json_summary[key] = value
        json.dump(json_summary, f, indent=2)
    
    print(f"\nK-fold training completed!")
    print(f"Results saved to: {config.save_dir}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Summary saved to: {summary_path}")
    
    return kfold_trainer, cv_results


def train_model_loso(config):
    """Train a model using Leave-One-Subject-Out cross-validation."""
    print("Starting Leave-One-Subject-Out cross-validation training...")
    print_config(config)
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Get device
    device = get_device(config.device)
    print(f"Using device: {device}")
    
    # Load speckle data
    print("Loading data...")
    with open(config.data.pickle_path, 'rb') as f:
        speckle_data = pickle.load(f)
    
    # Get all subjects from data if not specified
    all_subjects = list(speckle_data.keys())
    subjects_to_use = config.data.subjects if config.data.subjects else all_subjects
    
    # Filter subjects that exist in data
    subjects_to_use = [s for s in subjects_to_use if s in speckle_data]
    
    if len(subjects_to_use) < 2:
        raise ValueError(f"LOSO requires at least 2 subjects. Found: {subjects_to_use}")
    
    print(f"LOSO will be performed on {len(subjects_to_use)} subjects: {subjects_to_use}")
    
    # Update number of classes based on data
    from .data.preprocessing import extract_features_from_data
    from sklearn.preprocessing import LabelEncoder
    
    all_labels = []
    for subject in subjects_to_use:
        if subject in speckle_data:
            for data in speckle_data[subject]:
                label = data[1]
                if config.data.shape_filter is None or label in config.data.shape_filter:
                    all_labels.append(label)
    
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    class_names = label_encoder.classes_.tolist()
    config.model.num_classes = len(class_names)
    
    print(f"Number of classes: {config.model.num_classes}")
    print(f"Class names: {class_names}")
    
    # Create LOSO trainer
    from .training.loso_trainer import LOSOTrainer
    
    loso_trainer = LOSOTrainer(
        model_config=config.model.__dict__,
        training_config=config.training.__dict__,
        device=device,
        save_dir=os.path.join(config.save_dir, 'loso'),
        log_dir=os.path.join(config.log_dir, 'loso') if config.log_dir else None,
        verbose=config.verbose
    )
    
    # Perform LOSO cross-validation
    loso_results = loso_trainer.train_loso(
        speckle_data=speckle_data,
        subjects=subjects_to_use,
        class_names=class_names,
        shape_filter=config.data.shape_filter,
        feature_type=config.data.feature_type,
        n_chunks=config.data.n_chunks,
        random_state=getattr(config.training, 'loso_random_state', 42)
    )
    
    # Save best model
    best_model_path = os.path.join(config.save_dir, 'best_loso_model.pth')
    loso_trainer.save_best_model(best_model_path)
    
    # Create comprehensive results summary
    results_summary = {
        'config': config.__dict__,
        'loso_results': loso_results,
        'best_model_path': best_model_path,
        'class_names': class_names,
        'subjects': subjects_to_use
    }
    
    # Save results summary
    summary_path = os.path.join(config.results_dir, 'loso_training_summary.json')
    with open(summary_path, 'w') as f:
        # Convert to JSON-serializable format
        json_summary = {}
        for key, value in results_summary.items():
            if key == 'config':
                json_summary[key] = asdict(value)
            elif key == 'loso_results':
                # Skip complex nested structures
                json_summary[key] = {
                    'accuracy_mean': value.get('accuracy', {}).get('mean', 0),
                    'accuracy_std': value.get('accuracy', {}).get('std', 0),
                    'precision_mean': value.get('precision', {}).get('mean', 0),
                    'precision_std': value.get('precision', {}).get('std', 0),
                    'recall_mean': value.get('recall', {}).get('mean', 0),
                    'recall_std': value.get('recall', {}).get('std', 0),
                    'f1_score_mean': value.get('f1_score', {}).get('mean', 0),
                    'f1_score_std': value.get('f1_score', {}).get('std', 0)
                }
            else:
                json_summary[key] = value
        json.dump(json_summary, f, indent=2)
    
    print(f"\nLOSO training completed!")
    print(f"Results saved to: {config.save_dir}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Summary saved to: {summary_path}")
    
    return loso_trainer, loso_results


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Visual Cortex Speckle Recognition")
    parser.add_argument("--config", type=str, help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["train", "train_kfold", "train_loso", "eval", "create_configs"], 
                       default="train", help="Mode to run")
    parser.add_argument("--model_path", type=str, help="Path to trained model (for evaluation)")
    
    # Override arguments
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--learning_rate", type=float, help="Learning rate")
    parser.add_argument("--subjects", type=str, nargs="+", help="Subjects to include")
    parser.add_argument("--shapes", type=str, nargs="+", help="Shapes to include")
    parser.add_argument("--feature_type", type=str, help="Feature type to use")
    
    # K-fold cross-validation arguments
    parser.add_argument("--k_folds", type=int, help="Number of K-folds for cross-validation")
    parser.add_argument("--kfold_stratified", action="store_true", help="Use stratified K-fold")
    parser.add_argument("--kfold_random_state", type=int, help="Random state for K-fold")
    
    # LOSO cross-validation arguments
    parser.add_argument("--loso_random_state", type=int, help="Random state for LOSO")
    
    args = parser.parse_args()
    
    # Skip create_configs mode for now
    if args.mode == "create_configs":
        print("Config templates creation not implemented yet.")
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
    
    # K-fold specific overrides
    if args.k_folds:
        config.training.k_folds = args.k_folds
    if args.kfold_stratified:
        config.training.kfold_stratified = args.kfold_stratified
    if args.kfold_random_state:
        config.training.kfold_random_state = args.kfold_random_state
    
    # LOSO specific overrides  
    if args.loso_random_state:
        config.training.loso_random_state = args.loso_random_state
    
    # Create directories
    os.makedirs(config.save_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    if config.log_dir:
        os.makedirs(config.log_dir, exist_ok=True)
    
    # Run mode
    if args.mode == "train":
        if config.training.use_kfold:
            print("K-fold cross-validation enabled in config. Using K-fold training.")
            train_model_kfold(config)
        else:
            train_model(config)
    elif args.mode == "train_kfold":
        train_model_kfold(config)
    elif args.mode == "train_loso":
        train_model_loso(config)
    elif args.mode == "eval":
        if not args.model_path:
            parser.error("--model_path is required for evaluation mode")
        evaluate_model(config, args.model_path)


if __name__ == "__main__":
    main()
