#!/usr/bin/env python3
"""
Script to generate synthetic shape dataset and train models.

This script provides a complete pipeline for:
1. Generating synthetic shape videos
2. Processing them for speckle pattern analysis
3. Training classification models
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.data.shape_dataset_generator import ShapeDatasetGenerator

# ML-related imports (only needed for training, not generation)
try:
    from src.data.synthetic_shape_dataset import SyntheticShapeDataLoader
    from src.utils.config import Config
    from src.models.conv1d import Conv1DClassifier
    from src.training.trainer import Trainer
    import torch
    ML_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ML dependencies not available: {e}")
    print("Only dataset generation will be available.")
    ML_AVAILABLE = False


def generate_synthetic_dataset(config_or_args) -> str:
    """Generate synthetic shape dataset without ML dependencies."""
    print("=" * 60)
    print("GENERATING SYNTHETIC SHAPE DATASET")
    print("=" * 60)
    
    # Handle both config object and simple arguments
    if hasattr(config_or_args, 'data'):
        # Config object
        frame_width = config_or_args.data.frame_width
        frame_height = config_or_args.data.frame_height
        frames_per_video = config_or_args.data.frames_per_video
        fps = config_or_args.data.fps
        output_dir = config_or_args.data.output_dir
        videos_per_type = config_or_args.data.videos_per_type
        seed = config_or_args.seed
        use_colorized = getattr(config_or_args.data, 'use_colorized_shapes', False)
        motion_speed = getattr(config_or_args.data, 'motion_speed', 4.0)
    else:
        # Simple arguments dict
        frame_width = config_or_args.get('frame_width', 640)
        frame_height = config_or_args.get('frame_height', 480)
        frames_per_video = config_or_args.get('frames_per_video', 100)
        fps = config_or_args.get('fps', 30)
        output_dir = config_or_args.get('output_dir', 'data/synthetic_shapes')
        videos_per_type = config_or_args.get('videos_per_type', 50)
        seed = config_or_args.get('seed', 42)
        use_colorized = config_or_args.get('use_colorized_shapes', False)
        motion_speed = config_or_args.get('motion_speed', 4.0)
    
    # Initialize generator (no ML dependencies needed)
    generator = ShapeDatasetGenerator(
        frame_width=frame_width,
        frame_height=frame_height,
        frames_per_video=frames_per_video,
        fps=fps,
        use_colorized_shapes=use_colorized,
        motion_speed=motion_speed
    )
    
    # Generate dataset
    output_dir = output_dir
    stats = generator.generate_dataset(
        output_dir=output_dir,
        videos_per_type=videos_per_type,
        seed=seed
    )
    
    print(f"\\nDataset generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Total videos: {stats['total_videos']}")
    
    return output_dir


def train_synthetic_model(config, dataset_dir: str):
    """Train model on synthetic shape dataset."""
    if not ML_AVAILABLE:
        print("Error: ML dependencies (PyTorch, etc.) are not available!")
        print("Install full requirements.txt to enable training functionality.")
        return None, None, None
        
    print("\\n" + "=" * 60)
    print("TRAINING MODEL ON SYNTHETIC DATASET")
    print("=" * 60)
    
    # Create data loaders
    data_loader = SyntheticShapeDataLoader(
        batch_size=config.training.batch_size,
        num_workers=4
    )
    
    train_loader, val_loader, test_loader = data_loader.create_data_loaders(
        root_dir=dataset_dir,
        video_types=config.data.video_types if hasattr(config.data, 'video_types') else None,
        feature_type=config.processing.feature_type,
        n_chunks=config.processing.n_chunks,
        frames_limit=config.processing.frames_limit,
        train_split=0.7,
        val_split=0.15,
        test_split=0.15,
        shuffle=True,
        seed=config.seed
    )
    
    # Get a sample to determine input dimensions
    sample_features, _ = next(iter(train_loader))
    input_dim = sample_features.shape[-1]
    sequence_length = sample_features.shape[1]
    
    print(f"Input dimensions: {input_dim}")
    print(f"Sequence length: {sequence_length}")
    print(f"Number of classes: {config.model.num_classes}")
    
    # Create model
    model = Conv1DClassifier(
        input_dim=input_dim,
        num_classes=config.model.num_classes,
        num_filters=config.model.num_filters,
        kernel_size=config.model.kernel_size,
        dropout_rate=config.model.dropout_rate
    )
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() and config.device == "auto" else "cpu")
    model = model.to(device)
    
    print(f"Using device: {device}")
    print(f"Model architecture: {model}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )
    
    # Train model
    best_model, train_history = trainer.train()
    
    # Evaluate on test set
    test_metrics = trainer.evaluate(test_loader, dataset_name='test_synthetic')
    
    print(f"\\nTraining completed!")
    print(f"Best validation accuracy: {max([h['val_accuracy'] for h in train_history]):.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")
    
    return best_model, train_history, test_metrics


def main():
    """Main function to run the complete pipeline."""
    parser = argparse.ArgumentParser(description="Generate synthetic shape dataset and train models")
    parser.add_argument("--config", type=str, default="configs/synthetic_shapes.yaml",
                       help="Path to configuration file")
    parser.add_argument("--generate-only", action="store_true",
                       help="Only generate dataset, don't train")
    parser.add_argument("--train-only", action="store_true",
                       help="Only train (dataset must exist)")
    parser.add_argument("--dataset-dir", type=str, default=None,
                       help="Path to existing dataset (for train-only mode)")
    
    args = parser.parse_args()
    
    # Load configuration (only if ML dependencies available)
    config = None
    if ML_AVAILABLE and os.path.exists(args.config):
        config = Config(args.config)
        print(f"Using configuration: {args.config}")
        print(f"Random seed: {config.seed}")
    elif not args.generate_only:
        print("Warning: No config file found or ML dependencies missing.")
        print("Switching to generate-only mode.")
        args.generate_only = True
    
    # For generation-only with no config, use simple defaults
    if args.generate_only and config is None:
        config = {
            'frame_width': 640,
            'frame_height': 480,
            'frames_per_video': 100,
            'fps': 30,
            'output_dir': 'data/synthetic_shapes',
            'videos_per_type': 50,
            'seed': 42,
            'use_colorized_shapes': True,
            'motion_speed': 4.0
        }
        print("Using default generation parameters")
    
    # Set random seeds if available
    if ML_AVAILABLE and hasattr(config, 'seed'):
        torch.manual_seed(config.seed)
        seed = config.seed
    elif isinstance(config, dict):
        seed = config['seed']
    else:
        seed = 42
        
    # Set numpy seed (always available for generation)
    import numpy as np
    import random
    np.random.seed(seed)
    random.seed(seed)
    
    dataset_dir = None
    
    # Generate dataset if needed
    if not args.train_only:
        dataset_dir = generate_synthetic_dataset(config)
    else:
        if isinstance(config, dict):
            dataset_dir = args.dataset_dir or config['output_dir']
        else:
            dataset_dir = args.dataset_dir or config.data.output_dir
        if not os.path.exists(dataset_dir):
            print(f"Error: Dataset directory {dataset_dir} does not exist!")
            print("Run without --train-only to generate dataset first.")
            return
    
    # Train model if needed and ML dependencies available
    if not args.generate_only:
        if not ML_AVAILABLE:
            print("\\nSkipping training: ML dependencies not available.")
            print("Install full requirements.txt to enable training.")
        else:
            model, history, metrics = train_synthetic_model(config, dataset_dir)
            
            if model is not None:  # Training succeeded
                # Save results
                if hasattr(config, 'results_dir'):
                    results_dir = config.results_dir
                else:
                    results_dir = 'results'
                os.makedirs(results_dir, exist_ok=True)
                
                # Save training history
                import json
                history_path = os.path.join(results_dir, "training_history.json")
                with open(history_path, 'w') as f:
                    json.dump(history, f, indent=2)
                
                # Save test metrics
                metrics_path = os.path.join(results_dir, "test_metrics.json") 
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=2)
                
                print(f"\\nResults saved to: {results_dir}")
    
    print("\\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
