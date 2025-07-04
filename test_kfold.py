"""
Test script for K-fold cross-validation functionality.
"""

import torch
import numpy as np
import sys
import os
from unittest.mock import Mock, patch

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.training.kfold_trainer import KFoldTrainer
from src.data.dataset import SpeckleDataset
from src.models.base_model import ModelFactory


def create_mock_dataset(num_samples=100, num_classes=3, input_dim=128):
    """Create a mock dataset for testing."""
    # Generate random features
    features = np.random.randn(num_samples, input_dim, 1)
    
    # Generate balanced labels
    labels = np.repeat(np.arange(num_classes), num_samples // num_classes)
    if len(labels) < num_samples:
        labels = np.concatenate([labels, np.random.randint(0, num_classes, num_samples - len(labels))])
    
    # Shuffle
    indices = np.random.permutation(num_samples)
    features = features[indices]
    labels = labels[indices]
    
    return SpeckleDataset(features, labels)


def test_kfold_trainer():
    """Test K-fold trainer functionality."""
    print("Testing K-fold cross-validation trainer...")
    
    # Set device
    device = torch.device("cpu")  # Use CPU for testing
    print(f"Using device: {device}")
    
    # Create mock dataset
    dataset = create_mock_dataset(num_samples=100, num_classes=3, input_dim=128)
    class_names = ['Class_0', 'Class_1', 'Class_2']
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Class names: {class_names}")
    
    # Model and training configuration
    model_config = {
        'model_type': 'conv1d',
        'num_classes': 3,
        'input_channels': 1,
        'num_filters': 16,  # Small for testing
        'kernel_size': 3,
        'dropout_rate': 0.1
    }
    
    training_config = {
        'batch_size': 16,
        'epochs': 5,  # Small for testing
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'scheduler': 'plateau',
        'early_stopping_patience': 3,
        'save_every': 10,
        'save_best': True
    }
    
    # Create K-fold trainer
    kfold_trainer = KFoldTrainer(
        model_config=model_config,
        training_config=training_config,
        device=device,
        save_dir='test_kfold_results',
        log_dir='test_kfold_logs',
        stratified=True,
        verbose=True
    )
    
    print("Starting K-fold training...")
    
    try:
        # Perform K-fold cross-validation
        cv_results = kfold_trainer.train_kfold(
            dataset=dataset,
            class_names=class_names,
            k_folds=3,  # Small for testing
            random_state=42
        )
        
        print("\nK-fold training completed successfully!")
        
        # Print results
        print(f"Average accuracy: {cv_results['accuracy']['mean']:.4f} ± {cv_results['accuracy']['std']:.4f}")
        print(f"Average precision: {cv_results['precision']['mean']:.4f} ± {cv_results['precision']['std']:.4f}")
        print(f"Average recall: {cv_results['recall']['mean']:.4f} ± {cv_results['recall']['std']:.4f}")
        print(f"Average F1-score: {cv_results['f1_score']['mean']:.4f} ± {cv_results['f1_score']['std']:.4f}")
        
        # Test best model retrieval
        best_model, best_fold_idx = kfold_trainer.get_best_model()
        print(f"Best model from fold: {best_fold_idx + 1}")
        
        # Test model saving
        kfold_trainer.save_best_model('test_best_kfold_model.pth')
        print("Best model saved successfully!")
        
        # Verify files were created
        expected_files = [
            'test_kfold_results/kfold_summary.json',
            'test_kfold_results/kfold_metrics.csv',
            'test_kfold_results/kfold_results_visualization.png',
            'test_kfold_results/kfold_training_history.png',
            'test_best_kfold_model.pth'
        ]
        
        for file_path in expected_files:
            if os.path.exists(file_path):
                print(f"✓ {file_path} created")
            else:
                print(f"✗ {file_path} NOT created")
        
        print("\n🎉 K-fold trainer test completed successfully!")
        
        return True
        
    except Exception as e:
        print(f"❌ K-fold trainer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_factory():
    """Test model factory with different models."""
    print("\nTesting model factory...")
    
    try:
        # Test Conv1D model
        model1 = ModelFactory.create_model(
            'conv1d',
            num_classes=3,
            input_channels=1,
            num_filters=16,
            kernel_size=3,
            dropout_rate=0.1
        )
        print(f"✓ Conv1D model created: {model1.__class__.__name__}")
        
        # Test with sample input
        sample_input = torch.randn(2, 128, 1)
        output = model1(sample_input)
        print(f"✓ Conv1D output shape: {output.shape}")
        
        # Test ConvLSTM model if available
        available_models = ModelFactory.list_models()
        if 'convlstm' in available_models:
            model2 = ModelFactory.create_model(
                'convlstm',
                num_classes=3,
                input_channels=1,
                hidden_size=16,
                sequence_length=32
            )
            print(f"✓ ConvLSTM model created: {model2.__class__.__name__}")
        
        print("✓ Model factory test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Model factory test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def cleanup_test_files():
    """Clean up test files."""
    print("\nCleaning up test files...")
    
    import shutil
    
    # Remove test directories
    test_dirs = ['test_kfold_results', 'test_kfold_logs']
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print(f"✓ Removed {test_dir}")
    
    # Remove test files
    test_files = ['test_best_kfold_model.pth']
    for test_file in test_files:
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"✓ Removed {test_file}")


if __name__ == "__main__":
    print("K-fold Cross-Validation Test Suite")
    print("=" * 50)
    
    try:
        # Test model factory first
        if not test_model_factory():
            print("❌ Model factory test failed, skipping K-fold test")
            sys.exit(1)
        
        # Test K-fold trainer
        if not test_kfold_trainer():
            print("❌ K-fold trainer test failed")
            sys.exit(1)
        
        print("\n🎉 All tests passed successfully!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_test_files()
