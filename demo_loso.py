#!/usr/bin/env python3
"""
Demo script for Leave-One-Subject-Out (LOSO) cross-validation
with Visual Cortex Speckle Imaging data.

This script demonstrates how to:
1. Load speckle data from multiple subjects
2. Set up LOSO cross-validation
3. Train models with subject-independent validation
4. Analyze results for clinical/research applications

Usage:
    python demo_loso.py

Requirements:
    - Processed speckle data pickle file
    - At least 3-4 subjects for meaningful LOSO validation
"""

import os
import sys
import pickle
import torch
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from training.loso_trainer import LOSOTrainer
    from utils.config import load_config
    print("✅ Successfully imported LOSO trainer")
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Please ensure you're running from the project root directory")
    sys.exit(1)


def demo_loso_with_real_data():
    """Demo LOSO with real speckle data."""
    print("🧠 Visual Cortex Speckle Imaging - LOSO Cross-Validation Demo")
    print("=" * 60)
    
    # Configuration
    data_path = "data/vis_cortex_data.pickle"  # Update this path
    config_path = "configs/loso_convlstm.yaml"
    
    # Check if data exists
    if not os.path.exists(data_path):
        print(f"❌ Data file not found: {data_path}")
        print("Please update the data_path variable or run with synthetic data")
        return demo_loso_with_synthetic_data()
    
    # Load configuration
    if os.path.exists(config_path):
        config = load_config(config_path)
        print(f"✅ Loaded configuration from {config_path}")
    else:
        print(f"⚠️  Config file not found: {config_path}")
        print("Using default parameters")
        config = get_default_loso_config()
    
    # Load speckle data
    print(f"\n📂 Loading speckle data from {data_path}...")
    try:
        with open(data_path, 'rb') as f:
            speckle_data = pickle.load(f)
        
        subjects = list(speckle_data.keys())
        print(f"✅ Found data for {len(subjects)} subjects: {subjects}")
        
        if len(subjects) < 3:
            print("⚠️  LOSO requires at least 3 subjects for meaningful validation")
            print("Using synthetic data for demonstration...")
            return demo_loso_with_synthetic_data()
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return demo_loso_with_synthetic_data()
    
    # Set up LOSO trainer
    print("\n🤖 Setting up LOSO trainer...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_config = {
        'model_type': 'convlstm',
        'num_classes': 3,  # Will be updated based on data
        'hidden_size': 64,
        'sequence_length': 64,
        'dropout_rate': 0.1
    }
    
    training_config = {
        'batch_size': 32,
        'epochs': 50,  # Reduced for demo
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'early_stopping_patience': 10,
        'save_best': True
    }
    
    loso_trainer = LOSOTrainer(
        model_config=model_config,
        training_config=training_config,
        device=device,
        save_dir='demo_loso_results',
        verbose=True
    )
    
    # Run LOSO cross-validation
    print("\n🔄 Starting LOSO cross-validation...")
    print("This will train one model per subject, using the others for training")
    
    loso_results = loso_trainer.train_loso(
        speckle_data=speckle_data,
        subjects=subjects,
        class_names=['Circle', 'Rectangle', 'Triangle'],
        feature_type='manhattan',
        n_chunks=30,  # Reduced for demo
        random_state=42
    )
    
    # Analyze results
    print("\n📊 LOSO Cross-Validation Results:")
    print("=" * 40)
    
    print(f"Mean Accuracy: {loso_results['accuracy']['mean']:.3f} ± {loso_results['accuracy']['std']:.3f}")
    print(f"Mean F1-Score: {loso_results['f1_score']['mean']:.3f} ± {loso_results['f1_score']['std']:.3f}")
    
    print("\nPer-Subject Results:")
    for subject, result in loso_trainer.subject_results.items():
        print(f"  {subject}: Accuracy = {result['accuracy']:.3f}, F1 = {result['f1_score']:.3f}")
    
    # Clinical interpretation
    print("\n🏥 Clinical Interpretation:")
    accuracies = list(loso_results['accuracy']['values'])
    clinical_threshold = 0.75  # Example clinical threshold
    subjects_above_threshold = sum(1 for acc in accuracies if acc >= clinical_threshold)
    
    print(f"Subjects above clinical threshold ({clinical_threshold:.0%}): {subjects_above_threshold}/{len(subjects)}")
    print(f"Clinical readiness: {'✅ Ready' if subjects_above_threshold/len(subjects) >= 0.75 else '⚠️  Needs improvement'}")
    
    # Best model
    best_model, best_subject = loso_trainer.get_best_model()
    loso_trainer.save_best_model('demo_loso_best_model.pth')
    print(f"\n🏆 Best performing test subject: {best_subject}")
    print(f"Best model saved to: demo_loso_best_model.pth")
    
    print(f"\n📁 Results saved to: demo_loso_results/")
    print("Check the generated visualizations and reports!")


def demo_loso_with_synthetic_data():
    """Demo LOSO with synthetic data for testing."""
    print("🎲 Running LOSO demo with synthetic data...")
    print("=" * 50)
    
    # Create synthetic speckle data for multiple subjects
    subjects = ['Subject1', 'Subject2', 'Subject3', 'Subject4', 'Subject5']
    shapes = ['Circle', 'Rectangle', 'Triangle']
    
    print(f"Creating synthetic data for {len(subjects)} subjects...")
    
    speckle_data = {}
    for subject in subjects:
        subject_data = []
        for shape in shapes:
            for video_idx in range(2):  # 2 videos per shape per subject
                # Create synthetic speckle features
                features = np.random.randn(100).tolist()  # 100 frame differences
                video_data = (
                    f'synthetic_{subject}_{shape}_{video_idx}',  # path
                    shape,  # label
                    features,  # m_norm
                    features,  # z_norm
                    features,  # euclidean
                    features,  # manhattan
                    features   # ncc
                )
                subject_data.append(video_data)
        speckle_data[subject] = subject_data
    
    print(f"✅ Created synthetic data for {len(subjects)} subjects with {len(shapes)} shapes each")
    
    # Set up LOSO trainer
    device = torch.device('cpu')  # Use CPU for demo
    
    model_config = {
        'model_type': 'conv1d',  # Simpler model for demo
        'num_classes': 3,
        'num_filters': 32,
        'kernel_size': 3,
        'dropout_rate': 0.1
    }
    
    training_config = {
        'batch_size': 16,
        'epochs': 10,  # Very few epochs for demo
        'learning_rate': 0.01,
        'optimizer': 'adam',
        'early_stopping_patience': 5,
        'save_best': True
    }
    
    loso_trainer = LOSOTrainer(
        model_config=model_config,
        training_config=training_config,
        device=device,
        save_dir='demo_synthetic_loso_results',
        verbose=True
    )
    
    # Run LOSO
    print("\n🔄 Running LOSO with synthetic data (quick demo)...")
    
    loso_results = loso_trainer.train_loso(
        speckle_data=speckle_data,
        subjects=subjects,
        class_names=shapes,
        feature_type='manhattan',
        n_chunks=10,  # Small chunks for demo
        random_state=42
    )
    
    # Show results
    print("\n📊 Synthetic LOSO Results:")
    print("=" * 30)
    
    print(f"Mean Accuracy: {loso_results['accuracy']['mean']:.3f} ± {loso_results['accuracy']['std']:.3f}")
    
    print("\nPer-Subject Results:")
    for subject, result in loso_trainer.subject_results.items():
        print(f"  {subject}: {result['accuracy']:.3f}")
    
    print("\n✅ Demo completed successfully!")
    print("Check demo_synthetic_loso_results/ for outputs")


def get_default_loso_config():
    """Get default LOSO configuration."""
    from types import SimpleNamespace
    
    config = SimpleNamespace()
    config.data = SimpleNamespace()
    config.model = SimpleNamespace()
    config.training = SimpleNamespace()
    
    # Default values
    config.data.subjects = None
    config.data.shape_filter = ['Circle', 'Rectangle', 'Triangle']
    config.data.feature_type = 'manhattan'
    config.data.n_chunks = 30
    
    config.model.model_type = 'convlstm'
    config.model.num_classes = 3
    config.model.hidden_size = 64
    config.model.dropout_rate = 0.1
    
    config.training.batch_size = 32
    config.training.epochs = 50
    config.training.learning_rate = 0.001
    config.training.optimizer = 'adam'
    
    return config


if __name__ == "__main__":
    print("🚀 Starting Visual Cortex Speckle Imaging LOSO Demo")
    print("This demo shows Leave-One-Subject-Out cross-validation")
    print("for subject-independent neural activity classification\n")
    
    # Run the appropriate demo
    try:
        demo_loso_with_real_data()
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        print("Falling back to synthetic data demo...")
        demo_loso_with_synthetic_data()
    
    print("\n🎯 For more information, see:")
    print("  - configs/loso_convlstm.yaml for configuration options")
    print("  - README.md for detailed LOSO documentation")
    print("  - Use 'make train-loso' for production training")
