"""
Unit tests for K-fold cross-validation trainer.
"""

import unittest
import torch
import numpy as np
import tempfile
import shutil
import os
from unittest.mock import Mock, patch

from src.training.kfold_trainer import KFoldTrainer
from src.data.dataset import SpeckleDataset


class TestKFoldTrainer(unittest.TestCase):
    """Test cases for K-fold cross-validation trainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.device = torch.device('cpu')
        
        # Create mock dataset
        self.num_samples = 50
        self.num_classes = 3
        self.input_dim = 64
        
        features = np.random.randn(self.num_samples, self.input_dim, 1)
        labels = np.repeat(np.arange(self.num_classes), self.num_samples // self.num_classes)
        # Add remaining samples
        remaining = self.num_samples - len(labels)
        if remaining > 0:
            labels = np.concatenate([labels, np.random.randint(0, self.num_classes, remaining)])
        
        # Shuffle
        indices = np.random.permutation(self.num_samples)
        self.features = features[indices]
        self.labels = labels[indices]
        
        self.dataset = SpeckleDataset(self.features, self.labels)
        self.class_names = ['Class_0', 'Class_1', 'Class_2']
        
        # Model and training configurations
        self.model_config = {
            'model_type': 'conv1d',
            'num_classes': self.num_classes,
            'input_channels': 1,
            'num_filters': 8,  # Small for testing
            'kernel_size': 3,
            'dropout_rate': 0.1
        }
        
        self.training_config = {
            'batch_size': 8,
            'epochs': 2,  # Very small for testing
            'learning_rate': 0.01,
            'optimizer': 'adam',
            'scheduler': 'plateau',
            'early_stopping_patience': 1,
            'save_every': 1,
            'save_best': True
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_kfold_trainer_initialization(self):
        """Test K-fold trainer initialization."""
        trainer = KFoldTrainer(
            model_config=self.model_config,
            training_config=self.training_config,
            device=self.device,
            save_dir=self.test_dir,
            verbose=False
        )
        
        self.assertEqual(trainer.device, self.device)
        self.assertEqual(trainer.save_dir, self.test_dir)
        self.assertTrue(trainer.stratified)
        self.assertEqual(len(trainer.fold_results), 0)
        self.assertEqual(len(trainer.fold_histories), 0)
        self.assertEqual(len(trainer.fold_models), 0)
    
    def test_kfold_training(self):
        """Test K-fold training process."""
        trainer = KFoldTrainer(
            model_config=self.model_config,
            training_config=self.training_config,
            device=self.device,
            save_dir=self.test_dir,
            verbose=False
        )
        
        cv_results = trainer.train_kfold(
            dataset=self.dataset,
            class_names=self.class_names,
            k_folds=3,  # Small for testing
            random_state=42
        )
        
        # Check results structure
        self.assertIn('accuracy', cv_results)
        self.assertIn('precision', cv_results)
        self.assertIn('recall', cv_results)
        self.assertIn('f1_score', cv_results)
        self.assertIn('loss', cv_results)
        
        # Check statistics
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'loss']:
            self.assertIn('mean', cv_results[metric])
            self.assertIn('std', cv_results[metric])
            self.assertIn('min', cv_results[metric])
            self.assertIn('max', cv_results[metric])
            self.assertIn('values', cv_results[metric])
            self.assertEqual(len(cv_results[metric]['values']), 3)
        
        # Check that models were trained
        self.assertEqual(len(trainer.fold_results), 3)
        self.assertEqual(len(trainer.fold_histories), 3)
        self.assertEqual(len(trainer.fold_models), 3)
    
    def test_best_model_selection(self):
        """Test best model selection."""
        trainer = KFoldTrainer(
            model_config=self.model_config,
            training_config=self.training_config,
            device=self.device,
            save_dir=self.test_dir,
            verbose=False
        )
        
        # Train K-fold
        trainer.train_kfold(
            dataset=self.dataset,
            class_names=self.class_names,
            k_folds=3,
            random_state=42
        )
        
        # Get best model
        best_model, best_fold_idx = trainer.get_best_model()
        
        # Check that best model is returned
        self.assertIsNotNone(best_model)
        self.assertIsInstance(best_fold_idx, int)
        self.assertGreaterEqual(best_fold_idx, 0)
        self.assertLess(best_fold_idx, 3)
        
        # Check that it's actually the best
        accuracies = [result['accuracy'] for result in trainer.fold_results]
        expected_best_idx = np.argmax(accuracies)
        self.assertEqual(best_fold_idx, expected_best_idx)
    
    def test_model_saving(self):
        """Test model saving functionality."""
        trainer = KFoldTrainer(
            model_config=self.model_config,
            training_config=self.training_config,
            device=self.device,
            save_dir=self.test_dir,
            verbose=False
        )
        
        # Train K-fold
        trainer.train_kfold(
            dataset=self.dataset,
            class_names=self.class_names,
            k_folds=2,
            random_state=42
        )
        
        # Save best model
        model_path = os.path.join(self.test_dir, 'test_best_model.pth')
        trainer.save_best_model(model_path)
        
        # Check that file was created
        self.assertTrue(os.path.exists(model_path))
        
        # Load and check contents
        checkpoint = torch.load(model_path, map_location='cpu')
        self.assertIn('model_state_dict', checkpoint)
        self.assertIn('model_config', checkpoint)
        self.assertIn('training_config', checkpoint)
        self.assertIn('best_fold', checkpoint)
        self.assertIn('best_accuracy', checkpoint)
        self.assertIn('cv_results', checkpoint)
    
    def test_stratified_kfold(self):
        """Test stratified K-fold splitting."""
        trainer = KFoldTrainer(
            model_config=self.model_config,
            training_config=self.training_config,
            device=self.device,
            save_dir=self.test_dir,
            stratified=True,
            verbose=False
        )
        
        # Create imbalanced dataset
        imbalanced_features = np.random.randn(30, self.input_dim, 1)
        imbalanced_labels = np.array([0]*20 + [1]*8 + [2]*2)  # Imbalanced
        imbalanced_dataset = SpeckleDataset(imbalanced_features, imbalanced_labels)
        
        cv_results = trainer.train_kfold(
            dataset=imbalanced_dataset,
            class_names=self.class_names,
            k_folds=3,
            random_state=42
        )
        
        # Should complete without errors
        self.assertIn('accuracy', cv_results)
    
    def test_non_stratified_kfold(self):
        """Test non-stratified K-fold splitting."""
        trainer = KFoldTrainer(
            model_config=self.model_config,
            training_config=self.training_config,
            device=self.device,
            save_dir=self.test_dir,
            stratified=False,
            verbose=False
        )
        
        cv_results = trainer.train_kfold(
            dataset=self.dataset,
            class_names=self.class_names,
            k_folds=3,
            random_state=42
        )
        
        # Should complete without errors
        self.assertIn('accuracy', cv_results)
    
    def test_results_compilation(self):
        """Test results compilation."""
        trainer = KFoldTrainer(
            model_config=self.model_config,
            training_config=self.training_config,
            device=self.device,
            save_dir=self.test_dir,
            verbose=False
        )
        
        # Mock some results
        trainer.fold_results = [
            {'accuracy': 0.8, 'precision': 0.75, 'recall': 0.82, 'f1_score': 0.78, 'loss': 0.5,
             'classification_report': {'Class_0': {'precision': 0.8, 'recall': 0.7, 'f1-score': 0.75}}},
            {'accuracy': 0.85, 'precision': 0.80, 'recall': 0.85, 'f1_score': 0.82, 'loss': 0.45,
             'classification_report': {'Class_0': {'precision': 0.85, 'recall': 0.8, 'f1-score': 0.82}}}
        ]
        
        cv_results = trainer._compile_results(self.class_names)
        
        # Check compiled statistics
        self.assertAlmostEqual(cv_results['accuracy']['mean'], 0.825)
        self.assertAlmostEqual(cv_results['accuracy']['std'], 0.025)
        self.assertEqual(cv_results['accuracy']['min'], 0.8)
        self.assertEqual(cv_results['accuracy']['max'], 0.85)
    
    def test_file_creation(self):
        """Test that expected files are created."""
        trainer = KFoldTrainer(
            model_config=self.model_config,
            training_config=self.training_config,
            device=self.device,
            save_dir=self.test_dir,
            verbose=False
        )
        
        trainer.train_kfold(
            dataset=self.dataset,
            class_names=self.class_names,
            k_folds=2,
            random_state=42
        )
        
        # Check that expected files exist
        expected_files = [
            'kfold_summary.json',
            'kfold_metrics.csv',
            'kfold_results_visualization.png',
            'kfold_training_history.png'
        ]
        
        for filename in expected_files:
            filepath = os.path.join(self.test_dir, filename)
            self.assertTrue(os.path.exists(filepath), f"File {filename} was not created")
        
        # Check fold directories
        for fold in range(2):
            fold_dir = os.path.join(self.test_dir, f'fold_{fold + 1}')
            self.assertTrue(os.path.exists(fold_dir), f"Fold directory {fold_dir} was not created")


class TestKFoldIntegration(unittest.TestCase):
    """Integration tests for K-fold functionality."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.test_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up integration test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_different_model_types(self):
        """Test K-fold with different model types."""
        # Create small dataset
        features = np.random.randn(24, 32, 1)
        labels = np.array([0]*8 + [1]*8 + [2]*8)
        dataset = SpeckleDataset(features, labels)
        
        model_types = ['conv1d']  # Test available models
        
        for model_type in model_types:
            with self.subTest(model_type=model_type):
                model_config = {
                    'model_type': model_type,
                    'num_classes': 3,
                    'input_channels': 1,
                    'num_filters': 4,
                    'kernel_size': 3,
                    'dropout_rate': 0.0
                }
                
                training_config = {
                    'batch_size': 4,
                    'epochs': 1,
                    'learning_rate': 0.01,
                    'optimizer': 'adam',
                    'early_stopping_patience': 1
                }
                
                trainer = KFoldTrainer(
                    model_config=model_config,
                    training_config=training_config,
                    device=torch.device('cpu'),
                    save_dir=os.path.join(self.test_dir, model_type),
                    verbose=False
                )
                
                cv_results = trainer.train_kfold(
                    dataset=dataset,
                    k_folds=2,
                    random_state=42
                )
                
                self.assertIn('accuracy', cv_results)


if __name__ == '__main__':
    unittest.main()
