"""
Test Leave-One-Subject-Out (LOSO) cross-validation trainer functionality.
"""

import unittest
import tempfile
import shutil
import torch
import numpy as np
import os

from src.training.loso_trainer import LOSOTrainer, LOSOSubjectDataset
from src.data.dataset import SpeckleDataset


class TestLOSOTrainer(unittest.TestCase):
    """Test cases for LOSO cross-validation trainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.device = torch.device('cpu')
        
        # Create mock speckle data with multiple subjects
        self.subjects = ['subject1', 'subject2', 'subject3', 'subject4']
        self.speckle_data = {}
        
        for subject in self.subjects:
            subject_data = []
            # Create mock data for each subject
            for shape in ['Circle', 'Rectangle', 'Triangle']:
                for _ in range(3):  # 3 videos per shape per subject
                    # Mock video data: (path, label, m_norm, z_norm, euclidean, manhattan, ncc)
                    mock_features = list(np.random.randn(50))  # 50 frame differences
                    subject_data.append((
                        f'path_{subject}_{shape}',
                        shape,
                        mock_features,  # m_norm
                        mock_features,  # z_norm  
                        mock_features,  # euclidean
                        mock_features,  # manhattan
                        mock_features   # ncc
                    ))
            self.speckle_data[subject] = subject_data
        
        self.class_names = ['Circle', 'Rectangle', 'Triangle']
        
        # Model and training configs for testing
        self.model_config = {
            'model_type': 'conv1d',
            'num_classes': 3,
            'input_channels': 1,
            'num_filters': 32,
            'kernel_size': 3,
            'dropout_rate': 0.1
        }
        
        self.training_config = {
            'batch_size': 16,
            'epochs': 5,  # Small for testing
            'learning_rate': 0.01,
            'optimizer': 'adam',
            'scheduler': 'plateau',
            'early_stopping_patience': 3,
            'save_every': 10,
            'save_best': True
        }
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.test_dir)
    
    def test_loso_subject_dataset_creation(self):
        """Test LOSO subject dataset creation."""
        dataset = LOSOSubjectDataset(
            speckle_data=self.speckle_data,
            subjects=self.subjects,
            shape_filter=['Circle', 'Rectangle', 'Triangle'],
            feature_type='manhattan',
            n_chunks=5
        )
        
        # Check that all subjects are included
        self.assertEqual(len(dataset.subject_data), len(self.subjects))
        
        # Check class names
        self.assertEqual(len(dataset.class_names), 3)
        
        # Check that each subject has data
        for subject in self.subjects:
            self.assertIn(subject, dataset.subject_data)
            self.assertGreater(len(dataset.subject_data[subject]['features']), 0)
    
    def test_get_subject_dataset(self):
        """Test getting dataset for specific subject."""
        dataset = LOSOSubjectDataset(
            speckle_data=self.speckle_data,
            subjects=self.subjects,
            feature_type='manhattan',
            n_chunks=5
        )
        
        # Get dataset for first subject
        subject_dataset = dataset.get_subject_dataset(self.subjects[0])
        
        # Check dataset properties
        self.assertIsInstance(subject_dataset, SpeckleDataset)
        self.assertGreater(len(subject_dataset), 0)
        
        # Test getting features and labels
        features, label = subject_dataset[0]
        self.assertIsInstance(features, torch.Tensor)
        self.assertIsInstance(label, torch.Tensor)
    
    def test_get_multi_subject_dataset(self):
        """Test getting combined dataset for multiple subjects."""
        dataset = LOSOSubjectDataset(
            speckle_data=self.speckle_data,
            subjects=self.subjects,
            feature_type='manhattan',
            n_chunks=5
        )
        
        # Get dataset for multiple subjects (excluding one for LOSO)
        train_subjects = self.subjects[:-1]  # All but last subject
        multi_dataset = dataset.get_multi_subject_dataset(train_subjects)
        
        # Check dataset properties
        self.assertIsInstance(multi_dataset, SpeckleDataset)
        self.assertGreater(len(multi_dataset), 0)
        
        # Should have data from multiple subjects
        single_subject_size = len(dataset.get_subject_dataset(train_subjects[0]))
        self.assertGreaterEqual(len(multi_dataset), single_subject_size)
    
    def test_loso_trainer_creation(self):
        """Test LOSO trainer initialization."""
        trainer = LOSOTrainer(
            model_config=self.model_config,
            training_config=self.training_config,
            device=self.device,
            save_dir=self.test_dir,
            verbose=False
        )
        
        # Check trainer properties
        self.assertEqual(trainer.device, self.device)
        self.assertEqual(trainer.save_dir, self.test_dir)
        self.assertFalse(trainer.verbose)
        
        # Check that save directory was created
        self.assertTrue(os.path.exists(self.test_dir))
    
    def test_loso_training_process(self):
        """Test LOSO training process."""
        trainer = LOSOTrainer(
            model_config=self.model_config,
            training_config=self.training_config,
            device=self.device,
            save_dir=self.test_dir,
            verbose=False
        )
        
        # Run LOSO training
        loso_results = trainer.train_loso(
            speckle_data=self.speckle_data,
            subjects=self.subjects,
            class_names=self.class_names,
            feature_type='manhattan',
            n_chunks=5,
            random_state=42
        )
        
        # Check results structure
        self.assertIn('accuracy', loso_results)
        self.assertIn('precision', loso_results)
        self.assertIn('recall', loso_results)
        self.assertIn('f1_score', loso_results)
        self.assertIn('loss', loso_results)
        
        # Check statistics for each metric
        for metric in ['accuracy', 'precision', 'recall', 'f1_score', 'loss']:
            self.assertIn('mean', loso_results[metric])
            self.assertIn('std', loso_results[metric])
            self.assertIn('min', loso_results[metric])
            self.assertIn('max', loso_results[metric])
            self.assertIn('values', loso_results[metric])
            self.assertEqual(len(loso_results[metric]['values']), len(self.subjects))
        
        # Check that models were trained for each subject
        self.assertEqual(len(trainer.loso_results), len(self.subjects))
        self.assertEqual(len(trainer.loso_histories), len(self.subjects))
        self.assertEqual(len(trainer.loso_models), len(self.subjects))
        
        # Check subject-specific results
        self.assertEqual(len(trainer.subject_results), len(self.subjects))
        for subject in self.subjects:
            self.assertIn(subject, trainer.subject_results)
    
    def test_best_model_selection(self):
        """Test best model selection functionality."""
        trainer = LOSOTrainer(
            model_config=self.model_config,
            training_config=self.training_config,
            device=self.device,
            save_dir=self.test_dir,
            verbose=False
        )
        
        # Train LOSO
        trainer.train_loso(
            speckle_data=self.speckle_data,
            subjects=self.subjects,
            class_names=self.class_names,
            feature_type='manhattan',
            n_chunks=5,
            random_state=42
        )
        
        # Get best model
        best_model, best_subject = trainer.get_best_model()
        
        # Check that best model and subject are valid
        self.assertIsNotNone(best_model)
        self.assertIn(best_subject, self.subjects)
        
        # Save best model
        model_path = os.path.join(self.test_dir, 'test_best_loso_model.pth')
        trainer.save_best_model(model_path)
        
        # Check that file was created
        self.assertTrue(os.path.exists(model_path))
        
        # Load and check contents
        checkpoint = torch.load(model_path, map_location='cpu')
        self.assertIn('model_state_dict', checkpoint)
        self.assertIn('model_config', checkpoint)
        self.assertIn('training_config', checkpoint)
        self.assertIn('best_subject', checkpoint)
        self.assertIn('best_accuracy', checkpoint)
        self.assertIn('loso_results', checkpoint)
    
    def test_subject_filtering(self):
        """Test filtering by specific subjects."""
        # Test with subset of subjects
        subset_subjects = self.subjects[:2]  # Only first 2 subjects
        
        trainer = LOSOTrainer(
            model_config=self.model_config,
            training_config=self.training_config,
            device=self.device,
            save_dir=self.test_dir,
            verbose=False
        )
        
        loso_results = trainer.train_loso(
            speckle_data=self.speckle_data,
            subjects=subset_subjects,
            class_names=self.class_names,
            feature_type='manhattan',
            n_chunks=5,
            random_state=42
        )
        
        # Should only have results for subset of subjects
        self.assertEqual(len(trainer.subject_results), len(subset_subjects))
        for subject in subset_subjects:
            self.assertIn(subject, trainer.subject_results)
    
    def test_shape_filtering(self):
        """Test filtering by specific shapes."""
        # Test with subset of shapes
        shape_filter = ['Circle', 'Rectangle']  # Exclude Triangle
        
        trainer = LOSOTrainer(
            model_config=self.model_config,
            training_config=self.training_config,
            device=self.device,
            save_dir=self.test_dir,
            verbose=False
        )
        
        # Update model config for 2 classes
        model_config = self.model_config.copy()
        model_config['num_classes'] = 2
        trainer.model_config = model_config
        
        loso_results = trainer.train_loso(
            speckle_data=self.speckle_data,
            subjects=self.subjects,
            shape_filter=shape_filter,
            feature_type='manhattan',
            n_chunks=5,
            random_state=42
        )
        
        # Should have results for all subjects but filtered shapes
        self.assertEqual(len(trainer.subject_results), len(self.subjects))
    
    def test_insufficient_subjects(self):
        """Test behavior with insufficient subjects for LOSO."""
        # Test with only one subject
        single_subject_data = {self.subjects[0]: self.speckle_data[self.subjects[0]]}
        
        trainer = LOSOTrainer(
            model_config=self.model_config,
            training_config=self.training_config,
            device=self.device,
            save_dir=self.test_dir,
            verbose=False
        )
        
        # Should handle gracefully or raise appropriate error
        with self.assertRaises((ValueError, RuntimeError)):
            trainer.train_loso(
                speckle_data=single_subject_data,
                subjects=[self.subjects[0]],
                class_names=self.class_names,
                feature_type='manhattan',
                n_chunks=5,
                random_state=42
            )


if __name__ == '__main__':
    unittest.main()
