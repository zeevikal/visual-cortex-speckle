"""
Test interpretability functionality for visual cortex speckle imaging.
"""

import unittest
import torch
import numpy as np
import tempfile
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.interpretability import SpeckleInterpretabilityAnalyzer
from models.convlstm import ConvLSTMClassifier


class TestInterpretabilityAnalyzer(unittest.TestCase):
    """Test the interpretability analyzer functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cpu')  # Use CPU for testing
        self.class_names = ['Circle', 'Rectangle', 'Triangle']
        
        # Create a simple model for testing
        self.model = ConvLSTMClassifier(
            num_classes=3,
            hidden_size=32,  # Smaller for faster testing
            sequence_length=32,
            dropout_rate=0.1
        )
        self.model.eval()
        
        # Create test data
        self.batch_size = 8
        self.sequence_length = 32
        self.data_length = 64
        
        self.X_test = torch.randn(self.batch_size, self.sequence_length, 1, self.data_length)
        self.y_test = torch.randint(0, 3, (self.batch_size,))
        
        # Create temporary directory for results
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up temporary directory
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_analyzer_initialization(self):
        """Test that the analyzer initializes correctly."""
        analyzer = SpeckleInterpretabilityAnalyzer(
            model=self.model,
            device=self.device,
            class_names=self.class_names
        )
        
        self.assertEqual(analyzer.class_names, self.class_names)
        self.assertEqual(analyzer.device, self.device)
        self.assertIsNotNone(analyzer.model)
    
    def test_get_num_classes(self):
        """Test that the number of classes is detected correctly."""
        analyzer = SpeckleInterpretabilityAnalyzer(
            model=self.model,
            device=self.device,
            class_names=self.class_names
        )
        
        num_classes = analyzer._get_num_classes()
        self.assertEqual(num_classes, 3)
    
    def test_temporal_pattern_analysis(self):
        """Test temporal pattern analysis functionality."""
        analyzer = SpeckleInterpretabilityAnalyzer(
            model=self.model,
            device=self.device,
            class_names=self.class_names
        )
        
        # Run temporal analysis
        temporal_results = analyzer.analyze_temporal_patterns(
            X_test=self.X_test,
            y_test=self.y_test,
            sequence_length=self.sequence_length,
            save_dir=None  # Don't save during testing
        )
        
        self.assertIsInstance(temporal_results, dict)
        # Should have at least temporal_patterns
        self.assertIn('temporal_patterns', temporal_results)
        
        patterns = temporal_results['temporal_patterns']
        self.assertIn('labels', patterns)
        self.assertIn('pca_features', patterns)
        self.assertIn('tsne_features', patterns)
    
    def test_spatial_cortical_mapping(self):
        """Test spatial cortical mapping functionality."""
        analyzer = SpeckleInterpretabilityAnalyzer(
            model=self.model,
            device=self.device,
            class_names=self.class_names
        )
        
        # Define test cortical regions
        cortical_regions = {
            'V1_edge_detectors': (0, 512),
            'V1_orientation_columns': (512, 1024),
            'V1_spatial_frequency': (1024, 1536),
            'V1_higher_order': (1536, 2048),
        }
        
        # Run spatial analysis
        spatial_results = analyzer.analyze_spatial_cortical_mapping(
            X_test=self.X_test,
            y_test=self.y_test,
            cortical_regions=cortical_regions,
            save_dir=None  # Don't save during testing
        )
        
        self.assertIsInstance(spatial_results, dict)
        # Should have spatial gradients and shape patterns
        self.assertIn('spatial_gradients', spatial_results)
        self.assertIn('shape_patterns', spatial_results)
    
    def test_cortical_connectivity_analysis(self):
        """Test cortical connectivity analysis."""
        analyzer = SpeckleInterpretabilityAnalyzer(
            model=self.model,
            device=self.device,
            class_names=self.class_names
        )
        
        cortical_regions = {
            'V1_edge_detectors': (0, 256),
            'V1_orientation_columns': (256, 512),
        }
        
        # Test connectivity analysis
        connectivity = analyzer._analyze_cortical_connectivity(
            self.X_test, self.y_test, cortical_regions
        )
        
        self.assertIsInstance(connectivity, dict)
        # Should have connectivity data for each class
        for class_name in self.class_names:
            if class_name in connectivity:
                self.assertIn('connectivity_matrix', connectivity[class_name])
                self.assertIn('region_names', connectivity[class_name])
    
    def test_find_strongest_connections(self):
        """Test finding strongest connections between regions."""
        analyzer = SpeckleInterpretabilityAnalyzer(
            model=self.model,
            device=self.device,
            class_names=self.class_names
        )
        
        # Create test connectivity matrix
        connectivity_matrix = np.array([
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.6],
            [0.3, 0.6, 1.0]
        ])
        region_names = ['Region1', 'Region2', 'Region3']
        
        strongest = analyzer._find_strongest_connections(
            connectivity_matrix, region_names, top_k=2
        )
        
        self.assertEqual(len(strongest), 2)
        self.assertEqual(strongest[0]['strength'], 0.8)  # Strongest connection
        self.assertIn('region1', strongest[0])
        self.assertIn('region2', strongest[0])
    
    def test_feature_importance_calculation(self):
        """Test feature importance calculation from mock SHAP values."""
        analyzer = SpeckleInterpretabilityAnalyzer(
            model=self.model,
            device=self.device,
            class_names=self.class_names
        )
        
        # Create mock SHAP values
        mock_shap_values = [
            np.random.randn(5, 100),  # Class 0
            np.random.randn(5, 100),  # Class 1
            np.random.randn(5, 100),  # Class 2
        ]
        
        importance = analyzer._calculate_feature_importance(mock_shap_values)
        
        self.assertIsInstance(importance, dict)
        self.assertIn('overall', importance)
        for class_name in self.class_names:
            self.assertIn(class_name, importance)
        
        # Check shapes
        for class_name in self.class_names:
            self.assertEqual(len(importance[class_name]), 100)
    
    def test_class_specific_importance(self):
        """Test class-specific importance calculation."""
        analyzer = SpeckleInterpretabilityAnalyzer(
            model=self.model,
            device=self.device,
            class_names=self.class_names
        )
        
        # Create mock data
        mock_shap_values = [
            np.random.randn(6, 50),  # Class 0
            np.random.randn(6, 50),  # Class 1
            np.random.randn(6, 50),  # Class 2
        ]
        labels = np.array([0, 0, 1, 1, 2, 2])
        
        class_specific = analyzer._calculate_class_specific_importance(
            mock_shap_values, labels
        )
        
        self.assertIsInstance(class_specific, dict)
        # Should have data for each class
        for class_name in self.class_names:
            if class_name in class_specific:
                self.assertIn('correct_predictions', class_specific[class_name])
    
    def test_save_and_load_results(self):
        """Test saving and loading of results."""
        analyzer = SpeckleInterpretabilityAnalyzer(
            model=self.model,
            device=self.device,
            class_names=self.class_names
        )
        
        # Create mock results
        mock_results = {
            'feature_importance': {
                'overall': np.random.randn(100),
                'Circle': np.random.randn(100),
                'Rectangle': np.random.randn(100),
                'Triangle': np.random.randn(100)
            },
            'shap_values': np.random.randn(10, 100),
            'test_labels': np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0]),
            'test_samples': np.random.randn(10, 100),
            'background_samples': np.random.randn(5, 100)
        }
        
        # Test saving
        analyzer._save_shap_results(mock_results, self.temp_dir)
        
        # Check that files were created
        importance_file = os.path.join(self.temp_dir, 'feature_importance.csv')
        shap_file = os.path.join(self.temp_dir, 'shap_values.npy')
        
        self.assertTrue(os.path.exists(importance_file))
        self.assertTrue(os.path.exists(shap_file))
        
        # Test loading
        import pandas as pd
        loaded_importance = pd.read_csv(importance_file)
        loaded_shap = np.load(shap_file)
        
        self.assertEqual(loaded_importance.shape[0], 100)  # 100 features
        self.assertEqual(loaded_shap.shape, (10, 100))
    
    def test_model_forward_pass(self):
        """Test that the model forward pass works correctly."""
        # Test with different input shapes
        test_cases = [
            (8, 32, 1, 64),   # (batch, seq, channels, length)
            (4, 16, 1, 128),  # Different dimensions
        ]
        
        for batch_size, seq_len, channels, length in test_cases:
            with self.subTest(shape=(batch_size, seq_len, channels, length)):
                X = torch.randn(batch_size, seq_len, channels, length)
                
                with torch.no_grad():
                    output = self.model(X)
                
                self.assertEqual(output.shape, (batch_size, 3))  # 3 classes
                
                # Check that output is a valid probability distribution
                probs = torch.softmax(output, dim=1)
                self.assertTrue(torch.allclose(probs.sum(dim=1), torch.ones(batch_size)))


class TestInterpretabilityIntegration(unittest.TestCase):
    """Integration tests for interpretability analysis."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.device = torch.device('cpu')
        self.class_names = ['Circle', 'Rectangle', 'Triangle']
        
        # Create a more realistic model
        self.model = ConvLSTMClassifier(
            num_classes=3,
            hidden_size=16,  # Small for testing
            sequence_length=16,
            dropout_rate=0.0  # Disable dropout for consistent testing
        )
        self.model.eval()
        
        # Create test data that's more realistic
        self.X_test = torch.randn(12, 16, 1, 32)  # 12 samples, 16 sequence, 32 features
        self.y_test = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])  # Balanced classes
    
    def test_full_analysis_pipeline(self):
        """Test the complete analysis pipeline."""
        analyzer = SpeckleInterpretabilityAnalyzer(
            model=self.model,
            device=self.device,
            class_names=self.class_names
        )
        
        # Define minimal cortical regions for testing
        cortical_regions = {
            'V1_early': (0, 256),
            'V1_late': (256, 512),
        }
        
        # Run temporal analysis
        temporal_results = analyzer.analyze_temporal_patterns(
            X_test=self.X_test,
            y_test=self.y_test,
            sequence_length=16
        )
        
        # Run spatial analysis
        spatial_results = analyzer.analyze_spatial_cortical_mapping(
            X_test=self.X_test,
            y_test=self.y_test,
            cortical_regions=cortical_regions
        )
        
        # Check that both analyses completed
        self.assertIsInstance(temporal_results, dict)
        self.assertIsInstance(spatial_results, dict)
        
        # Check that results contain expected keys
        self.assertIn('temporal_patterns', temporal_results)
        self.assertIn('spatial_gradients', spatial_results)
        self.assertIn('shape_patterns', spatial_results)


def run_interpretability_tests():
    """Run all interpretability tests."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestInterpretabilityAnalyzer))
    test_suite.addTest(unittest.makeSuite(TestInterpretabilityIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("Running interpretability tests...")
    success = run_interpretability_tests()
    
    if success:
        print("\n✅ All interpretability tests passed!")
    else:
        print("\n❌ Some interpretability tests failed!")
        sys.exit(1)
