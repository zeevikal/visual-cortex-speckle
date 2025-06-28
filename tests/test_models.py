"""
Test model architectures.
"""

import torch
import numpy as np
import pytest


def test_conv1d_classifier():
    """Test Conv1D classifier model."""
    from src.models.conv1d import Conv1DClassifier
    
    # Create model
    model = Conv1DClassifier(num_classes=3, input_channels=1)
    
    # Test forward pass
    batch_size = 4
    sequence_length = 100
    x = torch.randn(batch_size, sequence_length, 1)
    
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, 3)
    
    # Test with different input shapes
    x_2d = torch.randn(batch_size, sequence_length)
    output_2d = model(x_2d)
    assert output_2d.shape == (batch_size, 3)


def test_model_info():
    """Test model information retrieval."""
    from src.models.conv1d import Conv1DClassifier
    
    model = Conv1DClassifier(num_classes=5, num_filters=32)
    info = model.get_model_info()
    
    # Check info structure
    assert 'model_name' in info
    assert 'num_classes' in info
    assert 'total_parameters' in info
    assert 'trainable_parameters' in info
    assert 'model_size_mb' in info
    
    assert info['num_classes'] == 5


def test_model_save_load():
    """Test model save and load functionality."""
    from src.models.conv1d import Conv1DClassifier
    import tempfile
    
    # Create and train a simple model
    model = Conv1DClassifier(num_classes=3)
    
    with tempfile.NamedTemporaryFile(suffix='.pth') as tmp:
        # Save model
        model.save_model(tmp.name, {'test_info': 'test_value'})
        
        # Load model
        loaded_model, info = Conv1DClassifier.load_model(tmp.name)
        
        # Check that models have same architecture
        assert loaded_model.num_classes == model.num_classes
        assert 'test_info' in info


def test_model_factory():
    """Test model factory functionality."""
    from src.models.base_model import ModelFactory
    
    # Test model creation
    model = ModelFactory.create_model('conv1d', num_classes=4)
    assert model.num_classes == 4
    
    # Test listing models
    available_models = ModelFactory.list_models()
    assert 'conv1d' in available_models


def test_feature_maps():
    """Test feature map extraction."""
    from src.models.conv1d import Conv1DClassifier
    
    model = Conv1DClassifier(num_classes=3)
    model.eval()
    
    # Create input
    x = torch.randn(1, 100, 1)
    
    # Get feature maps
    feature_maps = model.get_feature_maps(x, layer_idx=1)
    
    # Check output
    assert feature_maps.dim() == 3  # (batch, channels, sequence)
    assert feature_maps.size(0) == 1  # batch size
    assert feature_maps.size(1) == 64  # num filters


if __name__ == "__main__":
    pytest.main([__file__])
