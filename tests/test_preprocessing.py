"""
Test data preprocessing functionality.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch
import tempfile
import os

# Note: These tests would require the actual dependencies to run
# They serve as examples of how to structure tests for the project

def test_normalize():
    """Test the normalize function."""
    from src.data.preprocessing import normalize
    
    # Create test array
    arr = np.array([1, 2, 3, 4, 5])
    normalized = normalize(arr)
    
    # Check that values are in 0-255 range
    assert normalized.min() == 0
    assert normalized.max() == 255
    assert len(normalized) == len(arr)


def test_compare_images():
    """Test image comparison function."""
    from src.data.preprocessing import compare_images
    
    # Create test images
    img1 = np.random.rand(100, 100)
    img2 = np.random.rand(100, 100)
    
    # Test comparison
    m_norm, z_norm, euclidean, manhattan, ncc = compare_images(img1, img2)
    
    # Check that all metrics are computed
    assert isinstance(m_norm, (int, float))
    assert isinstance(z_norm, (int, float))
    assert isinstance(euclidean, (int, float))
    assert isinstance(manhattan, (int, float))
    assert isinstance(ncc, (int, float))


@patch('cv2.VideoCapture')
def test_video_processor(mock_video_capture):
    """Test video processing functionality."""
    from src.data.preprocessing import VideoProcessor
    
    # Mock video capture
    mock_cap = Mock()
    mock_cap.read.return_value = (True, np.random.rand(100, 100, 3))
    mock_video_capture.return_value = mock_cap
    
    processor = VideoProcessor(frames_limit=10)
    
    # Test processing (would need actual video file in real test)
    with tempfile.NamedTemporaryFile(suffix='.avi') as tmp:
        result = processor.process_single_video(tmp.name)
        
        # Check result structure
        assert len(result) == 5  # 5 different metrics


def test_prepare_dataset_splits():
    """Test dataset splitting functionality."""
    from src.data.preprocessing import prepare_dataset_splits
    
    # Create mock speckle data
    speckle_data = {
        'subject1': [
            ('path1', 'Circle', [1, 2, 3, 4, 5], [], [], [], []),
            ('path2', 'Rectangle', [2, 3, 4, 5, 6], [], [], [], [])
        ]
    }
    
    x_train, y_train, x_test, y_test = prepare_dataset_splits(
        speckle_data,
        train_size=0.5,
        n_chunks=2,
        subjects=['subject1'],
        shape_filter=['Circle', 'Rectangle'],
        feature_type='manhattan'
    )
    
    # Check output shapes
    assert len(x_train) > 0
    assert len(y_train) > 0
    assert len(x_test) > 0
    assert len(y_test) > 0
    assert len(x_train) == len(y_train)
    assert len(x_test) == len(y_test)


if __name__ == "__main__":
    pytest.main([__file__])
