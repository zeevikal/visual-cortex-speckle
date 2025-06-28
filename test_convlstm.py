"""
Test script for ConvLSTM models to verify they work correctly.
"""

import torch
import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.models.base_model import ModelFactory
from src.models.convlstm import ConvLSTMClassifier, EnhancedConvLSTMClassifier


def test_convlstm_models():
    """Test ConvLSTM models with sample data."""
    print("Testing ConvLSTM models...")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model parameters
    num_classes = 2
    batch_size = 4
    sequence_length = 10
    input_dim = 1
    data_length = 128
    
    # Create sample input data
    # Shape: (batch_size, sequence_length, input_dim, data_length)
    sample_input = torch.randn(batch_size, sequence_length, input_dim, data_length)
    print(f"Sample input shape: {sample_input.shape}")
    
    # Test basic ConvLSTM model
    print("\n1. Testing ConvLSTMClassifier...")
    model1 = ConvLSTMClassifier(
        num_classes=num_classes,
        input_dim=input_dim,
        sequence_length=sequence_length,
        hidden_dims=[64, 32],
        kernel_sizes=[3, 3],
        num_layers=2,
        dropout=0.3
    )
    model1.to(device)
    sample_input1 = sample_input.to(device)
    
    with torch.no_grad():
        output1 = model1(sample_input1)
    
    print(f"Output shape: {output1.shape}")
    print(f"Output values: {output1}")
    print(f"Model info: {model1.get_model_info()}")
    
    # Test enhanced ConvLSTM model
    print("\n2. Testing EnhancedConvLSTMClassifier...")
    model2 = EnhancedConvLSTMClassifier(
        num_classes=num_classes,
        input_size=input_dim,
        hidden_size=64,
        sequence_length=sequence_length,
        kernel_size=3,
        dropout_rate=0.3,
        use_attention=True
    )
    model2.to(device)
    
    with torch.no_grad():
        output2 = model2(sample_input1)
    
    print(f"Output shape: {output2.shape}")
    print(f"Output values: {output2}")
    print(f"Model info: {model2.get_model_info()}")
    
    # Test model factory
    print("\n3. Testing ModelFactory...")
    print(f"Available models: {ModelFactory.list_models()}")
    
    # Create model via factory
    factory_model = ModelFactory.create_model(
        'convlstm',
        num_classes=num_classes,
        input_dim=input_dim,
        sequence_length=sequence_length
    )
    factory_model.to(device)
    
    with torch.no_grad():
        output3 = factory_model(sample_input1)
    
    print(f"Factory model output shape: {output3.shape}")
    
    # Test different input shapes
    print("\n4. Testing different input shapes...")
    
    # Single sequence input (batch_size, input_dim, data_length)
    single_seq_input = torch.randn(batch_size, input_dim, data_length).to(device)
    print(f"Single sequence input shape: {single_seq_input.shape}")
    
    with torch.no_grad():
        output4 = model1(single_seq_input)
    
    print(f"Single sequence output shape: {output4.shape}")
    
    # Test 2D input (batch_size, data_length)
    input_2d = torch.randn(batch_size, data_length).to(device)
    print(f"2D input shape: {input_2d.shape}")
    
    with torch.no_grad():
        output5 = model1(input_2d)
    
    print(f"2D input output shape: {output5.shape}")
    
    print("\n✅ All ConvLSTM tests passed!")


def test_convlstm_training():
    """Test a simple training step."""
    print("\nTesting ConvLSTM training...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = ConvLSTMClassifier(
        num_classes=2,
        input_dim=1,
        sequence_length=10
    )
    model.to(device)
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Create sample data
    batch_size = 8
    x = torch.randn(batch_size, 10, 1, 128).to(device)
    y = torch.randint(0, 2, (batch_size,)).to(device)
    
    # Forward pass
    model.train()
    optimizer.zero_grad()
    
    outputs = model(x)
    loss = criterion(outputs, y)
    
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    print("✅ Training step completed successfully!")


if __name__ == "__main__":
    print("ConvLSTM Model Testing")
    print("=" * 50)
    
    try:
        test_convlstm_models()
        test_convlstm_training()
        print("\n🎉 All tests passed successfully!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
