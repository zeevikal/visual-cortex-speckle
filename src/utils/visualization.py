"""
Visualization utilities for speckle imaging analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from typing import List, Optional, Tuple, Dict
import os


def set_style():
    """Set default plotting style."""
    plt.style.use('default')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['legend.fontsize'] = 10


def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    Plot training history with loss and accuracy curves.
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    set_style()
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    if 'val_loss' in history and history['val_loss']:
        axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    if 'val_acc' in history and history['val_acc']:
        axes[1].plot(history['val_acc'], label='Validation Accuracy', linewidth=2)
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Learning rate plot
    if 'lr' in history and history['lr']:
        axes[2].plot(history['lr'], linewidth=2, color='red')
        axes[2].set_title('Learning Rate')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].set_yscale('log')
        axes[2].grid(True, alpha=0.3)
    else:
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_sample_data(x_data: np.ndarray, 
                    y_data: np.ndarray,
                    class_names: Optional[List[str]] = None,
                    num_samples: int = 5,
                    save_path: Optional[str] = None,
                    figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Plot sample speckle patterns for each class.
    
    Args:
        x_data: Feature data
        y_data: Labels
        class_names: List of class names
        num_samples: Number of samples per class to plot
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    set_style()
    
    unique_classes = np.unique(y_data)
    n_classes = len(unique_classes)
    
    fig, axes = plt.subplots(n_classes, num_samples, figsize=figsize)
    if n_classes == 1:
        axes = axes.reshape(1, -1)
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i, class_label in enumerate(unique_classes):
        class_indices = np.where(y_data == class_label)[0]
        sample_indices = np.random.choice(class_indices, 
                                        min(num_samples, len(class_indices)), 
                                        replace=False)
        
        class_name = class_names[class_label] if class_names else f'Class {class_label}'
        
        for j, idx in enumerate(sample_indices):
            ax = axes[i, j] if n_classes > 1 else axes[j]
            
            # Plot the speckle pattern
            if x_data.ndim == 3:
                data = x_data[idx, :, 0]  # Take first channel
            else:
                data = x_data[idx]
            
            ax.plot(data, alpha=0.7, linewidth=1)
            ax.set_title(f'{class_name}' if j == 0 else '')
            ax.grid(True, alpha=0.3)
            
            if i == n_classes - 1:  # Last row
                ax.set_xlabel('Time Step')
            if j == 0:  # First column
                ax.set_ylabel('Amplitude')
    
    plt.suptitle('Sample Speckle Patterns by Class', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_maps(model: torch.nn.Module,
                     input_sample: torch.Tensor,
                     layer_idx: int = 1,
                     num_filters: int = 8,
                     save_path: Optional[str] = None,
                     figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Visualize feature maps from a specific layer.
    
    Args:
        model: Trained model
        input_sample: Input sample tensor
        layer_idx: Layer index to visualize
        num_filters: Number of filters to show
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    set_style()
    
    model.eval()
    
    # Get feature maps
    if hasattr(model, 'get_feature_maps'):
        feature_maps = model.get_feature_maps(input_sample.unsqueeze(0), layer_idx)
        feature_maps = feature_maps.squeeze(0).detach().cpu().numpy()
    else:
        raise ValueError("Model doesn't support feature map extraction")
    
    # Plot feature maps
    n_filters = min(num_filters, feature_maps.shape[0])
    n_cols = 4
    n_rows = (n_filters + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    for i in range(n_filters):
        ax = axes[i]
        ax.plot(feature_maps[i], linewidth=1)
        ax.set_title(f'Filter {i+1}')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_filters, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle(f'Feature Maps - Layer {layer_idx}', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_activation_analysis(model: torch.nn.Module,
                           input_samples: torch.Tensor,
                           true_labels: np.ndarray,
                           class_names: Optional[List[str]] = None,
                           threshold: float = 0.8,
                           save_path: Optional[str] = None,
                           figsize: Tuple[int, int] = (15, 8)) -> plt.Figure:
    """
    Analyze and plot activations with threshold filtering.
    
    Args:
        model: Trained model
        input_samples: Input samples
        true_labels: True labels for samples
        class_names: List of class names
        threshold: Threshold for activation filtering
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    set_style()
    
    model.eval()
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    with torch.no_grad():
        for i, (sample, label) in enumerate(zip(input_samples[:4], true_labels[:4])):
            # Get feature maps
            if hasattr(model, 'get_feature_maps'):
                feature_maps = model.get_feature_maps(sample.unsqueeze(0), layer_idx=1)
                feature_maps = feature_maps.squeeze(0).cpu().numpy()
            else:
                continue
            
            ax = axes[i]
            
            # Plot original signal
            original_signal = sample.cpu().numpy()
            if original_signal.ndim > 1:
                original_signal = original_signal[:, 0]  # Take first channel
            
            # Normalize for plotting
            original_signal = original_signal / original_signal.max()
            ax.plot(original_signal, alpha=0.7, label='Original', linewidth=2)
            
            # Plot filtered feature maps
            for j, feature_map in enumerate(feature_maps[:5]):  # Show first 5 filters
                # Apply threshold filtering
                filtered_map = feature_map.copy()
                filtered_map = np.maximum(filtered_map, 0)  # ReLU
                filtered_map = filtered_map / filtered_map.max() if filtered_map.max() > 0 else filtered_map
                filtered_map[filtered_map < threshold] = 0
                
                # Interpolate to match original signal length
                if len(filtered_map) != len(original_signal):
                    x_old = np.linspace(0, 1, len(filtered_map))
                    x_new = np.linspace(0, 1, len(original_signal))
                    filtered_map = np.interp(x_new, x_old, filtered_map)
                
                ax.plot(filtered_map, alpha=0.6, linewidth=1)
            
            class_name = class_names[label] if class_names else f'Class {label}'
            ax.set_title(f'{class_name} - Sample {i+1}')
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Normalized Amplitude')
            
            if i == 0:
                ax.legend()
    
    plt.suptitle(f'Activation Analysis (Threshold: {threshold})', fontsize=16)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_data_distribution(y_data: np.ndarray,
                          class_names: Optional[List[str]] = None,
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
    """
    Plot class distribution in the dataset.
    
    Args:
        y_data: Label data
        class_names: List of class names
        save_path: Path to save the plot
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    set_style()
    
    unique_labels, counts = np.unique(y_data, return_counts=True)
    
    if class_names:
        labels = [class_names[i] for i in unique_labels]
    else:
        labels = [f'Class {i}' for i in unique_labels]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Bar plot
    bars = ax1.bar(labels, counts, alpha=0.7)
    ax1.set_title('Class Distribution')
    ax1.set_xlabel('Classes')
    ax1.set_ylabel('Number of Samples')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add count labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}', ha='center', va='bottom')
    
    # Pie chart
    ax2.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Class Distribution (Percentage)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_visualization_report(model: torch.nn.Module,
                              x_train: np.ndarray,
                              y_train: np.ndarray,
                              x_test: np.ndarray,
                              y_test: np.ndarray,
                              history: Dict[str, List[float]],
                              class_names: Optional[List[str]] = None,
                              save_dir: str = 'visualizations'):
    """
    Create a comprehensive visualization report.
    
    Args:
        model: Trained model
        x_train: Training features
        y_train: Training labels
        x_test: Test features
        y_test: Test labels
        history: Training history
        class_names: List of class names
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Training history
    plot_training_history(history, 
                         save_path=os.path.join(save_dir, 'training_history.png'))
    plt.close()
    
    # Data distribution
    plot_data_distribution(y_train, class_names,
                          save_path=os.path.join(save_dir, 'train_distribution.png'))
    plt.close()
    
    plot_data_distribution(y_test, class_names,
                          save_path=os.path.join(save_dir, 'test_distribution.png'))
    plt.close()
    
    # Sample data
    plot_sample_data(x_train, y_train, class_names,
                    save_path=os.path.join(save_dir, 'sample_patterns_train.png'))
    plt.close()
    
    plot_sample_data(x_test, y_test, class_names,
                    save_path=os.path.join(save_dir, 'sample_patterns_test.png'))
    plt.close()
    
    # Feature maps (if supported)
    if hasattr(model, 'get_feature_maps') and len(x_test) > 0:
        sample_input = torch.FloatTensor(x_test[0])
        plot_feature_maps(model, sample_input,
                         save_path=os.path.join(save_dir, 'feature_maps.png'))
        plt.close()
        
        # Activation analysis
        sample_inputs = torch.FloatTensor(x_test[:4])
        plot_activation_analysis(model, sample_inputs, y_test[:4], class_names,
                                save_path=os.path.join(save_dir, 'activation_analysis.png'))
        plt.close()
    
    print(f"Visualization report saved to {save_dir}")
