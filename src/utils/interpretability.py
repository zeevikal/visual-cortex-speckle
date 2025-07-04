"""
Model interpretability and explainability tools for visual cortex speckle imaging.

This module provides various interpretability methods to understand:
1. Which speckle features are most important for shape classification
2. How different temporal patterns contribute to predictions
3. Spatial mapping of important features to potential cortical regions
4. Visualization of model decision boundaries and feature importance

Key Methods:
- SHAP (SHapley Additive exPlanations) for feature importance
- Gradient-based attribution methods (Integrated Gradients, GradCAM)
- Temporal attention analysis for ConvLSTM models
- Spatial mapping and cortical region analysis
- Feature similarity analysis between shapes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import pandas as pd
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Optional imports for advanced interpretability
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Install with: pip install shap")

try:
    from captum.attr import (
        IntegratedGradients, 
        GradientShap, 
        DeepLift, 
        LayerConductance,
        LayerGradientXActivation,
        LayerGradCam,
        Occlusion
    )
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False
    print("Captum not available. Install with: pip install captum")

from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


class SpeckleInterpretabilityAnalyzer:
    """
    Comprehensive interpretability analysis for speckle imaging models.
    
    This class provides multiple methods to understand model decisions:
    - SHAP analysis for feature importance
    - Gradient-based attribution methods
    - Temporal pattern analysis
    - Spatial cortical mapping
    - Feature similarity and clustering
    """
    
    def __init__(self, 
                 model: nn.Module,
                 device: Optional[torch.device] = None,
                 class_names: Optional[List[str]] = None):
        """
        Initialize interpretability analyzer.
        
        Args:
            model: Trained model to analyze
            device: Device to run analysis on
            class_names: List of class names for visualization
        """
        self.model = model.eval()
        self.device = device or torch.device('cpu')
        self.class_names = class_names or [f'Class_{i}' for i in range(self._get_num_classes())]
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize attribution methods if available
        self._init_attribution_methods()
        
        # Storage for analysis results
        self.results = {}
        
    def _get_num_classes(self) -> int:
        """Get number of classes from model."""
        try:
            # Try to get from model's classifier layer
            if hasattr(self.model, 'classifier'):
                if hasattr(self.model.classifier, 'out_features'):
                    return self.model.classifier.out_features
                elif hasattr(self.model.classifier, 'linear'):
                    return self.model.classifier.linear.out_features
                elif isinstance(self.model.classifier, nn.Sequential):
                    for layer in reversed(self.model.classifier):
                        if hasattr(layer, 'out_features'):
                            return layer.out_features
            
            # Try to get from a dummy forward pass
            dummy_input = torch.randn(1, 64, 1, 128).to(self.device)  # Common input shape
            with torch.no_grad():
                output = self.model(dummy_input)
                return output.shape[-1]
        except:
            return 3  # Default for this project
    
    def _init_attribution_methods(self):
        """Initialize attribution methods if libraries are available."""
        self.attribution_methods = {}
        
        if CAPTUM_AVAILABLE:
            self.attribution_methods['integrated_gradients'] = IntegratedGradients(self.model)
            self.attribution_methods['gradient_shap'] = GradientShap(self.model)
            self.attribution_methods['deep_lift'] = DeepLift(self.model)
            self.attribution_methods['occlusion'] = Occlusion(self.model)
        
        if SHAP_AVAILABLE:
            self.shap_explainer = None  # Will be initialized when needed
    
    def analyze_feature_importance_shap(self, 
                                      X_test: torch.Tensor,
                                      y_test: torch.Tensor,
                                      background_size: int = 100,
                                      test_size: int = 50,
                                      save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze feature importance using SHAP.
        
        Args:
            X_test: Test data
            y_test: Test labels
            background_size: Size of background dataset for SHAP
            test_size: Number of test samples to analyze
            save_dir: Directory to save results
            
        Returns:
            Dictionary containing SHAP analysis results
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")
        
        print("Performing SHAP analysis...")
        
        # Prepare data
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)
        
        # Select random samples for analysis
        indices = np.random.choice(len(X_test), min(test_size, len(X_test)), replace=False)
        X_sample = X_test[indices]
        y_sample = y_test[indices]
        
        # Select background samples
        bg_indices = np.random.choice(len(X_test), min(background_size, len(X_test)), replace=False)
        X_background = X_test[bg_indices]
        
        # Reshape for SHAP if needed
        if X_sample.dim() == 4:  # (batch, seq, channels, length)
            X_sample_2d = X_sample.view(X_sample.size(0), -1)
            X_background_2d = X_background.view(X_background.size(0), -1)
        else:
            X_sample_2d = X_sample
            X_background_2d = X_background
        
        # Create SHAP explainer
        self.shap_explainer = shap.DeepExplainer(self.model, X_background_2d)
        
        # Calculate SHAP values
        shap_values = self.shap_explainer.shap_values(X_sample_2d, check_additivity=False)
        
        # Convert to numpy if needed
        if isinstance(shap_values, list):
            shap_values_np = [sv.cpu().numpy() if torch.is_tensor(sv) else sv for sv in shap_values]
        else:
            shap_values_np = shap_values.cpu().numpy() if torch.is_tensor(shap_values) else shap_values
        
        # Analysis results
        results = {
            'shap_values': shap_values_np,
            'test_samples': X_sample_2d.cpu().numpy(),
            'test_labels': y_sample.cpu().numpy(),
            'background_samples': X_background_2d.cpu().numpy(),
            'feature_importance': self._calculate_feature_importance(shap_values_np),
            'class_specific_importance': self._calculate_class_specific_importance(shap_values_np, y_sample.cpu().numpy())
        }
        
        # Save results
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self._save_shap_results(results, save_dir)
        
        self.results['shap'] = results
        return results
    
    def _calculate_feature_importance(self, shap_values: Union[np.ndarray, List[np.ndarray]]) -> Dict[str, np.ndarray]:
        """Calculate global feature importance from SHAP values."""
        if isinstance(shap_values, list):
            # Multi-class case
            global_importance = {}
            for class_idx, class_shap in enumerate(shap_values):
                class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f'Class_{class_idx}'
                global_importance[class_name] = np.mean(np.abs(class_shap), axis=0)
            
            # Overall importance (mean across classes)
            overall_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            global_importance['overall'] = overall_importance
        else:
            # Binary case
            global_importance = {'overall': np.mean(np.abs(shap_values), axis=0)}
        
        return global_importance
    
    def _calculate_class_specific_importance(self, shap_values: Union[np.ndarray, List[np.ndarray]], 
                                           labels: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        """Calculate class-specific feature importance."""
        class_specific = {}
        
        if isinstance(shap_values, list):
            # Multi-class case
            for class_idx, class_shap in enumerate(shap_values):
                class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f'Class_{class_idx}'
                class_specific[class_name] = {}
                
                # Importance for samples of this class
                class_mask = labels == class_idx
                if np.any(class_mask):
                    class_specific[class_name]['correct_predictions'] = np.mean(np.abs(class_shap[class_mask]), axis=0)
                
                # Importance for samples of other classes
                other_mask = labels != class_idx
                if np.any(other_mask):
                    class_specific[class_name]['other_predictions'] = np.mean(np.abs(class_shap[other_mask]), axis=0)
        else:
            # Binary case
            for class_idx in np.unique(labels):
                class_name = self.class_names[class_idx] if class_idx < len(self.class_names) else f'Class_{class_idx}'
                class_mask = labels == class_idx
                if np.any(class_mask):
                    class_specific[class_name] = {
                        'importance': np.mean(np.abs(shap_values[class_mask]), axis=0)
                    }
        
        return class_specific
    
    def analyze_temporal_patterns(self, 
                                X_test: torch.Tensor,
                                y_test: torch.Tensor,
                                sequence_length: int = 64,
                                save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze temporal patterns in speckle sequences.
        
        Args:
            X_test: Test data
            y_test: Test labels
            sequence_length: Length of temporal sequences
            save_dir: Directory to save results
            
        Returns:
            Dictionary containing temporal analysis results
        """
        print("Analyzing temporal patterns...")
        
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)
        
        # Ensure correct input shape for temporal analysis
        if X_test.dim() == 4:  # (batch, seq, channels, length)
            batch_size, seq_len, channels, length = X_test.shape
            temporal_data = X_test
        elif X_test.dim() == 3:  # (batch, length, channels)
            batch_size, length, channels = X_test.shape
            # Reshape to temporal sequences
            seq_len = min(sequence_length, length)
            temporal_data = X_test[:, :seq_len].unsqueeze(2)  # Add sequence dimension
        else:
            raise ValueError(f"Unsupported input shape: {X_test.shape}")
        
        results = {}
        
        # 1. Temporal attention analysis (if model supports it)
        if hasattr(self.model, 'get_attention_weights'):
            attention_weights = self._analyze_attention_weights(temporal_data, y_test)
            results['attention_weights'] = attention_weights
        
        # 2. Frame-by-frame importance using occlusion
        if CAPTUM_AVAILABLE:
            frame_importance = self._analyze_frame_importance(temporal_data, y_test)
            results['frame_importance'] = frame_importance
        
        # 3. Temporal gradient analysis
        temporal_gradients = self._analyze_temporal_gradients(temporal_data, y_test)
        results['temporal_gradients'] = temporal_gradients
        
        # 4. Temporal pattern clustering
        temporal_patterns = self._analyze_temporal_clusters(temporal_data, y_test)
        results['temporal_patterns'] = temporal_patterns
        
        # Save results
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self._save_temporal_results(results, save_dir)
        
        self.results['temporal'] = results
        return results
    
    def _analyze_attention_weights(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        """Analyze attention weights if model supports it."""
        try:
            attention_weights = []
            for i in range(min(50, len(X))):  # Analyze first 50 samples
                with torch.no_grad():
                    weights = self.model.get_attention_weights(X[i:i+1])
                    attention_weights.append(weights.cpu().numpy())
            
            return {
                'weights': np.concatenate(attention_weights, axis=0),
                'mean_attention': np.mean(attention_weights, axis=0),
                'std_attention': np.std(attention_weights, axis=0)
            }
        except:
            return {'message': 'Model does not support attention weight extraction'}
    
    def _analyze_frame_importance(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        """Analyze frame-by-frame importance using occlusion."""
        if not CAPTUM_AVAILABLE:
            return {'message': 'Captum not available'}
        
        # Use occlusion to test importance of different time frames
        occlusion = Occlusion(self.model)
        
        # Select a few samples for analysis
        sample_indices = np.random.choice(len(X), min(10, len(X)), replace=False)
        X_sample = X[sample_indices]
        y_sample = y[sample_indices]
        
        frame_attributions = []
        for i in range(len(X_sample)):
            # Occlude different time frames
            if X_sample.dim() == 4:  # (batch, seq, channels, length)
                occlusion_shape = (1, 1, 1, X_sample.size(3) // 4)  # Occlude 1/4 of the sequence
            else:
                occlusion_shape = (1, X_sample.size(1) // 4)  # Occlude 1/4 of the sequence
            
            try:
                attribution = occlusion.attribute(
                    X_sample[i:i+1], 
                    target=y_sample[i:i+1],
                    sliding_window_shapes=occlusion_shape,
                    strides=occlusion_shape
                )
                frame_attributions.append(attribution.cpu().numpy())
            except Exception as e:
                print(f"Error in occlusion analysis: {e}")
                continue
        
        if frame_attributions:
            return {
                'attributions': np.concatenate(frame_attributions, axis=0),
                'mean_attribution': np.mean(frame_attributions, axis=0),
                'std_attribution': np.std(frame_attributions, axis=0)
            }
        else:
            return {'message': 'Frame importance analysis failed'}
    
    def _analyze_temporal_gradients(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        """Analyze temporal gradients to understand time-dependent features."""
        X.requires_grad_(True)
        
        temporal_gradients = []
        for class_idx in range(len(self.class_names)):
            class_mask = y == class_idx
            if not torch.any(class_mask):
                continue
            
            X_class = X[class_mask]
            batch_grads = []
            
            for i in range(min(20, len(X_class))):  # Analyze first 20 samples per class
                self.model.zero_grad()
                
                output = self.model(X_class[i:i+1])
                loss = output[0, class_idx]  # Focus on target class
                loss.backward(retain_graph=True)
                
                if X_class.grad is not None:
                    grad = X_class.grad[i].cpu().numpy()
                    batch_grads.append(grad)
                    X_class.grad.zero_()
            
            if batch_grads:
                temporal_gradients.append({
                    'class': self.class_names[class_idx],
                    'gradients': np.stack(batch_grads),
                    'mean_gradient': np.mean(batch_grads, axis=0),
                    'std_gradient': np.std(batch_grads, axis=0)
                })
        
        return temporal_gradients
    
    def _analyze_temporal_clusters(self, X: torch.Tensor, y: torch.Tensor) -> Dict[str, Any]:
        """Analyze temporal patterns using clustering."""
        # Extract temporal features
        temporal_features = []
        labels = []
        
        for class_idx in range(len(self.class_names)):
            class_mask = y == class_idx
            if not torch.any(class_mask):
                continue
            
            X_class = X[class_mask]
            
            # Extract temporal statistics
            for i in range(min(100, len(X_class))):  # Analyze first 100 samples per class
                sample = X_class[i]
                
                if sample.dim() == 3:  # (seq, channels, length)
                    # Temporal statistics across the sequence
                    mean_val = torch.mean(sample, dim=0).flatten()
                    std_val = torch.std(sample, dim=0).flatten()
                    max_val = torch.max(sample, dim=0)[0].flatten()
                    min_val = torch.min(sample, dim=0)[0].flatten()
                    
                    features = torch.cat([mean_val, std_val, max_val, min_val]).cpu().numpy()
                else:
                    features = sample.flatten().cpu().numpy()
                
                temporal_features.append(features)
                labels.append(class_idx)
        
        temporal_features = np.array(temporal_features)
        labels = np.array(labels)
        
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=min(50, temporal_features.shape[1]))
        features_pca = pca.fit_transform(temporal_features)
        
        # Apply t-SNE for visualization
        tsne = TSNE(n_components=2, random_state=42)
        features_tsne = tsne.fit_transform(features_pca)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=len(self.class_names), random_state=42)
        cluster_labels = kmeans.fit_predict(features_pca)
        
        return {
            'features': temporal_features,
            'labels': labels,
            'pca_features': features_pca,
            'tsne_features': features_tsne,
            'cluster_labels': cluster_labels,
            'pca_explained_variance': pca.explained_variance_ratio_,
            'cluster_centers': kmeans.cluster_centers_
        }
    
    def analyze_spatial_cortical_mapping(self, 
                                       X_test: torch.Tensor,
                                       y_test: torch.Tensor,
                                       cortical_regions: Optional[Dict[str, Tuple[int, int]]] = None,
                                       save_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyze spatial mapping of important features to potential cortical regions.
        
        This method attempts to map important speckle features to potential
        regions of the visual cortex (V1 subareas) based on spatial patterns.
        
        Args:
            X_test: Test data
            y_test: Test labels
            cortical_regions: Dictionary mapping region names to (start, end) indices
            save_dir: Directory to save results
            
        Returns:
            Dictionary containing spatial mapping results
        """
        print("Analyzing spatial cortical mapping...")
        
        X_test = X_test.to(self.device)
        y_test = y_test.to(self.device)
        
        # Default cortical regions (hypothetical mapping)
        if cortical_regions is None:
            # Assume 64x64 speckle patterns -> 4096 features
            # Map to different V1 subareas based on feature spatial organization
            cortical_regions = {
                'V1_edge_detectors': (0, 1024),      # Early edge detection
                'V1_orientation_columns': (1024, 2048),  # Orientation processing
                'V1_spatial_frequency': (2048, 3072),    # Spatial frequency analysis
                'V1_higher_order': (3072, 4096),         # Higher-order processing
            }
        
        results = {}
        
        # 1. Feature importance by cortical region
        if 'shap' in self.results:
            region_importance = self._analyze_region_importance(cortical_regions)
            results['region_importance'] = region_importance
        
        # 2. Spatial gradient analysis
        spatial_gradients = self._analyze_spatial_gradients(X_test, y_test, cortical_regions)
        results['spatial_gradients'] = spatial_gradients
        
        # 3. Shape-specific spatial patterns
        shape_patterns = self._analyze_shape_spatial_patterns(X_test, y_test, cortical_regions)
        results['shape_patterns'] = shape_patterns
        
        # 4. Cortical region connectivity analysis
        connectivity = self._analyze_cortical_connectivity(X_test, y_test, cortical_regions)
        results['connectivity'] = connectivity
        
        # Save results
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            self._save_spatial_results(results, save_dir, cortical_regions)
        
        self.results['spatial'] = results
        return results
    
    def _analyze_region_importance(self, cortical_regions: Dict[str, Tuple[int, int]]) -> Dict[str, Any]:
        """Analyze feature importance by cortical region."""
        if 'shap' not in self.results:
            return {'message': 'SHAP analysis required first'}
        
        shap_results = self.results['shap']
        feature_importance = shap_results['feature_importance']
        
        region_analysis = {}
        
        for region_name, (start, end) in cortical_regions.items():
            region_analysis[region_name] = {}
            
            # Analyze importance for each class
            for class_name, importance in feature_importance.items():
                if class_name == 'overall':
                    continue
                
                # Extract region-specific importance
                region_importance = importance[start:end]
                
                region_analysis[region_name][class_name] = {
                    'mean_importance': np.mean(region_importance),
                    'max_importance': np.max(region_importance),
                    'std_importance': np.std(region_importance),
                    'importance_profile': region_importance
                }
            
            # Overall region importance
            overall_importance = feature_importance['overall'][start:end]
            region_analysis[region_name]['overall'] = {
                'mean_importance': np.mean(overall_importance),
                'max_importance': np.max(overall_importance),
                'std_importance': np.std(overall_importance),
                'importance_profile': overall_importance
            }
        
        return region_analysis
    
    def _analyze_spatial_gradients(self, X: torch.Tensor, y: torch.Tensor, 
                                 cortical_regions: Dict[str, Tuple[int, int]]) -> Dict[str, Any]:
        """Analyze spatial gradients for each cortical region."""
        X.requires_grad_(True)
        
        spatial_gradients = {}
        
        for region_name, (start, end) in cortical_regions.items():
            region_gradients = []
            
            for class_idx in range(len(self.class_names)):
                class_mask = y == class_idx
                if not torch.any(class_mask):
                    continue
                
                X_class = X[class_mask]
                class_gradients = []
                
                for i in range(min(10, len(X_class))):  # Analyze first 10 samples
                    self.model.zero_grad()
                    
                    # Reshape for gradient calculation
                    if X_class.dim() == 4:  # (batch, seq, channels, length)
                        sample = X_class[i:i+1].view(1, -1)
                    else:
                        sample = X_class[i:i+1]
                    
                    output = self.model(X_class[i:i+1])
                    loss = output[0, class_idx]
                    loss.backward(retain_graph=True)
                    
                    if sample.grad is not None:
                        # Extract gradients for this region
                        region_grad = sample.grad[0, start:end].cpu().numpy()
                        class_gradients.append(region_grad)
                        sample.grad.zero_()
                
                if class_gradients:
                    region_gradients.append({
                        'class': self.class_names[class_idx],
                        'gradients': np.stack(class_gradients),
                        'mean_gradient': np.mean(class_gradients, axis=0),
                        'std_gradient': np.std(class_gradients, axis=0)
                    })
            
            spatial_gradients[region_name] = region_gradients
        
        return spatial_gradients
    
    def _analyze_shape_spatial_patterns(self, X: torch.Tensor, y: torch.Tensor,
                                      cortical_regions: Dict[str, Tuple[int, int]]) -> Dict[str, Any]:
        """Analyze shape-specific spatial patterns in each cortical region."""
        shape_patterns = {}
        
        for region_name, (start, end) in cortical_regions.items():
            region_patterns = {}
            
            for class_idx in range(len(self.class_names)):
                class_name = self.class_names[class_idx]
                class_mask = y == class_idx
                
                if not torch.any(class_mask):
                    continue
                
                X_class = X[class_mask]
                
                # Extract regional patterns
                regional_patterns = []
                for i in range(min(50, len(X_class))):  # Analyze first 50 samples
                    if X_class.dim() == 4:  # (batch, seq, channels, length)
                        sample = X_class[i].view(-1)  # Flatten
                    else:
                        sample = X_class[i].flatten()
                    
                    region_pattern = sample[start:end].cpu().numpy()
                    regional_patterns.append(region_pattern)
                
                if regional_patterns:
                    patterns_array = np.stack(regional_patterns)
                    
                    region_patterns[class_name] = {
                        'patterns': patterns_array,
                        'mean_pattern': np.mean(patterns_array, axis=0),
                        'std_pattern': np.std(patterns_array, axis=0),
                        'pattern_variability': np.std(patterns_array, axis=0).mean(),
                        'max_activation': np.max(patterns_array, axis=0),
                        'min_activation': np.min(patterns_array, axis=0)
                    }
            
            shape_patterns[region_name] = region_patterns
        
        return shape_patterns
    
    def _analyze_cortical_connectivity(self, X: torch.Tensor, y: torch.Tensor,
                                     cortical_regions: Dict[str, Tuple[int, int]]) -> Dict[str, Any]:
        """Analyze connectivity between cortical regions."""
        connectivity = {}
        
        # Calculate cross-correlations between regions
        for shape_idx in range(len(self.class_names)):
            shape_name = self.class_names[shape_idx]
            class_mask = y == shape_idx
            
            if not torch.any(class_mask):
                continue
            
            X_shape = X[class_mask]
            
            # Calculate connectivity matrix for this shape
            region_names = list(cortical_regions.keys())
            connectivity_matrix = np.zeros((len(region_names), len(region_names)))
            
            for i, region1 in enumerate(region_names):
                start1, end1 = cortical_regions[region1]
                
                for j, region2 in enumerate(region_names):
                    start2, end2 = cortical_regions[region2]
                    
                    if i == j:
                        connectivity_matrix[i, j] = 1.0
                        continue
                    
                    # Calculate correlation between regions
                    correlations = []
                    for sample_idx in range(min(30, len(X_shape))):
                        if X_shape.dim() == 4:
                            sample = X_shape[sample_idx].view(-1)
                        else:
                            sample = X_shape[sample_idx].flatten()
                        
                        pattern1 = sample[start1:end1].cpu().numpy()
                        pattern2 = sample[start2:end2].cpu().numpy()
                        
                        # Calculate correlation
                        if len(pattern1) > 1 and len(pattern2) > 1:
                            corr, _ = pearsonr(pattern1, pattern2)
                            if not np.isnan(corr):
                                correlations.append(corr)
                    
                    if correlations:
                        connectivity_matrix[i, j] = np.mean(correlations)
            
            connectivity[shape_name] = {
                'connectivity_matrix': connectivity_matrix,
                'region_names': region_names,
                'mean_connectivity': np.mean(connectivity_matrix[connectivity_matrix != 1.0]),
                'max_connectivity': np.max(connectivity_matrix[connectivity_matrix != 1.0]),
                'strongest_connections': self._find_strongest_connections(connectivity_matrix, region_names)
            }
        
        return connectivity
    
    def _find_strongest_connections(self, connectivity_matrix: np.ndarray, 
                                  region_names: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """Find the strongest connections between regions."""
        connections = []
        
        for i in range(len(region_names)):
            for j in range(i+1, len(region_names)):
                connections.append({
                    'region1': region_names[i],
                    'region2': region_names[j],
                    'strength': connectivity_matrix[i, j]
                })
        
        # Sort by strength and return top connections
        connections.sort(key=lambda x: abs(x['strength']), reverse=True)
        return connections[:top_k]
    
    def create_interpretability_report(self, save_dir: str) -> None:
        """
        Create a comprehensive interpretability report.
        
        Args:
            save_dir: Directory to save the report
        """
        print("Creating comprehensive interpretability report...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Create visualizations
        self._create_feature_importance_plots(save_dir)
        self._create_temporal_analysis_plots(save_dir)
        self._create_spatial_mapping_plots(save_dir)
        self._create_connectivity_plots(save_dir)
        
        # Generate summary report
        self._generate_summary_report(save_dir)
    
    def _create_feature_importance_plots(self, save_dir: str) -> None:
        """Create feature importance visualization plots."""
        if 'shap' not in self.results:
            return
        
        shap_results = self.results['shap']
        
        # Plot 1: Overall feature importance
        plt.figure(figsize=(15, 8))
        
        if 'overall' in shap_results['feature_importance']:
            importance = shap_results['feature_importance']['overall']
            plt.subplot(2, 2, 1)
            plt.plot(importance, alpha=0.8)
            plt.title('Overall Feature Importance')
            plt.xlabel('Feature Index')
            plt.ylabel('SHAP Value')
            plt.grid(True, alpha=0.3)
        
        # Plot 2: Class-specific importance
        subplot_idx = 2
        for class_name, importance in shap_results['feature_importance'].items():
            if class_name == 'overall' or subplot_idx > 4:
                continue
            
            plt.subplot(2, 2, subplot_idx)
            plt.plot(importance, alpha=0.8, label=class_name)
            plt.title(f'{class_name} Feature Importance')
            plt.xlabel('Feature Index')
            plt.ylabel('SHAP Value')
            plt.grid(True, alpha=0.3)
            subplot_idx += 1
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_temporal_analysis_plots(self, save_dir: str) -> None:
        """Create temporal analysis visualization plots."""
        if 'temporal' not in self.results:
            return
        
        temporal_results = self.results['temporal']
        
        # Plot temporal patterns
        if 'temporal_patterns' in temporal_results:
            patterns = temporal_results['temporal_patterns']
            
            plt.figure(figsize=(15, 10))
            
            # t-SNE visualization
            plt.subplot(2, 2, 1)
            tsne_features = patterns['tsne_features']
            labels = patterns['labels']
            
            for class_idx in range(len(self.class_names)):
                mask = labels == class_idx
                if np.any(mask):
                    plt.scatter(tsne_features[mask, 0], tsne_features[mask, 1], 
                              label=self.class_names[class_idx], alpha=0.7)
            
            plt.title('Temporal Pattern Clustering (t-SNE)')
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # PCA explained variance
            plt.subplot(2, 2, 2)
            explained_var = patterns['pca_explained_variance']
            plt.bar(range(len(explained_var)), explained_var)
            plt.title('PCA Explained Variance')
            plt.xlabel('Component')
            plt.ylabel('Explained Variance Ratio')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'temporal_analysis.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_spatial_mapping_plots(self, save_dir: str) -> None:
        """Create spatial cortical mapping visualization plots."""
        if 'spatial' not in self.results:
            return
        
        spatial_results = self.results['spatial']
        
        # Plot cortical region importance
        if 'region_importance' in spatial_results:
            region_data = spatial_results['region_importance']
            
            plt.figure(figsize=(15, 10))
            
            # Create heatmap of region importance by class
            regions = list(region_data.keys())
            classes = [name for name in self.class_names]
            
            importance_matrix = np.zeros((len(regions), len(classes)))
            
            for i, region in enumerate(regions):
                for j, class_name in enumerate(classes):
                    if class_name in region_data[region]:
                        importance_matrix[i, j] = region_data[region][class_name]['mean_importance']
            
            sns.heatmap(importance_matrix, 
                       xticklabels=classes, 
                       yticklabels=regions,
                       annot=True, 
                       cmap='viridis',
                       cbar_kws={'label': 'Mean Importance'})
            
            plt.title('Cortical Region Importance by Shape Class')
            plt.xlabel('Shape Class')
            plt.ylabel('Cortical Region')
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, 'cortical_mapping.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_connectivity_plots(self, save_dir: str) -> None:
        """Create cortical connectivity visualization plots."""
        if 'spatial' not in self.results or 'connectivity' not in self.results['spatial']:
            return
        
        connectivity_results = self.results['spatial']['connectivity']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        plot_idx = 0
        for shape_name, connectivity_data in connectivity_results.items():
            if plot_idx >= 4:
                break
            
            ax = axes[plot_idx]
            
            # Plot connectivity matrix
            conn_matrix = connectivity_data['connectivity_matrix']
            region_names = connectivity_data['region_names']
            
            im = ax.imshow(conn_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
            ax.set_xticks(range(len(region_names)))
            ax.set_yticks(range(len(region_names)))
            ax.set_xticklabels(region_names, rotation=45, ha='right')
            ax.set_yticklabels(region_names)
            ax.set_title(f'{shape_name} Cortical Connectivity')
            
            # Add colorbar
            plt.colorbar(im, ax=ax, label='Correlation')
            
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, 4):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'cortical_connectivity.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_summary_report(self, save_dir: str) -> None:
        """Generate a summary report of all analyses."""
        report_path = os.path.join(save_dir, 'interpretability_report.md')
        
        with open(report_path, 'w') as f:
            f.write("# Visual Cortex Speckle Imaging - Interpretability Analysis Report\n\n")
            
            f.write("## Overview\n")
            f.write("This report presents the interpretability analysis of the ConvLSTM model ")
            f.write("for visual cortex speckle imaging classification. The analysis includes ")
            f.write("feature importance, temporal patterns, spatial cortical mapping, and ")
            f.write("connectivity analysis.\n\n")
            
            # SHAP Analysis Summary
            if 'shap' in self.results:
                f.write("## SHAP Feature Importance Analysis\n")
                f.write("SHAP (SHapley Additive exPlanations) analysis reveals which speckle ")
                f.write("features contribute most to shape classification decisions.\n\n")
                
                shap_results = self.results['shap']
                f.write("### Key Findings:\n")
                
                # Overall importance statistics
                if 'overall' in shap_results['feature_importance']:
                    overall_importance = shap_results['feature_importance']['overall']
                    f.write(f"- **Total features analyzed**: {len(overall_importance)}\n")
                    f.write(f"- **Mean importance**: {np.mean(overall_importance):.4f}\n")
                    f.write(f"- **Max importance**: {np.max(overall_importance):.4f}\n")
                    f.write(f"- **Features with high importance (>90th percentile)**: {np.sum(overall_importance > np.percentile(overall_importance, 90))}\n\n")
                
                # Class-specific findings
                for class_name, importance in shap_results['feature_importance'].items():
                    if class_name == 'overall':
                        continue
                    f.write(f"### {class_name} Specific Patterns:\n")
                    f.write(f"- Mean importance: {np.mean(importance):.4f}\n")
                    f.write(f"- Most important feature index: {np.argmax(importance)}\n")
                    f.write(f"- Number of highly important features: {np.sum(importance > np.percentile(importance, 95))}\n\n")
            
            # Temporal Analysis Summary
            if 'temporal' in self.results:
                f.write("## Temporal Pattern Analysis\n")
                f.write("Analysis of temporal dependencies in speckle patterns reveals ")
                f.write("how shape-specific neural responses evolve over time.\n\n")
                
                temporal_results = self.results['temporal']
                
                if 'temporal_patterns' in temporal_results:
                    patterns = temporal_results['temporal_patterns']
                    f.write("### Temporal Clustering Results:\n")
                    f.write(f"- **PCA components**: {len(patterns['pca_explained_variance'])}\n")
                    f.write(f"- **Explained variance (top 5 components)**: {patterns['pca_explained_variance'][:5].tolist()}\n")
                    f.write(f"- **Total samples analyzed**: {len(patterns['labels'])}\n\n")
            
            # Spatial Mapping Summary
            if 'spatial' in self.results:
                f.write("## Spatial Cortical Mapping Analysis\n")
                f.write("Mapping of important speckle features to hypothetical cortical regions ")
                f.write("provides insights into potential neural substrates of shape processing.\n\n")
                
                spatial_results = self.results['spatial']
                
                if 'region_importance' in spatial_results:
                    f.write("### Cortical Region Analysis:\n")
                    region_data = spatial_results['region_importance']
                    
                    for region_name, region_info in region_data.items():
                        f.write(f"#### {region_name}\n")
                        if 'overall' in region_info:
                            overall_data = region_info['overall']
                            f.write(f"- Mean importance: {overall_data['mean_importance']:.4f}\n")
                            f.write(f"- Max importance: {overall_data['max_importance']:.4f}\n")
                            f.write(f"- Variability (std): {overall_data['std_importance']:.4f}\n")
                        
                        # Class-specific patterns
                        class_importances = []
                        for class_name in self.class_names:
                            if class_name in region_info:
                                class_importances.append(region_info[class_name]['mean_importance'])
                        
                        if class_importances:
                            most_responsive_class = self.class_names[np.argmax(class_importances)]
                            f.write(f"- Most responsive to: {most_responsive_class}\n")
                        f.write("\n")
                
                # Connectivity Analysis
                if 'connectivity' in spatial_results:
                    f.write("### Cortical Connectivity Analysis:\n")
                    connectivity_data = spatial_results['connectivity']
                    
                    for shape_name, conn_info in connectivity_data.items():
                        f.write(f"#### {shape_name} Connectivity:\n")
                        f.write(f"- Mean connectivity: {conn_info['mean_connectivity']:.4f}\n")
                        f.write(f"- Max connectivity: {conn_info['max_connectivity']:.4f}\n")
                        
                        # Strongest connections
                        strongest = conn_info['strongest_connections']
                        f.write(f"- Strongest connections:\n")
                        for i, conn in enumerate(strongest[:3]):
                            f.write(f"  {i+1}. {conn['region1']} ↔ {conn['region2']}: {conn['strength']:.4f}\n")
                        f.write("\n")
            
            # Research Implications
            f.write("## Research Implications\n")
            f.write("### Neuroscientific Insights:\n")
            f.write("1. **Edge Detection Hypothesis**: If V1_edge_detectors region shows high importance ")
            f.write("for rectangles and triangles but low for circles, this supports the hypothesis ")
            f.write("that edge-based processing is crucial for polygon recognition.\n\n")
            
            f.write("2. **Temporal Integration**: Temporal pattern analysis reveals how speckle ")
            f.write("information is integrated over time, potentially corresponding to neural ")
            f.write("integration windows in visual cortex.\n\n")
            
            f.write("3. **Cortical Specialization**: Different importance patterns across cortical ")
            f.write("regions suggest functional specialization consistent with known V1 organization.\n\n")
            
            f.write("### Clinical Applications:\n")
            f.write("1. **Biomarker Development**: Identified important features could serve as ")
            f.write("biomarkers for visual cortex functionality.\n\n")
            
            f.write("2. **Diagnostic Potential**: Abnormal patterns in important feature regions ")
            f.write("might indicate visual processing disorders.\n\n")
            
            f.write("3. **Therapeutic Targets**: Understanding which cortical regions are most ")
            f.write("critical for shape processing could inform therapeutic interventions.\n\n")
            
            f.write("### Future Directions:\n")
            f.write("1. **Validation Studies**: Validate identified important features with ")
            f.write("direct neural recordings or fMRI studies.\n\n")
            
            f.write("2. **Cross-subject Analysis**: Examine consistency of important features ")
            f.write("across different subjects and populations.\n\n")
            
            f.write("3. **Stimuli Expansion**: Test interpretability patterns with more complex ")
            f.write("visual stimuli and natural images.\n\n")
    
    def _save_shap_results(self, results: Dict[str, Any], save_dir: str) -> None:
        """Save SHAP analysis results."""
        # Save feature importance as CSV
        importance_df = pd.DataFrame(results['feature_importance'])
        importance_df.to_csv(os.path.join(save_dir, 'feature_importance.csv'))
        
        # Save SHAP values as numpy arrays
        if isinstance(results['shap_values'], list):
            for i, shap_vals in enumerate(results['shap_values']):
                np.save(os.path.join(save_dir, f'shap_values_class_{i}.npy'), shap_vals)
        else:
            np.save(os.path.join(save_dir, 'shap_values.npy'), results['shap_values'])
    
    def _save_temporal_results(self, results: Dict[str, Any], save_dir: str) -> None:
        """Save temporal analysis results."""
        temporal_dir = os.path.join(save_dir, 'temporal')
        os.makedirs(temporal_dir, exist_ok=True)
        
        # Save temporal patterns
        if 'temporal_patterns' in results:
            patterns = results['temporal_patterns']
            np.save(os.path.join(temporal_dir, 'tsne_features.npy'), patterns['tsne_features'])
            np.save(os.path.join(temporal_dir, 'pca_features.npy'), patterns['pca_features'])
            np.save(os.path.join(temporal_dir, 'labels.npy'), patterns['labels'])
    
    def _save_spatial_results(self, results: Dict[str, Any], save_dir: str, 
                            cortical_regions: Dict[str, Tuple[int, int]]) -> None:
        """Save spatial analysis results."""
        spatial_dir = os.path.join(save_dir, 'spatial')
        os.makedirs(spatial_dir, exist_ok=True)
        
        # Save cortical regions definition
        regions_df = pd.DataFrame([
            {'region': name, 'start': start, 'end': end} 
            for name, (start, end) in cortical_regions.items()
        ])
        regions_df.to_csv(os.path.join(spatial_dir, 'cortical_regions.csv'), index=False)
        
        # Save connectivity matrices
        if 'connectivity' in results:
            connectivity_dir = os.path.join(spatial_dir, 'connectivity')
            os.makedirs(connectivity_dir, exist_ok=True)
            
            for shape_name, conn_data in results['connectivity'].items():
                np.save(os.path.join(connectivity_dir, f'{shape_name}_connectivity.npy'), 
                       conn_data['connectivity_matrix'])
