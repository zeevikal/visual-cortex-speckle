"""
Model evaluation and metrics for speckle imaging classification.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Optional
import os


class ModelEvaluator:
    """Utility class for model evaluation and metrics."""
    
    def __init__(self, 
                 model: nn.Module, 
                 device: Optional[torch.device] = None,
                 class_names: Optional[List[str]] = None):
        """
        Initialize evaluator.
        
        Args:
            model: Trained PyTorch model
            device: Device to run evaluation on
            class_names: List of class names
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions on a dataset.
        
        Args:
            data_loader: DataLoader for the dataset
            
        Returns:
            Tuple of (predictions, probabilities, true_labels)
        """
        all_predictions = []
        all_probabilities = []
        all_labels = []
        
        with torch.no_grad():
            for data, target in data_loader:
                data = data.to(self.device)
                
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                predictions = output.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_labels.extend(target.numpy())
        
        return (np.array(all_predictions), 
                np.array(all_probabilities), 
                np.array(all_labels))
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Args:
            data_loader: DataLoader for the dataset
            
        Returns:
            Dictionary with evaluation metrics
        """
        predictions, probabilities, true_labels = self.predict(data_loader)
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, predictions)
        
        # Per-class metrics
        if self.class_names:
            target_names = self.class_names
        else:
            target_names = [f'Class_{i}' for i in range(len(np.unique(true_labels)))]
        
        class_report = classification_report(
            true_labels, 
            predictions, 
            target_names=target_names,
            output_dict=True
        )
        
        results = {
            'accuracy': accuracy,
            'classification_report': class_report
        }
        
        return results
    
    def plot_confusion_matrix(self, 
                            data_loader: DataLoader,
                            save_path: Optional[str] = None,
                            normalize: bool = True,
                            figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            data_loader: DataLoader for the dataset
            save_path: Path to save the plot
            normalize: Whether to normalize the confusion matrix
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        predictions, _, true_labels = self.predict(data_loader)
        
        # Compute confusion matrix
        cm = confusion_matrix(true_labels, predictions)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Class names
        if self.class_names:
            labels = self.class_names
        else:
            labels = [f'Class_{i}' for i in range(cm.shape[0])]
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd',
                   xticklabels=labels, yticklabels=labels,
                   cmap='Blues', ax=ax)
        
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_classification_report(self,
                                 data_loader: DataLoader,
                                 save_path: Optional[str] = None,
                                 figsize: Tuple[int, int] = (10, 6)) -> plt.Figure:
        """
        Plot classification report as heatmap.
        
        Args:
            data_loader: DataLoader for the dataset
            save_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        results = self.evaluate(data_loader)
        class_report = results['classification_report']
        
        # Convert to DataFrame
        df = pd.DataFrame(class_report).transpose()
        
        # Remove non-numeric rows
        df = df.drop(['accuracy'], errors='ignore')
        df = df.iloc[:-2, :-1]  # Remove macro/weighted avg and support column
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot heatmap
        sns.heatmap(df, annot=True, fmt='.3f', cmap='Blues', ax=ax)
        
        ax.set_title('Classification Report')
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Classes')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def analyze_predictions(self, 
                          data_loader: DataLoader,
                          num_samples: int = 10) -> Dict:
        """
        Analyze model predictions with examples.
        
        Args:
            data_loader: DataLoader for the dataset
            num_samples: Number of samples to analyze
            
        Returns:
            Dictionary with analysis results
        """
        predictions, probabilities, true_labels = self.predict(data_loader)
        
        # Find correct and incorrect predictions
        correct_mask = predictions == true_labels
        incorrect_mask = ~correct_mask
        
        # Get sample indices
        correct_indices = np.where(correct_mask)[0][:num_samples//2]
        incorrect_indices = np.where(incorrect_mask)[0][:num_samples//2]
        
        analysis = {
            'total_samples': len(predictions),
            'correct_predictions': np.sum(correct_mask),
            'incorrect_predictions': np.sum(incorrect_mask),
            'accuracy': np.mean(correct_mask),
            'correct_samples': {
                'indices': correct_indices.tolist(),
                'predictions': predictions[correct_indices].tolist(),
                'true_labels': true_labels[correct_indices].tolist(),
                'confidence': np.max(probabilities[correct_indices], axis=1).tolist()
            },
            'incorrect_samples': {
                'indices': incorrect_indices.tolist(),
                'predictions': predictions[incorrect_indices].tolist(),
                'true_labels': true_labels[incorrect_indices].tolist(),
                'confidence': np.max(probabilities[incorrect_indices], axis=1).tolist()
            }
        }
        
        return analysis
    
    def save_evaluation_report(self, 
                             data_loader: DataLoader,
                             save_dir: str,
                             dataset_name: str = 'test'):
        """
        Generate and save comprehensive evaluation report.
        
        Args:
            data_loader: DataLoader for the dataset
            save_dir: Directory to save results
            dataset_name: Name of the dataset (for file naming)
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Evaluate model
        results = self.evaluate(data_loader)
        analysis = self.analyze_predictions(data_loader)
        
        # Save classification report as CSV
        class_report_df = pd.DataFrame(results['classification_report']).transpose()
        report_path = os.path.join(save_dir, f'{dataset_name}_classification_report.csv')
        class_report_df.to_csv(report_path)
        
        # Save confusion matrix plot
        cm_path = os.path.join(save_dir, f'{dataset_name}_confusion_matrix.png')
        self.plot_confusion_matrix(data_loader, save_path=cm_path)
        plt.close()
        
        # Save classification report plot
        cr_path = os.path.join(save_dir, f'{dataset_name}_classification_heatmap.png')
        self.plot_classification_report(data_loader, save_path=cr_path)
        plt.close()
        
        # Save analysis results
        import json
        analysis_path = os.path.join(save_dir, f'{dataset_name}_analysis.json')
        with open(analysis_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Evaluation results saved to {save_dir}")
        print(f"Accuracy: {results['accuracy']:.4f}")


def compare_models(models: Dict[str, nn.Module], 
                  test_loader: DataLoader,
                  class_names: Optional[List[str]] = None,
                  save_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Compare multiple models on the same test set.
    
    Args:
        models: Dictionary of model_name -> model
        test_loader: Test data loader
        class_names: List of class names
        save_dir: Directory to save comparison results
        
    Returns:
        DataFrame with comparison results
    """
    results = []
    
    for model_name, model in models.items():
        evaluator = ModelEvaluator(model, class_names=class_names)
        model_results = evaluator.evaluate(test_loader)
        
        # Extract key metrics
        accuracy = model_results['accuracy']
        class_report = model_results['classification_report']
        
        macro_avg = class_report['macro avg']
        weighted_avg = class_report['weighted avg']
        
        results.append({
            'Model': model_name,
            'Accuracy': accuracy,
            'Macro Precision': macro_avg['precision'],
            'Macro Recall': macro_avg['recall'],
            'Macro F1': macro_avg['f1-score'],
            'Weighted Precision': weighted_avg['precision'],
            'Weighted Recall': weighted_avg['recall'],
            'Weighted F1': weighted_avg['f1-score']
        })
    
    comparison_df = pd.DataFrame(results)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        comparison_path = os.path.join(save_dir, 'model_comparison.csv')
        comparison_df.to_csv(comparison_path, index=False)
    
    return comparison_df
