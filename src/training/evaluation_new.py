"""
Model evaluation and metrics for speckle imaging classification.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from typing import Dict, List, Optional, Tuple
import os
import json
from tqdm import tqdm


class ModelEvaluator:
    """Comprehensive model evaluation for speckle imaging classifiers."""
    
    def __init__(self, 
                 model: nn.Module,
                 device: Optional[torch.device] = None,
                 class_names: Optional[List[str]] = None):
        """
        Initialize model evaluator.
        
        Args:
            model: Trained PyTorch model
            device: Device to run evaluation on
            class_names: List of class names for reporting
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = class_names
        self.model.to(self.device)
    
    def evaluate(self, data_loader: DataLoader) -> Dict:
        """
        Comprehensive model evaluation.
        
        Args:
            data_loader: DataLoader for evaluation
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        
        all_predictions = []
        all_probabilities = []
        all_targets = []
        total_loss = 0.0
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, targets in tqdm(data_loader, desc="Evaluating"):
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                
                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                predictions = torch.argmax(outputs, dim=1)
                
                # Collect results
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                total_loss += loss.item()
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        all_targets = np.array(all_targets)
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        avg_loss = total_loss / len(data_loader)
        
        # Multi-class metrics
        precision = precision_score(all_targets, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_targets, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)
        
        # Classification report
        target_names = self.class_names if self.class_names else None
        class_report = classification_report(
            all_targets, all_predictions, 
            target_names=target_names, 
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(all_targets, all_predictions)
        
        # ROC AUC for binary/multi-class
        try:
            if len(np.unique(all_targets)) == 2:
                # Binary classification
                auc = roc_auc_score(all_targets, all_probabilities[:, 1])
            else:
                # Multi-class
                auc = roc_auc_score(all_targets, all_probabilities, multi_class='ovr', average='weighted')
        except ValueError:
            auc = None
        
        results = {
            'accuracy': accuracy,
            'loss': avg_loss,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'predictions': all_predictions,
            'probabilities': all_probabilities,
            'targets': all_targets
        }
        
        return results
    
    def plot_confusion_matrix(self, 
                            data_loader: DataLoader,
                            normalize: bool = True,
                            figsize: Tuple[int, int] = (8, 6),
                            save_path: Optional[str] = None) -> plt.Figure:
        """Plot confusion matrix."""
        results = self.evaluate(data_loader)
        cm = results['confusion_matrix']
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2f'
            title = 'Normalized Confusion Matrix'
        else:
            fmt = 'd'
            title = 'Confusion Matrix'
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create labels
        if self.class_names:
            labels = self.class_names
        else:
            labels = [f'Class {i}' for i in range(len(cm))]
        
        # Plot heatmap
        sns.heatmap(
            cm, 
            annot=True, 
            fmt=fmt, 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            ax=ax
        )
        
        ax.set_title(title)
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_evaluation_report(self,
                             data_loader: DataLoader,
                             save_dir: str,
                             dataset_name: str = 'test'):
        """Save comprehensive evaluation report."""
        os.makedirs(save_dir, exist_ok=True)
        
        # Evaluate model
        results = self.evaluate(data_loader)
        
        # Save classification report as CSV
        class_report_df = pd.DataFrame(results['classification_report']).transpose()
        class_report_path = os.path.join(save_dir, f'{dataset_name}_classification_report.csv')
        class_report_df.to_csv(class_report_path)
        
        # Save confusion matrix plot
        cm_path = os.path.join(save_dir, f'{dataset_name}_confusion_matrix.png')
        fig_cm = self.plot_confusion_matrix(data_loader, save_path=cm_path)
        plt.close(fig_cm)
        
        # Save numerical results
        numerical_results = {
            'accuracy': float(results['accuracy']),
            'loss': float(results['loss']),
            'precision': float(results['precision']),
            'recall': float(results['recall']),
            'f1_score': float(results['f1_score']),
            'auc': float(results['auc']) if results['auc'] is not None else None,
            'num_samples': len(results['targets']),
            'num_classes': len(np.unique(results['targets']))
        }
        
        results_path = os.path.join(save_dir, f'{dataset_name}_analysis.json')
        with open(results_path, 'w') as f:
            json.dump(numerical_results, f, indent=2)
        
        print(f"Evaluation Results for {dataset_name} dataset:")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Loss: {results['loss']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")
        if results['auc'] is not None:
            print(f"AUC: {results['auc']:.4f}")
        print(f"Results saved to {save_dir}")
