"""
K-fold cross-validation trainer for speckle imaging classification models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np
import pandas as pd
import os
import json
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

from .trainer import Trainer
from .evaluation import ModelEvaluator
from ..models.base_model import ModelFactory
from ..utils.visualization import plot_training_history


class KFoldTrainer:
    """K-fold cross-validation trainer for speckle imaging models."""
    
    def __init__(self,
                 model_config: Dict[str, Any],
                 training_config: Dict[str, Any],
                 device: Optional[torch.device] = None,
                 save_dir: str = 'kfold_results',
                 log_dir: Optional[str] = None,
                 stratified: bool = True,
                 verbose: bool = True):
        """
        Initialize K-fold trainer.
        
        Args:
            model_config: Model configuration dictionary
            training_config: Training configuration dictionary
            device: Device to train on (auto-detect if None)
            save_dir: Directory to save results
            log_dir: Directory for tensorboard logs
            stratified: Whether to use stratified K-fold
            verbose: Whether to print progress
        """
        self.model_config = model_config
        self.training_config = training_config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.stratified = stratified
        self.verbose = verbose
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Results storage
        self.fold_results = []
        self.fold_histories = []
        self.fold_models = []
        
    def train_kfold(self,
                   dataset: torch.utils.data.Dataset,
                   class_names: Optional[List[str]] = None,
                   k_folds: int = 5,
                   random_state: int = 42) -> Dict[str, Any]:
        """
        Perform K-fold cross-validation.
        
        Args:
            dataset: Full dataset to split
            class_names: List of class names for evaluation
            k_folds: Number of folds
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary containing cross-validation results
        """
        if self.verbose:
            print(f"Starting {k_folds}-fold cross-validation...")
            print(f"Dataset size: {len(dataset)}")
            print(f"Using device: {self.device}")
        
        # Get labels for stratified split
        all_labels = []
        for i in range(len(dataset)):
            _, label = dataset[i]
            all_labels.append(label.item() if isinstance(label, torch.Tensor) else label)
        all_labels = np.array(all_labels)
        
        # Initialize K-fold splitter
        if self.stratified:
            kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)
            splits = kfold.split(np.arange(len(dataset)), all_labels)
        else:
            kfold = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
            splits = kfold.split(np.arange(len(dataset)))
        
        # Perform K-fold training
        for fold, (train_idx, val_idx) in enumerate(splits):
            if self.verbose:
                print(f"\n{'='*50}")
                print(f"FOLD {fold + 1}/{k_folds}")
                print(f"{'='*50}")
                print(f"Train samples: {len(train_idx)}")
                print(f"Validation samples: {len(val_idx)}")
            
            # Create fold-specific datasets
            train_subset = Subset(dataset, train_idx)
            val_subset = Subset(dataset, val_idx)
            
            # Create data loaders
            train_loader = DataLoader(
                train_subset,
                batch_size=self.training_config.get('batch_size', 32),
                shuffle=True,
                num_workers=0
            )
            val_loader = DataLoader(
                val_subset,
                batch_size=self.training_config.get('batch_size', 32),
                shuffle=False,
                num_workers=0
            )
            
            # Train model for this fold
            model, history, results = self._train_single_fold(
                fold, train_loader, val_loader, class_names
            )
            
            # Store results
            self.fold_results.append(results)
            self.fold_histories.append(history)
            self.fold_models.append(model)
            
            if self.verbose:
                print(f"Fold {fold + 1} Results:")
                print(f"  Accuracy: {results['accuracy']:.4f}")
                print(f"  Loss: {results['loss']:.4f}")
                print(f"  Precision: {results['precision']:.4f}")
                print(f"  Recall: {results['recall']:.4f}")
                print(f"  F1-Score: {results['f1_score']:.4f}")
        
        # Compile final results
        cv_results = self._compile_results(class_names)
        
        # Save results
        self._save_results(cv_results, k_folds)
        
        if self.verbose:
            print(f"\n{'='*50}")
            print("K-FOLD CROSS-VALIDATION COMPLETED")
            print(f"{'='*50}")
            self._print_final_results(cv_results)
        
        return cv_results
    
    def _train_single_fold(self,
                          fold: int,
                          train_loader: DataLoader,
                          val_loader: DataLoader,
                          class_names: Optional[List[str]] = None) -> Tuple[nn.Module, Dict, Dict]:
        """
        Train a single fold.
        
        Args:
            fold: Fold number
            train_loader: Training data loader
            val_loader: Validation data loader
            class_names: List of class names
            
        Returns:
            Tuple of (model, history, results)
        """
        # Create model
        model = ModelFactory.create_model(
            self.model_config['model_type'],
            num_classes=self.model_config['num_classes'],
            **{k: v for k, v in self.model_config.items() if k != 'model_type'}
        )
        
        # Create fold-specific save directory
        fold_save_dir = os.path.join(self.save_dir, f'fold_{fold + 1}')
        os.makedirs(fold_save_dir, exist_ok=True)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            device=self.device,
            save_dir=fold_save_dir,
            log_dir=os.path.join(self.log_dir, f'fold_{fold + 1}') if self.log_dir else None
        )
        
        # Set up optimizer
        if self.training_config.get('optimizer', 'adam').lower() == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(), 
                lr=self.training_config.get('learning_rate', 0.001)
            )
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), 
                lr=self.training_config.get('learning_rate', 0.001),
                momentum=0.9
            )
        
        # Train model
        history = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=self.training_config.get('epochs', 100),
            optimizer=optimizer,
            scheduler=self.training_config.get('scheduler', 'plateau'),
            early_stopping={
                'patience': self.training_config.get('early_stopping_patience', 20),
                'mode': 'min'
            },
            save_every=self.training_config.get('save_every', 50),
            save_best=self.training_config.get('save_best', True)
        )
        
        # Load best model for evaluation
        best_model_path = os.path.join(fold_save_dir, 'best_model.pth')
        if os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Evaluate model
        evaluator = ModelEvaluator(
            model=model,
            device=self.device,
            class_names=class_names
        )
        
        results = evaluator.evaluate(val_loader)
        
        # Save fold-specific evaluation report
        evaluator.save_evaluation_report(
            val_loader,
            save_dir=fold_save_dir,
            dataset_name=f'fold_{fold + 1}_validation'
        )
        
        # Save training history plot
        history_plot_path = os.path.join(fold_save_dir, f'fold_{fold + 1}_training_history.png')
        plot_training_history(history, save_path=history_plot_path)
        plt.close()
        
        return model, history, results
    
    def _compile_results(self, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compile results from all folds.
        
        Args:
            class_names: List of class names
            
        Returns:
            Dictionary containing compiled results
        """
        # Extract metrics from all folds
        metrics = ['accuracy', 'loss', 'precision', 'recall', 'f1_score']
        
        compiled_results = {}
        for metric in metrics:
            values = [fold_result[metric] for fold_result in self.fold_results]
            compiled_results[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'values': values
            }
        
        # Per-class metrics (if available)
        if self.fold_results and 'classification_report' in self.fold_results[0]:
            class_metrics = self._compile_class_metrics(class_names)
            compiled_results['per_class_metrics'] = class_metrics
        
        # Store individual fold results
        compiled_results['fold_results'] = self.fold_results
        compiled_results['fold_histories'] = self.fold_histories
        
        return compiled_results
    
    def _compile_class_metrics(self, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compile per-class metrics from all folds.
        
        Args:
            class_names: List of class names
            
        Returns:
            Dictionary containing per-class metrics
        """
        if not class_names:
            # Extract class names from first fold's classification report
            first_report = self.fold_results[0]['classification_report']
            class_names = [k for k in first_report.keys() 
                          if k not in ['accuracy', 'macro avg', 'weighted avg']]
        
        class_metrics = {}
        for class_name in class_names:
            class_metrics[class_name] = {}
            for metric in ['precision', 'recall', 'f1-score']:
                values = []
                for fold_result in self.fold_results:
                    if class_name in fold_result['classification_report']:
                        values.append(fold_result['classification_report'][class_name][metric])
                
                if values:
                    class_metrics[class_name][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'values': values
                    }
        
        return class_metrics
    
    def _save_results(self, cv_results: Dict[str, Any], k_folds: int):
        """
        Save cross-validation results.
        
        Args:
            cv_results: Compiled results
            k_folds: Number of folds
        """
        # Save summary results
        summary_path = os.path.join(self.save_dir, 'kfold_summary.json')
        
        # Prepare JSON-serializable results
        json_results = {}
        for metric, stats in cv_results.items():
            if metric in ['fold_results', 'fold_histories', 'per_class_metrics']:
                continue  # Skip complex nested structures for JSON
            json_results[metric] = stats
        
        with open(summary_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save detailed results as CSV
        self._save_results_csv(cv_results, k_folds)
        
        # Create and save visualization
        self._create_results_visualization(cv_results, k_folds)
    
    def _save_results_csv(self, cv_results: Dict[str, Any], k_folds: int):
        """
        Save results as CSV files.
        
        Args:
            cv_results: Compiled results
            k_folds: Number of folds
        """
        # Main metrics CSV
        metrics_data = []
        for i in range(k_folds):
            fold_data = {'fold': i + 1}
            for metric in ['accuracy', 'loss', 'precision', 'recall', 'f1_score']:
                fold_data[metric] = cv_results[metric]['values'][i]
            metrics_data.append(fold_data)
        
        # Add summary statistics
        summary_data = {'fold': 'mean'}
        for metric in ['accuracy', 'loss', 'precision', 'recall', 'f1_score']:
            summary_data[metric] = cv_results[metric]['mean']
        metrics_data.append(summary_data)
        
        summary_data = {'fold': 'std'}
        for metric in ['accuracy', 'loss', 'precision', 'recall', 'f1_score']:
            summary_data[metric] = cv_results[metric]['std']
        metrics_data.append(summary_data)
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_csv_path = os.path.join(self.save_dir, 'kfold_metrics.csv')
        metrics_df.to_csv(metrics_csv_path, index=False)
        
        # Per-class metrics CSV (if available)
        if 'per_class_metrics' in cv_results:
            class_data = []
            for class_name, class_metrics in cv_results['per_class_metrics'].items():
                for metric, stats in class_metrics.items():
                    class_data.append({
                        'class': class_name,
                        'metric': metric,
                        'mean': stats['mean'],
                        'std': stats['std']
                    })
            
            if class_data:
                class_df = pd.DataFrame(class_data)
                class_csv_path = os.path.join(self.save_dir, 'kfold_per_class_metrics.csv')
                class_df.to_csv(class_csv_path, index=False)
    
    def _create_results_visualization(self, cv_results: Dict[str, Any], k_folds: int):
        """
        Create visualization of cross-validation results.
        
        Args:
            cv_results: Compiled results
            k_folds: Number of folds
        """
        # Create a comprehensive visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Metrics across folds
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        metric_means = [cv_results[m]['mean'] for m in metrics]
        metric_stds = [cv_results[m]['std'] for m in metrics]
        
        axes[0, 0].bar(metrics, metric_means, yerr=metric_stds, capsize=5, alpha=0.7)
        axes[0, 0].set_title('Average Metrics Across Folds')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(metric_means, metric_stds)):
            axes[0, 0].text(i, mean + std + 0.02, f'{mean:.3f}±{std:.3f}', 
                           ha='center', va='bottom', fontsize=9)
        
        # 2. Accuracy across folds
        fold_nums = list(range(1, k_folds + 1))
        accuracies = cv_results['accuracy']['values']
        
        axes[0, 1].plot(fold_nums, accuracies, 'o-', linewidth=2, markersize=8)
        axes[0, 1].axhline(y=cv_results['accuracy']['mean'], color='red', 
                          linestyle='--', alpha=0.7, label=f"Mean: {cv_results['accuracy']['mean']:.3f}")
        axes[0, 1].set_title('Accuracy Across Folds')
        axes[0, 1].set_xlabel('Fold')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Loss across folds
        losses = cv_results['loss']['values']
        
        axes[1, 0].plot(fold_nums, losses, 'o-', linewidth=2, markersize=8, color='orange')
        axes[1, 0].axhline(y=cv_results['loss']['mean'], color='red', 
                          linestyle='--', alpha=0.7, label=f"Mean: {cv_results['loss']['mean']:.3f}")
        axes[1, 0].set_title('Loss Across Folds')
        axes[1, 0].set_xlabel('Fold')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Metrics distribution
        metrics_data = []
        for metric in metrics:
            metrics_data.extend([(metric, value) for value in cv_results[metric]['values']])
        
        metrics_df = pd.DataFrame(metrics_data, columns=['Metric', 'Value'])
        sns.boxplot(data=metrics_df, x='Metric', y='Value', ax=axes[1, 1])
        axes[1, 1].set_title('Distribution of Metrics Across Folds')
        axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, 'kfold_results_visualization.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create training history comparison
        self._create_training_history_comparison(k_folds)
    
    def _create_training_history_comparison(self, k_folds: int):
        """
        Create training history comparison across folds.
        
        Args:
            k_folds: Number of folds
        """
        if not self.fold_histories:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training and validation loss/accuracy for all folds
        for fold, history in enumerate(self.fold_histories):
            color = plt.cm.tab10(fold % 10)
            
            # Training loss
            if 'train_loss' in history:
                axes[0, 0].plot(history['train_loss'], color=color, alpha=0.7, 
                               label=f'Fold {fold + 1}')
            
            # Validation loss
            if 'val_loss' in history:
                axes[0, 1].plot(history['val_loss'], color=color, alpha=0.7, 
                               label=f'Fold {fold + 1}')
            
            # Training accuracy
            if 'train_acc' in history:
                axes[1, 0].plot(history['train_acc'], color=color, alpha=0.7, 
                               label=f'Fold {fold + 1}')
            
            # Validation accuracy
            if 'val_acc' in history:
                axes[1, 1].plot(history['val_acc'], color=color, alpha=0.7, 
                               label=f'Fold {fold + 1}')
        
        # Set titles and labels
        axes[0, 0].set_title('Training Loss Across Folds')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Validation Loss Across Folds')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        axes[1, 0].set_title('Training Accuracy Across Folds')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Validation Accuracy Across Folds')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, 'kfold_training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _print_final_results(self, cv_results: Dict[str, Any]):
        """
        Print final cross-validation results.
        
        Args:
            cv_results: Compiled results
        """
        print("\nCross-Validation Results:")
        print("-" * 40)
        
        for metric in ['accuracy', 'loss', 'precision', 'recall', 'f1_score']:
            stats = cv_results[metric]
            print(f"{metric.capitalize():12}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                  f"[{stats['min']:.4f}, {stats['max']:.4f}]")
        
        # Per-class results (if available)
        if 'per_class_metrics' in cv_results:
            print("\nPer-Class Results:")
            print("-" * 40)
            for class_name, class_metrics in cv_results['per_class_metrics'].items():
                print(f"\n{class_name}:")
                for metric, stats in class_metrics.items():
                    print(f"  {metric:10}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    
    def get_best_model(self) -> Tuple[nn.Module, int]:
        """
        Get the best model from all folds based on validation accuracy.
        
        Returns:
            Tuple of (best_model, best_fold_index)
        """
        if not self.fold_results:
            raise ValueError("No fold results available. Run train_kfold first.")
        
        best_fold_idx = np.argmax([result['accuracy'] for result in self.fold_results])
        best_model = self.fold_models[best_fold_idx]
        
        return best_model, best_fold_idx
    
    def save_best_model(self, save_path: str):
        """
        Save the best model from cross-validation.
        
        Args:
            save_path: Path to save the best model
        """
        best_model, best_fold_idx = self.get_best_model()
        
        checkpoint = {
            'model_state_dict': best_model.state_dict(),
            'model_config': self.model_config,
            'training_config': self.training_config,
            'best_fold': best_fold_idx + 1,
            'best_accuracy': self.fold_results[best_fold_idx]['accuracy'],
            'cv_results': {
                'accuracy_mean': np.mean([r['accuracy'] for r in self.fold_results]),
                'accuracy_std': np.std([r['accuracy'] for r in self.fold_results])
            }
        }
        
        torch.save(checkpoint, save_path)
        print(f"Best model from fold {best_fold_idx + 1} saved to {save_path}")
