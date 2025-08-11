"""
Leave-One-Subject-Out (LOSO) cross-validation trainer for speckle imaging classification models.

This module implements LOSO cross-validation where the model is trained on data from
seven participants and evaluated on the held-out eighth participant, cycling through
all possible combinations.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import json
from typing import Dict, List, Optional, Tuple, Any
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
import pickle
from sklearn.preprocessing import LabelEncoder

from .trainer import Trainer
from .evaluation import ModelEvaluator
from ..models.base_model import ModelFactory
from ..utils.visualization import plot_training_history
from ..data.dataset import SpeckleDataset
from ..data.preprocessing import prepare_dataset_splits


class LOSOSubjectDataset(Dataset):
    """Dataset that organizes data by subject for LOSO cross-validation."""
    
    def __init__(self, speckle_data: Dict, subjects: List[str], shape_filter: Optional[List[str]] = None,
                 feature_type: str = 'manhattan', n_chunks: int = 10):
        """
        Initialize LOSO dataset.
        
        Args:
            speckle_data: Dictionary containing speckle data by subject
            subjects: List of subjects to include
            shape_filter: List of shapes to include (None for all)
            feature_type: Type of feature to extract
            n_chunks: Number of chunks per video
        """
        self.speckle_data = speckle_data
        self.subjects = subjects
        self.shape_filter = shape_filter
        self.feature_type = feature_type
        self.n_chunks = n_chunks
        
        # Organize data by subject
        self.subject_data = {}
        self.label_encoder = LabelEncoder()
        
        # Collect all labels first for encoding
        all_labels = []
        for subject in subjects:
            if subject in speckle_data:
                for data in speckle_data[subject]:
                    label = data[1]
                    if shape_filter is None or label in shape_filter:
                        all_labels.append(label)
        
        # Fit label encoder
        self.label_encoder.fit(all_labels)
        self.class_names = self.label_encoder.classes_.tolist()
        
        # Prepare data for each subject
        for subject in subjects:
            self._prepare_subject_data(subject)
    
    def _prepare_subject_data(self, subject: str):
        """Prepare data for a specific subject."""
        if subject not in self.speckle_data:
            self.subject_data[subject] = {'features': [], 'labels': []}
            return
        
        subject_features = []
        subject_labels = []
        
        for data in self.speckle_data[subject]:
            label = data[1]
            
            # Filter shapes if specified
            if self.shape_filter is not None and label not in self.shape_filter:
                continue
            
            # Extract features
            from ..data.preprocessing import extract_features_from_data
            features = extract_features_from_data(data, self.feature_type)
            
            # Split into chunks
            if len(features) > 0:
                chunks = np.array_split(features, self.n_chunks)
                for chunk in chunks:
                    if len(chunk) > 0:
                        subject_features.append(chunk)
                        subject_labels.append(label)
        
        # Encode labels
        if subject_labels:
            encoded_labels = self.label_encoder.transform(subject_labels)
            self.subject_data[subject] = {
                'features': np.array(subject_features),
                'labels': encoded_labels
            }
        else:
            self.subject_data[subject] = {'features': [], 'labels': []}
    
    def get_subject_dataset(self, subject: str) -> SpeckleDataset:
        """Get dataset for a specific subject."""
        if subject not in self.subject_data or len(self.subject_data[subject]['features']) == 0:
            # Return empty dataset
            return SpeckleDataset(np.empty((0, 1, 1)), np.array([]))
        
        features = self.subject_data[subject]['features']
        labels = self.subject_data[subject]['labels']
        return SpeckleDataset(features, labels)
    
    def get_multi_subject_dataset(self, subjects: List[str]) -> SpeckleDataset:
        """Get combined dataset for multiple subjects."""
        all_features = []
        all_labels = []
        
        for subject in subjects:
            if subject in self.subject_data and len(self.subject_data[subject]['features']) > 0:
                all_features.append(self.subject_data[subject]['features'])
                all_labels.append(self.subject_data[subject]['labels'])
        
        if all_features:
            combined_features = np.concatenate(all_features, axis=0)
            combined_labels = np.concatenate(all_labels, axis=0)
            return SpeckleDataset(combined_features, combined_labels)
        else:
            return SpeckleDataset(np.empty((0, 1, 1)), np.array([]))
    
    def get_subject_sample_counts(self) -> Dict[str, int]:
        """Get sample count for each subject."""
        counts = {}
        for subject in self.subjects:
            if subject in self.subject_data:
                counts[subject] = len(self.subject_data[subject]['features'])
            else:
                counts[subject] = 0
        return counts


class LOSOTrainer:
    """Leave-One-Subject-Out cross-validation trainer for speckle imaging models."""
    
    def __init__(self,
                 model_config: Dict[str, Any],
                 training_config: Dict[str, Any],
                 device: Optional[torch.device] = None,
                 save_dir: str = 'loso_results',
                 log_dir: Optional[str] = None,
                 verbose: bool = True):
        """
        Initialize LOSO trainer.
        
        Args:
            model_config: Dictionary containing model configuration
            training_config: Dictionary containing training configuration
            device: Device to train on
            save_dir: Directory to save results
            log_dir: Directory for logging (optional)
            verbose: Whether to print progress
        """
        self.model_config = model_config
        self.training_config = training_config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        self.log_dir = log_dir
        self.verbose = verbose
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # Results storage
        self.loso_results = []
        self.loso_histories = []
        self.loso_models = []
        self.subject_results = {}
    
    def train_loso(self,
                   speckle_data: Dict,
                   subjects: List[str],
                   class_names: Optional[List[str]] = None,
                   shape_filter: Optional[List[str]] = None,
                   feature_type: str = 'manhattan',
                   n_chunks: int = 10,
                   random_state: int = 42) -> Dict[str, Any]:
        """
        Perform Leave-One-Subject-Out cross-validation.
        
        Args:
            speckle_data: Dictionary of processed speckle data
            subjects: List of subjects to include
            class_names: List of class names for evaluation
            shape_filter: List of shapes to include
            feature_type: Type of feature to extract
            n_chunks: Number of chunks per video
            random_state: Random state for reproducibility
            
        Returns:
            Dictionary containing LOSO cross-validation results
        """
        if self.verbose:
            print(f"Starting Leave-One-Subject-Out cross-validation...")
            print(f"Total subjects: {len(subjects)}")
            print(f"Using device: {self.device}")
        
        # Set random seed
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        # Create LOSO dataset organizer
        loso_dataset = LOSOSubjectDataset(
            speckle_data=speckle_data,
            subjects=subjects,
            shape_filter=shape_filter,
            feature_type=feature_type,
            n_chunks=n_chunks
        )
        
        if class_names is None:
            class_names = loso_dataset.class_names
        
        # Print subject sample counts
        if self.verbose:
            print("\nSubject sample counts:")
            sample_counts = loso_dataset.get_subject_sample_counts()
            for subject, count in sample_counts.items():
                print(f"  {subject}: {count} samples")
        
        # Perform LOSO cross-validation
        for i, test_subject in enumerate(subjects):
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"LOSO FOLD {i + 1}/{len(subjects)}")
                print(f"Test Subject: {test_subject}")
                
                # Get train subjects
                train_subjects = [s for s in subjects if s != test_subject]
                print(f"Train Subjects: {', '.join(train_subjects)}")
                print(f"{'='*60}")
            
            # Create train and test datasets
            train_subjects = [s for s in subjects if s != test_subject]
            train_dataset = loso_dataset.get_multi_subject_dataset(train_subjects)
            test_dataset = loso_dataset.get_subject_dataset(test_subject)
            
            if len(train_dataset) == 0 or len(test_dataset) == 0:
                print(f"⚠️ Skipping {test_subject}: insufficient data")
                continue
            
            if self.verbose:
                print(f"Train samples: {len(train_dataset)}")
                print(f"Test samples: {len(test_dataset)}")
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.training_config.get('batch_size', 32),
                shuffle=True,
                num_workers=0
            )
            
            # Use 20% of training data for validation
            val_size = max(1, int(len(train_dataset) * 0.2))
            train_size = len(train_dataset) - val_size
            train_subset, val_subset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            
            val_loader = DataLoader(
                val_subset,
                batch_size=self.training_config.get('batch_size', 32),
                shuffle=False,
                num_workers=0
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=self.training_config.get('batch_size', 32),
                shuffle=False,
                num_workers=0
            )
            
            # Train model for this fold
            model, history, results = self._train_single_loso_fold(
                fold=i,
                test_subject=test_subject,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                class_names=class_names
            )
            
            # Store results
            self.loso_results.append(results)
            self.loso_histories.append(history)
            self.loso_models.append(model)
            self.subject_results[test_subject] = results
            
            if self.verbose:
                print(f"Subject {test_subject} Results:")
                print(f"  Accuracy: {results['accuracy']:.4f}")
                print(f"  Loss: {results['loss']:.4f}")
                print(f"  Precision: {results['precision']:.4f}")
                print(f"  Recall: {results['recall']:.4f}")
                print(f"  F1-Score: {results['f1_score']:.4f}")
        
        # Compile final results
        loso_results = self._compile_results(class_names)
        
        # Save results
        self._save_results(loso_results, subjects)
        
        if self.verbose:
            print(f"\n{'='*60}")
            print("LEAVE-ONE-SUBJECT-OUT CROSS-VALIDATION COMPLETED")
            print(f"{'='*60}")
            self._print_final_results(loso_results)
        
        return loso_results
    
    def _train_single_loso_fold(self,
                               fold: int,
                               test_subject: str,
                               train_loader: DataLoader,
                               val_loader: DataLoader,
                               test_loader: DataLoader,
                               class_names: Optional[List[str]] = None) -> Tuple[nn.Module, Dict, Dict]:
        """
        Train a single LOSO fold.
        
        Args:
            fold: Fold number
            test_subject: Subject being tested
            train_loader: Training data loader
            val_loader: Validation data loader  
            test_loader: Test data loader
            class_names: List of class names
            
        Returns:
            Tuple of (model, history, results)
        """
        # Create fold-specific save directory
        fold_save_dir = os.path.join(self.save_dir, f'subject_{test_subject}')
        os.makedirs(fold_save_dir, exist_ok=True)
        
        # Create model
        model = ModelFactory.create_model(**self.model_config)
        model.to(self.device)
        
        # Create trainer
        trainer = Trainer(
            model=model,
            device=self.device,
            save_dir=fold_save_dir,
            log_dir=os.path.join(self.log_dir, f'subject_{test_subject}') if self.log_dir else None
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
        
        # Evaluate model on test subject
        evaluator = ModelEvaluator(
            model=model,
            device=self.device,
            class_names=class_names
        )
        
        results = evaluator.evaluate(test_loader)
        
        # Save subject-specific evaluation report
        evaluator.save_evaluation_report(
            test_loader,
            save_dir=fold_save_dir,
            dataset_name=f'subject_{test_subject}_test'
        )
        
        # Save training history plot
        history_plot_path = os.path.join(fold_save_dir, f'subject_{test_subject}_training_history.png')
        plot_training_history(history, save_path=history_plot_path)
        plt.close()
        
        return model, history, results
    
    def _compile_results(self, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """Compile results across all LOSO folds."""
        if not self.loso_results:
            return {}
        
        # Collect metrics across folds
        metrics = ['accuracy', 'loss', 'precision', 'recall', 'f1_score']
        compiled_results = {}
        
        for metric in metrics:
            values = [result[metric] for result in self.loso_results if metric in result]
            if values:
                compiled_results[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'values': values
                }
        
        # Per-class metrics if available
        if class_names and 'per_class_metrics' in self.loso_results[0]:
            compiled_results['per_class_metrics'] = self._compile_class_metrics(class_names)
        
        # Store individual subject results
        compiled_results['subject_results'] = self.subject_results
        compiled_results['loso_histories'] = self.loso_histories
        
        return compiled_results
    
    def _compile_class_metrics(self, class_names: List[str]) -> Dict[str, Any]:
        """Compile per-class metrics across subjects."""
        class_metrics = {}
        
        for class_name in class_names:
            class_metrics[class_name] = {}
            
            # Collect per-class metrics across subjects
            for metric in ['precision', 'recall', 'f1_score']:
                values = []
                for result in self.loso_results:
                    if ('per_class_metrics' in result and 
                        class_name in result['per_class_metrics'] and
                        metric in result['per_class_metrics'][class_name]):
                        values.append(result['per_class_metrics'][class_name][metric])
                
                if values:
                    class_metrics[class_name][metric] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'values': values
                    }
        
        return class_metrics
    
    def _save_results(self, loso_results: Dict[str, Any], subjects: List[str]):
        """Save LOSO results to files."""
        # Save JSON summary
        summary_path = os.path.join(self.save_dir, 'loso_summary.json')
        
        # Convert numpy types to native Python types for JSON serialization
        json_results = {}
        for key, value in loso_results.items():
            if key in ['subject_results', 'loso_histories']:
                continue  # Skip complex nested structures
            elif isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, dict):
                        json_results[key][k] = {
                            sub_k: float(sub_v) if isinstance(sub_v, (np.floating, np.integer)) 
                                   else sub_v.tolist() if isinstance(sub_v, np.ndarray) 
                                   else sub_v
                            for sub_k, sub_v in v.items()
                        }
                    else:
                        json_results[key][k] = float(v) if isinstance(v, (np.floating, np.integer)) else v
            else:
                json_results[key] = float(value) if isinstance(value, (np.floating, np.integer)) else value
        
        with open(summary_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save CSV with detailed metrics
        self._save_results_csv(loso_results, subjects)
        
        # Create visualizations
        self._create_results_visualization(loso_results, subjects)
    
    def _save_results_csv(self, loso_results: Dict[str, Any], subjects: List[str]):
        """Save detailed metrics to CSV files."""
        # Main metrics CSV
        metrics_data = []
        
        for i, (subject, result) in enumerate(self.subject_results.items()):
            subject_data = {'subject': subject}
            for metric in ['accuracy', 'loss', 'precision', 'recall', 'f1_score']:
                if metric in result:
                    subject_data[metric] = result[metric]
            metrics_data.append(subject_data)
        
        # Add summary row
        summary_data = {'subject': 'MEAN'}
        for metric in ['accuracy', 'loss', 'precision', 'recall', 'f1_score']:
            if metric in loso_results:
                summary_data[metric] = loso_results[metric]['mean']
        metrics_data.append(summary_data)
        
        summary_data = {'subject': 'STD'}
        for metric in ['accuracy', 'loss', 'precision', 'recall', 'f1_score']:
            if metric in loso_results:
                summary_data[metric] = loso_results[metric]['std']
        metrics_data.append(summary_data)
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_csv_path = os.path.join(self.save_dir, 'loso_metrics.csv')
        metrics_df.to_csv(metrics_csv_path, index=False)
        
        # Per-class metrics CSV (if available)
        if 'per_class_metrics' in loso_results:
            class_data = []
            for class_name, class_metrics in loso_results['per_class_metrics'].items():
                for metric, stats in class_metrics.items():
                    class_data.append({
                        'class': class_name,
                        'metric': metric,
                        'mean': stats['mean'],
                        'std': stats['std']
                    })
            
            if class_data:
                class_df = pd.DataFrame(class_data)
                class_csv_path = os.path.join(self.save_dir, 'loso_per_class_metrics.csv')
                class_df.to_csv(class_csv_path, index=False)
    
    def _create_results_visualization(self, loso_results: Dict[str, Any], subjects: List[str]):
        """Create comprehensive visualization of LOSO results."""
        plt.rcParams.update({'font.size': 10})
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Leave-One-Subject-Out Cross-Validation Results', fontsize=16, fontweight='bold')
        
        # 1. Accuracy across subjects
        subject_accuracies = [self.subject_results[subject]['accuracy'] for subject in subjects
                            if subject in self.subject_results]
        subject_names = [subject for subject in subjects if subject in self.subject_results]
        
        axes[0, 0].bar(range(len(subject_names)), subject_accuracies, alpha=0.7, color='skyblue')
        axes[0, 0].axhline(y=loso_results['accuracy']['mean'], color='red', 
                          linestyle='--', alpha=0.7, label=f"Mean: {loso_results['accuracy']['mean']:.3f}")
        axes[0, 0].set_title('Accuracy by Test Subject')
        axes[0, 0].set_xlabel('Test Subject')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xticks(range(len(subject_names)))
        axes[0, 0].set_xticklabels(subject_names, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. F1-Score across subjects
        subject_f1_scores = [self.subject_results[subject]['f1_score'] for subject in subjects
                           if subject in self.subject_results]
        
        axes[0, 1].bar(range(len(subject_names)), subject_f1_scores, alpha=0.7, color='lightgreen')
        axes[0, 1].axhline(y=loso_results['f1_score']['mean'], color='red', 
                          linestyle='--', alpha=0.7, label=f"Mean: {loso_results['f1_score']['mean']:.3f}")
        axes[0, 1].set_title('F1-Score by Test Subject')
        axes[0, 1].set_xlabel('Test Subject')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].set_xticks(range(len(subject_names)))
        axes[0, 1].set_xticklabels(subject_names, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Loss across subjects
        subject_losses = [self.subject_results[subject]['loss'] for subject in subjects
                        if subject in self.subject_results]
        
        axes[1, 0].bar(range(len(subject_names)), subject_losses, alpha=0.7, color='lightcoral')
        axes[1, 0].axhline(y=loso_results['loss']['mean'], color='red', 
                          linestyle='--', alpha=0.7, label=f"Mean: {loso_results['loss']['mean']:.3f}")
        axes[1, 0].set_title('Loss by Test Subject')
        axes[1, 0].set_xlabel('Test Subject')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_xticks(range(len(subject_names)))
        axes[1, 0].set_xticklabels(subject_names, rotation=45, ha='right')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Metrics distribution
        metrics_data = []
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in metrics:
            if metric in loso_results:
                for value in loso_results[metric]['values']:
                    metrics_data.append({'Metric': metric.capitalize(), 'Value': value})
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            sns.boxplot(data=metrics_df, x='Metric', y='Value', ax=axes[1, 1])
            axes[1, 1].set_title('Distribution of Metrics Across Subjects')
            axes[1, 1].set_ylim(0, 1)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, 'loso_results_visualization.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create training history comparison
        self._create_training_history_comparison(subjects)
    
    def _create_training_history_comparison(self, subjects: List[str]):
        """Create comparison of training histories across subjects."""
        if not self.loso_histories:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History Comparison Across Subjects', fontsize=16, fontweight='bold')
        
        # Training and validation loss
        for i, (subject, history) in enumerate(zip(subjects, self.loso_histories)):
            if subject in self.subject_results:
                alpha = 0.7
                epochs = range(1, len(history['train_loss']) + 1)
                
                axes[0, 0].plot(epochs, history['train_loss'], alpha=alpha, label=f'{subject} (Train)')
                if 'val_loss' in history and history['val_loss']:
                    axes[0, 1].plot(epochs, history['val_loss'], alpha=alpha, label=f'{subject} (Val)')
        
        axes[0, 0].set_title('Training Loss Across Subjects')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].set_title('Validation Loss Across Subjects')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training and validation accuracy
        for i, (subject, history) in enumerate(zip(subjects, self.loso_histories)):
            if subject in self.subject_results:
                alpha = 0.7
                epochs = range(1, len(history['train_acc']) + 1)
                
                axes[1, 0].plot(epochs, history['train_acc'], alpha=alpha, label=f'{subject} (Train)')
                if 'val_acc' in history and history['val_acc']:
                    axes[1, 1].plot(epochs, history['val_acc'], alpha=alpha, label=f'{subject} (Val)')
        
        axes[1, 0].set_title('Training Accuracy Across Subjects')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy (%)')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].set_title('Validation Accuracy Across Subjects')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.save_dir, 'loso_training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _print_final_results(self, loso_results: Dict[str, Any]):
        """Print final LOSO results summary."""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'loss']
        
        print("LOSO Cross-Validation Results Summary:")
        print("-" * 50)
        
        for metric in metrics:
            if metric in loso_results:
                mean_val = loso_results[metric]['mean']
                std_val = loso_results[metric]['std']
                print(f"{metric.capitalize()}: {mean_val:.4f} ± {std_val:.4f}")
        
        print("-" * 50)
        print("Per-Subject Results:")
        
        for subject, results in self.subject_results.items():
            print(f"\n{subject}:")
            for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
                if metric in results:
                    print(f"  {metric}: {results[metric]:.4f}")
    
    def get_best_model(self) -> Tuple[nn.Module, str]:
        """Get the best performing model and its corresponding test subject."""
        if not self.loso_results:
            raise ValueError("No models have been trained yet")
        
        # Find best model based on highest accuracy
        best_idx = np.argmax([result['accuracy'] for result in self.loso_results])
        best_model = self.loso_models[best_idx]
        
        # Get corresponding test subject
        subjects = list(self.subject_results.keys())
        best_subject = subjects[best_idx]
        
        return best_model, best_subject
    
    def save_best_model(self, save_path: str):
        """Save the best performing model."""
        best_model, best_subject = self.get_best_model()
        best_accuracy = self.subject_results[best_subject]['accuracy']
        
        # Compile LOSO results
        loso_results = self._compile_results()
        
        checkpoint = {
            'model_state_dict': best_model.state_dict(),
            'model_config': self.model_config,
            'training_config': self.training_config,
            'best_subject': best_subject,
            'best_accuracy': best_accuracy,
            'loso_results': loso_results,
            'class_names': list(self.subject_results.keys())
        }
        
        torch.save(checkpoint, save_path)
        
        if self.verbose:
            print(f"Best model saved to: {save_path}")
            print(f"Best test subject: {best_subject}")
            print(f"Best accuracy: {best_accuracy:.4f}")
