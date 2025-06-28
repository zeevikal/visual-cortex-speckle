"""
Training pipeline for speckle imaging classification models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os
import time
from typing import Dict, List, Optional, Tuple, Callable
from tqdm import tqdm
import json


class EarlyStopping:
    """Early stopping utility."""
    
    def __init__(self, patience: int = 100, min_delta: float = 0, mode: str = 'min'):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for minimizing metric, 'max' for maximizing
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
        self.is_better = (lambda new, best: new < best - min_delta) if mode == 'min' else (lambda new, best: new > best + min_delta)
    
    def __call__(self, score: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            score: Current metric score
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
        elif self.is_better(score, self.best_score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class LRScheduler:
    """Learning rate scheduler wrapper."""
    
    def __init__(self, optimizer, scheduler_type: str = 'plateau', **kwargs):
        """
        Initialize learning rate scheduler.
        
        Args:
            optimizer: PyTorch optimizer
            scheduler_type: Type of scheduler ('plateau', 'step', 'cosine')
            **kwargs: Additional arguments for scheduler
        """
        self.optimizer = optimizer
        self.scheduler_type = scheduler_type
        
        if scheduler_type == 'plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode=kwargs.get('mode', 'min'),
                factor=kwargs.get('factor', 0.5),
                patience=kwargs.get('patience', 20),
                min_lr=kwargs.get('min_lr', 0.0001),
                verbose=kwargs.get('verbose', True)
            )
        elif scheduler_type == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=kwargs.get('step_size', 30),
                gamma=kwargs.get('gamma', 0.1)
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=kwargs.get('T_max', 100)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    def step(self, metric: Optional[float] = None):
        """Step the scheduler."""
        if self.scheduler_type == 'plateau' and metric is not None:
            self.scheduler.step(metric)
        else:
            self.scheduler.step()


class Trainer:
    """Training pipeline for speckle imaging models."""
    
    def __init__(self,
                 model: nn.Module,
                 device: Optional[torch.device] = None,
                 save_dir: str = 'checkpoints',
                 log_dir: Optional[str] = None):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to train on (auto-detect if None)
            save_dir: Directory to save checkpoints
            log_dir: Directory for tensorboard logs
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
        else:
            self.writer = None
        
        # Move model to device
        self.model.to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
    
    def train_epoch(self, 
                   train_loader: DataLoader, 
                   criterion: nn.Module, 
                   optimizer: optim.Optimizer) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch + 1} - Training')
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            total_correct += pred.eq(target).sum().item()
            total_samples += target.size(0)
            
            # Update progress bar
            avg_loss = total_loss / (batch_idx + 1)
            accuracy = 100. * total_correct / total_samples
            pbar.set_postfix({
                'Loss': f'{avg_loss:.4f}',
                'Acc': f'{accuracy:.2f}%'
            })
        
        return total_loss / len(train_loader), 100. * total_correct / total_samples
    
    def validate_epoch(self, 
                      val_loader: DataLoader, 
                      criterion: nn.Module) -> Tuple[float, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            criterion: Loss function
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f'Epoch {self.current_epoch + 1} - Validation')
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = criterion(output, target)
                
                # Statistics
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                total_correct += pred.eq(target).sum().item()
                total_samples += target.size(0)
                
                # Update progress bar
                avg_loss = total_loss / len(pbar)
                accuracy = 100. * total_correct / total_samples
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{accuracy:.2f}%'
                })
        
        return total_loss / len(val_loader), 100. * total_correct / total_samples
    
    def save_checkpoint(self, 
                       filepath: str, 
                       optimizer: optim.Optimizer, 
                       scheduler: Optional[LRScheduler] = None,
                       is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'model_info': self.model.get_model_info() if hasattr(self.model, 'get_model_info') else {}
        }
        
        if scheduler:
            checkpoint['scheduler_state_dict'] = scheduler.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_filepath = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(checkpoint, best_filepath)
    
    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader] = None,
              epochs: int = 500,
              criterion: Optional[nn.Module] = None,
              optimizer: Optional[optim.Optimizer] = None,
              scheduler: Optional[str] = 'plateau',
              scheduler_kwargs: Optional[Dict] = None,
              early_stopping: Optional[Dict] = None,
              save_every: int = 50,
              save_best: bool = True) -> Dict:
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            criterion: Loss function (default: CrossEntropyLoss)
            optimizer: Optimizer (default: Adam)
            scheduler: Learning rate scheduler type
            scheduler_kwargs: Arguments for scheduler
            early_stopping: Early stopping configuration
            save_every: Save checkpoint every N epochs
            save_best: Whether to save best model
            
        Returns:
            Training history dictionary
        """
        # Default criterion and optimizer
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        # Initialize scheduler
        lr_scheduler = None
        if scheduler:
            scheduler_kwargs = scheduler_kwargs or {}
            lr_scheduler = LRScheduler(optimizer, scheduler, **scheduler_kwargs)
        
        # Initialize early stopping
        early_stopper = None
        if early_stopping:
            early_stopper = EarlyStopping(**early_stopping)
        
        print(f"Training on device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        if hasattr(self.model, 'get_model_info'):
            model_info = self.model.get_model_info()
            print(f"Parameters: {model_info['trainable_parameters']:,}")
        
        start_time = time.time()
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Training
            train_loss, train_acc = self.train_epoch(train_loader, criterion, optimizer)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validation
            val_loss, val_acc = 0.0, 0.0
            if val_loader:
                val_loss, val_acc = self.validate_epoch(val_loader, criterion)
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(val_acc)
            
            # Learning rate
            current_lr = optimizer.param_groups[0]['lr']
            self.history['lr'].append(current_lr)
            
            # Scheduler step
            if lr_scheduler:
                if val_loader:
                    lr_scheduler.step(val_loss)
                else:
                    lr_scheduler.step()
            
            # Logging
            if self.writer:
                self.writer.add_scalar('Loss/Train', train_loss, epoch)
                self.writer.add_scalar('Accuracy/Train', train_acc, epoch)
                self.writer.add_scalar('LearningRate', current_lr, epoch)
                
                if val_loader:
                    self.writer.add_scalar('Loss/Validation', val_loss, epoch)
                    self.writer.add_scalar('Accuracy/Validation', val_acc, epoch)
            
            # Print progress
            if val_loader:
                print(f'Epoch {epoch + 1}/{epochs} - '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - '
                      f'LR: {current_lr:.6f}')
            else:
                print(f'Epoch {epoch + 1}/{epochs} - '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - '
                      f'LR: {current_lr:.6f}')
            
            # Save best model
            is_best = False
            if val_loader and save_best and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                is_best = True
            
            # Save checkpoint
            if (epoch + 1) % save_every == 0 or is_best:
                checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
                self.save_checkpoint(checkpoint_path, optimizer, lr_scheduler, is_best)
            
            # Early stopping
            if early_stopper:
                if early_stopper(val_loss if val_loader else train_loss):
                    print(f'Early stopping at epoch {epoch + 1}')
                    break
        
        # Final checkpoint
        final_path = os.path.join(self.save_dir, 'final_model.pth')
        self.save_checkpoint(final_path, optimizer, lr_scheduler)
        
        # Close tensorboard writer
        if self.writer:
            self.writer.close()
        
        training_time = time.time() - start_time
        print(f'Training completed in {training_time:.2f} seconds')
        
        # Save history
        history_path = os.path.join(self.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        return self.history
    
    def load_checkpoint(self, 
                       filepath: str, 
                       optimizer: Optional[optim.Optimizer] = None,
                       scheduler: Optional[LRScheduler] = None) -> Dict:
        """Load training checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if scheduler and 'scheduler_state_dict' in checkpoint:
            scheduler.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.history = checkpoint.get('history', {})
        
        return checkpoint
