"""
Configuration management for the speckle imaging project.
"""

import yaml
import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class DataConfig:
    """Data configuration parameters."""
    root_path: str = "../RawData/"
    pickle_path: str = "../Data/vis_cortex_data.pickle"
    subjects: Optional[list] = None
    shape_filter: Optional[list] = None
    feature_type: str = "manhattan"
    train_size: float = 0.5
    n_chunks: int = 10
    frames_limit: int = 20000


@dataclass
class ModelConfig:
    """Model configuration parameters."""
    model_type: str = "conv1d"
    num_classes: int = 3
    input_channels: int = 1
    num_filters: int = 64
    kernel_size: int = 3
    dropout_rate: float = 0.0


@dataclass
class TrainingConfig:
    """Training configuration parameters."""
    batch_size: int = 32
    epochs: int = 500
    learning_rate: float = 0.001
    optimizer: str = "adam"
    scheduler: str = "plateau"
    validation_split: float = 0.2
    early_stopping_patience: int = 100
    save_every: int = 50
    save_best: bool = True
    
    # K-fold cross-validation parameters
    use_kfold: bool = False
    k_folds: int = 5
    kfold_stratified: bool = True
    kfold_random_state: int = 42


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    mode: str = "min"
    factor: float = 0.5
    patience: int = 20
    min_lr: float = 0.0001
    verbose: bool = True


@dataclass
class Config:
    """Main configuration class."""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    scheduler: SchedulerConfig
    
    # Paths
    save_dir: str = "checkpoints"
    log_dir: str = "logs"
    results_dir: str = "results"
    
    # Device
    device: str = "auto"  # "auto", "cpu", "cuda"
    
    # Random seed
    seed: int = 42
    
    # Logging
    verbose: bool = True


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Configuration object
    """
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create configuration objects
    data_config = DataConfig(**config_dict.get('data', {}))
    model_config = ModelConfig(**config_dict.get('model', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    scheduler_config = SchedulerConfig(**config_dict.get('scheduler', {}))
    
    # Create main config
    main_config_dict = {k: v for k, v in config_dict.items() 
                       if k not in ['data', 'model', 'training', 'scheduler']}
    
    config = Config(
        data=data_config,
        model=model_config,
        training=training_config,
        scheduler=scheduler_config,
        **main_config_dict
    )
    
    return config


def save_config(config: Config, config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration object
        config_path: Path to save the YAML configuration file
    """
    config_dict = asdict(config)
    
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def get_default_config() -> Config:
    """
    Get default configuration.
    
    Returns:
        Default configuration object
    """
    return Config(
        data=DataConfig(),
        model=ModelConfig(),
        training=TrainingConfig(),
        scheduler=SchedulerConfig()
    )


def create_config_templates(output_dir: str = "configs"):
    """
    Create template configuration files.
    
    Args:
        output_dir: Directory to save template configs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Default config
    default_config = get_default_config()
    save_config(default_config, os.path.join(output_dir, "default.yaml"))
    
    # Config for 3 basic shapes
    basic_shapes_config = get_default_config()
    basic_shapes_config.data.shape_filter = ["Rectangle", "Circle", "Triangle"]
    basic_shapes_config.data.subjects = ["ZeevKal"]
    basic_shapes_config.data.n_chunks = 100
    basic_shapes_config.model.num_classes = 3
    save_config(basic_shapes_config, os.path.join(output_dir, "basic_shapes.yaml"))
    
    # Config for multi shapes
    multi_shapes_config = get_default_config()
    multi_shapes_config.data.shape_filter = ["M_Rectangle", "M_Circle", "M_Triangle"]
    multi_shapes_config.data.subjects = ["ZeevKal"]
    multi_shapes_config.data.n_chunks = 200
    multi_shapes_config.model.num_classes = 3
    save_config(multi_shapes_config, os.path.join(output_dir, "multi_shapes.yaml"))
    
    # Config for all data (single subject)
    all_data_config = get_default_config()
    all_data_config.data.subjects = ["Yevgeny"]
    all_data_config.data.n_chunks = 10
    all_data_config.model.num_classes = 5  # Adjust based on your data
    save_config(all_data_config, os.path.join(output_dir, "all_data_single_subject.yaml"))
    
    print(f"Configuration templates created in {output_dir}/")


def update_config_from_args(config: Config, args: Dict[str, Any]) -> Config:
    """
    Update configuration with command line arguments.
    
    Args:
        config: Base configuration
        args: Dictionary of arguments to update
        
    Returns:
        Updated configuration
    """
    # Update data config
    for key, value in args.items():
        if hasattr(config.data, key):
            setattr(config.data, key, value)
        elif hasattr(config.model, key):
            setattr(config.model, key, value)
        elif hasattr(config.training, key):
            setattr(config.training, key, value)
        elif hasattr(config.scheduler, key):
            setattr(config.scheduler, key, value)
        elif hasattr(config, key):
            setattr(config, key, value)
    
    return config


def print_config(config: Config):
    """
    Print configuration in a readable format.
    
    Args:
        config: Configuration to print
    """
    print("=" * 50)
    print("CONFIGURATION")
    print("=" * 50)
    
    print("\nData Configuration:")
    print(f"  Root Path: {config.data.root_path}")
    print(f"  Pickle Path: {config.data.pickle_path}")
    print(f"  Subjects: {config.data.subjects}")
    print(f"  Shape Filter: {config.data.shape_filter}")
    print(f"  Feature Type: {config.data.feature_type}")
    print(f"  Train Size: {config.data.train_size}")
    print(f"  Chunks: {config.data.n_chunks}")
    print(f"  Frames Limit: {config.data.frames_limit}")
    
    print("\nModel Configuration:")
    print(f"  Model Type: {config.model.model_type}")
    print(f"  Number of Classes: {config.model.num_classes}")
    print(f"  Input Channels: {config.model.input_channels}")
    print(f"  Number of Filters: {config.model.num_filters}")
    print(f"  Kernel Size: {config.model.kernel_size}")
    print(f"  Dropout Rate: {config.model.dropout_rate}")
    
    print("\nTraining Configuration:")
    print(f"  Batch Size: {config.training.batch_size}")
    print(f"  Epochs: {config.training.epochs}")
    print(f"  Learning Rate: {config.training.learning_rate}")
    print(f"  Optimizer: {config.training.optimizer}")
    print(f"  Scheduler: {config.training.scheduler}")
    print(f"  Validation Split: {config.training.validation_split}")
    print(f"  Early Stopping Patience: {config.training.early_stopping_patience}")
    print(f"  Use K-Fold: {config.training.use_kfold}")
    print(f"  K-Folds: {config.training.k_folds}")
    print(f"  Stratified K-Fold: {config.training.kfold_stratified}")
    print(f"  K-Fold Random State: {config.training.kfold_random_state}")
    
    print("\nPaths:")
    print(f"  Save Directory: {config.save_dir}")
    print(f"  Log Directory: {config.log_dir}")
    print(f"  Results Directory: {config.results_dir}")
    
    print(f"\nDevice: {config.device}")
    print(f"Seed: {config.seed}")
    print("=" * 50)
