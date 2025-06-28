# Visual Cortex Speckle Imaging for Shape Recognition

This repository implements a deep learning approach for shape recognition using visual cortex speckle imaging patterns. The project uses PyTorch to build Convolutional LSTM (ConvLSTM) networks that classify geometric shapes based on speckle imaging data, following the methodology described in our research paper. [Paper's link]()

## Abstract

This study introduces a non‑invasive approach for neurovisual classification of geometric shapes by capturing and decoding laser‑speckle patterns reflected from the human striate cortex. Using a fast digital camera and deep neural networks (DNN), we demonstrate that each visual stimulus - rectangle, triangle, circle, mixed shapes, or blank screen- arouses a uniquely distinguishable speckle signature. Our optimized DNN classifier achieved perfect recall (100 %) for rectangles and high recall (90 %) for triangles in single‑shape trials and sustained robust performance (80 % recall) when multiple shapes appeared simultaneously. Even complex multi‑shape and white‑screen controls were classified with exceptional reliability, underscoring the method’s sensitivity and generalizability. While circular stimuli produced subtler speckle dynamics, the results highlight clear avenues for refining the detection of curved geometries. By leveraging low‑cost optics and scalable AI processing, this technique paves the way for real‑time, portable monitoring of visual cortex activity, offering transformative potential for cognitive neuroscience, brain-machine interfaces, and clinical assessment of visual processing. Future work will expand stimulus complexity, optimize model architectures, and explore multimodal neurophotonic applications.

## Key Features

- **Advanced Deep Learning Models**: PyTorch implementation of ConvLSTM for speckle pattern classification
- **Temporal Pattern Analysis**: ConvLSTM architecture that captures both spatial and temporal dependencies in speckle patterns
- **Data Processing Pipeline**: Comprehensive tools for processing video files and extracting sequential speckle patterns
- **Multiple Feature Extraction**: Various distance metrics (Manhattan, Euclidean, Normalized Cross-Correlation)
- **Comprehensive Evaluation**: Advanced visualization and analysis tools for model performance
- **Multi-subject Support**: Framework for handling data from multiple subjects and experimental conditions
- **Configuration-based Training**: Flexible YAML-based configuration system for reproducible experiments

## Repository Structure

```
visual-cortex-speckle/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py          # Dataset classes and data loading
│   │   └── preprocessing.py    # Video processing and feature extraction
│   ├── models/
│   │   ├── __init__.py
│   │   ├── convlstm.py        # ConvLSTM architecture (default)
│   │   ├── conv1d.py          # 1D CNN architecture
│   │   └── base_model.py      # Base model class
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py         # Training pipeline
│   │   └── evaluation.py      # Model evaluation and metrics
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── visualization.py   # Plotting and visualization tools
│   │   └── config.py          # Configuration management
│   └── main.py                # Main entry point
├── configs/
│   ├── default.yaml           # Default ConvLSTM configuration
│   ├── basic_shapes.yaml      # Basic shapes configuration
│   └── convlstm.yaml         # Specific ConvLSTM configuration
├── data/                      # Data directory
├── models/                    # Saved model checkpoints
├── results/                   # Training results and plots
├── notebooks/                 # Jupyter notebooks for analysis
├── tests/                     # Unit tests
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── Makefile                   # Build automation
└── README.md                  # This file
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone git@github.com:zeevikal/visual-cortex-speckle.git
cd visual-cortex-speckle

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Or use make for quick setup
make setup
```

### 2. Training with Default ConvLSTM Configuration

```bash
# Train with default ConvLSTM settings
python src/main.py --mode train

# Train with custom configuration
python src/main.py --mode train --config configs/convlstm.yaml

# Quick start with make
make train
```

### 3. Training with Command Line Overrides

```bash
# Override specific parameters
python src/main.py --mode train \
    --config configs/convlstm.yaml \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 0.001

# Use make with custom parameters
make train-custom EPOCHS=50 BATCH_SIZE=64 LR=0.001
```

### 4. Data-specific Training

```bash
# Train on specific subjects and shapes
python src/main.py --mode train \
    --subjects ZeevKal Yevgeny \
    --shapes Circle Rectangle Triangle \
    --feature_type manhattan

# Train on multi-shapes
python src/main.py --mode train \
    --subjects ZeevKal \
    --shapes M_Circle M_Rectangle M_Triangle \
    --feature_type euclidean

# Train on basic shapes with make
make train-basic
```

### 5. Model Evaluation

```bash
# Evaluate a trained model
python src/main.py --mode eval \
    --model_path checkpoints/best_model.pth \
    --config configs/convlstm.yaml

# Use make for evaluation
make eval MODEL_PATH=checkpoints/best_model.pth
```

## Model Architecture

![alt text](figures/Fig4.png)

### ConvLSTM Architecture (Default)

The default model uses a Convolutional LSTM architecture as described in our research paper:

**ConvLSTM Features:**
- **Temporal Dependencies**: Captures both spatial and temporal patterns in sequential speckle data
- **ConvLSTM Cell**: Implements the full ConvLSTM equations with forget, input, and output gates
- **Spatial Preservation**: Maintains spatial dimensions while processing temporal sequences
- **Paper-based Configuration**: 64 hidden states, 3×3 kernels, processing 64 input frames

**Architecture Components:**
- **Input**: Sequential speckle pattern arrays (normalized 1×4096 tensors representing 64×64 frames)
- **ConvLSTM Layer**: Single layer with 64 hidden states and 3×3 state-to-state kernels
- **Classification Head**: Fully connected layer with 256 units and ReLU activation
- **Output**: Softmax classification for shape recognition

**ConvLSTM Equations:**
```
i_t = σ(W_xi * X_t + W_hi * H_{t-1} + W_ci ∘ C_{t-1} + b_i)
f_t = σ(W_xf * X_t + W_hf * H_{t-1} + W_cf ∘ C_{t-1} + b_f)
C_t = f_t ∘ C_{t-1} + i_t ∘ tanh(W_xc * X_t + W_hc * H_{t-1} + b_c)
o_t = σ(W_xo * X_t + W_ho * H_{t-1} + W_co ∘ C_t + b_o)
H_t = o_t ∘ tanh(C_t)
```

### Alternative 1D CNN Architecture

For comparison, a traditional 1D CNN architecture is also available:

- **Input**: 1D time series of speckle pattern differences
- **Conv1D Layers**: 3 convolutional layers with 64 filters each
- **Batch Normalization**: Applied after each convolution
- **Global Average Pooling**: Reduces spatial dimensions
- **Dense Layer**: Final classification layer with softmax activation

## Datasets

The project supports classification of:

1. **Basic Shapes**: Circle, Rectangle, Triangle
2. **Multi-shapes**: Multiple instances of basic shapes (M_Circle, M_Rectangle, M_Triangle)
3. **Extended Shapes**: Additional geometric patterns
4. **Sequential Patterns**: Temporal speckle patterns for ConvLSTM analysis

## Data Preparation

### 1. Directory Structure

Place your video files in the following structure:
```
data/raw/
├── subject1/
│   ├── Circle/
│   ├── Rectangle/
│   └── Triangle/
└── subject2/
    ├── Circle/
    ├── Rectangle/
    └── Triangle/
```

### 2. Process Video Data

```python
from src.data.preprocessing import VideoProcessor
from src.data.dataset import create_data_loaders_from_pickle

# Process raw video data
processor = VideoProcessor(frames_limit=20000)
speckle_data = processor.process_directory('../RawData/15032022/')

# Create data loaders from processed data
train_loader, val_loader, test_loader, data_loader = create_data_loaders_from_pickle(
    pickle_path='../Data/vis_cortex_data.pickle',
    subjects=['ZeevKal'],
    shape_filter=['Circle', 'Rectangle', 'Triangle'],
    feature_type='manhattan',
    batch_size=64
)
```

### 3. Using Make for Data Processing

```bash
# Process videos from raw data
make process-videos DATA_PATH=../RawData/
```

## Python API Usage

### Basic Training Script

```python
from src.utils.config import load_config
from src.main import train_model

# Load configuration
config = load_config('configs/convlstm.yaml')

# Train model
model, history, evaluator = train_model(config)
```

### ConvLSTM Model Creation and Training

```python
import torch
from src.models.convlstm import ConvLSTMClassifier
from src.training.trainer import Trainer

# Create ConvLSTM model (default)
model = ConvLSTMClassifier(
    num_classes=3,
    hidden_size=64,
    sequence_length=64,
    dropout_rate=0.1
)

# Create trainer
trainer = Trainer(
    model=model,
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    save_dir='checkpoints',
    log_dir='logs'
)

# Train with ConvLSTM-specific settings
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=50,  # ConvLSTM typically converges faster
    early_stopping={'patience': 10}
)
```

### Model Evaluation

```python
from src.training.evaluation import ModelEvaluator

# Load trained ConvLSTM model
model = ConvLSTMClassifier.load_model('checkpoints/best_model.pth')[0]

# Create evaluator
evaluator = ModelEvaluator(
    model=model,
    class_names=['Circle', 'Rectangle', 'Triangle']
)

# Evaluate
results = evaluator.evaluate(test_loader)
print(f"Accuracy: {results['accuracy']:.4f}")

# Save comprehensive evaluation report
evaluator.save_evaluation_report(
    test_loader,
    save_dir='results/evaluation',
    dataset_name='test'
)
```

### Visualization

```python
from src.utils.visualization import (
    plot_training_history,
    plot_sample_data,
    create_visualization_report
)

# Plot training history
plot_training_history(
    history,
    save_path='results/training_history.png'
)

# Plot sample data
plot_sample_data(
    x_test, y_test,
    class_names=['Circle', 'Rectangle', 'Triangle'],
    save_path='results/sample_patterns.png'
)

# Create comprehensive visualization report
create_visualization_report(
    model=model,
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    history=history,
    class_names=['Circle', 'Rectangle', 'Triangle'],
    save_dir='results/visualizations'
)
```

## Advanced Usage

### Custom ConvLSTM Architecture

```python
from src.models.base_model import BaseSpeckleModel, ModelFactory
from src.models.convlstm import ConvLSTMCell
import torch.nn as nn

class CustomConvLSTMModel(BaseSpeckleModel):
    def __init__(self, num_classes, **kwargs):
        super().__init__(num_classes)
        # Define your custom ConvLSTM architecture
        self.convlstm = ConvLSTMCell(
            input_channels=1,
            hidden_channels=128,  # Custom hidden size
            kernel_size=5,        # Custom kernel size
            padding=2
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Custom forward pass
        batch_size, seq_len, channels, height = x.size()
        h, c = self.init_hidden(batch_size, height)
        
        for t in range(seq_len):
            h, c = self.convlstm(x[:, t], h, c)
        
        # Global average pooling and classification
        out = torch.mean(h, dim=2)  # Average over spatial dimension
        return self.classifier(out)

# Register custom model
ModelFactory.register_model('custom_convlstm', CustomConvLSTMModel)
```

### Hyperparameter Search

```python
import itertools
from src.utils.config import get_default_config

# Define hyperparameter grid for ConvLSTM
param_grid = {
    'learning_rate': [0.001, 0.0005, 0.0001],
    'batch_size': [32, 64, 128],
    'hidden_size': [32, 64, 128],
    'dropout_rate': [0.0, 0.1, 0.2],
    'sequence_length': [32, 64, 96]
}

best_accuracy = 0
best_params = None

for params in itertools.product(*param_grid.values()):
    # Create config with current parameters
    config = get_default_config()
    config.training.learning_rate = params[0]
    config.training.batch_size = params[1]
    config.model.hidden_size = params[2]
    config.model.dropout_rate = params[3]
    config.model.sequence_length = params[4]
    config.model.model_type = 'convlstm'
    
    # Train and evaluate
    model, history, evaluator = train_model(config)
    results = evaluator.evaluate(test_loader)
    
    if results['accuracy'] > best_accuracy:
        best_accuracy = results['accuracy']
        best_params = dict(zip(param_grid.keys(), params))

print(f"Best accuracy: {best_accuracy}")
print(f"Best parameters: {best_params}")
```

## Configuration Reference

### Data Configuration
- `root_path`: Path to raw video data
- `pickle_path`: Path to processed pickle file
- `subjects`: List of subjects to include (null for all)
- `shape_filter`: List of shapes to include (null for all)
- `feature_type`: Feature extraction method ("manhattan", "euclidean", "ncc")
- `train_size`: Fraction of data for training
- `n_chunks`: Number of chunks to split each video
- `frames_limit`: Maximum frames to process per video

### Model Configuration
- `model_type`: Model architecture ("convlstm", "conv1d", "enhanced_convlstm")
- `num_classes`: Number of output classes
- `hidden_size`: Number of hidden units in ConvLSTM (default: 64)
- `sequence_length`: Length of input sequences (default: 64)
- `kernel_size`: ConvLSTM kernel size (default: 3)
- `dropout_rate`: Dropout rate
- `num_layers`: Number of ConvLSTM layers (for enhanced model)

### Training Configuration
- `batch_size`: Training batch size (recommended: 64 for ConvLSTM)
- `epochs`: Maximum number of epochs (ConvLSTM typically needs 50)
- `learning_rate`: Initial learning rate (default: 0.001)
- `optimizer`: Optimizer type ("adam", "sgd")
- `scheduler`: Learning rate scheduler ("plateau", "step", "cosine")
- `validation_split`: Fraction for validation
- `early_stopping_patience`: Patience for early stopping
- `save_every`: Save checkpoint every N epochs
- `save_best`: Whether to save best model

## Make Commands

The repository includes a comprehensive Makefile for easy project management:

```bash
# Setup and installation
make setup              # Complete project setup
make install            # Install dependencies
make install-dev        # Install with development dependencies

# Training commands
make train              # Train with default ConvLSTM configuration
make train-basic        # Train on basic shapes
make train-custom EPOCHS=50 BATCH_SIZE=64 LR=0.001  # Custom parameters

# Evaluation and testing
make eval MODEL_PATH=checkpoints/best_model.pth     # Evaluate model
make test               # Run unit tests
make test-coverage      # Run tests with coverage

# Data processing
make process-videos DATA_PATH=../RawData/           # Process raw videos

# Development
make lint               # Code linting
make format             # Code formatting
make clean              # Clean generated files

# Examples and demos
make demo               # Run ConvLSTM demo
make examples           # Show example commands
```

## File Structure After Training

```
visual-cortex-speckle/
├── checkpoints/
│   ├── best_model.pth           # Best ConvLSTM model
│   ├── final_model.pth          # Final training state
│   └── checkpoint_epoch_*.pth   # Periodic checkpoints
├── logs/
│   └── tensorboard_logs/        # TensorBoard training logs
├── results/
│   ├── test_classification_report.csv
│   ├── test_confusion_matrix.png
│   ├── test_analysis.json
│   └── visualizations/
│       ├── training_history.png
│       ├── sample_patterns.png
│       ├── convlstm_feature_maps.png
│       └── temporal_analysis.png
└── training_history.json        # Complete training metrics
```

## Results

#### 2.1 Detection of Speckle-Based Visual Cortex Reaction 

![alt text](figures/Fig1.png)

This study represents the first experimental setup designed to classify the speckle patterns reflected from the human visual cortex in response to visual stimuli. The experimental framework, as seen in Fig. 1, aimed to validate whether the laser speckle pattern imaging, combined with DNN data processing, could reliably classify the neural-visual brain activity caused by the observation of different geometric shapes.
Before each recording session, participants were shown a blank white screen to establish a baseline speckle pattern, ensuring the stability of neural responses and minimizing external noise. The speckle patterns were continuously recorded as the tested subjects viewed different shape stimuli. The captured videos were processed and analyzed to determine whether distinct neural responses could be extracted for different visual stimuli.
The results confirmed that the visual cortex speckle pattern imaging classification could differentiate between a structured and blank screen visual input. The model successfully classified polygonal shapes with higher accuracy, while circular shapes posed a greater challenge with low accuracy, as shown in the confusion matrix, Fig. 2. These findings validate the feasibility of using speckle-based imaging as a non-invasive technique to monitor visual cortex activity, setting the foundation for future advancements in the laser-based detection of neural activity. 

![alt text](figures/Fig2.png)

#### 2.2 Single-Shape Classification

![alt text](figures/Fig3.png)

The research was subdivided into several tasks. The first task evaluated the model’s ability to classify speckle patterns generated from the neural responses to single-shape stimuli. Each video presented the participants with a sequence of TV screens containing one geometric shape. A single circle, triangle, or rectangle was consequently displayed on a white background. This setup ensured that neural activity, detected through the speckle pattern classification, was primarily driven by shape recognition without interference from additional visual elements.
A sample set of the three video types shown to each participant is given in Fig. 3a. The video, displaying a circle, did not exhibit any noticeable speckle pattern variations, setting it apart from other shapes. The resulting poor classification accuracy is shown in Table 1. The validation set accuracy remained close to random, and the test set results indicated limited generalization capability.
The confusion matrix, Fig. 2, highlights the model failure to classify circles correctly (0% recall), whereas rectangles received the highest recall (100%) and F1-score (77%), triangles with a recall of 90% and an F1-score of 69%. These findings indicate that the neural response associated with polygonal shapes, particularly rectangles, was more distinguishable in the speckles pattern data, as shown in Table 1.

#### 2.3 Multi-Shape Classification
The second task tested the model’s ability to classify speckle patterns corresponding to visual stimuli containing multiple instances of the same geometric shape. In each video, participants were shown several copies of a single shape, either triangles, rectangles, or circles, randomly positioned across the screen. This setup was designed to examine whether the repetition of a shape altered neural responses and whether the model could still differentiate between shape categories when multiple instances were simultaneously introduced. The random positioning ensured that the entire visual field was engaged, reducing potential biases related to the spatial location.
A sample set of these videos is shown in Fig. 3b. The validation set accuracy varied, likely due to noise in the speckle signals. The test set accuracy reached 55%, with the rectangles achieving the highest recall (60%) and F1-score (71%), as summarized in Table 1.
for other classes, such as multi-triangles and multi-circles, was significantly lower, with multi-triangles achieving only 10% recall and multi-circles failing to achieve any correct predictions (0% recall). 

#### 2.4 Mixed-Shape Classification
The third task assessed the model’s ability to classify speckle patterns generated from visual stimuli containing multiple geometric shapes presented simultaneously in each screen frame. Unlike previous tasks, where stimuli consisted of either a single shape or multiple instances of the same shape, this task introduced frames in which a random number of circles, triangles, and rectangles appeared together in random positions. This setup aimed to evaluate the neural response to complex visual scenes where different shapes coexisted within a single stimulus, testing the model’s capacity to generalize across varying shape compositions.
A sample set of these videos is shown in Fig. 3c. As expected, the speckle signals generated from white-screen videos without shapes differ significantly from those of the mixed-shape videos. The presence of multiple shape types in each frame introduced additional complexity, requiring the model to distinguish the overlapping neural responses.
Test results showed that white-screen videos were classified with 100% recall, confirming the model’s ability to reliably detect the absence of structured visual information. Mixed-shape videos also achieved strong performance, with a recall of 80% and an F1-score of 76%, as summarized in Table 1. These findings suggest that while the neural response to complex visual stimuli introduces variability, the model was still able to extract distinguishing features from the speckle pattern data, likely leveraging shape-specific neural signals captured in previous tasks.

#### 2.5 General Observations
Before each recording session, the participants were shown a blank white screen to calibrate the session. This step ensured a stable baseline for neural activity assessment, minimizing variability in the speckle patterns caused by environmental factors or physiological fluctuations. The white-screen calibration also helped normalize the data, allowing for more accurate comparisons between shape-induced neural responses across different trials.
We found that the overall model’s classification performance consistently favored polygonal shapes, particularly rectangles with the highest recall and F1-scores in both single-shape and multi-shape tasks. The model struggled with circles, as shown in the confusion matrix, Fig. 2, where circular stimuli were often misclassified or failed to produce distinct speckle pattern responses. This trend suggests a potential correlation between circles and the blank white screen condition, implying that non-polygonal shapes may not elicit strong neural responses in the primary visual cortex (V1). This could indicate that the visual system processes circles differently or that the absence of edges in circular stimuli results in a weaker speckle signal, resembling the neural response to the blank screen. Further investigation is needed to explore this potential link and to better understand how the brain encodes different geometric forms.
The classification performance for mixed-shape and white-screen videos was notably high, indicating that the model effectively differentiated between the presence or absence of structured visual stimuli. However, the relatively poor performance on multi-triangle and multi-circle stimuli suggests that increased complexity in the neural response may reduce classification accuracy.
These findings underscore the potential of laser speckle pattern imaging in sensing the neural dynamics related to shape recognition. However, further research is needed to refine classification models, improve sensitivity to non-polygonal shapes, and explore how neural processing of complex visual stimuli influences speckle pattern response. Future work should also investigate the role of higher-order visual processing mechanisms and their effect on the speckle patterns to enhance the robustness and interpretability of this approach.


## Citation

If you use this code in your research, please cite:

```bibtex
@article{visual_cortex_speckle_2025,
  title={Visual Cortex Speckle Imaging for Shape Recognition},
  author={[Author Names]},
  journal={[Journal Name]},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Contact

For questions and support, please contact [zeevkal@biu.ac.il]
