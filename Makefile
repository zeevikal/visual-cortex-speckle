.PHONY: help install train eval test clean setup-dev
.DEFAULT_GOAL := help

# Variables
PYTHON = python
PIP = pip
PROJECT_NAME = visual-cortex-speckle-recognition

help: ## Show this help message
	@echo "Visual Cortex Speckle Recognition - Available Commands:"
	@echo "======================================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	$(PIP) install -r requirements.txt
	$(PIP) install pytest black flake8 mypy jupyter

setup: ## Set up the project (install + create configs)
	$(MAKE) install
	$(PYTHON) src/main.py --mode create_configs
	@echo "Project setup complete! Configuration templates created in configs/"

train: ## Train model with default configuration
	$(PYTHON) src/main.py --mode train

train-basic: ## Train model on basic shapes (Circle, Rectangle, Triangle)
	$(PYTHON) src/main.py --mode train --config configs/basic_shapes.yaml

train-kfold: ## Train model with K-fold cross-validation
	$(PYTHON) src/main.py --mode train_kfold --config configs/kfold_convlstm.yaml

train-kfold-basic: ## Train model with K-fold cross-validation on basic shapes
	$(PYTHON) src/main.py --mode train_kfold --config configs/kfold_basic_shapes.yaml

train-kfold-custom: ## Train with K-fold and custom parameters (usage: make train-kfold-custom K_FOLDS=10 EPOCHS=50)
	$(PYTHON) src/main.py --mode train_kfold \
		$(if $(K_FOLDS),--k_folds $(K_FOLDS)) \
		$(if $(EPOCHS),--epochs $(EPOCHS)) \
		$(if $(BATCH_SIZE),--batch_size $(BATCH_SIZE)) \
		$(if $(LR),--learning_rate $(LR)) \
		$(if $(SUBJECTS),--subjects $(SUBJECTS)) \
		$(if $(SHAPES),--shapes $(SHAPES)) \
		$(if $(STRATIFIED),--kfold_stratified)

train-loso: ## Train model with Leave-One-Subject-Out cross-validation
	$(PYTHON) src/main.py --mode train_loso --config configs/loso_convlstm.yaml

train-loso-basic: ## Train model with LOSO cross-validation on basic shapes
	$(PYTHON) src/main.py --mode train_loso --config configs/loso_basic_shapes.yaml

train-loso-custom: ## Train with LOSO and custom parameters (usage: make train-loso-custom EPOCHS=50 SUBJECTS="Subject1 Subject2")
	$(PYTHON) src/main.py --mode train_loso \
		$(if $(EPOCHS),--epochs $(EPOCHS)) \
		$(if $(BATCH_SIZE),--batch_size $(BATCH_SIZE)) \
		$(if $(LR),--learning_rate $(LR)) \
		$(if $(SUBJECTS),--subjects $(SUBJECTS)) \
		$(if $(SHAPES),--shapes $(SHAPES)) \
		$(if $(LOSO_SEED),--loso_random_state $(LOSO_SEED))

train-custom: ## Train with custom parameters (usage: make train-custom EPOCHS=100 BATCH_SIZE=64)
	$(PYTHON) src/main.py --mode train \
		$(if $(EPOCHS),--epochs $(EPOCHS)) \
		$(if $(BATCH_SIZE),--batch_size $(BATCH_SIZE)) \
		$(if $(LR),--learning_rate $(LR)) \
		$(if $(SUBJECTS),--subjects $(SUBJECTS)) \
		$(if $(SHAPES),--shapes $(SHAPES))

eval: ## Evaluate trained model (usage: make eval MODEL_PATH=checkpoints/best_model.pth)
	$(PYTHON) src/main.py --mode eval --model_path $(MODEL_PATH)

test: ## Run tests
	$(PYTHON) -m pytest tests/ -v

test-coverage: ## Run tests with coverage
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=html --cov-report=term

lint: ## Run code linting
	flake8 src/ tests/
	black --check src/ tests/
	mypy src/

format: ## Format code
	black src/ tests/

notebook: ## Run jupyter notebook
	jupyter notebook notebooks/

demo: ## Run the PyTorch conversion demo
	$(PYTHON) notebooks/pytorch_conversion_demo.py

clean: ## Clean generated files
	rm -rf __pycache__/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf checkpoints*/
	rm -rf logs*/
	rm -rf results*/
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete

clean-models: ## Clean only model files and results
	rm -rf checkpoints*/
	rm -rf logs*/
	rm -rf results*/

package: ## Create a package distribution
	$(PYTHON) setup.py sdist bdist_wheel

install-package: ## Install the package in development mode
	$(PIP) install -e .

# Data processing commands
process-videos: ## Process raw video data (usage: make process-videos DATA_PATH=../RawData/)
	$(PYTHON) -c "
	from src.data.preprocessing import VideoProcessor
	import pickle
	processor = VideoProcessor()
	data = processor.process_directory('$(DATA_PATH)')
	with open('processed_data.pickle', 'wb') as f:
		pickle.dump(data, f)
	print('Data processed and saved to processed_data.pickle')
	"

# Quick start commands
quickstart: ## Quick start: setup + train basic shapes
	$(MAKE) setup
	@echo "Starting training on basic shapes..."
	$(MAKE) train-basic

quickstart-kfold: ## Quick start: setup + K-fold training on basic shapes
	$(MAKE) setup
	@echo "Starting K-fold cross-validation on basic shapes..."
	$(MAKE) train-kfold-basic

# Testing K-fold functionality
test-kfold: ## Test K-fold cross-validation functionality
	$(PYTHON) test_kfold.py

# Examples and help
examples-kfold: ## Show K-fold training examples
	@echo "K-fold Cross-Validation Examples:"
	@echo "================================="
	@echo ""
	@echo "Basic K-fold training:"
	@echo "  make train-kfold"
	@echo "  make train-kfold-basic"
	@echo ""
	@echo "Custom K-fold parameters:"
	@echo "  make train-kfold-custom K_FOLDS=10 EPOCHS=50 BATCH_SIZE=32"
	@echo ""
	@echo "K-fold with specific subjects/shapes:"
	@echo "  python src/main.py --mode train_kfold --subjects ZeevKal --shapes Circle Rectangle Triangle --k_folds 5"
	@echo ""
	@echo "Evaluate K-fold best model:"
	@echo "  make eval MODEL_PATH=checkpoints/best_kfold_model.pth"
	@echo ""
	@echo "Test K-fold functionality:"
	@echo "  make test-kfold"

# Docker commands (if using Docker)
docker-build: ## Build Docker image
	docker build -t $(PROJECT_NAME) .

docker-run: ## Run container
	docker run -it --rm -v $(PWD):/workspace $(PROJECT_NAME)

# Development helpers
check: ## Run all checks (lint + test)
	$(MAKE) lint
	$(MAKE) test

pre-commit: ## Run pre-commit checks
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test

# Example training commands
examples: ## Show example training commands
	@echo "Example Training Commands:"
	@echo "=========================="
	@echo "1. Basic shapes training:"
	@echo "   make train-basic"
	@echo ""
	@echo "2. Custom training with parameters:"
	@echo "   make train-custom EPOCHS=200 BATCH_SIZE=64 LR=0.0005"
	@echo ""
	@echo "3. Training specific subjects and shapes:"
	@echo '   make train-custom SUBJECTS="ZeevKal Yevgeny" SHAPES="Circle Rectangle"'
	@echo ""
	@echo "4. Evaluation:"
	@echo "   make eval MODEL_PATH=checkpoints/best_model.pth"
	@echo ""
	@echo "5. Processing raw videos:"
	@echo "   make process-videos DATA_PATH=../RawData/15032022/"

info: ## Show project information
	@echo "Visual Cortex Speckle Recognition"
	@echo "================================="
	@echo "A PyTorch implementation for shape recognition using speckle imaging patterns"
	@echo ""
	@echo "Key Features:"
	@echo "- PyTorch-based 1D CNN for speckle pattern classification"
	@echo "- Modular and extensible architecture"
	@echo "- Configuration-based training pipeline"
	@echo "- Comprehensive evaluation and visualization tools"
	@echo "- Support for multiple subjects and shape types"
	@echo ""
	@echo "Supported Shapes:"
	@echo "- Basic: Circle, Rectangle, Triangle"
	@echo "- Multi: M_Circle, M_Rectangle, M_Triangle"
	@echo "- Custom shape patterns"
	@echo ""
	@echo "For more information, see README.md and USAGE.md"

# Interpretability and explainability analysis
interpretability: ## Run comprehensive interpretability analysis (SHAP, cortical mapping)
	$(PYTHON) examples/interpretability_analysis.py

interpret-shap: ## Run SHAP analysis only
	$(PYTHON) -c "
	from examples.interpretability_analysis import main
	import sys
	sys.argv = ['interpretability_analysis.py', '--shap-only']
	main()
	"

interpret-notebook: ## Launch interpretability analysis Jupyter notebook
	jupyter notebook notebooks/interpretability_analysis.ipynb

interpret-custom: ## Run interpretability with custom config (usage: make interpret-custom CONFIG=configs/interpretability.yaml)
	$(PYTHON) examples/interpretability_analysis.py \
		$(if $(CONFIG),--config $(CONFIG)) \
		$(if $(MODEL_PATH),--model_path $(MODEL_PATH)) \
		$(if $(SAVE_DIR),--save_dir $(SAVE_DIR))

install-interpretability: ## Install interpretability dependencies (SHAP, Captum, etc.)
	$(PIP) install shap captum lime umap-learn networkx plotly
	@echo "Interpretability packages installed successfully!"
