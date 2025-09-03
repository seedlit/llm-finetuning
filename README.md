# LLM Fine-tuning Pipeline

A complete pipeline for fine-tuning Language Models on German text using [Aleph Alpha's synthetic dataset](https://huggingface.co/datasets/Aleph-Alpha/Aleph-Alpha-GermanWeb). Optimized for **Apple M1 Macs** with memory-efficient training and **MLflow experiment tracking**.

## What This Does

**Fine-tunes existing LLMs** (not training from scratch):
- Takes pre-trained models like `microsoft/DialoGPT-small` or `gpt2`
- Fine-tunes them on German synthetic text data
- Uses **LoRA (Low-Rank Adaptation)** for efficient training on M1 Macs
- **Tracks experiments with MLflow** for reproducibility and comparison
- Produces models that can generate German text in different styles

**The model learns to predict the next word** in German text, giving it capabilities like:
- Text completion
- German text generation
- Style-aware generation (different prompt types from the dataset)

## Dataset Information

Using **Aleph Alpha's German synthetic dataset** with 5 prompt types:

| Prompt ID | Type | Description | Count |
|-----------|------|-------------|-------|
| 0 | Rephrasing | Text rephrasing tasks | 995K (26.5%) |
| 1 | Summarization | Text summarization | 219K (5.8%) |
| 2 | Wikipedia Style | Rephrasing in Wikipedia style | 964K (25.7%) |
| 3 | Questions | Formulating questions | 810K (21.6%) |
| 4 | Lists | Extracting lists | 770K (20.5%) |

**Total**: 3.7M samples, 2.5GB parquet file

## Quick Start

### 1. Setup Environment

```bash
# Clone and setup
git clone <your-repo>
cd llm-finetuning

# Install dependencies with uv (includes dev tools)
uv sync

# Install pre-commit hooks for code quality
uv run pre-commit install
```

### 2. Download Dataset

```bash
# Download the synthetic dataset (2.5GB)
uv run hf download Aleph-Alpha/Aleph-Alpha-GermanWeb --repo-type dataset --include "*synthetic/part.1.parquet*" --local-dir <download-dir>
```

### 3. Generate Sample Data

Start with smaller samples for testing:

```bash
# Generate stratified samples (100, 500, 1K, 5K rows)
uv run src/data_preprocessing/generate_sample_dataset.py
```

### 4. Training with MLflow Tracking

```bash
# Train on 1000 samples with MLflow tracking (10-15 minutes)
uv run src/llm_pipeline.py \
  --data data/sample/sample_1000.parquet \
  --model microsoft/DialoGPT-small \
  --epochs 2

# Train without MLflow tracking
uv run src/llm_pipeline.py \
  --data data/sample/sample_1000.parquet \
  --model microsoft/DialoGPT-small \
  --epochs 2 \
  --no_mlflow
```

### 5. View Experiments in MLflow

```bash
# Start MLflow UI (accessible at http://localhost:5000)
uv run mlflow ui --backend-store-uri experiments/mlruns

# Or specify a different port
uv run mlflow ui --backend-store-uri experiments/mlruns --port 5000
```

### 6. Testing the model
```bash
# Test the trained model
uv run src/models/test_llm.py \
  --model_path outputs/german_llm_microsoft_DialoGPT-small \
  --mode test

# Chat with your model (interactive)
uv run src/models/test_llm.py \
  --model_path outputs/german_llm_microsoft_DialoGPT-small \
  --mode chat

# Get model info
uv run src/models/test_llm.py \
  --model_path outputs/german_llm_microsoft_DialoGPT-small \
  --mode info
```

## Training Options

### Available Models

**Recommended for M1 Mac (tested):**
```bash
# Small and fast
--model microsoft/DialoGPT-small    # 117M params, conversational
--model gpt2                        # 124M params, general purpose
--model distilgpt2                  # 82M params, fastest

# Larger (if you have 16GB+ RAM)
--model microsoft/DialoGPT-medium   # 345M params
--model gpt2-medium                 # 345M params
```

### Training Parameters

All training parameters are now centralized in `configs/config.py` for easy modification:

```bash
# Basic training with default config
uv run src/llm_pipeline.py --data data/sample/sample_1000.parquet

# Override specific parameters
uv run src/llm_pipeline.py \
  --data data/sample/sample_5000.parquet \
  --model microsoft/DialoGPT-small \
  --epochs 3 \
  --batch_size 4 \
  --learning_rate 2e-4
```

**Default Configuration Values:**
- **Model**: `microsoft/DialoGPT-small` (117M parameters)
- **Epochs**: 3
- **Batch Size**: 4 (optimized for M1 Mac)
- **Learning Rate**: 2e-4
- **Max Length**: 512 tokens
- **LoRA**: Enabled (memory efficient fine-tuning)


## MLflow Experiment Tracking

### Experiment Organization

MLflow automatically organizes experiments:

```
experiments/mlruns/
├── german_llm_finetuning/          # Experiment name
│   ├── run_20240903_143022/        # Individual training runs
│   │   ├── params/                 # Training parameters
│   │   ├── metrics/                # Training metrics
│   │   ├── artifacts/              # Model files & logs
│   │   └── meta.yaml              # Run metadata
│   └── run_20240903_145511/        # Another training run
└── .../
```

## Directory Structure

```
llm-finetuning/
├── src/
│   ├── data_preprocessing/
│   │   └── generate_sample_dataset.py    # Create sample datasets
│   ├── models/
│   │   ├── llm_trainer.py                # Core LLM training logic
│   │   └── test_llm.py                   # Model testing & inference
│   ├── utils/
│   │   ├── helpers.py                    # Utility functions
│   │   └── mlflow_utilities.py           # MLflow tracking utilities
│   ├── llm_pipeline.py                   # Main training pipeline
│   └── constants.py                      # Project-wide constants
├── configs/
│   └── config.py                         # Centralized configuration
├── data/
│   └── sample/                           # Generated sample datasets
├── outputs/                              # Saved trained models
├── experiments/                          # MLflow experiments & tracking
│   └── mlruns/                           # MLflow run data & artifacts
├── tests/                                # Unit tests
└── pyproject.toml                        # Dependencies & tool config
```

## Configuration Management

The pipeline now uses centralized configuration for better maintainability:

- **`src/constants.py`**: Project paths, model recommendations, and static values
- **`configs/config.py`**: Training parameters, model configurations, and runtime settings

### Key Configuration Sections

```python
# LLM model settings
LLM_CONFIG = {
    "default_model": "microsoft/DialoGPT-small",
    "max_length": 512,
    "use_lora": True,
    # ... LoRA configuration
}

# Training parameters
TRAINING_CONFIG = {
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-4,
    # ... other training settings
}

# Sampling configuration
SAMPLING_CONFIG = {
    "default_sample_size": 1000,
    "stratified_sampling": True,
    # ... sampling parameters
}
```

## Technical Details

### What Happens During Training

1. **Load Data**: Sample German text from Aleph Alpha dataset
2. **Tokenize**: Convert text to tokens the model understands
3. **Fine-tune**: Update model weights to predict German text better
4. **Save**: Store the fine-tuned model for later use

### LoRA (Low-Rank Adaptation)

We use LoRA for efficient training:
- **Faster**: Only trains ~1% of model parameters
- **Memory Efficient**: Perfect for M1 Macs
- **Quality**: 95%+ of full fine-tuning performance
- **Flexible**: Easy to swap different LoRA adapters

### Memory Usage

**Estimated RAM usage:**
- `DialoGPT-small`: ~2-4GB
- `DialoGPT-medium`: ~6-8GB
- `GPT2-medium`: ~6-8GB

**M1 Mac recommendations:**
- 8GB RAM: Use small models, batch_size=2
- 16GB RAM: Use medium models, batch_size=4
- 32GB+ RAM: Any model, larger batch sizes

## Usage Examples


## Model Testing

### Test Mode
```bash
# Run predefined tests
uv run src/models/test_llm.py --model_path outputs/your_model --mode test
```

### Chat Mode (Interactive)
```bash
# Chat with your model
uv run src/models/test_llm.py --model_path outputs/your_model --mode chat

# Example interaction:
> Enter prompt: Die deutsche Sprache ist
> Model: sehr komplex und hat viele grammatische Regeln...
```

### Info Mode
```bash
# Get model information
uv run src/models/test_llm.py --model_path outputs/your_model --mode info
```

## Next Steps

1. **Experiment with hyperparameters** using MLflow tracking
2. **Compare different models** and training configurations
3. **Deploy the model** for inference
4. **Scale up training** with larger datasets

## Understanding the Output

When training completes, you'll see:

```
======================================================================
LLM TRAINING PIPELINE SUMMARY
======================================================================
Status: SUCCESS
Model: microsoft/DialoGPT-small
Training Samples: 400
Final Loss: 8.3107
Total Time: 425.3s
Model Path: ./outputs/german_llm_microsoft_DialoGPT-small

Next Steps:
   1. Test your model:
      uv run src/models/test_llm.py --model_path ./outputs/your_model
   2. View MLflow experiments:
      uv run mlflow ui --backend-store-uri experiments/mlruns
   3. Try different prompts and see how it performs!
======================================================================
```

**MLflow automatically tracks:**
- All training parameters and hyperparameters
- Training and validation loss over time
- Model artifacts and metadata
- Training duration and system metrics

Access the **MLflow UI** to compare runs and analyze training performance!
