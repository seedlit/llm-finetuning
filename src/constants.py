# Data paths
PARQUET_DATA_PATH = "./alephalpha_data/synthetic/part.1.parquet"

# Project directories
PROJECT_ROOT = "./llm-finetuning"
DATA_DIR = f"{PROJECT_ROOT}/data"
RAW_DATA_DIR = f"{DATA_DIR}/raw"
PROCESSED_DATA_DIR = f"{DATA_DIR}/processed"
SAMPLE_DATA_DIR = f"{DATA_DIR}/sample"

# Model and experiment directories
MODELS_DIR = f"{PROJECT_ROOT}/outputs"
EXPERIMENTS_DIR = f"{PROJECT_ROOT}/experiments"
CONFIGS_DIR = f"{PROJECT_ROOT}/configs"

# Prompt type mappings (from the Aleph Alpha documentation)
PROMPT_TYPES = {
    0: "Rephrasing",
    1: "Summarization",
    2: "Rephrasing in Wikipedia style",
    3: "Formulating questions",
    4: "Extracting lists",
}

# Default sample sizes
DEFAULT_SAMPLE_SIZES = [100, 500, 1000, 5000]

# MLflow settings
MLFLOW_TRACKING_URI = f"{EXPERIMENTS_DIR}/mlruns"
EXPERIMENT_NAME = "aleph_alpha_german_pipeline"

SAMPLE_DATA_FORMATS = ["parquet", "csv"]

# Default model configurations for M1 Mac
RECOMMENDED_MODELS = {
    "small": {
        "name": "microsoft/DialoGPT-small",
        "params": "117M",
        "description": "Small conversational model",
    },
    "gpt2": {"name": "gpt2", "params": "124M", "description": "General purpose model"},
    "distilgpt2": {
        "name": "distilgpt2",
        "params": "82M",
        "description": "Fastest, most memory efficient",
    },
    "medium": {
        "name": "microsoft/DialoGPT-medium",
        "params": "345M",
        "description": "Larger model (needs more RAM)",
    },
}

# Training constants
TRAIN_TEST_SPLIT_RATIO = 0.2
RANDOM_SEED = 42
DEFAULT_MAX_LENGTH = 512
PREVIEW_ROWS = 10000  # For data exploration
CONSOLE_WIDTH = 50  # For logging separators
DEFAULT_OUTPUT_FORMAT = "parquet"
