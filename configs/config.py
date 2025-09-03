"""
Configuration settings for the Aleph Alpha German pipeline.
"""

# Sampling configuration
SAMPLING_CONFIG = {
    "default_sample_size": 1000,
    "random_state": 42,
    "stratified_sampling": True,
    "min_samples_per_prompt": 10,
    "preview_rows": 10000,  # For data exploration
}

# Data preprocessing configuration
PREPROCESSING_CONFIG = {
    "max_text_length": 10000,
    "min_text_length": 50,
    "remove_duplicates": True,
    "normalize_whitespace": True,
    "filter_languages": ["de"],  # German only
}

# LLM Training configuration
LLM_CONFIG = {
    "default_model": "microsoft/DialoGPT-small",
    "max_length": 512,
    "train_test_split": 0.2,
    "random_seed": 42,
    "use_lora": True,
    "lora_config": {
        "r": 8,
        "lora_alpha": 32,
        "target_modules": ["c_attn", "c_proj", "c_fc"],
        "lora_dropout": 0.1,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
}

# Training parameters
TRAINING_CONFIG = {
    "num_epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "logging_steps": 10,
    "save_steps": 500,
    "eval_steps": 100,
    "max_grad_norm": 1.0,
    "weight_decay": 0.01,
    "dataloader_num_workers": 0,  # 0 for M1 Mac compatibility
}

# Text generation configuration
GENERATION_CONFIG = {
    "max_new_tokens": 150,
    "temperature": 0.8,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": None,  # Will be set dynamically
}

# Model configuration (legacy - keeping for backwards compatibility)
MODEL_CONFIG = {
    "model_name": "german_text_classifier",
    "max_length": 512,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "warmup_steps": 100,
}

# MLflow configuration
MLFLOW_CONFIG = {
    "experiment_name": "aleph_alpha_german_pipeline",
    "run_name_prefix": "run",
    "log_artifacts": True,
    "log_model": True,
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "filename": "pipeline.log",
}
