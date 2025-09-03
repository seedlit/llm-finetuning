"""
Utility functions for the Aleph Alpha German pipeline.
"""

import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src import constants


def setup_logging(
    log_level: str = "INFO", log_file: Optional[str] = None
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path

    Returns:
        Configured logger
    """
    logger = logging.getLogger("aa_pipeline")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, log_level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def ensure_directories(*dirs: str) -> None:
    """
    Ensure that directories exist, create them if they don't.

    Args:
        *dirs: Directory paths to create
    """
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in MB.

    Args:
        file_path: Path to the file

    Returns:
        File size in MB
    """
    return Path(file_path).stat().st_size / (1024 * 1024)


def save_metadata(data: Dict, file_path: str) -> None:
    """
    Save metadata as JSON file.

    Args:
        data: Dictionary containing metadata
        file_path: Path to save the metadata file
    """
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def load_metadata(file_path: str) -> Dict:
    """
    Load metadata from JSON file.

    Args:
        file_path: Path to the metadata file

    Returns:
        Dictionary containing metadata
    """
    with open(file_path, "r") as f:
        return json.load(f)


def get_prompt_type_name(prompt_id: int) -> str:
    """
    Get the human-readable name for a prompt type.

    Args:
        prompt_id: The prompt ID (0-4)

    Returns:
        Human-readable prompt type name
    """
    return constants.PROMPT_TYPES.get(prompt_id, f"Unknown ({prompt_id})")


def analyze_text_data(df: pd.DataFrame, text_column: str = "text") -> Dict:
    """
    Analyze text data and return statistics.

    Args:
        df: DataFrame containing text data
        text_column: Name of the text column

    Returns:
        Dictionary with text statistics
    """
    text_lengths = df[text_column].str.len()

    stats = {
        "total_samples": len(df),
        "text_stats": {
            "mean_length": float(text_lengths.mean()),
            "median_length": float(text_lengths.median()),
            "min_length": int(text_lengths.min()),
            "max_length": int(text_lengths.max()),
            "std_length": float(text_lengths.std()),
        },
    }

    # Add prompt distribution if available
    if "prompt_id" in df.columns:
        prompt_dist = df["prompt_id"].value_counts().sort_index()
        stats["prompt_distribution"] = {
            int(prompt_id): {
                "count": int(count),
                "percentage": float(count / len(df) * 100),
                "type_name": get_prompt_type_name(prompt_id),
            }
            for prompt_id, count in prompt_dist.items()
        }

    return stats


def create_data_summary(df: pd.DataFrame, output_path: str) -> None:
    """
    Create a comprehensive data summary and save it.

    Args:
        df: DataFrame to analyze
        output_path: Path to save the summary
    """
    stats = analyze_text_data(df)

    summary_text = f"""Aleph Alpha German Dataset Summary
{"=" * 50}

Dataset Overview:
- Total samples: {stats["total_samples"]:,}
- Columns: {list(df.columns)}

Text Statistics:
- Mean length: {stats["text_stats"]["mean_length"]:.1f} characters
- Median length: {stats["text_stats"]["median_length"]:.1f} characters
- Min length: {stats["text_stats"]["min_length"]} characters
- Max length: {stats["text_stats"]["max_length"]:,} characters
- Standard deviation: {stats["text_stats"]["std_length"]:.1f} characters

"""

    if "prompt_distribution" in stats:
        summary_text += "Prompt Type Distribution:\n"
        for prompt_id, info in stats["prompt_distribution"].items():
            summary_text += f"- {info['type_name']} (ID {prompt_id}): {info['count']:,} samples ({info['percentage']:.1f}%)\n"
        summary_text += "\n"

    # Add sample texts
    summary_text += "Sample Texts:\n"
    summary_text += "-" * 20 + "\n"
    for i, row in df.head(3).iterrows():
        if "prompt_id" in row:
            prompt_name = get_prompt_type_name(row["prompt_id"])
            summary_text += f"\nPrompt Type: {prompt_name}\n"
        summary_text += f"Text (first 200 chars): {row['text'][:200]}...\n"
        summary_text += "-" * 20 + "\n"

    # Save summary
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary_text)

    # Also save as JSON
    json_path = str(Path(output_path).with_suffix(".json"))
    save_metadata(stats, json_path)


def validate_sample_data(df: pd.DataFrame) -> List[str]:
    """
    Validate sample data and return any issues found.

    Args:
        df: DataFrame to validate

    Returns:
        List of validation issues (empty if no issues)
    """
    issues = []

    # Check required columns
    required_columns = ["text", "prompt_id"]
    for col in required_columns:
        if col not in df.columns:
            issues.append(f"Missing required column: {col}")

    if not issues:  # Only check further if basic structure is correct
        # Check for empty texts
        empty_texts = df["text"].isna() | (df["text"].str.strip() == "")
        if empty_texts.any():
            issues.append(f"Found {empty_texts.sum()} empty text entries")

        # Check prompt_id values
        valid_prompt_ids = set(constants.PROMPT_TYPES.keys())
        invalid_prompts = ~df["prompt_id"].isin(valid_prompt_ids)
        if invalid_prompts.any():
            invalid_ids = df[invalid_prompts]["prompt_id"].unique()
            issues.append(f"Found invalid prompt_id values: {invalid_ids}")

        # Check text lengths
        text_lengths = df["text"].str.len()
        very_short = text_lengths < 10
        very_long = text_lengths > 100000

        if very_short.any():
            issues.append(f"Found {very_short.sum()} very short texts (< 10 chars)")
        if very_long.any():
            issues.append(f"Found {very_long.sum()} very long texts (> 100k chars)")

    return issues
