"""
Text preprocessing pipeline for German text data.
"""

import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

from src import constants
from src.utils.helpers import save_metadata, setup_logging

logger = setup_logging()


class GermanTextPreprocessor:
    """Preprocessing pipeline for German text data."""

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the preprocessor with configuration.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {
            "min_text_length": 50,
            "max_text_length": 10000,
            "remove_duplicates": True,
            "normalize_whitespace": True,
            "remove_special_chars": False,
            "lowercase": False,  # Keep original case for German
        }

    def clean_text(self, text: str) -> str:
        """
        Clean individual text string.

        Args:
            text: Input text string

        Returns:
            Cleaned text string
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""

        # Normalize whitespace
        if self.config.get("normalize_whitespace", True):
            # Replace multiple whitespace with single space
            text = re.sub(r"\s+", " ", text)
            # Remove leading/trailing whitespace
            text = text.strip()

        # Remove special characters (optional, careful with German)
        if self.config.get("remove_special_chars", False):
            # Keep German special characters (ä, ö, ü, ß)
            text = re.sub(r"[^\w\säöüßÄÖÜ\.,!?\-\(\)]", "", text)

        # Convert to lowercase (optional, not recommended for German)
        if self.config.get("lowercase", False):
            text = text.lower()

        return text

    def filter_by_length(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Filter texts by length criteria.

        Args:
            df: DataFrame with text column

        Returns:
            Filtered DataFrame and statistics
        """
        logger.info("Filtering texts by length...")

        initial_count = len(df)
        min_length = self.config.get("min_text_length", 50)
        max_length = self.config.get("max_text_length", 10000)

        # Calculate text lengths
        text_lengths = df["text"].str.len()

        # Apply filters
        length_filter = (text_lengths >= min_length) & (text_lengths <= max_length)
        filtered_df = df[length_filter].copy()

        stats = {
            "initial_count": initial_count,
            "final_count": len(filtered_df),
            "removed_count": initial_count - len(filtered_df),
            "removal_rate": (initial_count - len(filtered_df)) / initial_count * 100,
            "min_length_filter": min_length,
            "max_length_filter": max_length,
            "length_stats": {
                "mean": float(filtered_df["text"].str.len().mean()),
                "median": float(filtered_df["text"].str.len().median()),
                "std": float(filtered_df["text"].str.len().std()),
            },
        }

        logger.info(
            f"Filtered {stats['removed_count']} texts ({stats['removal_rate']:.1f}%)"
        )
        logger.info(f"Remaining: {stats['final_count']} texts")

        return filtered_df, stats

    def remove_duplicates(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Remove duplicate texts.

        Args:
            df: DataFrame with text column

        Returns:
            Deduplicated DataFrame and statistics
        """
        if not self.config.get("remove_duplicates", True):
            return df, {"duplicates_removed": 0}

        logger.info("Removing duplicate texts...")

        initial_count = len(df)

        # Remove exact duplicates
        df_dedup = df.drop_duplicates(subset=["text"]).copy()

        stats = {
            "initial_count": initial_count,
            "final_count": len(df_dedup),
            "duplicates_removed": initial_count - len(df_dedup),
            "duplication_rate": (initial_count - len(df_dedup)) / initial_count * 100,
        }

        logger.info(
            f"Removed {stats['duplicates_removed']} duplicates ({stats['duplication_rate']:.1f}%)"
        )

        return df_dedup, stats

    def validate_german_content(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Basic validation that content appears to be German.

        Args:
            df: DataFrame with text column

        Returns:
            Validated DataFrame and statistics
        """
        logger.info("Validating German content...")

        initial_count = len(df)

        # Simple heuristics for German text
        def looks_german(text: str) -> bool:
            if not isinstance(text, str) or len(text) < 20:
                return False

            text_lower = text.lower()

            # Common German words (simple check)
            german_indicators = [
                "der",
                "die",
                "das",
                "und",
                "ist",
                "in",
                "zu",
                "den",
                "von",
                "mit",
                "auf",
                "für",
                "als",
                "an",
                "wird",
                "im",
                "nicht",
                "oder",
                "auch",
                "sich",
                "nach",
                "werden",
                "bei",
                "aus",
                "um",
                "sie",
                "über",
            ]

            # Check for German characters
            has_german_chars = any(char in text for char in ["ä", "ö", "ü", "ß"])

            # Check for common German words
            words = text_lower.split()
            german_word_count = sum(1 for word in words if word in german_indicators)
            german_word_ratio = german_word_count / len(words) if words else 0

            # Text is likely German if it has German chars OR good German word ratio
            return has_german_chars or german_word_ratio > 0.1

        # Apply German validation
        german_mask = df["text"].apply(looks_german)
        validated_df = df[german_mask].copy()

        stats = {
            "initial_count": initial_count,
            "final_count": len(validated_df),
            "non_german_removed": initial_count - len(validated_df),
            "non_german_rate": (initial_count - len(validated_df))
            / initial_count
            * 100,
        }

        logger.info(
            f"Removed {stats['non_german_removed']} non-German texts ({stats['non_german_rate']:.1f}%)"
        )

        return validated_df, stats

    def process_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Apply full preprocessing pipeline to dataset.

        Args:
            df: Input DataFrame

        Returns:
            Processed DataFrame and comprehensive statistics
        """
        logger.info("Starting preprocessing pipeline...")

        initial_df = df.copy()
        processing_stats = {
            "initial_count": len(df),
            "initial_prompt_distribution": df["prompt_id"].value_counts().to_dict(),
        }

        # Step 1: Clean text
        logger.info("Step 1: Cleaning text...")
        df["text"] = df["text"].apply(self.clean_text)

        # Remove empty texts after cleaning
        df = df[df["text"].str.strip() != ""].copy()
        processing_stats["after_cleaning"] = len(df)

        # Step 2: Filter by length
        df, length_stats = self.filter_by_length(df)
        processing_stats["length_filtering"] = length_stats

        # Step 3: Remove duplicates
        df, dedup_stats = self.remove_duplicates(df)
        processing_stats["deduplication"] = dedup_stats

        # Step 4: Validate German content
        df, german_stats = self.validate_german_content(df)
        processing_stats["german_validation"] = german_stats

        # Final statistics
        processing_stats.update(
            {
                "final_count": len(df),
                "final_prompt_distribution": df["prompt_id"].value_counts().to_dict(),
                "total_removal_rate": (len(initial_df) - len(df))
                / len(initial_df)
                * 100,
                "pipeline_config": self.config,
            }
        )

        logger.info("Preprocessing complete!")
        logger.info(
            f"Initial: {processing_stats['initial_count']} → Final: {processing_stats['final_count']}"
        )
        logger.info(
            f"Total removal rate: {processing_stats['total_removal_rate']:.1f}%"
        )

        return df, processing_stats

    def save_processed_data(
        self, df: pd.DataFrame, stats: Dict, output_name: str
    ) -> Dict[str, str]:
        """
        Save processed dataset and metadata.

        Args:
            df: Processed DataFrame
            stats: Processing statistics
            output_name: Base name for output files

        Returns:
            Dictionary with output file paths
        """
        output_dir = Path(constants.PROCESSED_DATA_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save processed data
        data_path = output_dir / f"{output_name}.parquet"
        df.to_parquet(data_path, index=False)

        # Save metadata
        metadata_path = output_dir / f"{output_name}_metadata.json"
        save_metadata(stats, str(metadata_path))

        # Save text summary
        summary_path = output_dir / f"{output_name}_summary.txt"
        self._create_processing_summary(df, stats, summary_path)

        paths = {
            "data": str(data_path),
            "metadata": str(metadata_path),
            "summary": str(summary_path),
        }

        logger.info(f"Saved processed data to: {data_path}")
        logger.info(f"File size: {data_path.stat().st_size / (1024 * 1024):.2f} MB")

        return paths

    def _create_processing_summary(
        self, df: pd.DataFrame, stats: Dict, output_path: Path
    ):
        """Create a human-readable processing summary."""
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("Text Preprocessing Summary\n")
            f.write("=" * 50 + "\n\n")

            f.write(f"Initial dataset: {stats['initial_count']:,} samples\n")
            f.write(f"Final dataset: {stats['final_count']:,} samples\n")
            f.write(
                f"Total removed: {stats['initial_count'] - stats['final_count']:,} samples\n"
            )
            f.write(f"Removal rate: {stats['total_removal_rate']:.1f}%\n\n")

            f.write("Processing Steps:\n")
            f.write(f"1. After text cleaning: {stats['after_cleaning']:,} samples\n")
            f.write(
                f"2. After length filtering: {stats['length_filtering']['final_count']:,} samples\n"
            )
            f.write(
                f"3. After deduplication: {stats['deduplication']['final_count']:,} samples\n"
            )
            f.write(
                f"4. After German validation: {stats['german_validation']['final_count']:,} samples\n\n"
            )

            f.write("Final Prompt Distribution:\n")
            for prompt_id, count in sorted(stats["final_prompt_distribution"].items()):
                prompt_name = constants.PROMPT_TYPES.get(
                    prompt_id, f"Unknown ({prompt_id})"
                )
                percentage = count / stats["final_count"] * 100
                f.write(f"  {prompt_name}: {count:,} samples ({percentage:.1f}%)\n")

            f.write("\nText Length Statistics:\n")
            text_lengths = df["text"].str.len()
            f.write(f"  Mean: {text_lengths.mean():.1f} characters\n")
            f.write(f"  Median: {text_lengths.median():.1f} characters\n")
            f.write(f"  Min: {text_lengths.min()} characters\n")
            f.write(f"  Max: {text_lengths.max()} characters\n")
            f.write(f"  Std: {text_lengths.std():.1f} characters\n")


def preprocess_sample_data(sample_size: str = "1000"):
    """
    Preprocess sample data.

    Args:
        sample_size: Size of sample to preprocess (100, 500, 1000, 5000)
    """
    # Load sample data
    sample_path = Path(constants.SAMPLE_DATA_DIR) / f"sample_{sample_size}.parquet"

    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    logger.info(f"Loading sample data from {sample_path}")
    df = pd.read_parquet(sample_path)

    logger.info(f"Loaded {len(df)} samples for preprocessing")

    # Initialize preprocessor
    preprocessor = GermanTextPreprocessor()

    # Process data
    processed_df, stats = preprocessor.process_dataset(df)

    # Save processed data
    output_name = f"processed_sample_{sample_size}"
    paths = preprocessor.save_processed_data(processed_df, stats, output_name)

    logger.info("Preprocessing complete!")
    logger.info(f"Processed data saved to: {paths['data']}")

    return processed_df, stats, paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess German text data")
    parser.add_argument(
        "--sample_size",
        default="1000",
        help="Sample size to preprocess (100, 500, 1000, 5000)",
    )

    args = parser.parse_args()

    # Process the specified sample
    processed_df, stats, paths = preprocess_sample_data(args.sample_size)

    print("\nPreprocessing Summary:")
    print(f"Processed {len(processed_df):,} samples")
    print("Files saved:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
