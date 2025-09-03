"""
Generate a sample dataset from the large Aleph Alpha GermanWeb synthetic dataset.

This script creates stratified samples to ensure representation across different prompt types.
"""

import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from configs.config import SAMPLING_CONFIG
from src import constants
from src.constants import (
    CONSOLE_WIDTH,
    DEFAULT_OUTPUT_FORMAT,
    DEFAULT_SAMPLE_SIZES,
    PREVIEW_ROWS,
    RANDOM_SEED,
    SAMPLE_DATA_DIR,
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SampleDatasetGenerator:
    """Generate sample datasets from the full Aleph Alpha GermanWeb data."""

    def __init__(self, data_path: str):
        """
        Initialize the sample generator.

        Args:
            data_path: Path to the full parquet file
        """
        self.data_path = data_path
        self.output_dir = Path(SAMPLE_DATA_DIR)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_full_dataset_info(self) -> dict:
        """Load basic information about the full dataset without loading it entirely."""
        logger.info(f"Loading dataset info from {self.data_path}")

        # Read parquet metadata
        parquet_file = pq.ParquetFile(self.data_path)
        total_rows = parquet_file.metadata.num_rows

        logger.info(f"Total rows in dataset: {total_rows:,}")

        # Sample a small portion to understand the data structure
        sample_df = (
            pq.read_table(self.data_path, columns=["prompt_id"])
            .to_pandas()
            .head(PREVIEW_ROWS)
        )

        prompt_distribution = sample_df["prompt_id"].value_counts().sort_index()
        logger.info(f"Prompt type distribution (first {PREVIEW_ROWS:,} rows):")
        for prompt_id, count in prompt_distribution.items():
            logger.info(f"  Prompt {prompt_id}: {count} rows")

        return {
            "total_rows": total_rows,
            "columns": parquet_file.schema.names,
            "prompt_distribution": prompt_distribution,
        }

    def generate_stratified_sample(
        self,
        sample_size: int = SAMPLING_CONFIG["default_sample_size"],
        random_state: int = RANDOM_SEED,
    ) -> pd.DataFrame:
        """
        Generate a stratified sample ensuring representation across prompt types.

        Args:
            sample_size: Total number of rows to sample
            random_state: Random seed for reproducibility

        Returns:
            Sampled DataFrame
        """
        logger.info(f"Generating stratified sample of {sample_size} rows")

        # First, let's understand the prompt distribution
        logger.info("Reading prompt_id column to understand distribution...")
        prompt_df = pq.read_table(self.data_path, columns=["prompt_id"]).to_pandas()

        prompt_counts = prompt_df["prompt_id"].value_counts().sort_index()
        logger.info("Full dataset prompt distribution:")
        for prompt_id, count in prompt_counts.items():
            logger.info(
                f"  Prompt {prompt_id}: {count:,} rows ({count / len(prompt_df) * 100:.1f}%)"
            )

        # Calculate sample sizes per prompt type (proportional)
        sample_per_prompt = {}
        remaining_samples = sample_size

        for prompt_id in sorted(prompt_counts.index):
            if (
                prompt_id == sorted(prompt_counts.index)[-1]
            ):  # Last prompt gets remaining
                sample_per_prompt[prompt_id] = remaining_samples
            else:
                proportion = prompt_counts[prompt_id] / len(prompt_df)
                samples = max(1, int(sample_size * proportion))  # At least 1 sample
                sample_per_prompt[prompt_id] = samples
                remaining_samples -= samples

        logger.info("Sampling strategy:")
        for prompt_id, samples in sample_per_prompt.items():
            logger.info(f"  Prompt {prompt_id}: {samples} samples")

        # Now sample from each prompt type
        sampled_indices = []
        np.random.seed(random_state)

        for prompt_id, num_samples in sample_per_prompt.items():
            prompt_indices = prompt_df[
                prompt_df["prompt_id"] == prompt_id
            ].index.tolist()
            if len(prompt_indices) < num_samples:
                logger.warning(
                    f"Not enough samples for prompt {prompt_id}. "
                    f"Requested: {num_samples}, Available: {len(prompt_indices)}"
                )
                selected = prompt_indices
            else:
                selected = np.random.choice(
                    prompt_indices, size=num_samples, replace=False
                )

            sampled_indices.extend(selected)

        # Sort indices for more efficient reading
        sampled_indices.sort()
        logger.info(f"Selected {len(sampled_indices)} total indices")

        # Read the full data and filter by indices
        logger.info("Reading full dataset...")
        full_df = pq.read_table(self.data_path).to_pandas()

        # Sample the data
        sample_df = full_df.iloc[sampled_indices].reset_index(drop=True)

        logger.info(f"Generated sample with {len(sample_df)} rows")
        logger.info("Sample prompt distribution:")
        sample_prompt_counts = sample_df["prompt_id"].value_counts().sort_index()
        for prompt_id, count in sample_prompt_counts.items():
            logger.info(f"  Prompt {prompt_id}: {count} rows")

        return sample_df

    def generate_random_sample(
        self,
        sample_size: int = SAMPLING_CONFIG["default_sample_size"],
        random_state: int = RANDOM_SEED,
    ) -> pd.DataFrame:
        """
        Generate a simple random sample (faster but may not be representative).

        Args:
            sample_size: Number of rows to sample
            random_state: Random seed for reproducibility

        Returns:
            Sampled DataFrame
        """
        logger.info(f"Generating random sample of {sample_size} rows")

        # Read the full dataset
        logger.info("Reading full dataset...")
        full_df = pq.read_table(self.data_path).to_pandas()

        # Sample randomly
        sample_df = full_df.sample(
            n=sample_size, random_state=random_state
        ).reset_index(drop=True)

        logger.info(f"Generated sample with {len(sample_df)} rows")
        logger.info("Sample prompt distribution:")
        sample_prompt_counts = sample_df["prompt_id"].value_counts().sort_index()
        for prompt_id, count in sample_prompt_counts.items():
            logger.info(f"  Prompt {prompt_id}: {count} rows")

        return sample_df

    def save_sample(
        self,
        sample_df: pd.DataFrame,
        filename: str = "sample_1k.parquet",
        format: str = DEFAULT_OUTPUT_FORMAT,
    ) -> Path:
        """
        Save the sample dataset to disk.

        Args:
            sample_df: The sampled DataFrame
            filename: Output filename
            format: Output format ('parquet', 'csv', 'json')

        Returns:
            Path to the saved file
        """
        output_path = self.output_dir / filename

        logger.info(f"Saving sample to {output_path}")

        if format == "parquet":
            sample_df.to_parquet(output_path, index=False)
        elif format == "csv":
            output_path = output_path.with_suffix(".csv")
            sample_df.to_csv(output_path, index=False)
        elif format == "json":
            output_path = output_path.with_suffix(".json")
            sample_df.to_json(output_path, orient="records", lines=True)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Sample saved successfully to {output_path}")
        logger.info(f"File size: {output_path.stat().st_size / (1024 * 1024):.2f} MB")

        return output_path

    def generate_multiple_samples(self):
        """Generate multiple sample sizes for different use cases."""
        sample_sizes = DEFAULT_SAMPLE_SIZES

        for size in sample_sizes:
            logger.info(f"\n{'=' * CONSOLE_WIDTH}")
            logger.info(f"Generating sample of size {size}")
            logger.info(f"{'=' * CONSOLE_WIDTH}")

            sample_df = self.generate_stratified_sample(sample_size=size)

            # Save in multiple formats
            for fmt in constants.SAMPLE_DATA_FORMATS:
                filename = f"sample_{size}.{fmt}"
                self.save_sample(sample_df, filename, fmt)

            # Generate basic statistics
            self.generate_sample_stats(sample_df, f"sample_{size}_stats.txt")

    def generate_sample_stats(self, sample_df: pd.DataFrame, stats_filename: str):
        """Generate and save basic statistics about the sample."""
        stats_path = self.output_dir / stats_filename

        with open(stats_path, "w") as f:
            f.write("Sample Dataset Statistics\n")
            f.write("=" * CONSOLE_WIDTH + "\n\n")

            f.write(f"Total rows: {len(sample_df):,}\n")
            f.write(f"Total columns: {len(sample_df.columns)}\n\n")

            f.write("Columns:\n")
            for col in sample_df.columns:
                f.write(f"  - {col}\n")
            f.write("\n")

            f.write("Prompt Type Distribution:\n")
            prompt_counts = sample_df["prompt_id"].value_counts().sort_index()
            for prompt_id, count in prompt_counts.items():
                percentage = count / len(sample_df) * 100
                f.write(f"  Prompt {prompt_id}: {count:,} rows ({percentage:.1f}%)\n")
            f.write("\n")

            f.write("Text Length Statistics:\n")
            text_lengths = sample_df["text"].str.len()
            f.write(f"  Mean: {text_lengths.mean():.1f} characters\n")
            f.write(f"  Median: {text_lengths.median():.1f} characters\n")
            f.write(f"  Min: {text_lengths.min()} characters\n")
            f.write(f"  Max: {text_lengths.max()} characters\n")
            f.write(f"  Std: {text_lengths.std():.1f} characters\n")

        logger.info(f"Statistics saved to {stats_path}")


def main():
    """Main function to generate sample datasets."""
    # Initialize the generator
    generator = SampleDatasetGenerator(constants.PARQUET_DATA_PATH)

    # First, get dataset info
    info = generator.load_full_dataset_info()
    print(f"\nDataset contains {info['total_rows']:,} total rows")
    print(f"Columns: {info['columns']}")

    # Generate multiple sample sizes
    generator.generate_multiple_samples()

    print("\nSample datasets generated successfully!")
    print("Check the 'data/sample/' directory for the generated files.")

    # Also generate a quick exploration sample
    logger.info("\nGenerating a quick exploration sample (100 rows)...")
    quick_sample = generator.generate_random_sample(sample_size=100, random_state=42)
    generator.save_sample(quick_sample, "quick_explore.parquet")

    print("\nSample generation complete!")


if __name__ == "__main__":
    main()
