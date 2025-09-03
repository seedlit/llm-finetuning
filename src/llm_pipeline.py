"""
Simplified LLM Training Pipeline for German Text.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import time

from configs.config import LLM_CONFIG, TRAINING_CONFIG
from src import constants
from src.models.llm_trainer import train_german_llm
from src.utils.helpers import setup_logging

logger = setup_logging()


class LLMPipeline:
    """Simplified pipeline focused on LLM training."""

    def __init__(
        self,
        data_path: str,
        model_name: str = LLM_CONFIG["default_model"],
        max_samples: int = None,
        num_epochs: int = TRAINING_CONFIG["num_epochs"],
        batch_size: int = TRAINING_CONFIG["batch_size"],
        learning_rate: float = TRAINING_CONFIG["learning_rate"],
    ):
        """
        Initialize LLM pipeline.

        Args:
            data_path: Path to training data
            model_name: HuggingFace model to fine-tune
            max_samples: Maximum samples to use (None for all)
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
        """
        self.data_path = data_path
        self.model_name = model_name
        self.max_samples = max_samples
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.start_time = time.time()

    def run(self) -> dict[str, any]:
        """Run the complete LLM training pipeline."""
        logger.info("Starting LLM Training Pipeline")
        logger.info(f"   Data: {self.data_path}")
        logger.info(f"   Model: {self.model_name}")
        logger.info(f"   Max Samples: {self.max_samples or 'All'}")
        logger.info(f"   Epochs: {self.num_epochs}")
        logger.info(f"   Batch Size: {self.batch_size}")
        logger.info(f"   Learning Rate: {self.learning_rate}")

        try:
            # Train model
            results = train_german_llm(
                data_path=self.data_path,
                model_name=self.model_name,
                max_samples=self.max_samples,
                num_epochs=self.num_epochs,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
            )

            # Calculate total time
            total_time = time.time() - self.start_time

            pipeline_results = {
                "status": "success",
                "training_results": results,
                "total_time": total_time,
                "model_path": results["model_path"],
            }

            logger.info("Pipeline completed successfully!")
            logger.info(f"   Total time: {total_time:.1f} seconds")
            logger.info(f"   Model saved: {results['model_path']}")

            return pipeline_results

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "total_time": time.time() - self.start_time,
            }


def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description="German LLM Training Pipeline")
    parser.add_argument(
        "--data", required=True, help="Path to training data (parquet file)"
    )
    parser.add_argument(
        "--model", default=LLM_CONFIG["default_model"], help="HuggingFace model name"
    )
    parser.add_argument(
        "--max_samples", type=int, help="Maximum samples to use (default: all)"
    )
    parser.add_argument(
        "--use_full_dataset",
        action="store_true",
        help="Use the full 2.5GB dataset instead of samples",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAINING_CONFIG["num_epochs"],
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=TRAINING_CONFIG["batch_size"],
        help="Training batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=TRAINING_CONFIG["learning_rate"],
        help="Learning rate",
    )

    args = parser.parse_args()

    # Determine data path
    if args.use_full_dataset:
        data_path = constants.PARQUET_DATA_PATH  # Full 2.5GB file
        logger.info("Using full dataset (2.5GB)")
    else:
        data_path = args.data

    # Suggested models for M1 Mac
    if args.model == "auto":
        suggested_models = [
            "microsoft/DialoGPT-small",  # 117M params
            "gpt2",  # 124M params
            "distilgpt2",  # 82M params
            "microsoft/DialoGPT-medium",  # 345M params (if you have more RAM)
        ]

        print("Suggested models for M1 Mac:")
        for i, model in enumerate(suggested_models, 1):
            print(f"   {i}. {model}")

        choice = input(
            "\nEnter model number (1-4) or press Enter for default: "
        ).strip()
        if choice.isdigit() and 1 <= int(choice) <= 4:
            args.model = suggested_models[int(choice) - 1]
        else:
            args.model = suggested_models[0]

        print(f"Selected: {args.model}")

    # Initialize and run pipeline
    pipeline = LLMPipeline(
        data_path=data_path,
        model_name=args.model,
        max_samples=args.max_samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    results = pipeline.run()

    # Print summary
    print("\n" + "=" * 70)
    print("LLM TRAINING PIPELINE SUMMARY")
    print("=" * 70)

    if results["status"] == "success":
        training_stats = results["training_results"]["training_stats"]
        print("Status: SUCCESS")
        print(f"Model: {args.model}")
        print(f"Training Samples: {training_stats['num_train_samples']:,}")
        print(f"Final Loss: {training_stats['eval_loss']:.4f}")
        print(f"Total Time: {results['total_time']:.1f}s")
        print(f"Model Path: {results['model_path']}")

        print("\nNext Steps:")
        print("   1. Test your model:")
        print(
            f"      uv run src/models/test_llm.py --model_path {results['model_path']}"
        )
        print("   2. Try different prompts and see how it performs!")

    else:
        print("Status: FAILED")
        print(f"Error: {results.get('error', 'Unknown error')}")

    print("=" * 70)


if __name__ == "__main__":
    main()
