"""
LLM Fine-tuning for German Text Generation using LoRA.
Optimized for M1 Mac with memory-efficient training.
"""

import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from configs.config import GENERATION_CONFIG, LLM_CONFIG, TRAINING_CONFIG
from src import constants
from src.constants import RANDOM_SEED, TRAIN_TEST_SPLIT_RATIO
from src.utils.helpers import save_metadata, setup_logging

logger = setup_logging()


class GermanLLMTrainer:
    """Fine-tune small LLMs for German text generation."""

    def __init__(
        self,
        model_name: str = LLM_CONFIG["default_model"],
        max_length: int = LLM_CONFIG["max_length"],
        use_lora: bool = LLM_CONFIG["use_lora"],
    ):
        """
        Initialize the LLM trainer.

        Args:
            model_name: HuggingFace model name (small models for M1 Mac)
            max_length: Maximum sequence length
            use_lora: Whether to use LoRA for efficient fine-tuning
        """
        self.model_name = model_name
        self.max_length = max_length
        self.use_lora = use_lora

        # Check device - prioritize MPS for M1 Mac
        if torch.backends.mps.is_available():
            self.device = "mps"
            logger.info("Using Apple Silicon (MPS)")
        elif torch.cuda.is_available():
            self.device = "cuda"
            logger.info("Using CUDA")
        else:
            self.device = "cpu"
            logger.info("Using CPU")

        self.tokenizer = None
        self.model = None
        self.trainer = None

    def load_model_and_tokenizer(self):
        """Load model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Fix pad token issue properly
        if self.tokenizer.pad_token is None:
            if (
                hasattr(self.tokenizer, "unk_token")
                and self.tokenizer.unk_token is not None
            ):
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                # Add a proper pad token
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        elif self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            # If pad and eos are the same, try to fix it
            if (
                hasattr(self.tokenizer, "unk_token")
                and self.tokenizer.unk_token is not None
            ):
                self.tokenizer.pad_token = self.tokenizer.unk_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16 if self.device != "cpu" else torch.float32,
            device_map="auto" if self.device != "mps" else None,
            trust_remote_code=True,
        )

        # Move to device for MPS
        if self.device == "mps":
            self.model = self.model.to(self.device)

        # Setup LoRA if requested
        if self.use_lora:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                inference_mode=False,
                r=LLM_CONFIG["lora_config"]["r"],
                lora_alpha=LLM_CONFIG["lora_config"]["lora_alpha"],
                lora_dropout=LLM_CONFIG["lora_config"]["lora_dropout"],
                target_modules=LLM_CONFIG["lora_config"]["target_modules"],
                bias=LLM_CONFIG["lora_config"]["bias"],
            )
            self.model = get_peft_model(self.model, lora_config)
            logger.info("LoRA configuration applied")

        # Print trainable parameters
        if self.use_lora:
            trainable_params = sum(
                p.numel() for p in self.model.parameters() if p.requires_grad
            )
            total_params = sum(p.numel() for p in self.model.parameters())
            logger.info(
                f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.1f}%)"
            )

    def prepare_dataset(self, df: pd.DataFrame, prompt_template: str = None) -> Dataset:
        """
        Prepare dataset for training.

        Args:
            df: DataFrame with text and prompt_id columns
            prompt_template: Optional template for prompt formatting

        Returns:
            Tokenized dataset
        """
        logger.info(f"Preparing dataset with {len(df)} samples")

        # Create training texts based on prompt type
        training_texts = []

        for _, row in df.iterrows():
            prompt_type = constants.PROMPT_TYPES.get(row["prompt_id"], "Unknown")
            text = row["text"]

            # Create instruction-following format
            if prompt_template:
                formatted_text = prompt_template.format(
                    prompt_type=prompt_type, text=text
                )
            else:
                # Simple format: [PROMPT_TYPE] text
                formatted_text = f"[{prompt_type.upper()}] {text}"

            training_texts.append(formatted_text)

        # Create dataset
        dataset = Dataset.from_dict({"text": training_texts})

        # Tokenize
        def tokenize_function(examples):
            tokenized = self.tokenizer(
                examples["text"],
                truncation=True,
                padding=True,  # Enable padding
                max_length=self.max_length,
                return_tensors=None,
            )
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].copy()
            return tokenized

        tokenized_dataset = dataset.map(
            tokenize_function, batched=True, remove_columns=dataset.column_names
        )

        logger.info(f"Dataset tokenized: {len(tokenized_dataset)} samples")
        return tokenized_dataset

    def train(
        self,
        dataset: Dataset,
        output_dir: str,
        num_epochs: int = TRAINING_CONFIG["num_epochs"],
        batch_size: int = TRAINING_CONFIG["batch_size"],
        learning_rate: float = TRAINING_CONFIG["learning_rate"],
        warmup_steps: int = TRAINING_CONFIG["warmup_steps"],
        save_steps: int = TRAINING_CONFIG["save_steps"],
        logging_steps: int = TRAINING_CONFIG["logging_steps"],
    ) -> dict:
        """
        Train the model.

        Args:
            dataset: Tokenized training dataset
            output_dir: Directory to save model
            num_epochs: Number of training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            save_steps: Save model every N steps
            logging_steps: Log every N steps

        Returns:
            Training statistics
        """
        logger.info("Starting training")

        # Split dataset (80/20 train/val)
        dataset = dataset.train_test_split(
            test_size=TRAIN_TEST_SPLIT_RATIO, seed=RANDOM_SEED
        )
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

        logger.info(f"Training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(eval_dataset)}")

        # Training arguments optimized for M1 Mac
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,  # Effective batch size = batch_size * 2
            learning_rate=learning_rate,
            weight_decay=TRAINING_CONFIG["weight_decay"],
            logging_steps=logging_steps,
            eval_strategy="steps",  # Fixed: changed from evaluation_strategy
            save_strategy="steps",
            eval_steps=save_steps,
            save_steps=save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            warmup_steps=warmup_steps,
            fp16=False,  # Set to True if not using MPS
            bf16=False,
            dataloader_pin_memory=False,  # Important for MPS
            dataloader_num_workers=TRAINING_CONFIG["dataloader_num_workers"],
            remove_unused_columns=False,
            report_to=None,  # Disable wandb for now
        )

        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
        )

        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        # Train
        logger.info("Starting training...")
        train_result = self.trainer.train()

        # Save model
        logger.info("Saving model...")
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)

        # Save LoRA adapter if using LoRA
        if self.use_lora:
            self.model.save_pretrained(output_dir)
            logger.info("LoRA adapter saved")

        # Training statistics
        training_stats = {
            "total_flos": train_result.metrics.get("train_runtime", 0),
            "train_loss": train_result.metrics.get("train_loss", 0),
            "eval_loss": self.trainer.evaluate().get("eval_loss", 0),
            "num_epochs": num_epochs,
            "num_train_samples": len(train_dataset),
            "num_eval_samples": len(eval_dataset),
            "model_name": self.model_name,
            "use_lora": self.use_lora,
            "device": self.device,
        }

        logger.info("Training completed!")
        logger.info(f"Final train loss: {training_stats['train_loss']:.4f}")
        logger.info(f"Final eval loss: {training_stats['eval_loss']:.4f}")

        return training_stats

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = GENERATION_CONFIG["max_new_tokens"],
        temperature: float = GENERATION_CONFIG["temperature"],
        do_sample: bool = GENERATION_CONFIG["do_sample"],
    ) -> str:
        """
        Generate text using the trained model.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling

        Returns:
            Generated text
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model_and_tokenizer() first.")

        # Tokenize input
        inputs = self.tokenizer.encode(prompt, return_tensors="pt")
        if self.device == "mps":
            inputs = inputs.to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=GENERATION_CONFIG["top_p"],
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the original prompt from output
        new_text = generated_text[len(prompt) :].strip()
        return new_text


def train_german_llm(
    data_path: str,
    model_name: str = LLM_CONFIG["default_model"],
    output_dir: str = None,
    max_samples: int = None,
    **training_kwargs,
) -> dict:
    """
    Train a German LLM on the dataset.

    Args:
        data_path: Path to the dataset (parquet file)
        model_name: HuggingFace model to fine-tune
        output_dir: Where to save the trained model
        max_samples: Maximum samples to use (None for all)
        **training_kwargs: Additional training arguments

    Returns:
        Training results
    """
    # Load data
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)

    if max_samples:
        df = df.sample(n=min(max_samples, len(df)), random_state=42)
        logger.info(f"Using {len(df)} samples (limited to {max_samples})")
    else:
        logger.info(f"Using all {len(df)} samples")

    # Set default output directory
    if output_dir is None:
        model_safe_name = model_name.replace("/", "_")
        output_dir = f"{constants.MODELS_DIR}/german_llm_{model_safe_name}"

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Initialize trainer
    trainer = GermanLLMTrainer(model_name=model_name)
    trainer.load_model_and_tokenizer()

    # Prepare dataset
    dataset = trainer.prepare_dataset(df)

    # Train
    training_stats = trainer.train(dataset, output_dir, **training_kwargs)

    # Save metadata
    metadata = {
        "model_name": model_name,
        "training_stats": training_stats,
        "dataset_info": {
            "samples": len(df),
            "prompt_distribution": df["prompt_id"].value_counts().to_dict(),
        },
    }

    metadata_path = Path(output_dir) / "training_metadata.json"
    save_metadata(metadata, str(metadata_path))

    # Test generation
    logger.info("Testing text generation...")
    test_prompts = ["[REPHRASING]", "[SUMMARIZATION]", "[FORMULATING QUESTIONS]"]

    for prompt in test_prompts:
        try:
            generated = trainer.generate_text(prompt, max_new_tokens=50)
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Generated: {generated[:100]}...")
        except Exception as e:
            logger.warning(f"Generation failed for {prompt}: {e}")

    return {
        "training_stats": training_stats,
        "model_path": output_dir,
        "metadata_path": str(metadata_path),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune German LLM")
    parser.add_argument("--data", required=True, help="Path to training data (parquet)")
    parser.add_argument(
        "--model", default=LLM_CONFIG["default_model"], help="HuggingFace model name"
    )
    parser.add_argument("--output_dir", help="Output directory for trained model")
    parser.add_argument("--max_samples", type=int, help="Maximum samples to use")
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAINING_CONFIG["num_epochs"],
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=TRAINING_CONFIG["batch_size"],
        help="Batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=TRAINING_CONFIG["learning_rate"],
        help="Learning rate",
    )

    args = parser.parse_args()

    # Train model
    results = train_german_llm(
        data_path=args.data,
        model_name=args.model,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )

    print("\n" + "=" * 60)
    print("LLM TRAINING COMPLETED")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Samples: {results['training_stats']['num_train_samples']:,}")
    print(f"Final Loss: {results['training_stats']['eval_loss']:.4f}")
    print(f"Saved to: {results['model_path']}")
    print("=" * 60)
