"""
Test trained German LLM models.
"""

import os
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.config import GENERATION_CONFIG
from src.utils.helpers import setup_logging

logger = setup_logging()


class LLMTester:
    """Test and interact with trained LLM."""

    def __init__(self, model_path: str):
        """
        Initialize LLM tester.

        Args:
            model_path: Path to trained model directory
        """
        self.model_path = Path(model_path)
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self.tokenizer = None
        self.model = None

        # Load metadata if available
        metadata_path = self.model_path / "training_metadata.json"
        self.metadata = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)

    def load_model(self):
        """Load the trained model and tokenizer."""
        logger.info(f"Loading model from {self.model_path}")
        logger.info(f"Device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        # Fix pad token issue
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Ensure different pad and eos tokens if possible
        if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
            # Try to use a different token for padding
            if (
                hasattr(self.tokenizer, "unk_token")
                and self.tokenizer.unk_token is not None
            ):
                self.tokenizer.pad_token = self.tokenizer.unk_token
            else:
                # Add a new pad token if none exists
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # Check if this is a LoRA model
        adapter_config_path = self.model_path / "adapter_config.json"
        is_lora = adapter_config_path.exists()

        if is_lora:
            logger.info("Loading LoRA adapter model")

            # Get base model name from metadata or adapter config
            if "model_name" in self.metadata:
                base_model_name = self.metadata["model_name"]
            else:
                with open(adapter_config_path) as f:
                    adapter_config = json.load(f)
                base_model_name = adapter_config.get(
                    "base_model_name_or_path", "microsoft/DialoGPT-small"
                )

            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device != "mps" else None,
            )

            # Load LoRA adapter
            self.model = PeftModel.from_pretrained(base_model, self.model_path)

        else:
            logger.info("📦 Loading full fine-tuned model")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
                device_map="auto" if self.device != "mps" else None,
            )

        # Move to device for MPS
        if self.device == "mps":
            self.model = self.model.to(self.device)

        logger.info("Model loaded successfully")

    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = GENERATION_CONFIG["max_new_tokens"],
        temperature: float = GENERATION_CONFIG["temperature"],
        do_sample: bool = GENERATION_CONFIG["do_sample"],
        top_p: float = GENERATION_CONFIG["top_p"],
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            top_p: Top-p (nucleus) sampling parameter

        Returns:
            Generated text
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        # Tokenize input with attention mask
        inputs = self.tokenizer(
            prompt, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        if self.device == "mps":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove the original prompt from output
        new_text = generated_text[len(prompt) :].strip()
        return new_text

    def interactive_chat(self):
        """Interactive chat session."""
        print("\nGerman LLM Interactive Chat")
        print("=" * 50)
        print("Type your prompts below. Use Ctrl+C to exit.")
        print("Try prompts like:")
        print("  [REPHRASING] ")
        print("  [SUMMARIZATION] ")
        print("  [FORMULATING QUESTIONS] ")
        print("=" * 50)

        while True:
            try:
                prompt = input("\nYou: ").strip()
                if not prompt:
                    continue

                print("AI: ", end="", flush=True)
                response = self.generate_text(
                    prompt, max_new_tokens=GENERATION_CONFIG["max_new_tokens"]
                )
                print(response)

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")

    def run_test_prompts(self):
        """Run a set of test prompts."""
        test_prompts = [
            "[REPHRASING] Das Wetter ist heute sehr schön.",
            "[SUMMARIZATION] Deutschland ist ein Land in Mitteleuropa mit einer reichen Geschichte und Kultur.",
            "[FORMULATING QUESTIONS] Berlin ist die Hauptstadt von Deutschland.",
            "[EXTRACTING LISTS] Ich mag Äpfel, Birnen und Bananen.",
            "[REPHRASING IN WIKIPEDIA STYLE] Fußball ist ein beliebter Sport.",
        ]

        print("\n🧪 Testing with sample prompts:")
        print("=" * 60)

        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n{i}. Prompt: {prompt}")
            print("-" * 40)
            try:
                response = self.generate_text(prompt, max_new_tokens=100)
                print(f"Response: {response}")
            except Exception as e:
                print(f"Error: {e}")

        print("=" * 60)

    def show_model_info(self):
        """Display model information."""
        print("\nModel Information:")
        print("=" * 40)
        print(f"Model Path: {self.model_path}")
        print(f"Device: {self.device}")

        if self.metadata:
            print(f"Base Model: {self.metadata.get('model_name', 'Unknown')}")

            if "training_stats" in self.metadata:
                stats = self.metadata["training_stats"]
                print(
                    f"Training Samples: {stats.get('num_train_samples', 'Unknown'):,}"
                )
                print(f"Final Loss: {stats.get('eval_loss', 'Unknown')}")
                print(f"Used LoRA: {stats.get('use_lora', 'Unknown')}")

            if "dataset_info" in self.metadata:
                dataset = self.metadata["dataset_info"]
                print(f"Dataset Samples: {dataset.get('samples', 'Unknown'):,}")

        print("=" * 40)


def main():
    """Main testing function."""
    parser = argparse.ArgumentParser(description="Test trained German LLM")
    parser.add_argument(
        "--model_path", required=True, help="Path to trained model directory"
    )
    parser.add_argument(
        "--mode", choices=["chat", "test", "info"], default="test", help="Testing mode"
    )

    args = parser.parse_args()

    # Check if model path exists
    if not Path(args.model_path).exists():
        print(f"Model path does not exist: {args.model_path}")
        return

    # Initialize tester
    tester = LLMTester(args.model_path)

    try:
        # Load model
        tester.load_model()

        # Run based on mode
        if args.mode == "info":
            tester.show_model_info()
        elif args.mode == "test":
            tester.show_model_info()
            tester.run_test_prompts()
        elif args.mode == "chat":
            tester.show_model_info()
            tester.interactive_chat()

    except Exception as e:
        print(f"Error: {e}")
        logger.error(f"Testing failed: {e}")


if __name__ == "__main__":
    main()
