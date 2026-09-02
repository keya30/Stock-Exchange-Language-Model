"""
CLI text generation with MicroLM.
Load a trained model and generate stock exchange text.
"""
import argparse
import os

import tensorflow as tf

from model.config import ModelConfig
from model.tokenizer import BPETokenizer
from model.transformer import MicroLM


def load_model(checkpoint_dir: str = "checkpoints"):
    """Load trained model and tokenizer from checkpoint directory."""
    # Load tokenizer
    tokenizer_path = os.path.join(checkpoint_dir, "tokenizer.json")
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. "
            "Please run train.py first."
        )

    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)

    # Build model with correct vocab size
    config = ModelConfig(vocab_size=tokenizer.vocab_size)
    model = MicroLM(config)

    # Build model by running dummy input
    dummy = tf.zeros((1, config.seq_length), dtype=tf.int32)
    model(dummy)

    # Load weights (try best first, then final)
    best_path = os.path.join(checkpoint_dir, "best_model.weights.h5")
    final_path = os.path.join(checkpoint_dir, "final_model.weights.h5")

    if os.path.exists(best_path):
        model.load_weights(best_path)
        print(f"[Model] Loaded best weights from {best_path}")
    elif os.path.exists(final_path):
        model.load_weights(final_path)
        print(f"[Model] Loaded final weights from {final_path}")
    else:
        raise FileNotFoundError(
            f"No model weights found in {checkpoint_dir}. "
            "Please run train.py first."
        )

    return model, tokenizer, config


def generate_text(
    model: MicroLM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_length: int = 500,
    temperature: float = 0.8,
    top_k: int = 40,
) -> str:
    """Generate text given a prompt."""
    prompt_ids = tokenizer.encode(prompt)
    generated_ids = model.generate(
        prompt_ids,
        max_new_tokens=max_length,
        temperature=temperature,
        top_k=top_k,
    )
    return tokenizer.decode(generated_ids)


def interactive_mode(model, tokenizer, config):
    """Interactive REPL for text generation."""
    print("\n" + "=" * 60)
    print("  MicroLM - Interactive Text Generation")
    print("  Type a prompt and press Enter to generate text.")
    print("  Type 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    while True:
        try:
            prompt = input("📈 Prompt: ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                print("Goodbye!")
                break
            if not prompt:
                continue

            print("Generating...\n")
            text = generate_text(model, tokenizer, prompt)
            print(f"📝 Generated:\n{text}\n")
            print("-" * 60)

        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate text with MicroLM")
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="Text prompt to complete"
    )
    parser.add_argument(
        "--length", type=int, default=500,
        help="Maximum number of characters to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature (higher = more creative)"
    )
    parser.add_argument(
        "--top-k", type=int, default=40,
        help="Top-k sampling (0 = disabled)"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints",
        help="Path to checkpoint directory"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run in interactive mode"
    )

    args = parser.parse_args()

    # Load model
    model, tokenizer, config = load_model(args.checkpoint)
    print(model.summary_str())

    if args.interactive or args.prompt is None:
        interactive_mode(model, tokenizer, config)
    else:
        text = generate_text(
            model, tokenizer, args.prompt,
            max_length=args.length,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(f"\n{text}")
