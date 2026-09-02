"""
Training pipeline for MicroLM — Stock Exchange Language Model.
Trains a character-level transformer on financial text data.
"""
import os
import json
import time
import argparse
import numpy as np
import tensorflow as tf

from model.config import ModelConfig
from model.tokenizer import BPETokenizer
from model.transformer import MicroLM
from data.corpus import get_corpus


def create_dataset(
    encoded_text: list[int],
    seq_length: int,
    batch_size: int,
    val_split: float = 0.1,
):
    """
    Create training and validation tf.data.Dataset pipelines.

    Each sample is (input_seq, target_seq) where target is input shifted by 1.
    """
    data = np.array(encoded_text, dtype=np.int32)

    # Create sequences using a sliding window
    num_sequences = (len(data) - 1) // seq_length
    # Trim data to fit exact sequences
    trimmed_length = num_sequences * seq_length + 1
    data = data[:trimmed_length]

    # Reshape into sequences
    inputs = []
    targets = []
    for i in range(0, len(data) - seq_length, seq_length // 2):  # 50% overlap
        if i + seq_length + 1 > len(data):
            break
        inputs.append(data[i : i + seq_length])
        targets.append(data[i + 1 : i + seq_length + 1])

    inputs = np.array(inputs, dtype=np.int32)
    targets = np.array(targets, dtype=np.int32)

    print(f"[Dataset] Created {len(inputs)} sequences of length {seq_length}")

    # Shuffle and split
    indices = np.random.permutation(len(inputs))
    inputs = inputs[indices]
    targets = targets[indices]

    if val_split > 0:
        val_size = max(1, int(len(inputs) * val_split))
        train_inputs, val_inputs = inputs[val_size:], inputs[:val_size]
        train_targets, val_targets = targets[val_size:], targets[:val_size]
        print(f"[Dataset] Train: {len(train_inputs)} | Validation: {len(val_inputs)}")
    else:
        train_inputs, train_targets = inputs, targets
        val_inputs, val_targets = None, None
        print(f"[Dataset] Train: {len(train_inputs)} | Validation: NONE (overfit mode)")

    # Build tf.data pipelines
    train_ds = (
        tf.data.Dataset.from_tensor_slices((train_inputs, train_targets))
        .shuffle(buffer_size=min(10000, len(train_inputs)))
        .batch(batch_size, drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = None
    if val_inputs is not None:
        val_ds = (
            tf.data.Dataset.from_tensor_slices((val_inputs, val_targets))
            .batch(batch_size, drop_remainder=False)
            .prefetch(tf.data.AUTOTUNE)
        )

    return train_ds, val_ds


class WarmupCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Learning rate schedule with linear warmup and cosine decay."""

    def __init__(self, peak_lr, warmup_steps, total_steps, min_lr=1e-5):
        super().__init__()
        self.peak_lr = peak_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        warmup = tf.minimum(step / tf.maximum(tf.cast(self.warmup_steps, tf.float32), 1.0), 1.0)
        decay_steps = tf.cast(self.total_steps - self.warmup_steps, tf.float32)
        decay_progress = tf.minimum(
            (step - tf.cast(self.warmup_steps, tf.float32)) / tf.maximum(decay_steps, 1.0),
            1.0,
        )
        decay_progress = tf.maximum(decay_progress, 0.0)
        cosine_decay = 0.5 * (1.0 + tf.cos(np.pi * decay_progress))
        lr = self.min_lr + (self.peak_lr - self.min_lr) * cosine_decay
        return lr * warmup

    def get_config(self):
        return {
            "peak_lr": self.peak_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "min_lr": self.min_lr,
        }


class SampleGenerationCallback(tf.keras.callbacks.Callback):
    """Generate sample text every N epochs to monitor training progress."""

    def __init__(self, tokenizer, prompts, every_n_epochs=5, max_tokens=150):
        super().__init__()
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.every_n_epochs = every_n_epochs
        self.max_tokens = max_tokens

    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.every_n_epochs == 0:
            print(f"\n{'-' * 60}")
            print(f"  Sample generations (epoch {epoch + 1})")
            print(f"{'-' * 60}")
            for prompt in self.prompts:
                prompt_ids = self.tokenizer.encode(prompt)
                generated_ids = self.model.generate(
                    prompt_ids,
                    max_new_tokens=self.max_tokens,
                    temperature=0.8,
                    top_k=40,
                )
                text = self.tokenizer.decode(generated_ids)
                # Show just first 200 chars
                preview = text[:200] + ("..." if len(text) > 200 else "")
                print(f"  Prompt: \"{prompt}\"")
                print(f"  Output: {preview}")
                print()
            print(f"{'-' * 60}\n")


def train(config: ModelConfig = None):
    """Run the full training pipeline."""
    if config is None:
        config = ModelConfig()

    print("\n" + "=" * 60)
    print("  MicroLM - Stock Exchange Language Model")
    print("  Training Pipeline")
    print("=" * 60 + "\n")

    # -- 1. Load corpus --
    print("Step 1: Loading corpus...")
    corpus = get_corpus()
    print()

    # -- 2. Build tokenizer --
    print("Step 2: Building tokenizer...")
    tokenizer = BPETokenizer()
    tokenizer.fit(corpus)
    config.vocab_size = tokenizer.vocab_size

    # Save tokenizer
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    tokenizer.save(os.path.join(config.checkpoint_dir, "tokenizer.json"))
    print()

    # -- 3. Encode corpus --
    print("Step 3: Encoding corpus...")
    encoded = tokenizer.encode(corpus)
    print(f"[Encode] {len(corpus):,} characters -> {len(encoded):,} tokens")
    print()

    # -- 4. Create dataset --
    print("Step 4: Creating dataset...")
    train_ds, val_ds = create_dataset(
        encoded, config.seq_length, config.batch_size, config.val_split
    )
    print()

    # -- 5. Build model --
    print("Step 5: Building model...")
    model = MicroLM(config)
    print(model.summary_str())
    print()

    # -- 6. Setup optimizer & compile --
    print("Step 6: Setting up optimizer...")
    steps_per_epoch = sum(1 for _ in train_ds)
    total_steps = steps_per_epoch * config.epochs

    lr_schedule = WarmupCosineDecay(
        peak_lr=config.learning_rate,
        warmup_steps=config.warmup_steps,
        total_steps=total_steps,
        min_lr=config.min_lr,
    )

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr_schedule,
        beta_1=0.9,
        beta_2=0.99,
        epsilon=1e-8,
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    print(f"[Optimizer] Adam with warmup cosine decay")
    print(f"[Schedule] {config.warmup_steps} warmup steps, {total_steps} total steps")
    print()

    has_val = val_ds is not None

    callbacks = [
        # Checkpoint best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(config.checkpoint_dir, "best_model.weights.h5"),
            monitor="val_loss" if has_val else "loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
    ]

    # -- 8. Train --
    print("Step 7: Training...")
    print(config.summary())
    print()

    fit_kwargs = dict(
        x=train_ds,
        epochs=config.epochs,
        callbacks=callbacks,
        verbose=1,
    )
    if val_ds is not None:
        fit_kwargs["validation_data"] = val_ds

    start_time = time.time()
    history = model.fit(**fit_kwargs)
    elapsed = time.time() - start_time

    print(f"\n[Training] Completed in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"[Training] Final train loss: {history.history['loss'][-1]:.4f}")
    if "val_loss" in history.history:
        print(f"[Training] Final val loss:   {history.history['val_loss'][-1]:.4f}")
    else:
        print(f"[Training] No validation (overfit mode)")

    # -- 9. Save training history --
    history_data = {
        "loss": [float(x) for x in history.history["loss"]],
        "accuracy": [float(x) for x in history.history["accuracy"]],
        "training_time_seconds": elapsed,
        "epochs_completed": len(history.history["loss"]),
        "overfit_mode": config.val_split == 0.0,
    }
    if "val_loss" in history.history:
        history_data["val_loss"] = [float(x) for x in history.history["val_loss"]]
        history_data["val_accuracy"] = [float(x) for x in history.history["val_accuracy"]]
        
    history_data["config"] = {
        "vocab_size": config.vocab_size,
        "embed_dim": config.embed_dim,
        "num_heads": config.num_heads,
        "num_layers": config.num_layers,
        "ff_dim": config.ff_dim,
        "seq_length": config.seq_length,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
    }

    with open(config.history_path, "w") as f:
        json.dump(history_data, f, indent=2)
    print(f"[Training] History saved to {config.history_path}")

    # -- 10. Save final model weights --
    final_path = os.path.join(config.checkpoint_dir, "final_model.weights.h5")
    model.save_weights(final_path)
    print(f"[Training] Final weights saved to {final_path}")

    # -- 11. Final sample generation --
    print(f"\n{'=' * 60}")
    print("  Training complete! Run 'python app.py' to launch the web demo.")
    print(f"{'=' * 60}\n")

    return model, tokenizer, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MicroLM")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--seq-length", type=int, default=None, help="Sequence length")
    parser.add_argument("--embed-dim", type=int, default=None, help="Embedding dimension")
    parser.add_argument("--num-layers", type=int, default=None, help="Number of layers")
    parser.add_argument("--num-heads", type=int, default=None, help="Number of heads")
    args = parser.parse_args()

    # Only override config with explicitly provided CLI arguments
    overrides = {}
    if args.epochs is not None: overrides["epochs"] = args.epochs
    if args.batch_size is not None: overrides["batch_size"] = args.batch_size
    if args.lr is not None: overrides["learning_rate"] = args.lr
    if args.seq_length is not None: overrides["seq_length"] = args.seq_length
    if args.embed_dim is not None: overrides["embed_dim"] = args.embed_dim
    if args.num_layers is not None: overrides["num_layers"] = args.num_layers
    if args.num_heads is not None: overrides["num_heads"] = args.num_heads

    config = ModelConfig(**overrides)
    train(config)
