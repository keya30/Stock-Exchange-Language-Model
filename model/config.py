"""
Model configuration for MicroLM — Stock Exchange Language Model.
Tuned for RTX 3050 6GB VRAM.
"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Hyperparameters for the micro transformer model."""

    # --- Model Architecture ---
    vocab_size: int = 128           # Set dynamically from tokenizer
    embed_dim: int = 256            # Embedding dimension
    num_heads: int = 8              # Number of attention heads
    num_layers: int = 6             # Number of transformer blocks
    ff_dim: int = 1024              # Feed-forward hidden dimension
    seq_length: int = 256           # Context window length
    dropout: float = 0.0            # Dropout rate (0.0 = no regularization)

    # --- Training ---
    batch_size: int = 32            # Standard batch size
    epochs: int = 50                # Severe overfit
    learning_rate: float = 1e-3     # Unconditional fitting
    warmup_steps: int = 50          # Quick warmup
    min_lr: float = 5e-4            # Keep LR high
    val_split: float = 0.0          # No validation

    # --- Generation ---
    default_temperature: float = 0.8
    default_top_k: int = 40
    default_max_length: int = 500

    # --- Paths ---
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    history_path: str = "training_history.json"

    def __post_init__(self):
        assert self.embed_dim % self.num_heads == 0, \
            f"embed_dim ({self.embed_dim}) must be divisible by num_heads ({self.num_heads})"

    @property
    def head_dim(self) -> int:
        return self.embed_dim // self.num_heads

    def summary(self) -> str:
        """Return a formatted summary of the config."""
        lines = [
            "+==========================================+",
            "|     MicroLM - Stock Exchange Model       |",
            "+==========================================+",
            f"|  Embedding dim   : {self.embed_dim:<21}|",
            f"|  Attention heads : {self.num_heads:<21}|",
            f"|  Transformer layers: {self.num_layers:<19}|",
            f"|  FF hidden dim   : {self.ff_dim:<21}|",
            f"|  Sequence length : {self.seq_length:<21}|",
            f"|  Dropout         : {self.dropout:<21}|",
            f"|  Vocab size      : {self.vocab_size:<21}|",
            "+==========================================+",
            f"|  Batch size      : {self.batch_size:<21}|",
            f"|  Epochs          : {self.epochs:<21}|",
            f"|  Learning rate   : {self.learning_rate:<21}|",
            f"|  Warmup steps    : {self.warmup_steps:<21}|",
            "+==========================================+",
        ]
        return "\n".join(lines)
