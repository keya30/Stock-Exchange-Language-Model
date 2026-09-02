"""
MicroLM — A micro GPT-style decoder-only transformer for stock exchange text.
Built from scratch in TensorFlow / Keras.
"""
import tensorflow as tf
import numpy as np
from .config import ModelConfig


class CausalSelfAttention(tf.keras.layers.Layer):
    """Multi-head causal (masked) self-attention."""

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.embed_dim = config.embed_dim

        self.query = tf.keras.layers.Dense(config.embed_dim, name="q_proj")
        self.key = tf.keras.layers.Dense(config.embed_dim, name="k_proj")
        self.value = tf.keras.layers.Dense(config.embed_dim, name="v_proj")
        self.out_proj = tf.keras.layers.Dense(config.embed_dim, name="out_proj")

        self.attn_dropout = tf.keras.layers.Dropout(config.dropout)
        self.proj_dropout = tf.keras.layers.Dropout(config.dropout)

    def call(self, x, training=False):
        batch_size = tf.shape(x)[0]
        seq_len = tf.shape(x)[1]

        # Project to Q, K, V
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        # Reshape to (batch, num_heads, seq_len, head_dim)
        q = tf.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        q = tf.transpose(q, perm=[0, 2, 1, 3])

        k = tf.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = tf.transpose(k, perm=[0, 2, 1, 3])

        v = tf.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        # Scaled dot-product attention
        scale = tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        attn_weights = tf.matmul(q, k, transpose_b=True) / scale

        # Causal mask — prevent attending to future tokens
        causal_mask = tf.linalg.band_part(
            tf.ones((seq_len, seq_len), dtype=tf.float32), -1, 0
        )
        causal_mask = tf.reshape(causal_mask, (1, 1, seq_len, seq_len))
        attn_weights = attn_weights * causal_mask + (1.0 - causal_mask) * (-1e9)

        attn_weights = tf.nn.softmax(attn_weights, axis=-1)
        attn_weights = self.attn_dropout(attn_weights, training=training)

        # Weighted sum of values
        attn_output = tf.matmul(attn_weights, v)

        # Reshape back to (batch, seq_len, embed_dim)
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        attn_output = tf.reshape(attn_output, (batch_size, seq_len, self.embed_dim))

        # Output projection
        output = self.out_proj(attn_output)
        output = self.proj_dropout(output, training=training)
        return output


class FeedForward(tf.keras.layers.Layer):
    """Position-wise feed-forward network with GELU activation."""

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.dense_1 = tf.keras.layers.Dense(
            config.ff_dim, activation="gelu", name="ff_up"
        )
        self.dense_2 = tf.keras.layers.Dense(config.embed_dim, name="ff_down")
        self.dropout = tf.keras.layers.Dropout(config.dropout)

    def call(self, x, training=False):
        x = self.dense_1(x)
        x = self.dense_2(x)
        x = self.dropout(x, training=training)
        return x


class TransformerBlock(tf.keras.layers.Layer):
    """Single transformer decoder block: Attention + FFN with pre-norm."""

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.attention = CausalSelfAttention(config)
        self.ffn = FeedForward(config)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="ln_attn")
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5, name="ln_ffn")

    def call(self, x, training=False):
        # Pre-norm architecture (more stable training)
        x = x + self.attention(self.ln1(x), training=training)
        x = x + self.ffn(self.ln2(x), training=training)
        return x


class MicroLM(tf.keras.Model):
    """
    Micro Language Model — GPT-style decoder-only transformer.

    Architecture:
    - Token embedding + learnable positional encoding
    - N transformer blocks (causal attention + FFN)
    - Layer normalization
    - Linear projection to vocabulary logits
    """

    def __init__(self, config: ModelConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # Embeddings
        self.token_embedding = tf.keras.layers.Embedding(
            config.vocab_size, config.embed_dim, name="token_emb"
        )
        self.position_embedding = tf.keras.layers.Embedding(
            config.seq_length, config.embed_dim, name="pos_emb"
        )
        self.embed_dropout = tf.keras.layers.Dropout(config.dropout)

        # Transformer blocks
        self.blocks = [
            TransformerBlock(config, name=f"block_{i}")
            for i in range(config.num_layers)
        ]

        # Output head
        self.ln_final = tf.keras.layers.LayerNormalization(
            epsilon=1e-5, name="ln_final"
        )
        self.lm_head = tf.keras.layers.Dense(
            config.vocab_size, name="lm_head"
        )

    def call(self, x, training=False):
        """
        Forward pass.

        Args:
            x: Integer token IDs of shape (batch_size, seq_length)
            training: Whether in training mode (for dropout)

        Returns:
            Logits of shape (batch_size, seq_length, vocab_size)
        """
        seq_len = tf.shape(x)[1]

        # Token + positional embeddings
        positions = tf.range(seq_len)
        tok_emb = self.token_embedding(x)
        pos_emb = self.position_embedding(positions)
        x = self.embed_dropout(tok_emb + pos_emb, training=training)

        # Transformer blocks
        for block in self.blocks:
            x = block(x, training=training)

        # Output projection
        x = self.ln_final(x)
        logits = self.lm_head(x)
        return logits

    def generate(
        self,
        prompt_ids: list[int],
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 40,
    ) -> list[int]:
        """
        Autoregressive text generation.

        Args:
            prompt_ids: List of token IDs for the prompt
            max_new_tokens: Number of new tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_k: Number of top tokens to sample from (0 = disabled)

        Returns:
            List of generated token IDs (including prompt)
        """
        generated = list(prompt_ids)
        seq_length = self.config.seq_length

        for _ in range(max_new_tokens):
            # Crop to max sequence length
            context = generated[-seq_length:]
            input_ids = tf.constant([context], dtype=tf.int32)

            # Forward pass
            logits = self(input_ids, training=False)
            # Get logits for the last position
            next_logits = logits[0, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                values, indices = tf.math.top_k(next_logits, k=min(top_k, self.config.vocab_size))
                # Create a mask that sets all logits outside top-k to -inf
                mask = tf.fill(tf.shape(next_logits), float("-inf"))
                mask = tf.tensor_scatter_nd_update(
                    mask, tf.expand_dims(indices, 1), values
                )
                next_logits = mask

            # Sample from distribution
            next_id = tf.random.categorical(
                next_logits[tf.newaxis, :], num_samples=1
            )
            next_id = int(next_id[0, 0].numpy())

            generated.append(next_id)

        return generated

    def get_num_params(self) -> int:
        """Count total trainable parameters."""
        return sum(
            tf.reduce_prod(var.shape).numpy()
            for var in self.trainable_variables
        )

    def summary_str(self) -> str:
        """Return a clean model summary string."""
        # Build the model by running a dummy input
        dummy = tf.zeros((1, self.config.seq_length), dtype=tf.int32)
        self(dummy)
        num_params = self.get_num_params()

        lines = [
            "=" * 50,
            "  MicroLM - Stock Exchange Language Model",
            "=" * 50,
            f"  Architecture: GPT-style Decoder-Only Transformer",
            f"  Layers:       {self.config.num_layers}",
            f"  Heads:        {self.config.num_heads}",
            f"  Embed dim:    {self.config.embed_dim}",
            f"  FF dim:       {self.config.ff_dim}",
            f"  Seq length:   {self.config.seq_length}",
            f"  Vocab size:   {self.config.vocab_size}",
            f"  Parameters:   {num_params:,}",
            f"  Size (est):   ~{num_params * 4 / 1024 / 1024:.1f} MB (float32)",
            "=" * 50,
        ]
        return "\n".join(lines)
