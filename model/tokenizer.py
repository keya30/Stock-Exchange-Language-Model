"""
Subword (BPE) tokenizer for MicroLM using HuggingFace tokenizers.
Maps subword tokens to integer IDs and back.
"""
import os
from typing import List
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace


class BPETokenizer:
    """BPE Tokenizer using HuggingFace."""

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self):
        self.tokenizer = Tokenizer(BPE(unk_token=self.UNK_TOKEN))
        self.tokenizer.pre_tokenizer = Whitespace()
        self.vocab_size: int = 0
        self._fitted = False

    def fit(self, text: str) -> "BPETokenizer":
        """Build vocabulary from a text corpus."""
        # Save text to a temporary file for training
        temp_file = "temp_corpus.txt"
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write(text)

        trainer = BpeTrainer(
            special_tokens=[self.PAD_TOKEN, self.UNK_TOKEN],
            vocab_size=8000,
            min_frequency=2
        )
        self.tokenizer.train([temp_file], trainer)
        
        self.vocab_size = self.tokenizer.get_vocab_size()
        self._fitted = True

        if os.path.exists(temp_file):
            os.remove(temp_file)
        print(f"[Tokenizer] Vocabulary built: {self.vocab_size} tokens")
        return self

    def encode(self, text: str) -> List[int]:
        """Convert text to a list of integer token IDs."""
        assert self._fitted, "Tokenizer must be fit() before encoding"
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        """Convert a list of integer token IDs back to text."""
        assert self._fitted, "Tokenizer must be fit() before decoding"
        return self.tokenizer.decode(ids)

    def save(self, path: str) -> None:
        """Save tokenizer vocabulary to a JSON file."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.tokenizer.save(path)
        print(f"[Tokenizer] Saved to {path}")

    def load(self, path: str) -> "BPETokenizer":
        """Load tokenizer vocabulary from a JSON file."""
        self.tokenizer = Tokenizer.from_file(path)
        self.vocab_size = self.tokenizer.get_vocab_size()
        self._fitted = True
        print(f"[Tokenizer] Loaded from {path} ({self.vocab_size} tokens)")
        return self

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        return f"BPETokenizer(vocab_size={self.vocab_size}, {status})"
