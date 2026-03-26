"""Streaming dataloader from pre-tokenized flat binary files.

Expects data/ directory with tokens_train.npy, tokens_val.npy,
bytes_train.npy, bytes_val.npy from scripts/tokenize_data.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from src.config import DataConfig

DATA_DIR = Path("data")


@dataclass
class Batch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    target_byte_lengths: torch.Tensor

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            input_ids=self.input_ids.to(device, non_blocking=True),
            target_ids=self.target_ids.to(device, non_blocking=True),
            attention_mask=self.attention_mask.to(device, non_blocking=True),
            position_ids=self.position_ids.to(device, non_blocking=True),
            target_byte_lengths=self.target_byte_lengths.to(device, non_blocking=True),
        )


class TokenStream:
    """Memory-mapped token stream that yields fixed-length sequences.

    Chunks the flat token array into (max_seq_len + 1) sequences.
    No padding, no waste — just contiguous slicing.
    """

    def __init__(self, split: str, max_seq_len: int) -> None:
        tokens_path = DATA_DIR / f"tokens_{split}.npy"
        bytes_path = DATA_DIR / f"bytes_{split}.npy"
        if not tokens_path.exists():
            raise FileNotFoundError(
                f"{tokens_path} not found. Run: uv run python -m scripts.tokenize_data"
            )
        self.tokens = np.load(tokens_path, mmap_mode="r")
        self.byte_lengths = np.load(bytes_path, mmap_mode="r")
        self.seq_len = max_seq_len + 1  # +1 for target shift
        self.num_sequences = len(self.tokens) // self.seq_len

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        start = index * self.seq_len
        end = start + self.seq_len
        return self.tokens[start:end], self.byte_lengths[start:end]


class StreamingBatchIterator:
    """Infinite iterator that yields Batch objects from a TokenStream.

    Pre-allocates pinned CPU buffers for efficient GPU transfer.
    Cycles through the data indefinitely.
    """

    def __init__(
        self,
        stream: TokenStream,
        batch_size: int,
        max_seq_len: int,
        pin_memory: bool = True,
        shuffle: bool = False,
        seed: int = 42,
    ) -> None:
        self.stream = stream
        self.batch_size = batch_size
        self.seq_len = max_seq_len
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)

        # Pre-allocate pinned CPU buffers
        self.input_buf = torch.empty(
            (batch_size, max_seq_len), dtype=torch.long, pin_memory=pin_memory
        )
        self.target_buf = torch.empty(
            (batch_size, max_seq_len), dtype=torch.long, pin_memory=pin_memory
        )
        self.byte_buf = torch.empty(
            (batch_size, max_seq_len), dtype=torch.long, pin_memory=pin_memory
        )

        # Static tensors (same every batch)
        self.attention_mask = torch.ones(
            (batch_size, max_seq_len), dtype=torch.bool, pin_memory=pin_memory
        )
        self.position_ids = (
            torch.arange(max_seq_len, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, -1)
            .contiguous()
        )
        if pin_memory:
            self.position_ids = self.position_ids.pin_memory()

        self._build_order()
        self.cursor = 0

    def _build_order(self) -> None:
        n = len(self.stream)
        self.order = self.rng.permutation(n) if self.shuffle else np.arange(n)

    def _next_index(self) -> int:
        if self.cursor >= len(self.order):
            self._build_order()
            self.cursor = 0
        idx = int(self.order[self.cursor])
        self.cursor += 1
        return idx

    def __iter__(self) -> StreamingBatchIterator:
        return self

    def __next__(self) -> Batch:
        for row in range(self.batch_size):
            idx = self._next_index()
            tokens, byte_lengths = self.stream[idx]
            # tokens is (seq_len+1,), split into input/target
            self.input_buf[row] = torch.from_numpy(tokens[:-1].copy())
            self.target_buf[row] = torch.from_numpy(tokens[1:].copy())
            self.byte_buf[row] = torch.from_numpy(byte_lengths[1:].copy())

        return Batch(
            input_ids=self.input_buf.clone(),
            target_ids=self.target_buf.clone(),
            attention_mask=self.attention_mask.clone(),
            position_ids=self.position_ids.clone(),
            target_byte_lengths=self.byte_buf.clone(),
        )


def build_dataloaders(
    config: DataConfig,
) -> tuple[StreamingBatchIterator, StreamingBatchIterator, int]:
    """Build train and val batch iterators from pre-tokenized data."""
    train_stream = TokenStream("train", config.max_seq_len)
    val_stream = TokenStream("val", config.max_seq_len)

    train_iter = StreamingBatchIterator(
        train_stream,
        batch_size=config.batch_size,
        max_seq_len=config.max_seq_len,
        pin_memory=config.pin_memory,
        shuffle=True,
    )
    val_iter = StreamingBatchIterator(
        val_stream,
        batch_size=config.eval_batch_size,
        max_seq_len=config.max_seq_len,
        pin_memory=config.pin_memory,
        shuffle=False,
    )

    # Load eot token from metadata
    import json

    meta_path = DATA_DIR / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        eot_token = meta["eot_token_id"]
    else:
        eot_token = 0

    return train_iter, val_iter, eot_token
