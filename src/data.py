"""Dataset loading and tokenization utilities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

from src.config import DataConfig
from src.tokenization import get_tokenizer


@dataclass
class Batch:
    input_ids: torch.Tensor
    target_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    target_byte_lengths: torch.Tensor

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            input_ids=self.input_ids.to(device),
            target_ids=self.target_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            position_ids=self.position_ids.to(device),
            target_byte_lengths=self.target_byte_lengths.to(device),
        )


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self, sequences: list[torch.Tensor], byte_lengths: list[torch.Tensor]
    ) -> None:
        self.sequences = sequences
        self.byte_lengths = byte_lengths

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.byte_lengths[index]


def cache_key(config: DataConfig) -> str:
    payload = {
        "cache_format_version": 2,
        "dataset_name": config.dataset_name,
        "dataset_split": config.dataset_split,
        "dataset_languages": config.dataset_languages,
        "streaming": config.streaming,
        "tokenizer_name": config.tokenizer_name,
        "max_seq_len": config.max_seq_len,
        "train_token_budget": config.train_token_budget,
        "val_token_budget": config.val_token_budget,
        "max_train_documents": config.max_train_documents,
        "max_val_documents": config.max_val_documents,
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def cache_paths(config: DataConfig) -> tuple[Path, Path]:
    root = Path(config.cache_dir)
    key = cache_key(config)
    return root / f"{key}.pt", root / f"{key}.json"


def _save_cache(
    config: DataConfig,
    train_sequences: list[torch.Tensor],
    train_byte_lengths: list[torch.Tensor],
    val_sequences: list[torch.Tensor],
    val_byte_lengths: list[torch.Tensor],
    eot_token: int,
) -> None:
    data_path, meta_path = cache_paths(config)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "train_sequences": train_sequences,
            "train_byte_lengths": train_byte_lengths,
            "val_sequences": val_sequences,
            "val_byte_lengths": val_byte_lengths,
            "eot_token": eot_token,
        },
        data_path,
    )
    meta = {
        "cache_key": cache_key(config),
        "dataset_name": config.dataset_name,
        "dataset_languages": list(config.dataset_languages),
        "tokenizer_name": config.tokenizer_name,
        "max_seq_len": config.max_seq_len,
        "train_examples": len(train_sequences),
        "val_examples": len(val_sequences),
        "train_tokens": int(sum(seq.numel() for seq in train_sequences)),
        "val_tokens": int(sum(seq.numel() for seq in val_sequences)),
        "train_bytes": int(sum(length.sum().item() for length in train_byte_lengths)),
        "val_bytes": int(sum(length.sum().item() for length in val_byte_lengths)),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _load_cache(
    config: DataConfig,
) -> tuple[SequenceDataset, SequenceDataset, int] | None:
    data_path, _ = cache_paths(config)
    if not config.use_cache or not data_path.exists():
        return None
    payload = torch.load(data_path, map_location="cpu")
    return (
        SequenceDataset(payload["train_sequences"], payload["train_byte_lengths"]),
        SequenceDataset(payload["val_sequences"], payload["val_byte_lengths"]),
        int(payload["eot_token"]),
    )


def _iter_finewiki_documents(config: DataConfig) -> Iterable[str]:
    iterators: list[tuple[str, Iterable[dict]]] = []
    for language in config.dataset_languages:
        dataset = load_dataset(
            config.dataset_name,
            language,
            split=config.dataset_split,
            streaming=config.streaming,
        )
        iterators.append((language, iter(dataset)))

    active = list(iterators)
    while active:
        next_active: list[tuple[str, Iterable[dict]]] = []
        for language, iterator in active:
            try:
                row = next(iterator)
            except StopIteration:
                continue
            text = row.get("text")
            if isinstance(text, str) and text:
                yield text
            next_active.append((language, iterator))
        active = next_active


def _split_long_document(tokens: list[int], max_tokens: int) -> Iterable[list[int]]:
    start = 0
    while start < len(tokens):
        yield tokens[start : start + max_tokens]
        start += max_tokens


def build_datasets(config: DataConfig) -> tuple[SequenceDataset, SequenceDataset, int]:
    cached = _load_cache(config)
    if cached is not None:
        return cached

    tokenizer = get_tokenizer(config.tokenizer_name)
    eot_token = tokenizer.eot_token_id
    max_tokens_per_example = config.max_seq_len + 1

    train_sequences: list[torch.Tensor] = []
    train_byte_lengths: list[torch.Tensor] = []
    val_sequences: list[torch.Tensor] = []
    val_byte_lengths: list[torch.Tensor] = []
    train_tokens = 0
    val_tokens = 0
    train_docs = 0
    val_docs = 0

    current_split = "train"
    for text in _iter_finewiki_documents(config):
        encoded = tokenizer.encode_document(text)
        doc_tokens = encoded.token_ids
        doc_byte_lengths = encoded.byte_lengths
        if len(doc_tokens) < 2:
            continue

        chunks = list(_split_long_document(doc_tokens, max_tokens_per_example))
        byte_chunks = list(
            _split_long_document(doc_byte_lengths, max_tokens_per_example)
        )
        if current_split == "train":
            train_docs += 1
        else:
            val_docs += 1

        for chunk, byte_chunk in zip(chunks, byte_chunks, strict=True):
            if len(chunk) < 2:
                continue
            tensor = torch.tensor(chunk, dtype=torch.long)
            byte_tensor = torch.tensor(byte_chunk, dtype=torch.long)
            if current_split == "train":
                if train_tokens >= config.train_token_budget:
                    current_split = "val"
                else:
                    train_sequences.append(tensor)
                    train_byte_lengths.append(byte_tensor)
                    train_tokens += tensor.numel()
                    continue
            if current_split == "val" and val_tokens < config.val_token_budget:
                val_sequences.append(tensor)
                val_byte_lengths.append(byte_tensor)
                val_tokens += tensor.numel()

        if (
            current_split == "train"
            and config.max_train_documents is not None
            and train_docs >= config.max_train_documents
        ):
            current_split = "val"
        if (
            current_split == "val"
            and config.max_val_documents is not None
            and val_docs >= config.max_val_documents
        ):
            break
        if current_split == "val" and val_tokens >= config.val_token_budget:
            break

    if not train_sequences:
        raise RuntimeError("No training sequences were materialized from the dataset.")
    if not val_sequences:
        raise RuntimeError(
            "No validation sequences were materialized from the dataset."
        )

    if config.use_cache:
        _save_cache(
            config,
            train_sequences,
            train_byte_lengths,
            val_sequences,
            val_byte_lengths,
            eot_token,
        )

    return (
        SequenceDataset(train_sequences, train_byte_lengths),
        SequenceDataset(val_sequences, val_byte_lengths),
        eot_token,
    )


def _collate_sequences(
    sequences_with_bytes: list[tuple[torch.Tensor, torch.Tensor]],
    pad_token_id: int,
) -> Batch:
    batch_size = len(sequences_with_bytes)
    max_tokens = max(seq.numel() for seq, _ in sequences_with_bytes)
    inputs = torch.full((batch_size, max_tokens - 1), pad_token_id, dtype=torch.long)
    targets = torch.full((batch_size, max_tokens - 1), -100, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_tokens - 1), dtype=torch.bool)
    position_ids = torch.zeros((batch_size, max_tokens - 1), dtype=torch.long)
    target_byte_lengths = torch.zeros((batch_size, max_tokens - 1), dtype=torch.long)

    for row, (seq, byte_lengths) in enumerate(sequences_with_bytes):
        seq_len = seq.numel() - 1
        inputs[row, :seq_len] = seq[:-1]
        targets[row, :seq_len] = seq[1:]
        attention_mask[row, :seq_len] = True
        position_ids[row, :seq_len] = torch.arange(seq_len, dtype=torch.long)
        target_byte_lengths[row, :seq_len] = byte_lengths[1:]

    return Batch(
        input_ids=inputs,
        target_ids=targets,
        attention_mask=attention_mask,
        position_ids=position_ids,
        target_byte_lengths=target_byte_lengths,
    )


def build_dataloaders(
    config: DataConfig,
) -> tuple[DataLoader[Batch], DataLoader[Batch], int]:
    train_dataset, val_dataset, pad_token_id = build_datasets(config)

    def collate_fn(seqs: list[tuple[torch.Tensor, torch.Tensor]]) -> Batch:
        return _collate_sequences(seqs, pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=collate_fn,
    )
    return train_loader, val_loader, pad_token_id
