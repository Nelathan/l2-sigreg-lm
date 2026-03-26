"""One-time tokenization of FineWiki into flat binary token files.

Usage:
    uv run python -m scripts.tokenize_data [--max-documents 50000] [--tokenizer lfm25]

Produces:
    data/tokens_train.npy   — flat int32 array of token IDs (train split)
    data/tokens_val.npy     — flat int32 array of token IDs (val split)
    data/bytes_train.npy    — flat int32 array of per-token byte lengths (train)
    data/bytes_val.npy      — flat int32 array of per-token byte lengths (val)
    data/meta.json          — metadata (tokenizer, vocab size, token counts, etc.)

These files are batch-size and seq-len independent. The dataloader slices them
on-the-fly into whatever shape the config requests.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset

from src.tokenization import get_tokenizer

DATA_DIR = Path("data")

LANGUAGES = ("en", "ar", "zh", "fr", "de", "ja", "ko", "es")


def iter_documents(
    dataset_name: str,
    languages: tuple[str, ...],
    split: str,
    max_documents: int | None = None,
) -> list[str]:
    """Stream documents from FineWiki, round-robin across languages.

    Collects all into a list so we only iterate HF streams once.
    """
    iterators = []
    for lang in languages:
        ds = load_dataset(dataset_name, lang, split=split, streaming=True)
        iterators.append(iter(ds))

    documents: list[str] = []
    active = list(range(len(iterators)))
    while active:
        next_active = []
        for idx in active:
            if max_documents is not None and len(documents) >= max_documents:
                return documents
            try:
                row = next(iterators[idx])
            except StopIteration:
                continue
            text = row.get("text")
            if isinstance(text, str) and text.strip():
                documents.append(text)
            next_active.append(idx)
        active = next_active
    return documents


def tokenize_documents(
    documents: list[str],
    tokenizer_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize documents into flat arrays with EOT separators between docs."""
    tokenizer = get_tokenizer(tokenizer_name)
    all_token_ids: list[int] = []
    all_byte_lengths: list[int] = []

    for i, text in enumerate(documents):
        encoded = tokenizer.encode_document(text)
        # encode_document already appends EOT at end
        all_token_ids.extend(encoded.token_ids)
        all_byte_lengths.extend(encoded.byte_lengths)
        if (i + 1) % 5000 == 0:
            print(
                f"  tokenized {i + 1}/{len(documents)} docs, {len(all_token_ids):,} tokens"
            )

    return (
        np.array(all_token_ids, dtype=np.int32),
        np.array(all_byte_lengths, dtype=np.int32),
    )


def shuffle_windows(
    tokens: np.ndarray,
    byte_lengths: np.ndarray,
    seq_len: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Chunk flat arrays into (seq_len+1) windows and shuffle them.

    Drops the tail that doesn't fill a complete window.
    Returns flattened shuffled arrays.
    """
    window = seq_len + 1  # +1 for target shift
    n_windows = len(tokens) // window
    usable = n_windows * window
    tokens = tokens[:usable].reshape(n_windows, window)
    byte_lengths = byte_lengths[:usable].reshape(n_windows, window)

    rng = np.random.default_rng(seed)
    order = rng.permutation(n_windows)
    tokens = tokens[order].reshape(-1)
    byte_lengths = byte_lengths[order].reshape(-1)

    print(f"  {n_windows:,} windows of {window} tokens, shuffled")
    return tokens, byte_lengths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tokenize FineWiki into flat binary files."
    )
    parser.add_argument("--max-train-documents", type=int, default=50_000)
    parser.add_argument("--max-val-documents", type=int, default=2_000)
    parser.add_argument("--tokenizer", default="lfm25")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--dataset", default="HuggingFaceFW/finewiki")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = get_tokenizer(args.tokenizer)
    seq_len = args.seq_len

    # Train split
    print(f"Fetching train documents (max {args.max_train_documents})...")
    t0 = time.time()
    train_docs = iter_documents(
        args.dataset, LANGUAGES, "train", args.max_train_documents
    )
    print(f"  fetched {len(train_docs)} docs in {time.time() - t0:.1f}s")

    print("Tokenizing train split...")
    t0 = time.time()
    train_tokens, train_bytes = tokenize_documents(train_docs, args.tokenizer)
    print(f"  {len(train_tokens):,} tokens in {time.time() - t0:.1f}s")

    print("Shuffling train windows...")
    train_tokens, train_bytes = shuffle_windows(train_tokens, train_bytes, seq_len)

    np.save(DATA_DIR / "tokens_train.npy", train_tokens)
    np.save(DATA_DIR / "bytes_train.npy", train_bytes)

    # Val split — use tail end of the same stream (different docs)
    print(f"Fetching val documents (max {args.max_val_documents})...")
    t0 = time.time()
    val_docs = iter_documents(
        args.dataset,
        LANGUAGES,
        "train",
        args.max_train_documents + args.max_val_documents,
    )
    val_docs = val_docs[args.max_train_documents :]
    print(f"  fetched {len(val_docs)} docs in {time.time() - t0:.1f}s")

    print("Tokenizing val split...")
    t0 = time.time()
    val_tokens, val_bytes = tokenize_documents(val_docs, args.tokenizer)
    print(f"  {len(val_tokens):,} tokens in {time.time() - t0:.1f}s")

    print("Shuffling val windows...")
    val_tokens, val_bytes = shuffle_windows(val_tokens, val_bytes, seq_len)

    np.save(DATA_DIR / "tokens_val.npy", val_tokens)
    np.save(DATA_DIR / "bytes_val.npy", val_bytes)

    # Metadata
    meta = {
        "tokenizer": args.tokenizer,
        "vocab_size": tokenizer.vocab_size,
        "eot_token_id": tokenizer.eot_token_id,
        "seq_len": seq_len,
        "train_documents": len(train_docs),
        "val_documents": len(val_docs),
        "train_tokens": int(len(train_tokens)),
        "val_tokens": int(len(val_tokens)),
    }
    (DATA_DIR / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"\nDone! Saved to {DATA_DIR}/")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)  # HF datasets leaves background threads alive
