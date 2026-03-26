"""Tokenizer adapters for GPT-2 and LFM2.5."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import tiktoken
from transformers import AutoTokenizer, PreTrainedTokenizerFast


LFM25_MODEL_ID = "LiquidAI/LFM2.5-1.2B-Base"


@dataclass(frozen=True)
class EncodedDocument:
    token_ids: list[int]
    byte_lengths: list[int]


class TokenizerAdapter:
    name: str
    vocab_size: int
    eot_token_id: int

    def encode_document(self, text: str) -> EncodedDocument:
        raise NotImplementedError


class GPT2TokenizerAdapter(TokenizerAdapter):
    def __init__(self) -> None:
        self.encoding = tiktoken.get_encoding("gpt2")
        self.name = "gpt2"
        self.vocab_size = self.encoding.n_vocab
        self.eot_token_id = self.encoding.eot_token

    def encode_document(self, text: str) -> EncodedDocument:
        token_ids = self.encoding.encode_ordinary(text)
        byte_lengths = [
            len(self.encoding.decode_single_token_bytes(token_id))
            for token_id in token_ids
        ]
        token_ids.append(self.eot_token_id)
        byte_lengths.append(0)
        return EncodedDocument(token_ids=token_ids, byte_lengths=byte_lengths)


class HuggingFaceTokenizerAdapter(TokenizerAdapter):
    def __init__(self, repo_id: str, name: str) -> None:
        tokenizer = AutoTokenizer.from_pretrained(repo_id, use_fast=True)
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            raise TypeError(f"{repo_id} did not resolve to a fast tokenizer.")
        if tokenizer.eos_token_id is None:
            raise ValueError(f"{repo_id} tokenizer does not expose an eos_token_id.")
        self.tokenizer = tokenizer
        self.name = name
        self.vocab_size = len(tokenizer)
        self.eot_token_id = int(tokenizer.eos_token_id)
        self.bos_token_id = (
            int(tokenizer.bos_token_id) if tokenizer.bos_token_id is not None else None
        )

    def encode_document(self, text: str) -> EncodedDocument:
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        token_ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
        byte_lengths = [len(text[start:end].encode("utf-8")) for start, end in offsets]
        # BOS at start, EOT at end — proper document framing
        if self.bos_token_id is not None:
            token_ids.insert(0, self.bos_token_id)
            byte_lengths.insert(0, 0)
        token_ids.append(self.eot_token_id)
        byte_lengths.append(0)
        return EncodedDocument(token_ids=token_ids, byte_lengths=byte_lengths)


@lru_cache(maxsize=4)
def get_tokenizer(name: str) -> TokenizerAdapter:
    if name == "gpt2":
        return GPT2TokenizerAdapter()
    if name == "lfm25":
        return HuggingFaceTokenizerAdapter(LFM25_MODEL_ID, name="lfm25")
    raise ValueError(f"Unsupported tokenizer '{name}'")
