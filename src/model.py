"""Shared transformer trunk and experiment heads."""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import ExperimentConfig, ModelConfig, ObjectiveConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return x * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, base: int = 10_000) -> None:
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = torch.einsum("bt,d->btd", position_ids.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


class SelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, dim = x.shape
        q = (
            self.q_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch_size, seq_len, self.n_heads, self.head_dim)
            .transpose(1, 2)
        )

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)
        return self.out_proj(y)


class MLP(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.fc = nn.Linear(config.d_model, config.ffn_dim, bias=False)
        self.proj = nn.Linear(config.ffn_dim, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.ffn_norm = RMSNorm(config.d_model)
        self.attn = SelfAttention(config)
        self.mlp = MLP(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
        token_mask: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, attention_mask)
        x = x * token_mask
        x = x + self.mlp(self.ffn_norm(x))
        x = x * token_mask
        return x


@dataclass
class ModelOutput:
    hidden_states: torch.Tensor
    prediction: torch.Tensor


class PackingTransformer(nn.Module):
    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self.config = config
        model_config = config.model
        self.token_embeddings = nn.Embedding(
            model_config.vocab_size, model_config.d_model
        )
        self.blocks = nn.ModuleList(
            [Block(model_config) for _ in range(model_config.n_layers)]
        )
        self.final_norm = RMSNorm(model_config.d_model)
        self.rope = RotaryEmbedding(
            model_config.d_model // model_config.n_heads, model_config.rope_base
        )
        self.prediction_head = nn.Linear(
            model_config.d_model, model_config.d_model, bias=False
        )
        self.output_scale_log = nn.Parameter(
            torch.tensor(math.log(model_config.output_scale_init))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(
            self.token_embeddings.weight,
            mean=0.0,
            std=self.config.model.embedding_init_std,
        )
        if self.config.model.prediction_head_init_std == 0.0:
            nn.init.zeros_(self.prediction_head.weight)
        else:
            nn.init.normal_(
                self.prediction_head.weight,
                mean=0.0,
                std=self.config.model.prediction_head_init_std,
            )
        for module in self.modules():
            if isinstance(module, nn.Linear) and module is not self.prediction_head:
                nn.init.xavier_uniform_(module.weight)

    def build_attention_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = attention_mask.shape
        causal = torch.tril(
            torch.ones(
                (seq_len, seq_len), dtype=torch.bool, device=attention_mask.device
            )
        )
        key_mask = attention_mask[:, None, None, :]
        query_mask = attention_mask[:, None, :, None]
        return (
            causal[None, None, :, :]
            & key_mask
            & query_mask.expand(batch_size, 1, seq_len, seq_len)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> ModelOutput:
        x = self.token_embeddings(input_ids)
        token_mask = attention_mask.unsqueeze(-1).to(dtype=x.dtype)
        x = x * token_mask
        cos, sin = self.rope(position_ids)
        attn_mask = self.build_attention_mask(attention_mask)
        for block in self.blocks:
            x = block(x, cos, sin, attn_mask, token_mask)
        x = self.final_norm(x) * token_mask

        if self.config.objective.name == "ce_baseline":
            prediction = x @ self.token_embeddings.weight.t()
        else:
            prediction = self.prediction_head(x)
            if self.config.objective.learned_output_scale:
                prediction = prediction * self.output_scale_log.exp()
        return ModelOutput(hidden_states=x, prediction=prediction)


def _load_lejepa() -> tuple[type[nn.Module], type[nn.Module]]:
    try:
        from lejepa.multivariate import SlicingUnivariateTest
        from lejepa.univariate import EppsPulley

        return SlicingUnivariateTest, EppsPulley
    except ModuleNotFoundError:
        repo_root = Path(__file__).resolve().parent.parent
        submodule_root = repo_root / "submodules" / "lejepa"
        if str(submodule_root) not in sys.path:
            sys.path.append(str(submodule_root))
        from lejepa.multivariate import SlicingUnivariateTest
        from lejepa.univariate import EppsPulley

        return SlicingUnivariateTest, EppsPulley


def build_sigreg_loss(objective: ObjectiveConfig) -> nn.Module | None:
    if objective.name != "l2_sigreg" or objective.lambda_sigreg <= 0.0:
        return None
    slicing_cls, epps_cls = _load_lejepa()
    try:
        univariate_test = epps_cls(num_points=objective.epps_pulley_points)
    except TypeError:
        univariate_test = epps_cls(n_points=objective.epps_pulley_points)
    return slicing_cls(univariate_test=univariate_test, num_slices=objective.num_slices)


def compute_l2_loss(
    model: PackingTransformer,
    prediction: torch.Tensor,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    valid_targets = target_ids[target_mask]
    if valid_targets.numel() == 0:
        raise RuntimeError("No valid targets found in batch.")
    target_embeddings = model.token_embeddings(valid_targets)
    predicted_embeddings = prediction[target_mask]
    squared_error = (predicted_embeddings - target_embeddings).pow(2).sum(dim=-1)
    normalized_squared_error = squared_error / model.config.model.d_model
    return normalized_squared_error.mean()


def compute_ce_loss(prediction: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(prediction.transpose(1, 2), target_ids, ignore_index=-100)
