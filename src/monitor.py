"""Embedding health monitoring utilities."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _to_cpu_float(embedding_table: torch.Tensor) -> torch.Tensor:
    return embedding_table.detach().to(device="cpu", dtype=torch.float32)


@torch.no_grad()
def singular_values(embedding_table: torch.Tensor) -> list[float]:
    values = torch.linalg.svdvals(_to_cpu_float(embedding_table))
    return values.cpu().tolist()


@torch.no_grad()
def effective_dimensionality(embedding_table: torch.Tensor) -> float:
    sigma = torch.tensor(singular_values(embedding_table), dtype=torch.float32)
    numerator = sigma.sum().pow(2)
    denominator = sigma.pow(2).sum().clamp_min(1e-8)
    return float((numerator / denominator).item())


@torch.no_grad()
def average_pairwise_cosine_similarity(
    embedding_table: torch.Tensor,
    num_pairs: int = 10_000,
) -> float:
    embedding_table = _to_cpu_float(embedding_table)
    n = embedding_table.shape[0]
    idx_a = torch.randint(0, n, (num_pairs,))
    idx_b = torch.randint(0, n, (num_pairs,))
    emb = F.normalize(embedding_table, dim=1)
    sim = (emb[idx_a] * emb[idx_b]).sum(dim=1)
    return float(sim.mean().item())


@torch.no_grad()
def nearest_neighbor_collision_rate(
    embedding_table: torch.Tensor,
    chunk_size: int = 1024,
) -> float:
    emb = F.normalize(_to_cpu_float(embedding_table), dim=1)
    n = emb.shape[0]
    nearest = torch.empty(n, dtype=torch.long)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sims = emb[start:end] @ emb.t()
        row_idx = torch.arange(start, end)
        sims[torch.arange(end - start), row_idx] = -1.0
        nearest[start:end] = sims.argmax(dim=1)

    counts = torch.bincount(nearest, minlength=n).float()
    collisions = counts[counts > 1].sum()
    return float((collisions / n).item())


@torch.no_grad()
def matrix_effective_rank(matrix: torch.Tensor) -> float:
    matrix = _to_cpu_float(matrix)
    if matrix.ndim != 2 or matrix.numel() == 0:
        return 0.0
    sigma = torch.linalg.svdvals(matrix)
    numerator = sigma.sum().pow(2)
    denominator = sigma.pow(2).sum().clamp_min(1e-8)
    return float((numerator / denominator).item())
