"""Retrieval-focused evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from src.model import PackingTransformer


@dataclass
class RetrievalMetrics:
    top1: float
    top5: float
    top10: float
    mrr: float
    average_rank: float
    median_rank: float
    ranks: list[int]


def _compute_ranks(scores: torch.Tensor, target_ids: torch.Tensor) -> torch.Tensor:
    target_scores = scores.gather(1, target_ids.unsqueeze(1))
    num_greater = (scores > target_scores).sum(dim=1).to(dtype=torch.float32)
    num_equal = (scores == target_scores).sum(dim=1).to(dtype=torch.float32)
    # Use average rank under ties so collapsed score vectors do not look perfect.
    return 1.0 + num_greater + 0.5 * (num_equal - 1.0)


@torch.no_grad()
def compute_retrieval_metrics(
    model: PackingTransformer,
    prediction: torch.Tensor,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    chunk_size: int = 128,
) -> RetrievalMetrics:
    valid_targets = target_ids[target_mask]
    if valid_targets.numel() == 0:
        raise RuntimeError("No valid targets available for retrieval metrics.")
    valid_prediction = prediction[target_mask]
    embedding_table = model.token_embeddings.weight

    ranks: list[torch.Tensor] = []
    top_hits = {1: 0, 5: 0, 10: 0}

    for start in range(0, valid_prediction.shape[0], chunk_size):
        end = start + chunk_size
        pred_chunk = valid_prediction[start:end]
        target_chunk = valid_targets[start:end]

        if model.config.objective.name == "ce_baseline":
            scores = pred_chunk
        else:
            pred_norm = pred_chunk.pow(2).sum(dim=1, keepdim=True)
            emb_norm = embedding_table.pow(2).sum(dim=1).unsqueeze(0)
            squared_distance = (
                pred_norm + emb_norm - (2.0 * pred_chunk @ embedding_table.t())
            )
            scores = -(squared_distance / model.config.model.d_model)

        chunk_ranks = _compute_ranks(scores, target_chunk)
        ranks.append(chunk_ranks)

        topk = torch.topk(scores, k=10, dim=1).indices
        for k in top_hits:
            top_hits[k] += (
                (topk[:, :k] == target_chunk.unsqueeze(1)).any(dim=1).sum().item()
            )

    rank_tensor = torch.cat(ranks).float()
    n = rank_tensor.numel()
    reciprocal = rank_tensor.reciprocal().mean().item()
    return RetrievalMetrics(
        top1=top_hits[1] / n,
        top5=top_hits[5] / n,
        top10=top_hits[10] / n,
        mrr=reciprocal,
        average_rank=float(rank_tensor.mean().item()),
        median_rank=float(rank_tensor.median().item()),
        ranks=rank_tensor.to(dtype=torch.int64).cpu().tolist(),
    )


@torch.no_grad()
def compute_ce_nll(
    prediction: torch.Tensor,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
) -> tuple[float, int]:
    logits = prediction[target_mask]
    targets = target_ids[target_mask]
    nll = torch.nn.functional.cross_entropy(logits, targets, reduction="sum")
    return float(nll.item()), int(targets.numel())


@torch.no_grad()
def compute_harmax_nll(
    model: PackingTransformer,
    prediction: torch.Tensor,
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    chunk_size: int = 128,
    exponent: float | None = None,
) -> tuple[float, int]:
    valid_targets = target_ids[target_mask]
    valid_prediction = prediction[target_mask]
    embedding_table = model.token_embeddings.weight
    pow_n = float(exponent if exponent is not None else model.config.model.d_model)
    total_nll = 0.0
    total_tokens = 0

    for start in range(0, valid_prediction.shape[0], chunk_size):
        end = start + chunk_size
        pred_chunk = valid_prediction[start:end]
        target_chunk = valid_targets[start:end]
        pred_norm = pred_chunk.pow(2).sum(dim=1, keepdim=True)
        emb_norm = embedding_table.pow(2).sum(dim=1).unsqueeze(0)
        squared_distance = (
            pred_norm + emb_norm - (2.0 * pred_chunk @ embedding_table.t())
        )
        dist = (squared_distance / model.config.model.d_model).clamp_min(1e-12)
        dist = dist / dist.min(dim=1, keepdim=True).values.clamp_min(1e-12)
        weights = dist.pow(-pow_n)
        probs = weights / weights.sum(dim=1, keepdim=True)
        target_probs = probs.gather(1, target_chunk.unsqueeze(1)).clamp_min(1e-12)
        total_nll += float((-target_probs.log()).sum().item())
        total_tokens += int(target_chunk.numel())

    return total_nll, total_tokens
