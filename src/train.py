"""Training entrypoint for L2 and CE experiment variants."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from dataclasses import replace

from dotenv import load_dotenv

load_dotenv()  # load project .env before any library imports that read env vars

import numpy as np  # noqa: E402
import torch  # noqa: E402

try:
    import wandb  # noqa: E402
except ImportError:
    wandb = None  # type: ignore[assignment]

from src.config import ExperimentConfig, get_config  # noqa: E402
from src.data import Batch, build_dataloaders  # noqa: E402
from src.eval import compute_ce_nll, compute_harmax_nll, compute_retrieval_metrics  # noqa: E402
from src.model import (  # noqa: E402
    PackingTransformer,
    build_sigreg_loss,
    compute_ce_loss,
    compute_l2_loss,
)
from src.monitor import (  # noqa: E402
    average_pairwise_cosine_similarity,
    effective_dimensionality,
    matrix_effective_rank,
    nearest_neighbor_collision_rate,
    singular_values,
)
from src.tokenization import get_tokenizer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train L2 or CE packing experiment.")
    parser.add_argument(
        "--config", required=True, help="Preset name, e.g. l2_debug or ce_debug"
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging.",
    )
    parser.add_argument(
        "--wandb-project",
        default="l2-sigreg-lm",
        help="W&B project name (default: l2-sigreg-lm).",
    )
    parser.add_argument(
        "--wandb-entity",
        default=None,
        help="W&B team/entity name (required if personal entities are disabled).",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def autodetect_device(device_name: str) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_optimizer(
    config: ExperimentConfig, model: PackingTransformer
) -> torch.optim.Optimizer:
    embedding_lr_scale = (
        config.optim.ce_embedding_lr_scale
        if config.objective.name == "ce_baseline"
        else config.optim.l2_embedding_lr_scale
    )
    embedding_params = [model.token_embeddings.weight]
    other_params = [
        parameter
        for name, parameter in model.named_parameters()
        if name != "token_embeddings.weight"
    ]
    return torch.optim.AdamW(
        [
            {
                "params": embedding_params,
                "lr": config.optim.learning_rate * embedding_lr_scale,
                "lr_scale": embedding_lr_scale,
            },
            {
                "params": other_params,
                "lr": config.optim.learning_rate,
                "lr_scale": 1.0,
            },
        ],
        betas=(config.optim.beta1, config.optim.beta2),
        weight_decay=config.optim.weight_decay,
    )


def learning_rate_for_step(config: ExperimentConfig, step: int) -> float:
    total_steps = max(config.runtime.train_steps, 1)
    warmup_steps = min(
        total_steps, max(1, math.ceil(total_steps * config.optim.warmup_frac))
    )
    decay_steps = min(
        total_steps - warmup_steps,
        max(0, math.ceil(total_steps * config.optim.decay_frac)),
    )
    stable_steps = max(0, total_steps - warmup_steps - decay_steps)

    if step < warmup_steps:
        return config.optim.learning_rate * (step + 1) / warmup_steps

    if step < warmup_steps + stable_steps:
        return config.optim.learning_rate

    if decay_steps == 0:
        return config.optim.learning_rate

    decay_step = step - warmup_steps - stable_steps
    progress = min(max(decay_step / decay_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = config.optim.learning_rate * config.optim.min_lr_ratio
    return min_lr + cosine * (config.optim.learning_rate - min_lr)


def apply_learning_rate(optimizer: torch.optim.Optimizer, learning_rate: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate * param_group.get("lr_scale", 1.0)


def finalize_config(config: ExperimentConfig) -> ExperimentConfig:
    tokenizer = get_tokenizer(config.data.tokenizer_name)
    return replace(config, model=replace(config.model, vocab_size=tokenizer.vocab_size))


def ensure_dirs(config: ExperimentConfig) -> None:
    config.output_path.mkdir(parents=True, exist_ok=True)
    config.checkpoint_path.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def checkpoint_path(config: ExperimentConfig, step: int) -> Path:
    return config.checkpoint_path / f"step_{step:06d}.pt"


def save_checkpoint(
    config: ExperimentConfig,
    step: int,
    model: PackingTransformer,
    optimizer: torch.optim.Optimizer,
) -> None:
    torch.save(
        {
            "step": step,
            "config": config.name,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        checkpoint_path(config, step),
    )


def sigreg_weight_for_step(config: ExperimentConfig, step: int) -> float:
    if config.objective.name != "l2_sigreg" or config.objective.lambda_sigreg <= 0.0:
        return 0.0
    warmup_steps = max(config.objective.sigreg_warmup_steps, 0)
    if warmup_steps == 0:
        return config.objective.lambda_sigreg
    progress = min(max((step + 1) / warmup_steps, 0.0), 1.0)
    return config.objective.lambda_sigreg * progress


def build_sigreg_inputs(
    config: ExperimentConfig,
    model: PackingTransformer,
    prediction: torch.Tensor,
    target_mask: torch.Tensor,
    target_ids: torch.Tensor,
) -> torch.Tensor:
    pieces: list[torch.Tensor] = []

    device = prediction.device
    vocab_size = model.token_embeddings.num_embeddings
    active_vocab = target_ids[target_mask].unique()

    # Always include embeddings of active batch tokens
    if config.objective.sigreg_include_active_predictions and active_vocab.numel() > 0:
        pieces.append(model.token_embeddings(active_vocab))

    # Add random vocab embeddings (excluding active tokens)
    random_vocab_size = max(config.objective.sigreg_random_vocab_size, 0)
    if random_vocab_size > 0:
        if active_vocab.numel() >= vocab_size:
            sampled_ids = torch.arange(vocab_size, device=device)
        else:
            sampled_ids = torch.randperm(vocab_size, device=device)
            if active_vocab.numel() > 0:
                excl_mask = torch.zeros(vocab_size, dtype=torch.bool, device=device)
                excl_mask[active_vocab] = True
                sampled_ids = sampled_ids[~excl_mask[sampled_ids]]
            sampled_ids = sampled_ids[: min(random_vocab_size, sampled_ids.numel())]
        if sampled_ids.numel() > 0:
            pieces.append(model.token_embeddings(sampled_ids))

    if not pieces:
        # Fallback: random vocab sample
        sampled_ids = torch.randperm(vocab_size, device=device)[:1024]
        return model.token_embeddings(sampled_ids)
    if len(pieces) == 1:
        return pieces[0]
    return torch.cat(pieces, dim=0)


def compute_loss(
    config: ExperimentConfig,
    model: PackingTransformer,
    batch: Batch,
    sigreg_loss_fn: torch.nn.Module | None,
    step: int,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    output = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        position_ids=batch.position_ids,
    )
    target_mask = batch.target_ids.ne(-100)

    if config.objective.name == "ce_baseline":
        loss = compute_ce_loss(output.prediction, batch.target_ids)
        metrics = {"loss_ce": float(loss.item())}
    else:
        pred_loss = compute_l2_loss(
            model, output.prediction, batch.target_ids, target_mask
        )
        sigreg_value = output.prediction.new_tensor(0.0)
        sigreg_weight = sigreg_weight_for_step(config, step)
        if sigreg_loss_fn is not None:
            sigreg_input = build_sigreg_inputs(
                config=config,
                model=model,
                prediction=output.prediction,
                target_mask=target_mask,
                target_ids=batch.target_ids,
            )
            sigreg_value = sigreg_loss_fn(sigreg_input)
        pred_scale = config.objective.pred_loss_scale
        loss = (pred_scale * pred_loss) + (sigreg_weight * sigreg_value)
        metrics = {
            "loss_pred": float(pred_loss.item()),
            "loss_sigreg": float(sigreg_value.item()),
            "sigreg_weight": sigreg_weight,
            "output_scale": float(model.output_scale_log.exp().item()),
        }
    metrics["loss_total"] = float(loss.item())
    return loss, metrics, output.prediction


def assert_target_embedding_grads(model: PackingTransformer, batch: Batch) -> None:
    grad = model.token_embeddings.weight.grad
    if grad is None:
        raise AssertionError("Expected embedding gradients to exist.")
    nonzero_rows = grad.abs().sum(dim=1).gt(0).nonzero(as_tuple=False).flatten()
    participating_tokens = torch.cat(
        [
            batch.input_ids[batch.attention_mask],
            batch.target_ids[batch.target_ids.ne(-100)],
        ]
    ).unique()
    extra = set(nonzero_rows.tolist()) - set(participating_tokens.tolist())
    if extra:
        preview = sorted(extra)[:10]
        raise AssertionError(
            f"Embedding gradient touched rows outside the batch token set, e.g. {preview}"
        )


def _module_grad_norm(module: torch.nn.Module) -> float:
    total = 0.0
    for parameter in module.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().pow(2).sum().item())
    return math.sqrt(total)


def collect_gradient_metrics(
    model: PackingTransformer, batch: Batch
) -> dict[str, float]:
    total_sq = 0.0
    max_abs = 0.0
    num_grad_tensors = 0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        total_sq += float(grad.pow(2).sum().item())
        max_abs = max(max_abs, float(grad.abs().max().item()))
        num_grad_tensors += 1

    embedding_grad = model.token_embeddings.weight.grad
    active_rows = 0
    active_density = 0.0
    active_grad_rank = 0.0
    active_grad_norm_mean = 0.0
    if embedding_grad is not None:
        row_norms = embedding_grad.detach().pow(2).sum(dim=1).sqrt()
        active_mask = row_norms > 0
        active_rows = int(active_mask.sum().item())
        active_density = active_rows / max(int(embedding_grad.shape[0]), 1)
        if active_rows > 0:
            active_grad_norm_mean = float(row_norms[active_mask].mean().item())
            active_grad_rank = matrix_effective_rank(embedding_grad[active_mask])

    participating_tokens = torch.cat(
        [
            batch.input_ids[batch.attention_mask],
            batch.target_ids[batch.target_ids.ne(-100)],
        ]
    ).unique()

    embedding_norm = _module_grad_norm(model.token_embeddings)
    prediction_head_norm = _module_grad_norm(model.prediction_head)
    trunk_norm = math.sqrt(
        max(total_sq - (embedding_norm**2) - (prediction_head_norm**2), 0.0)
    )

    return {
        "grad_global_norm": math.sqrt(total_sq),
        "grad_max_abs": max_abs,
        "grad_num_tensors": float(num_grad_tensors),
        "grad_embedding_norm": embedding_norm,
        "grad_prediction_head_norm": prediction_head_norm,
        "grad_trunk_norm": trunk_norm,
        "grad_embedding_active_rows": float(active_rows),
        "grad_embedding_active_density": active_density,
        "grad_embedding_active_row_norm_mean": active_grad_norm_mean,
        "grad_embedding_effective_rank": active_grad_rank,
        "batch_unique_tokens": float(participating_tokens.numel()),
    }


@torch.no_grad()
def run_validation(
    config: ExperimentConfig,
    model: PackingTransformer,
    val_iter: object,
    sigreg_loss_fn: torch.nn.Module | None,
    device: torch.device,
    max_val_batches: int = 50,
    autocast_ctx: torch.autocast | None = None,
) -> dict[str, float | list[float]]:
    model.eval()
    total_batches = 0
    total_positions = 0
    total_bytes = 0
    top1 = 0.0
    top5 = 0.0
    top10 = 0.0
    reciprocal_rank_sum = 0.0
    average_rank_sum = 0.0
    rank_values: list[int] = []
    total_nll = 0.0
    total_nll_tokens = 0
    total_harmax_nll = 0.0
    total_harmax_tokens = 0
    total_pred_loss = 0.0
    total_sigreg_loss = 0.0

    for _ in range(max_val_batches):
        batch = next(val_iter).to(device)
        if autocast_ctx is not None:
            with autocast_ctx:
                output = model(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    position_ids=batch.position_ids,
                )
        else:
            output = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                position_ids=batch.position_ids,
            )
        prediction = output.prediction
        total_batches += 1
        target_mask = batch.target_ids.ne(-100)
        retrieval = compute_retrieval_metrics(
            model=model,
            prediction=prediction,
            target_ids=batch.target_ids,
            target_mask=target_mask,
        )
        positions = int(target_mask.sum().item())
        total_positions += positions
        total_bytes += int(batch.target_byte_lengths[target_mask].sum().item())
        top1 += retrieval.top1 * positions
        top5 += retrieval.top5 * positions
        top10 += retrieval.top10 * positions
        reciprocal_rank_sum += retrieval.mrr * positions
        average_rank_sum += retrieval.average_rank * positions
        rank_values.extend(retrieval.ranks)
        if config.objective.name == "ce_baseline":
            nll, nll_tokens = compute_ce_nll(prediction, batch.target_ids, target_mask)
            total_nll += nll
            total_nll_tokens += nll_tokens
        else:
            pred_loss = compute_l2_loss(
                model, prediction, batch.target_ids, target_mask
            )
            total_pred_loss += float(pred_loss.item())
            if sigreg_loss_fn is not None:
                sigreg_input = build_sigreg_inputs(
                    config=config,
                    model=model,
                    prediction=prediction,
                    target_mask=target_mask,
                    target_ids=batch.target_ids,
                )
                sigreg_value = sigreg_loss_fn(sigreg_input)
                total_sigreg_loss += float(sigreg_value.item())
            harmax_nll, harmax_tokens = compute_harmax_nll(
                model=model,
                prediction=prediction,
                target_ids=batch.target_ids,
                target_mask=target_mask,
            )
            total_harmax_nll += harmax_nll
            total_harmax_tokens += harmax_tokens

    metrics = {
        "val_top1": top1 / max(total_positions, 1),
        "val_top5": top5 / max(total_positions, 1),
        "val_top10": top10 / max(total_positions, 1),
        "val_mrr": reciprocal_rank_sum / max(total_positions, 1),
        "val_average_rank": average_rank_sum / max(total_positions, 1),
        "val_median_rank": float(np.median(rank_values)) if rank_values else 0.0,
    }
    emb = model.token_embeddings.weight.detach()
    metrics["singular_values"] = singular_values(emb)

    if total_nll_tokens > 0:
        metrics["val_ce_loss"] = total_nll / total_nll_tokens
        metrics["val_nll"] = total_nll / total_nll_tokens
        metrics["val_ppl"] = math.exp(total_nll / total_nll_tokens)
        metrics["val_bits_per_token"] = total_nll / (math.log(2.0) * total_nll_tokens)
        if total_bytes > 0:
            metrics["val_bpb"] = total_nll / (math.log(2.0) * total_bytes)

    if config.objective.name == "l2_sigreg":
        metrics["val_pred_loss"] = total_pred_loss / max(total_batches, 1)
        metrics["val_sigreg_loss"] = total_sigreg_loss / max(total_batches, 1)
        metrics["output_scale"] = float(model.output_scale_log.exp().item())
        metrics["effective_dimensionality"] = effective_dimensionality(emb)
        metrics["avg_pairwise_cosine"] = average_pairwise_cosine_similarity(emb)
        metrics["nearest_neighbor_collision_rate"] = nearest_neighbor_collision_rate(
            emb
        )
        if total_harmax_tokens > 0:
            metrics["val_harmax_nll"] = total_harmax_nll / total_harmax_tokens
            metrics["val_harmax_ppl"] = math.exp(total_harmax_nll / total_harmax_tokens)
            metrics["val_harmax_bits_per_token"] = total_harmax_nll / (
                math.log(2.0) * total_harmax_tokens
            )
            if total_bytes > 0:
                metrics["val_harmax_bpb"] = total_harmax_nll / (
                    math.log(2.0) * total_bytes
                )

    model.train()
    return metrics


def _config_to_flat_dict(config: ExperimentConfig) -> dict[str, object]:
    """Flatten nested config dataclasses into a single dict for wandb."""
    from dataclasses import asdict

    raw = asdict(config)
    flat: dict[str, object] = {"name": raw.pop("name")}
    for section_name, section in raw.items():
        if isinstance(section, dict):
            for key, value in section.items():
                flat[f"{section_name}/{key}"] = value
        else:
            flat[section_name] = section
    return flat


class WandbLogger:
    """Deferred wandb logger — only creates a run on first log call."""

    def __init__(
        self,
        config: ExperimentConfig,
        project: str,
        entity: str | None,
        enabled: bool,
    ) -> None:
        self._config = config
        self._project = project
        self._entity = entity
        self._enabled = enabled and wandb is not None
        self._initialized = False
        if enabled and wandb is None:
            print("WARNING: wandb requested but not installed. Logging locally only.")

    def _ensure_init(self) -> None:
        if self._initialized or not self._enabled:
            return
        wandb.init(
            project=self._project,
            entity=self._entity,
            name=self._config.name,
            config=_config_to_flat_dict(self._config),
            reinit="finish_previous",
        )
        self._initialized = True

    def log(self, payload: dict[str, object], step: int) -> None:
        if not self._enabled:
            return
        self._ensure_init()
        loggable: dict[str, object] = {}
        for key, value in payload.items():
            if key in ("step", "split"):
                continue
            if isinstance(value, (int, float)):
                prefix = payload.get("split", "train")
                loggable[f"{prefix}/{key}"] = value
            elif key == "singular_values" and isinstance(value, list):
                loggable["val/singular_values"] = wandb.Histogram(value)
        wandb.log(loggable, step=step)

    def finish(self) -> None:
        if self._initialized and wandb is not None:
            wandb.finish()


def _aggregate_metrics(
    buffer: list[dict[str, float]],
) -> dict[str, float]:
    """Average a list of per-step metric dicts."""
    if not buffer:
        return {}
    keys = buffer[0].keys()
    return {k: sum(d[k] for d in buffer) / len(buffer) for k in keys}


def main() -> None:
    args = parse_args()
    config = finalize_config(get_config(args.config))
    ensure_dirs(config)
    set_seed(config.runtime.seed)
    device = autodetect_device(config.runtime.device)
    wb = WandbLogger(config, args.wandb_project, args.wandb_entity, not args.no_wandb)

    train_iter, val_iter, _ = build_dataloaders(config.data)

    model = PackingTransformer(config).to(device)
    optimizer = build_optimizer(config, model)
    sigreg_loss_fn = build_sigreg_loss(config.objective)
    if sigreg_loss_fn is not None:
        sigreg_loss_fn = sigreg_loss_fn.to(device)

    # Mixed precision: bf16 forward/backward, fp32 master weights and loss
    use_amp = device.type == "cuda" and config.runtime.dtype == "bfloat16"
    autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_amp else None

    log_path = config.output_path / f"{config.name}.jsonl"
    if log_path.exists():
        log_path.unlink()
    start_time = time.time()
    metric_buffer: list[dict[str, float]] = []

    def _run_and_log_val(step: int) -> None:
        payload = {
            "step": step,
            "split": "val",
            "elapsed_s": time.time() - start_time,
            **run_validation(
                config,
                model,
                val_iter,
                sigreg_loss_fn,
                device,
                autocast_ctx=autocast_ctx,
            ),
        }
        append_jsonl(log_path, payload)
        wb.log(payload, step)
        print(json.dumps(payload))

    # Initial validation baseline before any training
    _run_and_log_val(0)

    for step in range(config.runtime.train_steps):
        batch = next(train_iter).to(device)
        learning_rate = learning_rate_for_step(config, step)
        apply_learning_rate(optimizer, learning_rate)

        optimizer.zero_grad(set_to_none=True)
        if autocast_ctx is not None:
            with autocast_ctx:
                loss, train_metrics, _ = compute_loss(
                    config, model, batch, sigreg_loss_fn, step
                )
        else:
            loss, train_metrics, _ = compute_loss(
                config, model, batch, sigreg_loss_fn, step
            )
        loss.backward()

        if (
            config.objective.name == "l2_sigreg"
            and config.runtime.assert_target_embedding_grads
        ):
            assert_target_embedding_grads(model, batch)

        if config.runtime.log_gradient_metrics:
            train_metrics.update(collect_gradient_metrics(model, batch))

        torch.nn.utils.clip_grad_norm_(model.parameters(), config.optim.grad_clip_norm)
        optimizer.step()

        train_metrics["learning_rate"] = learning_rate
        metric_buffer.append(train_metrics)

        if (step + 1) % config.runtime.log_every == 0:
            aggregated = _aggregate_metrics(metric_buffer)
            metric_buffer.clear()
            payload = {
                "step": step,
                "split": "train",
                "elapsed_s": time.time() - start_time,
                **aggregated,
            }
            append_jsonl(log_path, payload)
            wb.log(payload, step)
            print(json.dumps(payload))

        if (step + 1) % config.runtime.eval_every == 0:
            _run_and_log_val(step)

        if (step + 1) % config.runtime.checkpoint_every == 0:
            save_checkpoint(config, step + 1, model, optimizer)

    save_checkpoint(config, config.runtime.train_steps, model, optimizer)
    wb.finish()


if __name__ == "__main__":
    main()
