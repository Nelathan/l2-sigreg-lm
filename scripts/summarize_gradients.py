"""Summarize gradient pressure from training logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize gradient metrics from a JSONL log."
    )
    parser.add_argument("log_path")
    return parser.parse_args()


def load_train_records(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return [record for record in records if record.get("split") == "train"]


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def main() -> None:
    args = parse_args()
    records = load_train_records(Path(args.log_path))
    if not records:
        raise SystemExit("No train records found.")

    embedding_share = [
        record["grad_embedding_norm"] / max(record["grad_global_norm"], 1e-12)
        for record in records
        if "grad_embedding_norm" in record and "grad_global_norm" in record
    ]
    head_share = [
        record["grad_prediction_head_norm"] / max(record["grad_global_norm"], 1e-12)
        for record in records
        if "grad_prediction_head_norm" in record and "grad_global_norm" in record
    ]
    trunk_share = [
        record["grad_trunk_norm"] / max(record["grad_global_norm"], 1e-12)
        for record in records
        if "grad_trunk_norm" in record and "grad_global_norm" in record
    ]
    active_density = [
        record["grad_embedding_active_density"]
        for record in records
        if "grad_embedding_active_density" in record
    ]
    active_rank = [
        record["grad_embedding_effective_rank"]
        for record in records
        if "grad_embedding_effective_rank" in record
    ]

    summary = {
        "steps": len(records),
        "mean_grad_global_norm": mean(
            [record["grad_global_norm"] for record in records]
        ),
        "mean_embedding_grad_share": mean(embedding_share),
        "mean_prediction_head_grad_share": mean(head_share),
        "mean_trunk_grad_share": mean(trunk_share),
        "mean_embedding_active_density": mean(active_density),
        "mean_embedding_effective_rank": mean(active_rank),
        "last_step": records[-1]["step"],
        "last_grad_global_norm": records[-1].get("grad_global_norm"),
        "last_embedding_grad_share": records[-1].get("grad_embedding_norm", 0.0)
        / max(records[-1].get("grad_global_norm", 1e-12), 1e-12),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
