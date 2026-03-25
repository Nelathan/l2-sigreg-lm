"""Compare final metrics between experiment variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CE and L2 experiment logs.")
    parser.add_argument("--l2-log", default="results/l2_debug.jsonl")
    parser.add_argument("--ce-log", default="results/ce_debug.jsonl")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--label", default="Comparison")
    parser.add_argument("--left-label", default="L2 + SIGReg")
    parser.add_argument("--right-label", default="CE Baseline")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def filter_split(records: list[dict], split: str) -> list[dict]:
    return [record for record in records if record.get("split") == split]


def last_record(records: list[dict], split: str) -> dict:
    split_records = filter_split(records, split)
    if not split_records:
        raise RuntimeError(f"No '{split}' records found.")
    return split_records[-1]


def format_metric(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, int):
        return str(value)
    if abs(value) >= 100:
        return f"{value:.2f}"
    return f"{value:.4f}"


def pick_metric(record: dict, keys: list[str]) -> float | None:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def plot_metric(
    train_records_l2: list[dict],
    train_records_ce: list[dict],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
    left_label: str,
    right_label: str,
) -> None:
    val_l2 = filter_split(train_records_l2, "val")
    val_ce = filter_split(train_records_ce, "val")
    l2_points = [(row["step"], row[metric_key]) for row in val_l2 if metric_key in row]
    ce_points = [(row["step"], row[metric_key]) for row in val_ce if metric_key in row]
    if not l2_points and not ce_points:
        return
    plt.figure(figsize=(8, 5))
    if l2_points:
        plt.plot(
            [step for step, _ in l2_points],
            [value for _, value in l2_points],
            label=left_label,
        )
    if ce_points:
        plt.plot(
            [step for step, _ in ce_points],
            [value for _, value in ce_points],
            label=right_label,
        )
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_metric_pair(
    train_records_l2: list[dict],
    train_records_ce: list[dict],
    left_metric_key: str,
    right_metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
    left_label: str,
    right_label: str,
) -> None:
    val_l2 = filter_split(train_records_l2, "val")
    val_ce = filter_split(train_records_ce, "val")
    l2_points = [
        (row["step"], row[left_metric_key]) for row in val_l2 if left_metric_key in row
    ]
    ce_points = [
        (row["step"], row[right_metric_key])
        for row in val_ce
        if right_metric_key in row
    ]
    if not l2_points and not ce_points:
        return
    plt.figure(figsize=(8, 5))
    if l2_points:
        plt.plot(
            [step for step, _ in l2_points],
            [value for _, value in l2_points],
            label=left_label,
        )
    if ce_points:
        plt.plot(
            [step for step, _ in ce_points],
            [value for _, value in ce_points],
            label=right_label,
        )
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_spectrum(
    l2_final: dict,
    ce_final: dict,
    output_path: Path,
    left_label: str,
    right_label: str,
) -> None:
    if "singular_values" not in l2_final:
        return
    plt.figure(figsize=(8, 5))
    plt.plot(l2_final["singular_values"], label=left_label)
    if "singular_values" in ce_final:
        plt.plot(ce_final["singular_values"], label=right_label)
    plt.xlabel("Index")
    plt.ylabel("Singular value")
    plt.title("Embedding Singular Value Spectrum")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    l2_records = load_jsonl(Path(args.l2_log))
    ce_records = load_jsonl(Path(args.ce_log))
    l2_final = last_record(l2_records, "val")
    ce_final = last_record(ce_records, "val")

    primary_loss_keys = ["val_pred_loss", "val_ce_loss", "val_nll"]
    aux_nll_keys = ["val_harmax_nll", "val_nll"]
    aux_ppl_keys = ["val_harmax_ppl", "val_ppl"]
    aux_bpt_keys = ["val_harmax_bits_per_token", "val_bits_per_token"]
    aux_bpb_keys = ["val_harmax_bpb", "val_bpb"]

    rows = [
        ("MRR", l2_final.get("val_mrr"), ce_final.get("val_mrr")),
        (
            "Average Rank",
            l2_final.get("val_average_rank"),
            ce_final.get("val_average_rank"),
        ),
        (
            "Median Rank",
            l2_final.get("val_median_rank"),
            ce_final.get("val_median_rank"),
        ),
        ("Top-10 Accuracy", l2_final.get("val_top10"), ce_final.get("val_top10")),
        ("Top-5 Accuracy", l2_final.get("val_top5"), ce_final.get("val_top5")),
        ("Top-1 Accuracy", l2_final.get("val_top1"), ce_final.get("val_top1")),
        (
            "Primary Loss",
            pick_metric(l2_final, primary_loss_keys),
            pick_metric(ce_final, primary_loss_keys),
        ),
        (
            "SIGReg Loss",
            l2_final.get("val_sigreg_loss"),
            ce_final.get("val_sigreg_loss"),
        ),
        (
            "Aux NLL",
            pick_metric(l2_final, aux_nll_keys),
            pick_metric(ce_final, aux_nll_keys),
        ),
        (
            "Aux Perplexity",
            pick_metric(l2_final, aux_ppl_keys),
            pick_metric(ce_final, aux_ppl_keys),
        ),
        (
            "Aux Bits/Token",
            pick_metric(l2_final, aux_bpt_keys),
            pick_metric(ce_final, aux_bpt_keys),
        ),
        (
            "Aux BPB",
            pick_metric(l2_final, aux_bpb_keys),
            pick_metric(ce_final, aux_bpb_keys),
        ),
        (
            "Effective Dimensionality",
            l2_final.get("effective_dimensionality"),
            ce_final.get("effective_dimensionality"),
        ),
        (
            "Avg Cosine Similarity",
            l2_final.get("avg_pairwise_cosine"),
            ce_final.get("avg_pairwise_cosine"),
        ),
        (
            "Nearest Neighbor Collision Rate",
            l2_final.get("nearest_neighbor_collision_rate"),
            ce_final.get("nearest_neighbor_collision_rate"),
        ),
    ]

    header = f"| Metric | {args.left_label} | {args.right_label} | Delta |"
    divider = "|---|---:|---:|---:|"
    lines = [header, divider]
    for name, l2_value, ce_value in rows:
        delta = None if l2_value is None or ce_value is None else l2_value - ce_value
        lines.append(
            f"| {name} | {format_metric(l2_value)} | {format_metric(ce_value)} | {format_metric(delta)} |"
        )

    report = "\n".join(lines)
    report_path = output_dir / "comparison.md"
    report_path.write_text(report + "\n", encoding="utf-8")
    print(report)

    plot_metric(
        l2_records,
        ce_records,
        metric_key="val_mrr",
        ylabel="MRR",
        title=f"{args.label}: Validation MRR",
        output_path=output_dir / "mrr_over_time.png",
        left_label=args.left_label,
        right_label=args.right_label,
    )
    plot_metric(
        l2_records,
        ce_records,
        metric_key="val_average_rank",
        ylabel="Average Rank",
        title=f"{args.label}: Validation Average Rank",
        output_path=output_dir / "average_rank_over_time.png",
        left_label=args.left_label,
        right_label=args.right_label,
    )
    plot_metric(
        l2_records,
        ce_records,
        metric_key="val_median_rank",
        ylabel="Median Rank",
        title=f"{args.label}: Validation Median Rank",
        output_path=output_dir / "median_rank_over_time.png",
        left_label=args.left_label,
        right_label=args.right_label,
    )
    plot_metric(
        l2_records,
        ce_records,
        metric_key="val_top10",
        ylabel="Top-10 Accuracy",
        title=f"{args.label}: Validation Top-10 Accuracy",
        output_path=output_dir / "top10_over_time.png",
        left_label=args.left_label,
        right_label=args.right_label,
    )
    plot_metric_pair(
        l2_records,
        ce_records,
        left_metric_key="val_harmax_bpb"
        if any("val_harmax_bpb" in row for row in filter_split(l2_records, "val"))
        else "val_bpb",
        right_metric_key="val_harmax_bpb"
        if any("val_harmax_bpb" in row for row in filter_split(ce_records, "val"))
        else "val_bpb",
        ylabel="BPB",
        title=f"{args.label}: Validation BPB",
        output_path=output_dir / "bpb_over_time.png",
        left_label=args.left_label,
        right_label=args.right_label,
    )
    plot_spectrum(
        l2_final,
        ce_final,
        output_path=output_dir / "singular_values.png",
        left_label=args.left_label,
        right_label=args.right_label,
    )


if __name__ == "__main__":
    main()
