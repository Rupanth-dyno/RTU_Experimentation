import argparse
import json
from pathlib import Path
from typing import Dict, List


STRATEGIES = ("Unranked_Baseline", "BM25_Ranked", "MiniLM_Dense_Ranked")


def summarize_budget(
    strategy: str,
    budget: int,
    instances: List[Dict],
    full_logs: List[Dict],
) -> tuple[Dict, List[Dict]]:
    if len(instances) != len(full_logs):
        raise ValueError(f"{strategy}: dataset and result logs have different lengths")

    predictions: List[Dict] = []
    for inst, log in zip(instances, full_logs):
        if inst["instance_id"] != log["instance_id"]:
            raise ValueError(f"{strategy}: instance order mismatch at {inst['instance_id']}")

        found_rank = log["found_at_rank"]
        supported_within_budget = found_rank is not None and (budget == 0 or found_rank <= budget)
        prediction = "SUPPORTS" if supported_within_budget else "NOT_ENOUGH_INFORMATION"
        checks = (
            int(found_rank)
            if supported_within_budget
            else (int(log["iterations_used"]) if budget == 0 else min(budget, len(inst["paragraphs"])))
        )
        predictions.append({
            "instance_id": inst["instance_id"],
            "label": inst["label"],
            "ground_truth": inst["ground_truth"],
            "predicted": prediction,
            "correct": prediction == inst["ground_truth"],
            "iterations_used": checks,
            "found_at_rank": found_rank if supported_within_budget else None,
        })

    positives = [row for row in predictions if row["label"] == 1]
    negatives = [row for row in predictions if row["label"] == 0]
    true_positives = sum(row["predicted"] == "SUPPORTS" for row in positives)
    false_positives = sum(row["predicted"] == "SUPPORTS" for row in negatives)
    true_negatives = len(negatives) - false_positives
    positive_ranks = [row["found_at_rank"] for row in positives if row["found_at_rank"] is not None]
    total_checks = sum(row["iterations_used"] for row in predictions)
    summary = {
        "strategy": strategy,
        "max_checks": budget,
        "instances_evaluated": len(predictions),
        "overall_accuracy": round(sum(row["correct"] for row in predictions) / max(1, len(predictions)), 4),
        "positive_source_accuracy": round(true_positives / max(1, len(positives)), 4),
        "negative_distractor_accuracy": round(true_negatives / max(1, len(negatives)), 4),
        "positive_precision": round(true_positives / max(1, true_positives + false_positives), 4),
        "positive_recall": round(true_positives / max(1, len(positives)), 4),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "true_negatives": true_negatives,
        "false_negatives": len(positives) - true_positives,
        "recall_at_1": round(sum(rank <= 1 for rank in positive_ranks) / max(1, len(positives)), 4),
        "recall_at_3": round(sum(rank <= 3 for rank in positive_ranks) / max(1, len(positives)), 4),
        "recall_at_5": round(sum(rank <= 5 for rank in positive_ranks) / max(1, len(positives)), 4),
        "total_llm_iterations": total_checks,
        "avg_iterations_per_instance": round(total_checks / max(1, len(predictions)), 2),
    }
    return summary, predictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate passage budgets from recorded full-run support ranks; makes no model calls."
    )
    parser.add_argument("--dataset", default="data/thesis_dataset.jsonl")
    parser.add_argument("--detailed-logs", default="results/detailed_logs.json")
    parser.add_argument("--budgets", default="1,3,5,10,20,0", help="Comma-separated limits; 0 means all passages")
    parser.add_argument("--output-dir", default="results/budget_eval")
    args = parser.parse_args()

    try:
        budgets = [int(value) for value in args.budgets.split(",")]
    except ValueError:
        parser.error("--budgets must be a comma-separated list of non-negative integers")
    if not budgets or any(budget < 0 for budget in budgets):
        parser.error("--budgets must contain non-negative integers")

    instances = [
        json.loads(line)
        for line in Path(args.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    full_logs = json.loads(Path(args.detailed_logs).read_text(encoding="utf-8"))
    missing = [strategy for strategy in STRATEGIES if strategy not in full_logs]
    if missing:
        parser.error(f"Detailed logs are missing strategies: {', '.join(missing)}")

    results = []
    detailed: Dict[str, Dict[str, List[Dict]]] = {}
    for budget in budgets:
        budget_key = "all" if budget == 0 else str(budget)
        detailed[budget_key] = {}
        for strategy in STRATEGIES:
            logs = full_logs[strategy]
            summary, predictions = summarize_budget(strategy, budget, instances, logs)
            results.append(summary)
            detailed[budget_key][strategy] = predictions

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "dataset": args.dataset,
        "source_logs": args.detailed_logs,
        "derived_from_full_run_ranks": True,
        "budgets": budgets,
        "results": results,
    }
    (output_dir / "budget_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "detailed_logs.json").write_text(json.dumps(detailed, indent=2), encoding="utf-8")
    lines = [
        "FIXED-BUDGET RETRIEVAL EVALUATION",
        f"Dataset: {args.dataset} ({len(instances)} instances)",
        f"Source: {args.detailed_logs}; derived from recorded full-run ranks; no model calls",
        "",
        "Budget\tStrategy\tAccuracy\tPositive accuracy\tNegative accuracy\tPrecision\tFP\tFN\tChecks",
    ]
    for row in results:
        budget = "all" if row["max_checks"] == 0 else str(row["max_checks"])
        lines.append(
            f"{budget}\t{row['strategy']}\t{row['overall_accuracy']:.4f}\t"
            f"{row['positive_source_accuracy']:.4f}\t{row['negative_distractor_accuracy']:.4f}\t"
            f"{row['positive_precision']:.4f}\t{row['false_positives']}\t"
            f"{row['false_negatives']}\t{row['total_llm_iterations']}"
        )
    (output_dir / "budget_metrics.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
