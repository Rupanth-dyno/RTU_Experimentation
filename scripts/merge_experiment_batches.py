import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


STRATEGIES = ("Unranked_Baseline", "BM25_Ranked", "MiniLM_Dense_Ranked")


def format_summary_text(comparison: List[Dict]) -> str:
    return "\n".join(
        [
            "================ FINAL THESIS EXPERIMENT SUMMARY ================",
            *(json.dumps(row, indent=2) for row in comparison),
            "",
        ]
    )


def summarize(strategy: str, logs: List[Dict]) -> Dict:
    total = len(logs)
    positives = [entry for entry in logs if entry["label"] == 1]
    negatives = [entry for entry in logs if entry["label"] == 0]
    positive_ranks = [
        entry["found_at_rank"]
        for entry in positives
        if entry["found_at_rank"] is not None
    ]
    iterations = sum(entry["iterations_used"] for entry in logs)
    ranking_sec = sum(entry["ranking_sec"] for entry in logs)
    llm_sec = sum(entry["llm_sec"] for entry in logs)
    correct = sum(entry["correct"] for entry in logs)
    positive_correct = sum(entry["correct"] for entry in positives)
    negative_correct = sum(entry["correct"] for entry in negatives)

    return {
        "strategy": strategy,
        "instances_evaluated": total,
        "overall_accuracy": round(correct / max(1, total), 4),
        "positive_source_accuracy": round(positive_correct / max(1, len(positives)), 4),
        "negative_distractor_accuracy": round(negative_correct / max(1, len(negatives)), 4),
        "recall_at_1": round(sum(rank <= 1 for rank in positive_ranks) / max(1, len(positives)), 4),
        "recall_at_3": round(sum(rank <= 3 for rank in positive_ranks) / max(1, len(positives)), 4),
        "recall_at_5": round(sum(rank <= 5 for rank in positive_ranks) / max(1, len(positives)), 4),
        "mrr": round(sum(1.0 / rank for rank in positive_ranks) / max(1, len(positives)), 4),
        "total_llm_iterations": iterations,
        "avg_iterations_per_instance": round(iterations / max(1, total), 2),
        "total_ranking_sec": round(ranking_sec, 3),
        "total_llm_sec": round(llm_sec, 3),
        "total_wall_clock_sec": round(ranking_sec + llm_sec, 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge completed experiment batch outputs.")
    parser.add_argument("--dataset", default="data/thesis_dataset.jsonl")
    parser.add_argument("--batches-dir", default="results/batches")
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    dataset_rows = [
        json.loads(line)
        for line in Path(args.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataset_ids = [entry["instance_id"] for entry in dataset_rows]
    batches_dir = Path(args.batches_dir)
    batch_files = sorted(
        batches_dir.glob("batch_*/detailed_logs.json"),
        key=lambda path: tuple(
            int(part)
            for part in re.fullmatch(r"batch_(\d+)_(\d+)", path.parent.name).groups()
        ),
    )
    if not batch_files:
        parser.error(f"No batch outputs found in {batches_dir}")

    combined: Dict[str, List[Dict]] = {strategy: [] for strategy in STRATEGIES}
    for detailed_path in batch_files:
        batch_logs = json.loads(detailed_path.read_text(encoding="utf-8"))
        missing = [strategy for strategy in STRATEGIES if strategy not in batch_logs]
        if missing:
            parser.error(f"{detailed_path} is missing strategies: {', '.join(missing)}")
        for strategy in STRATEGIES:
            combined[strategy].extend(batch_logs[strategy])

    expected_ids = [entry["instance_id"] for entry in combined[STRATEGIES[0]]]
    if len(expected_ids) != len(set(expected_ids)):
        parser.error("Batch outputs contain duplicate instance IDs")
    if any([entry["instance_id"] for entry in combined[name]] != expected_ids for name in STRATEGIES[1:]):
        parser.error("Strategies have different instance IDs or ordering")
    if expected_ids != dataset_ids:
        parser.error(
            f"Batch outputs cover {len(expected_ids)} instances, but the dataset contains "
            f"{len(dataset_ids)}; refusing to write incomplete full-dataset results"
        )

    comparison = [summarize(strategy, combined[strategy]) for strategy in STRATEGIES]
    base = comparison[0]
    for summary in comparison:
        iterations_saved = base["total_llm_iterations"] - summary["total_llm_iterations"]
        time_saved = base["total_wall_clock_sec"] - summary["total_wall_clock_sec"]
        summary["iterations_saved_vs_baseline"] = iterations_saved
        summary["iterations_saved_pct"] = round(
            100.0 * iterations_saved / max(1, base["total_llm_iterations"]), 2
        )
        summary["net_time_saved_sec_vs_baseline"] = round(time_saved, 3)
        summary["net_time_saved_pct"] = round(
            100.0 * time_saved / max(0.001, base["total_wall_clock_sec"]), 2
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary_metrics.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (output_dir / "detailed_logs.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    (output_dir / "summary_metrics.txt").write_text(format_summary_text(comparison), encoding="utf-8")
    print(f"Merged {len(expected_ids)} instances from {len(batch_files)} batches into {output_dir}")
    print(format_summary_text(comparison), end="")


if __name__ == "__main__":
    main()
