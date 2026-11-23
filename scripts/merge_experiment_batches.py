import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from experiment_utils import connected_groups, paired_cluster_permutation, summarize_metrics


STRATEGIES = ("Unranked_Baseline", "BM25_Ranked", "MiniLM_Dense_Ranked")


def format_summary_text(comparison: List[Dict], paired_tests: Dict) -> str:
    return "\n".join(
        [
            "================ MERGED BATCH SUMMARY ================",
            "Exploratory full-dataset batch merge; not a held-out test result.",
            "Final decision task: binary SUPPORTS vs NOT_ENOUGH_INFORMATION.",
            *(json.dumps(row, indent=2) for row in comparison),
            "Paired cluster-level permutation tests:",
            json.dumps(paired_tests, indent=2),
            "",
        ]
    )


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
    measured_seconds: Dict[str, float] = defaultdict(float)
    for detailed_path in batch_files:
        batch_logs = json.loads(detailed_path.read_text(encoding="utf-8"))
        missing = [strategy for strategy in STRATEGIES if strategy not in batch_logs]
        if missing:
            parser.error(f"{detailed_path} is missing strategies: {', '.join(missing)}")
        for strategy in STRATEGIES:
            combined[strategy].extend(batch_logs[strategy])
        summary_path = detailed_path.with_name("summary_metrics.json")
        if summary_path.exists():
            for entry in json.loads(summary_path.read_text(encoding="utf-8")):
                measured_seconds[entry["strategy"]] += entry.get(
                    "measured_strategy_wall_clock_sec",
                    entry.get("sum_ranking_and_model_api_sec", entry.get("total_wall_clock_sec", 0.0)),
                )

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

    all_group_ids, _ = connected_groups(dataset_rows)
    group_for_instance = {
        row["instance_id"]: all_group_ids[index]
        for index, row in enumerate(dataset_rows)
    }
    comparison = [
        summarize_metrics(strategy, combined[strategy], group_for_instance)
        for strategy in STRATEGIES
    ]
    all_requests_fresh = all(
        entry.get("cache_misses") is not None
        and entry.get("cache_hits") == 0
        for strategy in STRATEGIES
        for entry in combined[strategy]
    )
    base = comparison[0]
    for summary in comparison:
        summary["positive_source_accuracy"] = summary["support_recall"]
        summary["negative_distractor_accuracy"] = summary["negative_accuracy"]
        summary["total_llm_iterations"] = summary["total_checks"]
        summary["avg_iterations_per_instance"] = summary["mean_checks"]
        summary["measured_strategy_wall_clock_sec"] = round(
            measured_seconds[summary["strategy"]], 3
        )
        iterations_saved = base["total_checks"] - summary["total_checks"]
        time_saved = base["measured_strategy_wall_clock_sec"] - summary["measured_strategy_wall_clock_sec"]
        summary["iterations_saved_vs_baseline"] = iterations_saved
        summary["iterations_saved_pct"] = round(
            100.0 * iterations_saved / max(1, base["total_checks"]), 2
        )
        summary["net_time_saved_sec_vs_baseline"] = round(time_saved, 3)
        summary["net_time_saved_pct"] = round(
            100.0 * time_saved / max(0.001, base["measured_strategy_wall_clock_sec"]), 2
        )

    paired_tests = {
        f"{first}__vs__{second}": paired_cluster_permutation(
            combined[first], combined[second], group_for_instance
        )
        for first, second in (
            (STRATEGIES[0], STRATEGIES[1]),
            (STRATEGIES[0], STRATEGIES[2]),
            (STRATEGIES[1], STRATEGIES[2]),
        )
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary_metrics.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (output_dir / "detailed_logs.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    (output_dir / "paired_strategy_tests.json").write_text(json.dumps(paired_tests, indent=2), encoding="utf-8")
    (output_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "evaluation_status": "exploratory_full_dataset_batch_merge",
                "held_out_grouped_split": False,
                "label_source": "automatic citation/distractor construction; weak silver labels",
                "all_model_requests_fresh": all_requests_fresh,
                "note": "Batching is execution infrastructure; aggregate metrics are computed over merged instance logs.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (output_dir / "summary_metrics.txt").write_text(
        format_summary_text(comparison, paired_tests), encoding="utf-8"
    )
    print(f"Merged {len(expected_ids)} instances from {len(batch_files)} batches into {output_dir}")
    print(format_summary_text(comparison, paired_tests), end="")


if __name__ == "__main__":
    main()
