import argparse
import json
from pathlib import Path
from typing import Dict, List

from experiment_utils import connected_groups, paired_cluster_permutation, summarize_metrics


STRATEGIES = ("Unranked_Baseline", "BM25_Ranked", "MiniLM_Dense_Ranked")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reanalyse saved results with pair/source-clustered statistics."
    )
    parser.add_argument("--dataset", default="data/thesis_dataset.jsonl")
    parser.add_argument("--detailed-logs", default="results/detailed_logs.json")
    parser.add_argument("--output-dir", default="results/legacy_exploratory_reanalysis")
    args = parser.parse_args()

    instances = [
        json.loads(line)
        for line in Path(args.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    group_ids, _ = connected_groups(instances)
    group_for_instance = {
        row["instance_id"]: group_ids[index] for index, row in enumerate(instances)
    }
    original: Dict[str, List[Dict]] = json.loads(
        Path(args.detailed_logs).read_text(encoding="utf-8")
    )
    missing = [strategy for strategy in STRATEGIES if strategy not in original]
    if missing:
        parser.error(f"Missing strategies in detailed logs: {', '.join(missing)}")

    logs: Dict[str, List[Dict]] = {}
    summaries = []
    for strategy in STRATEGIES:
        logs[strategy] = []
        for row in original[strategy]:
            normalized = dict(row)
            normalized["first_verdict"] = (
                "SUPPORTS" if row.get("found_at_rank") == 1 else "NOT_ENOUGH_INFORMATION"
            )
            normalized["model_api_sec"] = row.get("llm_sec", 0.0)
            normalized["silver_label_error_category"] = (
                "positive_missed_within_budget"
                if row["ground_truth"] == "SUPPORTS" and row["predicted"] != "SUPPORTS"
                else "support_verdict_on_negative_distractor"
                if row["ground_truth"] != "SUPPORTS" and row["predicted"] == "SUPPORTS"
                else "correct_positive_decision"
                if row["ground_truth"] == "SUPPORTS"
                else "correct_negative_decision"
            )
            logs[strategy].append(normalized)
        summaries.append(
            summarize_metrics(strategy, logs[strategy], group_for_instance)
        )

    paired = {
        f"{first}__vs__{second}": paired_cluster_permutation(
            logs[first], logs[second], group_for_instance
        )
        for first, second in (
            (STRATEGIES[0], STRATEGIES[1]),
            (STRATEGIES[0], STRATEGIES[2]),
            (STRATEGIES[1], STRATEGIES[2]),
        )
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "exploratory_full-dataset_reanalysis_not_held_out_test",
        "dataset_size": len(instances),
        "independent_connected_groups": len(set(group_ids)),
        "label_source": "automatic citation/distractor construction; weak silver labels, not human gold",
        "latency_warning": (
            "Saved logs reuse cached model-call latencies; they are not fresh wall-clock measurements."
        ),
        "verifier_agreement": "Not measurable from the saved logs.",
        "results": summaries,
        "paired_cluster_tests": paired,
        "interpretation": (
            "The historical exhaustive results are exploratory. Identical 60% accuracy and small "
            "timing differences do not establish a ranking improvement; confirmatory claims require "
            "the grouped held-out test and fresh fixed-budget runs."
        ),
    }
    (output / "reanalysis.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "reanalysis.txt").write_text(
        report["status"].upper()
        + "\n"
        + report["interpretation"]
        + "\n"
        + report["latency_warning"]
        + "\n"
        + "\n".join(json.dumps(row, indent=2) for row in summaries)
        + "\n\nPaired cluster-level permutation tests:\n"
        + json.dumps(paired, indent=2)
        + "\n",
        encoding="utf-8",
    )
    print((output / "reanalysis.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
