import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

from experiment_utils import (
    connected_groups,
    paired_cluster_bootstrap_difference,
    paired_cluster_permutation,
    summarize_metrics,
)


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
    instance_by_id = {row["instance_id"]: row for row in instances}
    original: Dict[str, List[Dict]] = json.loads(
        Path(args.detailed_logs).read_text(encoding="utf-8")
    )
    missing = [strategy for strategy in STRATEGIES if strategy not in original]
    if missing:
        parser.error(f"Missing strategies in detailed logs: {', '.join(missing)}")
    for strategy in STRATEGIES:
        log_ids = [row["instance_id"] for row in original[strategy]]
        if len(set(log_ids)) != len(log_ids):
            parser.error(f"{strategy} logs contain duplicate instance IDs")
        for row in original[strategy]:
            instance = instance_by_id.get(row["instance_id"])
            if instance is None:
                parser.error(f"{strategy} log references unknown instance {row['instance_id']}")
            if row.get("label") != instance["label"]:
                parser.error(f"{strategy} label mismatch for instance {row['instance_id']}")

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
        f"{first}__vs__{second}": {
            **paired_cluster_permutation(
                logs[first], logs[second], group_for_instance
            ),
            **paired_cluster_bootstrap_difference(
                logs[first], logs[second], group_for_instance
            ),
        }
        for first, second in (
            (STRATEGIES[0], STRATEGIES[1]),
            (STRATEGIES[0], STRATEGIES[2]),
            (STRATEGIES[1], STRATEGIES[2]),
        )
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "post_hoc_exploratory_reanalysis_not_held_out_test",
        "dataset": args.dataset,
        "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        "detailed_logs": args.detailed_logs,
        "detailed_logs_sha256": hashlib.sha256(Path(args.detailed_logs).read_bytes()).hexdigest(),
        "historical_run_dataset_hash": (
            "not captured; instance IDs and labels validated against supplied dataset"
        ),
        "dataset_size": len(instances),
        "independent_connected_groups": len(set(group_ids)),
        "label_source": "automatic citation/distractor construction; weak silver labels, not human gold",
        "latency_warning": (
            "Per-call model latency is recorded, but whole-process wall-clock cannot be reconstructed."
        ),
        "verifier_agreement": "Not measurable from the saved logs.",
        "results": summaries,
        "paired_cluster_tests": paired,
        "interpretation": (
            "The full-dataset K=1 logs are exploratory, not held-out. They cannot establish a "
            "ranking improvement or gold evidence retrieval. Confirmatory claims require human "
            "labels, gold evidence passages, and a predeclared grouped test protocol."
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
    summary_lines = [
        "POST-HOC GROUPED REANALYSIS",
        "Silver-label diagnostics only; not human-gold or held-out test results.",
        f"Dataset SHA-256: {report['dataset_sha256']}",
        f"Detailed-log SHA-256: {report['detailed_logs_sha256']}",
        "Original run dataset hash was not captured; IDs and labels were validated.",
        "Original verifier configuration was not recorded; this is not a current-verifier result.",
        "",
        "Strategy | N | Accuracy [connected-group 95% CI] | Support P/R/F1 | TP/FP/TN/FN",
    ]
    for row in summaries:
        summary_lines.append(
            f"{row['strategy']} | {row['instances_evaluated']} | {row['accuracy']:.3f} "
            f"[{row['accuracy_cluster_bootstrap_95_ci'][0]:.3f}, "
            f"{row['accuracy_cluster_bootstrap_95_ci'][1]:.3f}] | "
            f"{row['support_precision']:.3f}/{row['support_recall']:.3f}/"
            f"{row['support_f1']:.3f} | {row['true_positive']}/{row['false_positive']}/"
            f"{row['true_negative']}/{row['false_negative']}"
        )
    summary_lines.extend(["", "PAIRED CONNECTED-GROUP COMPARISONS"])
    for name, result in paired.items():
        summary_lines.append(
            f"{name}: accuracy difference={result['accuracy_difference']:.3f}, "
            f"95% CI={result['accuracy_difference_cluster_bootstrap_95_ci']}, "
            f"two-sided p={result['two_sided_p_value']:.6f}"
        )
    summary_lines.extend([
        "",
        "No comparison is statistically significant at alpha=0.05.",
        "Gold evidence Recall@K/MRR and human inter-annotator agreement are unavailable.",
        "Per-instance historical logs cannot recover whole-process wall-clock time.",
    ])
    (output / "summary_metrics.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    print((output / "reanalysis.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
