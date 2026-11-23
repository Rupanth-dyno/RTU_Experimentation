import hashlib
import json
import math
import random
from collections import defaultdict
from itertools import product
from typing import Dict, List, Sequence, Tuple


SPLIT_NAMES = ("train", "development", "test")
SPLIT_RATIOS = {"train": 0.6, "development": 0.2, "test": 0.2}


def connected_groups(instances: Sequence[Dict]) -> Tuple[List[str], Dict[str, List[int]]]:
    parent = list(range(len(instances)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    first_by_key: Dict[Tuple[str, str], int] = {}
    for index, row in enumerate(instances):
        keys = (
            ("pair", str(row["pair_id"])),
            ("citing", str(row["citing_arxiv_id"])),
            ("source", str(row["source_arxiv_id"])),
        )
        for key in keys:
            previous = first_by_key.setdefault(key, index)
            union(index, previous)

    members: Dict[int, List[int]] = defaultdict(list)
    for index in range(len(instances)):
        members[find(index)].append(index)

    group_for_index: Dict[int, str] = {}
    group_rows: Dict[str, List[int]] = {}
    for indices in members.values():
        group_keys = sorted(
            f"{instances[index]['citing_arxiv_id']}|{instances[index]['source_arxiv_id']}|"
            f"{instances[index]['pair_id']}"
            for index in indices
        )
        group_id = hashlib.sha256("\n".join(group_keys).encode("utf-8")).hexdigest()[:16]
        group_rows[group_id] = indices
        for index in indices:
            group_for_index[index] = group_id
    return [group_for_index[index] for index in range(len(instances))], group_rows


def make_grouped_split(instances: Sequence[Dict], seed: int = 231104) -> Dict:
    group_for_index, groups = connected_groups(instances)
    rng = random.Random(seed)
    shuffled_groups = list(groups.items())
    rng.shuffle(shuffled_groups)
    shuffled_groups.sort(key=lambda item: len(item[1]), reverse=True)

    targets = {name: SPLIT_RATIOS[name] * len(instances) for name in SPLIT_NAMES}
    assigned: Dict[str, List[str]] = {name: [] for name in SPLIT_NAMES}
    counts = {name: 0 for name in SPLIT_NAMES}
    for group_id, indices in shuffled_groups:
        split_name = max(
            SPLIT_NAMES,
            key=lambda name: (targets[name] - counts[name]) / targets[name],
        )
        assigned[split_name].append(group_id)
        counts[split_name] += len(indices)

    index_to_split = {
        index: name
        for name, group_ids in assigned.items()
        for group_id in group_ids
        for index in groups[group_id]
    }
    splits = {
        name: [instances[index]["instance_id"] for index in range(len(instances)) if index_to_split[index] == name]
        for name in SPLIT_NAMES
    }
    return {
        "version": 1,
        "seed": seed,
        "ratios": SPLIT_RATIOS,
        "actual_instance_counts": counts,
        "groups": assigned,
        "group_for_instance": {
            instances[index]["instance_id"]: group_for_index[index]
            for index in range(len(instances))
        },
        "splits": splits,
    }


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> List[float]:
    if total == 0:
        return [0.0, 0.0]
    estimate = successes / total
    denominator = 1 + z * z / total
    center = (estimate + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(estimate * (1 - estimate) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _counts(logs: Sequence[Dict]) -> Dict[str, int]:
    truth_positive = lambda row: row["ground_truth"] == "SUPPORTS"
    pred_positive = lambda row: row["predicted"] == "SUPPORTS"
    return {
        "tp": sum(truth_positive(row) and pred_positive(row) for row in logs),
        "fp": sum(not truth_positive(row) and pred_positive(row) for row in logs),
        "tn": sum(not truth_positive(row) and not pred_positive(row) for row in logs),
        "fn": sum(truth_positive(row) and not pred_positive(row) for row in logs),
    }


def _point_metrics(logs: Sequence[Dict]) -> Dict[str, float]:
    counts = _counts(logs)
    tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
    total = max(1, tp + fp + tn + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "accuracy": (tp + tn) / total,
        "precision": precision,
        "recall": recall,
        "support_f1": 2 * precision * recall / max(1e-12, precision + recall),
        "negative_accuracy": tn / max(1, tn + fp),
    }


def summarize_metrics(
    strategy: str,
    logs: Sequence[Dict],
    group_for_instance: Dict[str, str],
    seed: int = 231104,
    bootstrap_samples: int = 2000,
) -> Dict:
    counts = _counts(logs)
    point = _point_metrics(logs)
    by_group: Dict[str, List[Dict]] = defaultdict(list)
    for row in logs:
        group_id = group_for_instance.get(row["instance_id"], row.get("cluster_id", row["instance_id"]))
        by_group[group_id].append(row)

    rng = random.Random(seed)
    group_rows = list(by_group.values())
    samples = {name: [] for name in point}
    if group_rows:
        for _ in range(bootstrap_samples):
            resampled = [row for _ in group_rows for row in rng.choice(group_rows)]
            current = _point_metrics(resampled)
            for name in samples:
                samples[name].append(current[name])
    intervals = {
        name: [
            round(sorted(values)[int(0.025 * (len(values) - 1))], 4),
            round(sorted(values)[int(0.975 * (len(values) - 1))], 4),
        ]
        if values else [0.0, 0.0]
        for name, values in samples.items()
    }
    positives = [row for row in logs if row["ground_truth"] == "SUPPORTS"]
    negatives = [row for row in logs if row["ground_truth"] != "SUPPORTS"]
    checks = sorted(row["iterations_used"] for row in logs)
    median_checks = checks[(len(checks) - 1) // 2] if checks else 0
    p95_checks = checks[min(len(checks) - 1, math.ceil(0.95 * len(checks)) - 1)] if checks else 0
    first_pass_tp = sum(
        row["ground_truth"] == "SUPPORTS" and row.get("first_verdict") == "SUPPORTS"
        for row in logs
    )
    first_pass_fp = sum(
        row["ground_truth"] != "SUPPORTS" and row.get("first_verdict") == "SUPPORTS"
        for row in logs
    )
    categories = {
        "positive_missed_within_budget": sum(
            row["ground_truth"] == "SUPPORTS" and row["predicted"] != "SUPPORTS" for row in logs
        ),
        "support_verdict_on_negative_distractor": counts["fp"],
        "correct_positive_decision": counts["tp"],
        "correct_negative_decision": counts["tn"],
    }
    return {
        "strategy": strategy,
        "instances_evaluated": len(logs),
        "independent_groups": len(by_group),
        "confusion_matrix": {
            "labels": ["NOT_ENOUGH_INFORMATION", "SUPPORTS"],
            "rows_true_columns_predicted": [[counts["tn"], counts["fp"]], [counts["fn"], counts["tp"]]],
        },
        "true_positive": counts["tp"],
        "false_positive": counts["fp"],
        "true_negative": counts["tn"],
        "false_negative": counts["fn"],
        "accuracy": round(point["accuracy"], 4),
        "accuracy_naive_row_level_wilson_95_ci": [
            round(x, 4) for x in wilson_interval(counts["tp"] + counts["tn"], len(logs))
        ],
        "accuracy_cluster_bootstrap_95_ci": intervals["accuracy"],
        "support_precision": round(point["precision"], 4),
        "support_precision_cluster_bootstrap_95_ci": intervals["precision"],
        "support_recall": round(point["recall"], 4),
        "support_recall_cluster_bootstrap_95_ci": intervals["recall"],
        "support_f1": round(point["support_f1"], 4),
        "support_f1_cluster_bootstrap_95_ci": intervals["support_f1"],
        "negative_accuracy": round(point["negative_accuracy"], 4),
        "negative_accuracy_cluster_bootstrap_95_ci": intervals["negative_accuracy"],
        "first_pass_support_recall": round(first_pass_tp / max(1, len(positives)), 4),
        "first_pass_support_accuracy_on_positives": round(first_pass_tp / max(1, len(positives)), 4),
        "first_pass_support_precision": round(first_pass_tp / max(1, first_pass_tp + first_pass_fp), 4),
        "median_checks": median_checks,
        "p95_checks": p95_checks,
        "mean_checks": round(sum(checks) / max(1, len(checks)), 2),
        "total_checks": sum(checks),
        "cache_hits": (
            sum(row["cache_hits"] for row in logs)
            if all("cache_hits" in row for row in logs)
            else None
        ),
        "cache_misses": (
            sum(row["cache_misses"] for row in logs)
            if all("cache_misses" in row for row in logs)
            else None
        ),
        "ranking_sec": round(sum(row.get("ranking_sec", 0.0) for row in logs), 3),
        "model_api_sec": round(sum(row.get("model_api_sec", row.get("llm_sec", 0.0)) for row in logs), 3),
        "sum_per_instance_recorded_elapsed_sec": round(
            sum(row.get("total_sec", 0.0) for row in logs), 3
        ),
        "outcome_diagnostic_categories_not_gold_attributions": categories,
        "verifier_agreement": "Not measured; run scripts/evaluate_verifier_agreement.py",
    }


def paired_cluster_permutation(
    first_logs: Sequence[Dict],
    second_logs: Sequence[Dict],
    group_for_instance: Dict[str, str],
) -> Dict:
    second_by_id = {row["instance_id"]: row for row in second_logs}
    if len(second_by_id) != len(second_logs):
        raise ValueError("Duplicate instance IDs in second strategy logs")
    differences: Dict[str, int] = defaultdict(int)
    for first in first_logs:
        second = second_by_id.get(first["instance_id"])
        if second is None:
            raise ValueError(f"Missing paired instance {first['instance_id']}")
        group_id = group_for_instance.get(first["instance_id"], first["instance_id"])
        differences[group_id] += int(first["correct"]) - int(second["correct"])

    values = list(differences.values())
    observed = abs(sum(values))
    if len(values) <= 16:
        extreme = sum(abs(sum(value * sign for value, sign in zip(values, signs))) >= observed for signs in product((-1, 1), repeat=len(values)))
        permutations = 2 ** len(values)
        p_value = extreme / permutations
    else:
        rng = random.Random(231104)
        permutations = 20000
        extreme = sum(
            abs(sum(value * rng.choice((-1, 1)) for value in values)) >= observed
            for _ in range(permutations)
        )
        p_value = (extreme + 1) / (permutations + 1)
    return {
        "groups": len(values),
        "accuracy_difference": round(sum(values) / max(1, len(first_logs)), 4),
        "two_sided_p_value": round(p_value, 6),
        "method": "two-sided paired sign-flip permutation over connected pair/citing/source groups",
    }


def paired_cluster_bootstrap_difference(
    first_logs: Sequence[Dict],
    second_logs: Sequence[Dict],
    group_for_instance: Dict[str, str],
    seed: int = 231104,
    repetitions: int = 2000,
) -> Dict:
    second_by_id = {row["instance_id"]: row for row in second_logs}
    if len(second_by_id) != len(second_logs):
        raise ValueError("Duplicate instance IDs in second strategy logs")
    first_ids = [row["instance_id"] for row in first_logs]
    if len(set(first_ids)) != len(first_ids):
        raise ValueError("Duplicate instance IDs in first strategy logs")
    if set(first_ids) != set(second_by_id):
        raise ValueError("Strategy logs do not contain the same instance IDs")

    differences: Dict[str, List[int]] = defaultdict(list)
    for first in first_logs:
        second = second_by_id[first["instance_id"]]
        group_id = group_for_instance.get(first["instance_id"], first["instance_id"])
        differences[group_id].append(int(first["correct"]) - int(second["correct"]))
    if not differences:
        raise ValueError("Cannot bootstrap an empty paired evaluation")

    clusters = list(differences.values())
    rng = random.Random(seed)
    estimates = []
    for _ in range(repetitions):
        sample = [rng.choice(clusters) for _ in clusters]
        denominator = sum(len(cluster) for cluster in sample)
        estimates.append(sum(sum(cluster) for cluster in sample) / denominator)
    estimates.sort()
    low = estimates[int(0.025 * (repetitions - 1))]
    high = estimates[int(0.975 * (repetitions - 1))]
    return {
        "accuracy_difference_cluster_bootstrap_95_ci": [round(low, 4), round(high, 4)],
        "bootstrap_unit": "connected_component(pair_id,citing_arxiv_id,source_arxiv_id)",
        "bootstrap_groups": len(clusters),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": seed,
    }
