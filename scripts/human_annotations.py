import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List


LABELS = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFORMATION")
csv.field_size_limit(sys.maxsize)
NEGATIVE_CATEGORIES = (
    "clearly_unrelated",
    "semantically_similar",
    "shared_terminology",
    "same_experiment_different_conditions",
    "contradictory",
    "same_topic_insufficient_evidence",
    "other",
)
ANNOTATION_FIELDS = [
    "annotator_1_verdict",
    "annotator_1_evidence_passage_ids",
    "annotator_1_negative_category",
    "annotator_2_verdict",
    "annotator_2_evidence_passage_ids",
    "annotator_2_negative_category",
]


def create_template(dataset: Path, output: Path, sample_pairs: int, seed: int) -> None:
    rows = [json.loads(line) for line in dataset.read_text(encoding="utf-8").splitlines() if line.strip()]
    pairs: Dict[str, List[Dict]] = {}
    for row in rows:
        pairs.setdefault(str(row["pair_id"]), []).append(row)
    unique_pairs = {}
    for pair_id, pair_rows in pairs.items():
        positive = next((row for row in pair_rows if row["label"] == 1), None)
        if positive is not None:
            key = (
                positive["citing_arxiv_id"],
                positive["source_arxiv_id"],
                positive["claim_context"],
            )
            unique_pairs.setdefault(key, pair_id)
    if sample_pairs > len(unique_pairs):
        raise ValueError(f"Requested {sample_pairs} unique claim/source pairs, only {len(unique_pairs)} are available")

    selected_keys = random.Random(seed).sample(sorted(unique_pairs), sample_pairs)
    selected_ids = [unique_pairs[key] for key in selected_keys]
    annotations = []
    for pair_id in selected_ids:
        pair_rows = sorted(pairs[pair_id], key=lambda row: row["label"], reverse=True)
        if len(pair_rows) != 2:
            continue
        for row in pair_rows:
            annotations.append({
                "pair_id": row["pair_id"],
                "instance_id": row["instance_id"],
                "label_source": "automatic_silver_label_only",
                "citing_arxiv_id": row["citing_arxiv_id"],
                "source_arxiv_id": row["source_arxiv_id"],
                "source_title": row["source_title"],
                "claim_context": row["claim_context"],
                "allowed_verdicts": "|".join(LABELS),
                "allowed_negative_categories": "|".join(NEGATIVE_CATEGORIES),
                "candidate_passages_json": json.dumps(
                    [{"passage_id": p.get("passage_id"), "text": p["text"]} for p in row["paragraphs"]],
                    ensure_ascii=False,
                ),
                **{field: "" for field in ANNOTATION_FIELDS},
            })

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(annotations[0].keys()))
        writer.writeheader()
        writer.writerows(annotations)
    print(json.dumps({
        "requested_pair_count": sample_pairs,
        "annotated_pair_count": len(annotations) // 2,
        "instance_count": len(annotations),
        "seed": seed,
        "output": str(output),
    }, indent=2))


def analyze_annotations(path: Path, output: Path) -> None:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    complete = [
        row for row in rows
        if row["annotator_1_verdict"] in LABELS and row["annotator_2_verdict"] in LABELS
    ]
    if not complete:
        raise ValueError("No rows have valid labels from both annotators")
    confusion = {a: {b: 0 for b in LABELS} for a in LABELS}
    for row in complete:
        confusion[row["annotator_1_verdict"]][row["annotator_2_verdict"]] += 1
    observed = sum(confusion[label][label] for label in LABELS) / len(complete)
    first = Counter(row["annotator_1_verdict"] for row in complete)
    second = Counter(row["annotator_2_verdict"] for row in complete)
    expected = sum(first[label] * second[label] for label in LABELS) / len(complete) ** 2
    kappa = (observed - expected) / (1 - expected) if expected < 1 else 1.0
    evidence_jaccards = []
    for row in complete:
        if row["annotator_1_verdict"] != "SUPPORTS" or row["annotator_2_verdict"] != "SUPPORTS":
            continue
        evidence_1 = set(filter(None, row["annotator_1_evidence_passage_ids"].split(";")))
        evidence_2 = set(filter(None, row["annotator_2_evidence_passage_ids"].split(";")))
        if evidence_1 or evidence_2:
            evidence_jaccards.append(
                len(evidence_1 & evidence_2) / max(1, len(evidence_1 | evidence_2))
            )
    category_pairs = [
        (row["annotator_1_negative_category"], row["annotator_2_negative_category"])
        for row in complete
        if row["annotator_1_negative_category"] in NEGATIVE_CATEGORIES
        and row["annotator_2_negative_category"] in NEGATIVE_CATEGORIES
    ]
    report = {
        "annotation_rows": len(rows),
        "jointly_labeled_rows": len(complete),
        "labels": list(LABELS),
        "confusion_matrix_annotator_1_rows_annotator_2_columns": confusion,
        "exact_agreement": observed,
        "cohen_kappa": kappa,
        "joint_support_rows": len(evidence_jaccards),
        "evidence_passage_set_jaccard_mean": (
            sum(evidence_jaccards) / len(evidence_jaccards) if evidence_jaccards else None
        ),
        "negative_category_pair_agreement": (
            sum(first == second for first, second in category_pairs) / len(category_pairs)
            if category_pairs else None
        ),
        "negative_category_jointly_labeled_rows": len(category_pairs),
        "negative_category_counts": {
            category: sum(
                row[f"annotator_{annotator}_negative_category"] == category
                for row in complete for annotator in (1, 2)
            )
            for category in NEGATIVE_CATEGORIES
        },
        "note": "Agreement is based only on human-filled annotation fields; silver labels are not treated as gold.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and analyze a manually annotated validation sample.")
    parser.add_argument("--mode", choices=("template", "analyze"), default="template")
    parser.add_argument("--dataset", default="data/thesis_dataset.jsonl")
    parser.add_argument("--sample-pairs", type=int, default=150)
    parser.add_argument("--seed", type=int, default=231104)
    parser.add_argument("--annotations", default="data/human_validation_annotations.csv")
    parser.add_argument("--output", default="results/human_annotation_agreement.json")
    args = parser.parse_args()
    if args.sample_pairs <= 0:
        parser.error("--sample-pairs must be positive")
    if args.mode == "template":
        create_template(Path(args.dataset), Path(args.annotations), args.sample_pairs, args.seed)
    else:
        analyze_annotations(Path(args.annotations), Path(args.output))


if __name__ == "__main__":
    main()
