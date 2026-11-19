
import json
from collections import Counter
from pathlib import Path


dataset_file = Path("processed/ranking_dataset.jsonl")

examples = []

with dataset_file.open("r", encoding="utf-8") as file:

    for line in file:
        examples.append(json.loads(line))




unique_claim_ids = set(
    example["claim_id"]
    for example in examples
)

candidate_passage_counts = [
    len(example["candidate_passages"])
    for example in examples
]

gold_evidence_counts = [
    len(example["gold_evidence_passage_ids"])
    for example in examples
]



label_counts = Counter()

for example in examples:

    for label in example["evidence_labels"]:
        label_counts[label] += 1




multiple_evidence_examples = sum(
    len(example["evidence_groups"]) > 1
    for example in examples
)




mixed_label_examples = sum(
    len(example["evidence_labels"]) > 1
    for example in examples
)




print("========== DATASET VALIDATION ==========")

print("Claim-document examples:", len(examples))

print("Unique claims:", len(unique_claim_ids))

print(
    "Examples per claim:",
    round(len(examples) / len(unique_claim_ids), 2)
)

print(
    "Average candidate passages:",
    round(
        sum(candidate_passage_counts)
        / len(candidate_passage_counts),
        2
    )
)

print(
    "Average gold evidence passages:",
    round(
        sum(gold_evidence_counts)
        / len(gold_evidence_counts),
        2
    )
)

print(
    "Maximum candidate passages:",
    max(candidate_passage_counts)
)

print(
    "Maximum gold evidence passages:",
    max(gold_evidence_counts)
)


print("\n LABEL DISTRIBUTION ")

for label, count in label_counts.items():
    print(f"{label}: {count}")


print("\n EVIDENCE STRUCTURE ")

print(
    "Examples with multiple evidence groups:",
    multiple_evidence_examples
)

print(
    "Examples with mixed evidence labels:",
    mixed_label_examples
)




invalid_gold_passage_ids = 0

for example in examples:

    candidate_ids = {
        passage["passage_id"]
        for passage in example["candidate_passages"]
    }

    gold_ids = set(
        example["gold_evidence_passage_ids"]
    )

    if not gold_ids.issubset(candidate_ids):
        invalid_gold_passage_ids += 1


print("\n INTEGRITY CHECKS ")

print(
    "Examples with invalid evidence passage IDs:",
    invalid_gold_passage_ids
)



largest_example = max(
    examples,
    key=lambda example: len(example["candidate_passages"])
)

print("\n LARGEST EXAMPLE ")

print("Claim ID:", largest_example["claim_id"])

print("Claim:", largest_example["claim"])

print(
    "Candidate passages:",
    len(largest_example["candidate_passages"])
)

print(
    "Gold evidence IDs:",
    largest_example["gold_evidence_passage_ids"]
)
