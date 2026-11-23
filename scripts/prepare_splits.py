import argparse
import hashlib
import json
from pathlib import Path

from experiment_utils import make_grouped_split


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create deterministic, leakage-resistant train/development/test splits."
    )
    parser.add_argument("--dataset", default="data/thesis_dataset.jsonl")
    parser.add_argument("--seed", type=int, default=231104)
    parser.add_argument("--output", default="results/splits.json")
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    instances = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not instances:
        parser.error("Dataset is empty")
    instance_ids = [row["instance_id"] for row in instances]
    if len(instance_ids) != len(set(instance_ids)):
        parser.error("Dataset contains duplicate instance_id values")

    split = make_grouped_split(instances, args.seed)
    split["dataset"] = str(dataset_path)
    split["dataset_sha256"] = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    split["grouping_keys"] = ["pair_id", "citing_arxiv_id", "source_arxiv_id"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(split, indent=2), encoding="utf-8")
    print(f"Saved grouped split manifest to {output}")
    print(json.dumps(split["actual_instance_counts"], indent=2))


if __name__ == "__main__":
    main()
