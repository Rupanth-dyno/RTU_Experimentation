import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List

from run_experiments import CachedLLMVerifier
from experiment_utils import wilson_interval


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure repeated-call agreement on a deterministic sample of test pairs."
    )
    parser.add_argument("--dataset", default="data/thesis_dataset.jsonl")
    parser.add_argument("--split-manifest", default="results/splits.json")
    parser.add_argument("--split", default="test", choices=("train", "development", "test"))
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=231104)
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--output", default="results/verifier_agreement.json")
    args = parser.parse_args()
    if args.sample_size <= 0:
        parser.error("--sample-size must be positive")

    rows = [
        json.loads(line)
        for line in Path(args.dataset).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    dataset_digest = hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()
    if manifest.get("dataset_sha256") != dataset_digest:
        parser.error("Split manifest does not match the current dataset bytes")
    selected_ids = set(manifest["splits"][args.split])
    candidates = [row for row in rows if row["instance_id"] in selected_ids]
    rng = random.Random(args.seed)
    sample = rng.sample(candidates, min(args.sample_size, len(candidates)))
    verifier = CachedLLMVerifier(
        args.ollama_url,
        args.model,
        Path(args.output).with_suffix(".sqlite3"),
        args.request_timeout,
        fresh_run=True,
    )

    records: List[Dict] = []
    for row in sample:
        selected_passage = rng.choice(row["paragraphs"])
        passage = selected_passage["text"]
        first, _ = verifier.verify(row["claim_context"], passage, force_refresh=True)
        second, _ = verifier.verify(row["claim_context"], passage, force_refresh=True)
        records.append({
            "instance_id": row["instance_id"],
            "passage_id": selected_passage["passage_id"],
            "label": row["label"],
            "first_verdict": first,
            "second_verdict": second,
            "agreement": first == second,
        })

    labels = ("SUPPORTS", "REFUTES", "NOT_ENOUGH_INFORMATION")
    confusion = {first: {second: 0 for second in labels} for first in labels}
    for row in records:
        confusion[row["first_verdict"]][row["second_verdict"]] += 1
    agreements = sum(row["agreement"] for row in records)
    first_counts = {label: sum(row["first_verdict"] == label for row in records) for label in labels}
    second_counts = {label: sum(row["second_verdict"] == label for row in records) for label in labels}
    expected_agreement = sum(
        first_counts[label] * second_counts[label] for label in labels
    ) / max(1, len(records) ** 2)
    observed_agreement = agreements / max(1, len(records))
    kappa = (
        (observed_agreement - expected_agreement) / (1 - expected_agreement)
        if expected_agreement < 1
        else 1.0
    )
    report = {
        "sample_size": len(records),
        "seed": args.seed,
        "split": args.split,
        "verifier_config": verifier.config,
        "exact_agreement": observed_agreement,
        "exact_agreement_wilson_95_ci": wilson_interval(agreements, len(records)),
        "cohen_kappa": kappa,
        "confusion_matrix_first_by_second": confusion,
        "records": records,
        "note": "Repeated deterministic model calls measure repeatability, not correctness against gold labels.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
