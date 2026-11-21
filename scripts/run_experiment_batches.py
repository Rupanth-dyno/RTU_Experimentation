import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict

import requests


STRATEGIES = ("Unranked_Baseline", "BM25_Ranked", "MiniLM_Dense_Ranked")


def ensure_ollama_ready(ollama_url: str, model: str) -> None:
    try:
        response = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=5)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(
            f"Ollama is not responding at {ollama_url}: {exc}. "
            "Start or restart Ollama, then run this batch command again."
        ) from exc

    available_models = {item["name"] for item in response.json().get("models", [])}
    if model not in available_models:
        raise SystemExit(
            f"Ollama model {model!r} is not available at {ollama_url}. "
            f"Pull it first with: ollama pull {model}"
        )


def is_complete(batch_dir: Path, expected_count: int) -> bool:
    detailed_path = batch_dir / "detailed_logs.json"
    summary_path = batch_dir / "summary_metrics.json"
    text_path = batch_dir / "summary_metrics.txt"
    if not detailed_path.exists() or not summary_path.exists() or not text_path.exists():
        return False

    try:
        details: Dict[str, list] = json.loads(detailed_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return all(len(details.get(strategy, [])) == expected_count for strategy in STRATEGIES)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full experiment in restartable, sequential dataset batches."
    )
    parser.add_argument("--dataset", default="data/thesis_dataset.jsonl")
    parser.add_argument("--start-index", type=int, default=0, help="Dataset row to resume from")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--max-checks", type=int, default=0, help="0 checks all paragraphs")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--results-dir", default="results/batches")
    args = parser.parse_args()
    if args.start_index < 0:
        parser.error("--start-index must be 0 or greater")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than 0")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be greater than 0")

    ollama_url = args.ollama_url
    if "://" not in ollama_url:
        ollama_url = f"http://{ollama_url}"
    ensure_ollama_ready(ollama_url, args.model)

    dataset_path = Path(args.dataset).resolve()
    rows = [line for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.start_index >= len(rows):
        parser.error(f"--start-index must be less than dataset size ({len(rows)})")

    root = Path(__file__).resolve().parent.parent
    experiment_script = Path(__file__).resolve().with_name("run_experiments.py")
    results_root = Path(args.results_dir)
    total = len(rows)

    for start in range(args.start_index, total, args.batch_size):
        count = min(args.batch_size, total - start)
        end = start + count
        batch_dir = results_root / f"batch_{start:04d}_{end:04d}"
        if is_complete(batch_dir, count):
            print(f"Skipping completed rows {start}-{end - 1}: {batch_dir}")
            continue

        print(f"\nRunning rows {start}-{end - 1} -> {batch_dir}", flush=True)
        command = [
            sys.executable,
            str(experiment_script),
            "--dataset",
            str(dataset_path),
            "--start-index",
            str(start),
            "--limit",
            str(count),
            "--max-checks",
            str(args.max_checks),
            "--model",
            args.model,
            "--ollama-url",
            ollama_url,
            "--request-timeout",
            str(args.request_timeout),
            "--results-dir",
            str(batch_dir),
        ]
        result = subprocess.run(command, cwd=root, check=False)
        if result.returncode:
            print(
                f"\nBatch rows {start}-{end - 1} failed with exit code {result.returncode}. "
                f"Restart this command with --start-index {start}; completed batches will be skipped.",
                file=sys.stderr,
                flush=True,
            )
            return result.returncode

    print(f"\nAll dataset rows have batch outputs in {results_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
