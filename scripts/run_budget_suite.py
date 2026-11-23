import argparse
import json
import subprocess
import sys
from pathlib import Path

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run real fixed-budget evaluations with fresh verifier calls."
    )
    parser.add_argument("--dataset", default="data/thesis_dataset.jsonl")
    parser.add_argument("--split-manifest", default="results/splits.json")
    parser.add_argument("--split", default="test", choices=("train", "development", "test"))
    parser.add_argument("--budgets", default="1,3,5,10,20")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--results-dir", default="results/fixed_budget_runs")
    parser.add_argument("--agreement-sample-size", type=int, default=100)
    parser.add_argument("--strategy-seed", type=int, default=231104)
    args = parser.parse_args()

    try:
        budgets = [int(value) for value in args.budgets.split(",")]
    except ValueError:
        parser.error("--budgets must be comma-separated non-negative integers")
    if not budgets or any(budget <= 0 for budget in budgets) or len(set(budgets)) != len(budgets):
        parser.error("--budgets must contain unique positive integers")

    script = Path(__file__).with_name("run_experiments.py")
    for budget in budgets:
        output = Path(args.results_dir) / f"{args.split}_k{budget}"
        command = [
            sys.executable,
            str(script),
            "--dataset",
            args.dataset,
            "--split-manifest",
            args.split_manifest,
            "--split",
            args.split,
            "--max-checks",
            str(budget),
            "--model",
            args.model,
            "--ollama-url",
            args.ollama_url,
            "--request-timeout",
            str(args.request_timeout),
            "--fresh-run",
            "--strategy-seed",
            str(args.strategy_seed),
            "--results-dir",
            str(output),
        ]
        print(f"\nRunning actual K={budget} evaluation for {args.split} split", flush=True)
        result = subprocess.run(command, check=False)
        if result.returncode:
            print(f"Budget K={budget} failed with exit code {result.returncode}", file=sys.stderr)
            return result.returncode

    agreement_path = Path(args.results_dir) / "verifier_agreement.json"
    agreement_script = Path(__file__).with_name("evaluate_verifier_agreement.py")
    agreement_command = [
        sys.executable,
        str(agreement_script),
        "--dataset",
        args.dataset,
        "--split-manifest",
        args.split_manifest,
        "--split",
        args.split,
        "--sample-size",
        str(args.agreement_sample_size),
        "--model",
        args.model,
        "--ollama-url",
        args.ollama_url,
        "--request-timeout",
        str(args.request_timeout),
        "--output",
        str(agreement_path),
    ]
    agreement_result = subprocess.run(agreement_command, check=False)
    if agreement_result.returncode:
        print(
            f"Verifier agreement evaluation failed with exit code {agreement_result.returncode}",
            file=sys.stderr,
        )
        return agreement_result.returncode
    agreement = json.loads(agreement_path.read_text(encoding="utf-8"))
    for budget in budgets:
        summary_path = Path(args.results_dir) / f"{args.split}_k{budget}" / "summary_metrics.json"
        summaries = json.loads(summary_path.read_text(encoding="utf-8"))
        for summary in summaries:
            summary["verifier_exact_agreement"] = round(agreement["exact_agreement"], 4)
            summary["verifier_exact_agreement_wilson_95_ci"] = [
                round(value, 4) for value in agreement["exact_agreement_wilson_95_ci"]
            ]
            summary["verifier_cohen_kappa"] = round(agreement["cohen_kappa"], 4)
            summary["verifier_agreement"] = {
                "sample_size": agreement["sample_size"],
                "exact": round(agreement["exact_agreement"], 4),
                "wilson_95_ci": [round(value, 4) for value in agreement["exact_agreement_wilson_95_ci"]],
                "cohen_kappa": round(agreement["cohen_kappa"], 4),
                "interpretation": "repeatability only; not correctness against gold",
            }
        summary_path.write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        text_path = summary_path.with_suffix(".txt")
        paired_path = summary_path.parent / "paired_strategy_tests.json"
        paired = json.loads(paired_path.read_text(encoding="utf-8"))
        text_path.write_text(
            "FIXED-BUDGET RUN; actual model calls with fresh cache namespace.\n"
            f"Verifier exact agreement: {agreement['exact_agreement']:.4f} "
            f"(95% Wilson CI {agreement['exact_agreement_wilson_95_ci'][0]:.4f}-"
            f"{agreement['exact_agreement_wilson_95_ci'][1]:.4f}); "
            f"Cohen's kappa={agreement['cohen_kappa']:.4f}\n\n"
            + "\n".join(json.dumps(row, indent=2) for row in summaries)
            + "\n\nPaired cluster permutation tests:\n"
            + json.dumps(paired, indent=2)
            + "\n",
            encoding="utf-8",
        )
    print(f"\nCompleted actual fixed-budget runs in {args.results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
