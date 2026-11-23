import argparse
import hashlib
import json
import os
import random
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

try:
    from .experiment_utils import (
        connected_groups,
        paired_cluster_bootstrap_difference,
        paired_cluster_permutation,
        summarize_metrics,
    )
except ImportError:
    from experiment_utils import (
        connected_groups,
        paired_cluster_bootstrap_difference,
        paired_cluster_permutation,
        summarize_metrics,
    )


PROMPT_VERSION = "citation-verifier-json-v3"
PARSER_VERSION = "exact-json-enum-v2"
VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["SUPPORTS", "REFUTES", "NOT_ENOUGH_INFORMATION"],
        }
    },
    "required": ["verdict"],
    "additionalProperties": False,
}
OLLAMA_OPTIONS = {"temperature": 0.0, "num_predict": 64}
SEGMENTATIONS = ("90-250-word", "single-sentence", "3-sentence", "section-oriented")


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def parse_verifier_json(raw_answer: str) -> Tuple[str, str]:
    try:
        def reject_duplicate_keys(pairs: List[Tuple[str, object]]) -> Dict:
            parsed = {}
            for key, value in pairs:
                if key in parsed:
                    raise ValueError(f"duplicate JSON key: {key}")
                parsed[key] = value
            return parsed

        decoded = json.loads(raw_answer, object_pairs_hook=reject_duplicate_keys)
    except (json.JSONDecodeError, ValueError):
        return "PARSE_FAILURE", "parse_failure"
    if not isinstance(decoded, dict) or set(decoded) != {"verdict"}:
        return "PARSE_FAILURE", "parse_failure"
    if decoded["verdict"] not in VERDICT_SCHEMA["properties"]["verdict"]["enum"]:
        return "PARSE_FAILURE", "parse_failure"
    return decoded["verdict"], "ok"


def rank_bm25(claim: str, paragraphs: List[Dict]) -> Tuple[List[int], float]:
    t0 = time.perf_counter()
    corpus = [tokenize(p["text"]) for p in paragraphs]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(claim))
    order = list(np.argsort(-scores))
    elapsed = time.perf_counter() - t0
    return [int(i) for i in order], elapsed


def rank_dense(claim: str, paragraphs: List[Dict], embedder: SentenceTransformer) -> Tuple[List[int], float]:
    t0 = time.perf_counter()
    texts = [p["text"] for p in paragraphs]
    c_emb = embedder.encode([claim], normalize_embeddings=True)[0]
    p_embs = embedder.encode(texts, normalize_embeddings=True)
    scores = np.matmul(p_embs, c_emb)
    order = list(np.argsort(-scores))
    elapsed = time.perf_counter() - t0
    return [int(i) for i in order], elapsed


class CachedLLMVerifier:
    def __init__(
        self,
        ollama_url: str,
        model: str,
        cache_path: Path,
        request_timeout: float,
        fresh_run: bool = False,
    ):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.cache_path = cache_path
        self.request_timeout = request_timeout
        self.fresh_run = fresh_run
        self.run_id = str(uuid.uuid4()) if fresh_run else "reusable-cache"
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(cache_path), timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA busy_timeout=30000")
        self.db.execute(
            """CREATE TABLE IF NOT EXISTS verdict_cache (
                cache_key TEXT PRIMARY KEY,
                raw_response TEXT NOT NULL,
                verdict TEXT NOT NULL,
                parse_status TEXT NOT NULL,
                api_latency_sec REAL NOT NULL,
                config_json TEXT NOT NULL
            )"""
        )
        columns = {
            row[1] for row in self.db.execute("PRAGMA table_info(verdict_cache)").fetchall()
        }
        if "parse_status" not in columns:
            self.db.execute(
                "ALTER TABLE verdict_cache ADD COLUMN parse_status TEXT NOT NULL DEFAULT 'ok'"
            )
        self.db.commit()
        version_response = requests.get(f"{self.ollama_url}/api/version", timeout=5)
        version_response.raise_for_status()
        self.ollama_version = version_response.json().get("version", "unknown")
        tags_response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
        tags_response.raise_for_status()
        models = tags_response.json().get("models", [])
        model_record = next(
            (item for item in models if item.get("name") == model or item.get("model") == model),
            None,
        )
        if model_record is None:
            raise ValueError(f"Ollama model {model!r} is not installed at {self.ollama_url}")
        self.model_digest = model_record.get("digest")
        if not self.model_digest:
            raise ValueError(f"Ollama did not provide a digest for model {model!r}")
        self.config = {
            "ollama_url": self.ollama_url,
            "model": model,
            "model_digest": self.model_digest,
            "ollama_version": self.ollama_version,
            "prompt_version": PROMPT_VERSION,
            "parser_version": PARSER_VERSION,
            "system_prompt": "You are a scientific citation fact-checker. Judge only the supplied evidence.",
            "options": OLLAMA_OPTIONS,
            "schema": VERDICT_SCHEMA,
        }
        self.config_json = json.dumps(self.config, sort_keys=True)
        self.cache_hits = 0
        self.cache_misses = 0

    def _key(self, claim: str, passage: str) -> str:
        raw = json.dumps(
            {
                "config": self.config,
                "run_id": self.run_id,
                "claim": claim.strip(),
                "passage": passage.strip(),
            },
            sort_keys=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def verify(
        self, claim: str, passage: str, force_refresh: bool = False
    ) -> Tuple[str, float, bool, str, str]:
        k = self._key(claim, passage)
        if not self.fresh_run and not force_refresh:
            row = self.db.execute(
                "SELECT raw_response, verdict, parse_status FROM verdict_cache WHERE cache_key = ?",
                (k,),
            ).fetchone()
            if row is not None:
                self.cache_hits += 1
                return str(row[1]), 0.0, True, str(row[0]), str(row[2])

        prompt = (
            "You are a scientific citation fact-checker. Given a Claim Context from a citing paper "
            "and a Candidate Paragraph from a source paper, decide whether the Candidate Paragraph "
            "directly supports the factual claim, contradicts it, or does not provide enough information. "
            "Return a JSON object with exactly one key, verdict. Use only one of the allowed labels.\n\n"
            f"Claim Context:\n{claim}\n\n"
            f"Candidate Paragraph:\n{passage}\n\n"
        )

        t0 = time.perf_counter()
        response = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self.config["system_prompt"]},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
                "format": VERDICT_SCHEMA,
                "options": OLLAMA_OPTIONS,
            },
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        response_data = response.json()
        latency = time.perf_counter() - t0
        raw_out = response_data["message"]["content"].strip()
        verdict, parse_status = parse_verifier_json(raw_out)
        self.db.execute(
            """INSERT INTO verdict_cache
                (cache_key, raw_response, verdict, parse_status, api_latency_sec, config_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    raw_response=excluded.raw_response,
                    verdict=excluded.verdict,
                    parse_status=excluded.parse_status,
                    api_latency_sec=excluded.api_latency_sec,
                    config_json=excluded.config_json""",
            (k, raw_out, verdict, parse_status, latency, self.config_json),
        )
        self.db.commit()
        self.cache_misses += 1
        return verdict, latency, False, raw_out, parse_status


def evaluate_strategy(
    name: str,
    instances: List[Dict],
    orders: List[List[int]],
    ranking_times: List[float],
    verifier: CachedLLMVerifier,
    max_checks: int,
    group_for_instance: Dict[str, str],
    strategy_started: float,
) -> Dict:
    verification_started = time.perf_counter()
    pos_total = 0

    total_ranking_sec = sum(ranking_times)
    total_llm_sec = 0.0
    first_support_ranks: List[int] = []
    per_instance_logs: List[Dict] = []
    parse_failure_count = 0

    for inst, order, r_time in tqdm(zip(instances, orders, ranking_times), total=len(instances), desc=name):
        claim = inst["claim_context"]
        paragraphs = inst["paragraphs"]
        gt = inst["ground_truth"]  # SUPPORTS (label=1) or NOT_ENOUGH_INFORMATION (label=0)

        checked_order = order[:max_checks] if max_checks > 0 else order
        iterations = 0
        llm_time = 0.0
        cache_hits = 0
        cache_misses = 0
        passage_outputs = []
        predicted = "NOT_ENOUGH_INFORMATION"
        found_rank = None
        first_verdict = None

        for rank_pos, p_idx in enumerate(checked_order, start=1):
            iterations += 1
            verdict, lat, cache_hit, raw_answer, parse_status = verifier.verify(
                claim, paragraphs[p_idx]["text"]
            )
            cache_hits += int(cache_hit)
            cache_misses += int(not cache_hit)
            parse_failure_count += int(parse_status != "ok")
            llm_time += lat
            passage_outputs.append({
                "rank": rank_pos,
                "passage_id": paragraphs[p_idx].get("passage_id"),
                "verdict": verdict,
                "raw_answer": raw_answer,
                "parse_status": parse_status,
                "cache_hit": cache_hit,
                "model_api_sec": round(lat, 6),
            })
            if first_verdict is None:
                first_verdict = verdict
            if verdict == "SUPPORTS":
                predicted = "SUPPORTS"
                found_rank = rank_pos
                break  # Early stop once supporting evidence is verified

        is_correct = predicted == gt
        if inst["label"] == 1:
            pos_total += 1
            if found_rank is not None:
                first_support_ranks.append(found_rank)

        total_llm_sec += llm_time

        per_instance_logs.append({
            "instance_id": inst["instance_id"],
            "label": inst["label"],
            "ground_truth": gt,
            "predicted": predicted,
            "correct": bool(is_correct),
            "num_paragraphs": len(paragraphs),
            "iterations_used": iterations,
            "found_at_rank": found_rank,
            "first_verdict": first_verdict,
            "passage_verifier_outputs": passage_outputs,
            "ranking_sec": round(r_time, 5),
            "model_api_sec": round(llm_time, 5),
            "llm_sec": round(llm_time, 5),
            "total_sec": round(r_time + llm_time, 5),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "silver_label_error_category": (
                "positive_missed_within_budget"
                if gt == "SUPPORTS" and predicted != "SUPPORTS"
                else "support_verdict_on_negative_distractor"
                if gt != "SUPPORTS" and predicted == "SUPPORTS"
                else "correct_positive_decision"
                if gt == "SUPPORTS"
                else "correct_negative_decision"
            ),
        })

    confirmed_at_1 = sum(1 for r in first_support_ranks if r <= 1) / max(1, pos_total)
    confirmed_at_3 = sum(1 for r in first_support_ranks if r <= 3) / max(1, pos_total)
    confirmed_at_5 = sum(1 for r in first_support_ranks if r <= 5) / max(1, pos_total)
    mrr = sum(1.0 / r for r in first_support_ranks) / max(1, pos_total)

    summary = summarize_metrics(
        name,
        per_instance_logs,
        group_for_instance,
    )
    summary.update({
        "parse_failure_count": parse_failure_count,
        "verifier_confirmed_positive_rate_at_rank_1_not_gold_recall": round(confirmed_at_1, 4),
        "verifier_confirmed_positive_rate_at_rank_3_not_gold_recall": round(confirmed_at_3, 4),
        "verifier_confirmed_positive_rate_at_rank_5_not_gold_recall": round(confirmed_at_5, 4),
        "verifier_confirmed_reciprocal_rank_not_gold_mrr": round(mrr, 4),
        "positive_source_accuracy": summary["support_recall"],
        "negative_distractor_accuracy": summary["negative_accuracy"],
        "total_llm_iterations": summary["total_checks"],
        "avg_iterations_per_instance": summary["mean_checks"],
        "total_ranking_sec": round(total_ranking_sec, 3),
        "total_llm_sec": round(total_llm_sec, 3),
        "sum_ranking_and_model_api_sec": round(total_ranking_sec + total_llm_sec, 3),
        "measured_verification_wall_clock_sec": round(time.perf_counter() - verification_started, 3),
        "measured_strategy_wall_clock_sec": round(time.perf_counter() - strategy_started, 3),
        "verifier_config": verifier.config,
    })
    return {"summary": summary, "logs": per_instance_logs}


def format_summary_text(comparison: List[Dict], paired_tests: Dict | None = None) -> str:
    sections = [
        "================ EXPERIMENT SUMMARY ================",
        "Final decision task: binary SUPPORTS vs NOT_ENOUGH_INFORMATION.",
        "REFUTES and NOT_ENOUGH_INFORMATION are both treated as negative; no refuted gold class is present.",
        *(json.dumps(row, indent=2) for row in comparison),
    ]
    if paired_tests:
        sections.extend(("Paired cluster-level permutation tests:", json.dumps(paired_tests, indent=2)))
    sections.append("")
    return "\n".join(
        sections
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate passage ranking + Ollama verification savings.")
    parser.add_argument("--dataset", default="data/thesis_dataset.jsonl")
    parser.add_argument("--start-index", type=int, default=0, help="First dataset row to evaluate")
    parser.add_argument("--limit", type=int, default=0, help="0 = all remaining rows; >0 = rows to evaluate")
    parser.add_argument("--max-checks", type=int, default=0, help="Max passages checked per source (0 = all)")
    parser.add_argument("--split-manifest", help="Optional output from scripts/prepare_splits.py")
    parser.add_argument("--split", choices=("train", "development", "test"))
    parser.add_argument("--fresh-run", action="store_true", help="Bypass cache reads and measure fresh API calls")
    parser.add_argument("--strategy-seed", type=int, default=231104, help="Seed for execution-order randomization")
    parser.add_argument("--segmentation", choices=SEGMENTATIONS, default="90-250-word")
    parser.add_argument("--model", default="qwen2.5:3b", help="Ollama model name")
    parser.add_argument("--request-timeout", type=float, default=120.0, help="Ollama request timeout in seconds")
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    if args.start_index < 0:
        parser.error("--start-index must be 0 or greater")
    if args.limit < 0:
        parser.error("--limit must be 0 or greater")
    if args.max_checks < 0:
        parser.error("--max-checks must be 0 or greater")
    if bool(args.split_manifest) != bool(args.split):
        parser.error("--split-manifest and --split must be provided together")
    if args.split and (args.start_index or args.limit):
        parser.error("--start-index/--limit cannot be combined with a grouped split")

    ollama_url = args.ollama_url
    if "://" not in ollama_url:
        ollama_url = f"http://{ollama_url}"

    raw_lines = Path(args.dataset).read_text(encoding="utf-8").splitlines()
    all_instances = [json.loads(line) for line in raw_lines if line.strip()]
    if args.split:
        manifest_path = Path(args.split_manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        dataset_digest = hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest()
        if manifest.get("dataset_sha256") != dataset_digest:
            parser.error("Split manifest does not match the current dataset bytes")
        selected_ids = set(manifest["splits"][args.split])
        known_ids = {row["instance_id"] for row in all_instances}
        if selected_ids - known_ids:
            parser.error("Split manifest references instance IDs missing from the dataset")
        instances = [row for row in all_instances if row["instance_id"] in selected_ids]
        group_for_instance = {
            instance_id: manifest["group_for_instance"][instance_id]
            for instance_id in selected_ids
        }
        split_name = args.split
    else:
        if args.start_index >= len(all_instances):
            parser.error(f"--start-index must be less than dataset size ({len(all_instances)})")
        end_index = len(all_instances) if args.limit == 0 else min(args.start_index + args.limit, len(all_instances))
        instances = all_instances[args.start_index:end_index]
        all_group_ids, _ = connected_groups(all_instances)
        group_for_instance = {
            row["instance_id"]: all_group_ids[index]
            for index, row in enumerate(all_instances)
        }
        split_name = "all"
    if not instances:
        parser.error(f"Selected {split_name} split is empty")
    if args.segmentation != "90-250-word":
        segmented_instances = []
        for instance in instances:
            passages = instance.get("passage_sets", {}).get(args.segmentation)
            if not passages:
                parser.error(
                    f"{args.segmentation} passages are absent; rebuild with "
                    "--include-segmentation-ablations"
                )
            updated = dict(instance)
            updated["paragraphs"] = passages
            updated["num_paragraphs"] = len(passages)
            segmented_instances.append(updated)
        instances = segmented_instances
    print(f"Loaded {len(instances)} instances from {args.dataset} ({split_name} split)")

    if args.request_timeout <= 0:
        parser.error("--request-timeout must be greater than 0")

    budget_name = "all" if args.max_checks == 0 else str(args.max_checks)
    cache_dir = Path("logs/verifier_cache") / args.model.replace(":", "_") / split_name / f"k{budget_name}"
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    strategy_names = ["Unranked_Baseline", "BM25_Ranked", "MiniLM_Dense_Ranked"]
    random.Random(args.strategy_seed).shuffle(strategy_names)
    results_by_name = {}
    for name in strategy_names:
        strategy_started = time.perf_counter()
        if name == "Unranked_Baseline":
            orders = [list(range(len(inst["paragraphs"]))) for inst in instances]
            ranking_times = [0.0 for _ in instances]
        elif name == "BM25_Ranked":
            orders, ranking_times = [], []
            for inst in tqdm(instances, desc="Computing BM25 rankings"):
                order, elapsed = rank_bm25(inst["claim_context"], inst["paragraphs"])
                orders.append(order)
                ranking_times.append(elapsed)
        else:
            orders, ranking_times = [], []
            for inst in tqdm(instances, desc="Computing dense rankings"):
                order, elapsed = rank_dense(inst["claim_context"], inst["paragraphs"], embedder)
                orders.append(order)
                ranking_times.append(elapsed)
        verifier = CachedLLMVerifier(
            ollama_url,
            args.model,
            cache_dir / f"{name}.sqlite3",
            args.request_timeout,
            fresh_run=args.fresh_run,
        )
        results_by_name[name] = evaluate_strategy(
                name,
                instances,
                orders,
                ranking_times,
                verifier,
                args.max_checks,
                group_for_instance,
                strategy_started,
            )
        verifier.db.close()
    res_base = results_by_name["Unranked_Baseline"]
    res_bm25 = results_by_name["BM25_Ranked"]
    res_dense = results_by_name["MiniLM_Dense_Ranked"]
    ordered_results = (res_base, res_bm25, res_dense)

    base_s = res_base["summary"]
    comparison = []
    for res in ordered_results:
        s = dict(res["summary"])
        iter_saved = base_s["total_llm_iterations"] - s["total_llm_iterations"]
        time_saved = base_s["measured_strategy_wall_clock_sec"] - s["measured_strategy_wall_clock_sec"]
        s["iterations_saved_vs_baseline"] = iter_saved
        s["iterations_saved_pct"] = round(100.0 * iter_saved / max(1, base_s["total_llm_iterations"]), 2)
        s["net_time_saved_sec_vs_baseline"] = round(time_saved, 3)
        s["net_time_saved_pct"] = round(
            100.0 * time_saved / max(0.001, base_s["measured_strategy_wall_clock_sec"]), 2
        )
        comparison.append(s)

    paired_tests = {
        f"{first['summary']['strategy']}__vs__{second['summary']['strategy']}": {
            **paired_cluster_permutation(
                first["logs"], second["logs"], group_for_instance
            ),
            **paired_cluster_bootstrap_difference(
                first["logs"], second["logs"], group_for_instance, seed=args.strategy_seed
            ),
        }
        for first, second in ((res_base, res_bm25), (res_base, res_dense), (res_bm25, res_dense))
    }
    out_dir = Path(args.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary_metrics.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (out_dir / "detailed_logs.json").write_text(
        json.dumps(
            {
                "Unranked_Baseline": res_base["logs"],
                "BM25_Ranked": res_bm25["logs"],
                "MiniLM_Dense_Ranked": res_dense["logs"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "paired_strategy_tests.json").write_text(json.dumps(paired_tests, indent=2), encoding="utf-8")
    (out_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "dataset": str(args.dataset),
                "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
                "instances": len(instances),
                "split": split_name,
                "evaluation_status": (
                    "held_out_test" if split_name == "test" else "development" if split_name == "development"
                    else "training" if split_name == "train" else "exploratory_unsplit"
                ),
                "label_source": "automatic citation/distractor construction; weak silver labels",
                "split_manifest": args.split_manifest,
                "max_checks": args.max_checks,
                "passage_segmentation": args.segmentation,
                "prediction_task": "binary SUPPORTS versus NOT_ENOUGH_INFORMATION",
                "verifier_label_handling": (
                    "REFUTES and NOT_ENOUGH_INFORMATION are both negative for source-level prediction"
                ),
                "fresh_run": args.fresh_run,
                "seed": 231104,
                "strategy_execution_seed": args.strategy_seed,
                "strategy_execution_order": strategy_names,
                "verifier_config": res_base["summary"]["verifier_config"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    summary_text = format_summary_text(comparison, paired_tests)
    (out_dir / "summary_metrics.txt").write_text(summary_text, encoding="utf-8")
    print(f"\n{summary_text}", end="")


if __name__ == "__main__":
    main()
