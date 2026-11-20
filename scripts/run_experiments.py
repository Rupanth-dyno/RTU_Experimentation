import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import requests
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


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
    """
    Verifies whether a single paragraph passage supports a claim.
    Caches verdict + latency by (model, claim, passage) hash so re-runs and cross-method
    comparisons are 100% deterministic, fast, and crash-safe.
    """

    def __init__(self, ollama_url: str, model: str, cache_path: Path):
        self.ollama_url = ollama_url.rstrip("/")
        self.model = model
        self.cache_path = cache_path
        self.cache: Dict[str, Dict] = {}
        if cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self.cache[rec["key"]] = rec

    def _key(self, claim: str, passage: str) -> str:
        raw = f"{self.model}||{claim.strip()}||{passage.strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def verify(self, claim: str, passage: str) -> Tuple[str, float]:
        k = self._key(claim, passage)
        if k in self.cache:
            rec = self.cache[k]
            return rec["verdict"], float(rec["latency_sec"])

        prompt = (
            "You are a scientific citation fact-checker. Given a Claim Context from a citing paper "
            "and a Candidate Paragraph from a source paper, decide whether the Candidate Paragraph "
            "directly provides evidence supporting the claim.\n\n"
            "Reply with ONLY one label: SUPPORTS, REFUTES, or NOT_ENOUGH_INFORMATION.\n\n"
            f"Claim Context:\n{claim}\n\n"
            f"Candidate Paragraph:\n{passage}\n\n"
            "Verdict:"
        )

        t0 = time.perf_counter()
        response = requests.post(
            f"{self.ollama_url}/api/chat",
            json={
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 10},
            },
            timeout=120,
        )
        response.raise_for_status()
        response_data = response.json()
        latency = time.perf_counter() - t0
        raw_out = response_data["message"]["content"].strip().upper()

        if "SUPPORTS" in raw_out:
            verdict = "SUPPORTS"
        elif "REFUTES" in raw_out:
            verdict = "REFUTES"
        else:
            verdict = "NOT_ENOUGH_INFORMATION"

        rec = {"key": k, "verdict": verdict, "latency_sec": latency}
        self.cache[k] = rec
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self.cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        return verdict, latency


def evaluate_strategy(
    name: str,
    instances: List[Dict],
    orders: List[List[int]],
    ranking_times: List[float],
    verifier: CachedLLMVerifier,
    max_checks: int,
) -> Dict:
    correct = 0
    pos_correct = 0
    neg_correct = 0
    pos_total = 0
    neg_total = 0

    total_iterations = 0
    total_ranking_sec = sum(ranking_times)
    total_llm_sec = 0.0
    first_support_ranks: List[int] = []
    per_instance_logs: List[Dict] = []

    for inst, order, r_time in tqdm(zip(instances, orders, ranking_times), total=len(instances), desc=name):
        claim = inst["claim_context"]
        paragraphs = inst["paragraphs"]
        gt = inst["ground_truth"]  # SUPPORTS (label=1) or NOT_ENOUGH_INFORMATION (label=0)

        checked_order = order[:max_checks] if max_checks > 0 else order
        iterations = 0
        llm_time = 0.0
        predicted = "NOT_ENOUGH_INFORMATION"
        found_rank = None

        for rank_pos, p_idx in enumerate(checked_order, start=1):
            iterations += 1
            verdict, lat = verifier.verify(claim, paragraphs[p_idx]["text"])
            llm_time += lat
            if verdict == "SUPPORTS":
                predicted = "SUPPORTS"
                found_rank = rank_pos
                break  # Early stop once supporting evidence is verified

        is_correct = int(predicted == gt)
        correct += is_correct
        if inst["label"] == 1:
            pos_total += 1
            pos_correct += is_correct
            if found_rank is not None:
                first_support_ranks.append(found_rank)
        else:
            neg_total += 1
            neg_correct += is_correct

        total_iterations += iterations
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
            "ranking_sec": round(r_time, 5),
            "llm_sec": round(llm_time, 5),
            "total_sec": round(r_time + llm_time, 5),
        })

    n = max(1, len(instances))
    recall_at_1 = sum(1 for r in first_support_ranks if r <= 1) / max(1, pos_total)
    recall_at_3 = sum(1 for r in first_support_ranks if r <= 3) / max(1, pos_total)
    recall_at_5 = sum(1 for r in first_support_ranks if r <= 5) / max(1, pos_total)
    mrr = sum(1.0 / r for r in first_support_ranks) / max(1, pos_total)

    summary = {
        "strategy": name,
        "instances_evaluated": len(instances),
        "overall_accuracy": round(correct / n, 4),
        "positive_source_accuracy": round(pos_correct / max(1, pos_total), 4),
        "negative_distractor_accuracy": round(neg_correct / max(1, neg_total), 4),
        "recall_at_1": round(recall_at_1, 4),
        "recall_at_3": round(recall_at_3, 4),
        "recall_at_5": round(recall_at_5, 4),
        "mrr": round(mrr, 4),
        "total_llm_iterations": total_iterations,
        "avg_iterations_per_instance": round(total_iterations / n, 2),
        "total_ranking_sec": round(total_ranking_sec, 3),
        "total_llm_sec": round(total_llm_sec, 3),
        "total_wall_clock_sec": round(total_ranking_sec + total_llm_sec, 3),
    }
    return {"summary": summary, "logs": per_instance_logs}


def format_summary_text(comparison: List[Dict]) -> str:
    return "\n".join(
        [
            "================ FINAL THESIS EXPERIMENT SUMMARY ================",
            *(json.dumps(row, indent=2) for row in comparison),
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate passage ranking + Ollama verification savings.")
    parser.add_argument("--dataset", default="data/thesis_dataset.jsonl")
    parser.add_argument("--limit", type=int, default=0, help="0 = full dataset; >0 = pilot subset size")
    parser.add_argument("--max-checks", type=int, default=0, help="Max passages checked per source (0 = all)")
    parser.add_argument("--model", default="qwen2.5:3b", help="Ollama model name")
    parser.add_argument(
        "--ollama-url",
        default=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama server URL (default: http://localhost:11434)",
    )
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    ollama_url = args.ollama_url
    if "://" not in ollama_url:
        ollama_url = f"http://{ollama_url}"

    raw_lines = Path(args.dataset).read_text(encoding="utf-8").splitlines()
    instances = [json.loads(line) for line in raw_lines if line.strip()]
    if args.limit > 0:
        instances = instances[: args.limit]

    print(f"Loaded {len(instances)} instances from {args.dataset}")

    verifier = CachedLLMVerifier(ollama_url, args.model, Path("logs/llm_verification_cache.jsonl"))
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # 1. Unranked Baseline (document order: 0, 1, ..., N-1)
    baseline_orders = [list(range(len(inst["paragraphs"]))) for inst in instances]
    baseline_r_times = [0.0 for _ in instances]

    # 2. BM25 Ranking
    bm25_orders, bm25_r_times = [], []
    for inst in tqdm(instances, desc="Precomputing BM25 rankings"):
        order, rt = rank_bm25(inst["claim_context"], inst["paragraphs"])
        bm25_orders.append(order)
        bm25_r_times.append(rt)

    # 3. Dense Semantic Ranking (all-MiniLM-L6-v2)
    dense_orders, dense_r_times = [], []
    for inst in tqdm(instances, desc="Precomputing Dense rankings"):
        order, rt = rank_dense(inst["claim_context"], inst["paragraphs"], embedder)
        dense_orders.append(order)
        dense_r_times.append(rt)

    res_base = evaluate_strategy("Unranked_Baseline", instances, baseline_orders, baseline_r_times, verifier, args.max_checks)
    res_bm25 = evaluate_strategy("BM25_Ranked", instances, bm25_orders, bm25_r_times, verifier, args.max_checks)
    res_dense = evaluate_strategy("MiniLM_Dense_Ranked", instances, dense_orders, dense_r_times, verifier, args.max_checks)

    base_s = res_base["summary"]
    comparison = []
    for res in (res_base, res_bm25, res_dense):
        s = dict(res["summary"])
        iter_saved = base_s["total_llm_iterations"] - s["total_llm_iterations"]
        time_saved = base_s["total_wall_clock_sec"] - s["total_wall_clock_sec"]
        s["iterations_saved_vs_baseline"] = iter_saved
        s["iterations_saved_pct"] = round(100.0 * iter_saved / max(1, base_s["total_llm_iterations"]), 2)
        s["net_time_saved_sec_vs_baseline"] = round(time_saved, 3)
        s["net_time_saved_pct"] = round(100.0 * time_saved / max(0.001, base_s["total_wall_clock_sec"]), 2)
        comparison.append(s)

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

    summary_text = format_summary_text(comparison)
    (out_dir / "summary_metrics.txt").write_text(summary_text, encoding="utf-8")
    print(f"\n{summary_text}", end="")


if __name__ == "__main__":
    main()
