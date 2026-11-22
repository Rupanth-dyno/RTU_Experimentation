# RTU Thesis Experimentation

This project builds a local dataset from cached research-paper PDFs and evaluates
whether passage-ranking methods help identify evidence for a citation claim.
It compares three strategies:

- **Unranked baseline:** checks paragraphs in their original document order.
- **BM25:** ranks paragraphs using lexical relevance to the claim.
- **MiniLM dense ranking:** ranks paragraphs using semantic embeddings from
  `all-MiniLM-L6-v2`.

The verifier checks ranked passages until it finds a `SUPPORTS` verdict or
reaches the configured passage limit. If no passage supports the claim, it
predicts `NOT_ENOUGH_INFORMATION`. This is a retrieval-and-verification
experiment; it does not train or fine-tune an LLM.

## Data pipeline

`scripts/build_dataset.py` works from locally cached PDFs in `data/pdfs/`.
PDF files are intentionally excluded from Git; their local filenames and
contents are left unchanged. Place the source PDFs in that directory before
building the dataset:

1. Extracts titles, abstracts, body text, references, and paragraphs locally
   using PyMuPDF.
2. Finds internal citations using arXiv IDs and normalized paper titles.
3. Extracts claim context around matched citations.
4. Pairs positive cited sources with semantically similar non-cited
   distractors using `all-MiniLM-L6-v2`.
5. Writes positive (`SUPPORTS`) and negative
   (`NOT_ENOUGH_INFORMATION`) instances as JSON Lines.

The committed dataset, `data/thesis_dataset.jsonl`, contains **800 instances**:
400 positive and 400 negative. The builder's default target is 600 positive
instances, but the final dataset contains only pairs successfully found and
processed from the available PDFs.

Build the dataset with:

```bash
.venv/bin/python scripts/build_dataset.py \
  --pdf-dir data/pdfs \
  --target-positives 600 \
  --output data/thesis_dataset.jsonl
```

The builder runs its PDF extraction and embedding model in offline mode. The
`all-MiniLM-L6-v2` model must already be available in the local Hugging Face
cache.

## LLM verifier

The experiment uses **Qwen 2.5 3B (`qwen2.5:3b`) served locally by Ollama**.
The recorded pilot used Ollama at `http://localhost:11434`; no OpenAI service,
API key, or cloud LLM is required. The experiment script calls Ollama's native
`/api/chat` endpoint. It asks for one of three labels:

- `SUPPORTS`
- `REFUTES`
- `NOT_ENOUGH_INFORMATION`

Only `SUPPORTS` changes the experiment's prediction to positive; both other
verdicts result in `NOT_ENOUGH_INFORMATION`. The script caches verdicts and
latencies in `logs/llm_verification_cache.jsonl`, keyed by model, claim, and
passage, so repeated checks can reuse prior responses.

Install the Python dependencies and download the Ollama model before running:

```bash
.venv/bin/python -m pip install -r requirements.txt
ollama pull qwen2.5:3b
ollama serve
```

If Ollama is listening on a different host, set `OLLAMA_HOST` or pass
`--ollama-url`. The server URL may include a scheme and port, for example
`http://localhost:11434`.

## Running experiments

Run the same 50-instance pilot, with at most five checked passages per source:

```bash
.venv/bin/python scripts/run_experiments.py \
  --dataset data/thesis_dataset.jsonl \
  --limit 50 \
  --max-checks 5 \
  --ollama-url http://localhost:11434 \
  --model qwen2.5:3b \
  --results-dir results_pilot
```

To evaluate the full dataset, omit `--limit` or set it to `0`. A
`--max-checks` value of `0` checks all paragraphs. Full evaluation can make
many local model requests and take substantially longer than the pilot.

For long full-dataset runs, use restartable 75-instance batches instead:

```bash
.venv/bin/python scripts/run_experiment_batches.py \
  --batch-size 75 \
  --max-checks 0 \
  --model qwen2.5:3b \
  --ollama-url http://localhost:11434
```

Each completed batch is saved under `results/batches/` and skipped on restart.
The 800-row dataset runs as 11 batches at this size.
If Ollama fails during a batch, restart the same command with
`--start-index` set to the beginning row of the failed batch; the verifier
cache avoids repeating successful passage checks. The batch runner checks that
Ollama is responding and that the requested model is installed before starting.
Each request has a 120-second timeout by default; adjust it with
`--request-timeout` if needed. After all batches complete, combine them into
the full-dataset reports with:

```bash
.venv/bin/python scripts/merge_experiment_batches.py
```

To compare retrieval at fixed passage limits using ranks and verdicts from the
completed full run (without making additional Ollama calls), run:

```bash
.venv/bin/python scripts/evaluate_cached_budgets.py \
  --dataset data/thesis_dataset.jsonl \
  --detailed-logs results/detailed_logs.json \
  --budgets 1,3,5,10,20,30,50,0 \
  --output-dir results/budget_eval
```

Here, budget `0` means all passages. The report includes precision and false
positive/negative counts. These are exploratory fixed-budget results derived
from the completed full run; they do not represent a new verifier or an
independent test set.

Each run writes:

- `summary_metrics.json`: aggregate metrics for the three strategies.
- `summary_metrics.txt`: the same summary in readable text form.
- `detailed_logs.json`: per-instance predictions, correctness, checks, and
  elapsed times.

The completed pilot outputs are in `results_pilot/`. The experiment also prints
the summary to the terminal.

## Recorded pilot results

The committed pilot evaluated the first **50 instances** from the dataset,
checking up to **five passages** per instance:

| Strategy | Accuracy | Positive-source accuracy | Recall@5 | LLM checks | Wall time |
|---|---:|---:|---:|---:|---:|
| Unranked baseline | 0.56 | 0.20 | 0.20 | 225 | 343.431 s |
| BM25 ranked | 0.54 | 0.12 | 0.12 | 242 | 382.582 s |
| MiniLM dense ranked | 0.52 | 0.12 | 0.12 | 239 | 379.442 s |

On this pilot, both ranking approaches performed below the unranked baseline
for overall accuracy, positive-source accuracy, and wall time. These results
are specific to this sample, model, and five-passage limit; they should not be
treated as full-dataset results.

## Full-scale experiment

The full-scale run evaluated all **800 instances** (400 positive and 400
negative) with Qwen 2.5 3B through local Ollama. The run used restartable
75-instance batches and `--max-checks 0` (check passages until support is found
or the passages are exhausted). Batch outputs were merged into the full-run
summary and per-instance logs.

| Task | Work completed | Output |
|---|---|---|
| 1. Run the full dataset | Evaluated all 800 instances for the unranked, BM25, and MiniLM strategies | `results/batches/` |
| 2. Merge batch results | Combined batch outputs into the aggregate metrics and detailed per-instance logs | `results/summary_metrics.json`, `results/summary_metrics.txt`, `results/detailed_logs.json` |
| 3. Compare retrieval budgets | Derived predictions for limits of 1, 3, 5, 10, 20, 30, and 50 passages, plus all passages, using recorded full-run ranks; no additional Ollama calls | `results/budget_eval/` |

### Full-scale results

| Strategy | Accuracy | Positive-source accuracy | Negative-distractor accuracy | Recall@5 | MRR | LLM checks | Avg. checks/instance | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Unranked baseline | 60.00% | 35.75% | 84.25% | 24.25% | 0.1833 | 31,796 | 39.74 | 13h 42m 57s |
| BM25 ranked | 60.00% | 35.75% | 84.25% | 21.50% | 0.1507 | 31,198 | 39.00 | 13h 30m 51s |
| MiniLM dense ranked | 60.00% | 35.75% | 84.25% | 22.25% | 0.1523 | 31,204 | 39.01 | 13h 32m 01s |

All three strategies produced the same final predictions when allowed to
exhaust the passages. Ranking therefore did not improve full-run accuracy.
BM25 and MiniLM used slightly fewer verifier checks than the baseline, but
ranking time reduced the net wall-time savings to about 1.5% and 1.3%,
respectively.

### Fixed-budget results

These are exploratory operating points reconstructed from the full-run
`found_at_rank` values. At each budget, a prediction is positive only if the
recorded support rank falls within that limit; otherwise it is negative. The
`all` rows reproduce the exhaustive full-scale results. This is not a new
verifier run or an independent test set.

| Budget | Strategy | Accuracy | Positive-source accuracy | Negative-distractor accuracy | Positive precision | FP | FN | Checks |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | Unranked baseline | 53.50% | 13.25% | 93.75% | 67.95% | 25 | 347 | 800 |
| 1 | BM25 ranked | 54.13% | 9.75% | 98.50% | 86.67% | 6 | 361 | 800 |
| 1 | MiniLM dense ranked | 53.12% | 8.50% | 97.75% | 79.07% | 9 | 366 | 800 |
| 3 | Unranked baseline | 56.37% | 21.25% | 91.50% | 71.43% | 34 | 315 | 2,217 |
| 3 | BM25 ranked | 56.13% | 16.75% | 95.50% | 78.82% | 18 | 333 | 2,289 |
| 3 | MiniLM dense ranked | 56.50% | 19.50% | 93.50% | 75.00% | 26 | 322 | 2,277 |
| 5 | Unranked baseline | 57.50% | 24.25% | 90.75% | 72.39% | 37 | 303 | 3,573 |
| 5 | BM25 ranked | 57.75% | 21.50% | 94.00% | 78.18% | 24 | 314 | 3,705 |
| 5 | MiniLM dense ranked | 57.63% | 22.25% | 93.00% | 76.07% | 28 | 311 | 3,661 |
| 10 | Unranked baseline | 58.50% | 27.00% | 90.00% | 72.97% | 40 | 292 | 6,864 |
| 10 | BM25 ranked | 59.00% | 26.75% | 91.25% | 75.35% | 35 | 293 | 7,090 |
| 10 | MiniLM dense ranked | 59.75% | 29.25% | 90.25% | 75.00% | 39 | 283 | 6,995 |
| 20 | Unranked baseline | 60.12% | 31.50% | 88.75% | 73.68% | 45 | 274 | 13,157 |
| 20 | BM25 ranked | 60.25% | 32.50% | 88.00% | 73.03% | 48 | 270 | 13,382 |
| 20 | MiniLM dense ranked | 60.00% | 32.75% | 87.25% | 71.98% | 51 | 269 | 13,181 |
| 30 | Unranked baseline | 59.38% | 32.50% | 86.25% | 70.27% | 55 | 270 | 18,597 |
| 30 | BM25 ranked | 60.50% | 34.75% | 86.25% | 71.65% | 55 | 261 | 18,745 |
| 30 | MiniLM dense ranked | 60.50% | 34.75% | 86.25% | 71.65% | 55 | 261 | 18,527 |
| 50 | Unranked baseline | 59.38% | 34.00% | 84.75% | 69.04% | 61 | 264 | 23,414 |
| 50 | BM25 ranked | 60.00% | 35.75% | 84.25% | 69.42% | 63 | 257 | 23,318 |
| 50 | MiniLM dense ranked | 60.38% | 35.50% | 85.25% | 70.65% | 59 | 258 | 23,154 |
| All | Unranked baseline | 60.00% | 35.75% | 84.25% | 69.42% | 63 | 257 | 31,796 |
| All | BM25 ranked | 60.00% | 35.75% | 84.25% | 69.42% | 63 | 257 | 31,198 |
| All | MiniLM dense ranked | 60.00% | 35.75% | 84.25% | 69.42% | 63 | 257 | 31,204 |

At 30 passages, BM25 and MiniLM have the highest overall accuracy (60.50%),
but positive-source accuracy (34.75%) remains below the exhaustive result
(35.75%). The budget curve shows a tradeoff between recovering positive
sources and rejecting negative distractors; these results alone do not
establish a general accuracy improvement.

## Repository contents

- `data/pdfs/`: locally cached source PDFs; excluded from Git.
- `data/pdf links.txt`: filename list with arXiv abstract and direct PDF links.
- `data/thesis_dataset.jsonl`: generated evaluation instances.
- `scripts/build_dataset.py`: local PDF and citation dataset builder.
- `scripts/run_experiments.py`: BM25, MiniLM, and Ollama evaluation.
- `logs/llm_verification_cache.jsonl`: cached local verifier responses.
- `results_pilot/`: pilot aggregate metrics, detailed logs, and text summary.
- `results/batches/`: restartable full-dataset batch outputs before merging.
- `results/budget_eval/`: fixed-budget evaluation using cached verifier judgments.
