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

For long full-dataset runs, use restartable 25-instance batches instead:

```bash
.venv/bin/python scripts/run_experiment_batches.py \
  --batch-size 25 \
  --max-checks 0 \
  --model qwen2.5:3b \
  --ollama-url http://localhost:11434
```

Each completed batch is saved under `results/batches/` and skipped on restart.
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

## Repository contents

- `data/pdfs/`: locally cached source PDFs; excluded from Git.
- `data/pdf links.txt`: filename list with arXiv abstract and direct PDF links.
- `data/thesis_dataset.jsonl`: generated evaluation instances.
- `scripts/build_dataset.py`: local PDF and citation dataset builder.
- `scripts/run_experiments.py`: BM25, MiniLM, and Ollama evaluation.
- `logs/llm_verification_cache.jsonl`: cached local verifier responses.
- `results_pilot/`: pilot aggregate metrics, detailed logs, and text summary.
- `results/batches/`: restartable full-dataset batch outputs before merging.
