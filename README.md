# RTU Thesis Experimentation

This project builds a local paper dataset and evaluates whether passage ranking
helps retrieve evidence for citation claims. Strategies are an unranked
baseline, BM25, and MiniLM dense ranking, followed by a local Qwen 2.5 3B
verifier through Ollama.

The source-level decision task is **binary SUPPORTS versus
NOT_ENOUGH_INFORMATION**. The per-passage verifier uses structured
`SUPPORTS`/`REFUTES`/`NOT_ENOUGH_INFORMATION` labels; REFUTES and
NOT_ENOUGH_INFORMATION both map to a negative source-level decision because
the dataset has no human-validated refuting-source class. Invalid, partial,
contradictory, or extra-key responses are retained as parse failures along
with their raw response.

## Dataset and limitations

Build the JSONL dataset from PDFs in `data/pdfs/`:

```bash
python3 scripts/build_dataset.py \
  --pdf-dir data/pdfs \
  --target-positives 500 \
  --output data/thesis_dataset.jsonl
```

The builder creates 500 positive and 500 negative rows (1,000 total), but
these are **silver labels**, not human gold judgments. Citation-derived
positives do not prove support; embedding-selected negatives do not prove
irrelevance. The current build is based on 2,495 available linked PDFs; 2,493
were valid and two excluded. The build report lists exclusions and source
audits. The 500 `pair_id`s include 100 explicitly marked alternative-distractor
augmentations; there are 400 initially resolved citing/source claim pairs.
Therefore, 1,000 rows must not be described as 1,000 independent observations.
The generated dataset and annotation CSV include passages extracted from
research papers, so they remain local and are not included in the public push.
The Git-tracked 800-row dataset from the previous protocol is not the same
artifact as the current local 1,000-row build.

PDF block ordering and extraction are heuristic. The audit fields record PDF
filename/hash, extracted text size, page/passage counts, and other construction
details, but do not establish that all PDF content was recovered. Current
passages use the 90–250-word segmentation. To store optional sentence,
three-sentence, and section-oriented segmentation ablations, build with
`--include-segmentation-ablations`; this increases dataset size. PDF blocks
are sorted by y/x coordinates, blocks under 15 words and
blocks beginning with selected figure/table/algorithm/equation markers are
filtered, and references are removed only when a heading is detected. There is
no explicit header/footer removal or two-column layout model; table fragments,
captions, equations, and incomplete sentences may remain. The recorded
extraction score is a length/passage-count heuristic, not a human-validated
quality measure.

`pair_id` connects the positive and negative instance for one paired candidate.
Rows can also share citing papers, sources, or claim contexts across pair IDs.
Statistical grouping therefore uses connected components over `pair_id`,
`citing_arxiv_id`, and `source_arxiv_id`. Do not treat individual rows or
processing batches as independent statistical units.

The generated dataset contains no human gold verdict or gold evidence
passages. `found_at_rank` is the verifier's predicted-support rank, not a gold
evidence rank. Consequently, gold Recall@K/MRR and human inter-annotator
agreement are unavailable until annotation is completed. Create an annotation
template for 150 pairs (300 claim/source instances) with two annotator fields
and candidate evidence passages:

```bash
python3 scripts/human_annotations.py --mode template --sample-pairs 150
```

Annotators should independently label `SUPPORTS`, `REFUTES`, or
`NOT_ENOUGH_INFORMATION`, identify exact supporting passage IDs for SUPPORTS,
and categorize negative sources as clearly unrelated, semantically similar,
shared terminology, same experiment/different conditions, contradictory, same
topic/insufficient evidence, or other. After both annotators complete the
template, calculate agreement with:

```bash
python3 scripts/human_annotations.py --mode analyze
```

No human labels, evidence gold, or inter-annotator agreement are currently
available. Do not report the template or automatic silver labels as completed
annotation.

## Grouped split and experiment protocol

Create a deterministic, dataset-hash-bound split manifest:

```bash
python3 scripts/prepare_splits.py \
  --dataset data/thesis_dataset.jsonl \
  --seed 231104 \
  --output results/splits_connected.json
```

Run a fresh evaluation with one fixed passage budget:

```bash
python3 scripts/run_experiments.py \
  --dataset data/thesis_dataset.jsonl \
  --split-manifest results/splits_connected.json --split development \
  --max-checks 3 --fresh-run --strategy-seed 231104 \
  --results-dir results/development_k3
```

`--max-checks 0` checks all passages. Use `scripts/run_budget_suite.py` for
separate fresh runs at K=1,3,5,10,20. The split manifest must be regenerated
whenever dataset bytes change. Use only development data to choose the primary
budget; reserve test for one final run. Earlier full-data K=1 runs used the
same examples later assigned to test and are exploratory, not held-out results.
Their original dataset hash and verifier configuration were not recorded, so
they do not validate the current 3-label structured-output implementation.

Outputs include per-instance logs, confusion matrices, support
precision/recall/F1, row-level Wilson intervals, connected-group bootstrap
intervals, paired connected-group bootstrap confidence intervals and
sign-flip tests, verifier/cache diagnostics, ranking time, and model-miss
latency. Cache hits have zero model latency. The runner reports
human gold evidence metrics only when gold passages exist. Whole-process
wall-clock must be measured from a fresh run, not reconstructed from cached
per-call latency.

`scripts/evaluate_cached_budgets.py` reconstructs predictions from an
exhaustive run and is **post hoc**, not an independent K-passage experiment.
`scripts/analyze_existing_results.py` recalculates grouped statistics from
completed logs but cannot recover unavailable wall-clock, gold, or human
annotation data.

Earlier pilot and full-run results used an older dataset/version and automatic
silver labels. They are exploratory only, not confirmatory held-out results
for the current dataset. Current conclusions remain inconclusive until the
human validation and gold evidence set are completed.

## Model setup

Install dependencies and retrieve the model:

```bash
python3 -m pip install -r requirements.txt
ollama pull qwen2.5:3b
ollama serve
```

The runner defaults to `http://localhost:11434`; use `--ollama-url` or
`OLLAMA_HOST` to select another endpoint.

## Main files

- `data/thesis_dataset.jsonl`: generated evaluation instances.
- `data/thesis_dataset.build_report.json`: PDF and dataset construction counts.
- `scripts/build_dataset.py`: PDF extraction and dataset builder.
- `scripts/download_pdfs.py`: resumable PDF downloader and validation report.
- `scripts/run_experiments.py`: ranking and verifier evaluation.
- `scripts/experiment_utils.py`: connected groups, grouped splits, bootstrap,
  and paired tests.
- `scripts/prepare_splits.py`: dataset-hash-bound split manifest generator.
- `scripts/human_annotations.py`: annotation template and agreement analysis.
- `scripts/analyze_existing_results.py`: grouped reanalysis of completed logs.
- `tests/`: focused verifier-parser and grouped-statistics tests.
- `results/fresh_k1_no_batches/`: earlier exploratory full-dataset K=1 runs,
  not held out and not human-gold evaluation.

PDFs under `data/pdfs/` are local and excluded from Git.
