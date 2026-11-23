# Confirmatory experiment protocol

## Dataset and labels

The fixed dataset contains 800 instances arranged as 400 claim pairs: one
automatically constructed citation-linked positive and one semantically similar,
non-cited-source negative per pair. These are **silver labels**, not human gold
annotations. Citation presence and distractor selection do not prove that the
claim is supported or unsupported. Results therefore measure agreement with
the construction labels and must not be presented as validated scientific
fact-checking accuracy.

The dataset is frozen at the SHA-256 recorded in `results/splits.json`. Do not
change its rows after preparing the split manifest.

## Grouping and data split

Use the deterministic seed `231104`. Connect instances that share any
`pair_id`, `citing_arxiv_id`, or `source_arxiv_id`; a connected component is
the indivisible group. The components are assigned approximately 60%/20%/20%
to train, development, and test. No pair, citing paper, or source paper may
cross split boundaries. The actual row counts are recorded in the split
manifest because whole components can make the ratios inexact.

The development set is the only set used to choose a passage budget. Freeze
one primary budget before test evaluation. The test set is evaluated once;
additional test-budget curves are descriptive and must not be used to select
the primary result.

For timing stability, run the development split three times in separate
result directories with fresh caches and different `--strategy-seed` values.
The execution order is randomized from that seed. Use these development runs
to report the median and range of elapsed time. The primary test run is done
once with a frozen budget; do not use repeated test runs to select a method.

## Task, methods, and hypotheses

The final source-level task is binary: `SUPPORTS` versus
`NOT_ENOUGH_INFORMATION`. The per-passage verifier is constrained to return
`SUPPORTS`, `REFUTES`, or `NOT_ENOUGH_INFORMATION`; because the dataset has no
refuting-source class, `REFUTES` and `NOT_ENOUGH_INFORMATION` both map to the
negative source-level prediction. No three-class accuracy is claimed.

Compare document-order baseline, BM25, and all-MiniLM-L6-v2 cosine ranking.
Use identical 90–250-word extracted passages and the same locally served
Qwen 2.5 3B verifier configuration for all methods. The prespecified
descriptive budget curve is K = 1, 3, 5, 10, and 20 passages. The primary K
must be selected using development data before inspecting test results.

The null hypothesis for each ranking method is no change in pair/citing/source
group-level binary accuracy versus document order. Report the cluster-level
paired sign-flip permutation test; the small number of independent groups
limits statistical power.

## Measurements

Report the confusion matrix, accuracy, support precision/recall/F1, negative
accuracy, first-passage support recall, median and 95th-percentile checks,
verifier-confirmed positive rates at ranks 1/3/5, and verifier-confirmed
reciprocal rank. These rank measures are not Recall@K or MRR against gold
evidence because no gold passages exist. Also report actual process wall time,
ranking time, model API latency, and cache hit/miss counts. Report Wilson
intervals with their row-level independence caveat and connected-group
bootstrap 95% intervals. Repeated verifier calls on a fixed sample measure
repeatability only, not correctness.

Use fresh cache namespaces for each fixed-budget run so elapsed time represents
actual model calls. Cache keys include the model digest, Ollama version, prompt
and parser versions, structured-output schema, sampling options, claim, and
passage. SQLite provides transactional local caching. Do not share a SQLite
cache over a network drive; each machine must keep a separate cache and result
directory.

## Interpretation limits

The verifier and the automatic citation/distractor construction are not gold
standards. Error categories in output are outcome diagnostics, not verified
causal attributions to retrieval versus verification. Historical exhaustive
results are exploratory and do not replace the grouped held-out experiment.
No improvement may be claimed from small timing differences without paired
uncertainty and a fresh held-out run.
