# Experimental Results

All formal baseline experiments use Corpus Version 2,
consisting of five research papers, 74 PDF pages,
and 433 indexed passages.

## Retrieval

- 25 manually annotated questions.
- Four retrieval architectures.
- Hit@K, Recall@K, Precision@K and MRR.
- Latency and paired statistical comparisons.

## Abstention

- 35 evaluation questions.
- 25 answerable and 10 presumed unanswerable questions.
- Relevance threshold fixed at 0.0 before evaluation.
- Results are limited to the constructed benchmark.

## Generation Baseline

- 10 answerable and 5 unanswerable questions.
- Purposively selected diagnostic subset.
- LLM-assisted manual rubric assessment.
- Independent human verification remains pending.

## Development

Contains threshold-selection experiments and
corpus development statistics.

These results are not independent test measurements.

## V5 Diagnostics

Contains targeted document-aware retrieval experiments.

Pipeline V5 introduces document recognition,
bibliography penalties, context diversity,
structural validation and evidence fallback.

V5 has not undergone independent full-benchmark
evaluation.

## Limitations

- Small research corpus.
- Non-exhaustive relevance annotations.
- PDF extraction artifacts.
- Slow CPU-based language model generation.
- Incomplete factual and citation support.
- Citation ID validity does not establish factual accuracy.
- Results should not be generalized beyond the benchmark.
