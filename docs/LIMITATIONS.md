# Research Limitations

## Corpus Scope

The experimental corpus contains five NLP research
papers, 74 PDF pages and 433 indexed passages.
Results cannot be generalized to arbitrary documents
or larger document collections.

## Retrieval Evaluation

The formal retrieval benchmark contains 25
answerable questions with non-exhaustive,
AI-assisted relevance annotations.

Cross-encoder reranking achieved 80% Hit@5.
Its improvement over dense retrieval was statistically
significant in the constructed benchmark, but its
improvement over BM25 was not.

## Relevance Gate

The relevance gate achieved 100% accuracy on the
constructed 35-question held-out benchmark.

However, a separate standalone diagnostic involving
GPT-4 parameter counts produced a false acceptance.
The language model subsequently abstained correctly.

Therefore, perfect benchmark performance does not
establish universal abstention reliability.

## Answer Generation

The Qwen2.5-1.5B-Instruct model demonstrated
inconsistent citation formatting and incomplete
answer generation.

Baseline generation evaluation achieved:

- Correctness rubric score: 75%
- Completeness rubric score: 50%
- Citation support rubric score: 25%

These are rubric scores, not general QA accuracy.

## Structural Validation

Citation validation checks citation identifiers and
selected formatting requirements.

It does not establish factual correctness or semantic
support for every generated claim.

## Computational Constraints

All experiments were conducted using CPU resources.
Language model generation can take several minutes.

## Reproducibility

Standalone execution has been verified using the
existing indexed corpus and installed environment.

Fresh-install reproducibility and exact equivalence
between notebook and standalone execution remain
unverified.

## Future Work

- Larger and more diverse document collections
- Independent human relevance annotation
- Improved document extraction
- Semantic citation verification
- Better calibrated relevance gating
- Stronger instruction-following language models
- Independent evaluation of Pipeline V5
- Automated fresh-environment reproduction
