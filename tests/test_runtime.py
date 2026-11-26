import sys
from pathlib import Path

from src import pipeline

root = Path(sys.argv[1])

print("STANDALONE RUNTIME TEST")
print("=" * 65)

print("Initializing retrieval models...")

pipeline.initialize(
    root=root,
    load_llm=False
)

assert len(pipeline.chunks_df) == 433
assert pipeline.index.ntotal == 433

print("[PASS] Corpus initialization")
print("[PASS] FAISS initialization")
print("[PASS] BM25 initialization")
print("[PASS] Embedding and reranking models")

query = "How does DPR calculate similarity?"

methods = {
    "BM25": pipeline.keyword_search,
    "Dense Semantic": pipeline.semantic_search,
    "Hybrid RRF": pipeline.hybrid_search,
    "Hybrid + Cross-Encoder": pipeline.reranked_search,
    "Document-Aware": pipeline.document_aware_search
}

for name, method in methods.items():

    results = method(query, top_k=3)

    assert not results.empty
    assert len(results) <= 3

    print("\n[PASS]", name)

    print(
        results[
            ["chunk_id", "document", "page"]
        ].to_string(index=False)
    )

print("\nAll standalone retrieval tests passed!")
