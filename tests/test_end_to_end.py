import sys
import json
from pathlib import Path

from src import pipeline

root = Path(sys.argv[1])

print("DOCUMIND AI — END-TO-END TEST")
print("=" * 65)

print("Initializing complete RAG pipeline...")

pipeline.initialize(
    root=root,
    load_llm=True
)

print("[PASS] Complete pipeline initialization")

test_cases = [
    {
        "name": "Evidence Search",
        "query": "How does DPR calculate similarity?",
        "evidence_only": True
    },
    {
        "name": "Unanswerable Question",
        "query": "How many parameters does GPT-4 have?",
        "evidence_only": False
    },
    {
        "name": "AI Question Answering",
        "query": "What is the purpose of masked language modeling in BERT?",
        "evidence_only": False
    }
]

test_results = []

for case in test_cases:

    print("\n" + "=" * 65)
    print("TEST:", case["name"])
    print("QUESTION:", case["query"])

    result = pipeline.documind_qa_v5(
        query=case["query"],
        top_k=3,
        max_new_tokens=96,
        evidence_only=case["evidence_only"]
    )

    print("\nSTATUS:", result["status"])
    print("\nANSWER:", result["answer"])
    print("\nDETECTED DOCUMENTS:",
          result["detected_documents"])
    print("RELEVANCE SCORE:",
          result["top_relevance_score"])
    print("RETRIEVAL TIME:",
          result["retrieval_time"])
    print("GENERATION TIME:",
          result["generation_time"])
    print("TOTAL TIME:",
          result["total_time"])
    print("VALIDATION ISSUES:",
          result["validation_issues"])

    print("\nSOURCES:")

    for source in result["sources"]:

        print(
            source["source_id"],
            source["document"],
            "Page:", source["page"]
        )

    if case["name"] == "Evidence Search":

        assert result["status"] == "evidence_only"
        assert len(result["sources"]) > 0
        assert result["generated_tokens"] == 0

    elif case["name"] == "Unanswerable Question":

        assert result["status"] in [
            "rejected_low_relevance",
            "model_abstained"
        ]

        assert result["answer"].strip() == (
            "Insufficient evidence in the provided documents."
        )

        if result["status"] == "rejected_low_relevance":

            assert result["generated_tokens"] == 0

            print("[PASS] Relevance gate rejection")

        else:

            print("[WARNING] Relevance gate false acceptance")
            print("[PASS] Secondary LLM abstention")

            assert result["generated_tokens"] > 0

    else:

        assert result["generated_tokens"] > 0

        assert result["status"] in [
            "structurally_validated_answer",
            "evidence_fallback",
            "model_abstained"
        ]

    test_results.append({
        "test": case["name"],
        "query": case["query"],
        "status": result["status"],
        "retrieval_time": result["retrieval_time"],
        "generation_time": result["generation_time"],
        "total_time": result["total_time"],
        "generated_tokens": result["generated_tokens"],
        "validation_issues": result["validation_issues"],
        "source_count": len(result["sources"])
    })

    print("\n[PASS] Functional test:", case["name"])

output_path = (
    root / "results/standalone_end_to_end_test.json"
)

output_path.write_text(
    json.dumps(test_results, indent=4),
    encoding="utf-8"
)

print("\n" + "=" * 65)
print("All end-to-end functional tests passed!")
print("Results saved:", output_path)
