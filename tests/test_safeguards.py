from src.pipeline import (
    detect_documents,
    validate_answer_structure,
    build_evidence_fallback
)

sources = [
    {
        "source_id": "S1",
        "document": "Test Document",
        "page": 1,
        "chunk_id": "test_1",
        "text": "DPR uses dot-product similarity."
    }
]

assert detect_documents(
    "How does Sentence-BERT work?"
) == ["sentence_bert"]

assert set(detect_documents(
    "Compare BERT and Sentence-BERT"
)) == {"bert", "sentence_bert"}

assert detect_documents(
    "What causes vitamin D deficiency?"
) == []

print("[PASS] Document recognition")

cases = [
    (
        "[S1] DPR uses dot-product similarity.",
        False,
        True
    ),
    (
        "DPR uses dot-product similarity.",
        False,
        False
    ),
    (
        "[S9] DPR uses dot-product similarity.",
        False,
        False
    ),
    (
        "[S1] DPR uses",
        True,
        False
    ),
    (
        "Insufficient evidence in the provided documents.",
        False,
        True
    )
]

for answer, truncated, expected in cases:

    result = validate_answer_structure(
        answer,
        sources,
        reached_token_limit=truncated
    )

    assert result["passed"] == expected

print("[PASS] All five structural validation cases")

fallback = build_evidence_fallback(sources)

assert "[S1]" in fallback
assert "Test Document" in fallback
assert "DPR uses dot-product similarity." in fallback

print("[PASS] Evidence fallback")

print("\nAll automated safeguard tests passed!")
