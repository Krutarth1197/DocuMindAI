def calculate_retrieval_metrics(
    retrieved_ids,
    relevant_ids
):

    retrieved_ids = list(
        dict.fromkeys(retrieved_ids)
    )

    relevant_ids = set(relevant_ids)

    assert relevant_ids, "Missing ground truth labels"

    metrics = {}

    for k in [1, 3, 5]:

        top_k_ids = retrieved_ids[:k]

        relevant_retrieved = len(
            set(top_k_ids) & relevant_ids
        )

        metrics[f"hit_at_{k}"] = int(
            relevant_retrieved > 0
        )

        metrics[f"recall_at_{k}"] = (
            relevant_retrieved / len(relevant_ids)
        )

        metrics[f"precision_at_{k}"] = (
            relevant_retrieved / k
        )

    reciprocal_rank = 0.0

    for rank, chunk_id in enumerate(
        retrieved_ids[:5],
        start=1
    ):

        if chunk_id in relevant_ids:

            reciprocal_rank = 1.0 / rank

            break

    metrics["reciprocal_rank_at_5"] = reciprocal_rank

    metrics["all_annotated_chunks_at_5"] = int(
        relevant_ids.issubset(
            set(retrieved_ids[:5])
        )
    )

    return metrics
