from __future__ import annotations

from .schemas import RetrievedChunk


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]],
    *,
    k: int = 60,
    top_k: int = 20,
) -> list[RetrievedChunk]:
    """Combine ranked lists with Reciprocal Rank Fusion.

    score(d) = sum_i 1 / (k + rank_i(d))
    """
    fused: dict[str, RetrievedChunk] = {}
    scores: dict[str, float] = {}

    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            chunk_id = item.id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (k + rank))
            if chunk_id not in fused:
                fused[chunk_id] = item.model_copy(deep=True)

    ranked_ids = sorted(scores.keys(), key=lambda cid: scores[cid], reverse=True)
    output: list[RetrievedChunk] = []
    for chunk_id in ranked_ids[:top_k]:
        chunk = fused[chunk_id]
        chunk.score = scores[chunk_id]
        chunk.source = "hybrid"
        output.append(chunk)
    return output
