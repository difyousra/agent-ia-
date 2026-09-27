from __future__ import annotations

from typing import Sequence

from .schemas import RerankedChunk, RetrievedChunk, SourceRef


def build_context(results: Sequence[RerankedChunk | RetrievedChunk]) -> str:
    """Build agent context while preserving source metadata."""
    if not results:
        return (
            "NO RETRIEVED CONTEXT.\n"
            "There is insufficient contractual evidence in the knowledge base "
            "to answer the question."
        )

    blocks: list[str] = []
    for index, item in enumerate(results, start=1):
        if isinstance(item, RerankedChunk):
            meta = item.metadata
            content = item.content
            score = item.rerank_score
        else:
            meta = {
                "document_name": item.document_name,
                "contract_id": item.contract_id,
                "clause_number": item.clause_number,
                "clause_type": item.clause_type,
                "section": item.section,
                "page": item.page,
            }
            content = item.content
            score = item.score

        blocks.append(
            "\n".join(
                [
                    f"[SOURCE {index}]",
                    f"[DOCUMENT]\n{meta.get('document_name') or 'unknown'}",
                    f"[CONTRACT_ID]\n{meta.get('contract_id') or 'unknown'}",
                    f"[CLAUSE]\n{meta.get('clause_number') or 'n/a'}",
                    f"[TYPE]\n{meta.get('clause_type') or 'n/a'}",
                    f"[SECTION]\n{meta.get('section') or 'n/a'}",
                    f"[PAGE]\n{meta.get('page') if meta.get('page') is not None else 'n/a'}",
                    f"[SCORE]\n{score:.4f}",
                    f"[CONTENT]\n{content}",
                ]
            )
        )
    return "\n\n".join(blocks)


def sources_from_results(results: Sequence[RerankedChunk | RetrievedChunk]) -> list[SourceRef]:
    sources: list[SourceRef] = []
    for item in results:
        if isinstance(item, RerankedChunk):
            meta = item.metadata
            content = item.content
        else:
            meta = {
                "document_name": item.document_name,
                "contract_id": item.contract_id,
                "clause_number": item.clause_number,
                "clause_type": item.clause_type,
                "page": item.page,
            }
            content = item.content
        sources.append(
            SourceRef(
                document_name=meta.get("document_name"),
                contract_id=meta.get("contract_id"),
                clause_number=meta.get("clause_number"),
                clause_type=meta.get("clause_type"),
                page=meta.get("page"),
                text=(content[:500] + "...") if len(content) > 500 else content,
            )
        )
    return sources
