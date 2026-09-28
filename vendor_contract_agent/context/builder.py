from __future__ import annotations

from typing import Any


def build_routed_context(
    *,
    contract_payload: dict[str, Any] | None = None,
    legal_payload: dict[str, Any] | None = None,
) -> str:
    """Build LLM context with strict separation of contract vs legal sources."""
    blocks: list[str] = []

    if contract_payload is not None:
        if not contract_payload.get("available", True):
            blocks.append(
                "[INTERNAL CONTRACT]\n"
                f"Retrieval error: {contract_payload.get('error') or 'unavailable'}"
            )
        else:
            results = contract_payload.get("results") or []
            if not results:
                blocks.append(
                    "[INTERNAL CONTRACT]\n"
                    "No matching contractual evidence was retrieved."
                )
            for index, item in enumerate(results, start=1):
                blocks.append(
                    "\n".join(
                        [
                            f"[INTERNAL CONTRACT {index}]",
                            f"Contract:\n{item.get('contract_id') or 'unknown'}",
                            f"Document:\n{item.get('document') or 'unknown'}",
                            f"Clause:\n{item.get('clause_number') or 'n/a'}",
                            f"Type:\n{item.get('clause_type') or 'n/a'}",
                            f"Page:\n{item.get('page') if item.get('page') is not None else 'n/a'}",
                            f"Score:\n{item.get('score')}",
                            f"Content:\n{item.get('content') or ''}",
                        ]
                    )
                )

    if legal_payload is not None:
        jurisdiction = legal_payload.get("jurisdiction") or "DZ"
        if not legal_payload.get("available", True):
            blocks.append(
                "[EXTERNAL LEGAL]\n"
                f"Jurisdiction: {jurisdiction}\n"
                f"Retrieval error: {legal_payload.get('error') or 'unavailable'}"
            )
        else:
            results = legal_payload.get("results") or []
            if not results:
                blocks.append(
                    f"[EXTERNAL LEGAL]\nJurisdiction: {jurisdiction}\n"
                    "No matching legal evidence was retrieved."
                )
            for index, item in enumerate(results, start=1):
                blocks.append(
                    "\n".join(
                        [
                            f"[EXTERNAL LEGAL {index}]",
                            f"Jurisdiction:\n{jurisdiction}",
                            f"Title:\n{item.get('title') or 'n/a'}",
                            f"Reference:\n{item.get('reference') or 'n/a'}",
                            f"Document type:\n{item.get('document_type') or 'n/a'}",
                            f"Date:\n{item.get('date') or 'n/a'}",
                            f"URL:\n{item.get('url') or 'n/a'}",
                            f"Content:\n{item.get('content') or ''}",
                        ]
                    )
                )

    if not blocks:
        return (
            "NO RETRIEVED CONTEXT.\n"
            "Answer using general knowledge only when the route is NO_RETRIEVAL. "
            "Do not invent contractual or legal citations."
        )
    return "\n\n".join(blocks)


def sources_from_payloads(
    *,
    contract_payload: dict[str, Any] | None = None,
    legal_payload: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    if contract_payload and contract_payload.get("available", True):
        for item in contract_payload.get("results") or []:
            content = item.get("content") or ""
            sources.append(
                {
                    "source_type": "contract",
                    "document": item.get("document"),
                    "contract_id": item.get("contract_id"),
                    "clause": item.get("clause_number"),
                    "page": item.get("page"),
                    "text": (content[:500] + "...") if len(content) > 500 else content,
                }
            )
    if legal_payload and legal_payload.get("available", True):
        for item in legal_payload.get("results") or []:
            content = item.get("content") or ""
            sources.append(
                {
                    "source_type": "legal",
                    "title": item.get("title"),
                    "reference": item.get("reference"),
                    "jurisdiction": legal_payload.get("jurisdiction") or "DZ",
                    "date": item.get("date"),
                    "url": item.get("url"),
                    "text": (content[:500] + "...") if len(content) > 500 else content,
                }
            )
    return sources
