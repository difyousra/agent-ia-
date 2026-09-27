from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .ingest import extract_contract_text
from .llm import LLMClient
from .normalize import extract_json_payload, normalize_list_items, normalize_result
from .prompts import RAG_SYSTEM_PROMPT, RAG_USER_PROMPT_TEMPLATE, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE
from .rag.pipeline import RAGPipeline
from .rag.schemas import ContractQueryResponse, FactWithSource, RetrievalDebug, SourceRef

logger = logging.getLogger(__name__)


class VendorContractAgent:
    """Agent unique d'analyse de contrat fournisseur (optionnellement augmenté par RAG)."""

    def __init__(self, llm: LLMClient | None = None, rag: RAGPipeline | None = None) -> None:
        self.llm = llm or LLMClient()
        self.rag = rag

    def analyze_file(self, path: str | Path) -> dict[str, Any]:
        contract_text = extract_contract_text(path)
        return self.analyze_text(contract_text)

    def analyze_text(self, contract_text: str) -> dict[str, Any]:
        text = (contract_text or "").strip()
        if not text:
            raise ValueError("Le texte du contrat est vide.")

        user_prompt = USER_PROMPT_TEMPLATE.replace("<<<CONTRACT_TEXT>>>", text)
        raw = self.llm.complete(SYSTEM_PROMPT, user_prompt)
        parsed = extract_json_payload(raw)
        normalized = normalize_result(parsed)
        return normalize_list_items(normalized)

    def answer_query(
        self,
        query: str,
        *,
        contract_id: str | None = None,
        debug: bool = False,
        top_k: int | None = None,
        top_n: int | None = None,
    ) -> ContractQueryResponse:
        """Question answering via hybrid RAG + existing LLM agent."""
        if not query or not str(query).strip():
            return ContractQueryResponse(
                answer=None,
                insufficient_information=True,
                confidence=0.0,
                facts=[],
                actions=[],
                risks=[],
                sources=[],
            )

        pipeline = self.rag or RAGPipeline()
        try:
            retrieval = pipeline.retrieve(
                query.strip(),
                contract_id=contract_id,
                top_k=top_k,
                top_n=top_n,
            )
        except ValueError as exc:
            logger.warning("RAG retrieval validation error: %s", exc)
            return ContractQueryResponse(
                answer=None,
                insufficient_information=True,
                confidence=0.0,
                facts=[],
                actions=[],
                risks=[],
                sources=[],
                analysis={"error": str(exc)},
            )
        except Exception as exc:
            logger.exception("RAG retrieval failed")
            return ContractQueryResponse(
                answer=None,
                insufficient_information=True,
                confidence=0.0,
                facts=[],
                actions=[],
                risks=[],
                sources=[],
                analysis={"error": f"retrieval_unavailable: {exc}"},
            )

        if retrieval["insufficient_information"]:
            response = ContractQueryResponse(
                answer=None,
                insufficient_information=True,
                confidence=0.0,
                facts=[],
                actions=[],
                risks=[],
                sources=retrieval["sources"],
            )
            if debug:
                response.debug = RetrievalDebug(
                    semantic_results=retrieval["semantic_results"],
                    keyword_results=retrieval["keyword_results"],
                    rrf_results=retrieval["rrf_results"],
                    reranked_results=retrieval["reranked_results"],
                )
            return response

        user_prompt = (
            RAG_USER_PROMPT_TEMPLATE.replace("<<<USER_QUERY>>>", query.strip()).replace(
                "<<<RETRIEVED_CONTEXT>>>",
                retrieval["context"],
            )
        )
        try:
            raw = self.llm.complete(RAG_SYSTEM_PROMPT, user_prompt)
            parsed = extract_json_payload(raw)
        except Exception as exc:
            logger.exception("LLM JSON generation failed")
            return ContractQueryResponse(
                answer=None,
                insufficient_information=True,
                confidence=0.0,
                facts=[],
                actions=[],
                risks=[],
                sources=retrieval["sources"],
                analysis={"error": f"invalid_llm_json: {exc}"},
                debug=RetrievalDebug(
                    semantic_results=retrieval["semantic_results"],
                    keyword_results=retrieval["keyword_results"],
                    rrf_results=retrieval["rrf_results"],
                    reranked_results=retrieval["reranked_results"],
                )
                if debug
                else None,
            )

        response = _parse_rag_response(parsed, retrieval["sources"])
        if debug:
            response.debug = RetrievalDebug(
                semantic_results=retrieval["semantic_results"],
                keyword_results=retrieval["keyword_results"],
                rrf_results=retrieval["rrf_results"],
                reranked_results=retrieval["reranked_results"],
            )
        return response


def _parse_rag_response(parsed: dict[str, Any], fallback_sources: list[SourceRef]) -> ContractQueryResponse:
    facts_raw = parsed.get("facts") or []
    facts: list[FactWithSource] = []
    for item in facts_raw:
        if not isinstance(item, dict):
            continue
        source = item.get("source") if isinstance(item.get("source"), dict) else {}
        fact_text = item.get("fact")
        if not fact_text:
            continue
        facts.append(
            FactWithSource(
                fact=str(fact_text),
                source=SourceRef(
                    document_name=source.get("document_name"),
                    contract_id=source.get("contract_id"),
                    clause_number=source.get("clause_number"),
                    clause_type=source.get("clause_type"),
                    page=source.get("page"),
                    text=source.get("text"),
                ),
            )
        )

    sources_raw = parsed.get("sources") or []
    sources: list[SourceRef] = []
    for item in sources_raw:
        if isinstance(item, dict):
            sources.append(SourceRef(**{k: item.get(k) for k in SourceRef.model_fields}))
    if not sources:
        sources = fallback_sources

    confidence = parsed.get("confidence")
    try:
        confidence_value = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence_value = None

    analysis = parsed.get("analysis")
    if isinstance(analysis, str):
        analysis = {"note": analysis}
    elif analysis is not None and not isinstance(analysis, dict):
        analysis = {"value": analysis}

    actions = parsed.get("actions") or []
    if not isinstance(actions, list):
        actions = []
    actions = [item if isinstance(item, dict) else {"action": str(item)} for item in actions]

    risks = parsed.get("risks") or []
    if not isinstance(risks, list):
        risks = []
    risks = [item if isinstance(item, dict) else {"risk": str(item)} for item in risks]

    return ContractQueryResponse(
        answer=parsed.get("answer") if parsed.get("answer") is None or isinstance(parsed.get("answer"), str) else str(parsed.get("answer")),
        facts=facts,
        actions=actions,
        risks=risks,
        sources=sources,
        confidence=confidence_value,
        analysis=analysis,
        insufficient_information=bool(parsed.get("insufficient_information")),
    )
