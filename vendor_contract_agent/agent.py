from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import AgentConfig
from .context.builder import build_routed_context, sources_from_payloads
from .ingest import extract_contract_text
from .llm import LLMClient
from .normalize import extract_json_payload, normalize_list_items, normalize_result
from .prompts import (
    INTELLIGENCE_SYSTEM_PROMPT,
    INTELLIGENCE_USER_PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from .rag.pipeline import RAGPipeline
from .router import Route, RoutingDecision, classify_route
from .schemas.response import AgentQueryResponse, RoutingDebug, TraceableSource
from .tools import contract_search, legal_search, no_retrieval

logger = logging.getLogger(__name__)


class VendorContractAgent:
    """Single agent: full-contract analysis + routed Q&A (RAG / legal / both / none)."""

    def __init__(
        self,
        llm: LLMClient | None = None,
        rag: RAGPipeline | None = None,
        config: AgentConfig | None = None,
        legal_provider=None,
    ) -> None:
        self.config = config or AgentConfig.from_env()
        self.llm = llm or LLMClient()
        self.rag = rag
        self.legal_provider = legal_provider

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
        jurisdiction: str | None = None,
        debug: bool | None = None,
        top_k: int | None = None,
        top_n: int | None = None,
    ) -> AgentQueryResponse:
        """Route the question, call only required tools, then produce structured JSON."""
        text = (query or "").strip()
        debug_mode = self.config.debug_agent_routing if debug is None else bool(debug)
        jur = (jurisdiction or self.config.default_jurisdiction or "DZ").upper()
        k = top_n or top_k or 5

        if not text:
            return AgentQueryResponse(
                answer=None,
                route=Route.NO_RETRIEVAL.value,
                insufficient_information=True,
                confidence=0.0,
                missing_information=["Empty query"],
            )

        decision = classify_route(text, config=self.config, llm=self.llm)
        contract_payload: dict[str, Any] | None = None
        legal_payload: dict[str, Any] | None = None

        try:
            if decision.route == Route.CONTRACT_ONLY:
                contract_payload = contract_search(
                    text,
                    contract_id=contract_id,
                    top_k=k,
                    rag=self.rag,
                )
            elif decision.route == Route.LEGAL_ONLY:
                legal_payload = legal_search(
                    text,
                    jurisdiction=jur,
                    top_k=k,
                    config=self.config,
                    provider=self.legal_provider,
                )
            elif decision.route == Route.CONTRACT_AND_LEGAL:
                contract_payload = contract_search(
                    text,
                    contract_id=contract_id,
                    top_k=k,
                    rag=self.rag,
                )
                legal_payload = legal_search(
                    text,
                    jurisdiction=jur,
                    top_k=k,
                    config=self.config,
                    provider=self.legal_provider,
                )
            else:
                no_retrieval()
        except Exception as exc:
            logger.exception("Tool execution failed")
            return _error_response(
                decision,
                missing=[f"Tool execution failed: {exc}"],
                debug_mode=debug_mode,
            )

        # Graceful degradation when retrieval backends fail
        contract_ok = contract_payload is None or contract_payload.get("available", True)
        legal_ok = legal_payload is None or legal_payload.get("available", True)

        if decision.route == Route.CONTRACT_ONLY and not contract_ok:
            return _error_response(
                decision,
                missing=[contract_payload.get("error") if contract_payload else "contract_rag_unavailable"],
                debug_mode=debug_mode,
                answer="Insufficient information.",
            )

        if decision.route == Route.LEGAL_ONLY and not legal_ok:
            return _error_response(
                decision,
                missing=[legal_payload.get("error") if legal_payload else "legal_api_unavailable"],
                debug_mode=debug_mode,
                answer="Insufficient information.",
            )

        if decision.route == Route.CONTRACT_AND_LEGAL and not contract_ok and not legal_ok:
            return _error_response(
                decision,
                missing=["Required retrieval sources are unavailable."],
                debug_mode=debug_mode,
                answer="Insufficient information.",
            )

        context = build_routed_context(
            contract_payload=contract_payload,
            legal_payload=legal_payload,
        )
        if decision.route == Route.NO_RETRIEVAL:
            context = (
                "NO RETRIEVAL SELECTED.\n"
                "Answer with general knowledge only. "
                "Do not invent contract clauses, legal citations, or URLs."
            )

        fallback_sources = sources_from_payloads(
            contract_payload=contract_payload,
            legal_payload=legal_payload,
        )
        missing_retrieval: list[str] = []
        if decision.route == Route.CONTRACT_AND_LEGAL and not legal_ok:
            missing_retrieval.append(
                f"Legal verification unavailable: {(legal_payload or {}).get('error')}"
            )
        if decision.route == Route.CONTRACT_AND_LEGAL and not contract_ok:
            missing_retrieval.append(
                f"Contract retrieval unavailable: {(contract_payload or {}).get('error')}"
            )

        user_prompt = (
            INTELLIGENCE_USER_PROMPT_TEMPLATE.replace("<<<USER_QUERY>>>", text)
            .replace("<<<ROUTE>>>", decision.route.value)
            .replace("<<<RETRIEVED_CONTEXT>>>", context)
        )
        try:
            raw = self.llm.complete(INTELLIGENCE_SYSTEM_PROMPT, user_prompt)
            parsed = extract_json_payload(raw)
        except Exception as exc:
            logger.exception("LLM JSON generation failed")
            return _error_response(
                decision,
                missing=[f"invalid_llm_json: {exc}"],
                sources=fallback_sources,
                debug_mode=debug_mode,
            )

        response = _parse_intelligence_response(
            parsed,
            decision=decision,
            fallback_sources=fallback_sources,
            extra_missing=missing_retrieval,
        )
        if debug_mode:
            response.debug = RoutingDebug(**decision.to_debug_dict())
            response.route = decision.route.value
        elif response.route is None:
            # Always expose final route in the structured output (not chain-of-thought)
            response.route = decision.route.value
        return response


def _error_response(
    decision: RoutingDecision,
    *,
    missing: list[Any],
    debug_mode: bool,
    answer: str | None = None,
    sources: list[dict[str, Any]] | None = None,
) -> AgentQueryResponse:
    response = AgentQueryResponse(
        answer=answer or "Insufficient information.",
        route=decision.route.value,
        missing_information=[str(m) for m in missing if m],
        sources=[TraceableSource(**s) for s in (sources or []) if isinstance(s, dict)],
        insufficient_information=True,
        confidence=0.0,
    )
    if debug_mode:
        response.debug = RoutingDebug(**decision.to_debug_dict())
    return response


def _parse_intelligence_response(
    parsed: dict[str, Any],
    *,
    decision: RoutingDecision,
    fallback_sources: list[dict[str, Any]],
    extra_missing: list[str],
) -> AgentQueryResponse:
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        return [value]

    sources_raw = _as_list(parsed.get("sources"))
    sources: list[TraceableSource] = []
    for item in sources_raw:
        if isinstance(item, dict):
            # Map legacy RAG source keys if the model returns them
            mapped = {
                "source_type": item.get("source_type"),
                "document": item.get("document") or item.get("document_name"),
                "contract_id": item.get("contract_id"),
                "clause": item.get("clause") or item.get("clause_number"),
                "page": item.get("page"),
                "title": item.get("title"),
                "reference": item.get("reference"),
                "jurisdiction": item.get("jurisdiction"),
                "date": item.get("date"),
                "url": item.get("url"),
                "text": item.get("text"),
            }
            sources.append(TraceableSource(**mapped))
    if not sources:
        sources = [TraceableSource(**s) for s in fallback_sources]

    confidence = parsed.get("confidence")
    try:
        confidence_value = float(confidence) if confidence is not None else None
    except (TypeError, ValueError):
        confidence_value = None

    answer = parsed.get("answer")
    if answer is not None and not isinstance(answer, str):
        answer = str(answer)

    missing = [str(x) for x in _as_list(parsed.get("missing_information"))]
    for item in extra_missing:
        if item and item not in missing:
            missing.append(item)

    contract_facts = _as_list(parsed.get("contract_facts"))
    # Backward-compatible "facts" mirror for older UI
    facts = contract_facts or _as_list(parsed.get("facts"))

    route = parsed.get("route") or decision.route.value
    if isinstance(route, str):
        route = route.strip().upper()

    analysis = parsed.get("analysis")
    if isinstance(analysis, str):
        analysis = {"note": analysis}
    elif analysis is not None and not isinstance(analysis, dict):
        analysis = {"value": analysis}

    return AgentQueryResponse(
        answer=answer,
        route=route,
        contract_facts=contract_facts,
        legal_facts=_as_list(parsed.get("legal_facts")),
        comparison=_as_list(parsed.get("comparison")),
        obligations=_as_list(parsed.get("obligations")),
        actions=_as_list(parsed.get("actions")),
        risks=_as_list(parsed.get("risks")),
        missing_information=missing,
        sources=sources,
        confidence=confidence_value,
        insufficient_information=bool(parsed.get("insufficient_information")),
        analysis=analysis,
        facts=facts,
    )
