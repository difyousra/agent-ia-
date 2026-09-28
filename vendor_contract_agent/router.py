from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .config import AgentConfig
from .normalize import extract_json_payload

logger = logging.getLogger(__name__)


class Route(str, Enum):
    CONTRACT_ONLY = "CONTRACT_ONLY"
    LEGAL_ONLY = "LEGAL_ONLY"
    CONTRACT_AND_LEGAL = "CONTRACT_AND_LEGAL"
    NO_RETRIEVAL = "NO_RETRIEVAL"


ROUTE_TOOLS: dict[Route, list[str]] = {
    Route.CONTRACT_ONLY: ["contract_search"],
    Route.LEGAL_ONLY: ["legal_search"],
    Route.CONTRACT_AND_LEGAL: ["contract_search", "legal_search"],
    Route.NO_RETRIEVAL: ["no_retrieval"],
}


@dataclass(frozen=True)
class RoutingDecision:
    route: Route
    tools_used: list[str]
    reason: str = ""

    def to_debug_dict(self) -> dict[str, Any]:
        return {
            "route": self.route.value,
            "tools_used": list(self.tools_used),
            "reason": self.reason or None,
        }


_COMPARISON_PATTERNS = [
    r"\bcompl(y|ies|iance)\b",
    r"\bconform(e|es|ity)?\b",
    r"\bcompare\b",
    r"\bconsistent with\b",
    r"\bin accordance with\b",
    r"\blegal framework\b.*\bcontract\b",
    r"\bcontract\b.*\blegal framework\b",
    r"\bcontract\b.*\b(law|legislation|regulation)\b",
    r"\b(law|legislation|regulation)\b.*\bcontract\b",
    r"\bclause\b.*\b(law|algerian|algeria|juridique)\b",
    r"\b(conforme|conformité|comparer|comparaison)\b",
]

_LEGAL_PATTERNS = [
    r"\balgerian law\b",
    r"\blaw of algeria\b",
    r"\balgeria\b.*\b(law|legal|legislation|regulation|framework)\b",
    r"\b(law|legal|legislation|regulation|jurisprudence)\b.*\balgeria\b",
    r"\blegal (framework|requirements?|obligations?|provisions?)\b",
    r"\bwhat does .+ law say\b",
    r"\bjurisdiction\b",
    r"\bréglementation\b",
    r"\bdroit algérien\b",
    r"\bcadre juridique\b",
    r"\bexigences? légales?\b",
]

_CONTRACT_PATTERNS = [
    r"\bcontract\b",
    r"\bctr[-_]?\d+\b",
    r"\bvendor'?s?\b",
    r"\bsupplier'?s?\b",
    r"\bpayment terms?\b",
    r"\bcontract value\b",
    r"\bexpir(e|es|ation|y)\b",
    r"\btermination\b",
    r"\bnotice period\b",
    r"\bsla\b",
    r"\bpenalt(y|ies)\b",
    r"\binsurance\b",
    r"\bdeliverables?\b",
    r"\bobligations?\b",
    r"\bclause\b",
    r"\brenewal\b",
    r"\bwarrant(y|ies)\b",
    r"\bcontrat\b",
    r"\bfournisseur\b",
]

_NO_RETRIEVAL_PATTERNS = [
    r"^\s*what is an?\s+[\w/-]+\??\s*$",
    r"^\s*what does .+ mean\??\s*$",
    r"^\s*explain\b",
    r"^\s*define\b",
    r"difference between",
    r"^\s*qu['’]est[- ]ce qu(un|une)\b",
    r"^\s*explique\b",
    r"^\s*défini[rt]\b",
]


def _compile(patterns: list[str]) -> list[re.Pattern[str]]:
    return [re.compile(p, re.IGNORECASE) for p in patterns]


_COMPARISON_RE = _compile(_COMPARISON_PATTERNS)
_LEGAL_RE = _compile(_LEGAL_PATTERNS)
_CONTRACT_RE = _compile(_CONTRACT_PATTERNS)
_NO_RETRIEVAL_RE = _compile(_NO_RETRIEVAL_PATTERNS)


def _matches(patterns: list[re.Pattern[str]], text: str) -> bool:
    return any(p.search(text) for p in patterns)


def classify_route_rules(query: str) -> RoutingDecision:
    """Deterministic intent classifier (local, no paid API)."""
    text = (query or "").strip()
    if not text:
        return RoutingDecision(Route.NO_RETRIEVAL, ROUTE_TOOLS[Route.NO_RETRIEVAL], "empty_query")

    has_comparison = _matches(_COMPARISON_RE, text)
    has_legal = _matches(_LEGAL_RE, text)
    has_contract = _matches(_CONTRACT_RE, text)
    looks_definitional = _matches(_NO_RETRIEVAL_RE, text)
    has_contract_anchor = bool(
        re.search(
            r"\bctr[-_]?\d+\b|\bin (this |our |the )?contract\b|\bcontract (value|terms?)\b|\bclause\b",
            text,
            re.I,
        )
    )

    # 1) Explicit compliance / comparison against law
    if has_comparison and (has_legal or has_contract or has_contract_anchor):
        route = Route.CONTRACT_AND_LEGAL
        reason = "comparison_contract_vs_law"
    # 2) External legal dependency (may mention topics like termination)
    elif has_legal and not has_comparison:
        route = Route.LEGAL_ONLY
        reason = "external_legal_dependency"
    # 3) Pure definitions without a concrete contract reference
    elif looks_definitional and not has_contract_anchor:
        route = Route.NO_RETRIEVAL
        reason = "general_definition"
    # 4) Internal contract dependency
    elif has_contract:
        route = Route.CONTRACT_ONLY
        reason = "internal_contract_dependency"
    elif looks_definitional:
        route = Route.NO_RETRIEVAL
        reason = "general_definition"
    else:
        route = Route.NO_RETRIEVAL
        reason = "no_retrieval_needed"

    return RoutingDecision(route=route, tools_used=ROUTE_TOOLS[route], reason=reason)


def classify_route(
    query: str,
    *,
    config: AgentConfig | None = None,
    llm=None,
) -> RoutingDecision:
    cfg = config or AgentConfig.from_env()
    if cfg.router_backend == "llm" and llm is not None:
        try:
            return _classify_route_llm(query, llm=llm)
        except Exception:
            logger.exception("LLM routing failed; falling back to rules")
    return classify_route_rules(query)


def _classify_route_llm(query: str, *, llm) -> RoutingDecision:
    system = (
        "You are a routing classifier for a vendor contract intelligence agent. "
        "Return ONLY JSON: {\"route\": \"CONTRACT_ONLY\"|\"LEGAL_ONLY\"|\"CONTRACT_AND_LEGAL\"|\"NO_RETRIEVAL\"}. "
        "Do not explain."
    )
    user = f"Classify this question:\n{query}"
    raw = llm.complete(system, user)
    parsed = extract_json_payload(raw)
    route_raw = str(parsed.get("route") or "").strip().upper()
    route = Route(route_raw)
    return RoutingDecision(route=route, tools_used=ROUTE_TOOLS[route], reason="llm")
