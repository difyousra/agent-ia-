from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path

import pytest

from vendor_contract_agent.agent import VendorContractAgent
from vendor_contract_agent.rag.clause_splitter import split_into_clauses
from vendor_contract_agent.rag.config import RAGConfig
from vendor_contract_agent.rag.context import build_context
from vendor_contract_agent.rag.db import InMemoryChunkStore
from vendor_contract_agent.rag.embeddings import EmbeddingService
from vendor_contract_agent.rag.hybrid import hybrid_search
from vendor_contract_agent.rag.ingestion import DocumentIngestionPipeline
from vendor_contract_agent.rag.keyword import keyword_search
from vendor_contract_agent.rag.pipeline import RAGPipeline
from vendor_contract_agent.rag.rerank import RerankerService
from vendor_contract_agent.rag.rrf import reciprocal_rank_fusion
from vendor_contract_agent.rag.schemas import RetrievedChunk
from vendor_contract_agent.rag.semantic import semantic_search


SAMPLE_CONTRACT = """
VENDOR SERVICES AGREEMENT
Contract No.: VSA-2026-047

1. PARTIES
This Agreement is entered into between Algeria Telecom Services S.p.A. ("Customer")
and Atlas Digital Solutions SARL ("Vendor").

3. TERM
The Agreement shall commence on 1 October 2026 and shall remain in force until 30 September 2027.

7. SERVICE LEVEL AGREEMENT
The Vendor shall provide support 24 hours per day, 7 days per week.
Failure to meet the Critical incident restoration target may result in a service credit of 2% of the monthly service fee per incident, capped at 10% of the monthly service fee.

16. INSURANCE
During the term, the Vendor shall maintain professional liability insurance with minimum coverage of 20,000,000 DZD.

15. INTELLECTUAL PROPERTY
Customer-specific reports, documentation and deliverables created and paid for under this Agreement shall belong to the Customer.

22. TERMINATION
The Customer may terminate the Agreement for convenience with 60 days' written notice.
Either Party may terminate the Agreement for material breach if the breach is not cured within 30 days after written notice.

4. CONTRACT VALUE AND PRICING
The annual contract value is 12,000,000 DZD excluding VAT.
"""


@pytest.fixture
def rag_stack():
    config = RAGConfig(
        database_url="memory://",
        embedding_backend="hash",
        reranker_backend="lexical",
        semantic_top_k=10,
        keyword_top_k=10,
        hybrid_top_k=20,
        rerank_top_n=5,
    )
    store = InMemoryChunkStore()
    embedder = EmbeddingService(config)
    reranker = RerankerService(config)
    ingestion = DocumentIngestionPipeline(store=store, embedder=embedder, config=config)
    pipeline = RAGPipeline(store=store, embedder=embedder, reranker=reranker, config=config)
    return SimpleNamespace(
        config=config,
        store=store,
        embedder=embedder,
        reranker=reranker,
        ingestion=ingestion,
        pipeline=pipeline,
    )


def test_clause_splitter_prefers_numbered_clauses():
    chunks = split_into_clauses(SAMPLE_CONTRACT)
    assert len(chunks) >= 4
    numbers = {c.clause_number for c in chunks if c.clause_number}
    assert "22" in numbers or "7" in numbers


def test_document_ingestion_and_idempotency(rag_stack):
    first = rag_stack.ingestion.ingest_text(
        SAMPLE_CONTRACT,
        contract_id="CTR-001",
        document_name="Vendor_Agreement.pdf",
    )
    assert first["chunks_written"] > 0
    count_after_first = rag_stack.store.count("CTR-001")

    second = rag_stack.ingestion.ingest_text(
        SAMPLE_CONTRACT,
        contract_id="CTR-001",
        document_name="Vendor_Agreement.pdf",
        replace_existing=True,
    )
    assert second["chunks_written"] == first["chunks_written"]
    assert rag_stack.store.count("CTR-001") == count_after_first


def test_semantic_search(rag_stack):
    rag_stack.ingestion.ingest_text(
        SAMPLE_CONTRACT,
        contract_id="CTR-001",
        document_name="Vendor_Agreement.pdf",
    )
    results = semantic_search(
        "termination notice period",
        rag_stack.store,
        rag_stack.embedder,
        top_k=5,
        contract_id="CTR-001",
    )
    assert results
    assert all(r.contract_id == "CTR-001" for r in results)
    assert any("60" in r.content or "terminat" in r.content.lower() for r in results)


def test_keyword_search(rag_stack):
    rag_stack.ingestion.ingest_text(
        SAMPLE_CONTRACT,
        contract_id="CTR-001",
        document_name="Vendor_Agreement.pdf",
    )
    for query in ["60 days", "termination", "12,000,000 DZD", "Clause 7", "service credit"]:
        results = keyword_search(query, rag_stack.store, top_k=5, contract_id="CTR-001")
        assert results, f"No keyword hits for {query!r}"


def test_rrf_ranking():
    a = [
        RetrievedChunk(
            id="1",
            content="a",
            contract_id="c",
            document_name="d",
            score=0.9,
            source="semantic",
        ),
        RetrievedChunk(
            id="2",
            content="b",
            contract_id="c",
            document_name="d",
            score=0.8,
            source="semantic",
        ),
    ]
    b = [
        RetrievedChunk(
            id="2",
            content="b",
            contract_id="c",
            document_name="d",
            score=0.7,
            source="keyword",
        ),
        RetrievedChunk(
            id="3",
            content="c",
            contract_id="c",
            document_name="d",
            score=0.6,
            source="keyword",
        ),
    ]
    fused = reciprocal_rank_fusion([a, b], k=60, top_k=3)
    assert fused[0].id == "2"
    assert fused[0].source == "hybrid"


def test_hybrid_search(rag_stack):
    rag_stack.ingestion.ingest_text(
        SAMPLE_CONTRACT,
        contract_id="CTR-001",
        document_name="Vendor_Agreement.pdf",
    )
    semantic_results, keyword_results, fused = hybrid_search(
        "What penalties apply when the supplier misses the SLA?",
        rag_stack.store,
        rag_stack.embedder,
        top_k=10,
        contract_id="CTR-001",
        config=rag_stack.config,
    )
    assert semantic_results or keyword_results
    assert fused
    assert fused[0].source == "hybrid"


def test_reranking_and_source_preservation(rag_stack):
    rag_stack.ingestion.ingest_text(
        SAMPLE_CONTRACT,
        contract_id="CTR-001",
        document_name="Vendor_Agreement.pdf",
    )
    _, _, fused = hybrid_search(
        "What insurance coverage is required?",
        rag_stack.store,
        rag_stack.embedder,
        contract_id="CTR-001",
        config=rag_stack.config,
    )
    reranked = rag_stack.reranker.rerank("What insurance coverage is required?", fused, top_n=3)
    assert reranked
    assert "document_name" in reranked[0].metadata
    assert "contract_id" in reranked[0].metadata
    context = build_context(reranked)
    assert "[DOCUMENT]" in context
    assert "[CLAUSE]" in context
    assert "[CONTENT]" in context


def test_contract_filtering(rag_stack):
    rag_stack.ingestion.ingest_text(
        SAMPLE_CONTRACT,
        contract_id="CTR-001",
        document_name="Vendor_Agreement.pdf",
    )
    other = SAMPLE_CONTRACT.replace("60 days", "90 days").replace("VSA-2026-047", "VSA-OTHER")
    rag_stack.ingestion.ingest_text(
        other,
        contract_id="CTR-002",
        document_name="Other.pdf",
    )
    results = keyword_search("90 days", rag_stack.store, top_k=5, contract_id="CTR-002")
    assert results
    assert all(r.contract_id == "CTR-002" for r in results)


def test_no_result_queries(rag_stack):
    rag_stack.ingestion.ingest_text(
        SAMPLE_CONTRACT,
        contract_id="CTR-001",
        document_name="Vendor_Agreement.pdf",
    )
    retrieval = rag_stack.pipeline.retrieve(
        "quantum teleportation warranty for Mars colony",
        contract_id="CTR-001",
        top_n=5,
    )
    # May still retrieve weak lexical/semantic neighbors; ensure pipeline stays stable.
    assert "context" in retrieval
    assert "reranked_results" in retrieval


def test_invalid_contract_id(rag_stack):
    with pytest.raises(ValueError, match="Invalid contract_id"):
        rag_stack.pipeline.retrieve("termination", contract_id="DOES-NOT-EXIST")


def test_end_to_end_question_answering(rag_stack, monkeypatch):
    rag_stack.ingestion.ingest_text(
        SAMPLE_CONTRACT,
        contract_id="CTR-001",
        document_name="Vendor_Agreement.pdf",
    )

    class FakeLLM:
        def complete(self, system_prompt: str, user_prompt: str) -> str:
            assert "RETRIEVED CONTRACT CONTEXT" in user_prompt or "retrieved" in system_prompt.lower() or "Retrieved" in user_prompt or "60" in user_prompt or "TERMINATION" in user_prompt
            return """
            {
              "answer": "Termination for convenience requires 60 days written notice.",
              "facts": [
                {
                  "fact": "Termination requires 60 days written notice.",
                  "source": {
                    "document_name": "Vendor_Agreement.pdf",
                    "contract_id": "CTR-001",
                    "clause_number": "22",
                    "clause_type": "Termination",
                    "page": null,
                    "text": "The Customer may terminate the Agreement for convenience with 60 days' written notice."
                  }
                }
              ],
              "actions": [],
              "risks": [],
              "sources": [],
              "confidence": 0.91,
              "insufficient_information": false,
              "analysis": null
            }
            """

    agent = VendorContractAgent(llm=FakeLLM(), rag=rag_stack.pipeline)
    questions = [
        "What is the termination notice period?",
        "What are the vendor's SLA obligations?",
        "What penalties apply when the supplier misses the SLA?",
        "When does the contract expire?",
        "What insurance coverage is required?",
        "Who owns the deliverables?",
    ]
    for question in questions:
        response = agent.answer_query(question, contract_id="CTR-001", debug=True)
        assert response.debug is not None
        assert response.route == "CONTRACT_ONLY"
        assert response.debug.tools_used == ["contract_search"]
        assert response.answer or response.insufficient_information is False or response.facts


def test_empty_query_does_not_crash(rag_stack):
    agent = VendorContractAgent(llm=SimpleNamespace(complete=lambda *a, **k: "{}"), rag=rag_stack.pipeline)
    response = agent.answer_query("   ")
    assert response.insufficient_information is True
