# Vendor Contract Intelligence Agent + Chroma RAG

Single-agent vendor contract analysis with **intelligent routing** and a **ChromaDB** hybrid RAG layer.  
The **Atticus Open Contract Dataset (AOK)** is ingested as the contract knowledge base.

## Architecture

```
User Question
  → Single Agent
  → Route: CONTRACT_ONLY | LEGAL_ONLY | CONTRACT_AND_LEGAL | NO_RETRIEVAL
  → Tools (only when needed):
       contract_search (Chroma + BGE-M3 + hybrid/RRF + BGE-reranker)
       legal_search   (external Legal API adapter)
       no_retrieval
  → Separated context [INTERNAL CONTRACT] / [EXTERNAL LEGAL]
  → LLM → structured JSON
```

Vector backend: **ChromaDB** only (no PostgreSQL / pgvector).

## Installation

```powershell
cd C:\Users\dell\Desktop\agen_contract
.\.venv\Scripts\Activate.ps1
pip install -e .
pip install -r requirements.txt
```

Optional local embedding/reranker models:

```powershell
pip install FlagEmbedding sentence-transformers torch
```

Offline / fast mode (recommended for full Atticus ingest on CPU):

```
VECTOR_BACKEND=chroma
CHROMA_PATH=./chroma_db
EMBEDDING_BACKEND=hash
RERANKER_BACKEND=lexical
```

## Chroma setup

No Docker/PostgreSQL required.

Chroma persists under `CHROMA_PATH` (default `./chroma_db`).

```powershell
# check
python -c "from vendor_contract_agent.rag import create_store, RAGConfig; s=create_store(RAGConfig.from_env()); print(s.count())"
```

## Ingest Atticus dataset → RAG (PDF only)

Download + ingest **full-contract PDFs** into Chroma (CSV annotations are excluded):

```powershell
$env:EMBEDDING_BACKEND="hash"
$env:RERANKER_BACKEND="lexical"
$env:VECTOR_BACKEND="chroma"
$env:CHROMA_PATH="$PWD\chroma_db"
python scripts/ingest_atticus_chroma.py --reset --pdf-only
```

Debug with a small subset:

```powershell
python scripts/ingest_atticus_chroma.py --reset --pdf-only --limit 5
```

Optional (not used by default): CSV annotations with `--csv-only`.

## Run API

```powershell
python -m vendor_contract_agent --api
# or
uvicorn vendor_contract_agent.api:app --reload --port 8000
```

Query:

```powershell
curl -X POST http://localhost:8000/api/contracts/query -H "Content-Type: application/json" -d "{\"query\":\"What is the termination notice period?\",\"debug\":true}"
```

## Testing

```powershell
$env:EMBEDDING_BACKEND="hash"
$env:RERANKER_BACKEND="lexical"
$env:VECTOR_BACKEND="memory"
pytest tests -q
```

## Notes

- Existing `analyze_file` full-document JSON extraction is unchanged.
- RAG Q&A uses `answer_query` / `/api/contracts/query`.
- Duplicate ingest is idempotent via `(contract_id, content_hash)`.
