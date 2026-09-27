from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from vendor_contract_agent.agent import VendorContractAgent
from vendor_contract_agent.dataset import download_atticus_dataset
from vendor_contract_agent.ingest import extract_contract_text
from vendor_contract_agent.rag.config import RAGConfig
from vendor_contract_agent.rag.db import create_store
from vendor_contract_agent.rag.embeddings import EmbeddingService
from vendor_contract_agent.rag.pipeline import RAGPipeline
from vendor_contract_agent.rag.rerank import RerankerService

load_dotenv(override=True)

st.set_page_config(
    page_title="Vendor Contract Agent",
    page_icon="📄",
    layout="wide",
)

EXAMPLE_PROMPTS = [
    "What is the termination notice period?",
    "What are the vendor's SLA obligations?",
    "What penalties apply when the supplier misses the SLA?",
    "When does the contract expire?",
    "What insurance coverage is required?",
    "Who owns the deliverables?",
    "Governing Law",
    "Governing Law Pennsylvania",
    "Audit Rights",
    "Expiration Date",
    "Termination for Convenience",
]


@st.cache_resource(show_spinner=False)
def get_agent() -> VendorContractAgent:
    config = RAGConfig.from_env()
    store = create_store(config)
    embedder = EmbeddingService(config)
    reranker = RerankerService(config)
    rag = RAGPipeline(store=store, embedder=embedder, reranker=reranker, config=config)
    return VendorContractAgent(rag=rag)


@st.cache_resource(show_spinner=False)
def get_atticus_pdf_index() -> dict[str, str]:
    """Map document_name / stem -> absolute PDF path."""
    root = download_atticus_dataset()
    pdf_dir = root / "Final Publication" / "full_contracts"
    index: dict[str, str] = {}
    if not pdf_dir.exists():
        return index
    for path in pdf_dir.iterdir():
        if path.suffix.lower() != ".pdf":
            continue
        index[path.name] = str(path)
        index[path.name.lower()] = str(path)
        index[path.stem] = str(path)
        index[path.stem.lower()] = str(path)
    return index


def resolve_pdf_path(document_name: str | None = None, contract_id: str | None = None) -> Path | None:
    index = get_atticus_pdf_index()
    for key in [document_name, contract_id]:
        if not key:
            continue
        if key in index:
            return Path(index[key])
        if key.lower() in index:
            return Path(index[key.lower()])
        # fuzzy: startswith
        for name, path in index.items():
            if name.lower().startswith(str(key).lower()[:40]):
                return Path(path)
    return None


def render_pdf(path: Path | bytes | None, *, height: int = 720, key: str | None = None) -> None:
    """Display a PDF with st.pdf when available, else iframe fallback."""
    if path is None:
        st.warning("PDF introuvable.")
        return

    data: bytes
    label = "document.pdf"
    if isinstance(path, Path):
        if not path.exists():
            st.warning(f"Fichier introuvable : {path}")
            return
        data = path.read_bytes()
        label = path.name
    else:
        data = path

    st.caption(f"Visualisation : **{label}** ({len(data) // 1024} Ko)")
    try:
        st.pdf(data, height=height, key=key)
        return
    except Exception:
        # Fallback: embedded browser PDF viewer
        b64 = base64.b64encode(data).decode("ascii")
        components.html(
            f"""
            <iframe
              src="data:application/pdf;base64,{b64}"
              width="100%"
              height="{height}"
              style="border:1px solid #444;border-radius:8px;"
            ></iframe>
            """,
            height=height + 20,
        )


def _save_upload(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower() or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        return Path(tmp.name)


def _metric_count(items) -> int:
    return len(items) if isinstance(items, list) else 0


st.title("Agent d'analyse de contrat fournisseur")
st.caption(
    "Posez une question (RAG Atticus), analysez un PDF, ou visualisez les contrats sources."
)

mode = st.radio(
    "Mode",
    [
        "Question RAG (prompt utilisateur)",
        "Analyse complète d'un PDF",
        "Visualisation PDF Atticus",
    ],
    horizontal=True,
)

# ---------------------------------------------------------------------------
# MODE 1: RAG Q&A + PDF des sources
# ---------------------------------------------------------------------------
if mode.startswith("Question"):
    st.subheader("Votre prompt")
    st.write(
        "L'agent cherche dans Chroma (Atticus), fusionne sémantique + mots-clés (RRF), "
        "rerank, puis répond uniquement à partir du contexte récupéré."
    )

    agent = get_agent()
    store = agent.rag.store if agent.rag else None
    chunk_count = store.count() if store else 0
    st.info(f"Base RAG Chroma : **{chunk_count}** chunks indexés.")

    chosen = st.selectbox("Exemples de prompts", ["(saisie libre)"] + EXAMPLE_PROMPTS)
    user_prompt = st.text_area(
        "Prompt utilisateur",
        height=120,
        placeholder="Ex: What is the termination notice period?",
        key="prompt_input",
    )
    if chosen != "(saisie libre)" and st.session_state.get("last_example") != chosen:
        st.session_state["last_example"] = chosen
        st.session_state["prompt_input"] = chosen
        st.rerun()

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        contract_id = st.text_input(
            "Filtrer par contract_id (optionnel)",
            value="",
            placeholder="ACCELERATEDTECHNOLOGIESHOLDINGCORP_...",
        )
    with col_b:
        debug = st.checkbox("Debug retrieval", value=False)
    with col_c:
        ask = st.button("Envoyer le prompt", type="primary", use_container_width=True)

    if ask:
        query = (user_prompt or "").strip()
        if not query:
            st.warning("Saisissez un prompt.")
        else:
            with st.spinner("RAG + agent en cours…"):
                try:
                    response = agent.answer_query(
                        query,
                        contract_id=contract_id.strip() or None,
                        debug=debug,
                    )
                    st.session_state["rag_response"] = response.model_dump()
                    st.session_state["last_prompt"] = query
                except Exception as exc:
                    st.error(str(exc))

    rag_response = st.session_state.get("rag_response")
    if rag_response:
        st.success(f"Prompt : {st.session_state.get('last_prompt')}")
        if rag_response.get("insufficient_information"):
            st.warning("Informations insuffisantes dans le contexte récupéré.")
        st.markdown("### Réponse")
        st.write(rag_response.get("answer") or "—")

        m1, m2, m3 = st.columns(3)
        m1.metric(
            "Confidence",
            rag_response.get("confidence") if rag_response.get("confidence") is not None else "—",
        )
        m2.metric("Facts", _metric_count(rag_response.get("facts")))
        m3.metric("Sources", _metric_count(rag_response.get("sources")))

        tabs = st.tabs(["PDF sources", "Facts", "Actions", "Risques", "Sources", "Debug", "JSON"])

        with tabs[0]:
            sources = rag_response.get("sources") or []
            pdf_options: list[str] = []
            for src in sources:
                name = src.get("document_name") or src.get("contract_id")
                if name and name not in pdf_options:
                    pdf_options.append(name)
            # Also collect from facts
            for item in rag_response.get("facts") or []:
                src = item.get("source") or {}
                name = src.get("document_name") or src.get("contract_id")
                if name and name not in pdf_options:
                    pdf_options.append(name)

            if not pdf_options:
                st.info("Aucune source PDF à visualiser pour cette réponse.")
            else:
                selected_pdf = st.selectbox("PDF source à afficher", pdf_options, key="rag_pdf_select")
                pdf_path = resolve_pdf_path(document_name=selected_pdf, contract_id=selected_pdf)
                if pdf_path:
                    st.download_button(
                        "Télécharger ce PDF",
                        data=pdf_path.read_bytes(),
                        file_name=pdf_path.name,
                        mime="application/pdf",
                    )
                    render_pdf(pdf_path, height=780, key=f"rag_pdf_{selected_pdf}")
                else:
                    st.warning("Ce PDF n'a pas été trouvé dans le dataset Atticus local.")

        with tabs[1]:
            facts = rag_response.get("facts") or []
            if not facts:
                st.write("Aucun fact extrait.")
            else:
                for item in facts:
                    source = item.get("source") or {}
                    st.markdown(f"**{item.get('fact')}**")
                    st.caption(
                        f"{source.get('document_name')} | "
                        f"{source.get('clause_type')} | "
                        f"clause {source.get('clause_number')} | "
                        f"page {source.get('page')}"
                    )
                    if source.get("text"):
                        st.code(source.get("text"), language=None)

        with tabs[2]:
            st.dataframe(rag_response.get("actions") or [], use_container_width=True)
        with tabs[3]:
            st.dataframe(rag_response.get("risks") or [], use_container_width=True)
        with tabs[4]:
            st.dataframe(rag_response.get("sources") or [], use_container_width=True)
        with tabs[5]:
            debug_payload = rag_response.get("debug")
            if not debug_payload:
                st.info("Cochez Debug retrieval puis renvoyez le prompt.")
            else:
                st.write(
                    f"semantic={len(debug_payload.get('semantic_results') or [])} | "
                    f"keyword={len(debug_payload.get('keyword_results') or [])} | "
                    f"rrf={len(debug_payload.get('rrf_results') or [])} | "
                    f"rerank={len(debug_payload.get('reranked_results') or [])}"
                )
                st.json(debug_payload)
        with tabs[6]:
            st.json(rag_response)

# ---------------------------------------------------------------------------
# MODE 2: Full PDF analysis + viewer
# ---------------------------------------------------------------------------
elif mode.startswith("Analyse"):
    st.subheader("Analyse PDF complète")
    st.write(
        "Mode classique : le fichier entier est envoyé à l'agent, qui produit le JSON "
        "structuré (parties, SLA, obligations, actions, risques, etc.)."
    )

    uploaded = st.file_uploader(
        "Contrat",
        type=["pdf", "txt", "md", "docx"],
        help="Formats acceptés : PDF, TXT, MD, DOCX",
    )
    col_run, col_sample = st.columns([1, 1])
    with col_run:
        run = st.button("Analyser le contrat", type="primary", disabled=uploaded is None)
    with col_sample:
        use_sample = st.button("Utiliser le contrat d'exemple")

    if use_sample:
        sample = Path(__file__).resolve().parent / "sample_vendor_contract_test.pdf"
        if sample.exists():
            st.session_state["contract_path"] = str(sample)
            st.session_state["display_name"] = sample.name
            st.session_state["pending_analysis"] = True
        else:
            st.error("Le fichier d'exemple est introuvable.")

    if run and uploaded is not None:
        saved = _save_upload(uploaded)
        st.session_state["contract_path"] = str(saved)
        st.session_state["display_name"] = uploaded.name
        st.session_state["pending_analysis"] = True

    contract_path = Path(st.session_state["contract_path"]) if st.session_state.get("contract_path") else None
    display_name = st.session_state.get("display_name")

    if contract_path and contract_path.suffix.lower() == ".pdf":
        with st.expander("Aperçu PDF", expanded=True):
            render_pdf(contract_path, height=640, key="analysis_preview_pdf")

    if contract_path and st.session_state.get("pending_analysis"):
        with st.spinner(f"Analyse de {display_name} en cours…"):
            try:
                preview = extract_contract_text(contract_path)
                result = VendorContractAgent().analyze_file(contract_path)
                st.session_state["result"] = result
                st.session_state["preview"] = preview
                st.session_state["pending_analysis"] = False
            except Exception as exc:
                st.session_state.pop("result", None)
                st.session_state["pending_analysis"] = False
                st.error(str(exc))
                st.stop()

    result = st.session_state.get("result")
    if not result:
        st.info("Chargez un contrat puis cliquez sur Analyser.")
        st.stop()

    contract = result.get("contract") or {}
    parties = result.get("parties") or {}
    vendor = parties.get("vendor") or {}
    customer = parties.get("customer") or {}

    st.success(f"Analyse terminée : {display_name}")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("N° contrat", contract.get("contract_number") or "—")
    m2.metric("Actions", _metric_count(result.get("actions")))
    m3.metric("Risques", _metric_count(result.get("risks")))
    m4.metric("Pénalités", _metric_count(result.get("penalties")))

    left, right = st.columns(2)
    with left:
        st.subheader("Parties")
        st.write(f"**Client :** {customer.get('name') or '—'}")
        st.write(customer.get("address") or "")
        st.write(f"**Fournisseur :** {vendor.get('name') or '—'}")
        st.write(vendor.get("address") or "")
    with right:
        st.subheader("Dates")
        st.write(f"**Début :** {contract.get('start_date') or '—'}")
        st.write(f"**Fin :** {contract.get('end_date') or '—'}")
        st.write(f"**Valeur :** {(result.get('financial') or {}).get('contract_value') or '—'}")

    tabs = st.tabs(["PDF", "Actions", "Risques", "Obligations", "JSON", "Texte extrait"])
    with tabs[0]:
        if contract_path and contract_path.suffix.lower() == ".pdf":
            render_pdf(contract_path, height=780, key="analysis_result_pdf")
        else:
            st.info("Le fichier chargé n'est pas un PDF.")
    with tabs[1]:
        st.dataframe(result.get("actions") or [], use_container_width=True)
    with tabs[2]:
        st.dataframe(result.get("risks") or [], use_container_width=True)
    with tabs[3]:
        obligations = result.get("obligations") or {}
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Fournisseur**")
            for item in obligations.get("vendor") or []:
                st.write(f"- {item}")
        with c2:
            st.markdown("**Client**")
            for item in obligations.get("customer") or []:
                st.write(f"- {item}")
    with tabs[4]:
        payload = json.dumps(result, ensure_ascii=False, indent=2)
        st.download_button(
            "Télécharger outputs.json",
            data=payload.encode("utf-8"),
            file_name="outputs.json",
            mime="application/json",
        )
        st.json(result)
    with tabs[5]:
        st.text_area("Texte du contrat", st.session_state.get("preview") or "", height=320)

# ---------------------------------------------------------------------------
# MODE 3: Browse / visualize Atticus PDFs
# ---------------------------------------------------------------------------
else:
    st.subheader("Visualisation des PDF Atticus")
    st.write("Parcourez les contrats du dataset indexé dans Chroma et ouvrez le PDF original.")

    index = get_atticus_pdf_index()
    # unique real paths
    pdf_paths = sorted({Path(p) for p in index.values() if Path(p).exists()}, key=lambda p: p.name.lower())
    st.info(f"**{len(pdf_paths)}** PDF disponibles dans Atticus.")

    search = st.text_input("Filtrer par nom", placeholder="joint venture, license, co-branding...")
    filtered = [
        p for p in pdf_paths if not search.strip() or search.strip().lower() in p.name.lower()
    ]
    if not filtered:
        st.warning("Aucun PDF ne correspond au filtre.")
        st.stop()

    names = [p.name for p in filtered]
    selected_name = st.selectbox("Contrat PDF", names, index=0)
    selected_path = next(p for p in filtered if p.name == selected_name)

    c1, c2 = st.columns([1, 3])
    with c1:
        st.write(f"**Fichier :** {selected_path.name}")
        st.write(f"**Taille :** {selected_path.stat().st_size // 1024} Ko")
        st.download_button(
            "Télécharger",
            data=selected_path.read_bytes(),
            file_name=selected_path.name,
            mime="application/pdf",
            use_container_width=True,
        )
        if st.button("Analyser ce PDF", use_container_width=True):
            st.session_state["contract_path"] = str(selected_path)
            st.session_state["display_name"] = selected_path.name
            st.session_state["pending_analysis"] = True
            st.info("Passez en mode « Analyse complète d'un PDF » pour voir le résultat.")
    with c2:
        render_pdf(selected_path, height=820, key=f"browse_{selected_name}")
