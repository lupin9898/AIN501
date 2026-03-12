"""HyDE RAG Chatbot — Streamlit UI with file upload and streaming.

Run:
    cd hyde_rag
    streamlit run ui.py
"""

from __future__ import annotations

import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings
from graph.nodes import (
    assemble_context_node,
    embed_hypothetical_node,
    generate_hypothetical_doc_node,
    retrieve_documents_node,
    validate_query_node,
)
from ingestion.ingest import run as run_ingestion
from prompts.answer_prompt import ANSWER_PROMPT
from retrieval.qdrant_client import QdrantRetriever

# ── Page config ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HyDE RAG Chatbot",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    /* Overall background */
    .stApp { background-color: #212121; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #171717;
        border-right: 1px solid #2f2f2f;
    }

    /* Chat messages — user bubble */
    [data-testid="stChatMessageContent"] {
        font-size: 0.97rem;
        line-height: 1.65;
    }

    /* Tighten main chat column */
    .main .block-container {
        max-width: 780px;
        padding-top: 1rem;
        padding-bottom: 5rem;
    }

    /* Chat input bar — sticky bottom feel */
    [data-testid="stChatInput"] textarea {
        background: #2f2f2f !important;
        border: 1px solid #3f3f3f !important;
        border-radius: 12px !important;
        color: #ececec !important;
        font-size: 0.97rem;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #555 !important;
        box-shadow: none !important;
    }

    /* Source card */
    .source-card {
        background: #2a2a2a;
        border: 1px solid #3a3a3a;
        border-radius: 8px;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.4rem;
        font-size: 0.83rem;
    }

    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state init ──────────────────────────────────────────────────


def _init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
    if "last_hyde_doc" not in st.session_state:
        st.session_state.last_hyde_doc = ""


_init_session()


# ── Helper: get Qdrant collection stats ─────────────────────────────────


def _get_collection_stats() -> dict | None:
    """Return collection point count, or None if Qdrant is unreachable."""
    try:
        retriever = QdrantRetriever()
        info = retriever.client.get_collection(settings.COLLECTION_NAME)
        return {"points": info.points_count, "status": info.status.value}
    except Exception:
        return None


# ── Helper: stream direct chat (no RAG) ─────────────────────────────────


def _stream_direct(query: str, history: list[dict]):
    """Stream a plain Gemini response when no knowledge base exists yet."""
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        temperature=0.7,
        google_api_key=settings.GEMINI_API_KEY,
        streaming=True,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{query}"),
        ]
    )
    chat_history_messages = []
    recent = history[-20:] if len(history) > 20 else history
    for turn in recent:
        if turn["role"] == "user":
            chat_history_messages.append(HumanMessage(content=turn["content"]))
        else:
            chat_history_messages.append(AIMessage(content=turn["content"]))

    for chunk in (prompt | llm).stream(
        {"query": query, "chat_history": chat_history_messages}
    ):
        yield chunk.content


# ── Helper: stream the final answer ─────────────────────────────────────


def _stream_answer(query: str, context: str, history: list[dict]):
    """Stream tokens from the LLM for the final answer generation.

    Uses the REAL retrieved context (not the hypothetical doc) to ground
    the response in actual knowledge-base data.
    """
    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        temperature=0,
        google_api_key=settings.GEMINI_API_KEY,
        streaming=True,
    )

    # Build chat history messages (last 10 turns)
    chat_history_messages = []
    recent = history[-20:] if len(history) > 20 else history
    for turn in recent:
        if turn["role"] == "user":
            chat_history_messages.append(HumanMessage(content=turn["content"]))
        else:
            chat_history_messages.append(AIMessage(content=turn["content"]))

    chain = ANSWER_PROMPT | llm
    for chunk in chain.stream(
        {"context": context, "query": query, "chat_history": chat_history_messages}
    ):
        yield chunk.content


# ── Helper: run HyDE pipeline steps 1-5 (silent) ────────────────────────


def _run_hyde_pipeline(query: str):
    """Run the HyDE retrieval pipeline silently and return the assembled state."""
    state = {
        "query": query,
        "hypothetical_doc": "",
        "hypothetical_vector": [],
        "retrieved_docs": [],
        "context": "",
        "answer": "",
        "conversation_history": [],
        "error": None,
    }

    for node_fn in [
        validate_query_node,
        generate_hypothetical_doc_node,
        embed_hypothetical_node,
        retrieve_documents_node,
        assemble_context_node,
    ]:
        state.update(node_fn(state))
        if state.get("error"):
            return state

    return state


# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔍 HyDE RAG Chatbot")
    st.caption("Hypothetical Document Embedding")
    st.divider()

    # ── New conversation ────────────────────────────────────────────
    if st.button("🗨️  New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.last_sources = []
        st.session_state.last_hyde_doc = ""
        st.rerun()

    st.divider()

    # ── File upload & ingestion ─────────────────────────────────────
    st.markdown("### 📁 Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload documents to index",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Supported: PDF, TXT, Markdown",
    )

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} file(s) selected")

        if st.button("⬆️  Ingest into RAG", use_container_width=True, type="primary"):
            progress_area = st.empty()
            status_msgs: list[str] = []

            def _ui_progress(msg: str):
                status_msgs.append(msg)
                progress_area.info("\n".join(status_msgs))

            total_stats = {"docs_loaded": 0, "chunks_created": 0, "chunks_upserted": 0}

            for uf in uploaded_files:
                # Save uploaded file to a temp path
                suffix = Path(uf.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uf.getbuffer())
                    tmp_path = Path(tmp.name)

                _ui_progress(f"--- Processing: {uf.name} ---")
                try:
                    stats = run_ingestion(source=tmp_path, on_progress=_ui_progress)
                    for k in total_stats:
                        total_stats[k] += stats.get(k, 0)
                except Exception as e:
                    _ui_progress(f"Error: {e}")

            progress_area.empty()
            st.success(
                f"Ingested {total_stats['docs_loaded']} doc(s) → "
                f"{total_stats['chunks_upserted']} chunks indexed"
            )

    st.divider()

    # ── Collection stats ────────────────────────────────────────────
    st.markdown("### 📊 Collection Info")
    stats = _get_collection_stats()
    if stats:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Vectors", f"{stats['points']:,}")
        with col2:
            st.metric("Status", stats["status"].upper())
        st.caption(f"Collection: `{settings.COLLECTION_NAME}`")
    else:
        st.warning("Qdrant not reachable")
        st.caption(f"URL: `{settings.QDRANT_URL}`")

    st.divider()

    # ── Last query details ──────────────────────────────────────────
    if st.session_state.last_hyde_doc:
        with st.expander("🧠 Last HyDE Document", expanded=False):
            st.markdown(
                f"<div style='font-size:0.82rem; color:#c9d1d9;'>"
                f"{st.session_state.last_hyde_doc}</div>",
                unsafe_allow_html=True,
            )

    if st.session_state.last_sources:
        with st.expander("📚 Last Retrieved Sources", expanded=False):
            for src in st.session_state.last_sources:
                score_pct = src["score"] * 100
                st.markdown(
                    f"<div class='source-card'>"
                    f"📄 {src['source']}"
                    f"{'  (p.' + str(src['page']) + ')' if src.get('page') else ''}"
                    f" — <span class='score'>{score_pct:.1f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

    st.divider()
    st.caption(f"Model: `{settings.LLM_MODEL}`")
    st.caption(f"Embeddings: `{settings.EMBEDDING_MODEL}`")


# ═══════════════════════════════════════════════════════════════════════
#  MAIN CHAT AREA
# ═══════════════════════════════════════════════════════════════════════

# Empty welcome state
if not st.session_state.messages:
    st.markdown(
        """
        <div style="text-align:center; padding: 6rem 1rem 2rem;">
            <p style="font-size:2rem; margin-bottom:0.5rem;">🔍</p>
            <h2 style="font-weight:600; margin-bottom:0.4rem;">HyDE RAG Chatbot</h2>
            <p style="color:#8b949e; font-size:0.95rem;">
                Hỏi đáp từ knowledge base của bạn.<br/>
                Upload tài liệu ở sidebar để bắt đầu.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources inline for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} source(s) used", expanded=False):
                for src in msg["sources"]:
                    score_pct = src["score"] * 100
                    label = src["source"]
                    if src.get("page"):
                        label += f" (p.{src['page']})"
                    st.caption(f"📄 {label} — **{score_pct:.1f}%** relevance")

# ── Chat input ──────────────────────────────────────────────────────────

if prompt := st.chat_input("Nhập câu hỏi ..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Check KB before running the full pipeline
        kb_stats = _get_collection_stats()
        has_kb = kb_stats and kb_stats.get("points", 0) > 0

        if not has_kb:
            # Stream directly without running HyDE pipeline
            full_answer = st.write_stream(
                _stream_direct(prompt, st.session_state.conversation_history)
            )
            sources = []
        else:
            # Run HyDE pipeline silently, then stream
            with st.spinner(""):
                state = _run_hyde_pipeline(prompt)

            error_msg = state.get("error")
            no_kb = error_msg and (
                "chưa có dữ liệu" in error_msg or "doesn't exist" in error_msg
            )

            if error_msg and not no_kb:
                full_answer = error_msg
                st.markdown(full_answer)
                sources = []
            else:
                st.session_state.last_hyde_doc = state.get("hypothetical_doc", "")

                docs = state.get("retrieved_docs", [])
                sources = [
                    {
                        "source": d.metadata.get("source", "unknown"),
                        "page": d.metadata.get("page", ""),
                        "score": d.metadata.get("score", 0),
                    }
                    for d in docs
                ]
                st.session_state.last_sources = sources

                context = state.get("context", "")
                if not context:
                    full_answer = st.write_stream(
                        _stream_direct(prompt, st.session_state.conversation_history)
                    )
                else:
                    full_answer = st.write_stream(
                        _stream_answer(
                            query=prompt,
                            context=context,
                            history=st.session_state.conversation_history,
                        )
                    )
                    if sources:
                        with st.expander(f"📚 {len(sources)} nguồn", expanded=False):
                            for src in sources:
                                label = src["source"]
                                if src.get("page"):
                                    label += f" (tr.{src['page']})"
                                st.caption(f"📄 {label} — **{src['score']*100:.1f}%**")

        st.session_state.messages.append(
            {"role": "assistant", "content": full_answer, "sources": sources}
        )
        st.session_state.conversation_history.append({"role": "user", "content": prompt})
        st.session_state.conversation_history.append({"role": "assistant", "content": full_answer})
