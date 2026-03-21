"""HyDE RAG Chatbot — Streamlit UI."""

from __future__ import annotations

import html as html_lib
import re
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

# ── Page config ──────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ──────────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
/* ── Base ─────────────────────────────────────────────── */
.stApp { background-color: #1a1a1a; color: #ececec; }

/* ── Sidebar ──────────────────────────────────────────── */
section[data-testid="stSidebar"] {
    background-color: #171717 !important;
    border-right: 1px solid #272727 !important;
}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] div { color: #c9c9c9 !important; }

/* ── Main content width ───────────────────────────────── */
.main .block-container {
    max-width: 700px !important;
    margin: 0 auto !important;
    padding: 1.5rem 1.25rem 7rem !important;
}

/* ── User bubble (right-aligned) ──────────────────────── */
.msg-user {
    display: flex;
    justify-content: flex-end;
    margin: 4px 0 20px;
}
.msg-user-bubble {
    background: #2d2d2d;
    border-radius: 18px 18px 4px 18px;
    padding: 10px 18px;
    max-width: 78%;
    color: #ececec;
    font-size: 0.95rem;
    line-height: 1.65;
    word-break: break-word;
    white-space: pre-wrap;
}

/* ── Assistant text (left-aligned, no bg) ─────────────── */
.msg-assistant {
    margin: 4px 0 6px;
}
.msg-assistant p,
.msg-assistant li,
.msg-assistant span {
    color: #ececec !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
}
.msg-assistant strong { color: #fff !important; }
.msg-assistant code {
    background: #2a2a2a !important;
    color: #e06c75 !important;
    border-radius: 4px !important;
    padding: 1px 6px !important;
    font-size: 0.87rem !important;
}
.msg-assistant pre {
    background: #222 !important;
    border: 1px solid #333 !important;
    border-radius: 10px !important;
    padding: 0.9rem !important;
}
.msg-assistant a { color: #7eb8f7 !important; }

/* Global text colour for assistant streaming placeholder */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] td {
    color: #ececec !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
}

/* ── Chat input ───────────────────────────────────────── */
[data-testid="stChatInput"] {
    background: #252525 !important;
    border: 1.5px solid #363636 !important;
    border-radius: 14px !important;
}
[data-testid="stChatInput"]:focus-within { border-color: #505050 !important; }
[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #ececec !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea::placeholder { color: #555 !important; }

/* ── Expander (sources) ───────────────────────────────── */
[data-testid="stExpander"] {
    background: #202020 !important;
    border: 1px solid #2d2d2d !important;
    border-radius: 10px !important;
    margin-top: 8px !important;
}
[data-testid="stExpander"] summary {
    color: #777 !important;
    font-size: 0.81rem !important;
    padding: 7px 12px !important;
}
[data-testid="stExpander"] summary:hover { color: #aaa !important; }

/* ── Buttons ──────────────────────────────────────────── */
[data-testid="stBaseButton-primary"] {
    background: #3a3a3a !important;
    color: #ececec !important;
    border: none !important;
    border-radius: 8px !important;
}
[data-testid="stBaseButton-primary"]:hover { background: #484848 !important; }
[data-testid="stBaseButton-secondary"] {
    background: transparent !important;
    border: 1px solid #333 !important;
    border-radius: 8px !important;
    color: #bbb !important;
}

/* ── File uploader ────────────────────────────────────── */
[data-testid="stFileUploader"] section {
    background: #1e1e1e !important;
    border: 1.5px dashed #303030 !important;
    border-radius: 10px !important;
}

/* ── Metrics ──────────────────────────────────────────── */
[data-testid="stMetricValue"] { color: #ececec !important; font-weight: 600 !important; }
[data-testid="stMetricLabel"] { color: #666 !important; font-size: 0.78rem !important; }

/* ── Caption ──────────────────────────────────────────── */
.stCaption, [data-testid="stCaptionContainer"] { color: #555 !important; font-size: 0.79rem !important; }

/* ── Alert ────────────────────────────────────────────── */
[data-testid="stAlert"] {
    background: #1a221a !important;
    border: 1px solid #2a3d2a !important;
    color: #8fc98f !important;
    border-radius: 8px !important;
    font-size: 0.82rem !important;
}

/* ── Hide Streamlit chrome ────────────────────────────── */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }

hr { border-color: #272727 !important; margin: 0.5rem 0 !important; }

/* GK badge */
.gk-badge {
    display: inline-block;
    background: #1e1e1e;
    border: 1px solid #2d2d2d;
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.74rem;
    color: #555;
    margin-top: 6px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state ────────────────────────────────────────────────────────


def _init_session() -> None:
    defaults: dict = {
        "messages": [],
        "conversation_history": [],
        "thread_id": str(uuid.uuid4()),
        "last_sources": [],
        "last_hyde_doc": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()

# ── Utilities ────────────────────────────────────────────────────────────


def _get_collection_stats() -> dict | None:
    try:
        r = QdrantRetriever()
        info = r.client.get_collection(settings.COLLECTION_NAME)
        return {"points": info.points_count, "status": info.status.value}
    except Exception:
        return None


def _build_history_messages(history: list[dict]) -> list:
    recent = history[-20:] if len(history) > 20 else history
    return [
        (HumanMessage if t["role"] == "user" else AIMessage)(content=t["content"])
        for t in recent
    ]


def _stream_direct(query: str, history: list[dict]):
    """Stream plain LLM response (no KB context)."""
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
    for chunk in (prompt | llm).stream(
        {"query": query, "chat_history": _build_history_messages(history)}
    ):
        yield chunk.content


def _stream_answer(query: str, context: str, history: list[dict]):
    """Stream RAG answer. Yields (token, used_kb | None).

    LLM prefixes its response with [KB] or [GK] to signal whether it used
    the knowledge-base context or fell back to general knowledge.
    """
    llm = ChatGoogleGenerativeAI(
        model=settings.LLM_MODEL,
        temperature=0,
        google_api_key=settings.GEMINI_API_KEY,
        streaming=True,
    )
    buf = ""
    routing_done = False
    used_kb: bool | None = None

    for chunk in (ANSWER_PROMPT | llm).stream(
        {"context": context, "query": query, "chat_history": _build_history_messages(history)}
    ):
        token = chunk.content
        if not routing_done:
            buf += token
            if len(buf) >= 4:
                routing_done = True
                if buf.startswith("[KB]"):
                    used_kb = True
                    buf = buf[4:].lstrip()
                elif buf.startswith("[GK]"):
                    used_kb = False
                    buf = buf[4:].lstrip()
                else:
                    used_kb = True
                yield buf, used_kb
                buf = ""
        else:
            yield token, used_kb


def _run_hyde_pipeline(query: str) -> dict:
    state: dict = {
        "query": query,
        "hypothetical_doc": "",
        "hypothetical_vector": [],
        "sparse_vector": None,
        "retrieved_docs": [],
        "context": "",
        "answer": "",
        "conversation_history": [],
        "error": None,
    }
    for fn in [
        validate_query_node,
        generate_hypothetical_doc_node,
        embed_hypothetical_node,
        retrieve_documents_node,
        assemble_context_node,
    ]:
        state.update(fn(state))
        if state.get("error"):
            return state
    return state


# ── Intent classifier (zero-latency heuristic) ───────────────────────────

# Patterns that are almost certainly NOT KB-related queries
_CONVERSATIONAL_RE = re.compile(
    r"""^(
        # greetings
        hello|hi+|hey|howdy|hola|sup|yo|
        chào|xin\s+chào|alo|
        good\s+(morning|afternoon|evening|night)|
        # small talk
        how\s+are\s+you|bạn\s+(có\s+)?khỏe|
        bạn\s+là\s+ai|you\s+are\s+who|
        what\s+(is|are)\s+you[r]?|
        # thanks / ack
        thanks?|thank\s+you|cảm\s+ơn|
        ok(ay)?|sure|got\s+it|understood|
        no\s+problem|you['\u2019]?re\s+welcome|
        # farewells
        bye+|goodbye|tạm\s+biệt|see\s+you|
        # affirmations
        yes|no|yep|nope|correct|wrong|
        # emoji-only / single char
        [\U00010000-\U0010ffff\u2600-\u26FF\u2700-\u27BF]+|
        [!?.]+
    )\b""",
    re.VERBOSE | re.IGNORECASE,
)


def _skip_rag(query: str) -> bool:
    """Return True when the query is clearly conversational/off-topic.

    Uses zero-latency regex heuristics so the HyDE pipeline is bypassed
    entirely — no LLM call, no embedding, no Qdrant round-trip.

    Falls back to False (run full pipeline) for ambiguous queries.
    """
    q = query.strip()
    # Very short: 1-3 words — likely chitchat
    if len(q.split()) <= 3 and not q.endswith("?"):
        return True
    # Matches known conversational patterns
    if _CONVERSATIONAL_RE.match(q):
        return True
    return False


def _source_label(src: dict) -> str:
    path = src.get("source", "unknown")
    name = Path(path).name if path != "unknown" else "unknown"
    page = f"  tr.{src['page']}" if src.get("page") else ""
    score = f"  {src['score'] * 100:.0f}%" if src.get("score") else ""
    return f"📄  {name}{page}{score}"


# ── Message render helpers ────────────────────────────────────────────────


def _render_user(text: str) -> None:
    """Right-aligned dark bubble."""
    safe = html_lib.escape(text)
    st.markdown(
        f'<div class="msg-user"><div class="msg-user-bubble">{safe}</div></div>',
        unsafe_allow_html=True,
    )


def _render_assistant(content: str, sources: list | None = None, from_kb: bool = True) -> None:
    """Plain left-aligned text + optional sources expander."""
    st.markdown(content)
    if sources:
        with st.expander(f"📄 {len(sources)} nguồn tham khảo", expanded=False):
            for s in sources:
                st.caption(_source_label(s))
    elif not from_kb:
        st.markdown(
            "<span class='gk-badge'>💬 general knowledge</span>",
            unsafe_allow_html=True,
        )


# ════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        "<p style='font-size:1.05rem;font-weight:600;margin-bottom:0'>✦ RAG Assistant</p>"
        "<p style='font-size:0.74rem;color:#444;margin-top:2px'>HyDE · Hybrid BM25 · Gemini</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    if st.button("＋  New chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = []
        st.session_state.last_sources = []
        st.session_state.last_hyde_doc = ""
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()

    st.divider()
    st.markdown(
        "<p style='font-size:0.83rem;font-weight:600;margin-bottom:4px'>Knowledge Base</p>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "docs",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded:
        st.caption(f"{len(uploaded)} file(s) chờ index")
        if st.button("⬆  Ingest", use_container_width=True, type="primary"):
            log_area = st.empty()
            log_lines: list[str] = []

            def _prog(msg: str) -> None:
                log_lines.append(msg)
                log_area.info("\n".join(log_lines[-5:]))

            totals = {"docs_loaded": 0, "chunks_created": 0, "chunks_upserted": 0}
            for uf in uploaded:
                suffix = Path(uf.name).suffix
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uf.getbuffer())
                    tmp_path = Path(tmp.name)
                _prog(f"→ {uf.name}")
                try:
                    s = run_ingestion(source=tmp_path, on_progress=_prog)
                    for k in totals:
                        totals[k] += s.get(k, 0)
                except Exception as e:
                    _prog(f"⚠ {e}")
            log_area.empty()
            st.success(f"✓  {totals['chunks_upserted']} chunks đã được index")

    st.divider()

    col_stats = _get_collection_stats()
    if col_stats:
        c1, c2 = st.columns(2)
        c1.metric("Chunks", f"{col_stats['points']:,}")
        c2.metric("DB", col_stats["status"].upper()[:2])
        st.caption(f"`{settings.COLLECTION_NAME}`")
    else:
        st.caption("⚠  Qdrant offline")

    st.divider()

    if st.session_state.last_hyde_doc:
        with st.expander("🧠 HyDE document", expanded=False):
            st.markdown(
                f"<div style='font-size:0.78rem;color:#777;white-space:pre-wrap'>"
                f"{html_lib.escape(st.session_state.last_hyde_doc[:600])}…</div>",
                unsafe_allow_html=True,
            )

    if st.session_state.last_sources:
        with st.expander(f"📄 {len(st.session_state.last_sources)} sources", expanded=False):
            for s in st.session_state.last_sources:
                st.caption(_source_label(s))

    st.divider()
    st.caption(f"Model  `{settings.LLM_MODEL}`")
    st.caption(f"Embed  `{settings.EMBEDDING_MODEL.split('/')[-1]}`")

# ════════════════════════════════════════════════════════════════════════
#  MAIN CHAT AREA
# ════════════════════════════════════════════════════════════════════════

if not st.session_state.messages:
    st.markdown(
        """
        <div style="text-align:center;padding:5rem 0 2rem;">
            <div style="font-size:2.6rem;margin-bottom:0.6rem;opacity:.8">✦</div>
            <h2 style="font-weight:600;font-size:1.5rem;color:#ececec;margin:0 0 0.5rem">
                RAG Assistant
            </h2>
            <p style="color:#4a4a4a;font-size:0.92rem;line-height:1.7;max-width:360px;margin:0 auto">
                Hỏi đáp từ tài liệu của bạn.<br>
                Tự động fallback sang general AI khi không có context.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Render chat history (no st.chat_message — no action buttons)
for msg in st.session_state.messages:
    if msg["role"] == "user":
        _render_user(msg["content"])
    else:
        _render_assistant(
            msg["content"],
            sources=msg.get("sources"),
            from_kb=msg.get("from_kb", True),
        )

# ── Chat input ───────────────────────────────────────────────────────────

if prompt := st.chat_input("Nhập câu hỏi …"):
    # Render user bubble immediately
    _render_user(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ── Assistant response ────────────────────────────────────────────
    kb_stats = _get_collection_stats()
    has_kb = bool(kb_stats and kb_stats.get("points", 0) > 0)

    sources: list[dict] = []
    from_kb = False

    # Fast path: skip HyDE pipeline for obvious conversational queries
    if has_kb and _skip_rag(prompt):
        has_kb = False  # treat as no-KB so we stream direct immediately

    if not has_kb:
        # No KB → direct LLM
        answer_placeholder = st.empty()
        parts: list[str] = []
        for token in _stream_direct(prompt, st.session_state.conversation_history):
            parts.append(token)
            answer_placeholder.markdown("".join(parts))
        full_answer = "".join(parts)
        st.markdown(
            "<span class='gk-badge'>💬 general knowledge</span>",
            unsafe_allow_html=True,
        )

    else:
        with st.spinner(""):
            state = _run_hyde_pipeline(prompt)

        error = state.get("error")
        is_missing = error and (
            "chưa có dữ liệu" in error or "doesn't exist" in error
        )

        if error and not is_missing:
            full_answer = f"⚠  {error}"
            st.markdown(full_answer)

        else:
            st.session_state.last_hyde_doc = state.get("hypothetical_doc", "")
            context = state.get("context", "")
            docs = state.get("retrieved_docs", [])

            if context:
                # RAG path — LLM decides [KB] or [GK]
                candidate_sources = [
                    {
                        "source": d.metadata.get("source", "unknown"),
                        "page": d.metadata.get("page", ""),
                        "score": d.metadata.get("score", 0),
                    }
                    for d in docs
                ]
                answer_parts: list[str] = []
                used_kb: bool | None = None
                answer_placeholder = st.empty()

                for token, routing in _stream_answer(
                    prompt, context, st.session_state.conversation_history
                ):
                    answer_parts.append(token)
                    used_kb = routing
                    answer_placeholder.markdown("".join(answer_parts))

                full_answer = "".join(answer_parts)

                if used_kb:
                    from_kb = True
                    sources = candidate_sources
                    st.session_state.last_sources = sources
                    if sources:
                        with st.expander(
                            f"📄 {len(sources)} nguồn tham khảo", expanded=False
                        ):
                            for s in sources:
                                st.caption(_source_label(s))
                else:
                    st.markdown(
                        "<span class='gk-badge'>💬 general knowledge</span>",
                        unsafe_allow_html=True,
                    )

            else:
                # No relevant context → direct LLM
                answer_placeholder = st.empty()
                parts = []
                for token in _stream_direct(prompt, st.session_state.conversation_history):
                    parts.append(token)
                    answer_placeholder.markdown("".join(parts))
                full_answer = "".join(parts)
                st.markdown(
                    "<span class='gk-badge'>💬 general knowledge</span>",
                    unsafe_allow_html=True,
                )

    # Persist
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_answer,
            "sources": sources,
            "from_kb": from_kb,
        }
    )
    st.session_state.conversation_history.append({"role": "user", "content": prompt})
    st.session_state.conversation_history.append(
        {"role": "assistant", "content": full_answer}
    )
