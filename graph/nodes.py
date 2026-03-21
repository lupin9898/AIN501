"""All six HyDE RAG graph nodes.

Each node receives the full GraphState and returns a *partial* dict that
LangGraph merges back into the state.

HyDE key insight (repeated at the critical embed step):
  We embed the *hypothetical document* — NOT the original query — because
  the hypothetical doc is stylistically similar to the real KB chunks
  (both are "answer-style" prose), so cosine similarity is much more
  meaningful than question-vs-answer similarity.
"""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from config import settings
from graph.state import GraphState
from prompts.answer_prompt import ANSWER_PROMPT
from prompts.hyde_prompt import HYDE_PROMPT
from retrieval.embedder import HyDEEmbedder
from retrieval.sparse_embedder import BM25Embedder
from retrieval.qdrant_client import CollectionNotFoundError, QdrantRetriever

# ── Lazy singletons (deferred until first use so dotenv is loaded) ─────
_llm_creative = None
_llm_factual = None
_embedder = None
_sparse_embedder = None
_retriever = None


def _get_llm_creative():
    global _llm_creative
    if _llm_creative is None:
        _llm_creative = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=0.7,
            google_api_key=settings.GEMINI_API_KEY,
        )
    return _llm_creative


def _get_llm_factual():
    global _llm_factual
    if _llm_factual is None:
        _llm_factual = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            temperature=0,
            google_api_key=settings.GEMINI_API_KEY,
        )
    return _llm_factual


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = HyDEEmbedder()
    return _embedder


def _get_sparse_embedder():
    global _sparse_embedder
    if _sparse_embedder is None:
        _sparse_embedder = BM25Embedder()
    return _sparse_embedder


def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = QdrantRetriever()
    return _retriever


# ── 1. Validate Query ──────────────────────────────────────────────────

def validate_query_node(state: GraphState) -> dict:
    """Check that the query is non-empty and within length limits."""
    query = (state.get("query") or "").strip()

    if not query:
        return {"error": "Query is empty. Please enter a question."}

    if len(query) > 2000:
        return {"error": "Query exceeds 2000 characters. Please shorten it."}

    return {"query": query, "error": None}


# ── 2. Generate Hypothetical Document ─────────────────────────────────

def generate_hypothetical_doc_node(state: GraphState) -> dict:
    """Ask the LLM to imagine an ideal answer document for the query.

    This synthetic document is used ONLY for embedding — it is never
    shown to the user as the final answer.
    """
    query = state["query"]

    try:
        chain = HYDE_PROMPT | _get_llm_creative()
        response = chain.invoke({"query": query})
        hypothetical_doc = response.content

        # Visible log so the operator can inspect the HyDE generation
        print(f"\n[HyDE] Hypothetical document generated ({len(hypothetical_doc)} chars)")

        return {"hypothetical_doc": hypothetical_doc}

    except Exception as exc:
        # Graceful degradation: fall back to embedding the raw query
        # (equivalent to standard RAG without HyDE)
        print(f"\n[HyDE] LLM failed ({exc}), falling back to raw query embedding")
        return {"hypothetical_doc": query}


# ── 3. Embed Hypothetical Document ────────────────────────────────────

def embed_hypothetical_node(state: GraphState) -> dict:
    """Embed the hypothetical document — NOT the original user query.

    Why?  The KB chunks are answer-style text.  The hypothetical doc is
    also answer-style text.  Embedding both in the same "style space"
    produces far better cosine-similarity scores than embedding a short
    question against long answer passages.

    When HYBRID_SEARCH is enabled, also computes a BM25 sparse vector from
    the same hypothetical document for use in RRF fusion retrieval.
    """
    hypothetical_doc = state["hypothetical_doc"]
    dense_vector = _get_embedder().embed_query(hypothetical_doc)

    sparse_vector = None
    if settings.HYBRID_SEARCH:
        try:
            sparse_vector = _get_sparse_embedder().embed_query(hypothetical_doc)
        except Exception as exc:
            print(f"\n[Hybrid] BM25 sparse embedding failed ({exc}), falling back to dense-only")

    return {"hypothetical_vector": dense_vector, "sparse_vector": sparse_vector}


# ── 4. Retrieve Documents from Qdrant ─────────────────────────────────

def retrieve_documents_node(state: GraphState) -> dict:
    """Search Qdrant using the hypothetical-document vector.

    The search vector comes from the HyDE embedding (step 3), NOT from
    embedding the user's raw query.  This is the core of the HyDE pattern.

    When HYBRID_SEARCH is enabled and a sparse_vector is present, performs
    RRF-fused hybrid retrieval (dense + BM25).
    """
    vector = state["hypothetical_vector"]
    sparse_vector = state.get("sparse_vector")

    try:
        docs = _get_retriever().search(vector=vector, sparse_vector=sparse_vector)
    except CollectionNotFoundError as exc:
        return {"retrieved_docs": [], "error": str(exc)}
    except ConnectionError as exc:
        return {"retrieved_docs": [], "error": str(exc)}

    if not docs:
        return {
            "retrieved_docs": [],
            "error": "No relevant documents found in the knowledge base.",
        }

    return {"retrieved_docs": docs}


# ── 5. Assemble Context ───────────────────────────────────────────────

def assemble_context_node(state: GraphState) -> dict:
    """Sort retrieved docs by score and build a context string.

    Enforces a token budget (MAX_CONTEXT_TOKENS) by truncating the list
    of documents once the budget is exceeded.
    """
    docs = state.get("retrieved_docs", [])

    if not docs:
        return {"context": ""}

    # Sort by relevance score (highest first)
    docs_sorted = sorted(docs, key=lambda d: d.metadata.get("score", 0), reverse=True)

    context_parts: list[str] = []
    token_count = 0

    for doc in docs_sorted:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "")
        score = doc.metadata.get("score", 0)
        source_label = f"{source}" + (f" (p.{page})" if page else "")

        block = f"[Source: {source_label} | score: {score:.3f}]\n{doc.page_content}\n---"

        # Rough token estimate: 1 token ≈ 4 characters
        block_tokens = len(block) // 4
        if token_count + block_tokens > settings.MAX_CONTEXT_TOKENS:
            break

        context_parts.append(block)
        token_count += block_tokens

    return {"context": "\n\n".join(context_parts)}


# ── 6. Generate Final Answer ───────────────────────────────────────────

def generate_answer_node(state: GraphState) -> dict:
    """Generate the user-facing answer from retrieved context.

    IMPORTANT: The context is built from *real KB documents* retrieved
    via Qdrant — NOT from the hypothetical document.  The hypothetical
    doc was only a search probe; the answer must be grounded in actual data.
    """
    query = state["query"]
    context = state.get("context", "")
    history = state.get("conversation_history", [])

    # If no context available, refuse to answer from general knowledge
    if not context:
        answer = (
            "I could not find relevant documents in the knowledge base to answer "
            "your question. Please try rephrasing or ensure the relevant documents "
            "have been ingested."
        )
        return {
            "answer": answer,
            "conversation_history": [
                {"role": "user", "content": query},
                {"role": "assistant", "content": answer},
            ],
        }

    # Build chat_history messages for the prompt (last 10 turns = 20 messages)
    chat_history_messages = []
    recent = history[-20:] if len(history) > 20 else history
    for turn in recent:
        if turn["role"] == "user":
            chat_history_messages.append(HumanMessage(content=turn["content"]))
        else:
            chat_history_messages.append(AIMessage(content=turn["content"]))

    chain = ANSWER_PROMPT | _get_llm_factual()
    response = chain.invoke({
        "context": context,
        "query": query,
        "chat_history": chat_history_messages,
    })
    answer = response.content

    # Append this turn to conversation history
    return {
        "answer": answer,
        "conversation_history": [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer},
        ],
    }
