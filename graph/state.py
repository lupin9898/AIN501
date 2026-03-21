"""LangGraph state definition for the HyDE RAG pipeline."""

from __future__ import annotations

import operator
from typing import Annotated, Optional

from langchain_core.documents import Document
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """Shared state flowing through every node of the HyDE graph.

    Fields updated by each node are merged back automatically by LangGraph.
    conversation_history uses operator.add so each node can *append* turns
    without overwriting previous history.
    """

    query: str  # Original user question
    hypothetical_doc: str  # LLM-generated ideal answer document
    hypothetical_vector: list[float]  # Dense embedding of hypothetical_doc
    sparse_vector: Optional[dict]  # BM25 sparse vector {"indices", "values"} or None
    retrieved_docs: list[Document]  # Documents retrieved from Qdrant
    context: str  # Assembled context string for final generation
    answer: str  # Final answer returned to the user
    conversation_history: Annotated[list[dict], operator.add]  # Accumulated chat turns
    error: Optional[str]  # Error message, if any
