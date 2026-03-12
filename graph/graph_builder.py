"""Build and compile the LangGraph HyDE RAG workflow.

Graph topology:
  START → validate_query → (conditional) → generate_hypothetical_doc
        → embed_hypothetical → retrieve_documents → assemble_context
        → generate_answer → END

If validation fails the graph short-circuits to END with an error message.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from graph.nodes import (
    assemble_context_node,
    embed_hypothetical_node,
    generate_answer_node,
    generate_hypothetical_doc_node,
    retrieve_documents_node,
    validate_query_node,
)
from graph.state import GraphState


def _route_after_validation(state: GraphState) -> str:
    """Conditional edge: skip the pipeline if the query is invalid."""
    if state.get("error"):
        return END
    return "generate_hypothetical_doc"


def build_graph() -> StateGraph:
    """Construct the HyDE RAG state graph (uncompiled)."""
    graph = StateGraph(GraphState)

    # Register nodes
    graph.add_node("validate_query", validate_query_node)
    graph.add_node("generate_hypothetical_doc", generate_hypothetical_doc_node)
    graph.add_node("embed_hypothetical", embed_hypothetical_node)
    graph.add_node("retrieve_documents", retrieve_documents_node)
    graph.add_node("assemble_context", assemble_context_node)
    graph.add_node("generate_answer", generate_answer_node)

    # Wire edges
    graph.add_edge(START, "validate_query")
    graph.add_conditional_edges("validate_query", _route_after_validation)
    graph.add_edge("generate_hypothetical_doc", "embed_hypothetical")
    graph.add_edge("embed_hypothetical", "retrieve_documents")
    graph.add_edge("retrieve_documents", "assemble_context")
    graph.add_edge("assemble_context", "generate_answer")
    graph.add_edge("generate_answer", END)

    return graph


def compile_graph():
    """Build, attach a memory checkpointer, and compile the graph."""
    graph = build_graph()
    checkpointer = MemorySaver()
    return graph.compile(checkpointer=checkpointer)
