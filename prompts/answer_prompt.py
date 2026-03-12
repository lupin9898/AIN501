"""Prompt template for generating the final answer from retrieved context."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer based ONLY on the provided context. "
            "If the context doesn't contain enough information, say so clearly. "
            "Do not hallucinate or make up facts beyond what the context provides.\n"
            "Always cite the source when possible using [Source: ...] notation.",
        ),
        # Inject prior conversation turns for multi-turn memory
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {query}",
        ),
    ]
)
