"""Prompt template for generating the final answer from retrieved context."""

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant.\n"
            "A context block from a knowledge base is provided. "
            "If the context is relevant to the question, start your reply with the token [KB] then answer using it. "
            "If the context is NOT relevant to the question, start your reply with the token [GK] then answer naturally "
            "from your general knowledge — do NOT mention the context at all.\n"
            "Do NOT include source citations, file paths, or [Source: ...] in your answer.",
        ),
        # Inject prior conversation turns for multi-turn memory
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        (
            "human",
            "Context:\n{context}\n\nQuestion: {query}",
        ),
    ]
)
