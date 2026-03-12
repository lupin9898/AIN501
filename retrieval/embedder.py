"""Embedding wrapper used for both indexing and HyDE query embedding.

Why embed the *hypothetical document* instead of the raw query?
─────────────────────────────────────────────────────────────────
Traditional RAG embeds the user's short question and searches for similar
chunks.  The problem is that a *question* lives in a different region of
embedding space than an *answer* — so the nearest-neighbor search is
cross-domain (question → answer).

HyDE first asks an LLM to *imagine* what a perfect answer document would
look like, then embeds that synthetic document.  Because the synthetic
document and the real KB chunks are both *answers*, they occupy the same
region of embedding space, yielding much better cosine-similarity matches.
"""

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from config import settings


class HyDEEmbedder:
    """Thin wrapper around the embedding model for consistent reuse."""

    def __init__(self) -> None:
        self._model = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
        )

    def embed_query(self, text: str) -> list[float]:
        """Embed a single text (hypothetical doc or fallback query)."""
        return self._model.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts (used during ingestion)."""
        return self._model.embed_documents(texts)

    @property
    def underlying(self) -> GoogleGenerativeAIEmbeddings:
        """Expose the raw LangChain embedding object (needed by Qdrant ingestion)."""
        return self._model
