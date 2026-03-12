"""Qdrant vector-store retriever with retry logic."""

from __future__ import annotations

import time
from typing import Optional

from langchain_core.documents import Document
from qdrant_client import QdrantClient, models


class CollectionNotFoundError(Exception):
    """Raised when the Qdrant collection does not exist yet."""

from config import settings


class QdrantRetriever:
    """Handles all interactions with the Qdrant vector database.

    search() accepts the *hypothetical_vector* (NOT the raw query vector)
    because HyDE embeds the synthetic answer document — this places the
    search vector in the same embedding subspace as the stored KB chunks,
    resulting in higher-quality nearest-neighbor matches.
    """

    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds

    def __init__(self) -> None:
        api_key = settings.QDRANT_API_KEY or None
        self._client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=api_key,
        )
        self._collection = settings.COLLECTION_NAME

    # ── public API ──────────────────────────────────────────────

    def search(
        self,
        vector: list[float],
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> list[Document]:
        """Search Qdrant using the hypothetical-document embedding vector.

        Returns LangChain Document objects with score stored in metadata.
        Retries up to MAX_RETRIES times on transient connection failures.
        """
        top_k = top_k or settings.TOP_K
        score_threshold = score_threshold or settings.SCORE_THRESHOLD

        last_err: Exception | None = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                results = self._client.query_points(
                    collection_name=self._collection,
                    query=vector,
                    limit=top_k,
                    score_threshold=score_threshold,
                    with_payload=True,
                )
                return self._to_documents(results.points)
            except Exception as exc:
                err_str = str(exc)
                # 404 = collection chưa tồn tại — không cần retry
                if "404" in err_str or "doesn't exist" in err_str or "Not found" in err_str:
                    raise CollectionNotFoundError(
                        f"Collection '{self._collection}' chưa có dữ liệu. "
                        "Hãy upload và ingest tài liệu trước."
                    ) from exc
                last_err = exc
                if attempt < self.MAX_RETRIES:
                    time.sleep(self.RETRY_DELAY)

        raise ConnectionError(
            f"Không thể kết nối Qdrant sau {self.MAX_RETRIES} lần thử: {last_err}"
        )

    @property
    def client(self) -> QdrantClient:
        """Expose underlying client for admin operations (e.g. ingestion)."""
        return self._client

    # ── helpers ─────────────────────────────────────────────────

    @staticmethod
    def _to_documents(points: list) -> list[Document]:
        docs: list[Document] = []
        for pt in points:
            payload = pt.payload or {}
            content = payload.get("page_content", payload.get("text", ""))
            metadata = payload.get("metadata", {})
            metadata["score"] = pt.score
            docs.append(Document(page_content=content, metadata=metadata))
        return docs
