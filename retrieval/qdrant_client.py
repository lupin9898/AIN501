"""Qdrant vector-store retriever with retry logic and hybrid BM25+dense search."""

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

    When HYBRID_SEARCH is enabled and a sparse_vector is supplied, the
    retriever performs Reciprocal Rank Fusion (RRF) over:
      • dense cosine-similarity results (from the HyDE embedding)
      • sparse BM25 results (from the fastembed Qdrant/bm25 model)
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
        sparse_vector: Optional[dict] = None,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> list[Document]:
        """Search Qdrant using dense (and optionally sparse BM25) vectors.

        When hybrid search is enabled and sparse_vector is provided, uses
        Qdrant's native RRF fusion over dense + sparse prefetch results.
        Falls back to dense-only search otherwise.

        Returns LangChain Document objects with score stored in metadata.
        Retries up to MAX_RETRIES times on transient connection failures.
        """
        top_k = top_k or settings.TOP_K
        score_threshold = score_threshold or settings.SCORE_THRESHOLD

        use_hybrid = (
            settings.HYBRID_SEARCH
            and sparse_vector is not None
            and len(sparse_vector.get("indices", [])) > 0
        )

        last_err: Exception | None = None
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                if use_hybrid:
                    points = self._hybrid_search(vector, sparse_vector, top_k)  # type: ignore[arg-type]
                else:
                    points = self._dense_search(vector, top_k, score_threshold)
                return self._to_documents(points)
            except Exception as exc:
                err_str = str(exc)
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

    # ── private search strategies ────────────────────────────────

    def _dense_search(
        self,
        vector: list[float],
        top_k: int,
        score_threshold: float,
    ) -> list:
        """Single-vector cosine-similarity search."""
        # Support both old (unnamed) and new (named "dense") vector formats.
        try:
            results = self._client.query_points(
                collection_name=self._collection,
                query=vector,
                using=settings.DENSE_VECTOR_NAME,
                limit=top_k,
                score_threshold=score_threshold,
                with_payload=True,
            )
        except Exception as exc:
            # Fallback for old collections that use unnamed (flat) vector format.
            if "wrong input" in str(exc).lower() or "not found" in str(exc).lower():
                results = self._client.query_points(
                    collection_name=self._collection,
                    query=vector,
                    limit=top_k,
                    score_threshold=score_threshold,
                    with_payload=True,
                )
            else:
                raise
        return results.points

    def _hybrid_search(
        self,
        dense_vector: list[float],
        sparse_vector: dict,
        top_k: int,
    ) -> list:
        """Hybrid RRF fusion over dense + BM25 sparse results.

        Qdrant's query API with prefetch + Fusion.RRF re-ranks results from
        both retrieval arms — no score_threshold is applied here because RRF
        scores are not comparable to cosine similarity values.
        """
        prefetch_limit = top_k * 3  # cast wider net before fusion
        sv = models.SparseVector(
            indices=sparse_vector["indices"],
            values=sparse_vector["values"],
        )
        results = self._client.query_points(
            collection_name=self._collection,
            prefetch=[
                models.Prefetch(
                    query=dense_vector,
                    using=settings.DENSE_VECTOR_NAME,
                    limit=prefetch_limit,
                ),
                models.Prefetch(
                    query=sv,
                    using=settings.SPARSE_VECTOR_NAME,
                    limit=prefetch_limit,
                ),
            ],
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return results.points

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
