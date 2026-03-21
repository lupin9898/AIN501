"""BM25 sparse embedder backed by fastembed (Qdrant/bm25 model)."""

from __future__ import annotations

import logging
from typing import Optional

from fastembed import SparseTextEmbedding

log = logging.getLogger(__name__)

# SparseVector data as a plain dict for LangGraph state serialisability.
# Schema: {"indices": list[int], "values": list[float]}
SparseVectorDict = dict


class BM25Embedder:
    """Thin wrapper around fastembed SparseTextEmbedding for BM25 sparse vectors.

    Uses the Qdrant/bm25 model which produces sparse vectors compatible with
    Qdrant's native sparse vector index.  Model weights are downloaded on
    first instantiation and cached locally by fastembed.
    """

    MODEL_NAME = "Qdrant/bm25"

    def __init__(self) -> None:
        self._model: Optional[SparseTextEmbedding] = None

    def _get_model(self) -> SparseTextEmbedding:
        """Lazy-load the model on first use."""
        if self._model is None:
            log.info("[BM25] Loading fastembed model '%s' …", self.MODEL_NAME)
            self._model = SparseTextEmbedding(model_name=self.MODEL_NAME)
        return self._model

    def embed(self, texts: list[str]) -> list[SparseVectorDict]:
        """Embed multiple texts into BM25 sparse vectors.

        Returns a list of dicts with keys 'indices' (list[int]) and
        'values' (list[float]).
        """
        results = list(self._get_model().embed(texts))
        return [
            {"indices": r.indices.tolist(), "values": r.values.tolist()}
            for r in results
        ]

    def embed_query(self, text: str) -> SparseVectorDict:
        """Embed a single query text into a BM25 sparse vector."""
        return self.embed([text])[0]
