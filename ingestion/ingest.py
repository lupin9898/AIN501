"""Standalone ingestion script — load documents, chunk, embed, upsert to Qdrant.

Usage:
    cd hyde_rag
    python -m ingestion.ingest          # index everything in ./data/
    python -m ingestion.ingest path/to/file.pdf   # index a single file

Supported formats: .pdf, .txt, .md
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Callable, Optional

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import models

from config import settings
from retrieval.embedder import HyDEEmbedder

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
MIN_CHUNK_LENGTH = 10  # Chunks shorter than this are skipped (validation)
SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md")


def _load_documents(source: Path):
    """Load documents from a directory or single file."""
    if source.is_file():
        ext = source.suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(str(source))
        else:
            loader = TextLoader(str(source), encoding="utf-8")
        return loader.load()

    # Directory — load all supported types
    loaders = []
    for pattern, loader_cls, kwargs in [
        ("**/*.pdf", PyPDFLoader, {}),
        ("**/*.txt", TextLoader, {"encoding": "utf-8"}),
        ("**/*.md", TextLoader, {"encoding": "utf-8"}),
    ]:
        loaders.append(
            DirectoryLoader(
                str(source),
                glob=pattern,
                loader_cls=loader_cls,
                loader_kwargs=kwargs,
                silent_errors=True,
            )
        )

    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    return docs


def _ensure_collection(client, collection_name: str, vector_size: int, log=None):
    """Create the Qdrant collection if it doesn't exist; else check vector size matches."""
    log = log or (lambda msg: print(msg))
    collections = [c.name for c in client.get_collections().collections]
    if collection_name not in collections:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        log(f"[Ingest] Đã tạo collection '{collection_name}'")
    else:
        info = client.get_collection(collection_name)
        existing_size = info.config.params.vectors.size
        if existing_size != vector_size:
            raise ValueError(
                f"Kích thước vector không khớp: collection '{collection_name}' có size {existing_size}, "
                f"trong khi model embedding trả về {vector_size}. Hãy dùng collection mới hoặc cùng model embedding."
            )
        log(f"[Ingest] Collection '{collection_name}' đã tồn tại")


def run(
    source: Path | None = None,
    on_progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """Main ingestion pipeline.

    Args:
        source: File or directory to ingest. Defaults to ./data/.
        on_progress: Optional callback for progress updates.

    Returns:
        dict with keys: docs_loaded, chunks_created, chunks_upserted
    """
    source = source or DATA_DIR
    log = on_progress or (lambda msg: print(msg))

    # 0. Validate source exists
    if not source.exists():
        log(f"Lỗi: đường dẫn không tồn tại: {source}")
        return {"docs_loaded": 0, "chunks_created": 0, "chunks_upserted": 0}
    # When source is a file, only allow supported formats
    if source.is_file() and source.suffix.lower() not in SUPPORTED_EXTENSIONS:
        log(f"Lỗi: định dạng '{source.suffix}' không hỗ trợ. Chỉ dùng: {', '.join(SUPPORTED_EXTENSIONS)}")
        return {"docs_loaded": 0, "chunks_created": 0, "chunks_upserted": 0}

    # 1. Load
    log(f"Đang tải tài liệu từ {source.name} ...")
    try:
        docs = _load_documents(source)
    except Exception as e:
        log(f"Lỗi khi tải tài liệu: {e}")
        return {"docs_loaded": 0, "chunks_created": 0, "chunks_upserted": 0}
    if not docs:
        log("Không tìm thấy tài liệu nào.")
        return {"docs_loaded": 0, "chunks_created": 0, "chunks_upserted": 0}
    log(f"Đã tải {len(docs)} tài liệu.")

    # 2. Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    # Filter out empty or too-short chunks (noise, save embedding cost)
    original_count = len(chunks)
    chunks = [c for c in chunks if (c.page_content or "").strip() and len((c.page_content or "").strip()) >= MIN_CHUNK_LENGTH]
    if len(chunks) < original_count:
        log(f"Đã bỏ {original_count - len(chunks)} chunk rỗng hoặc quá ngắn.")
    if not chunks:
        log("Không còn chunk nào sau khi lọc.")
        return {"docs_loaded": len(docs), "chunks_created": 0, "chunks_upserted": 0}
    log(f"Đã chia thành {len(chunks)} chunk.")

    # 3. Embed (with error handling)
    embedder = HyDEEmbedder()
    texts = [c.page_content for c in chunks]
    log("Đang embedding các chunk ...")
    try:
        vectors = embedder.embed_documents(texts)
    except Exception as e:
        log(f"Lỗi embedding: {e}")
        return {
            "docs_loaded": len(docs),
            "chunks_created": len(chunks),
            "chunks_upserted": 0,
        }
    if not vectors:
        log("Không tạo được vector nào.")
        return {"docs_loaded": len(docs), "chunks_created": len(chunks), "chunks_upserted": 0}
    # Ensure we got one vector per chunk (avoid silent drop if API returns fewer)
    if len(vectors) != len(chunks):
        log(f"Lỗi: embedding trả về {len(vectors)} vector cho {len(chunks)} chunk. Đã dừng.")
        return {"docs_loaded": len(docs), "chunks_created": len(chunks), "chunks_upserted": 0}
    vector_size = len(vectors[0])

    # 4. Ensure Qdrant collection exists (and vector size matches if existing)
    from retrieval.qdrant_client import QdrantRetriever

    try:
        retriever = QdrantRetriever()
        _ensure_collection(retriever.client, settings.COLLECTION_NAME, vector_size, log=log)
    except ValueError as e:
        log(str(e))
        return {"docs_loaded": len(docs), "chunks_created": len(chunks), "chunks_upserted": 0}
    except Exception as e:
        log(f"Không thể kết nối Qdrant: {e}")
        return {"docs_loaded": len(docs), "chunks_created": len(chunks), "chunks_upserted": 0}

    # 5. Upsert — use UUID-based IDs to avoid overwriting previous uploads
    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "page_content": chunk.page_content,
                    "metadata": chunk.metadata,
                },
            )
        )

    # Batch upsert (100 points per batch)
    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        try:
            retriever.client.upsert(
                collection_name=settings.COLLECTION_NAME,
                points=batch,
            )
        except Exception as e:
            log(f"Lỗi ghi batch {i // batch_size + 1}: {e}")
            return {
                "docs_loaded": len(docs),
                "chunks_created": len(chunks),
                "chunks_upserted": i,  # partial
            }
        log(f"Đã ghi {min(i + batch_size, len(points))}/{len(points)} chunk ...")

    log(f"Hoàn tất! Đã index {len(points)} chunk.")
    return {
        "docs_loaded": len(docs),
        "chunks_created": len(chunks),
        "chunks_upserted": len(points),
    }


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    run(target)
