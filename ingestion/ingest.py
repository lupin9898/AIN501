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
import re
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
from retrieval.sparse_embedder import BM25Embedder

DATA_DIRECTORY = Path(__file__).resolve().parent.parent / "data"

CHUNK_SIZE_LIMIT = 1200 
CHUNK_OVERLAP_SIZE = 200
MINIMUM_CHUNK_LENGTH = 50  
SUPPORTED_FILE_EXTENSIONS = (".pdf", ".txt", ".md")

#  alias duoi day giup tuong thich nguoc.
DATA_DIR = DATA_DIRECTORY
CHUNK_SIZE = CHUNK_SIZE_LIMIT
CHUNK_OVERLAP = CHUNK_OVERLAP_SIZE
MIN_CHUNK_LENGTH = MINIMUM_CHUNK_LENGTH
SUPPORTED_EXTENSIONS = SUPPORTED_FILE_EXTENSIONS


def _load_documents(source: Path):
    return load_documents_from_source(source)


def _ensure_collection(client, collection_name: str, vector_size: int, log=None):
    return ensure_qdrant_collection_exists(
        qdrant_client=client,
        target_collection_name=collection_name,
        embedding_vector_size=vector_size,
        logger_function=log,
    )


def run(source: Path | None = None, on_progress: Optional[Callable[[str], None]] = None) -> dict:
    """Wrapper tuong thich nguoc cho pipeline ingestion.

    Tra ve cac key cu de cac noi khac (vd `ui.py`) van chay:
      docs_loaded, chunks_created, chunks_upserted
    """
    stats = execute_ingestion_pipeline(
        source_path_input=source,
        progress_callback_function=on_progress,
    )
    return {
        "docs_loaded": stats.get("documents_loaded_count", 0),
        "chunks_created": stats.get("chunks_created_count", 0),
        "chunks_upserted": stats.get("chunks_upserted_count", 0),
    }


def clean_extracted_pdf_text_content(raw_text_content: str) -> str:
    """Tiền xử lý văn bản từ PDF: Gộp các từ bị rớt dòng và khôi phục cấu trúc đoạn văn."""
    text_lines_list = raw_text_content.split('\n')
    cleaned_text_lines_list = []
    
    for text_line in text_lines_list:
        stripped_text_line = text_line.strip()
        if stripped_text_line:
            cleaned_text_lines_list.append(stripped_text_line)
            
    reconstructed_text_content = ""
    for text_line in cleaned_text_lines_list:
        if text_line.startswith("+") or text_line.startswith("-") or text_line.startswith("*"):
            reconstructed_text_content += "\n" + text_line + " "
        else:
            reconstructed_text_content += text_line + " "
            
    reconstructed_text_content = reconstructed_text_content.replace(". ", ".\n\n")
    final_cleaned_text_content = re.sub(r' +', ' ', reconstructed_text_content)
    
    return final_cleaned_text_content.strip()


def load_documents_from_source(source_path: Path):
    """Load documents from a directory or single file."""
    if source_path.is_file():
        file_extension = source_path.suffix.lower()
        if file_extension == ".pdf":
            pdf_loader = PyPDFLoader(str(source_path))
            loaded_documents = pdf_loader.load()
        else:
            text_loader = TextLoader(str(source_path), encoding="utf-8")
            loaded_documents = text_loader.load()
            
        if file_extension == ".pdf":
            for document_item in loaded_documents:
                document_item.page_content = clean_extracted_pdf_text_content(document_item.page_content)
        return loaded_documents

    document_loaders_list = []
    for search_pattern, loader_class, keyword_arguments in [
        ("**/*.pdf", PyPDFLoader, {}),
        ("**/*.txt", TextLoader, {"encoding": "utf-8"}),
        ("**/*.md", TextLoader, {"encoding": "utf-8"}),
    ]:
        document_loaders_list.append(
            DirectoryLoader(
                str(source_path),
                glob=search_pattern,
                loader_cls=loader_class,
                loader_kwargs=keyword_arguments,
                silent_errors=True,
            )
        )

    all_loaded_documents = []
    for document_loader in document_loaders_list:
        documents_from_loader = document_loader.load()
        for document_item in documents_from_loader:
            document_source_path = document_item.metadata.get("source", "")
            if document_source_path.lower().endswith(".pdf"):
                document_item.page_content = clean_extracted_pdf_text_content(document_item.page_content)
        all_loaded_documents.extend(documents_from_loader)
        
    return all_loaded_documents


def ensure_qdrant_collection_exists(qdrant_client, target_collection_name: str, embedding_vector_size: int, logger_function=None):
    """Create (or recreate) the Qdrant collection with named dense + sparse vectors.

    When HYBRID_SEARCH is enabled the collection uses:
      • named vector  "dense"  — cosine similarity, size = embedding_vector_size
      • sparse vector "sparse" — BM25 via fastembed

    If an existing collection was created with the old flat-vector format it
    is automatically recreated so that hybrid upsert/search works correctly.
    All existing data will be lost and must be re-ingested.
    """
    logger_function = logger_function or (lambda message: print(message))
    existing_names = [c.name for c in qdrant_client.get_collections().collections]

    if target_collection_name in existing_names:
        info = qdrant_client.get_collection(target_collection_name)
        vectors_cfg = info.config.params.vectors

        # Detect old flat-vector format (VectorParams, not a dict of named vectors)
        is_flat = not isinstance(vectors_cfg, dict)
        if settings.HYBRID_SEARCH and is_flat:
            logger_function(
                f"[Ingest] Collection '{target_collection_name}' dùng định dạng vector cũ. "
                "Đang xóa và tạo lại với named vectors để hỗ trợ hybrid search …"
            )
            qdrant_client.delete_collection(target_collection_name)
        elif not settings.HYBRID_SEARCH and is_flat:
            # Old flat format is fine for dense-only
            existing_size = vectors_cfg.size  # type: ignore[union-attr]
            if existing_size != embedding_vector_size:
                raise ValueError(
                    f"Kích thước vector không khớp: collection '{target_collection_name}' "
                    f"có kích thước {existing_size}, model trả về {embedding_vector_size}."
                )
            logger_function(f"[Ingest] Collection '{target_collection_name}' đã tồn tại và hợp lệ")
            return
        elif isinstance(vectors_cfg, dict):
            dense_params = vectors_cfg.get(settings.DENSE_VECTOR_NAME)
            if dense_params and dense_params.size != embedding_vector_size:
                raise ValueError(
                    f"Kích thước vector dense không khớp: collection có {dense_params.size}, "
                    f"model trả về {embedding_vector_size}."
                )
            logger_function(f"[Ingest] Collection '{target_collection_name}' đã tồn tại và hợp lệ")
            return

    if settings.HYBRID_SEARCH:
        qdrant_client.create_collection(
            collection_name=target_collection_name,
            vectors_config={
                settings.DENSE_VECTOR_NAME: models.VectorParams(
                    size=embedding_vector_size,
                    distance=models.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                settings.SPARSE_VECTOR_NAME: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False)
                )
            },
        )
        logger_function(
            f"[Ingest] Đã tạo collection hybrid '{target_collection_name}' "
            f"(dense={settings.DENSE_VECTOR_NAME}, sparse={settings.SPARSE_VECTOR_NAME})"
        )
    else:
        qdrant_client.create_collection(
            collection_name=target_collection_name,
            vectors_config=models.VectorParams(
                size=embedding_vector_size,
                distance=models.Distance.COSINE,
            ),
        )
        logger_function(f"[Ingest] Đã tạo collection mới: '{target_collection_name}'")


def execute_ingestion_pipeline(
    source_path_input: Path | None = None,
    progress_callback_function: Optional[Callable[[str], None]] = None,
) -> dict:
    """Main ingestion pipeline."""
    source_path_input = source_path_input or DATA_DIRECTORY
    logger_function = progress_callback_function or (lambda message: print(message))

    if not source_path_input.exists():
        logger_function(f"Lỗi: đường dẫn dữ liệu không tồn tại: {source_path_input}")
        return {"documents_loaded_count": 0, "chunks_created_count": 0, "chunks_upserted_count": 0}
        
    if source_path_input.is_file() and source_path_input.suffix.lower() not in SUPPORTED_FILE_EXTENSIONS:
        logger_function(f"Lỗi: định dạng tệp '{source_path_input.suffix}' không được hỗ trợ. Chỉ hỗ trợ: {', '.join(SUPPORTED_FILE_EXTENSIONS)}")
        return {"documents_loaded_count": 0, "chunks_created_count": 0, "chunks_upserted_count": 0}

    logger_function(f"Đang tiến hành tải tài liệu từ {source_path_input.name} ...")
    try:
        loaded_documents_list = load_documents_from_source(source_path_input)
    except Exception as document_loading_error:
        logger_function(f"Lỗi trong quá trình đọc tài liệu: {document_loading_error}")
        return {"documents_loaded_count": 0, "chunks_created_count": 0, "chunks_upserted_count": 0}
        
    if not loaded_documents_list:
        logger_function("Không tìm thấy dữ liệu văn bản nào trong đường dẫn.")
        return {"documents_loaded_count": 0, "chunks_created_count": 0, "chunks_upserted_count": 0}
    logger_function(f"Đã tải thành công {len(loaded_documents_list)} tài liệu gốc.")

    text_document_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_LIMIT,
        chunk_overlap=CHUNK_OVERLAP_SIZE,
        separators=["\n\n", "\n# ", "\n## ", "\n### ", "\n+ ", "\n- ", "\n* ", "\n", " ", ""],
        length_function=len,
    )
    
    document_chunks_list = text_document_splitter.split_documents(loaded_documents_list)
    valid_document_chunks_list = []
    
    for document_chunk in document_chunks_list:
        chunk_text_content = (document_chunk.page_content or "").strip()
        
        # BƯỚC DỌN DẸP HẬU KỲ CHUNK (POST-PROCESSING)
        chunk_text_content = re.sub(r'\n[ \t]+', '\n', chunk_text_content)
        chunk_text_content = re.sub(r'\n{3,}', '\n\n', chunk_text_content)
        chunk_text_content = chunk_text_content.strip()
        
        if chunk_text_content and len(chunk_text_content) >= MINIMUM_CHUNK_LENGTH:
            document_chunk.page_content = chunk_text_content
            valid_document_chunks_list.append(document_chunk)
            
    if not valid_document_chunks_list:
        logger_function("Không tạo được đoạn chunk nào hợp lệ sau quá trình lọc.")
        return {"documents_loaded_count": len(loaded_documents_list), "chunks_created_count": 0, "chunks_upserted_count": 0}
        
    logger_function(f"Đã chia tài liệu thành {len(valid_document_chunks_list)} đoạn chunk chứa ngữ nghĩa.")

    hyde_embedding_model = HyDEEmbedder()
    chunk_text_strings_list = [document_chunk.page_content for document_chunk in valid_document_chunks_list]
    logger_function("Đang tiến hành tạo vector embedding (dense) cho các đoạn chunk ...")

    try:
        embedding_vectors_list = hyde_embedding_model.embed_documents(chunk_text_strings_list)
    except Exception as embedding_generation_error:
        logger_function(f"Lỗi trong quá trình tạo embedding: {embedding_generation_error}")
        return {
            "documents_loaded_count": len(loaded_documents_list),
            "chunks_created_count": len(valid_document_chunks_list),
            "chunks_upserted_count": 0,
        }

    if not embedding_vectors_list or len(embedding_vectors_list) != len(valid_document_chunks_list):
        logger_function("Hệ thống khởi tạo vector thất bại hoặc số lượng vector không khớp với chunk. Đã dừng tiến trình an toàn.")
        return {"documents_loaded_count": len(loaded_documents_list), "chunks_created_count": len(valid_document_chunks_list), "chunks_upserted_count": 0}

    embedding_vector_dimensions_size = len(embedding_vectors_list[0])

    sparse_vectors_list: list[dict | None] = [None] * len(valid_document_chunks_list)
    if settings.HYBRID_SEARCH:
        logger_function("Đang tiến hành tạo BM25 sparse vector cho các đoạn chunk ...")
        try:
            bm25_embedder = BM25Embedder()
            sparse_vectors_list = bm25_embedder.embed(chunk_text_strings_list)
            logger_function(f"Đã tạo {len(sparse_vectors_list)} BM25 sparse vector.")
        except Exception as sparse_error:
            logger_function(f"Cảnh báo: BM25 sparse embedding thất bại ({sparse_error}). Tiếp tục với dense-only.")

    from retrieval.qdrant_client import QdrantRetriever

    try:
        qdrant_database_retriever = QdrantRetriever()
        ensure_qdrant_collection_exists(
            qdrant_client=qdrant_database_retriever.client, 
            target_collection_name=settings.COLLECTION_NAME, 
            embedding_vector_size=embedding_vector_dimensions_size, 
            logger_function=logger_function
        )
    except Exception as qdrant_connection_error:
        logger_function(f"Lỗi kết nối cơ sở dữ liệu: {qdrant_connection_error}")
        return {"documents_loaded_count": len(loaded_documents_list), "chunks_created_count": len(valid_document_chunks_list), "chunks_upserted_count": 0}

    qdrant_point_structures_list = []
    for document_chunk, dense_vec, sparse_vec in zip(
        valid_document_chunks_list, embedding_vectors_list, sparse_vectors_list
    ):
        if settings.HYBRID_SEARCH and sparse_vec is not None:
            vector_payload: dict | list = {
                settings.DENSE_VECTOR_NAME: dense_vec,
                settings.SPARSE_VECTOR_NAME: models.SparseVector(
                    indices=sparse_vec["indices"],
                    values=sparse_vec["values"],
                ),
            }
        elif settings.HYBRID_SEARCH:
            # sparse generation failed for this chunk — store dense only in named slot
            vector_payload = {settings.DENSE_VECTOR_NAME: dense_vec}
        else:
            # Dense-only mode: use flat (unnamed) vector for backward compat
            vector_payload = dense_vec  # type: ignore[assignment]

        qdrant_point_structures_list.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector_payload,
                payload={
                    "page_content": document_chunk.page_content,
                    "metadata": document_chunk.metadata,
                },
            )
        )

    upsert_batch_size = 100
    total_upserted_points_count = 0
    
    for batch_start_index in range(0, len(qdrant_point_structures_list), upsert_batch_size):
        current_batch_points = qdrant_point_structures_list[batch_start_index : batch_start_index + upsert_batch_size]
        try:
            qdrant_database_retriever.client.upsert(
                collection_name=settings.COLLECTION_NAME,
                points=current_batch_points,
            )
            total_upserted_points_count += len(current_batch_points)
        except Exception as batch_upsert_error:
            current_batch_number = (batch_start_index // upsert_batch_size) + 1
            logger_function(f"Lỗi khi ghi dữ liệu tại lô thứ {current_batch_number}: {batch_upsert_error}")
            return {
                "documents_loaded_count": len(loaded_documents_list),
                "chunks_created_count": len(valid_document_chunks_list),
                "chunks_upserted_count": total_upserted_points_count, 
            }
            
        progress_count = min(batch_start_index + upsert_batch_size, len(qdrant_point_structures_list))
        logger_function(f"Đã lưu thành công {progress_count}/{len(qdrant_point_structures_list)} đoạn chunk vào Qdrant ...")

    logger_function(f"Tiến trình hoàn tất! Đã đưa vào index thành công tổng cộng {len(qdrant_point_structures_list)} đoạn chunk.")
    return {
        "documents_loaded_count": len(loaded_documents_list),
        "chunks_created_count": len(valid_document_chunks_list),
        "chunks_upserted_count": len(qdrant_point_structures_list),
    }

if __name__ == "__main__":
    target_source_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    execute_ingestion_pipeline(target_source_path)