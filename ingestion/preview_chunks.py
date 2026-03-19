"""Preview chunking output by exporting chunks to a txt file.

Usage (from project root):
    py -m ingestion.preview_chunks data/sample.txt
    py -m ingestion.preview_chunks data/samplepdf.pdf --out data/chunks_samplepdf.txt

This script ONLY tests: load -> chunk -> filter. It does NOT embed or upsert.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

import ingestion.ingest as ingest_module


def retrieve_first_matching_attribute(module_target, attribute_names_list: list[str]):
    for attribute_name in attribute_names_list:
        if hasattr(module_target, attribute_name):
            return getattr(module_target, attribute_name)
    raise AttributeError(f"Không tìm thấy thuộc tính nào trong {module_target.__name__}: {attribute_names_list}")


LOAD_DOCUMENTS_FUNCTION = retrieve_first_matching_attribute(ingest_module, ["_load_documents", "load_documents_from_source"])
CHUNK_SIZE_LIMIT = retrieve_first_matching_attribute(ingest_module, ["CHUNK_SIZE", "CHUNK_SIZE_LIMIT"])
CHUNK_OVERLAP_SIZE = retrieve_first_matching_attribute(ingest_module, ["CHUNK_OVERLAP", "CHUNK_OVERLAP_SIZE"])
MINIMUM_CHUNK_LENGTH = retrieve_first_matching_attribute(ingest_module, ["MIN_CHUNK_LENGTH", "MINIMUM_CHUNK_LENGTH"])
SUPPORTED_FILE_EXTENSIONS = retrieve_first_matching_attribute(
    ingest_module, ["SUPPORTED_EXTENSIONS", "SUPPORTED_FILE_EXTENSIONS"]
)


def build_argument_parser() -> argparse.ArgumentParser:
    argument_parser = argparse.ArgumentParser(description="Export chunked documents to a txt file.")
    argument_parser.add_argument(
        "source",
        type=str,
        help="Input file or directory to chunk (pdf/txt/md). Example: data/samplepdf.pdf",
    )
    argument_parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output txt path. Default: data/chunks_<source-stem>.txt",
    )
    argument_parser.add_argument(
        "--max-chunks",
        type=int,
        default=0,
        help="Limit number of chunks written (0 = no limit).",
    )
    return argument_parser


def export_document_chunks_to_file(source_path: Path, output_file_path: Path, maximum_chunks_limit: int = 0) -> dict:
    if not source_path.exists():
        raise FileNotFoundError(f"Đường dẫn không tồn tại: {source_path}")
    if source_path.is_file() and source_path.suffix.lower() not in SUPPORTED_FILE_EXTENSIONS:
        raise ValueError(
            f"Định dạng '{source_path.suffix}' không hỗ trợ. Chỉ dùng: {', '.join(SUPPORTED_FILE_EXTENSIONS)}"
        )

    loaded_documents_list = LOAD_DOCUMENTS_FUNCTION(source_path)
    text_document_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE_LIMIT,
        chunk_overlap=CHUNK_OVERLAP_SIZE,
        separators=["\n\n", "\n+ ", "\n- ", "\n* ", "\n", " ", ""],
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

    total_chunks_count = len(valid_document_chunks_list)
    if maximum_chunks_limit and maximum_chunks_limit > 0:
        valid_document_chunks_list = valid_document_chunks_list[:maximum_chunks_limit]

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    with output_file_path.open("w", encoding="utf-8") as file_handler:
        file_handler.write("=== Chunk Preview Export ===\n")
        file_handler.write(f"source: {source_path}\n")
        file_handler.write(f"chunk_size: {CHUNK_SIZE_LIMIT}\n")
        file_handler.write(f"chunk_overlap: {CHUNK_OVERLAP_SIZE}\n")
        file_handler.write(f"min_chunk_length: {MINIMUM_CHUNK_LENGTH}\n")
        file_handler.write(f"docs_loaded: {len(loaded_documents_list)}\n")
        file_handler.write(f"chunks_total: {total_chunks_count}\n")
        
        if maximum_chunks_limit and maximum_chunks_limit > 0:
            file_handler.write(f"chunks_written: {len(valid_document_chunks_list)} (limited by --max-chunks)\n")
        else:
            file_handler.write(f"chunks_written: {len(valid_document_chunks_list)}\n")
        file_handler.write("\n")

        for chunk_index, document_chunk in enumerate(valid_document_chunks_list):
            chunk_metadata_dictionary = document_chunk.metadata or {}
            file_handler.write(f"\n===== CHUNK {chunk_index} | len={len(document_chunk.page_content)} =====\n")
            file_handler.write("metadata: ")
            file_handler.write(json.dumps(chunk_metadata_dictionary, ensure_ascii=False))
            file_handler.write("\n\n")
            file_handler.write(document_chunk.page_content)
            file_handler.write("\n")

    return {"documents_loaded_count": len(loaded_documents_list), "total_chunks_count": total_chunks_count, "output_file_path": str(output_file_path)}


def main_execution() -> None:
    command_line_arguments = build_argument_parser().parse_args()
    source_path_input = Path(command_line_arguments.source)

    if command_line_arguments.out:
        output_file_path = Path(command_line_arguments.out)
    else:
        default_file_name = f"chunks_{source_path_input.stem}.txt"
        output_file_path = Path("data") / default_file_name

    execution_result = export_document_chunks_to_file(
        source_path=source_path_input, 
        output_file_path=output_file_path, 
        maximum_chunks_limit=int(command_line_arguments.max_chunks)
    )
    
    # Avoid UnicodeEncodeError on some Windows consoles (cp949/cp1252)
    print(f"Da ghi du lieu chunk ra file: {execution_result['output_file_path']}")
    print(f"Tong tai lieu = {execution_result['documents_loaded_count']}, Tong chunks = {execution_result['total_chunks_count']}")


if __name__ == "__main__":
    main_execution()