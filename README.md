# HyDE RAG Chatbot

Chatbot hỏi đáp dựa trên tài liệu sử dụng kỹ thuật **HyDE (Hypothetical Document Embedding)**, xây dựng với LangGraph + LangChain + Google Gemini + Qdrant.

> **Đồ án môn AIN501 — FPT University**

## Mục lục

- [HyDE là gì?](#hyde-là-gì)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Tech Stack](#tech-stack)
- [Cấu trúc project](#cấu-trúc-project)
- [Cài đặt & Chạy](#cài-đặt--chạy)
- [Cấu hình](#cấu-hình)
- [Sử dụng](#sử-dụng)
- [Ingestion Pipeline](#ingestion-pipeline)
- [Ví dụ minh họa](#ví-dụ-minh-họa)

## HyDE là gì?

RAG truyền thống embed **câu hỏi** của user rồi tìm các chunk tương tự. Vấn đề: một câu hỏi ngắn ("Lạm phát là gì?") nằm ở vùng embedding space rất khác so với đoạn văn chứa câu trả lời.

**HyDE** giải quyết bằng cách nhờ LLM **tưởng tượng** ra một tài liệu trả lời lý tưởng, rồi embed *tài liệu giả định đó*. Vì cả tài liệu giả định và các chunk thật trong KB đều là văn bản dạng "câu trả lời", cosine similarity hoạt động tốt hơn rất nhiều.

```
Standard RAG:                           HyDE RAG:

  User Query ──embed──► Query Vector     User Query ──LLM──► Hypothetical Doc
                              │                                     │
                              │                                  embed
                              │                                     │
                              ▼                                     ▼
                         Qdrant Search                        Qdrant Search
                              │                                     │
                              ▼                                     ▼
                        Retrieved Docs                       Retrieved Docs
                              │                                     │
                              ▼                                     ▼
                         LLM Answer                            LLM Answer

  Question ↔ Answer mismatch!         Answer ↔ Answer match! Much better recall.
```

## Kiến trúc hệ thống

### LangGraph Workflow (6 nodes)

```
┌─────────────────────────────────────────────────────────────────┐
│                        HyDE RAG Graph                           │
│                                                                 │
│  START                                                          │
│    │                                                            │
│    ▼                                                            │
│  ┌──────────────────┐                                           │
│  │ validate_query    │──── invalid ──► END (error)              │
│  └──────────────────┘                                           │
│    │ valid                                                      │
│    ▼                                                            │
│  ┌──────────────────────────┐                                   │
│  │ generate_hypothetical_doc │  LLM tưởng tượng câu trả lời    │
│  └──────────────────────────┘                                   │
│    │                                                            │
│    ▼                                                            │
│  ┌──────────────────────┐                                       │
│  │ embed_hypothetical    │  Embed tài liệu GIẢ ĐỊNH            │
│  └──────────────────────┘  (KHÔNG embed câu hỏi gốc!)          │
│    │                                                            │
│    ▼                                                            │
│  ┌──────────────────────┐                                       │
│  │ retrieve_documents    │  Tìm kiếm Qdrant bằng HyDE vector   │
│  └──────────────────────┘                                       │
│    │                                                            │
│    ▼                                                            │
│  ┌──────────────────────┐                                       │
│  │ assemble_context      │  Sắp xếp theo score, giới hạn token │
│  └──────────────────────┘                                       │
│    │                                                            │
│    ▼                                                            │
│  ┌──────────────────────┐                                       │
│  │ generate_answer       │  LLM trả lời từ tài liệu THẬT      │
│  └──────────────────────┘                                       │
│    │                                                            │
│    ▼                                                            │
│   END                                                           │
└─────────────────────────────────────────────────────────────────┘
```

### Mô tả từng node

| Node | Chức năng | Chi tiết |
|------|-----------|----------|
| `validate_query` | Kiểm tra input | Từ chối query rỗng hoặc vượt 2000 ký tự |
| `generate_hypothetical_doc` | Tạo tài liệu giả định | Dùng LLM (temperature=0.7) sinh đoạn văn 150-250 từ mô tả câu trả lời lý tưởng |
| `embed_hypothetical` | Embed tài liệu giả định | Dùng `gemini-embedding-001` — đây là bước cốt lõi của HyDE |
| `retrieve_documents` | Truy xuất tài liệu | Tìm top-K chunk gần nhất trong Qdrant bằng cosine similarity |
| `assemble_context` | Ghép context | Sắp xếp theo score giảm dần, cắt khi vượt token budget |
| `generate_answer` | Sinh câu trả lời | LLM (temperature=0) trả lời dựa trên tài liệu **thật**, có trích nguồn |

### Fallback behavior

- Nếu bước `generate_hypothetical_doc` lỗi → fallback về embed câu hỏi gốc (tương đương standard RAG)
- Nếu chưa có knowledge base → Streamlit UI tự động chuyển sang chat trực tiếp với Gemini (không qua RAG)
- Retry logic: Qdrant client tự động retry 3 lần khi mất kết nối

## Tech Stack

| Thành phần | Công nghệ | Phiên bản |
|------------|-----------|-----------|
| **LLM** | Google Gemini | gemini-2.5-flash |
| **Embedding** | Google Generative AI Embeddings | gemini-embedding-001 |
| **Vector Database** | Qdrant | v1.12.4 (Docker) |
| **Orchestration** | LangGraph + LangChain | langgraph 0.2.60 |
| **Web UI** | Streamlit | 1.41.1 |
| **Config** | Pydantic Settings | 2.7.1 |
| **PDF Parsing** | PyPDF | 5.1.0 |
| **Tokenizer** | tiktoken | 0.8.0 |

## Cấu trúc project

```
AIN501/
├── config.py                  # Cấu hình tập trung (pydantic-settings, đọc từ .env)
├── main.py                    # CLI entry point
├── ui.py                      # Streamlit Web UI (streaming, upload, citations)
├── docker-compose.yml         # Qdrant container
├── requirements.txt           # Python dependencies
├── .env.example               # Template biến môi trường
│
├── graph/                     # LangGraph state machine
│   ├── state.py               #   GraphState TypedDict (shared state)
│   ├── nodes.py               #   6 node functions của pipeline
│   └── graph_builder.py       #   Build & compile graph với MemorySaver
│
├── ingestion/                 # Document ingestion pipeline
│   ├── ingest.py              #   Load → chunk → embed → upsert Qdrant
│   └── preview_chunks.py      #   Debug tool: export chunks ra file txt
│
├── retrieval/                 # Vector search layer
│   ├── embedder.py            #   Wrapper Google Generative AI Embeddings
│   └── qdrant_client.py       #   Qdrant client với retry & error handling
│
├── prompts/                   # Prompt templates
│   ├── hyde_prompt.py         #   Prompt tạo hypothetical document
│   └── answer_prompt.py       #   Prompt sinh câu trả lời cuối cùng
│
└── data/                      # Thư mục chứa tài liệu (gitignored)
```

## Cài đặt & Chạy

### Yêu cầu

- Python 3.11+
- Docker & Docker Compose
- Google Gemini API Key

### 1. Clone & tạo virtual environment

```bash
git clone <repo-url>
cd AIN501
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Khởi động Qdrant

```bash
docker compose up -d
```

Qdrant chạy tại `http://localhost:6333` với persistent storage qua Docker volume.

| Lệnh | Mô tả |
|-------|--------|
| `docker compose up -d` | Khởi động Qdrant |
| `docker compose down` | Dừng (giữ dữ liệu) |
| `docker compose down -v` | Dừng & xóa toàn bộ dữ liệu |

### 3. Cấu hình environment

```bash
cp .env.example .env
```

Sửa file `.env` và điền `GEMINI_API_KEY`:

```env
GEMINI_API_KEY=AIza...your-key-here
```

### 4. Nạp tài liệu (Ingestion)

Đặt file PDF/TXT/MD vào thư mục `data/`, sau đó:

```bash
python -m ingestion.ingest                    # Nạp toàn bộ thư mục data/
python -m ingestion.ingest path/to/file.pdf   # Nạp một file cụ thể
```

### 5. Chạy chatbot

**Web UI (khuyến nghị):**

```bash
streamlit run ui.py
```

**CLI:**

```bash
python main.py
```

## Cấu hình

Tất cả cấu hình được quản lý qua file `.env` và load bởi `pydantic-settings`:

| Biến | Mặc định | Mô tả |
|------|----------|-------|
| `GEMINI_API_KEY` | *(bắt buộc)* | API key Google Gemini |
| `LLM_MODEL` | `gemini-2.5-flash` | Model LLM dùng cho generation |
| `QDRANT_URL` | `http://localhost:6333` | URL kết nối Qdrant |
| `QDRANT_API_KEY` | *(trống)* | API key Qdrant (nếu có) |
| `COLLECTION_NAME` | `hyde_rag_kb` | Tên collection trong Qdrant |
| `EMBEDDING_MODEL` | `models/gemini-embedding-001` | Model embedding |
| `TOP_K` | `5` | Số lượng document trả về khi search |
| `SCORE_THRESHOLD` | `0.6` | Ngưỡng similarity tối thiểu |
| `MAX_CONTEXT_TOKENS` | `3000` | Giới hạn token cho context (ước lượng 1 token ~ 4 chars) |

## Sử dụng

### Web UI (Streamlit)

- **Upload tài liệu:** Kéo thả file PDF/TXT/MD vào sidebar → nhấn "Ingest into RAG"
- **Chat:** Nhập câu hỏi ở thanh chat phía dưới
- **Xem nguồn:** Mỗi câu trả lời có expander hiển thị nguồn tài liệu + relevance score
- **Sidebar:** Hiển thị collection stats, HyDE document preview, lịch sử nguồn
- **Fallback:** Nếu chưa có KB, chatbot tự động trả lời trực tiếp qua Gemini

### CLI Commands

| Lệnh | Mô tả |
|-------|--------|
| `/quit` | Thoát chatbot |
| `/clear` | Xóa lịch sử hội thoại |
| `/sources` | Hiển thị nguồn từ câu hỏi gần nhất |

### Debug Chunking

Kiểm tra kết quả chunking mà không cần embed/upsert:

```bash
python -m ingestion.preview_chunks data/sample.pdf
python -m ingestion.preview_chunks data/sample.pdf --out data/chunks_preview.txt
python -m ingestion.preview_chunks data/sample.pdf --max-chunks 10
```

## Ingestion Pipeline

Chi tiết quy trình nạp tài liệu:

```
File (PDF/TXT/MD)
    │
    ▼
Load documents (PyPDFLoader / TextLoader)
    │
    ▼
PDF text cleaning (gộp dòng, khôi phục đoạn văn)
    │
    ▼
Recursive chunking (chunk_size=1200, overlap=200)
    │
    ▼
Post-processing (xóa khoảng trắng thừa, lọc chunk < 50 ký tự)
    │
    ▼
Embed chunks (Google gemini-embedding-001)
    │
    ▼
Upsert vào Qdrant (batch size=100)
```

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `CHUNK_SIZE` | 1200 | Kích thước tối đa mỗi chunk (ký tự) |
| `CHUNK_OVERLAP` | 200 | Số ký tự overlap giữa các chunk liên tiếp |
| `MIN_CHUNK_LENGTH` | 50 | Chunk ngắn hơn sẽ bị loại bỏ |
| Separators | `\n\n`, `\n# `, `\n## `, `\n### `, `\n+ `, `\n- `, `\n* `, `\n`, ` ` | Thứ tự ưu tiên khi chia chunk |
| Định dạng hỗ trợ | `.pdf`, `.txt`, `.md` | — |

## Ví dụ minh họa

### CLI

```
============================================================
  HyDE RAG Chatbot  (type /quit to exit)
============================================================

You: What is the role of attention mechanism in transformers?

[HyDE] Hypothetical document generated (487 chars)

[HyDE Doc]: The attention mechanism is the fundamental building block of
the Transformer architecture, introduced in the seminal paper "Attention
Is All You Need" (Vaswani et al., 2017). Unlike recurrent neural networks
that process sequences step-by-step, the attention mechanism allows the
model to directly compute relationships between all positions in a se...

[Retrieved 3 source(s)]
  - transformer_architecture.pdf (score: 0.847)
  - attention_survey.pdf (score: 0.812)
  - deep_learning_basics.md (score: 0.734)