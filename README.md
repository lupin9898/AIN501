# HyDE RAG Chatbot

A production-ready Retrieval-Augmented Generation chatbot using the **HyDE (Hypothetical Document Embedding)** pattern, built with LangGraph + LangChain + Qdrant.

## What is HyDE?

Traditional RAG embeds the user's **question** and searches for similar chunks. The problem: a short question ("What causes inflation?") lives in a very different region of embedding space than the answer passage it's looking for.

HyDE solves this by first asking an LLM to **imagine** what a perfect answer would look like, then embedding *that* synthetic document. Since both the synthetic doc and the real KB chunks are answer-style prose, cosine similarity works much better.

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

## HyDE LangGraph Workflow

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
│  │ generate_hypothetical_doc │  LLM imagines ideal answer       │
│  └──────────────────────────┘                                   │
│    │                                                            │
│    ▼                                                            │
│  ┌──────────────────────┐                                       │
│  │ embed_hypothetical    │  Embed the HYPOTHETICAL doc          │
│  └──────────────────────┘  (NOT the original query!)            │
│    │                                                            │
│    ▼                                                            │
│  ┌──────────────────────┐                                       │
│  │ retrieve_documents    │  Search Qdrant with HyDE vector      │
│  └──────────────────────┘                                       │
│    │                                                            │
│    ▼                                                            │
│  ┌──────────────────────┐                                       │
│  │ assemble_context      │  Sort by score, enforce token budget  │
│  └──────────────────────┘                                       │
│    │                                                            │
│    ▼                                                            │
│  ┌──────────────────────┐                                       │
│  │ generate_answer       │  LLM answers from REAL docs only     │
│  └──────────────────────┘                                       │
│    │                                                            │
│    ▼                                                            │
│   END                                                           │
└─────────────────────────────────────────────────────────────────┘
```

## Setup

### 1. Start Qdrant

```bash
docker compose up -d
```

This starts Qdrant with persistent storage via `docker-compose.yml`. Data survives container restarts.

To stop: `docker compose down` (data is kept in the `qdrant_data` volume).
To wipe data: `docker compose down -v`.

### 2. Install dependencies

```bash
cd hyde_rag
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

### 4. Ingest documents

Place your documents (PDF, TXT, MD) into the `data/` folder, then:

```bash
python -m ingestion.ingest
```

### 5. Run the chatbot

**Option A — Web UI (recommended):**

```bash
streamlit run ui.py
```

Features:
- Drag-and-drop file upload (PDF, TXT, MD) with one-click ingestion
- Streaming token-by-token responses
- Real-time HyDE pipeline progress indicator
- Expandable source citations per message
- Sidebar with collection stats, HyDE doc preview, and source history
- Multi-turn conversation memory

**Option B — CLI:**

```bash
python main.py
```

## CLI Commands

| Command    | Description                        |
|------------|------------------------------------|
| `/quit`    | Exit the chatbot                   |
| `/clear`   | Reset conversation history         |
| `/sources` | Show sources from the last query   |

## Example Session

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