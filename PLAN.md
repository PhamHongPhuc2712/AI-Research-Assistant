# AI Research Assistant — Project Plan

## Overview
A full-stack AI-powered research assistant with two distinct modes:

- **Mode 1 — General Query:** User asks any question → system retrieves relevant document chunks via hybrid search → LLM synthesizes a grounded answer with citations.
- **Mode 2 — Paper Discovery:** User inputs a topic → system searches internal vector store + external sources (ArXiv API) → returns a ranked list of recommended papers to read, with summaries and relevance scores.

Built with LangChain, FastAPI, ChromaDB, and a Streamlit frontend.

---

## Repo Structure

```
ai-research-assistant/
│
├── backend/                        # FastAPI backend
│   ├── main.py                     # App entry point, registers all routers
│   ├── config.py                   # Environment variables, model names, paths
│   ├── dependencies.py             # Shared dependencies (DB client, LLM instance)
│   │
│   ├── routers/                    # API route handlers
│   │   ├── query.py                # POST /query — Mode 1: runs RAG chain, returns answer + sources
│   │   ├── papers.py               # POST /papers — Mode 2: takes a topic, returns ranked paper suggestions
│   │   └── ingest.py               # POST /ingest — accepts PDF/URL, processes into vector DB
│   │
│   ├── services/                   # Core business logic
│   │   ├── rag_chain.py            # LangChain RAG pipeline (retrieval + LLM synthesis) — used by Mode 1
│   │   ├── hybrid_search.py        # Combines ChromaDB (semantic) + BM25 (keyword) search
│   │   ├── paper_discovery.py      # Mode 2: queries ArXiv API + internal vectorstore, ranks + summarizes papers
│   │   ├── recommender.py          # Finds top-K related docs/topics after each query
│   │   ├── ingestion.py            # Loads, chunks, embeds, and stores documents
│   │   └── memory.py               # Manages conversation history for follow-up queries
│   │
│   ├── models/                     # Pydantic request/response schemas
│   │   ├── query.py                # QueryRequest, QueryResponse (Mode 1)
│   │   ├── papers.py               # PaperRequest, PaperResponse, PaperItem (Mode 2)
│   │   └── ingest.py               # IngestRequest, IngestResponse
│   │
│   └── tests/                      # Backend unit tests
│       ├── test_rag_chain.py
│       ├── test_hybrid_search.py
│       ├── test_paper_discovery.py
│       └── test_ingestion.py
│
├── frontend/                       # Streamlit UI
│   ├── app.py                      # Main entry point — renders mode toggle (Mode 1 / Mode 2)
│   ├── components/
│   │   ├── chat.py                 # Mode 1: Chat interface — renders messages, handles input
│   │   ├── sources.py              # Mode 1: Displays source documents with citations
│   │   ├── paper_list.py           # Mode 2: Renders ranked paper cards (title, authors, summary, relevance)
│   │   └── sidebar.py              # Shared: document upload widget + mode description
│   └── utils/
│       └── api_client.py           # HTTP client to call FastAPI backend (both modes)
│
├── data/                           # Data storage
│   ├── raw/                        # Original uploaded PDFs / raw text files
│   ├── processed/                  # Chunked text ready for embedding
│   └── vectorstore/                # ChromaDB persistent storage
│
├── scripts/                        # Standalone utility scripts
│   ├── ingest_bulk.py              # Bulk ingest a folder of PDFs at setup time
│   └── evaluate.py                 # Evaluate retrieval quality (precision, recall)
│
├── notebooks/                      # Jupyter notebooks for experimentation
│   ├── 01_chunking_strategies.ipynb
│   ├── 02_embedding_comparison.ipynb
│   └── 03_hybrid_search_tuning.ipynb
│
├── .env                            # API keys, model config (never commit this)
├── .env.example                    # Template for environment variables
├── requirements.txt                # Python dependencies
├── docker-compose.yml              # Runs backend + frontend together
├── Dockerfile.backend
├── Dockerfile.frontend
└── README.md
```

---

## File & Folder Descriptions

### `backend/`

| File | Purpose |
|---|---|
| `main.py` | Initializes FastAPI app, registers routers, sets up CORS |
| `config.py` | Loads `.env` vars: OpenAI key, model name, chunk size, ChromaDB path |
| `dependencies.py` | Creates shared singletons (LLM client, vector DB connection) injected via FastAPI `Depends()` |

#### `backend/routers/`
| File | Endpoint | Purpose |
|---|---|---|
| `query.py` | `POST /query` | **Mode 1** — receives user question, runs RAG chain, returns answer + sources |
| `papers.py` | `POST /papers` | **Mode 2** — receives a topic, queries ArXiv + vectorstore, returns ranked paper list |
| `ingest.py` | `POST /ingest` | Accepts file upload or URL, triggers ingestion pipeline |

#### `backend/services/`
| File | Purpose |
|---|---|
| `rag_chain.py` | **Mode 1** — LangChain chain: embeds query → retrieves chunks → builds prompt → calls LLM |
| `hybrid_search.py` | Runs semantic search (ChromaDB) + keyword search (BM25) in parallel, fuses via Reciprocal Rank Fusion |
| `paper_discovery.py` | **Mode 2** — takes a topic → queries ArXiv API + internal vectorstore → ranks results by relevance → generates short LLM summaries per paper |
| `recommender.py` | After a query, finds top-K semantically similar docs not in the current answer |
| `ingestion.py` | Loads PDF/text → splits into chunks → embeds → stores in ChromaDB |
| `memory.py` | Stores conversation history per session so follow-up questions retain context |

#### `backend/models/`
Pydantic schemas for request validation and response serialization. Keeps API contracts explicit and documented automatically in Swagger.

---

### `frontend/`

| File | Purpose |
|---|---|
| `app.py` | Entry point — renders a mode toggle (Mode 1: General Query / Mode 2: Paper Discovery), manages session state |
| `components/chat.py` | **Mode 1** — renders chat window, handles user input, displays streamed LLM responses |
| `components/sources.py` | **Mode 1** — shows retrieved source documents with page references below each answer |
| `components/paper_list.py` | **Mode 2** — renders ranked paper cards showing title, authors, abstract summary, and relevance score |
| `components/sidebar.py` | Shared — file upload button to add new documents, mode description |
| `utils/api_client.py` | Wraps all `httpx` calls to FastAPI — one function per endpoint (`/query`, `/papers`, `/ingest`) |

---

### `data/`

| Folder | Purpose |
|---|---|
| `raw/` | Original files as uploaded by the user — never modified |
| `processed/` | Text chunks with metadata (source file, page number, chunk index) |
| `vectorstore/` | ChromaDB files persisted to disk so data survives restarts |

---

### `scripts/`

| File | Purpose |
|---|---|
| `ingest_bulk.py` | One-time script to load an entire folder of PDFs into ChromaDB at project setup |
| `evaluate.py` | Runs retrieval benchmarks — measures how often the correct chunk is in the top-K results |

---

### `notebooks/`
Used during development to experiment before writing production code:
- `01` — compare chunking strategies (fixed size vs sentence vs paragraph)
- `02` — compare embedding models (OpenAI vs HuggingFace)
- `03` — tune hybrid search weights (how much to weight semantic vs keyword score)

---

## Key Dependencies (`requirements.txt`)

```
fastapi
uvicorn
langchain
langchain-openai
chromadb
rank-bm25
sentence-transformers
arxiv                   # ArXiv API client for Mode 2 paper discovery
streamlit
httpx
pydantic
pypdf
python-dotenv
```

---

## Environment Variables (`.env.example`)

```
OPENAI_API_KEY=your_key_here
LLM_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-ada-002
CHROMA_PATH=./data/vectorstore
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K_RETRIEVAL=5
```

---

## Implementation Order

1. **Setup** — repo structure, `.env`, install dependencies
2. **Ingestion pipeline** — `ingestion.py` + `ingest_bulk.py` + ChromaDB
3. **Mode 1: Basic RAG chain** — `rag_chain.py` + `POST /query` endpoint
4. **Hybrid search** — add BM25, fuse with semantic search in `hybrid_search.py`
5. **Mode 2: Paper discovery** — `paper_discovery.py` + ArXiv API + `POST /papers` endpoint
6. **Recommendation module** — `recommender.py` linked to both modes
7. **Conversation memory** — `memory.py` integrated into Mode 1 RAG chain
8. **Frontend** — mode toggle UI, wire Mode 1 chat + Mode 2 paper list to FastAPI
9. **Evaluation** — `evaluate.py` to measure retrieval quality for both modes
10. **Docker** — containerize for clean deployment