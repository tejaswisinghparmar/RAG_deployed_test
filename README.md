# 📚 PDF RAG Pipeline — Document Q&A with Qdrant + LangChain + Gemini

> **Domain-specific Retrieval-Augmented Generation (RAG)** pipeline that ingests PDF documents, indexes them in a Qdrant vector database, and answers natural-language questions with page-level citations — powered by Google Gemini and FastEmbed.

<!-- Replace with your own demo GIF / screenshot -->
<!-- ![Demo](docs/demo.gif) -->

---

## ✨ Key Features

| Feature | Details |
|---|---|
| **PDF Ingestion** | Automatic page-by-page loading via `PyPDFLoader` |
| **Smart Chunking** | Recursive character splitting (1 000 tokens, 400-token overlap) to preserve context across chunk boundaries |
| **Fast Embeddings** | `BAAI/bge-small-en-v1.5` (384-dim) via FastEmbed — lightweight, runs on CPU |
| **Vector Storage** | Qdrant (Cosine similarity) with Docker for easy setup & persistence |
| **LLM Generation** | Google Gemini (`gemini-2.5-flash-lite` / `gemini-2.0-flash-lite`) with grounded, citation-aware prompts |
| **Two Chat Modes** | Single-query mode (`chat.py`) and interactive multi-turn loop (`chat_autorun.py`) |
| **Page Citations** | Every answer references the source page number so you can verify in the original PDF |

---

## 🏗️ Architecture

```
┌──────────┐    ┌────────────┐    ┌──────────────┐    ┌────────┐    ┌────────────┐
│  PDF Doc  │──▶│  Chunking   │──▶│  FastEmbed    │──▶│ Qdrant  │    │   Gemini   │
│ (PyPDF)   │   │ (Recursive) │   │ bge-small-en  │   │ VectorDB│    │    LLM     │
└──────────┘    └────────────┘    └──────────────┘    └───┬────┘    └─────┬──────┘
                                                          │               │
                                                          ▼               │
                                                   ┌────────────┐        │
                                          Query ──▶│  Retriever  │───────▶│
                                                   │  (top-k=4)  │  context  ──▶ Answer
                                                   └────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Framework | LangChain |
| Vector DB | Qdrant (Docker) |
| Embeddings | FastEmbed (`BAAI/bge-small-en-v1.5`, 384-dim) |
| LLM | Google Gemini (Flash Lite) |
| PDF Loader | PyPDF |
| Environment | python-dotenv |

---

## 📂 Project Structure

```
.
├── rag/
│   ├── index.py              # Ingestion — load PDF, chunk, embed, store in Qdrant
│   ├── chat.py               # Single-query RAG chat
│   ├── chat_autorun.py       # Interactive multi-turn RAG chat loop
│   └── docker-compose.yml    # One-command Qdrant setup
├── .env.example              # Template for required environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Setup & Run

### Prerequisites

- **Python 3.10+**
- **Docker** (for Qdrant)
- A **Google Gemini API key** — get one free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 1. Clone the repo

```bash
git clone https://github.com/tejaswisinghparmar/RAG.git
cd RAG
```

### 2. Create & activate a virtual environment

```bash
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

```bash
cp .env.example rag/.env
```

Open `rag/.env` and paste your Google API key:

```
GOOGLE_API_KEY="your_actual_key_here"
```

### 5. Start Qdrant (vector database)

```bash
cd rag
docker compose up -d
```

Qdrant dashboard will be available at **http://localhost:6333/dashboard**.

### 6. Index your PDF

Place your PDF file in the `rag/` folder (update `PDF_FILENAME` in `index.py` if needed), then:

```bash
python rag/index.py
```

### 7. Chat with your document

**Single question:**
```bash
python rag/chat.py
```

**Interactive session (multi-turn):**
```bash
python rag/chat_autorun.py
```

---

## 💡 How It Works

1. **Ingestion (`index.py`)** — The PDF is loaded page-by-page, split into overlapping chunks of ~1 000 tokens, embedded with `bge-small-en-v1.5`, and stored in a Qdrant collection.
2. **Retrieval** — When the user asks a question, the query is embedded and the top-k most similar chunks are retrieved from Qdrant using cosine similarity.
3. **Generation** — The retrieved chunks (with page numbers) are injected into a system prompt, and Google Gemini generates a grounded answer with page citations.

---

## 🧠 Challenges & Learnings

- **Chunking strategy matters** — Recursive splitting with 400-token overlap significantly improved retrieval accuracy for questions spanning two pages.
- **FastEmbed vs cloud embeddings** — Switched from cloud-based embedding APIs to FastEmbed for zero-cost, offline-capable, and faster indexing on CPU.
- **Prompt engineering for citation** — Explicitly instructing the LLM to cite page numbers reduced hallucinated answers and improved verifiability.

---

## 🔮 Future Scope

- **Agentic RAG** — Add tool-calling to let the LLM decide when to search, summarise, or ask for clarification.
- **Hybrid Retrieval** — Combine dense (vector) + sparse (BM25) search for better recall.
- **Reranking** — Add a cross-encoder reranker (e.g., `ms-marco-MiniLM`) to improve precision after retrieval.
- **Streamlit / Gradio UI** — Web interface for a more polished demo experience.
- **Multi-document support** — Ingest multiple PDFs and filter by source at retrieval time.
- **Evaluation with RAGAS** — Measure faithfulness, answer relevance, and context precision.

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).

---

> Built with ❤️ using LangChain, Qdrant, FastEmbed, and Google Gemini.
