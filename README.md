# 📚 DocMind RAG — Chat with Any PDF using AI

> **Upload any PDF and ask questions** — get accurate, page-cited answers powered by Retrieval-Augmented Generation (RAG) with HuggingFace LLMs, FastEmbed, and in-memory Qdrant.

<!-- Add your demo GIF/screenshot here after deploying -->
<!-- ![Demo](docs/demo.gif) -->

**🔗 [Live Demo](https://ragdeployedtest.streamlit.app/) · [Get Free HuggingFace Token](https://huggingface.co/settings/tokens)**

---

## ✨ Key Features

| Feature | Details |
|---|---|
| **Upload Any PDF** | Users upload their own PDF — no pre-indexed data needed |
| **BYOK (Bring Your Own Key)** | Each user provides their own free Gemini API key — your API key stays safe |
| **In-Memory Processing** | PDF is chunked, embedded, and stored in-memory — nothing is saved after the session |
| **Smart Chunking** | Recursive splitting (1 000 tokens, 400-token overlap) preserves context across pages |
| **Fast Embeddings** | `BAAI/bge-small-en-v1.5` (384-dim) via FastEmbed — runs on CPU, zero API cost |
| **Page Citations** | Every answer references the exact page number for easy verification |
| **ChatGPT-style UI** | Clean, dark-themed chat interface built with Streamlit |
| **Privacy-first** | No data stored, no API keys saved — everything dies when you close the tab |
| **CLI Tools** | Bonus: CLI scripts for local batch indexing and terminal-based chat |

---

## 🏗️ Architecture

```
                          ┌─────────────────────────────────────────────┐
                          │            Streamlit Web App                │
                          └─────────────────────────────────────────────┘
                                            │
                     ┌──────────────────────┼──────────────────────┐
                     ▼                      ▼                      ▼
              ┌────────────┐      ┌──────────────┐       ┌──────────────┐
  User ──▶   │  Upload PDF │      │  User Query   │       │  Gemini Key  │
              └─────┬──────┘      └──────┬───────┘       └──────┬───────┘
                    │                    │                       │
                    ▼                    ▼                       │
           ┌──────────────┐    ┌──────────────────┐             │
           │  PyPDF Load   │    │     Embed Query   │             │
           │  + Chunking   │    │  (FastEmbed/CPU)  │             │
           └──────┬───────┘    └────────┬─────────┘             │
                  │                     │                       │
                  ▼                     ▼                       │
           ┌──────────────┐    ┌──────────────────┐             │
           │   FastEmbed   │    │  Cosine Search    │             │
           │  (bge-small)  │    │  Qdrant In-Memory │             │
           └──────┬───────┘    └────────┬─────────┘             │
                  │                     │                       │
                  ▼                     ▼                       ▼
           ┌──────────────┐    ┌──────────────────────────────────┐
           │ Qdrant Store  │    │  System Prompt + Context + Query │
           │  (in-memory)  │    │         → Google Gemini LLM      │
           └──────────────┘    └──────────────┬───────────────────┘
                                              │
                                              ▼
                                     ┌──────────────┐
                                     │   Answer +    │
                                     │  Page Cited   │
                                     └──────────────┘
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Framework | LangChain |
| Frontend | Streamlit (ChatGPT-style UI) |
| Vector DB | Qdrant (in-memory for web app / Docker for CLI) |
| Embeddings | FastEmbed — `BAAI/bge-small-en-v1.5` (384-dim, CPU) |
| LLM | HuggingFace Inference API (Mistral 7B, Zephyr 7B, Phi-3, Qwen 2.5) |
| PDF Loader | PyPDF |

---

## 📂 Project Structure

```
.
├── app.py                    # 🌐 Streamlit web UI (upload PDF + chat)
├── rag/
│   ├── index.py              # CLI: Ingest PDF → chunk → embed → Qdrant
│   ├── chat.py               # CLI: Single-query RAG chat
│   ├── chat_autorun.py       # CLI: Interactive multi-turn chat loop
│   └── docker-compose.yml    # One-command Qdrant setup (for CLI mode)
├── .streamlit/
│   └── config.toml           # Streamlit theme (dark mode)
├── .env.example              # Template for environment variables
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### Option A: Web UI (Recommended)

```bash
git clone https://github.com/tejaswisinghparmar/RAG.git
cd RAG
python -m venv venv && source venv/bin/activate  # Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open **http://localhost:8501** → paste your [free Gemini API key](https://aistudio.google.com/apikey) → upload a PDF → start chatting!

### Option B: CLI Mode (with Docker Qdrant)

```bash
# 1. Start Qdrant
cd rag && docker compose up -d && cd ..

# 2. Configure
cp .env.example rag/.env
# Edit rag/.env with your GOOGLE_API_KEY

# 3. Index a PDF
python rag/index.py

# 4. Chat
python rag/chat_autorun.py
```

---

## ☁️ Free Deployment (Streamlit Community Cloud)

The web app needs **zero external services** — no database, no paid APIs.

| What | Where | Cost |
|---|---|---|
| Web App | [Streamlit Community Cloud](https://streamlit.io/cloud) | Free (public repos) |
| LLM | Users bring their own [Gemini key](https://aistudio.google.com/apikey) | Free for users |
| Vector DB | In-memory (no setup needed) | Free |
| Embeddings | FastEmbed (runs on CPU) | Free |

### Deploy in 3 Steps

1. **Push to GitHub** — Make sure your repo is public
2. **Go to [share.streamlit.io](https://share.streamlit.io/)** → Connect your GitHub repo → Set main file to `app.py`
3. **Done!** Share the URL on your resume. No secrets needed — users bring their own key.

---

## 🔒 Security & Privacy

- **BYOK Model** — Users enter their own Gemini API key. Your key is never exposed.
- **No Persistent Storage** — PDFs are processed in-memory and discarded when the session ends.
- **API Keys Not Stored** — Keys exist only in the browser session state.
- **`.gitignore` Protection** — `.env` files and PDFs are excluded from version control.

---

## 💡 How It Works

1. **Upload** — User uploads a PDF via the Streamlit sidebar.
2. **Chunk** — The PDF is split into overlapping chunks (~1 000 tokens each, 400-token overlap) to preserve context across page boundaries.
3. **Embed** — Each chunk is embedded using `bge-small-en-v1.5` (384-dim vectors) running locally on CPU via FastEmbed.
4. **Store** — Vectors are stored in an in-memory Qdrant instance (no external database).
5. **Retrieve** — When the user asks a question, the query is embedded and the top-4 most similar chunks are retrieved via cosine similarity.
6. **Generate** — Retrieved chunks + page numbers are injected into a system prompt, and Google Gemini generates a grounded, cited answer.

---

## 🧠 Challenges & Learnings

- **Chunking strategy matters** — Recursive splitting with 400-token overlap significantly improved retrieval accuracy for questions spanning multiple pages.
- **FastEmbed vs cloud embeddings** — Switched from cloud-based embedding APIs to FastEmbed for zero-cost, offline-capable, and faster indexing on CPU.
- **BYOK for free deployment** — Instead of burning through a shared API quota, letting users bring their own key makes the app sustainably free.
- **Prompt engineering for citations** — Explicitly instructing the LLM to cite page numbers reduced hallucinated answers and improved verifiability.
- **In-memory Qdrant** — Using `:memory:` mode eliminates the need for an external database in deployment while keeping the same LangChain API.

---

## 🔮 Future Scope

- **Agentic RAG** — Add tool-calling to let the LLM decide when to search, summarise, or ask for clarification
- **Hybrid Retrieval** — Combine dense (vector) + sparse (BM25) search for better recall
- **Reranking** — Cross-encoder reranker (e.g., `ms-marco-MiniLM`) for improved precision
- **Multi-document support** — Upload multiple PDFs and filter by source at retrieval time
- **Streaming responses** — Token-by-token output for a more responsive feel
- **Evaluation with RAGAS** — Measure faithfulness, answer relevance, and context precision
- **Chat history export** — Download conversation as PDF/Markdown

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).

---

> Built with ❤️ using LangChain, Qdrant, FastEmbed, HuggingFace, and Streamlit.
