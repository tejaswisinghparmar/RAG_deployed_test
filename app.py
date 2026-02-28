"""
app.py — Streamlit ChatGPT-style RAG Interface

Upload any PDF, enter your free Gemini API key, and chat with your
document. Everything runs in-memory — no external database needed.

Deploy for free on Streamlit Community Cloud.

Usage (local):
    streamlit run app.py
"""

import tempfile
import time
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Constants ───────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 400
TOP_K = 4
MAX_RETRIES = 3
RETRY_DELAYS = [5, 15, 30]  # seconds to wait between retries

AVAILABLE_MODELS = {
    "gemini-2.0-flash": "Fast & capable (15 RPM free)",
    "gemini-1.5-flash": "Stable & reliable (15 RPM free)",
    "gemini-1.5-flash-8b": "Lightweight (30 RPM free)",
    "gemini-2.0-flash-lite": "Cheapest (lower free quota)",
}

# ── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind RAG",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #212121; }

    .main-header { text-align: center; padding: 1.2rem 0 0.3rem 0; }
    .main-header h1 { color: #ECECEC; font-size: 1.8rem; font-weight: 600; margin-bottom: 0.2rem; }
    .main-header p  { color: #9A9A9A; font-size: 0.95rem; margin-top: 0; }

    .setup-card {
        background: #2A2A2A; border: 1px solid #3A3A3A; border-radius: 12px;
        padding: 1.5rem; margin: 1rem 0;
    }
    .setup-card h3 { color: #ECECEC; margin-top: 0; }

    .stChatMessage { background-color: transparent !important; border: none !important; padding: 0.8rem 0 !important; }

    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Cached Embedding Model (shared across sessions) ────────────────
@st.cache_resource(show_spinner="⏳ Loading embedding model (first time only)...")
def get_embedding_model():
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


# ── PDF Processing ──────────────────────────────────────────────────
def process_pdf(uploaded_file, embedding_model):
    """Load PDF -> chunk -> embed -> return in-memory vector store."""
    # Save to temp file (PyPDFLoader needs a file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    # Load
    docs = PyPDFLoader(file_path=tmp_path).load()

    # Clean up temp file
    Path(tmp_path).unlink(missing_ok=True)

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    # Embed & store in-memory (no external DB needed)
    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        location=":memory:",
        collection_name="uploaded_pdf",
    )
    return vector_store, len(docs), len(chunks)


# ── RAG Pipeline ────────────────────────────────────────────────────
def retrieve_and_generate(query: str, vector_db, api_key: str, model: str):
    """Retrieve -> Prompt -> Generate with auto-retry on rate limits."""
    # 1. Retrieve
    results = vector_db.similarity_search(query=query, k=TOP_K)

    # 2. Build context
    context_parts, sources = [], []
    for r in results:
        page = r.metadata.get("page_label", r.metadata.get("page", "N/A"))
        context_parts.append(f"Page Content: {r.page_content}\nPage Number: {page}")
        sources.append(f"Page {page}")

    context = "\n\n".join(context_parts)

    # 3. Prompt
    system_prompt = f"""You are a helpful AI Assistant. Answer the user's
question ONLY based on the context below. If the context doesn't contain
the answer, say you don't know. Always cite the relevant Page Number.

Context:
{context}

User Query: {query}"""

    # 4. Generate with retry
    llm = ChatGoogleGenerativeAI(
        model=model, temperature=0.3, google_api_key=api_key
    )

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = llm.invoke(system_prompt)
            return response.content, list(dict.fromkeys(sources))
        except Exception as e:
            err = str(e)
            last_error = e
            if "RESOURCE_EXHAUSTED" in err or "429" in err:
                if attempt < MAX_RETRIES - 1:
                    delay = RETRY_DELAYS[attempt]
                    st.toast(f"⏳ Rate limited — retrying in {delay}s (attempt {attempt + 2}/{MAX_RETRIES})...")
                    time.sleep(delay)
                    continue
            else:
                raise  # non-rate-limit errors propagate immediately

    raise last_error  # all retries exhausted


# ── Session State Init ──────────────────────────────────────────────
for key, default in {
    "messages": [],
    "vector_db": None,
    "pdf_name": None,
    "page_count": 0,
    "chunk_count": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ── Header ──────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📚 DocMind RAG</h1>
    <p>Upload a PDF, paste your free Gemini API key, and start chatting with your document</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar — Setup Panel ──────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Setup")

    # API Key input
    api_key = st.text_input(
        "🔑 Google Gemini API Key",
        type="password",
        placeholder="Paste your API key here",
        help="Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)",
    )

    # Model selector
    selected_model = st.selectbox(
        "🧠 Gemini Model",
        options=list(AVAILABLE_MODELS.keys()),
        format_func=lambda m: f"{m}  —  {AVAILABLE_MODELS[m]}",
        index=0,
        help="If one model hits rate limits, switch to another — each has its own quota.",
    )

    st.divider()

    # PDF upload
    uploaded_file = st.file_uploader(
        "📄 Upload a PDF",
        type=["pdf"],
        help="Your file is processed in-memory and never stored on any server.",
    )

    # Process button
    if uploaded_file and api_key:
        is_new_file = uploaded_file.name != st.session_state.pdf_name

        if is_new_file:
            if st.button("⚡ Process & Index PDF", use_container_width=True, type="primary"):
                embedding_model = get_embedding_model()
                with st.spinner(f"Processing **{uploaded_file.name}**..."):
                    vector_db, pages, chunks = process_pdf(uploaded_file, embedding_model)
                    st.session_state.vector_db = vector_db
                    st.session_state.pdf_name = uploaded_file.name
                    st.session_state.page_count = pages
                    st.session_state.chunk_count = chunks
                    st.session_state.messages = []  # reset chat for new doc
                st.success(f"✅ Indexed **{pages}** pages into **{chunks}** chunks")
        else:
            st.success(f"✅ **{uploaded_file.name}** ready ({st.session_state.page_count} pages)")

    elif not api_key:
        st.info("👆 Paste your free [Gemini API key](https://aistudio.google.com/apikey) to get started.")

    st.divider()

    # About section
    st.markdown(
        """
        ### ℹ️ About
        **DocMind RAG** answers questions from your PDF
        with page-level citations.

        **Your data is safe:**
        - PDF is processed in-memory only
        - API key is never stored
        - Nothing is saved after you close the tab

        **Tech Stack:**
        🔍 Qdrant (in-memory) · 🧠 Gemini · ⚡ FastEmbed · 🐍 LangChain
        """
    )

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main Area ───────────────────────────────────────────────────────
if not api_key or not st.session_state.vector_db:
    # Onboarding state
    st.markdown("""
    <div class="setup-card">
        <h3>👋 Welcome! Get started in 2 steps:</h3>
        <ol style="color: #B0B0B0; line-height: 2;">
            <li>Paste your <strong>free</strong> <a href="https://aistudio.google.com/apikey" target="_blank">Google Gemini API key</a> in the sidebar</li>
            <li>Upload a <strong>PDF</strong> file and click <strong>Process</strong></li>
        </ol>
        <p style="color: #777; font-size: 0.85rem; margin-bottom: 0;">
            🔒 Your API key and document stay in your browser session — nothing is stored.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Chat Interface ──────────────────────────────────────────────
    st.caption(f"💬 Chatting with **{st.session_state.pdf_name}**")

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 Sources"):
                    st.markdown(" · ".join(msg["sources"]))

    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        # Assistant response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = retrieve_and_generate(
                        prompt, st.session_state.vector_db, api_key, selected_model
                    )
                    st.markdown(answer)
                    if sources:
                        with st.expander("📄 Sources"):
                            st.markdown(" · ".join(sources))
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as e:
                    err = str(e)
                    if "API_KEY_INVALID" in err or "PERMISSION_DENIED" in err:
                        error_msg = "❌ Invalid API key. Please check your Gemini API key in the sidebar."
                    elif "RESOURCE_EXHAUSTED" in err or "429" in err:
                        error_msg = (
                            f"⏳ Rate limit exhausted for **{selected_model}** after {MAX_RETRIES} retries.\n\n"
                            "**Try one of these:**\n"
                            "- Switch to a **different model** in the sidebar (each has its own quota)\n"
                            "- Wait a minute and try again\n"
                            "- If daily quota is hit, try again tomorrow or use a different API key"
                        )
                    else:
                        error_msg = f"⚠️ Error: {err}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )
