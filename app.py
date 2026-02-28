"""
app.py — Streamlit ChatGPT-style RAG Interface

Upload any PDF, enter your free HuggingFace token, and chat with your
document. Everything runs in-memory — no external database needed.
Uses HuggingFace's free Inference API — no rate limit worries.

Deploy for free on Streamlit Community Cloud.

Usage (local):
    streamlit run app.py
"""

import tempfile
from pathlib import Path

import streamlit as st
from huggingface_hub import InferenceClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Constants ───────────────────────────────────────────────────────
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE = 1500       # larger chunks = fewer embeddings = faster indexing
CHUNK_OVERLAP = 300
TOP_K = 4

AVAILABLE_MODELS = {
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral 7B — Fast & good",
    "HuggingFaceH4/zephyr-7b-beta": "Zephyr 7B — Conversational",
    "microsoft/Phi-3-mini-4k-instruct": "Phi-3 Mini — Compact & smart",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen 2.5 7B — Multilingual",
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


# ── Cached Embedding Model ─────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading embedding model (first time only)...")
def get_embedding_model():
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


# ── PDF Processing with Progress ───────────────────────────────────
def process_pdf(uploaded_file, embedding_model):
    """Load PDF -> chunk -> embed -> return in-memory vector store."""
    progress = st.progress(0, text="Loading PDF...")

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    # Load
    docs = PyPDFLoader(file_path=tmp_path).load()
    Path(tmp_path).unlink(missing_ok=True)
    progress.progress(20, text=f"Loaded {len(docs)} pages. Chunking...")

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    progress.progress(40, text=f"Created {len(chunks)} chunks. Embedding...")

    # Embed & store in-memory
    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        location=":memory:",
        collection_name="uploaded_pdf",
    )
    progress.progress(100, text="✅ Done!")
    return vector_store, len(docs), len(chunks)


# ── RAG Pipeline ────────────────────────────────────────────────────
def retrieve_and_generate(query: str, vector_db, hf_token: str, model_id: str):
    """Retrieve -> Prompt -> Generate using HuggingFace Inference API."""
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
    system_msg = (
        "You are a helpful AI Assistant. Answer the user's question ONLY "
        "based on the context below. If the context doesn't contain the "
        "answer, say you don't know. Always cite the relevant Page Number."
    )
    user_msg = f"Context:\n{context}\n\nUser Query: {query}"

    # 4. Generate via HuggingFace Inference API
    client = InferenceClient(token=hf_token)
    response = client.chat_completion(
        model=model_id,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=1024,
        temperature=0.3,
    )
    answer = response.choices[0].message.content
    return answer, list(dict.fromkeys(sources))


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
    <p>Upload a PDF, paste your free HuggingFace token, and chat with your document</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔧 Setup")

    # Token input
    hf_token = st.text_input(
        "🔑 HuggingFace Token",
        type="password",
        placeholder="hf_xxxxxxxxxxxxxxxxxxxx",
        help="Get a free token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)",
    )

    # Model selector
    selected_model = st.selectbox(
        "🧠 Model",
        options=list(AVAILABLE_MODELS.keys()),
        format_func=lambda m: AVAILABLE_MODELS[m],
        index=0,
        help="All models are free. Switch if one is slow.",
    )

    st.divider()

    # PDF upload
    uploaded_file = st.file_uploader(
        "📄 Upload a PDF",
        type=["pdf"],
        help="Processed in-memory only — never stored.",
    )

    # Process button
    if uploaded_file and hf_token:
        is_new_file = uploaded_file.name != st.session_state.pdf_name

        if is_new_file:
            if st.button("⚡ Process & Index PDF", use_container_width=True, type="primary"):
                embedding_model = get_embedding_model()
                vector_db, pages, chunks = process_pdf(uploaded_file, embedding_model)
                st.session_state.vector_db = vector_db
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.page_count = pages
                st.session_state.chunk_count = chunks
                st.session_state.messages = []
                st.success(f"✅ Indexed **{pages}** pages → **{chunks}** chunks")
        else:
            st.success(f"✅ **{uploaded_file.name}** ready ({st.session_state.page_count} pages)")

    elif not hf_token:
        st.info("👆 Paste your free [HuggingFace token](https://huggingface.co/settings/tokens) to start.")

    st.divider()

    st.markdown(
        """
        ### ℹ️ About
        **DocMind RAG** answers questions from your PDF
        with page-level citations.

        **Your data is safe:**
        - PDF processed in-memory only
        - Token never stored
        - Nothing saved after you close

        **Tech Stack:**
        🔍 Qdrant · 🧠 HuggingFace · ⚡ FastEmbed · 🐍 LangChain
        """
    )

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main Area ───────────────────────────────────────────────────────
if not hf_token or not st.session_state.vector_db:
    st.markdown("""
    <div class="setup-card">
        <h3>👋 Welcome! Get started in 2 steps:</h3>
        <ol style="color: #B0B0B0; line-height: 2;">
            <li>Paste your <strong>free</strong> <a href="https://huggingface.co/settings/tokens" target="_blank">HuggingFace token</a> in the sidebar</li>
            <li>Upload a <strong>PDF</strong> file and click <strong>Process</strong></li>
        </ol>
        <p style="color: #777; font-size: 0.85rem; margin-bottom: 0;">
            🔒 100% free — no credit card, no rate limits, no data stored.<br>
            💡 Get your token: <a href="https://huggingface.co/settings/tokens" target="_blank">huggingface.co/settings/tokens</a> → New token → Read access
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Chat Interface ──────────────────────────────────────────────
    st.caption(f"💬 Chatting with **{st.session_state.pdf_name}** · Model: `{selected_model.split('/')[-1]}`")

    # Display history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 Sources"):
                    st.markdown(" · ".join(msg["sources"]))

    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = retrieve_and_generate(
                        prompt, st.session_state.vector_db, hf_token, selected_model
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
                    if "401" in err or "Unauthorized" in err or "Invalid" in err:
                        error_msg = "❌ Invalid token. Check your HuggingFace token in the sidebar."
                    elif "loading" in err.lower() or "unavailable" in err.lower():
                        error_msg = "⏳ Model is waking up (~30s). Try again in a moment or switch models."
                    else:
                        error_msg = f"⚠️ Error: {err}\n\nTry switching to a different model in the sidebar."
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )
