"""
app.py — Inkwell: AI-Powered PDF Q&A

Upload any PDF and chat with your document instantly.
Everything runs in-memory — no external database needed.
Uses HuggingFace's free Inference API.

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
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
TOP_K = 4

LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"
HF_TOKEN = st.secrets["HF_TOKEN"]

# ── Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Inkwell",
    page_icon="🖋️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Dark Aesthetic Theme ───────────────────────────────
st.markdown("""
<style>
    /* ── Import font ─────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:ital,wght@1,400;1,600&display=swap');

    /* ── Base ────────────────────────────────────────────────────── */
    .stApp {
        background: linear-gradient(160deg, #0d0d0d 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar — hover to reveal when collapsed ────────────────── */
    section[data-testid="stSidebar"] {
        background: rgba(13, 13, 13, 0.95);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
        transition: transform 0.3s ease, opacity 0.3s ease;
    }
    section[data-testid="stSidebar"][aria-expanded="false"] {
        transform: translateX(-100%);
        opacity: 0;
    }
    section[data-testid="stSidebar"][aria-expanded="false"]:hover {
        transform: translateX(0);
        opacity: 1;
    }
    /* Invisible hover trigger strip on the left edge */
    section[data-testid="stSidebar"]::before {
        content: "";
        position: fixed;
        left: 0;
        top: 0;
        width: 18px;
        height: 100vh;
        z-index: 999;
    }

    /* ── Header ──────────────────────────────────────────────────── */
    .main-header {
        text-align: center;
        padding: 2.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-family: 'Playfair Display', serif;
        font-style: italic;
        font-weight: 600;
        font-size: 3rem;
        background: linear-gradient(135deg, #e0c3fc 0%, #8ec5fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.3rem;
        letter-spacing: 0.02em;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.45);
        font-size: 0.95rem;
        font-weight: 300;
        margin-top: 0;
        letter-spacing: 0.03em;
    }

    /* ── Welcome card ────────────────────────────────────────────── */
    .setup-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        backdrop-filter: blur(10px);
    }
    .setup-card h3 {
        color: #e0e0e0;
        font-weight: 500;
        margin-top: 0;
        font-size: 1.15rem;
    }

    /* ── Chat messages ───────────────────────────────────────────── */
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
        padding: 0.8rem 0 !important;
    }

    /* ── Chat input ──────────────────────────────────────────────── */
    .stChatInput > div {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
    }
    .stChatInput textarea {
        color: #e0e0e0 !important;
    }

    /* ── Buttons ─────────────────────────────────────────────────── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 500 !important;
        letter-spacing: 0.03em;
        transition: all 0.3s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(118, 75, 162, 0.4) !important;
    }
    .stButton > button {
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        color: #c0c0c0 !important;
        transition: all 0.2s ease;
    }

    /* ── File uploader ───────────────────────────────────────────── */
    section[data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px dashed rgba(255, 255, 255, 0.12);
        border-radius: 12px;
        padding: 0.5rem;
    }

    /* ── Expander (sources) ──────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 8px !important;
        color: rgba(255, 255, 255, 0.6) !important;
        font-size: 0.85rem !important;
    }

    /* ── Divider ─────────────────────────────────────────────────── */
    hr {
        border-color: rgba(255, 255, 255, 0.06) !important;
    }

    /* ── Scrollbar ───────────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

    /* ── Progress bar ────────────────────────────────────────────── */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2) !important;
    }

    /* ── Caption ─────────────────────────────────────────────────── */
    .stCaption { color: rgba(255,255,255,0.35) !important; }

    /* ── Hide Streamlit chrome ───────────────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }

    /* ── Sidebar text colors ─────────────────────────────────────── */
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: rgba(255, 255, 255, 0.55);
        font-size: 0.88rem;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: rgba(255, 255, 255, 0.8);
        font-size: 0.95rem;
        font-weight: 500;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached Embedding Model ─────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model (first time only)...")
def get_embedding_model():
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)


# ── PDF Processing with Progress ───────────────────────────────────
def process_pdf(uploaded_file, embedding_model):
    """Load PDF -> chunk -> embed -> return in-memory vector store."""
    progress = st.progress(0, text="Loading PDF...")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    docs = PyPDFLoader(file_path=tmp_path).load()
    Path(tmp_path).unlink(missing_ok=True)
    progress.progress(20, text=f"Loaded {len(docs)} pages. Chunking...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    progress.progress(40, text=f"Created {len(chunks)} chunks. Embedding...")

    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        location=":memory:",
        collection_name="uploaded_pdf",
    )
    progress.progress(100, text="Done!")
    return vector_store, len(docs), len(chunks)


# ── RAG Pipeline ────────────────────────────────────────────────────
def retrieve_and_generate(query: str, vector_db, model_id: str):
    """Retrieve -> Prompt -> Generate using HuggingFace Inference API."""
    results = vector_db.similarity_search(query=query, k=TOP_K)

    context_parts, sources = [], []
    for r in results:
        page = r.metadata.get("page_label", r.metadata.get("page", "N/A"))
        context_parts.append(f"Page Content: {r.page_content}\nPage Number: {page}")
        sources.append(f"Page {page}")

    context = "\n\n".join(context_parts)

    system_msg = (
        "You are a helpful AI Assistant. Answer the user's question ONLY "
        "based on the context below. If the context doesn't contain the "
        "answer, say you don't know. Always cite the relevant Page Number."
    )
    user_msg = f"Context:\n{context}\n\nUser Query: {query}"

    client = InferenceClient(token=HF_TOKEN)
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
    <h1>Inkwell</h1>
    <p>Upload a PDF and start a conversation with your document</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Upload Document")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        help="Processed in-memory only — never stored.",
        label_visibility="collapsed",
    )

    if uploaded_file:
        is_new_file = uploaded_file.name != st.session_state.pdf_name

        if is_new_file:
            if st.button("Process & Index", use_container_width=True, type="primary"):
                embedding_model = get_embedding_model()
                vector_db, pages, chunks = process_pdf(uploaded_file, embedding_model)
                st.session_state.vector_db = vector_db
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.page_count = pages
                st.session_state.chunk_count = chunks
                st.session_state.messages = []
                st.success(f"Indexed **{pages}** pages into **{chunks}** chunks")
        else:
            st.success(f"**{uploaded_file.name}** ready ({st.session_state.page_count} pages)")

    st.divider()

    st.markdown(
        """
        ### About
        **Inkwell** answers questions from your PDF
        with page-level citations.

        Your data stays safe — processed in-memory,
        nothing saved after you close the tab.

        **Stack:** Qdrant / HuggingFace / FastEmbed / LangChain
        """
    )

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main Area ───────────────────────────────────────────────────────
if not st.session_state.vector_db:
    st.markdown("""
    <div class="setup-card">
        <h3>Welcome — get started in one step</h3>
        <ol style="color: rgba(255,255,255,0.5); line-height: 2.2; padding-left: 1.2rem;">
            <li>Upload a <strong style="color:rgba(255,255,255,0.75);">PDF</strong> in the sidebar and click <strong style="color:rgba(255,255,255,0.75);">Process</strong></li>
        </ol>
        <p style="color: rgba(255,255,255,0.3); font-size: 0.82rem; margin-bottom: 0; margin-top: 1rem;">
            100% free — no sign-up, no API keys, no data stored.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Chat Interface ──────────────────────────────────────────────
    st.caption(f"Chatting with {st.session_state.pdf_name}")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🖋️"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    st.markdown(" · ".join(msg["sources"]))

    if prompt := st.chat_input("Ask something about your document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🖋️"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = retrieve_and_generate(
                        prompt, st.session_state.vector_db, LLM_MODEL
                    )
                    st.markdown(answer)
                    if sources:
                        with st.expander("Sources"):
                            st.markdown(" · ".join(sources))
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer, "sources": sources}
                    )
                except Exception as e:
                    err = str(e)
                    if "401" in err or "Unauthorized" in err or "Invalid" in err:
                        error_msg = "Authentication error. Please try again later."
                    elif "loading" in err.lower() or "unavailable" in err.lower():
                        error_msg = "Model is waking up (~30s). Try again in a moment."
                    else:
                        error_msg = f"Something went wrong: {err}\n\nPlease try again."
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )
