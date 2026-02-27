"""
chat.py — Single-query RAG Chat

Retrieves relevant chunks from Qdrant and generates an answer
using Google Gemini. Accepts one question, prints the answer, then exits.

Usage:
    python rag/chat.py
"""

import os

from dotenv import load_dotenv
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore

# ── Configuration ───────────────────────────────────────────────────
COLLECTION_NAME = "harry_potter_fast"
QDRANT_URL = "http://localhost:6333"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
LLM_MODEL = "gemini-2.5-flash-lite"
TOP_K = 4  # number of chunks to retrieve

load_dotenv()


def main() -> None:
    # 1. Connect to Qdrant
    embedding_model = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
    vector_db = QdrantVectorStore.from_existing_collection(
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model,
    )

    # 2. Accept user query
    user_query = input("Ask the expert:: ")

    # 3. Retrieve relevant chunks
    search_results = vector_db.similarity_search(query=user_query, k=TOP_K)

    context = "\n\n".join(
        f"Page Content: {r.page_content}\n"
        f"Page Number: {r.metadata.get('page_label', 'N/A')}"
        for r in search_results
    )

    # 4. Build prompt & generate answer
    system_prompt = f"""
You are a helpful AI Assistant who answers user queries based on the
available context retrieved from a PDF file.

Rules:
- Answer ONLY based on the context below.
- If the context doesn't contain the answer, say you don't know.
- Always cite the relevant Page Number so the user can look it up.

Context:
{context}

User Query: {user_query}
"""

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    response = llm.invoke(system_prompt)
    print(f"\n🤖: {response.content}")


if __name__ == "__main__":
    main()