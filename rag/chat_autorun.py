"""
chat_autorun.py — Interactive Multi-turn RAG Chat Loop

Runs a persistent chat session: retrieves relevant chunks from Qdrant
for every query and generates answers using Google Gemini.
Type 'exit' / 'quit' / 'q' to stop.

Usage:
    python rag/chat_autorun.py
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
LLM_MODEL = "gemini-2.0-flash"
TOP_K = 3  # number of chunks to retrieve per query

load_dotenv()


def main() -> None:
    # Initialise models & DB connection once
    embedding_model = FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)

    vector_db = QdrantVectorStore.from_existing_collection(
        url=QDRANT_URL,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model,
    )

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0.3,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )

    print("--- 📚 RAG Chatbot Active (type 'exit' to stop) ---")

    while True:
        user_query = input("\nAsk the expert:: ").strip()

        if not user_query:
            continue
        if user_query.lower() in ("exit", "quit", "q"):
            print("Goodbye! 🤖")
            break

        # 1. Retrieve
        search_results = vector_db.similarity_search(query=user_query, k=TOP_K)

        # 2. Build context
        context = "\n\n".join(
            f"Page Content: {r.page_content}\n"
            f"Page Number: {r.metadata.get('page_label', 'N/A')}"
            for r in search_results
        )

        # 3. Prompt
        system_prompt = f"""
You are a helpful AI Assistant. Answer the user query ONLY based on the
context below. If the context doesn't have the answer, say you don't know.
Always mention the Page Number if available.

Context:
{context}

User Query: {user_query}
"""

        # 4. Generate
        try:
            response = llm.invoke(system_prompt)
            print(f"\n🤖: {response.content}")
        except Exception as e:
            print(f"⚠️  Error: {e}. Please try again in a few seconds.")


if __name__ == "__main__":
    main()