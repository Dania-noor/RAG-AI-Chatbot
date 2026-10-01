from S2_TextSplitting import chunks
from S3_Embeddings import embeddings
from langchain_chroma import Chroma


print("Loading ChromaDB...")

vectorstore = Chroma(
    collection_name="rag_documents",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)


# =========================
# USER QUESTION
# =========================

query = "What is RAG?"


# =========================
# RETRIEVAL
# =========================

print("Searching for relevant information...")

results = vectorstore.similarity_search(
    query,
    k=3
)


# =========================
# RESULTS
# =========================

print("=" * 60)
print("RETRIEVAL RESULTS")
print("=" * 60)

print("Question:", query)

for i, document in enumerate(results):

    print(f"\n--- Result {i + 1} ---")
    print(document.page_content)
    print("\nSource:", document.metadata.get("source"))