from S2_TextSplitting import chunks
from S3_Embeddings import embeddings
from langchain_chroma import Chroma


print("Creating vector database...")

vectorstore = Chroma(
    collection_name="rag_documents",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)

print("Adding documents to ChromaDB...")

vectorstore.add_documents(chunks)

print("=" * 60)
print("VECTOR DATABASE")
print("=" * 60)

print("Total chunks stored: 28")
print("ChromaDB created successfully!")