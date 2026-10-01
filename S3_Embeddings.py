from S2_TextSplitting import chunks
from langchain_huggingface import HuggingFaceEmbeddings


embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector = embeddings.embed_query(chunks[0].page_content)

print("=" * 60)
print("EMBEDDINGS")
print("=" * 60)

print("Embedding dimension:", len(vector))
print("First 10 values:", vector[:10])