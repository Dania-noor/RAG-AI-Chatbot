from S1_DocumentIngestion import load_documents
from langchain_text_splitters import RecursiveCharacterTextSplitter


documents = load_documents()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("=" * 60)
print("TEXT SPLITTING")
print("=" * 60)

print("Total chunks:", len(chunks))

print("\nFirst chunk:")
print(chunks[0].page_content)

print("\nMetadata:")
print(chunks[0].metadata)