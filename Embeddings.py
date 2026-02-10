# Embeddings.py
"""
Builds the ChromaDB Vector Store from enriched documents.

Pipeline:
1. Loads `enriched_documents.json`.
2. Splits documents into chunks (512 chars).
3. Serializes metadata (belief states) for storage.
4. Generates embeddings using `intfloat/e5-large-v2`.
5. Persists to `schopenhauer_vector_db`.
"""
import json
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

# --- Configuration ---
INPUT_FILE = "enriched_documents.json"
VECTORSTORE_DIR = "schopenhauer_vector_db"
EMBEDDING_MODEL = "intfloat/e5-large-v2"

# --- Main Execution Logic ---
if __name__ == "__main__":
    # 1. Load the enriched documents
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            docs_from_json = json.load(f)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{INPUT_FILE}'.")
        print("Please run '02_enrich_documents.py' first.")
        exit()

    # Convert dictionaries back into LangChain Document objects
    enriched_docs = [
        Document(page_content=d['metadata']['heading']+d['page_content'],
                  metadata=d['metadata'])
        for d in docs_from_json
    ]
    print(f"Loaded {len(enriched_docs)} enriched documents from '{INPUT_FILE}'.")

    # 2. Split the documents into smaller chunks
    # These parameters are a good starting point for retrieval.
    final_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,  # The size of each chunk in characters
        chunk_overlap=100  # The number of characters to overlap between chunks
    )
    final_chunks = final_splitter.split_documents(enriched_docs)

    print(f"Split {len(enriched_docs)} documents into {len(final_chunks)} final chunks.")

    # **FIX**: Convert complex metadata (dicts) into JSON strings for ChromaDB.
    print("Serializing complex metadata for ChromaDB compatibility...")
    for chunk in final_chunks:
        if 'belief_state' in chunk.metadata and isinstance(chunk.metadata['belief_state'], dict):
            # json.dumps converts a Python dictionary to a JSON formatted string
            chunk.metadata['belief_state'] = json.dumps(chunk.metadata['belief_state'])

    # You can inspect a chunk's metadata to see that it was preserved
    if final_chunks:
        print("\n--- Example Final Chunk Metadata (After Serialization) ---")
        print(final_chunks[0].metadata)
        print("----------------------------------------------------------\n")

    # 3. Initialize the embedding model
    # This uses the same model you originally chose.
    print(f"Initializing embedding model: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"} # Use "cpu" if you don't have a CUDA-enabled GPU
    )

    # 4. Create and persist the vector store
    if os.path.exists(VECTORSTORE_DIR):
        print(f"Vector store directory '{VECTORSTORE_DIR}' already exists. Skipping creation.")
        print("If you want to recreate it, please delete the directory first.")
    else:
        print(f"Creating vector store in '{VECTORSTORE_DIR}'... (This may take a while)")
        vectorstore = Chroma.from_documents(
            documents=final_chunks,
            embedding=embeddings,
            persist_directory=VECTORSTORE_DIR
        )
        print("Successfully created and persisted the vector store.")

    print("\n searchable vector database is ready.")

