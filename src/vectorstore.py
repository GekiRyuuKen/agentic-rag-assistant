import chromadb
from sentence_transformers import SentenceTransformer
from ingest import load_documents, chunk_documents, filter_noise

# Load embedding model - runs on CPU to keep GPU free for Ollama/LLama
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

def build_vectorstore(chunks, persist_dir="chroma_db", collection_name="rag_papers"):
    """Embed chunks and store them in a persistent ChromaDB collection."""
    client = chromadb.PersistentClient(path=persist_dir)

    # Fresh start each time - delete old collection if it exists
    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)

    collection = client.create_collection(collection_name)

    texts = [chunk["text"] for chunk in chunks]
    ids = [chunk["chunk_id"] for chunk in chunks]
    metadatas = [{"source": chunk["source"]} for chunk in chunks]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True).tolist()

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    print(f"Stored {collection.count()} chunks in ChromaDB collection '{collection_name}'.")
    return collection

def query_vectorstore(query, persist_dir="chroma_db", collection_name="rag_papers", n_results=3):
    """Test retrieval - embed a query and find the most similar chunks."""
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_collection(collection_name)

    query_embedding = embedding_model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)
    return results

if __name__ == "__main__":
    docs = load_documents()
    chunks = chunk_documents(docs)
    chunks = filter_noise(chunks)

    build_vectorstore(chunks)

    # Quick retrieval test
    test_query = "What is the difference between naive RAG and agentic RAG?"
    print(f"\n--- Test query: '{test_query}' ---")
    results = query_vectorstore(test_query)

    for i, doc in enumerate(results["documents"][0]):
        source = results["metadatas"][0][i]["source"]
        print(f"\n[Result {i+1}] Source: {source}")
        print(doc[:300])