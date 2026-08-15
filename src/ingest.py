import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = "data"

def load_documents(data_dir=DATA_DIR):
    """Load all PDFs from the data folder and extract their text."""
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".pdf"):
            filepath = os.path.join(data_dir, filename)
            reader = PdfReader(filepath)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            documents.append({"source": filename, "text": text})
            print(f"Loaded {filename} - {len(text)} characters")
    return documents

def chunk_documents(documents, chunk_size=500, chunk_overlap=50):
    """Split each document's text into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    all_chunks = []
    for doc in documents:
        chunks = splitter.split_text(doc["text"])
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "source": doc["source"],
                "chunk_id": f"{doc['source']}_{i}",
                "text": chunk
            })
    return all_chunks

def filter_noise(chunks, min_length=100):
    """Remove chunks that are likely navigation/boilerplate noise."""
    noise_keywords = ["cookie", "privacy notice", "search docs", "all rights reserved", "was this doc helpful", "edit this doc", "copyright ©", "terms of service", "dmca"]
    filtered = []
    for chunk in chunks:
        text_lower = chunk["text"].lower()
        if len(chunk["text"]) < min_length:
            continue # too short, likely nav/TOC fragment
        if any(keyword in text_lower for keyword in noise_keywords):
            continue # likely boilerplate
        filtered.append(chunk)
    return filtered

if __name__ == "__main__":
    docs = load_documents()
    print(f"\nLoaded {len(docs)} documents total.\n")

    chunks = chunk_documents(docs)
    chunks = filter_noise(chunks)
    print(f"Split into {len(chunks)} chunks, {len(chunks)} after filtering noise.\n")

    # Print a few chunks from different points to check quality
    if chunks:
        print("\n--- Sample chunks from different points ---")
        sample_indices = [0, len(chunks)//4, len(chunks)//2, (3*len(chunks))//4, len(chunks)-1]
        for idx in sample_indices:
            print(f"\n[Chunk {idx}] Source: {chunks[idx]['source']}")
            print(chunks[idx]['text'][:250])