import re
import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = "data"

def trim_references(text):
    """Cut text at the References/Bibliography section, wherever it starts."""
    search_start = len(text) // 2  # only look in the back half, avoids false matches mid-paper
    tail = text[search_start:]

    match = re.search(r"\bREFERENCES\b|\bReferences\b|\bBibliography\b", tail)
    if match:
        cut_point = search_start + match.start()
        return text[:cut_point]
    return text

def is_likely_navigation(text, max_avg_line_length=25):
    """Detect nav-menu-style chunks: many short lines (letter-spaced sidebar text)."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 3:
        return False
    avg_len = sum(len(l) for l in lines) / len(lines)
    return avg_len < max_avg_line_length

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
            text = trim_references(text)
            documents.append({"source": filename, "text": text})
            print(f"Loaded {filename} - {len(text)} characters")
    return documents

def chunk_with_parents(documents, parent_chunk_size=2000, parent_overlap=200, child_chunk_size=500, child_overlap=50):
    """Split each document into large parent chunks, then each parent into small child chunks for embedding."""
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=parent_chunk_size,
        chunk_overlap=parent_overlap
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=child_chunk_size,
        chunk_overlap=child_overlap
    )

    parent_store = {}
    child_chunks = []

    for doc in documents:
        parents = parent_splitter.split_text(doc["text"])
        for p_i, parent_text in enumerate(parents):
            parent_id = f"{doc['source']}_parent_{p_i}"
            parent_store[parent_id] = {
                "text": parent_text,
                "source": doc["source"]
            }
            children = child_splitter.split_text(parent_text)
            for c_i, child_text in enumerate(children):
                child_chunks.append({
                    "source": doc["source"],
                    "chunk_id": f"{parent_id}_child_{c_i}",
                    "text": child_text,
                    "parent_id": parent_id
                })

    return parent_store, child_chunks

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
        if is_likely_navigation(chunk["text"]):
            continue # likely navigation menu
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