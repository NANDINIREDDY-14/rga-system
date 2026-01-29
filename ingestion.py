from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from vector_store import VectorStore
import os

model = SentenceTransformer("all-MiniLM-L6-v2")
store = VectorStore()

CHUNK_SIZE = 500
OVERLAP = 50

def read_file(path):
    if path.endswith(".txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    elif path.endswith(".pdf"):
        reader = PdfReader(path)
        return " ".join(page.extract_text() or "" for page in reader.pages)

def chunk_text(text):
    words = text.split()
    chunks = []

    for i in range(0, len(words), CHUNK_SIZE - OVERLAP):
        chunk = " ".join(words[i:i + CHUNK_SIZE])
        chunks.append(chunk)

    return chunks

def ingest_document(path):
    text = read_file(path)
    chunks = chunk_text(text)
    embeddings = model.encode(chunks)
    store.add(embeddings, chunks)
