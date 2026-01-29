import faiss
import pickle
import os
import numpy as np

INDEX_FILE = "faiss.index"
DATA_FILE = "chunks.pkl"

class VectorStore:
    def __init__(self, dim=384):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.text_chunks = []

        if os.path.exists(INDEX_FILE):
            self.index = faiss.read_index(INDEX_FILE)
            with open(DATA_FILE, "rb") as f:
                self.text_chunks = pickle.load(f)

    def add(self, embeddings, chunks):
        self.index.add(np.array(embeddings).astype("float32"))
        self.text_chunks.extend(chunks)

        faiss.write_index(self.index, INDEX_FILE)
        with open(DATA_FILE, "wb") as f:
            pickle.dump(self.text_chunks, f)

    def search(self, query_embedding, top_k=3):
        D, I = self.index.search(
            np.array([query_embedding]).astype("float32"), top_k
        )
        return [self.text_chunks[i] for i in I[0]]
