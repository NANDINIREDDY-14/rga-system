from fastapi import FastAPI, UploadFile, BackgroundTasks
from ingestion import ingest_document
from sentence_transformers import SentenceTransformer
from vector_store import VectorStore
import shutil
import os
import time

app = FastAPI()

UPLOAD_DIR = "uploaded_docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")
store = VectorStore()

@app.post("/upload")
async def upload_document(
    file: UploadFile,
    background_tasks: BackgroundTasks
):
    file_path = f"{UPLOAD_DIR}/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(ingest_document, file_path)

    return {"message": "Document uploaded and processing started"}

@app.post("/ask")
async def ask_question(question: str):
    start_time = time.time()

    query_embedding = model.encode(question)
    context_chunks = store.search(query_embedding)

    context = "\n".join(context_chunks)

    answer = f"""
Based on the uploaded documents:

{context}

Answer:
The information relevant to your question is provided above.
"""

    latency = time.time() - start_time

    return {
        "answer": answer.strip(),
        "latency_seconds": round(latency, 2)
    }
