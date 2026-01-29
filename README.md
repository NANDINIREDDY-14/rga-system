## RAG-Based Question Answering System

### Setup
pip install -r requirements.txt
uvicorn main:app --reload

### Usage
1. Upload PDF/TXT document via /upload
2. Ask questions via /ask
3. Receive answers based on document content

### Tech Stack
- FastAPI
- FAISS
- Sentence Transformers
- Python

### Metrics
Latency measured for query response
