# RAG Setup (Production-Ready, Minimal, Clean)

## 1. Project Structure

Base directory:
C:\Users\91638\Desktop\AI_LEARNING\RAG

Final structure:

RAG/
│
├── app/
│   ├── ingestion/
│   │   └── loader.py
│   ├── retrieval/
│   │   └── retriever.py
│   ├── chains/
│   │   └── rag_chain.py
│   ├── config/
│   │   └── settings.py
│   └── main.py
│
├── data/
│   └── documents/   (move your docs here)
│
├── .env
├── pyproject.toml
└── README.md

---

## 2. Install uv (if not installed)

pip install uv

Verify:
uv --version

---

## 3. Initialize project

cd C:\Users\91638\Desktop\AI_LEARNING\RAG

uv init

---

## 4. Dependencies (clean, correct)

Run:

uv add \
langchain \
langchain-openai \
langchain-community \
langgraph \
qdrant-client \
tiktoken \
python-dotenv \
docling \
pymupdf \
rank-bm25 \
pandas

---

## Why these only?

REMOVED:
- groq → not needed
- chromadb → replaced with qdrant
- faiss → not needed now
- sentence-transformers → using OpenAI embeddings
- unstructured/pdfminer → replaced by docling

---

## 5. Environment Variables

Create `.env` file:

OPENAI_API_KEY=your_openai_key
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=rag_collection

---

## 6. Run Qdrant (Docker)

docker run -p 6333:6333 qdrant/qdrant

Check:
http://localhost:6333/dashboard

---

## 7. Ingestion Pipeline

app/ingestion/loader.py

```python
from docling.document_converter import DocumentConverter
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_documents(path):
    converter = DocumentConverter()
    docs = converter.convert(path)

    texts = [d.text for d in docs]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    return splitter.create_documents(texts)