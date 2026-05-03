# RAG Learning Roadmap (Production-Level)

## 0. What is RAG (Real Definition)

RAG = Retrieval + Reasoning + Generation

Pipeline:
User Query  
→ Query Processing  
→ Retrieval  
→ Context Selection  
→ Prompt Construction  
→ LLM Generation  
→ Post-processing  

---

## 1. Types of RAG

### 1.1 Naive RAG
query → embedding → vector search → top-k → LLM

Pros:
- Simple
- Fast to build

Cons:
- Low accuracy
- Hallucinations

---

### 1.2 Advanced RAG

Adds:
- Query rewriting
- Reranking
- Metadata filtering

Flow:
query → rewrite → retrieve → rerank → compress → LLM

---

### 1.3 Hybrid RAG

Dense + Sparse retrieval

---

### 1.4 Graph RAG

Used for relationships

Example:
Dish → Ingredient → Cuisine → Region

---

### 1.5 Agentic RAG

LLM decides:
- what to retrieve
- when to retrieve
- which tool to use

---

### 1.6 Multi-hop RAG

Used for complex queries

---

### 1.7 Context Compression RAG

Reduce tokens by extracting only relevant parts

---

### 1.8 Real-time RAG

Used in:
- News
- Stocks
- Live systems

---

## 2. Chunking Strategies

### 2.1 Basic Chunking
Split by tokens or characters

### 2.2 Recursive Chunking
Uses multiple separators to split text

### 2.3 Semantic Chunking
Split based on meaning using embeddings

### 2.4 Sliding Window
Overlapping chunks improve recall

### 2.5 Metadata Chunking
Attach metadata like cuisine, ingredient

### 2.6 Parent-Child Chunking
Small chunks for retrieval, large for context

---

## 3. Core Tech Stack

Frameworks:
- LangChain
- LangGraph

Vector DB:
- Qdrant

Observability:
- LangSmith

Deployment:
- Docker
- AWS
- Render

---

## 4. Minimal RAG Code

### Embedding + Store

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant

embeddings = OpenAIEmbeddings()

db = Qdrant.from_texts(
    texts=chunks,
    embedding=embeddings,
    url="http://localhost:6333",
    collection_name="food_data"
)
```

---

### Retrieval

```python
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 5}
)
```

---

### Prompt

```python
from langchain.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("""
You are a food expert.

Context:
{context}

Question:
{question}
""")
```

---

### Chain

```python
from langchain.chains import RetrievalQA

chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

response = chain.run("High protein Indian dishes?")
```

---

## 5. Interview Q&A

1. What is RAG?
2. Why not fine-tuning?
3. What are embeddings?
4. What is cosine similarity?
5. Chunk size trade-offs?
6. What is top-k?
7. What is reranking?
8. What is hybrid search?
9. Why hallucinations happen?
10. How to evaluate RAG?

---

## 6. Learning Phases

Phase 1:
- Build naive RAG

Phase 2:
- Add hybrid search
- Add reranker

Phase 3:
- Build LangGraph agent

---

## 7. Project Mapping

Menu → Dish → Ingredients → Nutrition → Region

Build:
- Ingredient-aware RAG
- Cuisine reasoning agent
- Nutrition filtering system
