# app/retrieval/retriever.py
# -----------------------------------------------------------
# Production-grade Qdrant retriever
# Stack: Qdrant (vector DB) + OpenAI embeddings + LangChain
# Features:
#   - Collection creation with cosine distance
#   - Batch upsert with rich metadata payloads
#   - Similarity search (top-k)
#   - MMR search (Max Marginal Relevance — diversity-aware)
#   - Similarity search with scores
#   - Payload filtering (query by filename)
#   - Collection health check
#   - Interactive query loop for validation
# -----------------------------------------------------------

from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

from dotenv import load_dotenv
import os
import uuid
import sys

# Load .env from project root (two levels up from retrieval/)
load_dotenv(
    dotenv_path=os.path.join(
        os.path.dirname(__file__), "..", "..", ".env"
    )
)

# -----------------------------------------------------------
# CONFIG
# -----------------------------------------------------------
QDRANT_URL      = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY", None)   # None for local Docker
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "rag_collection")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIM   = 1536   # text-embedding-3-small = 1536 dims
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise EnvironmentError(
        "OPENAI_API_KEY not found in environment.\n"
        "Add it to your .env file: OPENAI_API_KEY=sk-..."
    )


# -----------------------------------------------------------
# CLIENTS
# -----------------------------------------------------------
def get_qdrant_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=30)


def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=EMBEDDING_MODEL, openai_api_key=OPENAI_API_KEY)


# -----------------------------------------------------------
# COLLECTION MANAGEMENT
# -----------------------------------------------------------
def create_collection_if_not_exists(client: QdrantClient, collection_name: str) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if collection_name in existing:
        print(f"  [Qdrant] Collection '{collection_name}' already exists — skipping creation")
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"  [Qdrant] ✅ Collection '{collection_name}' created (dim={EMBEDDING_DIM}, cosine)")


def delete_collection(client: QdrantClient, collection_name: str) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if collection_name not in existing:
        print(f"  [Qdrant] Collection '{collection_name}' does not exist — nothing to delete")
        return
    client.delete_collection(collection_name)
    print(f"  [Qdrant] 🗑️  Collection '{collection_name}' deleted")


def collection_info(client: QdrantClient, collection_name: str) -> dict:
    info = client.get_collection(collection_name)
    return {
        "name":         collection_name,
        "status":       info.status,
        "vector_count": info.points_count,
        "dimension":    info.config.params.vectors.size,
        "distance":     info.config.params.vectors.distance,
    }


# -----------------------------------------------------------
# UPSERT PIPELINE
# -----------------------------------------------------------
def upsert_documents(
    chunks: list[Document],
    collection_name: str = COLLECTION_NAME,
    batch_size: int = 100,
    recreate: bool = False,
) -> int:
    """
    Embeds and upserts LangChain Document chunks into Qdrant.

    Args:
        chunks:          list[Document] from loader.load_documents()
        collection_name: target Qdrant collection
        batch_size:      chunks per OpenAI embedding API call
        recreate:        if True, wipes and recreates the collection first

    Under the hood:
        1. Get/create Qdrant collection
        2. Batch chunks → OpenAI embeddings API → 1536-dim vectors
        3. Wrap as PointStruct (id, vector, payload)
        4. client.upsert() — idempotent, wait=True for consistency
    """
    if not chunks:
        raise ValueError("No chunks provided to upsert_documents()")

    client     = get_qdrant_client()
    embeddings = get_embeddings()

    if recreate:
        delete_collection(client, collection_name)
    create_collection_if_not_exists(client, collection_name)

    total_upserted = 0

    for batch_start in range(0, len(chunks), batch_size):
        batch       = chunks[batch_start : batch_start + batch_size]
        batch_num   = batch_start // batch_size + 1
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        print(f"  [Upsert] Batch {batch_num}/{total_batches} — embedding {len(batch)} chunks...")

        texts   = [doc.page_content for doc in batch]
        vectors = embeddings.embed_documents(texts)

        points = []
        for doc, vector in zip(batch, vectors):
            payload = {
                "text":        doc.page_content,
                "source":      doc.metadata.get("source", ""),
                "filename":    doc.metadata.get("filename", ""),
                "toc_entries": doc.metadata.get("toc_entries", 0),
                "toc":         doc.metadata.get("toc", []),
            }
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload,
                )
            )

        client.upsert(collection_name=collection_name, points=points, wait=True)
        total_upserted += len(points)
        print(f"  [Upsert] ✅ Batch {batch_num} done ({total_upserted}/{len(chunks)} total)")

    print(f"\n✅ Upsert complete — {total_upserted} vectors in '{collection_name}'\n")
    return total_upserted


# -----------------------------------------------------------
# VECTOR STORE
# -----------------------------------------------------------
def get_vector_store(collection_name: str = COLLECTION_NAME) -> QdrantVectorStore:
    """
    Returns LangChain QdrantVectorStore connected to existing collection.
    Used by all search methods and get_retriever().

    Under the hood:
        Wraps QdrantClient + OpenAIEmbeddings.
        .similarity_search()  → embed query → HNSW ANN → top-k Documents
        .as_retriever()       → BaseRetriever for LCEL chains
    """
    return QdrantVectorStore(
        client=get_qdrant_client(),
        collection_name=collection_name,
        embedding=get_embeddings(),
        content_payload_key="text",
        metadata_payload_key=None,
    )


# -----------------------------------------------------------
# SEARCH METHODS
# -----------------------------------------------------------
def similarity_search(
    query: str,
    k: int = 5,
    collection_name: str = COLLECTION_NAME,
    filter_filename: str | None = None,
) -> list[Document]:
    """
    Standard cosine similarity search — returns top-k most relevant chunks.

    Under the hood:
        query → embed (1536-dim) → Qdrant HNSW ANN → cosine nearest neighbors → Documents

    When to use:
        Simple Q&A, fact lookups, single-topic questions.
    """
    store = get_vector_store(collection_name)

    qdrant_filter = None
    if filter_filename:
        qdrant_filter = Filter(
            must=[FieldCondition(key="filename", match=MatchValue(value=filter_filename))]
        )

    return store.similarity_search(query=query, k=k, filter=qdrant_filter)


def mmr_search(
    query: str,
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    collection_name: str = COLLECTION_NAME,
    filter_filename: str | None = None,
) -> list[Document]:
    """
    Max Marginal Relevance (MMR) — balances relevance AND diversity.

    Under the hood:
        1. Fetch fetch_k candidates by cosine similarity
        2. Iteratively pick next doc maximizing:
               score = lambda * sim(doc, query) - (1-lambda) * max(sim(doc, selected))
        3. Returns k diverse-yet-relevant docs

    Args:
        lambda_mult: 0.0 = max diversity, 1.0 = max relevance, 0.5 = balanced

    When to use:
        Multi-topic questions, risk factors, summaries — avoids repetitive chunks.
    """
    store = get_vector_store(collection_name)

    qdrant_filter = None
    if filter_filename:
        qdrant_filter = Filter(
            must=[FieldCondition(key="filename", match=MatchValue(value=filter_filename))]
        )

    return store.max_marginal_relevance_search(
        query=query,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult,
        filter=qdrant_filter,
    )


def similarity_search_with_score(
    query: str,
    k: int = 5,
    collection_name: str = COLLECTION_NAME,
) -> list[tuple[Document, float]]:
    """
    Returns (Document, cosine_score) tuples.
    Score range: 0.0 (unrelated) → 1.0 (identical).
    Use for debugging retrieval quality and building re-rankers.
    """
    store = get_vector_store(collection_name)
    return store.similarity_search_with_score(query=query, k=k)


# -----------------------------------------------------------
# LANGCHAIN RETRIEVER (for LCEL chains)
# -----------------------------------------------------------
def get_retriever(
    search_type: str = "similarity",
    k: int = 5,
    fetch_k: int = 20,
    lambda_mult: float = 0.5,
    collection_name: str = COLLECTION_NAME,
):
    """
    Returns LangChain BaseRetriever for use in LCEL chains.

    Usage in rag_chain.py:
        retriever = get_retriever(search_type="mmr", k=5)
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt | llm | StrOutputParser()
        )

    Under the hood:
        as_retriever() wraps QdrantVectorStore in VectorStoreRetriever.
        Chain calls retriever.invoke(query) → list[Document].
    """
    store = get_vector_store(collection_name)
    search_kwargs = {"k": k}

    if search_type == "mmr":
        search_kwargs["fetch_k"]     = fetch_k
        search_kwargs["lambda_mult"] = lambda_mult

    return store.as_retriever(search_type=search_type, search_kwargs=search_kwargs)


# -----------------------------------------------------------
# HEALTH CHECK
# -----------------------------------------------------------
def health_check(collection_name: str = COLLECTION_NAME) -> None:
    try:
        client = get_qdrant_client()
        info   = collection_info(client, collection_name)
        print(f"\n========== COLLECTION HEALTH ==========")
        print(f"  Name:      {info['name']}")
        print(f"  Status:    {info['status']}")
        print(f"  Vectors:   {info['vector_count']}")
        print(f"  Dimension: {info['dimension']}")
        print(f"  Distance:  {info['distance']}")
        print(f"=======================================\n")
    except Exception as e:
        print(f"  [Health] ❌ Qdrant unreachable: {e}")
        print(f"  Run: docker start qdrant_local")


# -----------------------------------------------------------
# DISPLAY HELPERS
# -----------------------------------------------------------
def _print_separator(label: str = "") -> None:
    width = 58
    if label:
        side = (width - len(label) - 2) // 2
        print(f"\n{'='*side} {label} {'='*side}")
    else:
        print("=" * width)


def _display_results(results: list[Document], max_chars: int = 400) -> None:
    if not results:
        print("  ⚠️  No results returned.")
        return
    for i, doc in enumerate(results, 1):
        print(f"\n  --- Result {i} ---")
        print(f"  File: {doc.metadata.get('filename', 'unknown')}")
        print(f"  Text:\n  {doc.page_content[:max_chars]}")
        if len(doc.page_content) > max_chars:
            print(f"  ... [{len(doc.page_content) - max_chars} more chars]")


def _display_scored_results(
    results: list[tuple[Document, float]], max_chars: int = 300
) -> None:
    if not results:
        print("  ⚠️  No results returned.")
        return
    for i, (doc, score) in enumerate(results, 1):
        bar = "█" * int(score * 20)
        print(f"\n  --- Result {i} | Score: {score:.4f} {bar}")
        print(f"  File: {doc.metadata.get('filename', 'unknown')}")
        print(f"  Text:\n  {doc.page_content[:max_chars]}")


# -----------------------------------------------------------
# INTERACTIVE QUERY LOOP
# -----------------------------------------------------------
def interactive_query_loop() -> None:
    """
    Interactive REPL for testing retrieval quality.
    Commands:
        1  → Similarity search
        2  → MMR search
        3  → Similarity with scores
        h  → Health check
        q  → Quit
    """
    print("\n" + "=" * 58)
    print("  🔍  RAG RETRIEVAL TESTER — Interactive Mode")
    print("=" * 58)
    print("  Commands:")
    print("    1  → Similarity search")
    print("    2  → MMR search (diversity-aware)")
    print("    3  → Similarity search with scores")
    print("    h  → Collection health check")
    print("    q  → Quit")
    print("=" * 58)

    while True:
        try:
            cmd = input("\n  Enter command [1/2/3/h/q]: ").strip().lower()

            if cmd == "q":
                print("\n  👋 Exiting retrieval tester.")
                break

            elif cmd == "h":
                health_check()

            elif cmd in ("1", "2", "3"):
                query = input("  Enter your query: ").strip()
                if not query:
                    print("  ⚠️  Query cannot be empty.")
                    continue

                k_input = input("  Number of results [default=5]: ").strip()
                k = int(k_input) if k_input.isdigit() else 5

                if cmd == "1":
                    _print_separator("SIMILARITY SEARCH")
                    print(f"  Query : {query}")
                    print(f"  Top-k : {k}")
                    results = similarity_search(query, k=k)
                    _display_results(results)

                elif cmd == "2":
                    fetch_input  = input("  fetch_k (candidate pool) [default=20]: ").strip()
                    lambda_input = input("  lambda_mult [0=diverse → 1=relevant, default=0.5]: ").strip()
                    fetch_k      = int(fetch_input) if fetch_input.isdigit() else 20
                    try:
                        lambda_mult = float(lambda_input) if lambda_input else 0.5
                        lambda_mult = max(0.0, min(1.0, lambda_mult))
                    except ValueError:
                        lambda_mult = 0.5

                    _print_separator("MMR SEARCH")
                    print(f"  Query      : {query}")
                    print(f"  k          : {k}")
                    print(f"  fetch_k    : {fetch_k}")
                    print(f"  lambda_mult: {lambda_mult}")
                    results = mmr_search(query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)
                    _display_results(results)

                elif cmd == "3":
                    _print_separator("SIMILARITY + SCORES")
                    print(f"  Query : {query}")
                    print(f"  Top-k : {k}")
                    scored = similarity_search_with_score(query, k=k)
                    _display_scored_results(scored)

            else:
                print("  ⚠️  Unknown command. Use 1, 2, 3, h, or q.")

        except KeyboardInterrupt:
            print("\n\n  👋 Interrupted. Exiting.")
            break
        except Exception as e:
            print(f"\n  ❌ Error: {e}")


# -----------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------
if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from ingestion.loader import load_documents

    FILE_PATH = "../../data/documents/Mcdonalds_report/mcd-20251231.pdf"

    # -------------------------------------------------------
    # STEP 1: Load + chunk
    # -------------------------------------------------------
    _print_separator("STEP 1: Load + Chunk")
    chunks = load_documents(FILE_PATH)

    # -------------------------------------------------------
    # STEP 2: Upsert
    # Change recreate=False after first run to skip re-embedding
    # -------------------------------------------------------
    _print_separator("STEP 2: Upsert to Qdrant")
    upsert_documents(chunks, recreate=True)

    # -------------------------------------------------------
    # STEP 3: Health check
    # -------------------------------------------------------
    health_check()

    # -------------------------------------------------------
    # STEP 4: Interactive query loop
    # -------------------------------------------------------
    interactive_query_loop()