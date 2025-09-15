from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
import uvicorn
from src.embeddings import get_embedding
from src.db import store
from src.retrieve import retrieve_cosine, retrieve_hybrid, retrieve_fts, retrieve_faiss, get_context_from_results, retrieve
from src.llm import generate
from config import TOP_K, HYBRID, DB_PATH, INDEX_DIR, LLM_MODEL, EMBED_MODEL_NAME, PROVIDER
try:
    from config import USE_FAISS  # optional flag
except Exception:
    USE_FAISS = False

app = FastAPI(title="RAG API", description="Retrieval-Augmented Generation API")

class DocumentRequest(BaseModel):
    text: str
    metadata: Optional[Dict] = None

class QueryRequest(BaseModel):
    query: str
    k: int = None
    method: str = None  # "cosine", "hybrid", "fts", or None for auto
    cosine_weight: float = 0.7
    fts_weight: float = 0.3

class QueryResponse(BaseModel):
    query: str
    results: List[Dict]
    context: str
    answer: Optional[str] = None

@app.post("/ingest", response_model=Dict)
async def ingest_document(doc: DocumentRequest):
    """Ingest a document into the vector store"""
    try:
        embedding = get_embedding(doc.text)
        doc_id = store.add_document(doc.text, embedding, doc.metadata)
        return {
            "success": True,
            "doc_id": doc_id,
            "message": f"Document ingested with ID {doc_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrieve", response_model=QueryResponse)
async def retrieve_documents(query: QueryRequest):
    """Retrieve documents based on query"""
    try:
        # Use environment defaults if not specified
        k = query.k or TOP_K
        method = query.method or ("hybrid" if HYBRID else "cosine")

        if method == "cosine":
            results = retrieve_cosine(query.query, k)
        elif method == "hybrid":
            results = retrieve_hybrid(query.query, k, query.cosine_weight, query.fts_weight)
        elif method == "fts":
            results = retrieve_fts(query.query, k)
        elif method == "faiss":
            results = retrieve_faiss(query.query, k)
        else:
            raise HTTPException(status_code=400, detail="Invalid method. Use 'cosine', 'hybrid', or 'fts'")

        context = get_context_from_results(results)

        return QueryResponse(
            query=query.query,
            results=results,
            context=context
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/rag", response_model=QueryResponse)
async def rag_query(query: QueryRequest):
    """Full RAG pipeline: retrieve + generate"""
    try:
        # Use environment defaults if not specified
        k = query.k or TOP_K
        method = query.method or ("hybrid" if HYBRID else "cosine")

        # Retrieve documents
        if method == "cosine":
            results = retrieve_cosine(query.query, k)
        elif method == "hybrid":
            results = retrieve_hybrid(query.query, k, query.cosine_weight, query.fts_weight)
        elif method == "fts":
            results = retrieve_fts(query.query, k)
        elif method == "faiss":
            results = retrieve_faiss(query.query, k)
        else:
            raise HTTPException(status_code=400, detail="Invalid method. Use 'cosine', 'hybrid', or 'fts'")

        context = get_context_from_results(results)

        # Generate answer
        prompt = f"Use the context to answer.\n\nContext:\n{context}\n\nQuestion: {query.query}\nAnswer:"
        answer = generate(prompt)

        return QueryResponse(
            query=query.query,
            results=results,
            context=context,
            answer=answer
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get database statistics"""
    try:
        stats = store.get_stats()
        return {
            "document_count": stats['documents'],
            "vector_count": stats['vectors'],
            "status": "healthy"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/config")
async def get_config():
    """Surface runtime configuration (non-sensitive) to verify alignment."""
    return {
        "db_path": DB_PATH,
        "index_dir": INDEX_DIR,
        "retrieval": {
            "top_k": TOP_K,
            "hybrid": HYBRID,
            "use_faiss": USE_FAISS,
        },
        "models": {
            "llm_model": LLM_MODEL,
            "embed_model": EMBED_MODEL_NAME,
        },
        "provider": PROVIDER,
        "status": "ok",
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
