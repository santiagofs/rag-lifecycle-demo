from src.embeddings import get_embedding
from src.db import store
from typing import List, Dict, Optional
from config import TOP_K, HYBRID, USE_FAISS

try:
    from src.store.faiss_index import search as faiss_search  # type: ignore
except Exception:
    faiss_search = None  # FAISS optional

def retrieve_cosine(query: str, k: int = None) -> List[Dict]:
    """Retrieve documents using cosine similarity"""
    if k is None:
        k = TOP_K
    query_embedding = get_embedding(query)
    results = store.search_cosine(query_embedding, k)
    return results

def retrieve_hybrid(query: str, k: int = None, cosine_weight: float = 0.7, fts_weight: float = 0.3) -> List[Dict]:
    """Retrieve documents using hybrid search (cosine + FTS5)"""
    if k is None:
        k = TOP_K
    query_embedding = get_embedding(query)
    results = store.search_hybrid(query_embedding, query, k, cosine_weight, fts_weight)
    return results

def retrieve_fts(query: str, k: int = None) -> List[Dict]:
    """Retrieve documents using FTS5 full-text search"""
    if k is None:
        k = TOP_K
    results = store.search_fts(query, k)
    return results

def retrieve_faiss(query: str, k: int = None) -> List[Dict]:
    """Retrieve documents using FAISS index if available, else fallback to cosine"""
    if k is None:
        k = TOP_K
    if faiss_search is None:
        return retrieve_cosine(query, k)
    try:
        hits = faiss_search(query, k)
        return [
            {"id": r["id"], "text": r.get("text"), "metadata": r.get("metadata"), "similarity": r.get("score", 0.0)}
            for r in hits
        ]
    except Exception:
        return retrieve_cosine(query, k)

def retrieve(query: str, k: int = None) -> List[Dict]:
    """Main retrieval function that uses HYBRID setting"""
    if k is None:
        k = TOP_K

    if HYBRID:
        return retrieve_hybrid(query, k)
    if USE_FAISS and faiss_search is not None:
        try:
            results = faiss_search(query, k)
            # Map to common shape used by cosine/hybrid
            return [
                {"id": r["id"], "text": r.get("text"), "metadata": r.get("metadata"), "similarity": r.get("score", 0.0)}
                for r in results
            ]
        except Exception:
            # Fallback to cosine if FAISS fails
            pass
    return retrieve_cosine(query, k)

def get_context_from_results(results: List[Dict]) -> str:
    """Convert retrieval results to context string"""
    return "\n---\n".join([r['text'] for r in results])
