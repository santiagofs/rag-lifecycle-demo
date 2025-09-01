from src.embeddings import get_embedding
from src.db import store
from typing import List, Dict, Optional

def retrieve_cosine(query: str, k: int = 3) -> List[Dict]:
    """Retrieve documents using cosine similarity"""
    query_embedding = get_embedding(query)
    results = store.search_cosine(query_embedding, k)
    return results

def retrieve_hybrid(query: str, k: int = 3, cosine_weight: float = 0.7, fts_weight: float = 0.3) -> List[Dict]:
    """Retrieve documents using hybrid search (cosine + FTS5)"""
    query_embedding = get_embedding(query)
    results = store.search_hybrid(query_embedding, query, k, cosine_weight, fts_weight)
    return results

def retrieve_fts(query: str, k: int = 3) -> List[Dict]:
    """Retrieve documents using FTS5 full-text search"""
    results = store.search_fts(query, k)
    return results

def get_context_from_results(results: List[Dict]) -> str:
    """Convert retrieval results to context string"""
    return "\n---\n".join([r['text'] for r in results])
